# sumac: Local In-Process Agent for `sumac ask`

**Status:** design record, drawn from an external ideation session; implementation not started
**Author:** drafted with Claude, 2026-09-01
**Scope:** new `sumac/llm.py` (or equivalent), `cli.py`'s `ask` command, no changes to `decide.py`/`ledger.py`/`events.py` domain logic

---

## 1. Motivation

The domain model (event log, ledger fold, `decide` gate) is already correct and already tested against a real household's usage. The friction is entirely at the interface: a single physical action — "take a ragu out of the freezer and a tin of tomatoes out of the pantry, put ragu-for-one and a jam jar of tomatoes in the fridge, second shelf" — decomposes into several `sumac add` invocations, each requiring information the user must look up by hand first: the exact product string as stored (`"Homemade Ragu"` vs `"Home Made Ragu"`, two distinct products), the exact location id (`fridge-main-shelf-2`, not "second shelf"), and the exact unit already on record (`bags`, not `bag` — `decide_change` rejects the singular with `unit_unconvertible` since unit conversion is nominal per `Config.convert`, not a pluralisation rule). A worked session doing exactly this — recorded in full below in §2 — took thirteen commands and three dead-end attempts to complete a task a person would describe in one sentence.

The session's own conclusion, reached before any code existed, is the thesis this entry inherits: the CLI's command surface is a fine *domain API* and a poor *human interface* for compound, underspecified requests. The fix is not to make the CLI's argument parsing smarter — `sumac add`'s job is to be an unambiguous, scriptable operation, and it should stay that way — but to add a second, narrower entry point that accepts an underspecified sentence, resolves it against the real inventory, and asks before writing anything.

## 2. How the idea developed

The starting point was a real interactive session (transcript preserved in this repo's chat history, not reproduced verbatim here) in which the user asked an external LLM, command by command, how to record that a jar of jam was eaten, then how to execute the ragu/tomatoes compound request above. That session surfaced two facts that shaped everything after it:

- **The chore was real and specific, not a vague UX complaint.** `sumac find` already returns exactly the disambiguating information a human needs (product name, quantity, unit, full location path) — the burden was manually transcribing that output back into `sumac add`'s positional arguments, correctly, every time.
- **The compound request was not a `movement`.** The ragu was cooked from frozen base ragu and repackaged into a Tupperware box under a new name ("ragu for one"); the tomatoes were decanted from a half-used tin into a jam jar. Both product identity *and* unit changed. The session's own resolution was two `consumption` events (against the original products, from their original locations) followed by two `discovery` events (the new products, into the fridge) — four writes, not one `movement` with a relabelled endpoint. This is the shape any agent built for this domain has to reproduce: **one user utterance can require an ordered sequence of distinct domain operations**, not a single tool call.

From there the design question moved through several rejected shapes, each rejected for a stated reason rather than by default:

- **A rewritten, "smarter" `sumac add`** (inferring source location from a unique `find` match, accepting plurals, taking a free-text destination like "fridge second shelf") — rejected as the first idea, because it still requires the human to decompose a compound request into separate CLI invocations; it improves each command without addressing the ragu/tomatoes case at all.
- **A general chat/GUI product** (conversational history, a `sumac` REPL with a chat transcript, eventually a web UI with a command palette) — the broader idea explored the shape of this at length, but it was set aside as out of scope for a first slice: it multiplies the surface area (rendering, session state, undo) before the core translation — sentence to domain calls — is shown to work at all.
- **Ollama** — rejected as a dependency: it runs as a separate local server process, which conflicts directly with the constraint that `sumac ask` is a plain CLI invocation with no daemon and no network port, local or otherwise.
- **`llama-cpp-python`** — considered viable (in-process, CPU-capable, GGUF, has tool-calling support) and not rejected, but superseded once mistral.rs's Python SDK was found to offer the same shape with a tool-callback API already built around exactly this pattern.
- **mistral.rs's `mistralrs serve --agent`** — this was the one genuine near-miss, because the documentation page found for it (`docs/agents/build-an-agent`) is titled "agents" and looked at first glance like the intended entry point. It is not: that mode runs a server process (`mistralrs serve`) and hands the model a fixed, generalist toolset — web search, arbitrary Python execution, shell execution — none of which sumac's domain permits an LLM to touch. Distinguishing this from the Python SDK's *in-process* agentic callback API (`examples/python/agentic_tools.py`, quoted in full in the originating request) is the pivot the design settled on.
- **LangChain, CrewAI, MCP, or any other agent-orchestration framework** — rejected throughout on the same grounds: the tool surface is fixed at roughly five typed Python functions, and the loop that dispatches a tool call and feeds back its result is exactly what mistral.rs's `Runner.tool_callbacks` + `max_tool_rounds` already does, in-process, with no added framework.

## 3. What this is

`sumac ask "<sentence>"` is a CLI command, alongside the existing ones, that:

- Loads a small GGUF instruct model (roughly 0.6B–1.7B parameters, chosen for CPU inference speed over raw capability — the model's job is narrow semantic parsing against a five-tool vocabulary, not general reasoning) via mistral.rs's Python `Runner`, in-process, inside the same Python interpreter running the CLI. No server, no port, no daemon left running after the command exits.
- Registers a small, fixed set of Python tool callbacks on that `Runner` — not the CLI's own commands, but thin wrappers around the domain layer's existing entry points (`ledger.build_inventory` for search/lookup, `decide.decide_change` for consumption/movement/discovery, `decide.decide_correct` for corrections). Registered via `Runner(..., tool_callbacks={...})` and driven with `max_tool_rounds` set on the `ChatCompletionRequest`, per the SDK example that motivated this design — the engine executes the loop of "model proposes a tool call → sumac's Python code runs it → the result is fed back to the model" without any additional orchestration code.
- Can and is expected to produce **more than one domain call per invocation** — the ragu/tomatoes case (§2) is the reference example: one sentence, four writes (two `consumption`, two `discovery`), each going through the same `decide_change` gate `sumac add` uses today. The agent's role is to plan and sequence those calls, not to reimplement or bypass what `decide_change` already validates.
- Searches before acting: given "take a ragu out of the freezer," the tool wrapping inventory lookup is called first, and its actual result (the real matching products, quantities, units, and locations — not anything the model asserts on its own) is what the model reasons over for the next tool call. This mirrors `sumac find`'s existing output exactly; the agent is not given a dump of the whole inventory, only what a query returns.
- Asks the user to disambiguate rather than guessing, when a search returns more than one plausible match with no basis in the sentence for picking one — the stated example being `find_inventory("ragu")` returning both "Home Made Ragu" (1 bag) and "Homemade Ragu" (7 bags): the agent presents both and stops rather than choosing.
- Surfaces the same domain errors `sumac add` would produce — `unit_unconvertible`, `unknown_location`, `missing_endpoint`, and the rest of the rejection catalogue in `docs/journal/2026-08-30_decide-pattern-data-integrity-upgrade.md` §4 — back to the model as a tool result, not as a crash, so the model can retry with a corrected call or explain the failure to the user rather than the process exiting on an unhandled exception.

## 4. What this is not

- **Not a replacement for the existing CLI.** `sumac add`, `sumac find`, `sumac status`, and every other command are unchanged and remain the deterministic, scriptable path — the point of `mv`-style commands (fully specified by their arguments, no interpretation required) is precisely that they do not need a confirmation step, and nothing about adding `ask` changes that property for the rest of the CLI.
- **Not the mistral.rs generalist agent mode.** `mistralrs serve --agent`'s built-in web-search, shell, and Python-execution tools are explicitly not used. The LLM's only capabilities are the specific typed functions sumac registers; it has no path to the filesystem, the database, or arbitrary code, satisfying the constraint that the domain layer stays authoritative and the model can only invoke what is explicitly exposed to it.
- **Not a chat product.** The interaction is one request in, one proposed action (or one clarifying question) out, then either confirmation or cancellation — not a persisted multi-turn conversation with history that itself becomes a feature. `sumac`'s existing model deliberately does not track chat history between calls (matching mistral.rs's own note that its `Runner` does not track chat history across separate `send_chat_completion_request` calls) — each `sumac ask` invocation is independent.
- **Not a source of new domain rules.** Every operation the model can trigger already exists in `decide.py`. Nothing about this design adds a new `ChangeKind`, a new event type, or a new validation path — the "discovery" and "consumption" events used in the ragu/tomatoes example (§2) are the existing `discovery` and `consumption` kinds `sumac add` already accepts, invoked in sequence rather than added to.
- **Not restricted to interactive human use.** `sumac ask` is a CLI command like any other and can be driven by a script or by another agent, not only typed at a terminal by a person. The propose-then-confirm behaviour described in §5 belongs to the interactive path; a non-interactive caller (a script, a cron job, another agent) is a different consumer of the same underlying "sentence in, proposed writes out" mechanism and owns its own decision about whether and how to gate execution — this design does not assume a human is always the one pressing confirm.

## 5. Why a confirm step, and why only here

Every other sumac command is unambiguous by construction: `sumac add consumption "Strawberry Jam" 1 jar --from fridge-door` fully determines what will be written before it runs, the same way `mv a b` fully determines the filesystem operation before it runs — there is nothing to interpret, so there is nothing to confirm. `sumac ask` is different in kind, not degree: the input is a sentence that requires interpretation — resolving "a ragu" against two candidate products, resolving "second shelf" against a location id, deciding that "a jam jar full of tomatoes" describes a new product-and-unit pair rather than a straight move — before it can become a determinate operation at all.

That interpretation step is exactly where a small model can be wrong in a way `sumac add`'s argument parser cannot: parsing `"1"` as an amount has one meaning, but resolving "the ragu" against inventory has several. The proposal is therefore: `sumac ask` resolves the sentence into a concrete, fully-specified sequence of the same writes `sumac add` would produce, shows that sequence to the user before any of it is written (mirroring the search-first, ask-when-ambiguous behaviour in §3), and only calls into `decide_change`/`store.append` after the user accepts it. This keeps the domain layer's guarantees intact — every write, agent-proposed or hand-typed, passes through the identical `decide_change` gate — while adding a review step at the one point in the system where the input itself is ambiguous rather than the operation.

## 6. What is not yet decided

- The exact tool function signatures (parameter shapes for search/consume/move/discover) and how closely they mirror `decide.decide_change`'s existing keyword-argument shape versus a narrower agent-facing wrapper.
- How the proposed write sequence is rendered to the user for confirmation — plain text, a rich table via `render.py`, or output shared with `sumac doctor`'s existing anomaly-table rendering.
- Which specific GGUF model and quantisation is used by default, and where the model file is expected to live on disk (a cache directory under `paths.py`'s existing layout, or left to mistral.rs's own Hugging Face cache resolution).
- Whether `sumac` with no arguments becomes an interactive agent REPL, deferred in the originating request until `sumac ask` itself is proven.
- Test strategy for the agent loop without invoking a real model — the originating request calls for separating orchestration (sequencing tool calls, applying writes, rendering the proposal) from inference (the actual `Runner` call), so the orchestration can be driven by a fake model/fake tool-call sequence in tests.

---

The section below records repository state in the format specified by `docs/JOURNAL.md`. §1-§6 above predate that format's adoption for this entry and follow the `docs/journal/2026-08-31-*` entries' convention of a free-form design record followed by a compliant tail section.

---

# 2026-09-01: `sumac ask` Scaffolding State

## Current State

- `pyproject.toml` declares an optional dependency group named `ask` containing `mistralrs>=0.9.2`, both under `[project.optional-dependencies]` and `[dependency-groups]`, and lists `ask` in `[tool.uv] default-groups` alongside `dev` (pyproject.toml:17-19, pyproject.toml:30-32, pyproject.toml:50-51).
- `uv.lock` resolves `mistralrs` 0.9.2 with prebuilt wheels for macOS arm64, Linux aarch64, Linux x86_64, and Windows x86_64, and lists it under `sumac-home`'s `optional-dependencies.ask` and `metadata.requires-dev.ask` (uv.lock:223-231, uv.lock:481-484, uv.lock:507).
- `src/sumac/cli.py` defines an `ask` command taking a positional `prompt: str` and the same `--data-dir` option every other command takes (cli.py:373-377).
- `ask`'s docstring lists three example invocations — `"where is the jam?"`, `"consume 1 jar of jam"`, `"move the ragu to the fridge"` (cli.py:378-384).
- `ask` calls `_key(data_dir)` before attempting to import `sumac.llm`, so a missing vault raises `VaultNotFoundError` ahead of any agent-specific error (cli.py:385, cli.py:47-49).
- `ask` imports `sumac.llm` inside a `try/except ImportError` block with the comment "Import llm here so mistralrs is optional" (cli.py:387-393).
- On successful import, `ask` constructs `llm.AgentRunner(data_dir, key)`, calls `.run(prompt)` on it, and passes the return value to `render.print_success` (cli.py:395-398).
- `ask` catches `FileNotFoundError` and prints it via `render.print_error` with exit code 1, and catches any other `Exception` and prints it prefixed `"Agent error: "` with exit code 1 (cli.py:399-404).
- No other command in `cli.py` imports from or references `sumac.llm` or `mistralrs` (cli.py, full file).
- `README.md` contains no reference to `ask`, `agent`, or `mistral` in any case (README.md, full file).

## Stubbed

- `ask`'s `except ImportError` branch (cli.py:390-393) fires identically whether `mistralrs` itself is absent or `sumac.llm` is absent — both raise `ModuleNotFoundError`, a subclass of `ImportError` — so the printed message "Agent requires mistralrs. Install with: pip install mistralrs" prints even when `mistralrs` is installed and only `sumac.llm` is missing, which is the actual state of this repository (§ Missing, below).
- `ask`'s `except Exception` branch (cli.py:402-404) catches every exception `llm.AgentRunner`'s constructor or `.run()` could raise, including `SumacError` subclasses that other commands (`add`, `snapshot`, `correct`) let propagate to `cli.main`'s own handler (cli.py:407-412) — so a `Rejected` raised inside an agent tool call would print as `"Agent error: ..."` rather than through the same error path `sumac add` uses for the identical rejection.

## Missing

- `src/sumac/llm.py` does not exist in the repository (confirmed: no file at that path under `src/sumac/`). Every name `cli.py`'s `ask` command references from it — `llm.AgentRunner`, its `__init__(data_dir, key)` signature, and its `.run(prompt) -> str` method — has no implementation anywhere in the tree.
- No module defines tool functions wrapping `ledger.build_inventory`, `decide.decide_change`, or `decide.decide_correct` for agent use — `decide.py`, `ledger.py`, and `config.py` are unchanged by the two commits that introduced `ask` (git log: `4e437c4` touches only `pyproject.toml`/`uv.lock`; `013a638` touches only `cli.py`).
- No code constructs a `mistralrs.Runner`, registers `tool_callbacks`, or sets `max_tool_rounds` anywhere in `src/sumac/` (no match for `mistralrs`, `Runner`, or `tool_callbacks` outside `pyproject.toml`/`uv.lock`).
- No model file, model-path configuration, or download mechanism exists — `paths.py` defines no function or constant naming a model cache directory (paths.py, full file).
- No test file references `ask`, `llm`, `AgentRunner`, or `mistralrs` (no match under `tests/`).
- No propose/confirm rendering exists for a multi-write agent proposal — `render.py` defines `print_success`, `print_error`, `print_warning`, `print_status`, `print_find`, `print_log`, `print_verify`, `print_doctor`, and `print_anomaly_banner`, none of which take a list of pending writes or a confirmation prompt (render.py, function definitions).

## Divergence

- `ask`'s docstring example `sumac ask "consume 1 jar of jam"` (cli.py:382) names an operation no code path can currently perform — `sumac.llm` does not exist (§ Missing, above), so every invocation of `ask` on the current tree reaches the `except ImportError` branch and exits without attempting to parse the prompt.
