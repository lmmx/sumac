"""A `SendsCompletions` implementation that talks to a Modal-hosted,
OpenAI-compatible vLLM endpoint over plain HTTP instead of loading a local
GGUF through `mistralrs`. Exists purely to buy more, faster eval epochs —
see docs/journal/2026-09-04-modal-remote-inference-backend.md for the full
design rationale, the fidelity risks this module deliberately does and
doesn't paper over, and why this is opt-in, never the default.

No dependency on `mistralrs`, `sumac.llm`, or the `modal` package: this
module only ever calls a deployed endpoint's public HTTPS URL, with the
stdlib's own `urllib`. `mistralrs.ChatCompletionRequest`/
`ChatCompletionResponse` are opaque PyO3 objects a non-mistralrs backend
can neither construct nor read back — `llm._build_request` already
returns a plain dict for exactly this reason, and `llm.ChatResponse`/
`llm.SendsCompletions` are structural Protocols, so the dataclasses below
satisfy them with no import of `llm` at all — importing it would drag in
`mistralrs` transitively (`llm.py` imports it unconditionally), defeating
the point of a smoke-test script runnable with nothing but this package's
own stdlib-only deps. `_GATE_TOOL_SCHEMA` below is a representative
stand-in for `llm._FIND_INVENTORY_SCHEMA`'s shape, not that exact schema.

**Transport failures must never become scenario verdicts** — every
network/parse/HTTP-status failure below raises `ModalTransportError`. It
is never caught here, and must never be caught by a caller either; letting
a cold-container 503 or a timeout read as "the model got the answer wrong"
would silently corrupt every eval run against this backend.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

_DEFAULT_TIMEOUT_S = 60.0

# A representative tool schema for the deploy-time gate below — same
# shape (name, one string arg, "strict") as the real
# `llm._FIND_INVENTORY_SCHEMA`, but defined locally rather than imported
# from `sumac.llm`, which would pull in `mistralrs` transitively. The gate
# only needs *a* realistic tool call to round-trip, not this exact schema.
_GATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sumac_find_inventory",
        "description": "Search current inventory by product name, case-insensitive.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class ModalTransportError(RuntimeError):
    """A transport-layer failure talking to the Modal endpoint: a timeout,
    connection reset, non-2xx HTTP status, or a response body that isn't
    parseable as the OpenAI chat-completions JSON shape. Callers (the eval
    harness above all) must let this propagate and abort the epoch rather
    than record it as a failed verdict — see the module docstring."""


class ModalToolGateError(RuntimeError):
    """Raised by `verify_tool_calling` when a deployed endpoint doesn't
    round-trip a tool call correctly for one fixed, canonical request —
    almost always a serving-stack misconfiguration (wrong
    `--tool-call-parser` for the model, or a reasoning parser leaving
    `<think>`/tool-call XML in `content` instead of a structured
    `tool_calls` array), not a real model regression. Distinct from
    `ModalTransportError` so a caller can tell "the server answered, but
    wrongly" apart from "the server didn't answer"."""


@dataclass(frozen=True, slots=True)
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    avg_compl_tok_per_sec: float
    total_time_sec: float


@dataclass(frozen=True, slots=True)
class _FunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class _ToolCall:
    function: _FunctionCall


@dataclass(frozen=True, slots=True)
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None


@dataclass(frozen=True, slots=True)
class _Choice:
    message: _Message


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Satisfies `llm.ChatResponse` structurally — the exact minimal shape
    audited in the journal entry, nothing more. `raw_body` is extra (not
    part of that Protocol): the pre-parse HTTP response body verbatim,
    read via `getattr(response, "raw_body", None)` at every
    `ToolCallRecord` construction site in `llm.py`, so a tool-call-parser
    mismatch (`tool_calls` empty, the real call landed as text in
    `content`) is diagnosable from the trace after the fact instead of
    just reading as "the model didn't call the tool."."""

    choices: list[_Choice]
    usage: _Usage | None
    raw_body: str = field(compare=False, default="")


def _build_body(request: dict, *, served_model_name: str) -> dict:
    """Translates `llm._build_request`'s backend-agnostic dict into an
    OpenAI chat-completions JSON body. `request["model"]` is sumac's own
    local GGUF repo id (`ModelPreset.quantized_model_id`) — irrelevant to
    what a Modal deployment actually serves, so it's deliberately ignored
    here in favor of `served_model_name` (the vLLM `--served-model-name`
    this `ModalCompletions` was constructed with), not passed through.
    Every entry in `tool_schemas` is already a full OpenAI-style
    `{"type": "function", "function": {...}}` JSON string (the same shape
    `_GATE_TOOL_SCHEMA` above is), so no restructuring is needed — just
    `json.loads` each one straight into `tools`."""
    body: dict = {
        "model": served_model_name,
        "messages": request["messages"],
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "max_tokens": request["max_tokens"],
        # vLLM's OpenAI-compatible server takes `enable_thinking` as a
        # top-level `chat_template_kwargs` field, not a sampling param —
        # see the journal entry's "enable_thinking" section for why this
        # has to be this specific shape or thinking silently turns back on.
        "chat_template_kwargs": {"enable_thinking": request["enable_thinking"]},
    }
    schemas = request["tool_schemas"]
    if schemas:
        body["tools"] = [json.loads(s) for s in schemas]
        body["tool_choice"] = request["tool_choice"]
    # `.get`, not `["seed"]` — `verify_tool_calling`'s own fixed gate
    # request builds its dict by hand and doesn't set one; a real
    # `llm.AgentRunner` request always does (`None` when no `--eval-seed`
    # was passed), vLLM's OpenAI-compatible server accepts `seed` as a
    # standard top-level field. Previously never threaded through at all —
    # every Modal request ran against vLLM's own unseeded default with no
    # way to reproduce a specific run. See the journal entry.
    seed = request.get("seed")
    if seed is not None:
        body["seed"] = seed
    return body


class ModalCompletions:
    """A `llm.SendsCompletions` backed by a deployed Modal/vLLM
    OpenAI-compatible endpoint. `endpoint` is the server's base URL (e.g.
    `https://<workspace>--<app>-server.modal.direct`, printed by `modal
    deploy`); `served_model_name` must match whatever the deployment's
    `vllm serve ... --served-model-name` was actually launched with."""

    def __init__(
        self, endpoint: str, served_model_name: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._served_model_name = served_model_name
        self._timeout_s = timeout_s

    def send_chat_completion_request(
        self, request: dict, model_id: str | None = None
    ) -> ChatResponse:
        body = _build_body(request, served_model_name=self._served_model_name)
        payload = json.dumps(body).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self._endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout_s) as resp:
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise ModalTransportError(
                f"Modal endpoint {self._endpoint} returned HTTP {e.code}: "
                f"{e.read().decode('utf-8', errors='replace')[:500]}"
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ModalTransportError(
                f"could not reach Modal endpoint {self._endpoint}: {e}"
            ) from e
        elapsed_s = time.monotonic() - started

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise ModalTransportError(
                f"Modal endpoint {self._endpoint} returned a non-JSON body: {raw_body[:500]!r}"
            ) from e

        try:
            choices = [_parse_choice(c) for c in parsed["choices"]]
        except (KeyError, IndexError, TypeError) as e:
            raise ModalTransportError(
                f"Modal endpoint {self._endpoint} returned an unexpected response shape "
                f"(no usable 'choices'): {raw_body[:500]!r}"
            ) from e

        usage_json = parsed.get("usage") or {}
        completion_tokens = usage_json.get("completion_tokens", 0)
        usage = _Usage(
            prompt_tokens=usage_json.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            # vLLM's OpenAI-compatible `usage` typically carries only token
            # counts, not mistral.rs's engine-internal tok/s — computed
            # here from measured wall-clock instead. Never compare this
            # number against a local mistral.rs run's `avg_compl_tok_per_sec`
            # — one is engine-internal generation time, the other wall-clock
            # including HTTP round-trip and (when cold) container
            # scheduling. See the journal entry's "usage accounting" section.
            avg_compl_tok_per_sec=(completion_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
            total_time_sec=elapsed_s,
        )
        return ChatResponse(choices=choices, usage=usage, raw_body=raw_body)


def _parse_choice(choice_json: dict) -> _Choice:
    message_json = choice_json["message"]
    tool_calls_json = message_json.get("tool_calls") or []
    tool_calls = [
        _ToolCall(
            function=_FunctionCall(
                name=tc["function"]["name"], arguments=tc["function"]["arguments"]
            )
        )
        for tc in tool_calls_json
    ]
    return _Choice(
        message=_Message(content=message_json.get("content"), tool_calls=tool_calls or None)
    )


_GATE_MESSAGES = [
    {
        "role": "system",
        "content": "You must call sumac_find_inventory to answer any question about "
        "what's currently in stock. Never answer from your own knowledge.",
    },
    {"role": "user", "content": "Do we have any jam?"},
]


_STARTUP_TIMEOUT_S = 14 * 60.0  # matches serve_qwen3_5_4b.py's own startup_timeout=14*MINUTES
_HEALTH_POLL_INTERVAL_S = 2.0


def wait_until_ready(
    endpoint: str,
    *,
    startup_timeout_s: float = _STARTUP_TIMEOUT_S,
    poll_interval_s: float = _HEALTH_POLL_INTERVAL_S,
) -> None:
    """Polls `{endpoint}/health` (vLLM's own readiness endpoint) until it
    returns HTTP 200, tolerating both a 503 and a refused/timed-out
    connection — a cold, scale-to-zero container returns either while it's
    still pulling the image, downloading weights into a cold HF cache
    volume, and (with FAST_BOOT off) compiling — for up to
    `startup_timeout_s`. Modal's own reference vLLM example handles this
    identically in its `local_entrypoint` test. Treating the first request
    after a fresh deploy or an idle period as a real failure, rather than
    "not warm yet," is exactly the mistake the journal entry's "cold-start
    floor management" note warns against — this is the fix, called by
    `verify_tool_calling` below so every caller gets it for free rather
    than needing to remember to wait first. Raises `ModalTransportError`
    if the endpoint never becomes ready in time, or on any non-503 HTTP
    failure (a real error, not a cold start, should surface immediately —
    only 503/connection-refused are treated as "still starting")."""
    endpoint = endpoint.rstrip("/")
    started = time.monotonic()
    deadline = started + startup_timeout_s
    last_error: BaseException | None = None
    printed_waiting = False
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{endpoint}/health", method="GET")
            with urllib.request.urlopen(req, timeout=10.0):
                if printed_waiting:
                    print(f"endpoint ready after {time.monotonic() - started:.0f}s")
                return
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise ModalTransportError(
                    f"health check for {endpoint} returned HTTP {e.code} (not a cold-start 503)"
                ) from e
            last_error = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
        if not printed_waiting:
            # A cold container can take real minutes (image pull, weight
            # download into an empty HF cache volume, and — with FAST_BOOT
            # off — Torch compile/CUDA graph capture) — printed once, not
            # per poll, so this isn't silent without spamming the terminal.
            print(f"waiting for {endpoint} to become ready (up to {startup_timeout_s:.0f}s)...")
            printed_waiting = True
        time.sleep(poll_interval_s)
    raise ModalTransportError(
        f"endpoint {endpoint} never became ready within {startup_timeout_s:.0f}s "
        f"(last error: {last_error})"
    )


def verify_tool_calling(
    endpoint: str,
    served_model_name: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    startup_timeout_s: float = _STARTUP_TIMEOUT_S,
) -> None:
    """The deploy-time gate the journal entry calls the single highest-value
    pre-flight check for this whole backend: wait for the endpoint to be
    warm (`wait_until_ready`), then send one fixed, canonical request —
    `_GATE_TOOL_SCHEMA`, a prompt that unambiguously requires calling it —
    through it, and assert the response comes back with a non-empty
    `tool_calls` array whose `arguments` parses as JSON. Raises
    `ModalToolGateError` on failure (a misconfigured `--tool-call-parser`
    reads as "the model didn't call the tool" if nothing checks this
    explicitly) or `ModalTransportError` if the endpoint never came up or
    couldn't be reached at all. Callers (an eval-session fixture, or the
    `__main__` entry below) should treat either as a hard stop, never a
    skip: **never let a misconfigured serving stack present as "a prompt
    change hurt every scenario."**"""
    wait_until_ready(endpoint, startup_timeout_s=startup_timeout_s)
    backend = ModalCompletions(endpoint, served_model_name, timeout_s=timeout_s)
    request = {
        "messages": _GATE_MESSAGES,
        "model": served_model_name,
        "tool_schemas": [json.dumps(_GATE_TOOL_SCHEMA)],
        "tool_choice": "auto",
        "enable_thinking": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
    }
    response = backend.send_chat_completion_request(request)
    if not response.choices:
        raise ModalToolGateError(f"deploy-time gate: no choices in response: {response.raw_body!r}")
    message = response.choices[0].message
    if not message.tool_calls:
        raise ModalToolGateError(
            "deploy-time gate: the server never returned a tool_calls array for a prompt "
            "that unambiguously requires one calling sumac_find_inventory. Almost always "
            "a wrong --tool-call-parser (or a reasoning parser left on with thinking "
            "'off') for this model on the serving stack, not a prompt problem — see "
            "docs/journal/2026-09-04-modal-remote-inference-backend.md. "
            f"content was: {message.content!r}"
        )
    try:
        json.loads(message.tool_calls[0].function.arguments)
    except json.JSONDecodeError as e:
        raise ModalToolGateError(
            f"deploy-time gate: tool call arguments did not parse as JSON: {e}"
        ) from e
    if message.content and "<think>" in message.content:
        # A parse-time assertion, not just "did tool_calls come back
        # non-empty" — every sumac request sends `enable_thinking: false`,
        # but a chat template that defaults to thinking mode on (Qwen3.5's
        # does) can still leak a `<think>` block into `content` alongside a
        # structured tool call if that flag isn't fully honored by whatever
        # the serving stack actually renders. See the journal entry's
        # "enable_thinking" section.
        raise ModalToolGateError(
            "deploy-time gate: tool_calls came back structured, but content still "
            "contains a '<think>' block — enable_thinking is not being fully honored "
            f"by this deployment. content was: {message.content!r}"
        )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Deploy-time smoke test: verify a Modal endpoint round-trips tool calls "
        "correctly before any eval run is allowed to trust it."
    )
    parser.add_argument("--endpoint", required=True, help="Modal endpoint base URL.")
    parser.add_argument(
        "--served-model-name", required=True, help="The vLLM --served-model-name in use."
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=_STARTUP_TIMEOUT_S,
        help="Seconds to wait for a cold container to become ready before giving up "
        f"(default: {_STARTUP_TIMEOUT_S:.0f}).",
    )
    args = parser.parse_args()
    try:
        verify_tool_calling(
            args.endpoint, args.served_model_name, startup_timeout_s=args.startup_timeout
        )
    except (ModalTransportError, ModalToolGateError) as e:
        print(f"GATE FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print("GATE PASSED: tool calls round-trip correctly.")
