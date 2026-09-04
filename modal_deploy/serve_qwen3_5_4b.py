"""Deploys Qwen3.5-4B behind an OpenAI-compatible vLLM server on Modal —
the Modal-side half of the eval-speed backend in
docs/journal/2026-09-04-modal-remote-inference-backend.md. Everything
sumac-specific (the HTTP client, the deploy-time tool-calling gate) lives
in `src/sumac/modal_backend.py` instead; this file has no dependency on
the sumac package and can be deployed on its own.

Modelled directly on Modal's own current `vllm_inference.py` example
(`modal-labs/modal-examples`, `06_gpu_and_ml/llm-serving/`) — the
`@app.server` decorator, not the older `@app.function` +
`asgi_app`/`web_server` pattern, is Modal's current recommended way to put
a raw TCP/HTTP server (vLLM's own `vllm serve` process) behind a public
URL.

**Quantization decision (not GGUF, not a community AWQ checkpoint):**
serves the official bf16 `Qwen/Qwen3.5-4B` checkpoint with vLLM's
on-the-fly `--quantization fp8` (no calibration, no separate checkpoint —
computed from the bf16 weights at load time). Two things this deliberately
avoids: (1) vLLM's own GGUF support — confirmed current and real, not
just carried over from the journal entry's original caution: it's now an
out-of-tree plugin, vLLM's docs call it "highly experimental and
under-optimized," and a cited benchmark shows ~93 tok/s, "better suited
for llama.cpp than vLLM" — so reusing this repo's local
`unsloth/Qwen3.5-4B-GGUF` file directly on Modal is not actually the fast
path it looks like; (2) a community AWQ/GPTQ requantization (e.g.
`cyankiwi/Qwen3.5-4B-AWQ-4bit` — the only 4B-sized option found; Qwen only
publishes official GPTQ at 27B/122B/397B) — weight-only, no activation-
quantization-overhead risk, and would likely be faster than FP8, but
carries an unverified-provenance gap the official-weights + FP8 path
avoids entirely. **Known, accepted risk of the choice actually made
here:** FP8 weight+activation quantization's overhead can outweigh its
gains specifically on small/memory-bound models (a cited vLLM benchmark
showed a 256M model getting *slower*, not faster, under FP8 W8A8) — 4B is
larger and likely fares better, but this was not benchmarked for this
model before choosing it. **Benchmark this after deploying** (time a few
`scripts/modal-smoke-test.sh` runs, or compare against the same file with
`"--quantization", "fp8"` removed) before trusting FP8 saves Modal
compute — if it doesn't, drop the flag and reconsider.

Before running `modal deploy` on this file:

1. `MODEL_NAME`/`MODEL_REVISION` — `Qwen/Qwen3.5-4B` was confirmed to
   exist and be the instruction/chat-tuned checkpoint (not `-Base`) via a
   web fetch this session; `MODEL_REVISION` below was read off that
   fetch's summary of the repo's commit history, which is more error-prone
   for an exact hash string than reading it directly — re-confirm it
   against the repo's own "Files and versions" → commit history tab before
   trusting it blindly. An unpinned floating reference means an upstream
   update silently changes the benchmark model mid-experiment; see the
   journal entry's "Modal deployment specifics" section.
2. Confirm the installed `vllm` version actually supports Qwen3.5, the
   `TOOL_CALL_PARSER` chosen below, and `--quantization fp8` for this
   architecture — Qwen tool-call parser support has shifted across vLLM
   releases (`hermes` vs `qwen3_xml` vs `qwen3_coder`); the journal entry's
   "tool-call parser" section explains why the wrong choice fails silently
   (empty `tool_calls`, not an error) rather than loudly.
   `scripts/modal-smoke-test.sh` (run after deploying) is the actual gate
   that catches a wrong parser choice — treat it as required, not
   optional, precisely because this step can't be verified in advance.
   `Qwen/Qwen3.5-4B`'s own model card says it "operates in thinking mode by
   default" — if `chat_template_kwargs: {"enable_thinking": false}"` (what
   every sumac request sends) isn't fully respected by whatever chat
   template vLLM renders, `<think>` text could still land in `content`;
   `verify_tool_calling` in `src/sumac/modal_backend.py` checks for this
   explicitly, not just "did tool_calls come back non-empty."

Deploy:  modal deploy modal_deploy/serve_qwen3_5_4b.py
Test:    modal run modal_deploy/serve_qwen3_5_4b.py
"""

import json

import modal

MINUTES = 60
VLLM_PORT = 8000

MODEL_NAME = "Qwen/Qwen3.5-4B"  # official, instruction-tuned — confirmed via HF, not "-Instruct"
MODEL_REVISION = "851bf6e"  # RE-CONFIRM against the repo's commit history — see docstring point 1

# The name every client (including `sumac.modal_backend.ModalCompletions`)
# addresses this deployment's model as — must match
# `evals/conftest.py`'s `_MODAL_SERVED_MODEL_NAMES["qwen3.5-4b"]` exactly.
SERVED_MODEL_NAME = "qwen3.5-4b-instruct"

# vLLM's OpenAI-compatible server needs to be told which parser understands
# this model family's tool-call format — the "silent-failure fork" the
# journal entry names: the wrong choice doesn't error, it just makes
# `tool_calls` come back empty. CONFIRMED via `scripts/modal-smoke-test.sh`
# against a real deployment: `hermes` fails silently here — this model
# actually emits `<function=name><parameter=key>value</parameter></function>`
# (XML-tag style), not Hermes's JSON-in-`<tool_call>` style, so
# `hermes_tool_parser` throws `json.decoder.JSONDecodeError` internally and
# drops the call as unparsed text into `content`. `qwen3_xml` is the parser
# built for exactly this tag shape.
TOOL_CALL_PARSER = "qwen3_xml"

# No `--reasoning-parser` flag anywhere below, deliberately: sumac always
# sends `enable_thinking: false` (see llm.py's `_build_request`), and the
# journal entry names a specific reported failure mode where a reasoning
# parser left enabled alongside a tool-call parser makes Qwen3.5 return
# tool calls as unparsed XML text in `content` instead of a structured
# `tool_calls` array, even with thinking off in the request. Don't add one
# back without re-reading that section.

# Trades a slower first boot for better steady-state generation throughput
# (Torch compilation + CUDA graph capture) — right for a session firing
# many hundreds of eval requests against one warm container, per the
# journal entry's "boot vs. throughput tradeoff" section. Flip to True
# while iterating on this deployment's own configuration, where fast
# restarts matter more than steady-state speed.
FAST_BOOT = False  # measured: full compile ran past several minutes on a first boot —
# see the comment above; flip back to False deliberately (and bump
# serve_qwen3_5_4b.py's own @app.server(startup_timeout=...) well past
# whatever that full sweep actually measures) once steady-state throughput
# is the goal, not "does the pipeline work."

# A further, separate boot-speed lever once FAST_BOOT is False again: vLLM
# logs the exact `--kv-cache-memory <value>` that reproduces its measured
# allocation on every boot. Passing that value back on the next boot skips
# the memory-profiling + CUDA-graph memory-estimation pass entirely — but
# it's only valid for the same GPU/model/config with the same free memory
# at boot, and a stale value either caps concurrency (too conservative) or
# OOMs (too optimistic), so it needs a real logged value from an actual
# boot, not a guess. Not wired in here for that reason — grab it from your
# own deploy's logs once you're tuning steady-state boot time.

# A 4B model in bf16 is ~8GB of weights — L4 (24GB) has ample headroom for
# weights plus KV cache without reaching for a bigger/pricier GPU tier. See
# the journal entry's "GPU sizing" note. Bump this if you hit an OOM.
GPU = "L4:1"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        # Fastest to use nightly releases
        "vllm",
        extra_index_url="https://wheels.vllm.ai/nightly",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            # NOT `VLLM_FORCE_AOT_LOAD=1` here, deliberately (previously
            # was — removed after it turned a legitimate cache-invalidating
            # config change into a hard crash instead of a fresh compile):
            # *any* change to the model, config, relevant `VLLM_*` vars,
            # torch build, or GPU model invalidates the persisted
            # `vllm_cache_vol` AOT cache — including switching vLLM
            # versions (nightly moves every day) or adding
            # `--speculative-config`, both of which this file does. Forcing
            # AOT-only loading is only useful once you've stopped changing
            # things and want drift caught loudly — re-add it then, not
            # while actively tuning FAST_BOOT/quantization/parser/version.
        }
    )
)

# Two volumes, not one: caching HF weights but not vLLM's own JIT-compiled
# kernel cache is a reported common cause of multi-minute cold starts even
# with weights already warm. See the journal entry's "Modal deployment
# specifics" section.
hf_cache_vol = modal.Volume.from_name("sumac-modal-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("sumac-modal-vllm-cache", create_if_missing=True)

app = modal.App("sumac-qwen3-5-4b")


@app.server(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=15 * MINUTES,
    startup_timeout=14 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    target_concurrency=32,  # tune after measuring per the journal's "concurrency" section
    unauthenticated=True,  # for a plain HTTPS endpoint; add Modal's auth if this needs privacy
)
class Server:
    @modal.enter()
    def start(self) -> None:
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            TOOL_CALL_PARSER,
            # On-the-fly FP8 (E4M3, per-tensor, dynamic activation scales) —
            # no calibration, no separate checkpoint. See the module
            # docstring's "quantization decision" section for why this was
            # chosen over GGUF/community-AWQ, and its own known,
            # unbenchmarked risk at this model size.
            "--quantization",
            "fp8",
            # No multimodal input anywhere in sumac's tool-calling loop.
            "--limit-mm-per-prompt",
            json.dumps({"image": 0, "video": 0, "audio": 0}),
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":1}',
        ]
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES) -> None:
    """`modal run modal_deploy/serve_qwen3_5_4b.py` — a bare health check plus one
    plain (non-tool-calling) request, just to confirm the server comes up
    and answers at all. This is NOT the tool-calling gate — run
    `scripts/modal-smoke-test.sh` against the deployed URL for that; it's
    the one that actually proves `tool_calls` round-trips correctly, which
    a plain chat reply here can't tell you."""
    import asyncio
    import time

    import aiohttp

    url = await Server.get_url.aio()

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        deadline = time.time() + test_timeout - 1 * MINUTES
        while time.time() < deadline:
            async with session.get("/health", timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    break
                if resp.status == 503:
                    await asyncio.sleep(1)
                    continue
                raise RuntimeError(f"health check failed: HTTP {resp.status}")
        else:
            raise RuntimeError("health check never returned 200 within test_timeout")
        print(f"Successful health check for server at {url}")

        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with session.post("/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            body = await resp.json()
            print(body["choices"][0]["message"]["content"])
