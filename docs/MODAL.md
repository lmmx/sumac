# Modal backend: deploying it

An optional, opt-in backend for the eval suite (`evals/`, see `evals/README.md`) that runs
inference against a deployed Modal/vLLM endpoint instead of a local `mistralrs.Runner` — faster
epochs, at the cost of a quantization/fidelity gap from the local GGUF benchmark. Full design
rationale, the fidelity risks, and what's deliberately *not* built (concurrency modes, a
chat-template-diff gate, statistical hygiene) are in
`docs/journal/2026-09-04-modal-remote-inference-backend.md` — this doc is just the "how do I stand
one up" instructions. **Local `mistralrs` remains the authoritative benchmark regardless** — a
prompt that scores well on Modal still needs a local confirming run before being trusted.

Everything below runs from this repo, using the `modal` package already in the `dev` dependency
group (`uv sync`). None of it needs to run inside the same environment that runs `sumac` itself —
`src/sumac/modal_backend.py` (the HTTP client the eval suite actually uses) has zero dependency on
`mistralrs` or the `modal` package; only the deploy step below needs `modal`.

## Deploy

```sh
uv run modal deploy modal_deploy/serve_qwen3_5_4b.py
```

This prints your own endpoint URL on success (`https://<workspace>--sumac-qwen3-5-4b-server.<region>.modal.direct`)
— it's specific to your Modal workspace, keep it out of anything you share (it's an unauthenticated
public endpoint, per `unauthenticated=True` in the deploy file — anyone with the URL can hit it).
Don't commit it; export it instead (see "Running evals against it" below).

Before deploying, read `modal_deploy/serve_qwen3_5_4b.py`'s own module docstring — it documents
exactly what to confirm (model revision hash, tool-call parser, quantization choice) and why, in
more depth than repeated here.

**Redeploying after a config change:** Modal's default deploy strategy is *rolling* — an
already-warm container keeps serving old code until it drains on its own. A code-only redeploy can
report a ~2 second "deployed" and still be answering with the previous configuration. Force an
immediate cutover with:

```sh
uv run modal deploy --strategy recreate modal_deploy/serve_qwen3_5_4b.py
```

## Verify it before trusting it

```sh
./scripts/modal-smoke-test.sh <your-endpoint-url> qwen3.5-4b-instruct
```

This is the deploy-time gate the journal entry calls the single highest-value check for this whole
backend: it sends one fixed request through the endpoint and asserts `tool_calls` actually comes
back structured (not as unparsed text) and free of a leaked `<think>` block. **Treat a failure here
as a real, fixable serving-stack misconfiguration, not a reason to distrust the model** — the wrong
`--tool-call-parser` for a given model is a documented, easy-to-hit silent-failure mode (see
`modal_deploy/serve_qwen3_5_4b.py`'s `TOOL_CALL_PARSER` comment for the specific one already found
and fixed for Qwen3.5).

It also tolerates a cold container: the first request after a fresh deploy or an idle period can
take several minutes (image pull, weight download into an empty cache volume, and — with
`FAST_BOOT=False` — `torch.compile`/CUDA graph capture); the script polls `/health` for up to 10
minutes rather than failing on the first 503. This is not a hang — watch `modal app logs
sumac-qwen3-5-4b` if you want to see it happen live.

## Running evals against it

```sh
export SUMAC_MODAL_ENDPOINT=<your-endpoint-url>
uv run pytest evals --eval-backend modal --eval-model qwen3.5-4b
```

Every other `--eval-*` flag (`--eval-seed`, `--eval-prompt-variant`, `--eval-json`, ...) works
exactly as it does against the local backend — see `evals/README.md`. `--eval-json`'s output
records which backend produced each epoch, and `evals/epoch_report.py` never folds a Modal epoch
into the same row as a local one.

## Keeping a container warm

A scaled-to-zero container costs nothing but adds real cold-start latency to whatever request hits
it first. To keep one instance warm for an iteration session, and release it afterward:

```sh
uv run modal run modal_deploy/warm.py --on
uv run modal run modal_deploy/warm.py --off
```

## Boot speed

`FAST_BOOT` in `serve_qwen3_5_4b.py` trades a slower first boot (full `torch.compile` + CUDA graph
capture, better steady-state throughput) against a fast one (`--enforce-eager`, worse steady-state
throughput). Measured on this deployment: the full-compile path ran well past ten minutes on a
first boot — bump `@app.server`'s own `startup_timeout` to comfortably clear whatever your own
measurement shows before ever setting `FAST_BOOT = False` again, or Modal will kill the container
as failed-to-start mid-compile. The compile cache persists on `vllm_cache_vol`, so this cost is
paid once, not on every cold start after that — `VLLM_FORCE_AOT_LOAD=1` (already set in the image)
makes a cache miss fail loudly instead of silently re-paying it.
