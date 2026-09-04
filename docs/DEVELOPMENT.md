# Development

Moved out of the main README, which is user-facing — this is for working on sumac itself.

```sh
uv sync
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

An eval suite for `sumac ask`'s agent, run against a real local model (or, optionally, a deployed
Modal endpoint), lives in `evals/` — not under `src/`, so it doesn't ship. See `evals/README.md`
for running it, and `docs/MODAL.md` for deploying the optional Modal backend.
