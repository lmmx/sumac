#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?usage: scripts/modal-smoke-test.sh ENDPOINT SERVED_MODEL_NAME}"
SERVED_MODEL_NAME="${2:?usage: scripts/modal-smoke-test.sh ENDPOINT SERVED_MODEL_NAME}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The deploy-time gate docs/journal/2026-09-04-modal-remote-inference-backend.md
# calls the single highest-value pre-flight check for this backend: send one
# fixed request through the deployed endpoint and confirm tool_calls actually
# comes back structured, not as unparsed text. `evals/conftest.py` runs this
# same check automatically before any `--eval-backend modal` run — this script
# is for checking it by hand right after `modal deploy`, before wiring up a
# full eval run. `sumac.modal_backend` is stdlib-only (no `mistralrs`, no
# `modal` package) — this needs nothing beyond sumac's own base install.
uv run python -m sumac.modal_backend \
  --endpoint "$ENDPOINT" \
  --served-model-name "$SERVED_MODEL_NAME"
