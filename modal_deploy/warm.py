"""Toggle a warm-container floor on the deployed `serve_qwen3_5_4b.py`
server, without a full redeploy — for the "keep one instance warm for an
iteration session, release it afterward" need in
docs/journal/2026-09-04-modal-remote-inference-backend.md's "Modal
deployment specifics" section. A cold, scaled-to-zero container adds real
latency to the first request of a session; this trades that for the cost
of an idle GPU, only while explicitly turned on. Deliberately doesn't
import `serve_qwen3_5_4b` — it looks the deployed app/class up by name
instead (`modal.Server.from_name`), so this only ever touches the running
deployment's autoscaler settings, never redefines or redeploys it.

Usage:
    modal run modal_deploy/warm.py --on
    modal run modal_deploy/warm.py --off
"""

import modal

# Must match `serve_qwen3_5_4b.py`'s `modal.App(...)` name and `Server`
# class exactly.
APP_NAME = "sumac-qwen3-5-4b"
SERVER_CLASS_NAME = "Server"

app = modal.App("sumac-qwen3-5-4b-warm-toggle")


@app.local_entrypoint()
def main(on: bool = False, off: bool = False) -> None:
    if on == off:
        raise ValueError("pass exactly one of --on or --off")
    server = modal.Server.from_name(APP_NAME, SERVER_CLASS_NAME)
    settings = server.update_autoscaler(min_containers=1 if on else 0)
    print(f"min_containers now {settings.min_containers}")
