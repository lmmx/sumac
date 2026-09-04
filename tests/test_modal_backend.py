"""No real network call in any test here — every `urllib.request.urlopen`
call is monkeypatched to a canned in-memory response, the same "fake the
one seam that touches the outside world" approach `tests/test_llm.py`'s
`FakeRunner` uses for mistral.rs. See
docs/journal/2026-09-04-modal-remote-inference-backend.md.
"""

from __future__ import annotations

import json
import urllib.error
from email.message import Message
from typing import Any

import pytest

from sumac import modal_backend


class _FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _openai_response(*, content: str | None = None, tool_call: tuple[str, dict] | None = None):
    if tool_call is None:
        message: dict[str, Any] = {"role": "assistant", "content": content}
    else:
        name, args = tool_call
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
    return json.dumps(
        {
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def _fixed_request() -> dict:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "unsloth/Qwen3.5-4B-GGUF",
        "tool_schemas": [
            json.dumps(
                {
                    "type": "function",
                    "function": {"name": "sumac_find_inventory", "parameters": {}},
                }
            )
        ],
        "tool_choice": "auto",
        "enable_thinking": False,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 1024,
        "seed": None,
    }


def test_build_body_uses_served_model_name_not_local_gguf_id() -> None:
    """The dict's own `model` field is sumac's local GGUF repo id — the
    Modal side must substitute the actual vLLM `--served-model-name`
    instead of passing that straight through."""
    body = modal_backend._build_body(_fixed_request(), served_model_name="qwen3.5-4b-instruct")

    assert body["model"] == "qwen3.5-4b-instruct"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tools"][0]["function"]["name"] == "sumac_find_inventory"
    assert body["tool_choice"] == "auto"


def test_build_body_passes_seed_when_present() -> None:
    request = _fixed_request()
    request["seed"] = 12345

    body = modal_backend._build_body(request, served_model_name="m")

    assert body["seed"] == 12345


def test_build_body_omits_seed_when_none_or_absent() -> None:
    request = _fixed_request()
    assert "seed" not in modal_backend._build_body(request, served_model_name="m")

    del request["seed"]  # the gate's own hand-built request never sets this key at all
    assert "seed" not in modal_backend._build_body(request, served_model_name="m")


def test_build_body_omits_tools_when_no_schemas() -> None:
    request = _fixed_request()
    request["tool_schemas"] = []

    body = modal_backend._build_body(request, served_model_name="m")

    assert "tools" not in body
    assert "tool_choice" not in body


def test_send_chat_completion_request_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _openai_response(tool_call=("sumac_find_inventory", {"query": "jam"}))
    monkeypatch.setattr(
        modal_backend.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(raw)
    )

    backend = modal_backend.ModalCompletions("https://example.modal.direct", "qwen3.5-4b-instruct")
    response = backend.send_chat_completion_request(_fixed_request())

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    call = tool_calls[0].function
    assert call.name == "sumac_find_inventory"
    assert json.loads(call.arguments) == {"query": "jam"}
    usage = response.usage
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert response.raw_body == raw


def test_send_chat_completion_request_parses_plain_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _openai_response(content="the jam is in the pantry")
    monkeypatch.setattr(
        modal_backend.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(raw)
    )

    backend = modal_backend.ModalCompletions("https://example.modal.direct", "qwen3.5-4b-instruct")
    response = backend.send_chat_completion_request(_fixed_request())

    message = response.choices[0].message
    assert message.tool_calls is None
    assert message.content == "the jam is in the pantry"


def test_transport_error_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: object, **k: object):
        raise urllib.error.HTTPError("url", 503, "Service Unavailable", Message(), None)

    monkeypatch.setattr(modal_backend.urllib.request, "urlopen", _raise)
    backend = modal_backend.ModalCompletions("https://example.modal.direct", "m")

    with pytest.raises(modal_backend.ModalTransportError, match="503"):
        backend.send_chat_completion_request(_fixed_request())


def test_wait_until_ready_retries_through_cold_start_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact failure the user hit right after `modal deploy`: the
    container is still cold (image pull / weight download / compile), and
    a 503 during that window must be tolerated, not treated as broken."""
    attempts = {"n": 0}

    def _urlopen(*a: object, **k: object):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError("url", 503, "Service Unavailable", Message(), None)
        return _FakeHTTPResponse("")

    monkeypatch.setattr(modal_backend.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(modal_backend.time, "sleep", lambda _s: None)

    modal_backend.wait_until_ready("https://example.modal.direct", startup_timeout_s=5.0)

    assert attempts["n"] == 3


def test_wait_until_ready_raises_immediately_on_non_503_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real error (not a cold start) must surface right away, not get
    silently retried for the full startup timeout."""

    def _raise(*a: object, **k: object):
        raise urllib.error.HTTPError("url", 500, "Internal Server Error", Message(), None)

    monkeypatch.setattr(modal_backend.urllib.request, "urlopen", _raise)
    monkeypatch.setattr(modal_backend.time, "sleep", lambda _s: pytest.fail("should not retry"))

    with pytest.raises(modal_backend.ModalTransportError, match="500"):
        modal_backend.wait_until_ready("https://example.modal.direct", startup_timeout_s=60.0)


def test_wait_until_ready_raises_after_startup_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: object, **k: object):
        raise urllib.error.HTTPError("url", 503, "Service Unavailable", Message(), None)

    monkeypatch.setattr(modal_backend.urllib.request, "urlopen", _raise)
    monkeypatch.setattr(modal_backend.time, "sleep", lambda _s: None)

    with pytest.raises(modal_backend.ModalTransportError, match="never became ready"):
        modal_backend.wait_until_ready("https://example.modal.direct", startup_timeout_s=0.01)


def test_transport_error_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        modal_backend.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse("not json")
    )
    backend = modal_backend.ModalCompletions("https://example.modal.direct", "m")

    with pytest.raises(modal_backend.ModalTransportError):
        backend.send_chat_completion_request(_fixed_request())


def test_verify_tool_calling_passes_when_tool_calls_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _openai_response(tool_call=("sumac_find_inventory", {"query": "jam"}))
    monkeypatch.setattr(
        modal_backend.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(raw)
    )

    modal_backend.verify_tool_calling("https://example.modal.direct", "qwen3.5-4b-instruct")


def test_verify_tool_calling_raises_gate_error_when_tool_calls_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silent-failure mode the journal entry names directly: a
    misconfigured `--tool-call-parser` leaves the model's real call as
    unparsed text instead of a structured `tool_calls` array."""
    raw = _openai_response(content="<tool_call>sumac_find_inventory(query=jam)</tool_call>")
    monkeypatch.setattr(
        modal_backend.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(raw)
    )

    with pytest.raises(modal_backend.ModalToolGateError, match="tool_calls array"):
        modal_backend.verify_tool_calling("https://example.modal.direct", "qwen3.5-4b-instruct")


def test_verify_tool_calling_raises_gate_error_on_leaked_think_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structured tool call alone isn't enough to trust the deployment —
    a chat template that defaults to thinking mode on can still leak a
    `<think>` block into `content` even when `enable_thinking: false` was
    sent and `tool_calls` came back correctly parsed."""
    raw = _openai_response(
        content="<think>let me check inventory</think>",
        tool_call=("sumac_find_inventory", {"query": "jam"}),
    )
    monkeypatch.setattr(
        modal_backend.urllib.request, "urlopen", lambda *a, **k: _FakeHTTPResponse(raw)
    )

    with pytest.raises(modal_backend.ModalToolGateError, match="<think>"):
        modal_backend.verify_tool_calling("https://example.modal.direct", "qwen3.5-4b-instruct")
