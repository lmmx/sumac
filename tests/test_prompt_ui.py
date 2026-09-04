"""`prompt_ui`'s two input paths, driven without a terminal.

The interactive path is exercised by monkeypatching `interactive()` to True
and `read_key` to a scripted list of keypresses — the same shape
`tests/test_llm.py` uses for a scripted model. Rich renders its `Live` region
into a non-terminal console without error, so the assertions here are about
what `select`/`multiselect` return, which is the whole of their contract to
`cli.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sumac import prompt_ui


def _keys(monkeypatch: pytest.MonkeyPatch, *presses: str) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    stream: Iterator[str] = iter(presses)
    monkeypatch.setattr(prompt_ui, "read_key", lambda: next(stream))


OPTIONS = [
    prompt_ui.Option("a", "Accept"),
    prompt_ui.Option("r", "Reject"),
    prompt_ui.Option("e", "Edit"),
    prompt_ui.Option("(anything else)", "Feedback", prompt_for_text=True),
]


def test_not_interactive_without_a_terminal() -> None:
    """pytest's own stdin is not a tty, so this is the state every test in
    the suite and every piped invocation actually runs in."""
    assert prompt_ui.interactive() is False


def test_non_tty_falls_back_to_the_typed_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, str]] = []

    def fake_prompt(text: str, default: str = "") -> str:
        asked.append((text, default))
        return "a"

    monkeypatch.setattr(prompt_ui.typer, "prompt", fake_prompt)

    assert prompt_ui.select(OPTIONS, default="a") == "a"
    assert asked == [("Choice", "a")]


def test_enter_takes_the_default_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, "\r")
    assert prompt_ui.select(OPTIONS, default="a") == "a"


def test_arrow_down_then_enter_takes_the_next_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, prompt_ui.DOWN, "\r")
    assert prompt_ui.select(OPTIONS, default="a") == "r"


def test_arrow_up_wraps_to_the_last_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, prompt_ui.UP, "\r")
    monkeypatch.setattr(prompt_ui.typer, "prompt", lambda *a, **k: "try again with the jam")

    assert prompt_ui.select(OPTIONS, default="a") == "try again with the jam"


def test_a_row_key_chooses_it_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accelerator is the same character the non-TTY path accepts as a
    typed answer — one value, both paths."""
    _keys(monkeypatch, "e")
    assert prompt_ui.select(OPTIONS, default="a") == "e"


def test_an_unbound_key_is_ignored_rather_than_chosen(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, "z", "\r")
    assert prompt_ui.select(OPTIONS, default="a") == "a"


def test_escape_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escape and Ctrl-C answer "r", not an exception: a plan someone
    escaped out of is a rejected plan, which every caller already handles."""
    _keys(monkeypatch, prompt_ui.ESC)
    assert prompt_ui.select(OPTIONS, default="a") == "r"


def test_ctrl_c_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, prompt_ui.CTRL_C)
    assert prompt_ui.select(OPTIONS, default="a") == "r"


def test_choosing_the_text_row_returns_what_was_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, prompt_ui.DOWN, prompt_ui.DOWN, prompt_ui.DOWN, "\r")
    monkeypatch.setattr(prompt_ui.typer, "prompt", lambda *a, **k: "  it's the other jam  ")

    assert prompt_ui.select(OPTIONS, default="a") == "it's the other jam"


CHOICES = [
    prompt_ui.Choice("consumption 1 jar jam"),
    prompt_ui.Choice("movement 1 tub ragu"),
    prompt_ui.Choice("discovery 1 bag rice"),
]


def test_multiselect_returns_none_without_a_terminal() -> None:
    assert prompt_ui.multiselect(CHOICES, title="Apply which changes?") is None


def test_multiselect_starts_with_everything_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, "\r")
    assert prompt_ui.multiselect(CHOICES, title="t") == [0, 1, 2]


def test_space_unchecks_the_row_under_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, prompt_ui.DOWN, " ", "\r")
    assert prompt_ui.multiselect(CHOICES, title="t") == [0, 2]


def test_n_then_space_selects_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, "n", prompt_ui.DOWN, prompt_ui.DOWN, " ", "\r")
    assert prompt_ui.multiselect(CHOICES, title="t") == [2]


def test_escape_cancels_the_checklist(monkeypatch: pytest.MonkeyPatch) -> None:
    _keys(monkeypatch, " ", prompt_ui.ESC)
    assert prompt_ui.multiselect(CHOICES, title="t") is None
