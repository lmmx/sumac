"""`prompt_ui`'s key reading, driven over a real pty.

`tests/test_prompt_ui.py` monkeypatches `read_key` to a scripted list, which
tests the menu's logic and nothing about how a keypress is actually read —
and a real Down arrow was misread as Escape (so it rejected the plan and
exited) for exactly as long as that was the only coverage. These tests write
the bytes a terminal really sends into a pty and let `read_key` do its job.

`os.fdopen(slave, "r")` matters: the bug was Python's buffered
`TextIOWrapper` reading ahead past the byte it returned, so a test that
substituted an unbuffered stand-in for `sys.stdin` would pass against the
broken code.
"""

from __future__ import annotations

import os
import pty
import sys
import threading
import time
from collections.abc import Iterator

import pytest

from sumac import prompt_ui


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """Yields the pty master fd; bytes written to it arrive as keypresses.
    `sys.stdin` is the slave, wrapped exactly the way the real one is."""
    master, slave = pty.openpty()
    stdin = os.fdopen(slave, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    try:
        yield master
    finally:
        stdin.close()
        os.close(master)


def _read(terminal: int, sequence: bytes) -> str:
    """Writes *after* entering key mode, never before: `raw_mode` changes the
    terminal with `TCSAFLUSH`, which discards anything already queued (see its
    docstring — typeahead from before a plan was drawn is not a decision about
    it). A test that wrote first would be racing that flush."""
    with prompt_ui.raw_mode():
        os.write(terminal, sequence)
        return prompt_ui.read_key()


def _type(terminal: int, *sequences: bytes, delay: float = 0.05) -> None:
    """Feeds keypresses from a thread, spaced out, so they arrive after the
    menu has entered key mode and drawn itself — which is also how a person
    types them, one at a time rather than as one buffered burst."""

    def run() -> None:
        for sequence in sequences:
            time.sleep(delay)
            os.write(terminal, sequence)

    threading.Thread(target=run, daemon=True).start()


def test_down_arrow_reads_as_down_not_escape(terminal: int) -> None:
    """The regression: three bytes in one burst, of which `sys.stdin.read(1)`
    returned the first and buffered the rest where `select` could not see
    them."""
    assert _read(terminal, b"\x1b[B") in prompt_ui.DOWN


def test_up_arrow_reads_as_up(terminal: int) -> None:
    assert _read(terminal, b"\x1b[A") in prompt_ui.UP


def test_application_mode_cursor_keys_are_recognized(terminal: int) -> None:
    """What a terminal in DECCKM (tmux, among others) sends for the same
    keys."""
    assert _read(terminal, b"\x1bOB") in prompt_ui.DOWN
    assert _read(terminal, b"\x1bOA") in prompt_ui.UP


def test_a_plain_character_reads_as_itself(terminal: int) -> None:
    assert _read(terminal, b"a") == "a"


def test_enter_reads_as_enter(terminal: int) -> None:
    assert _read(terminal, b"\r") in prompt_ui.ENTER


def test_a_lone_escape_stays_an_escape(terminal: int) -> None:
    """Nothing follows it, so the poll times out and it is a real Escape —
    the case the broken code got right by accident and the sequence case
    wrong for the same reason."""
    assert _read(terminal, b"\x1b") == prompt_ui.ESC


def test_a_split_escape_sequence_is_still_read_whole(terminal: int) -> None:
    """A sequence arriving in pieces, as it can over a slow link: the first
    read returns a partial, and the poll picks up the rest."""
    with prompt_ui.raw_mode():
        os.write(terminal, b"\x1b")
        time.sleep(0.01)
        os.write(terminal, b"[B")
        assert prompt_ui.read_key() in prompt_ui.DOWN


def test_select_moves_and_chooses_from_real_keypresses(
    terminal: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole menu, end to end, on the bytes a terminal sends: two downs
    then Enter lands on the third option."""
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    options = [
        prompt_ui.Option("a", "Accept"),
        prompt_ui.Option("r", "Reject"),
        prompt_ui.Option("e", "Edit"),
    ]
    _type(terminal, b"\x1b[B", b"\x1b[B", b"\r")

    assert prompt_ui.select(options, default="a") == "e"


def test_multiselect_toggles_from_real_keypresses(
    terminal: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    choices = [prompt_ui.Choice("one"), prompt_ui.Choice("two"), prompt_ui.Choice("three")]
    _type(terminal, b"\x1b[B", b" ", b"\r")

    assert prompt_ui.multiselect(choices, title="pick") == [0, 2]


def test_two_keypresses_arriving_together_are_read_separately(terminal: int) -> None:
    """Typed fast enough to land in the buffer together. A chunked read would
    return "ar" as one string, match no option, and drop both."""
    with prompt_ui.raw_mode():
        os.write(terminal, b"ar")
        assert prompt_ui.read_key() == "a"
        assert prompt_ui.read_key() == "r"


def test_an_escape_sequence_followed_by_a_key_does_not_swallow_it(terminal: int) -> None:
    with prompt_ui.raw_mode():
        os.write(terminal, b"\x1b[B\r")
        assert prompt_ui.read_key() in prompt_ui.DOWN
        assert prompt_ui.read_key() in prompt_ui.ENTER


def test_a_non_ascii_keypress_is_read_as_one_character(terminal: int) -> None:
    """Its continuation bytes must not come back as separate keypresses."""
    with prompt_ui.raw_mode():
        os.write(terminal, "é".encode())
        assert prompt_ui.read_key() == "é"


ROWS = [
    prompt_ui.Row("fridge", "Fridge  (fridge)", "Fridge fridge"),
    prompt_ui.Row("fridge-door", "Fridge > Door  (fridge-door)", "Fridge > Door fridge-door"),
    prompt_ui.Row("hob-left", "Hob Left > Top  (hob-left)", "Hob Left > Top hob-left"),
    prompt_ui.Row("pantry", "Pantry  (pantry)", "Pantry pantry"),
]


def test_pick_returns_the_row_under_the_cursor(
    terminal: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"\x1b[B", b"\r")

    assert prompt_ui.pick(ROWS, title="Where?") == "fridge-door"


def test_pick_starts_on_the_current_value(terminal: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening the picker on the location the write already names means
    Enter alone changes nothing."""
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"\r")

    assert prompt_ui.pick(ROWS, title="Where?", current="hob-left") == "hob-left"


def test_typing_filters_the_list(terminal: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A household's layout runs to dozens of locations; scrolling to one is
    not the interaction, typing a few letters of it is."""
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"h", b"o", b"b", b"\r")

    assert prompt_ui.pick(ROWS, title="Where?") == "hob-left"


def test_the_filter_matches_the_id_as_well_as_the_path(
    terminal: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"f", b"r", b"i", b"d", b"g", b"e", b"-", b"d", b"\r")

    assert prompt_ui.pick(ROWS, title="Where?") == "fridge-door"


def test_backspace_widens_the_filter_again(terminal: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"p", b"a", b"n", b"\x7f", b"\x7f", b"\x7f", b"\r")

    assert prompt_ui.pick(ROWS, title="Where?") == "fridge"


def test_enter_on_an_empty_filter_result_does_nothing(
    terminal: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing matches, so Enter cannot choose — backspacing brings the list
    back rather than the picker having exited on a phantom row."""
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"z", b"\r", b"\x7f", b"\r")

    assert prompt_ui.pick(ROWS, title="Where?") == "fridge"


def test_escape_cancels_the_picker(terminal: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)
    _type(terminal, b"\x1b")

    assert prompt_ui.pick(ROWS, title="Where?") is None


def test_pick_returns_none_without_a_terminal() -> None:
    assert prompt_ui.pick(ROWS, title="Where?") is None


def test_pick_of_an_empty_list_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_ui, "interactive", lambda: True)

    assert prompt_ui.pick([], title="Where?") is None
