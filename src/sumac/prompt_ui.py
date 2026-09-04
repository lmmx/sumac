"""Keypress-driven pickers for `sumac ask`'s decision prompts, with the
line-typed prompt they replace kept as the non-TTY path.

Two rules shape everything here:

*Nothing behaves differently when stdin is not a terminal.* `tests/test_cli.py`
drives every `ask` test through `CliRunner(input=...)` and `evals/` runs
headless; both feed a line per decision, on the same keys the option table has
always printed. `select` checks `interactive()` first and, when it is false,
prints that same table and reads that same line — the keys, their meanings, and
the free-text-is-feedback fallback are identical on both paths, so a script, a
pipe, and a test see exactly what they saw before this module existed.

*No new dependency.* Raw-mode key reads are `termios`/`tty` from the standard
library, guarded at import so a platform without them (Windows) simply never
reports itself interactive; the rendering is `rich`, already a hard dependency
of the package.
"""

from __future__ import annotations

import os
import select as _select
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import typer
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from sumac.render import console, print_decision_options

try:  # POSIX only — a platform without it never reports itself interactive.
    import termios
    import tty

    _RAW_MODE_AVAILABLE = True
except ImportError:  # pragma: no cover - Windows
    _RAW_MODE_AVAILABLE = False


# Both cursor-key encodings: a terminal in normal mode sends `ESC [ A`, and one
# in application-cursor mode (`DECCKM`, which tmux and some terminals turn on)
# sends `ESC O A` for the same key. Accepting only the first is another way for
# an arrow key to silently do nothing.
UP = ("\x1b[A", "\x1bOA")
DOWN = ("\x1b[B", "\x1bOB")
ENTER = ("\r", "\n")
ESC = "\x1b"
CTRL_C = "\x03"
SPACE = " "

# A read that returned exactly this much of an escape sequence has more of it
# still coming — a terminal usually delivers all three bytes in one burst, but
# over ssh or a slow link they can arrive split across reads.
_PARTIAL_ESCAPES = (b"\x1b", b"\x1b[", b"\x1bO")
_ESCAPE_TIMEOUT = 0.05


def _utf8_continuation_bytes(lead: int) -> int:
    """How many bytes follow a UTF-8 lead byte. No menu key is non-ASCII, so
    this only keeps such a keypress from arriving as replacement characters
    and, worse, leaving its continuation bytes in the buffer to be read as
    keys of their own."""
    if lead >= 0xF0:
        return 3
    if lead >= 0xE0:
        return 2
    if lead >= 0xC0:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class Option:
    """One row of a `select` menu. `key` is what a typed answer must equal on
    the non-TTY path *and* the accelerator character on the interactive one —
    the two paths never drift apart because there is only the one value.
    `prompt_for_text` marks the row whose meaning is "type something instead
    of choosing" (`sumac ask`'s free-text feedback), which is the one option
    a single keystroke cannot express: choosing it opens a line editor and
    `select` returns what was typed."""

    key: str
    description: str
    prompt_for_text: bool = False
    text_prompt: str = "Feedback"


def interactive() -> bool:
    """Whether a single keypress can be read at all: raw mode available, and
    both stdin and stdout attached to a terminal. Checked before every
    keypress-driven interaction rather than once at import, so a caller under
    a `CliRunner` in the same process as a real terminal still gets the
    line-typed path."""
    if not _RAW_MODE_AVAILABLE:
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
        # Probes what `raw_mode` will need before promising a menu: an
        # `isatty()` that says yes over a stdin whose attributes cannot
        # actually be read would otherwise reach `tty.setcbreak` and raise
        # there, mid-decision, instead of falling back to the typed prompt.
        termios.tcgetattr(sys.stdin.fileno())
    except (AttributeError, ValueError, OSError, termios.error):
        return False
    return True


@contextmanager
def raw_mode() -> Iterator[None]:
    """Puts the terminal in single-keypress mode for the length of one menu,
    not one keypress.

    Held across the whole loop on purpose: restoring canonical mode between
    reads means a keystroke arriving while the menu is redrawing is echoed to
    the screen and line-buffered by the tty, so fast keypresses corrupt the
    `Live` region and go missing until Enter.

    `tty.setcbreak`, not `tty.setraw`: `setraw` also clears `OPOST`, which is
    what turns `\n` into `\r\n` on the way out — every line `Live` redraws
    inside a `setraw` block staircases across the screen. `setcbreak` clears
    only `ECHO`/`ICANON`, which is all a single-keypress read needs. It also
    leaves `ISIG` on, so Ctrl-C stays a `KeyboardInterrupt` rather than
    arriving as a `\x03` byte — `select`/`multiselect` catch it and treat it
    the same way."""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # TCSAFLUSH discards whatever is already sitting in the input queue as
        # the mode changes — deliberate here, and the safer of the two: model
        # inference runs for seconds before a plan appears, and anything typed
        # into that wait was not a decision about a plan nobody had seen yet.
        # A stray Enter from before the preview must not accept it.
        tty.setcbreak(fd, termios.TCSAFLUSH)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def read_key() -> str:
    """One keypress, with an escape sequence returned whole. Call inside
    `raw_mode()`.

    Reads the file descriptor with `os.read`, not `sys.stdin.read`. That is
    the whole point of this function: `sys.stdin` is a buffered
    `TextIOWrapper`, and `sys.stdin.read(1)` on a terminal pulls every byte
    already available into Python's own buffer before returning the first
    one. An arrow key's three bytes arrive in a single burst, so after
    returning `\x1b` the remaining `[B` sits in the wrapper's buffer where
    `select.select` — which polls the file descriptor — cannot see it. The
    sequence then reads as a bare Escape, which `select()` answers as
    "reject", so pressing Down silently discarded the plan and returned to
    the shell. Found by a real run, not by a test: every test in
    `tests/test_prompt_ui.py` monkeypatched this function out. `tests/
    test_prompt_ui_pty.py` now drives it over a real pty instead.

    One byte at a time, extended only for an escape sequence or a multi-byte
    UTF-8 character. Reading a larger chunk would be fewer syscalls and is
    wrong: two keypresses already waiting in the tty buffer — typed quickly,
    or arriving while the menu redrew — would come back merged into one
    string that matches no key, and both would be dropped. The poll covers an
    escape sequence split across reads, as one can be over a slow link."""
    fd = sys.stdin.fileno()
    data = os.read(fd, 1)
    if data == ESC.encode():
        while data in _PARTIAL_ESCAPES and _select.select([fd], [], [], _ESCAPE_TIMEOUT)[0]:
            data += os.read(fd, 1)
    elif data and (extra := _utf8_continuation_bytes(data[0])):
        data += os.read(fd, extra)
    return data.decode("utf-8", errors="replace")


def _menu(options: list[Option], cursor: int, title: str | None, hint: str) -> Group:
    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    for i, option in enumerate(options):
        selected = i == cursor
        marker = Text("❯" if selected else " ", style="bold cyan")
        key = Text(option.key, style="bold cyan" if selected else "cyan")
        description = Text(option.description, style="bold" if selected else "dim")
        table.add_row(marker, key, description)
    parts = [table, Text(hint, style="dim")]
    if title:
        parts.insert(0, Text(title, style="bold"))
    return Group(*parts)


def _index_of(options: list[Option], key: str) -> int:
    for i, option in enumerate(options):
        if option.key.lower() == key.lower():
            return i
    return 0


def select(
    options: list[Option],
    *,
    default: str,
    title: str | None = None,
    prompt: str = "Choice",
) -> str:
    """The chosen option's `key`, or — for a `prompt_for_text` option — the
    line the person typed after choosing it. `default` is the key the cursor
    starts on, and the value a bare Enter produces on the non-TTY path, so
    "accept" stays one keystroke either way.

    A returned string is exactly what `typer.prompt(...).strip()` returned
    before: callers keep matching on `"a"`/`"r"`/`"e"` and treating anything
    unmatched as feedback."""
    if not interactive():
        print_decision_options([(o.key, o.description) for o in options])
        return typer.prompt(prompt, default=default).strip()

    cursor = _index_of(options, default)
    hint = "↑/↓ move · enter choose · esc reject"
    with (
        raw_mode(),
        Live(_menu(options, cursor, title, hint), console=console, auto_refresh=False) as live,
    ):
        while True:
            try:
                key = read_key()
            except (KeyboardInterrupt, EOFError):
                key = CTRL_C
            if key in UP or key == "k":
                cursor = (cursor - 1) % len(options)
            elif key in DOWN or key == "j":
                cursor = (cursor + 1) % len(options)
            elif key in (ESC, CTRL_C):
                # Same answer a typed "r" gives: reject this proposal and
                # write nothing. Never an exception, so the caller's own
                # accept/reject branch handles it like any other decision.
                return "r"
            elif key in ENTER:
                break
            else:
                match = next((o for o in options if o.key.lower() == key.lower()), None)
                if match is None:
                    continue
                cursor = options.index(match)
                break
            live.update(_menu(options, cursor, title, hint), refresh=True)

    chosen = options[cursor]
    # Redrawn once outside the Live block so the transcript keeps a record of
    # what was chosen — a Live region that simply disappears leaves a scrollback
    # in which the decision that led to a commit is nowhere to be seen.
    console.print(f"[cyan]❯[/cyan] {chosen.description}")
    if chosen.prompt_for_text:
        return typer.prompt(chosen.text_prompt).strip()
    return chosen.key


@dataclass(frozen=True, slots=True)
class Choice:
    """One row of a `multiselect` list: `label` is what the person reads,
    `checked` its initial state."""

    label: str
    checked: bool = True


def _checklist(choices: list[Choice], checked: list[bool], cursor: int, title: str) -> Group:
    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    for i, choice in enumerate(choices):
        selected = i == cursor
        table.add_row(
            Text("❯" if selected else " ", style="bold cyan"),
            Text("[x]" if checked[i] else "[ ]", style="green" if checked[i] else "dim"),
            Text(choice.label, style="bold" if selected else ""),
        )
    return Group(
        Text(title, style="bold"),
        table,
        Text("↑/↓ move · space toggle · a all · n none · enter apply · esc cancel", style="dim"),
    )


def multiselect(choices: list[Choice], *, title: str) -> list[int] | None:
    """The indices left checked, or `None` if the person cancelled. Returns
    `None` immediately when not interactive rather than falling back to a
    typed equivalent: `sumac ask` only ever offers this option on a TTY (see
    `cli.py`'s `_decision_options`), so there is no non-TTY caller to serve,
    and inventing a line-typed index syntax for one would be a second
    interface to keep in step with this one for no existing user."""
    if not interactive():
        return None

    checked = [c.checked for c in choices]
    cursor = 0
    with (
        raw_mode(),
        Live(
            _checklist(choices, checked, cursor, title), console=console, auto_refresh=False
        ) as live,
    ):
        while True:
            try:
                key = read_key()
            except (KeyboardInterrupt, EOFError):
                key = CTRL_C
            if key in UP or key == "k":
                cursor = (cursor - 1) % len(choices)
            elif key in DOWN or key == "j":
                cursor = (cursor + 1) % len(choices)
            elif key == SPACE:
                checked[cursor] = not checked[cursor]
            elif key == "a":
                checked = [True] * len(choices)
            elif key == "n":
                checked = [False] * len(choices)
            elif key in (ESC, CTRL_C):
                return None
            elif key in ENTER:
                break
            live.update(_checklist(choices, checked, cursor, title), refresh=True)

    return [i for i, on in enumerate(checked) if on]
