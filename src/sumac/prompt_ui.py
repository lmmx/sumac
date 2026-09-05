"""Keypress-driven pickers for `sumac ask`'s decision prompts, with the
line-typed prompt they replace kept as the non-TTY path.

Two rules shape everything here:

*Nothing behaves differently when stdin is not a terminal.* `tests/test_cli.py`
drives every `ask` test through `CliRunner(input=...)` and `evals/` runs
headless; both feed a line per decision, on the same keys the option table has
always printed. `select` checks `interactive()` first and, when it is false,
prints that same table and reads that same line. The keys, their meanings, and
the free-text-is-feedback fallback are identical on both paths, so a script, a
pipe, and a test read the same as before this module existed.

*No new dependency.* Raw-mode key reads are `termios`/`tty` from the standard
library, guarded at import so a platform without them (Windows) never reports
itself interactive; the rendering is `rich`, already a hard dependency of the
package.
"""

from __future__ import annotations

import os
import select as _select
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import typer
from rich.console import Group, RenderableType
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
# sends `ESC O A` for the same key. Accepting only the first leaves the arrow
# key doing nothing under those terminals.
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
    this only prevents such a keypress arriving as replacement characters and
    leaving its continuation bytes in the buffer to be read as further
    keypresses."""
    if lead >= 0xF0:
        return 3
    if lead >= 0xE0:
        return 2
    if lead >= 0xC0:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class Option:
    """One row of a `select` menu. `key` is both what a typed answer must
    equal on the non-TTY path and the accelerator character on the interactive
    one, so the two paths cannot diverge. `prompt_for_text` marks a row that
    takes typed input rather than being chosen outright (`sumac ask`'s
    free-text feedback): choosing it opens a line editor and `select` returns
    what was typed."""

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
        # Checks what `raw_mode` will need before a menu is offered: an
        # `isatty()` that returns True over a stdin whose attributes cannot be
        # read would otherwise raise inside `tty.setcbreak` mid-decision,
        # instead of falling back to the typed prompt.
        termios.tcgetattr(sys.stdin.fileno())
    except (AttributeError, ValueError, OSError, termios.error):
        return False
    return True


@contextmanager
def raw_mode() -> Iterator[None]:
    """Puts the terminal in single-keypress mode for the length of one menu,
    not one keypress.

    Held across the whole loop deliberately: restoring canonical mode between
    reads means a keystroke arriving while the menu is redrawing is echoed to
    the screen and line-buffered by the tty, so fast keypresses corrupt the
    `Live` region and are not delivered until Enter.

    `tty.setcbreak`, not `tty.setraw`: `setraw` also clears `OPOST`, which
    converts `\n` to `\r\n` on output, so every line `Live` redraws inside a
    `setraw` block is indented one column further than the last. `setcbreak`
    clears only `ECHO`/`ICANON`, which is all a single-keypress read requires.
    It also leaves `ISIG` set, so Ctrl-C raises `KeyboardInterrupt` rather than
    arriving as a `\x03` byte; `select`/`multiselect` catch it and treat it
    the same way."""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # TCSAFLUSH discards whatever is already queued as the mode changes.
        # Deliberate: model inference runs for seconds before a plan appears,
        # and anything typed during that wait was not a decision about a plan
        # that had not yet been displayed. A stray Enter from before the
        # preview must not accept it.
        tty.setcbreak(fd, termios.TCSAFLUSH)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def read_key() -> str:
    """One keypress, with an escape sequence returned whole. Call inside
    `raw_mode()`.

    Reads the file descriptor with `os.read`, not `sys.stdin.read`, which is
    the reason this function exists: `sys.stdin` is a buffered
    `TextIOWrapper`, and `sys.stdin.read(1)` on a terminal pulls every byte
    already available into Python's own buffer before returning the first
    one. An arrow key's three bytes arrive in a single burst, so after
    returning `\x1b` the remaining `[B` remains in the wrapper's buffer, where
    `select.select` — which polls the file descriptor — cannot detect it. The
    sequence then reads as a bare Escape, which `select()` answers as
    "reject", so pressing Down discarded the plan and returned to the shell.
    Found by a real run rather than by a test: every test in
    `tests/test_prompt_ui.py` monkeypatched this function out.
    `tests/test_prompt_ui_pty.py` now drives it over a real pty.

    One byte at a time, extended only for an escape sequence or a multi-byte
    UTF-8 character. A larger chunk would be fewer syscalls but incorrect:
    two keypresses already waiting in the tty buffer — typed quickly, or
    arriving while the menu redrew — would be returned merged into one string
    matching no key, and both would be dropped. The poll covers an escape
    sequence split across reads, which can happen over a slow link."""
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
    "accept" takes one keystroke on both paths.

    The returned string matches what `typer.prompt(...).strip()` returned
    before: callers still match on `"a"`/`"r"`/`"e"` and treat anything
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
                # The same answer a typed "r" gives: reject this proposal
                # and write nothing. Not an exception, so the caller's
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
    # Redrawn once outside the Live block so the transcript records what was
    # chosen: a Live region that disappears leaves no record in the scrollback
    # of the decision that led to a commit.
    console.print(f"[cyan]❯[/cyan] {chosen.description}")
    if chosen.prompt_for_text:
        return typer.prompt(chosen.text_prompt).strip()
    return chosen.key


# How many rows of a long list are on screen at once. The rest scroll under
# the cursor, with a count of what is off each end: a list of every location
# in a household runs to dozens, and a menu taller than the terminal would
# scroll its own title off the screen.
_PICK_HEIGHT = 12
BACKSPACE = ("\x7f", "\b")


@dataclass(frozen=True, slots=True)
class Row:
    """One row of a `pick` list. `value` is what `pick` returns, `label` what
    is shown, and `search` what typing filters against — for a location, its
    path and its id, so either matches."""

    value: str
    label: str
    search: str = ""

    def matches(self, needle: str) -> bool:
        return needle in (self.search or self.label).lower()


def _pick_view(
    rows: list[Row], cursor: int, filter_text: str, title: str, total: int, hint: str = ""
) -> Group:
    start = 0
    if len(rows) > _PICK_HEIGHT:
        start = max(0, min(cursor - _PICK_HEIGHT // 2, len(rows) - _PICK_HEIGHT))
    window = rows[start : start + _PICK_HEIGHT]

    table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
    for i, row in enumerate(window, start=start):
        selected = i == cursor
        table.add_row(
            Text("❯" if selected else " ", style="bold cyan"),
            Text(row.label, style="bold" if selected else ""),
        )

    parts: list[RenderableType] = [Text(title, style="bold")]
    typed = f"filter: {filter_text}" if filter_text else "type to filter"
    shown = f"{len(rows)} of {total}" if len(rows) != total else f"{total}"
    parts.append(Text(f"{typed}   [{shown}]", style="dim"))
    if start:
        parts.append(Text(f"  ↑ {start} more", style="dim"))
    parts.append(table)
    if start + _PICK_HEIGHT < len(rows):
        parts.append(Text(f"  ↓ {len(rows) - start - _PICK_HEIGHT} more", style="dim"))
    if not rows:
        parts.append(Text("  (nothing matches)", style="yellow"))
    parts.append(Text(hint or "↑/↓ move · type to filter · enter choose · esc cancel", style="dim"))
    return Group(*parts)


def _visible_rows(rows: list[Row], filter_text: str, allow_new: bool, new_hint: str) -> list[Row]:
    """The rows a filter leaves, plus — when `allow_new` is set and the filter
    names something not already in the list — one row carrying the typed text.

    That row goes last, not first. The cursor resets to the top on every
    keystroke, so typing a value that already exists selects it with one
    Enter. When nothing matches, the added row is the only row and is already
    under the cursor, so typing a new unit or product and pressing Enter
    takes two steps rather than three."""
    needle = filter_text.lower()
    matches = [row for row in rows if row.matches(needle)] if filter_text else list(rows)
    if allow_new and filter_text and not any(row.value == filter_text for row in rows):
        matches.append(Row(value=filter_text, label=f'+ "{filter_text}"  ({new_hint})'))
    return matches


def pick(
    rows: list[Row],
    *,
    title: str,
    current: str | None = None,
    allow_new: bool = False,
    new_hint: str = "new",
) -> str | None:
    """One value from a long list, filtered as you type. The counterpart to
    `select`, which handles a handful of fixed options each with its own
    accelerator key. Here there are no accelerators — every printable key is
    filter text — plus a scrolling window, and the cursor starts on `current`
    when that value is in the list.

    `allow_new` adds a row for whatever has been typed when it is not already
    in the list, so a value the vault has never recorded is still reachable.
    A product or a unit may legitimately be new — `decide` registers one on
    first use — while a location may not: `decide` rejects any location that
    is not configured.

    Returns `None` when cancelled, and — like `multiselect` — when there is
    no terminal to read keypresses from: a caller wanting a value off a
    terminal asks for one its own way."""
    if not interactive() or (not rows and not allow_new):
        return None

    filter_text = ""
    footer = (
        "↑/↓ move · type to filter or add · enter choose · esc cancel"
        if allow_new
        else "↑/↓ move · type to filter · enter choose · esc cancel"
    )
    visible = _visible_rows(rows, filter_text, allow_new, new_hint)
    cursor = next((i for i, row in enumerate(visible) if row.value == current), 0)

    with (
        raw_mode(),
        Live(
            _pick_view(visible, cursor, filter_text, title, len(rows), footer),
            console=console,
            auto_refresh=False,
        ) as live,
    ):
        while True:
            try:
                key = read_key()
            except (KeyboardInterrupt, EOFError):
                key = CTRL_C
            if key in UP:
                cursor = (cursor - 1) % len(visible) if visible else 0
            elif key in DOWN:
                cursor = (cursor + 1) % len(visible) if visible else 0
            elif key in (ESC, CTRL_C):
                return None
            elif key in ENTER:
                if visible:
                    break
            elif key in BACKSPACE:
                filter_text = filter_text[:-1]
                visible = _visible_rows(rows, filter_text, allow_new, new_hint)
                cursor = 0
            elif len(key) == 1 and key >= " ":
                filter_text += key
                visible = _visible_rows(rows, filter_text, allow_new, new_hint)
                cursor = 0
            live.update(_pick_view(visible, cursor, filter_text, title, len(rows)), refresh=True)

    chosen = visible[cursor]
    console.print(f"[cyan]❯[/cyan] {chosen.label}")
    return chosen.value


_DIGITS = "0123456789."


def _as_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _plain(value: Decimal) -> str:
    """Decimal formatting without an exponent: `format(..., "f")` rather than
    `str()`, which uses scientific notation for some values a repeated
    decrement produces."""
    return format(value, "f")


def _number_view(text: str, title: str, hint: str, valid: bool) -> Group:
    return Group(
        Text(title, style="bold"),
        Text(
            f"  {text or '—'}" + ("" if valid else "   not a number"),
            style="bold" if valid else "yellow",
        ),
        Text(hint, style="dim"),
    )


def number(
    current: str,
    *,
    title: str,
    step: Decimal = Decimal(1),
    minimum: Decimal = Decimal(0),
) -> str | None:
    """A numeric field. Returns the accepted value, or `None` if cancelled or
    there is no terminal.

    A key that is not a digit, a decimal point, or a control this widget
    handles does nothing. Enter does nothing until what has been typed
    parses as a decimal, so an invalid value cannot be returned. The previous
    field was a free-text prompt, which accepted "three" and reported it
    invalid several keystrokes later, while the edit was being applied.

    Up and Down step by `step`, which covers the common edit: a quantity is
    usually wrong by one. `minimum` clamps the stepping (not below zero;
    `decide` rejects a non-positive amount) but does not restrict typing,
    since a partly-typed number passes through values a clamp would
    change."""
    if not interactive():
        return None

    text = current
    hint = "↑/↓ adjust · digits to type · enter accept · esc cancel"
    with (
        raw_mode(),
        Live(
            _number_view(text, title, hint, _as_decimal(text) is not None),
            console=console,
            auto_refresh=False,
        ) as live,
    ):
        while True:
            try:
                key = read_key()
            except (KeyboardInterrupt, EOFError):
                key = CTRL_C
            if key in UP or key in DOWN:
                delta = step if key in UP else -step
                text = _plain(max(minimum, (_as_decimal(text) or Decimal(0)) + delta))
            elif key in (ESC, CTRL_C):
                return None
            elif key in ENTER:
                if _as_decimal(text) is not None:
                    break
            elif key in BACKSPACE:
                text = text[:-1]
            elif len(key) == 1 and key in _DIGITS:
                text += key
            live.update(
                _number_view(text, title, hint, _as_decimal(text) is not None), refresh=True
            )

    console.print(f"[cyan]❯[/cyan] {text}")
    return text


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
    typed equivalent: `sumac ask` offers this option only on a TTY (see
    `cli.py`'s `_decision_options`), so there is no non-TTY caller for it, and
    a line-typed index syntax would be a second interface to maintain with no
    user."""
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
