"""Smoke test for `scripts/preview-ask-ui.py`.

The harness is how a rendering change is compared, so if it breaks — a
renamed `render` function, a changed `ProposedWrite` field — the comparison
breaks with it, silently. This runs every scene against a throwaway console
and asserts each drew something.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from rich.console import Console

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "preview-ask-ui.py"


@pytest.fixture(autouse=True)
def _restore_consoles() -> Iterator[None]:
    """Both the harness and these tests reassign `render.console`, a module
    global every other test's `CliRunner` output goes through. Restoring it
    keeps this file's position in the run order from affecting the rest of the
    suite."""
    from sumac import prompt_ui, render

    saved = (render.console, prompt_ui.console)
    yield
    render.console, prompt_ui.console = saved


@pytest.fixture(scope="module")
def preview() -> ModuleType:
    spec = importlib.util.spec_from_file_location("preview_ask_ui", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_scene_renders(preview: ModuleType) -> None:
    from sumac import prompt_ui, render

    for name, scene in preview.SCENES.items():
        console = Console(record=True, width=96)
        render.console = console
        prompt_ui.console = console
        scene()
        assert console.export_text().strip(), f"scene {name!r} drew nothing"


def test_the_fabricated_scene_is_flagged_as_ungrounded(preview: ModuleType) -> None:
    """The scene demonstrates `review`'s one check that `decide` cannot make;
    a rendering change that dropped the badge would leave the scene showing
    nothing."""
    from sumac import prompt_ui, render

    console = Console(record=True, width=96)
    render.console = console
    prompt_ui.console = console
    preview.scene_fabricated()

    assert "[unverified]" in console.export_text()
