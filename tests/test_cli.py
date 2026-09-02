from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import pytest
from sealedlog import Vault
from sealedlog.errors import WrongPassphraseError
from typer.testing import CliRunner

from sumac import ledger, llm, paths, queue, store
from sumac import vault as sumac_vault
from sumac.cli import app
from sumac.errors import RetireNonemptyError, VaultExistsError
from sumac.models import ChangeKind

runner = CliRunner()
PASSPHRASE_ENV = {"SUMAC_PASSPHRASE": "test-pass"}


def _real_key(data_dir: Path) -> bytes:
    """The actual key behind a data dir `_run(data_dir, "init")` created —
    derived directly rather than via `sumac.passphrase`, whose env-var
    resolution only applies inside a `CliRunner.invoke` call, not after."""
    vault = Vault.from_dict(json.loads(paths.vault_path(data_dir).read_text(encoding="utf-8")))
    return sumac_vault.unlock(vault, PASSPHRASE_ENV["SUMAC_PASSPHRASE"])


def _append_raw_change(
    data_dir: Path,
    actor: str,
    product_id: str,
    amount: str,
    unit: str,
    *,
    to_location: str,
) -> None:
    """Appends a change bypassing `decide` entirely — for tests that need a
    record `sumac add` would now reject (e.g. an unregistered location), to
    exercise the read-side tolerance (`status`/`find`/`doctor`) rather than
    the write-side gate."""
    key = _real_key(data_dir)
    store.append(
        data_dir,
        key,
        f"log:{actor}",
        {
            "schema_version": 1,  # v1 shape (kind/quantity/*_location), not SCHEMA_VERSION
            "type": "change",
            "id": "raw-1",
            "ts": datetime.now(UTC).isoformat(),
            "actor": actor,
            "supersedes": None,
            "payload": {
                "kind": "purchase",
                "product_id": product_id,
                "quantity": {"amount": amount, "unit": unit},
                "from_location": None,
                "to_location": to_location,
                "metadata": {},
            },
        },
    )


@pytest.fixture(autouse=True)
def _osuser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getuser", lambda: "alice")


def _run(
    data_dir: Path, *args: str, env: dict[str, str] = PASSPHRASE_ENV, input: str | None = None
):
    return runner.invoke(app, [*args, "--data-dir", str(data_dir)], env=env, input=input)


def test_init_creates_vault(data_dir: Path) -> None:
    result = _run(data_dir, "init")
    assert result.exit_code == 0, result.output
    assert paths.vault_path(data_dir).exists()
    assert paths.log_dir(data_dir).exists()


def test_init_twice_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "init")
    assert result.exit_code != 0
    assert isinstance(result.exception, VaultExistsError)


def test_wrong_passphrase_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "show", env={"SUMAC_PASSPHRASE": "nope"})
    assert result.exit_code != 0
    assert isinstance(result.exception, WrongPassphraseError)


def test_add_location_and_show(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    assert "Fridge" in result.output


def test_config_show_locations_only_omits_products(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    result = _run(data_dir, "config", "show", "--locations-only")
    assert result.exit_code == 0, result.output
    assert "Fridge" in result.output
    assert "Milk" not in result.output


def test_config_show_products_only_omits_locations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    result = _run(data_dir, "config", "show", "--products-only")
    assert result.exit_code == 0, result.output
    assert "Milk" in result.output
    assert "Fridge" not in result.output


def test_config_show_both_flags_rejected(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "show", "--locations-only", "--products-only")
    assert result.exit_code != 0


def test_retire_location_shows_in_config(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    result = _run(data_dir, "config", "retire-location", "fridge")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert "retired" in result.output


def test_retire_unknown_location_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "retire-location", "nonexistent")
    assert result.exit_code != 0


def test_retire_nonempty_location_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "config", "retire-location", "pantry")
    assert result.exit_code != 0
    assert isinstance(result.exception, RetireNonemptyError)
    assert "milk" in str(result.exception)


def test_retire_location_with_stock_only_in_sublocation_succeeds(data_dir: Path) -> None:
    """`retire-location` checks the named location's own holdings, not its
    sub-locations' — each sub-location is retired (and checked) on its own."""
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Door", "--id", "fridge-door", "--parent", "fridge")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge-door")
    result = _run(data_dir, "config", "retire-location", "fridge")
    assert result.exit_code == 0, result.output


def test_add_product_and_show(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    assert "Milk" in result.output


def test_retire_product_shows_in_config(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    result = _run(data_dir, "config", "retire-product", "milk")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert "retired" in result.output


def test_retire_product_with_stock_succeeds(data_dir: Path) -> None:
    """Unlike a location, retiring a product is permitted at any time."""
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "config", "add-product", "Milk", "l", "--id", "milk")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    status = _run(data_dir, "status")
    assert "milk" in status.output, "setup didn't actually produce stock"

    result = _run(data_dir, "config", "retire-product", "milk")
    assert result.exit_code == 0, result.output


def test_retire_unknown_product_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "retire-product", "nonexistent")
    assert result.exit_code != 0


def test_check_units_clean_when_nothing_observed(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 0
    assert "every observed" in result.output


def test_check_units_suggests_command_for_unregistered_product(data_dir: Path) -> None:
    """`sumac add` now auto-registers on first use (Phase 3), so an
    unregistered product can no longer arise through it — check-units is a
    legacy-data tool now. Simulate a pre-decide record directly."""
    _run(data_dir, "init")
    _append_raw_change(data_dir, "alice", "milk", "1", "l", to_location="pantry")
    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 1
    assert "milk" in result.output
    assert "add-product" in result.output


def test_check_units_flags_unconvertible_unit_for_registered_product(data_dir: Path) -> None:
    """Same reasoning: decide now rejects unit_unconvertible at write time,
    so this shape is legacy-data-only too."""
    _run(data_dir, "init")
    _run(data_dir, "config", "add-product", "Flour", "kg", "--id", "flour")
    _append_raw_change(data_dir, "alice", "flour", "1", "lb", to_location="pantry")
    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 1
    assert "lb" in result.output
    assert "unregistered" not in result.output.lower()


def test_check_units_clean_when_registered_and_convertible(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "config", "add-product", "Flour", "kg", "--id", "flour")
    add_result = _run(data_dir, "add", "purchase", "flour", "1", "kg", "--to", "pantry")
    assert add_result.exit_code == 0, add_result.output

    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 0
    assert "every observed" in result.output


def test_check_units_reports_unconfirmed_auto_registration(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    add_result = _run(data_dir, "add", "purchase", "kimchi", "1", "jar", "--to", "pantry")
    assert add_result.exit_code == 0, add_result.output

    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 1
    assert "never confirmed" in result.output
    assert "kimchi" in result.output


def test_check_units_does_not_flag_deliberately_registered_product(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "config", "add-product", "Kimchi", "jar", "--id", "kimchi")
    add_result = _run(data_dir, "add", "purchase", "kimchi", "1", "jar", "--to", "pantry")
    assert add_result.exit_code == 0, add_result.output

    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 0
    assert "never confirmed" not in result.output


def test_check_units_confirming_an_auto_registration_clears_the_flag(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "kimchi", "1", "jar", "--to", "pantry")

    # confirm it — redefining clears the auto marker, per §3.5a
    _run(data_dir, "config", "add-product", "Kimchi", "jar", "--id", "kimchi")

    result = _run(data_dir, "config", "check-units")
    assert result.exit_code == 0
    assert "never confirmed" not in result.output


def test_add_change_and_status(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    result = _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "pantry")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "status")
    assert result.exit_code == 0
    assert "milk" in result.output
    assert "2" in result.output


def test_snapshot_and_find(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    result = _run(data_dir, "snapshot", "fridge", "milk=3/l")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "find", "milk")
    assert result.exit_code == 0
    assert "Fridge" in result.output


def test_find_shows_anomaly_banner(data_dir: Path) -> None:
    _run(data_dir, "init")
    _append_raw_change(data_dir, "alice", "milk", "1", "l", to_location="hob-right-below-bottom")
    result = _run(data_dir, "find", "milk")
    assert "could not be applied" in result.output


def _seed_butter_tiers(data_dir: Path) -> None:
    """One product per tier for a "butter" query: `Butter` (exact),
    `Salted Butter` (whole-word), `Butternut Squash` (substring only —
    "butter" isn't a whole word inside "butternut")."""
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "add", "purchase", "Butter", "1", "packet", "--to", "fridge")
    _run(data_dir, "add", "purchase", "Salted Butter", "1", "block", "--to", "fridge")
    _run(data_dir, "add", "purchase", "Butternut Squash", "1", "pack", "--to", "fridge")


def test_find_shows_one_table_per_match_kind_in_tier_order(data_dir: Path) -> None:
    _seed_butter_tiers(data_dir)

    result = _run(data_dir, "find", "Butter")

    assert result.exit_code == 0, result.output
    for heading in ("Exact matches", "Whole-word matches", "Substring matches"):
        assert heading in result.output
    exact_idx = result.output.index("Exact matches")
    whole_idx = result.output.index("Whole-word matches")
    substring_idx = result.output.index("Substring matches")
    assert exact_idx < whole_idx < substring_idx


def test_find_omits_tables_for_absent_tiers(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "add", "purchase", "Butter", "1", "packet", "--to", "fridge")

    result = _run(data_dir, "find", "Butter")

    assert "Exact matches" in result.output
    assert "Whole-word matches" not in result.output
    assert "Substring matches" not in result.output


def test_find_no_only_flags_shows_every_kind(data_dir: Path) -> None:
    _seed_butter_tiers(data_dir)

    result = _run(data_dir, "find", "Butter")

    assert "Salted Butter" in result.output
    assert "Butternut Squash" in result.output


def test_find_single_only_flag_restricts_to_that_kind(data_dir: Path) -> None:
    _seed_butter_tiers(data_dir)

    result = _run(data_dir, "find", "Butter", "--exact")

    assert "Exact matches" in result.output
    assert "Whole-word matches" not in result.output
    assert "Substring matches" not in result.output
    assert "Salted Butter" not in result.output
    assert "Butternut Squash" not in result.output


def test_find_combined_only_flags_are_unioned(data_dir: Path) -> None:
    """Passing --exact and --whole-word together means "only these two
    kinds", not "only records passing both" — substring is still excluded."""
    _seed_butter_tiers(data_dir)

    result = _run(data_dir, "find", "Butter", "--exact", "--whole-word")

    assert "Exact matches" in result.output
    assert "Whole-word matches" in result.output
    assert "Substring matches" not in result.output
    assert "Salted Butter" in result.output
    assert "Butternut Squash" not in result.output


def test_find_substring_only_flag_shows_only_substring_tier(data_dir: Path) -> None:
    _seed_butter_tiers(data_dir)

    result = _run(data_dir, "find", "Butter", "--substring")

    assert "Exact matches" not in result.output
    assert "Whole-word matches" not in result.output
    assert "Substring matches" in result.output
    assert "Butternut Squash" in result.output


def test_find_no_match_reports_not_found(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "find", "nonexistent")
    assert result.exit_code == 0, result.output
    assert "not found" in result.output


def test_verify_clean(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "verify")
    assert result.exit_code == 0
    assert "verified" in result.output


def test_verify_detects_tampering(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    log_path = paths.log_path(data_dir, "alice")
    log_path.write_text("not-valid-base64!!!\n")
    result = _run(data_dir, "verify")
    assert result.exit_code != 0


def test_doctor_clean_log(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "doctor")
    assert result.exit_code == 0, result.output
    assert "no anomalies" in result.output


def test_doctor_flags_unknown_location(data_dir: Path) -> None:
    _run(data_dir, "init")
    _append_raw_change(data_dir, "alice", "milk", "1", "l", to_location="hob-right-below-bottom")
    result = _run(data_dir, "doctor")
    assert result.exit_code == 1
    assert "unknown_location" in result.output


def test_doctor_suggests_a_ready_to_paste_correction(data_dir: Path) -> None:
    _run(data_dir, "init")
    _append_raw_change(data_dir, "alice", "milk", "1", "l", to_location="hob-right-below-bottom")
    result = _run(data_dir, "doctor")
    assert "sumac correct raw-1 --reason" in result.output


def test_doctor_suggests_only_one_correction_for_a_record_with_two_anomalies(
    data_dir: Path,
) -> None:
    """A duplicated line trips both seq_duplicate and duplicate_record on the
    same record id — doctor must offer `sumac correct` for it once, not
    once per anomaly (a second `correct` of the same target would just fail
    on supersede_already_applied)."""
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    log_path = paths.log_path(data_dir, "alice")
    line = log_path.read_text()
    log_path.write_text(line + line)

    result = _run(data_dir, "doctor")
    assert result.output.count("sumac correct") == 1


def test_correct_cancels_record_and_removes_it_from_status(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "pantry")

    key = _real_key(data_dir)
    record_id = ledger.load_records(data_dir, key)[0].id

    result = _run(data_dir, "correct", record_id, "--reason", "typo, wrong product")
    assert result.exit_code == 0, result.output

    result = _run(data_dir, "status")
    assert "milk" not in result.output

    result = _run(data_dir, "log")
    assert "correction" in result.output
    assert "supersedes" in result.output


def test_correct_unknown_record_id_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    result = _run(data_dir, "correct", "nope", "--reason", "typo")
    assert result.exit_code != 0


def test_correct_already_corrected_record_fails(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "2", "l", "--to", "pantry")

    key = _real_key(data_dir)
    record_id = ledger.load_records(data_dir, key)[0].id
    _run(data_dir, "correct", record_id, "--reason", "typo")

    result = _run(data_dir, "correct", record_id, "--reason", "again")
    assert result.exit_code != 0


def test_log_shows_recorded_events(data_dir: Path) -> None:
    """A "purchase" kind is now stored (and shown) as an Acquired v2 event —
    "purchase" survives only as the CLI-facing ChangeKind vocabulary."""
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")
    result = _run(data_dir, "log")
    assert result.exit_code == 0
    assert "acquired" in result.output


def test_add_array_creates_numbered_sublocations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    result = _run(data_dir, "config", "add-array", "Shelf", "--parent", "fridge", "--count", "3")
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    for i in (1, 2, 3):
        assert f"Shelf {i}" in result.output


def test_add_grid_creates_grid_sublocations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    result = _run(
        data_dir, "config", "add-grid", "Bin", "--parent", "pantry", "--rows", "2", "--cols", "2"
    )
    assert result.exit_code == 0, result.output
    result = _run(data_dir, "config", "show")
    assert result.exit_code == 0
    for cell in ("Bin R1C1", "Bin R1C2", "Bin R2C1", "Bin R2C2"):
        assert cell in result.output


def test_status_rolls_up_sublocations(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Door", "--id", "fridge-door", "--parent", "fridge")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")
    _run(data_dir, "add", "purchase", "eggs", "6", "ct", "--to", "fridge-door")

    result = _run(data_dir, "status", "fridge")
    assert result.exit_code == 0
    assert "milk" in result.output
    assert "eggs" in result.output


def test_status_on_leaf_excludes_siblings(data_dir: Path) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Fridge", "--id", "fridge")
    _run(data_dir, "config", "add-location", "Door", "--id", "fridge-door", "--parent", "fridge")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "fridge")
    _run(data_dir, "add", "purchase", "eggs", "6", "ct", "--to", "fridge-door")

    result = _run(data_dir, "status", "fridge-door")
    assert result.exit_code == 0
    assert "eggs" in result.output
    assert "milk" not in result.output


def test_another_user_cannot_write_into_alices_log(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    _run(data_dir, "config", "add-location", "Pantry", "--id", "pantry")
    _run(data_dir, "add", "purchase", "milk", "1", "l", "--to", "pantry")

    monkeypatch.setattr("getpass.getuser", lambda: "bob")
    result = _run(data_dir, "add", "purchase", "eggs", "6", "ct", "--to", "pantry")
    assert result.exit_code == 0, result.output

    assert paths.log_path(data_dir, "alice").exists()
    assert paths.log_path(data_dir, "bob").exists()
    alice_lines = paths.log_path(data_dir, "alice").read_text().splitlines()
    assert len(alice_lines) == 1


# --- ask (docs/journal/2026-09-01-ask-agent-design.md §14) -----------------
#
# `sumac ask`'s own interactive loop (accept/reject/feedback, --dry-run) is
# CLI plumbing around `llm.AgentRunner`'s propose/revise/commit interface —
# `llm.AgentRunner`'s own logic (tool dispatch, self-review, re-deciding on
# commit) is exercised against a real vault in tests/test_llm.py instead.
# These tests stand in a fake `AgentRunner` so the loop can be driven without
# a real model or GGUF download.


def _consumption_plan(amount: str = "1") -> llm.AgentPlan:
    return llm.AgentPlan(
        reply_text=f"consumed {amount} jar of jam",
        writes=(
            llm.ProposedWrite(
                kind=ChangeKind.CONSUMPTION,
                product_id="jam",
                amount=Decimal(amount),
                unit="jar",
                from_location="pantry",
                to_location=None,
            ),
        ),
    )


class _FakeAgentRunner:
    """Scripted stand-in for `llm.AgentRunner`, driven by a per-test queue of
    `AgentPlan`s. `commits` records each accepted plan so a test can assert
    whether `commit` was reached without depending on real domain state.
    `plans` is set per-test by `_patch_agent_runner` as a class attribute on
    a dynamically built subclass — declared here so its type is known."""

    # All three are ClassVars, not per-instance: `cli.py`'s loop mode
    # constructs a fresh `AgentRunner` per request, and tests have no
    # handle to those instances — only to `fake_cls`, the dynamically
    # built subclass `_patch_agent_runner` returns. Each test gets fresh
    # empty lists there, so this sharing doesn't leak across tests.
    plans: ClassVar[list[llm.AgentPlan | Exception]] = []
    commits: ClassVar[list[llm.AgentPlan]] = []
    prompts: ClassVar[list[str]] = []

    def __init__(self, data_dir: Path, key: bytes, *, debug: bool = False) -> None:
        self.data_dir = data_dir
        self.key = key
        self.debug = debug

    def _next(self) -> llm.AgentPlan:
        item = self.plans.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def propose(self, prompt: str) -> llm.AgentPlan:
        self.prompts.append(prompt)
        return self._next()

    def revise(self, feedback: str) -> llm.AgentPlan:
        return self._next()

    def commit(self, plan: llm.AgentPlan) -> list[str]:
        self.commits.append(plan)
        return [
            f"Recorded {w.kind.value} of {w.amount} {w.unit} {w.product_id}" for w in plan.writes
        ]


def _patch_agent_runner(
    monkeypatch: pytest.MonkeyPatch, plans: list[llm.AgentPlan | Exception]
) -> type[_FakeAgentRunner]:
    """Builds a fresh `_FakeAgentRunner` subclass carrying its own `plans`
    queue, and fresh `commits`/`prompts` accumulators (all class
    attributes, since `cli.py` constructs the instance itself — there's no
    seam to pass a queue through `AgentRunner(data_dir, key)`, and loop
    mode may construct more than one instance per test), and monkeypatches
    `sumac.llm.AgentRunner` to it."""
    fake_cls = type(
        "_ScriptedAgentRunner", (_FakeAgentRunner,), {"plans": plans, "commits": [], "prompts": []}
    )
    monkeypatch.setattr(llm, "AgentRunner", fake_cls)
    return fake_cls


def test_ask_read_only_reply_prints_text_and_does_not_prompt(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    plan = llm.AgentPlan(reply_text="the jam is in the pantry", writes=())
    _patch_agent_runner(monkeypatch, [plan])

    result = _run(data_dir, "ask", "where is the jam?")

    assert result.exit_code == 0, result.output
    assert "the jam is in the pantry" in result.output


def test_ask_dry_run_on_read_only_request_still_shows_dry_run_indicator(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only request has no writes to withhold either way, so
    `--dry-run` and a normal run produce identical output unless something
    says otherwise — this line is that something."""
    _run(data_dir, "init")
    plan = llm.AgentPlan(reply_text="the jam is in the pantry", writes=())
    _patch_agent_runner(monkeypatch, [plan])

    result = _run(data_dir, "ask", "where is the jam?", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "the jam is in the pantry" in result.output
    assert "--dry-run" in result.output


def test_ask_shows_tool_call_trace_before_reply(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    plan = llm.AgentPlan(
        reply_text="the jam is in the pantry",
        writes=(),
        trace=(
            llm.ToolCallRecord(
                name="sumac_find_inventory",
                arguments={"query": "jam"},
                result='{"matches": [{"product_id": "jam", "location_path": "Pantry"}]}',
            ),
        ),
    )
    _patch_agent_runner(monkeypatch, [plan])

    result = _run(data_dir, "ask", "where is the jam?")

    assert result.exit_code == 0, result.output
    assert "sumac_find_inventory" in result.output
    assert "Pantry" in result.output


def test_ask_dry_run_shows_plan_and_never_commits(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--dry-run` withholds the write, not the accept/reject/feedback
    interface — the decision prompt still appears (labeled to make clear
    accepting won't write anything), and "accept" prints a preview instead
    of calling `agent.commit`."""
    _run(data_dir, "init")
    fake_cls = _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", "consume 1 jar of jam", "--dry-run", input="a\n")

    assert result.exit_code == 0, result.output
    assert "consumption" in result.output
    assert "jam" in result.output
    assert "would record" in result.output
    assert "Recorded" not in result.output
    assert fake_cls.plans == []  # propose() was called, revise()/commit() were not
    assert fake_cls.commits == []
    # A prior run showed no visible difference between --dry-run and a
    # normal invocation for a read-only request, making the flag look like
    # it did nothing; an explicit line now confirms it was honored.
    assert "--dry-run" in result.output


def test_ask_accept_commits_and_prints_summary(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", "consume 1 jar of jam", input="a\n")

    assert result.exit_code == 0, result.output
    assert "Recorded consumption of 1 jar jam" in result.output


def test_ask_reject_does_not_commit(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(data_dir, "init")
    fake_cls = _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", "consume 1 jar of jam", input="r\n")

    assert result.exit_code == 0, result.output
    assert "Recorded" not in result.output
    assert fake_cls.plans == []


def test_ask_feedback_revises_then_accept_commits_the_revised_plan(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    _patch_agent_runner(monkeypatch, [_consumption_plan(amount="1"), _consumption_plan(amount="2")])

    result = _run(data_dir, "ask", "consume some jam", input="actually make it 2 jars\na\n")

    assert result.exit_code == 0, result.output
    assert "Recorded consumption of 2 jar jam" in result.output


# --- ask --loop / queue ----------------------------------------------------


def test_ask_loop_with_no_prompt_enters_loop_mode(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", input="consume 1 jar of jam\na\nquit\n")

    assert result.exit_code == 0, result.output
    assert "Recorded consumption of 1 jar jam" in result.output


def test_ask_loop_flag_runs_a_given_prompt_first_then_keeps_prompting(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    fake_cls = _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", "consume 1 jar of jam", "--loop", input="a\nquit\n")

    assert result.exit_code == 0, result.output
    assert "Recorded consumption of 1 jar jam" in result.output
    assert fake_cls.prompts == ["consume 1 jar of jam"]


def test_ask_loop_failure_is_queued_not_fatal(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The specific failure this exists for: a request that raises (e.g.
    mistral.rs's max-seq-len KV-cache exhaustion) must not end the loop —
    it goes on the local queue and the session keeps going."""
    _run(data_dir, "init")
    _patch_agent_runner(monkeypatch, [RuntimeError("boom")])

    result = _run(data_dir, "ask", input="add something\nquit\n")

    assert result.exit_code == 0, result.output
    assert "Agent error: boom" in result.output
    assert "queued for later" in result.output
    items = queue.load(data_dir)
    assert len(items) == 1
    assert items[0].prompt == "add something"
    assert items[0].reason == "error: boom"
    assert items[0].attempts == 0


def test_ask_loop_defer_queues_without_committing(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    fake_cls = _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", input="consume 1 jar of jam\nd\nquit\n")

    assert result.exit_code == 0, result.output
    assert "queued for later" in result.output
    assert fake_cls.commits == []
    items = queue.load(data_dir)
    assert len(items) == 1
    assert items[0].prompt == "consume 1 jar of jam"
    assert items[0].reason == "deferred"


def test_ask_loop_dry_run_still_prompts_but_never_commits(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As with `_ask_one`, `--dry-run` in loop mode withholds the write,
    not the decision itself — accepting still prints a preview rather than
    silently doing nothing, which previously made the flag look broken."""
    _run(data_dir, "init")
    fake_cls = _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", "--dry-run", input="consume 1 jar of jam\na\nquit\n")

    assert result.exit_code == 0, result.output
    assert "would record" in result.output
    assert "Recorded" not in result.output
    assert fake_cls.commits == []


def test_ask_loop_queue_command_lists_pending_requests(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    queue.enqueue(data_dir, "add 6 cartons of milk", "deferred")
    queue.enqueue(data_dir, "consume 1 jar of jam", "error: boom", attempts=1)
    _patch_agent_runner(monkeypatch, [])

    result = _run(data_dir, "ask", input="queue\nquit\n")

    assert result.exit_code == 0, result.output
    assert "[0] add 6 cartons of milk" in result.output
    assert "[1] consume 1 jar of jam" in result.output
    assert "1 attempt(s)" in result.output


def test_ask_loop_retry_dequeues_and_reuses_the_queued_prompt(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    queue.enqueue(data_dir, "consume 1 jar of jam", "error: boom")
    fake_cls = _patch_agent_runner(monkeypatch, [_consumption_plan()])

    result = _run(data_dir, "ask", input="retry 0\na\nquit\n")

    assert result.exit_code == 0, result.output
    assert "Recorded consumption of 1 jar jam" in result.output
    assert fake_cls.prompts == ["consume 1 jar of jam"]
    assert queue.load(data_dir) == []


def test_ask_loop_retry_failing_again_requeues_with_incremented_attempts(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    queue.enqueue(data_dir, "consume 1 jar of jam", "error: boom", attempts=1)
    _patch_agent_runner(monkeypatch, [RuntimeError("boom again")])

    result = _run(data_dir, "ask", input="retry 0\nquit\n")

    assert result.exit_code == 0, result.output
    items = queue.load(data_dir)
    assert len(items) == 1
    assert items[0].attempts == 2
    assert items[0].reason == "error: boom again"


def test_ask_loop_retry_with_invalid_index_reports_error_and_continues(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(data_dir, "init")
    _patch_agent_runner(monkeypatch, [])

    result = _run(data_dir, "ask", input="retry 5\nquit\n")

    assert result.exit_code == 0, result.output
    assert "No queued request at index" in result.output
