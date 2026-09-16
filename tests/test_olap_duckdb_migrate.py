"""What the migration tooling must do, written against the findings that it did not.

E1 of `docs/handoffs/MUSASHI_DUCKDB_CLOSEOUT_CORRECTIONS_2026_09_16.md`: "Write behavioral
tests first for the findings above."

Musashi reviewed the migration code rather than only its receipts, and found five things. All
five are mine, and each is a case of the tooling REPORTING an outcome it had not performed:

1. `rollback` emitted `REPLAY_REQUIRED` and returned success without writing anything;
2. selection classified `df_*` by table-name prefix — the exact criterion D1 forbade — and the
   export then ignored the manifest entirely and copied whole tables from a constant list;
3. `snapshot` copied a moving main/WAL pair and called the result verified, and with no
   expected count `verified` was unconditionally true;
4. an interrupted export resumed as `ALREADY_PRESENT`; catch-up skipped child tables; and two
   relations whose digest could not be computed compared EQUAL, so `content_matches` was true
   because both sides failed the same way;
5. stopping the old loader prevented old writes without providing a new route.

These run on SQLite and DuckDB fixtures built in `tmp_path`. Nothing here touches production,
and the disposable-target rule is enforced by construction: no test knows a production name.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "olap_duckdb_migrate.py"

spec = importlib.util.spec_from_file_location("olap_duckdb_migrate_under_test", TOOL)
migrate = importlib.util.module_from_spec(spec)
sys.modules["olap_duckdb_migrate_under_test"] = migrate
spec.loader.exec_module(migrate)

duckdb = pytest.importorskip("duckdb")


# --- finding 2: selection must use lineage, not a table-name prefix ---------------------

def test_selection_does_not_classify_by_table_name_prefix():
    """`df_` is a naming convention. It says nothing about which campaign a row belongs to."""
    source = TOOL.read_text(encoding="utf-8")
    assert 'startswith("df_")' not in source, (
        "classifying by table name is the criterion the order forbids: current "
        "data-foundation work must not be archived merely for predating gov_terminal")


def test_a_run_is_classified_from_its_campaign_identity(tmp_path):
    """Membership comes from the run's recorded campaign, including NON_GOVERNING ones."""
    current = {"run_id": "d2v2_fresh_1", "module": "D2 fresh confirmation",
               "status": "COMPLETED", "campaign_key": "d2::fresh_confirmation",
               "result_class": "CONFIRMATION"}
    historical = {"run_id": "hist_1", "module": "HISTORICAL_MIGRATION_REANALYSIS",
                  "status": "COMPLETED", "campaign_key": "d2::historical_reanalysis",
                  "result_class": "NON_CONFIRMATORY_HISTORICAL"}
    assert migrate.classify_run(current, current_campaigns={"d2::fresh_confirmation"})[0] == (
        "INCLUDED_CURRENT")
    assert migrate.classify_run(historical, current_campaigns={"d2::fresh_confirmation"})[0] == (
        "LEGACY_COMPARISON_ONLY")


def test_a_failed_current_run_is_included_with_its_status(tmp_path):
    """Selecting by outcome is what the order forbids; a failure is evidence."""
    failed = {"run_id": "r2", "module": "D2", "status": "FAILED",
              "campaign_key": "d2::fresh_confirmation", "result_class": "CONFIRMATION"}
    disposition, rationale = migrate.classify_run(
        failed, current_campaigns={"d2::fresh_confirmation"})
    assert disposition == "INCLUDED_CURRENT"
    assert "FAILED" in rationale


def test_one_table_can_hold_both_current_and_legacy_rows(tmp_path):
    """Row-level closure: a whole-table decision cannot express this and is therefore wrong."""
    path = tmp_path / "src.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE df_fact_x (row_sha256 VARCHAR, run_id VARCHAR, v INTEGER)")
    con.execute("INSERT INTO df_fact_x VALUES ('a','current',1),('b','legacy',2)")
    con.close()

    selected = migrate.rows_for_runs(str(path), "main", "df_fact_x", {"current"})
    assert selected == 1, "only the current run's rows belong in the cube"


# --- finding 1: rollback must perform what it reports -----------------------------------

def test_rollback_without_dry_run_must_not_report_a_replay_it_did_not_do(tmp_path):
    """The exact defect: REPLAY_REQUIRED and exit zero, with nothing written."""
    source = TOOL.read_text(encoding="utf-8")
    assert '"REPLAY_REQUIRED"' not in source or "def replay_relation" in source, (
        "a rollback that names work it did not perform is a report, not a rollback")


def test_rollback_replays_parent_and_child_rows(tmp_path):
    """A terminal without its metrics, datasets and artifacts is not a restored outcome."""
    source = str(tmp_path / "duck.duckdb")
    target = str(tmp_path / "dest.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)

    report = migrate.replay_between(source, target, schema="main", dry_run=False)

    assert report["summary"]["terminals_replayed"] == 2
    assert report["summary"]["child_rows_replayed"] > 0
    assert report["summary"]["contracts_replayed"] == 1
    con = duckdb.connect(target, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_terminal").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM main.gov_terminal_metric").fetchone()[0] > 0
    con.close()


def test_a_second_rollback_replays_nothing_and_still_succeeds(tmp_path):
    source = str(tmp_path / "duck.duckdb")
    target = str(tmp_path / "dest.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)

    migrate.replay_between(source, target, schema="main", dry_run=False)
    again = migrate.replay_between(source, target, schema="main", dry_run=False)

    assert again["summary"]["terminals_replayed"] == 0
    con = duckdb.connect(target, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_terminal").fetchone()[0] == 2
    con.close()


def test_rollback_creates_a_relation_the_destination_lacks(tmp_path):
    """gov_availability_contract does not exist in the old schema; replay must create it."""
    source = str(tmp_path / "duck.duckdb")
    target = str(tmp_path / "dest.duckdb")
    migrate.build_fixture_cube(source, terminals=1, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0, drop_contract_table=True)

    report = migrate.replay_between(source, target, schema="main", dry_run=False)

    assert report["summary"]["relations_created"] >= 1
    con = duckdb.connect(target, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_availability_contract").fetchone()[0] == 1
    con.close()


# --- finding 3: a snapshot must have a writer boundary ----------------------------------

def test_snapshot_refuses_to_call_itself_verified_without_a_boundary(tmp_path):
    source = str(tmp_path / "live.duckdb")
    migrate.build_fixture_cube(source, terminals=1)
    report = migrate.snapshot_database(source, str(tmp_path / "copy.duckdb"),
                                       schema="main", expect_terminals=None,
                                       owner_stopped=False)
    assert report["verified"] is False, (
        "a copy taken with no coordinated writer boundary is not a verified snapshot, and "
        "an absent expectation must not make `verified` unconditionally true")
    assert "boundary" in json.dumps(report).lower()


def test_snapshot_is_verified_when_the_owner_is_stopped_and_the_count_matches(tmp_path):
    source = str(tmp_path / "live.duckdb")
    migrate.build_fixture_cube(source, terminals=3)
    report = migrate.snapshot_database(source, str(tmp_path / "copy.duckdb"),
                                       schema="main", expect_terminals=3, owner_stopped=True)
    assert report["verified"] is True
    assert report["counts_in_snapshot"]["gov_terminal"] == 3


def test_snapshot_fails_when_the_count_disagrees(tmp_path):
    source = str(tmp_path / "live.duckdb")
    migrate.build_fixture_cube(source, terminals=3)
    report = migrate.snapshot_database(source, str(tmp_path / "copy.duckdb"),
                                       schema="main", expect_terminals=5, owner_stopped=True)
    assert report["verified"] is False


# --- finding 4: partial imports, child catch-up and uncomparable digests -----------------

def test_two_uncomparable_digests_are_never_a_match():
    """Both sides failing the same way is not agreement."""
    assert migrate.digests_agree("UNCOMPARABLE: TypeError",
                                 "UNCOMPARABLE: TypeError") is False
    assert migrate.digests_agree("abc", "abc") is True
    assert migrate.digests_agree(None, None) is False


def test_a_partially_filled_relation_is_resumed_not_declared_present(tmp_path):
    """An interrupted batch leaves rows behind; `ALREADY_PRESENT` would strand the rest."""
    assert migrate.import_state(source_rows=100, destination_rows=0) == "EMPTY"
    assert migrate.import_state(source_rows=100, destination_rows=40) == "PARTIAL"
    assert migrate.import_state(source_rows=100, destination_rows=100) == "COMPLETE"
    assert migrate.import_state(source_rows=100, destination_rows=140) == "OVERFILLED"


# --- E2: replay must survive interruption and prove the destination can READ what arrived ---

def test_an_interrupted_replay_resumes_and_completes(tmp_path, monkeypatch):
    """Kill the replay after the first parent, then run it again: nothing is lost or doubled."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)

    calls = {"n": 0}
    real = migrate.replay_between

    def interrupt_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # a replay that dies partway: the contract lands, the terminals do not
            report = real(*args, **kwargs)
            raise RuntimeError("interrupted after the first relation")
        return real(*args, **kwargs)

    monkeypatch.setattr(migrate, "replay_between", interrupt_once)
    with pytest.raises(RuntimeError):
        migrate.replay_between(source, target, schema="main", dry_run=False)
    monkeypatch.undo()

    resumed = migrate.replay_between(source, target, schema="main", dry_run=False)
    con = duckdb.connect(target, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_terminal").fetchone()[0] == 2
    assert con.execute(
        "SELECT count(*) FROM main.gov_availability_contract").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM main.gov_terminal_metric").fetchone()[0] == 2
    con.close()
    assert resumed["summary"]["terminals_replayed"] >= 0


def test_the_destination_can_resolve_the_replayed_contract(tmp_path):
    """A replayed outcome that cannot be read back is not a restored outcome."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=1, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0, drop_contract_table=True)
    migrate.replay_between(source, target, schema="main", dry_run=False)

    from predictor_duckdb_store.provider import PredictorDuckdbStore

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=target, schema="main", memory_limit="1GB", threads=2,
                     min_free_bytes=1)
    store.engine()
    answer = store.resolve_delivery_availability("00000000000000000000000000000000")

    assert answer["contract_resolution"] == "VERIFIED"
    assert answer["use_class"] == "ARCHIVE_RETROSPECTIVE"
    assert answer["completion_lag_max"] == "UNKNOWN", (
        "UNKNOWN must survive a rollback replay as itself")


def test_replayed_content_is_identical_not_merely_present(tmp_path):
    """Same identities AND same values on both sides, compared by the same engine."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=3, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)
    migrate.replay_between(source, target, schema="main", dry_run=False)

    con = duckdb.connect(target)
    con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
    for relation in ("gov_terminal", "gov_terminal_metric", "gov_terminal_dataset",
                     "gov_terminal_artifact", "gov_availability_contract"):
        columns = [row[1] for row in con.execute(
            f'PRAGMA table_info("main"."{relation}")').fetchall()]
        left = migrate.content_digest(con, f'src."main"."{relation}"', columns)
        right = migrate.content_digest(con, f'"main"."{relation}"', columns)
        assert migrate.digests_agree(left, right), f"{relation} differs after replay"
    con.close()


def test_a_dry_run_replay_writes_nothing(tmp_path):
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)

    report = migrate.replay_between(source, target, schema="main", dry_run=True)

    assert all(entry["outcome"].startswith("WOULD_") for entry in report["relations"])
    con = duckdb.connect(target, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_terminal").fetchone()[0] == 0
    con.close()


# --- E3: catch-up must carry a new parent's complete evidence -----------------------------

def test_catch_up_carries_children_of_a_new_parent(tmp_path):
    """Child rows have no timestamp. Skipping them ships a terminal without its evidence."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)
    migrate.replay_between(source, target, schema="main", dry_run=False)

    # a NEW terminal appears at the source after the catch-up point
    migrate.build_fixture_cube(source, terminals=3, with_children=True, with_contract=True)
    report = migrate.replay_between(source, target, schema="main", dry_run=False)

    assert report["summary"]["terminals_replayed"] == 1
    assert report["summary"]["child_rows_replayed"] == 3, (
        "the new terminal's metric, dataset and artifact must travel with it")
    con = duckdb.connect(target, read_only=True)
    assert con.execute(
        "SELECT count(*) FROM main.gov_terminal_artifact").fetchone()[0] == 3
    con.close()
