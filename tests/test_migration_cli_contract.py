"""Musashi's own probes, frozen, plus the command paths they were never run through.

F1 of `docs/handoffs/MUSASHI_E1_E6_REVIEW_AND_F1_F5_2026_09_16.md`. He did not read my
receipts; he imported the tool and called it. Two things fell out immediately, and both are
mine:

1. `copy_relation` restarts at the first source row instead of skipping the prefix a previous
   attempt already wrote, so resuming a constrained table raises `Duplicate key` and resuming
   an unconstrained one silently duplicates rows;
2. `replay_between` matches parents by identity alone, so a destination that already holds the
   parent is declared in sync while its metrics, datasets and artifacts are missing.

And a third thing, which is the shape of the problem rather than a single bug: every proof I
offered was a helper call. The shipped command paths - `main()` with real arguments - were
never exercised, so a correct helper and a wrong CLI looked identical from the outside.

Everything here runs on databases created inside `tmp_path`. No production name appears.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "olap_duckdb_migrate.py"

spec = importlib.util.spec_from_file_location("olap_duckdb_migrate_cli_under_test", TOOL)
migrate = importlib.util.module_from_spec(spec)
sys.modules["olap_duckdb_migrate_cli_under_test"] = migrate
spec.loader.exec_module(migrate)

duckdb = pytest.importorskip("duckdb")


def table(path: str, *, name="source", rows=(1, 2, 3), key=True):
    con = duckdb.connect(path)
    constraint = " PRIMARY KEY" if key else ""
    con.execute(f"CREATE TABLE {name} (id INTEGER{constraint}, v VARCHAR)")
    for row in rows:
        con.execute(f"INSERT INTO {name} VALUES ({row}, 'v{row}')")
    con.close()


# --- probe 1, exactly as the reviewer ran it ---------------------------------------------

def test_resuming_a_constrained_table_skips_the_completed_prefix(tmp_path):
    """The reviewer's first probe: source [1,2,3], target [1], batch of 1."""
    path = str(tmp_path / "probe.duckdb")
    table(path, name="source", rows=(1, 2, 3))
    table(path, name="target", rows=(1,))
    con = duckdb.connect(path)

    copied = migrate.copy_relation(con, "source", "target", 3, "id", 1)

    held = [row[0] for row in con.execute("SELECT id FROM target ORDER BY id").fetchall()]
    con.close()
    assert held == [1, 2, 3], "the completed prefix must be skipped, not reinserted"
    assert copied == 2, "two rows were outstanding, so two rows were copied"


def test_resuming_an_unconstrained_table_does_not_duplicate(tmp_path):
    """Without a key constraint nothing raises — it just silently doubles rows."""
    path = str(tmp_path / "bag.duckdb")
    table(path, name="source", rows=(1, 2, 3), key=False)
    table(path, name="target", rows=(1,), key=False)
    con = duckdb.connect(path)

    migrate.copy_relation(con, "source", "target", 3, "id", 1)

    held = sorted(row[0] for row in con.execute("SELECT id FROM target").fetchall())
    con.close()
    assert held == [1, 2, 3], f"duplicated rows: {held}"


# --- probe 2, exactly as the reviewer ran it ---------------------------------------------

def test_an_existing_parent_does_not_hide_missing_children(tmp_path):
    """The reviewer's second probe: the parent is already there, the children are not."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=1, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=1)          # parent only, no children

    report = migrate.replay_between(source, target, schema="main", dry_run=False)

    con = duckdb.connect(target, read_only=True)
    metrics = con.execute("SELECT count(*) FROM main.gov_terminal_metric").fetchone()[0]
    datasets = con.execute("SELECT count(*) FROM main.gov_terminal_dataset").fetchone()[0]
    con.close()
    assert metrics == 1, "the parent was present; its metric was not, and must be restored"
    assert datasets == 1
    assert report["summary"]["child_rows_replayed"] >= 2


def test_conflicting_parent_content_is_refused_not_ignored(tmp_path):
    """Same identity, different content: that is an unresolved state, not a match."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=1, with_children=True)
    migrate.build_fixture_cube(target, terminals=1, with_children=True)
    con = duckdb.connect(target)
    con.execute("UPDATE main.gov_terminal SET status = 'FAILED'")
    con.close()

    report = migrate.replay_between(source, target, schema="main", dry_run=False)

    assert report["summary"]["conflicting_parents"] == 1
    assert report["summary"]["unresolved"] is True
    assert any(entry.get("outcome") == "CONFLICT_REFUSED"
               for entry in report["relations"]), report["relations"]


# --- the command paths themselves ---------------------------------------------------------

def test_the_export_command_refuses_an_overfilled_destination_with_a_nonzero_exit(tmp_path):
    """`OVERFILLED_REFUSED` used to be reported and then exit 0, which reads as success."""
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=1)
    migrate.build_fixture_cube(target, terminals=3)
    out = tmp_path / "report.json"

    code = migrate.main(["copy-cube", "--source", source, "--destination", target,
                         "--schema", "main", "--out", str(out)])

    assert code != 0, "refused work must not exit zero"
    report = json.loads(out.read_text())
    assert any(entry["outcome"] == "OVERFILLED_REFUSED" for entry in report["relations"])


def test_the_rollback_command_runs_through_main_and_reports_its_summary(tmp_path):
    source = str(tmp_path / "src.duckdb")
    target = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(target, terminals=0)
    out = tmp_path / "rollback.json"

    code = migrate.main(["rollback", "--duckdb", source, "--target", target,
                         "--target-engine", "duckdb", "--schema", "main", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text())
    assert report["summary"]["terminals_replayed"] == 2
    assert report["summary"]["child_rows_replayed"] == 6


def test_a_row_count_alone_never_declares_a_relation_complete(tmp_path):
    """Equal counts, different content: the destination is NOT complete."""
    path = str(tmp_path / "counts.duckdb")
    table(path, name="source", rows=(1, 2, 3))
    con = duckdb.connect(path)
    con.execute("CREATE TABLE target (id INTEGER PRIMARY KEY, v VARCHAR)")
    con.execute("INSERT INTO target VALUES (1,'v1'),(2,'DIFFERENT'),(3,'v3')")
    con.close()

    con = duckdb.connect(path)
    state = migrate.relation_state(con, "source", "target", ["id", "v"])
    con.close()
    assert state == "CONTENT_DIFFERS", (
        "row-count equality is not completion; the contents disagree")


# --- F2: catch-up must carry children that have no timestamp of their own ------------------

def test_the_catchup_command_carries_timestamp_free_children(tmp_path):
    """A terminal caught up without its metrics is a half-outcome, not a caught-up one."""
    source = str(tmp_path / "src.duckdb")
    destination = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=2, with_children=True, with_contract=True)
    migrate.build_fixture_cube(destination, terminals=0)
    out = tmp_path / "catchup.json"

    code = migrate.main(["catchup", "--source", source, "--destination", destination,
                         "--source-engine", "duckdb", "--schema", "main", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text())
    assert report["summary"]["terminals_replayed"] == 2
    assert report["summary"]["child_rows_replayed"] == 6, (
        "gov_terminal_metric, _dataset and _artifact have no received_at and were skipped")
    con = duckdb.connect(destination, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_terminal_artifact").fetchone()[0] == 2
    con.close()


def test_a_partially_present_outcome_is_completed_by_catchup(tmp_path):
    """The parent is there and its children are not: catch-up must finish it."""
    source = str(tmp_path / "src.duckdb")
    destination = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=1, with_children=True, with_contract=True)
    migrate.build_fixture_cube(destination, terminals=1)
    out = tmp_path / "catchup.json"

    migrate.main(["catchup", "--source", source, "--destination", destination,
                  "--source-engine", "duckdb", "--schema", "main", "--out", str(out)])

    con = duckdb.connect(destination, read_only=True)
    assert con.execute("SELECT count(*) FROM main.gov_terminal_metric").fetchone()[0] == 1
    con.close()


def test_the_catchup_command_exits_nonzero_on_an_unresolved_conflict(tmp_path):
    source = str(tmp_path / "src.duckdb")
    destination = str(tmp_path / "dst.duckdb")
    migrate.build_fixture_cube(source, terminals=1, with_children=True)
    migrate.build_fixture_cube(destination, terminals=1, with_children=True)
    con = duckdb.connect(destination)
    con.execute("UPDATE main.gov_terminal SET actor = 'somebody-else'")
    con.close()
    out = tmp_path / "catchup.json"

    code = migrate.main(["catchup", "--source", source, "--destination", destination,
                         "--source-engine", "duckdb", "--schema", "main", "--out", str(out)])

    assert code == 2, "an unresolved conflict is not a completed catch-up"


# --- G2: a resume must prove the prefix it is resuming from --------------------------------

def test_a_hole_in_the_destination_is_not_a_completed_prefix(tmp_path):
    """The reviewer's third probe: source [1,2,3], destination [1,3]. Two is missing.

    Resuming from MAX(key) jumps past the hole and reports nothing to do. The high-water mark
    says where the destination STOPS, not that everything below it is there.
    """
    path = str(tmp_path / "hole.duckdb")
    table(path, name="source", rows=(1, 2, 3))
    table(path, name="target", rows=(1, 3))
    con = duckdb.connect(path)

    copied = migrate.copy_relation(con, "source", "target", 3, "id", 10)

    held = sorted(row[0] for row in con.execute("SELECT id FROM target").fetchall())
    con.close()
    assert held == [1, 2, 3], f"the hole was not repaired: {held}"
    assert copied == 1


def test_a_modified_existing_row_is_refused_rather_than_resumed_past(tmp_path):
    """Same identity, different content: resuming would leave the wrong row in place."""
    path = str(tmp_path / "modified.duckdb")
    table(path, name="source", rows=(1, 2, 3))
    con = duckdb.connect(path)
    con.execute("CREATE TABLE target (id INTEGER PRIMARY KEY, v VARCHAR)")
    con.execute("INSERT INTO target VALUES (1,'WRONG')")
    con.close()

    con = duckdb.connect(path)
    with pytest.raises(RuntimeError) as refusal:
        migrate.copy_relation(con, "source", "target", 3, "id", 10)
    con.close()
    assert "content" in str(refusal.value).lower()


def test_an_extra_row_in_the_destination_is_refused(tmp_path):
    path = str(tmp_path / "extra.duckdb")
    table(path, name="source", rows=(1, 2))
    table(path, name="target", rows=(1, 2, 9))
    con = duckdb.connect(path)
    with pytest.raises(RuntimeError) as refusal:
        migrate.copy_relation(con, "source", "target", 2, "id", 10)
    con.close()
    assert "not in the source" in str(refusal.value).lower()


def test_the_snapshot_path_without_a_measured_boundary_is_named_unverified(tmp_path):
    """A caller's flag is an assertion. The name of the artefact must say what it is."""
    source = str(tmp_path / "live.duckdb")
    migrate.build_fixture_cube(source, terminals=1)
    report = migrate.snapshot_database(source, str(tmp_path / "copy.duckdb"),
                                       schema="main", expect_terminals=1,
                                       owner_stopped=True)
    assert report["kind"] in ("VERIFIED_SNAPSHOT", "UNVERIFIED_COPY")
    assert report["boundary"]["measured"] is True, (
        "the boundary must be MEASURED by the snapshot itself, not asserted by its caller")
    assert report["boundary"]["method"]


def test_a_snapshot_taken_while_another_process_holds_the_file_is_an_unverified_copy(tmp_path):
    """Measured, not asserted — and measured against the condition that actually occurs.

    My first version of this rule opened a second connection in the SAME process and expected a
    conflict. DuckDB's lock is per PROCESS, so it saw none: the premise was wrong, not the
    code. The real case is the one that matters anyway — the warehouse service is a separate
    process — so the holder here is a subprocess, and the limitation is stated in the tool.
    """
    import subprocess
    import time

    source = str(tmp_path / "held.duckdb")
    migrate.build_fixture_cube(source, terminals=1)
    holder_script = tmp_path / "holder.py"
    holder_script.write_text(
        "import duckdb, sys, time\n"
        "con = duckdb.connect(sys.argv[1])\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n", encoding="utf-8")
    holder = subprocess.Popen([sys.executable, str(holder_script), source],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        time.sleep(0.3)
        report = migrate.snapshot_database(source, str(tmp_path / "copy2.duckdb"),
                                           schema="main", expect_terminals=1,
                                           owner_stopped=True)
    finally:
        holder.kill()
        holder.wait(timeout=30)

    assert report["boundary"]["measured"] is True
    assert report["boundary"]["writer_present"] is True, (
        "another PROCESS holds the file, so there is no boundary to copy at")
    assert report["kind"] == "UNVERIFIED_COPY"
    assert report["verified"] is False
    assert report["caller_claimed_owner_stopped"] is True, (
        "the caller's claim is recorded beside the measurement that contradicts it")
