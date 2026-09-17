"""R6 selects a coverage version. It must never delete, dedupe, or lose the other one.

Follow-up to `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`: R6 was
the one measured block on D3 and is applied here, so the rules that make the application safe
have to exist too.

`df_fact_coverage` holds one run id under TWO code digests. A coverage figure read without a
selection therefore counts a doubled population, and "fixing" that by deleting a digest would
destroy evidence. R6's answer is a selection table plus two views: current shows exactly one
declared matrix, history keeps every version with its provenance.

Everything here builds its own throwaway cube under `tmp_path`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "df_r6_apply.py"
spec = importlib.util.spec_from_file_location("df_r6_apply_under_test", TOOL)
r6 = importlib.util.module_from_spec(spec)
sys.modules["df_r6_apply_under_test"] = r6
spec.loader.exec_module(r6)

duckdb = pytest.importorskip("duckdb")

RUN_V1 = "c140_17e79fa33a11f298c70180ec"
RUN_V2 = "c170_17e79fa33a11f298c70180ec"
CODE_A = "31e3376d" + "0" * 56
CODE_B = "b4c9157c206dc914953c0ca52d6ae60fa79780d38aaef318ba2eb3cf1d53cb86"


def build(path: Path, *, v1_rows: int = 6, v2_rows: int = 6) -> Path:
    """A cube shaped like the real one: v1 under two code digests, v2 under one."""
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE df_fact_coverage (run_id TEXT, code_sha256 TEXT,"
                " dataset_id TEXT, variable_id TEXT, metric TEXT, operator TEXT, state TEXT,"
                " loaded_at TEXT)")
    con.execute("CREATE TABLE df_fact_coverage_v2 (run_id TEXT, code_sha256 TEXT,"
                " dataset_id TEXT, variable_id TEXT, metric TEXT, operator TEXT, state TEXT,"
                " loaded_at TEXT, partition TEXT, policy TEXT, applicability TEXT)")
    for index in range(v1_rows):
        code = CODE_A if index % 2 else CODE_B
        con.execute("INSERT INTO df_fact_coverage VALUES (?,?,?,?,?,?,?,?)",
                    [RUN_V1, code, f"d{index % 3}", f"v{index}", "m", "o", "s", "t"])
    for index in range(v2_rows):
        con.execute("INSERT INTO df_fact_coverage_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [RUN_V2, CODE_B, f"d{index % 3}", f"v{index}", "m", "o", "s", "t",
                     "train", "p", "a"])
    con.close()
    return path


def counts(path: Path, *names) -> dict:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return {name: con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                for name in names}
    finally:
        con.close()


def apply(path: Path, tmp_path: Path, *extra):
    out = tmp_path / "receipt.json"
    code = r6.main(["--cube", str(path), "--out", str(out), *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))


# --- the statement splitter, which is where the first attempt broke ------------------------

def test_a_semicolon_inside_a_quoted_string_does_not_split_a_statement():
    """R6's own reason reads "(9 states + applicability); supersedes v1 c140"."""
    text = "INSERT INTO t VALUES ('a; b'); SELECT 1;"
    assert r6.split_statements(text) == ["INSERT INTO t VALUES ('a; b')", "SELECT 1"]


def test_a_comment_marker_inside_a_quoted_string_is_not_a_comment():
    assert r6.split_statements("SELECT 'a -- b';") == ["SELECT 'a -- b'"]


def test_comments_are_removed():
    assert r6.split_statements("-- a comment\nSELECT 1; -- trailing\n") == ["SELECT 1"]


def test_the_shipped_sql_parses_into_the_statements_it_declares():
    statements = r6.statements()
    assert sum(1 for s in statements if s.upper().startswith("CREATE TABLE")) == 1
    assert sum(1 for s in statements if s.upper().startswith("INSERT")) == 1
    assert sum(1 for s in statements if "CREATE OR REPLACE VIEW" in s.upper()) == 3


# --- what the application must produce -----------------------------------------------------

def test_the_current_view_is_exactly_the_selected_matrix(tmp_path):
    cube = build(tmp_path / "c.duckdb")
    code, receipt = apply(cube, tmp_path)
    assert code == 0 and receipt["outcome"] == "APPLIED"
    assert receipt["applied"]["selection"]["run_id"] == RUN_V2
    held = counts(cube, "df_coverage_current", "df_fact_coverage_v2")
    assert held["df_coverage_current"] == held["df_fact_coverage_v2"]


def test_the_current_view_carries_one_run_and_one_code_digest(tmp_path):
    """The whole point: a coverage figure must be able to say what it counted."""
    cube = build(tmp_path / "c.duckdb")
    apply(cube, tmp_path)
    con = duckdb.connect(str(cube), read_only=True)
    try:
        runs, codes = con.execute(
            "SELECT count(DISTINCT run_id), count(DISTINCT code_sha256)"
            " FROM df_coverage_current").fetchone()
    finally:
        con.close()
    assert (runs, codes) == (1, 1)


def test_history_keeps_every_version(tmp_path):
    cube = build(tmp_path / "c.duckdb", v1_rows=6, v2_rows=6)
    apply(cube, tmp_path)
    held = counts(cube, "df_coverage_history", "df_fact_coverage", "df_fact_coverage_v2")
    assert held["df_coverage_history"] == (held["df_fact_coverage"]
                                           + held["df_fact_coverage_v2"])


def test_nothing_is_deleted_or_deduplicated(tmp_path):
    cube = build(tmp_path / "c.duckdb")
    before = counts(cube, "df_fact_coverage", "df_fact_coverage_v2")
    _code, receipt = apply(cube, tmp_path)
    after = counts(cube, "df_fact_coverage", "df_fact_coverage_v2")
    assert before == after
    assert receipt["applied"]["fact_rows_before"] == receipt["applied"]["fact_rows_after"]


def test_the_denominator_is_not_doubled(tmp_path):
    """v1's two digests must not reach the denominator the current view is measured against."""
    cube = build(tmp_path / "c.duckdb", v1_rows=6, v2_rows=6)
    apply(cube, tmp_path)
    con = duckdb.connect(str(cube), read_only=True)
    try:
        cells = con.execute("SELECT sum(cells) FROM df_coverage_current_denominator"
                            ).fetchone()[0]
        v2 = con.execute("SELECT count(*) FROM df_fact_coverage_v2").fetchone()[0]
    finally:
        con.close()
    assert cells == v2


def test_the_selection_carries_an_explicit_reason(tmp_path):
    cube = build(tmp_path / "c.duckdb")
    _code, receipt = apply(cube, tmp_path)
    reason = receipt["applied"]["selection"]["reason"]
    assert "supersedes" in reason and RUN_V1.split("_")[0] in reason


def test_the_receipt_names_every_dialect_translation(tmp_path):
    """A silent translation is a change nobody reviewed."""
    cube = build(tmp_path / "c.duckdb")
    _code, receipt = apply(cube, tmp_path)
    spellings = {entry["postgres"] for entry in receipt["translations"]}
    assert "TIMESTAMPTZ" in spellings and "::TEXT" in spellings


def test_no_write_ahead_log_is_left_beside_the_cube(tmp_path):
    cube = build(tmp_path / "c.duckdb")
    _code, receipt = apply(cube, tmp_path)
    assert receipt["applied"]["write_ahead_log_bytes_after_checkpoint"] == 0
    assert not Path(str(cube) + ".wal").exists()


# --- what must stop it ---------------------------------------------------------------------

def test_a_rehearsal_that_does_not_come_out_right_never_touches_the_cube(tmp_path,
                                                                        monkeypatch):
    cube = build(tmp_path / "c.duckdb")
    monkeypatch.setattr(r6, "verify", lambda outcome: ["a deliberate objection"])
    code, receipt = apply(cube, tmp_path)
    assert code == 1 and receipt["outcome"] == "REFUSED_ON_REHEARSAL"
    con = duckdb.connect(str(cube), read_only=True)
    try:
        with pytest.raises(Exception):
            con.execute("SELECT count(*) FROM df_coverage_current")
    finally:
        con.close()


def test_rehearse_only_leaves_the_cube_untouched(tmp_path):
    cube = build(tmp_path / "c.duckdb")
    code, receipt = apply(cube, tmp_path, "--rehearse-only")
    assert code == 0 and receipt["outcome"] == "REHEARSED_ONLY"
    assert receipt["applied"] is None
    con = duckdb.connect(str(cube), read_only=True)
    try:
        with pytest.raises(Exception):
            con.execute("SELECT count(*) FROM df_coverage_current")
    finally:
        con.close()


def test_a_deleted_fact_row_is_caught_by_the_verifier():
    outcome = {"fact_rows_before": {"df_fact_coverage": 10},
               "fact_rows_after": {"df_fact_coverage": 9},
               "selection": {"table_name": "t"}, "relation_rows": {}, "expected_current_rows": 0,
               "write_ahead_log_bytes_after_checkpoint": 0}
    reasons = r6.verify(outcome)
    assert any("never deletes" in reason for reason in reasons)


def test_a_history_short_of_its_sources_is_caught_by_the_verifier():
    outcome = {"fact_rows_before": {"a": 5, "b": 5}, "fact_rows_after": {"a": 5, "b": 5},
               "selection": {"table_name": "b"},
               "relation_rows": {"df_coverage_version_selection": 1, "df_coverage_current": 5,
                                 "df_coverage_history": 5,
                                 "df_coverage_current_denominator": 1},
               "expected_current_rows": 5,
               "write_ahead_log_bytes_after_checkpoint": 0}
    reasons = r6.verify(outcome)
    assert any("every version" in reason for reason in reasons)


def test_applying_twice_keeps_one_current_matrix_and_both_selections(tmp_path):
    """Re-running records a new selection; it must not double the current view."""
    cube = build(tmp_path / "c.duckdb")
    apply(cube, tmp_path)
    first = counts(cube, "df_coverage_current", "df_coverage_version_selection")
    apply(cube, tmp_path)
    second = counts(cube, "df_coverage_current", "df_coverage_version_selection")
    assert second["df_coverage_current"] == first["df_coverage_current"]
    assert second["df_coverage_version_selection"] == 2
