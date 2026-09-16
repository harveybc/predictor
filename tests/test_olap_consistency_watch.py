"""The watch must see both faults, must cost what it says, and must never be able to write.

Block 2 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`:

    "Keep a bounded read-only reconciliation/monitoring check for warehouse multiplicity and
     predicate-versus-scan consistency, with measured query cost. Define cadence from that cost
     instead of running full scans on every request. Record discrepancies and alert; no
     automatic deletion, reindexing or repair."

Both faults it watches for were real: four metric rows present twice, and an index that reached
533 rows of a 537-row table. Neither was visible in a terminal count, and the second was not
visible to any filtered read at all.

The service here is a stand-in backed by a throwaway DuckDB file. No production service,
database or port is touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "olap_consistency_watch.py"
spec = importlib.util.spec_from_file_location("olap_consistency_watch_under_test", TOOL)
watch = importlib.util.module_from_spec(spec)
sys.modules["olap_consistency_watch_under_test"] = watch
spec.loader.exec_module(watch)

duckdb = pytest.importorskip("duckdb")

TOKEN = "watch-test-token"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def build(path: Path, *, duplicate: bool = False) -> Path:
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA IF NOT EXISTS main")
    con.execute('CREATE TABLE main.gov_terminal (terminal_sha256 TEXT PRIMARY KEY)')
    con.execute('CREATE TABLE main.gov_terminal_metric (terminal_sha256 TEXT, metric TEXT,'
                " value DOUBLE)")
    con.execute('CREATE TABLE main.gov_terminal_dataset (terminal_sha256 TEXT, sha256 TEXT)')
    con.execute('CREATE TABLE main.gov_terminal_artifact (terminal_sha256 TEXT, role TEXT)')
    for digest in (DIGEST_A, DIGEST_B):
        con.execute("INSERT INTO main.gov_terminal VALUES (?)", [digest])
        con.execute("INSERT INTO main.gov_terminal_metric VALUES (?, 'bytes_delivered', 1.0)",
                    [digest])
        con.execute("INSERT INTO main.gov_terminal_dataset VALUES (?, 'd')", [digest])
        con.execute("INSERT INTO main.gov_terminal_artifact VALUES (?, 'log')", [digest])
    if duplicate:
        con.execute("INSERT INTO main.gov_terminal_metric VALUES (?, 'bytes_delivered', 1.0)",
                    [DIGEST_A])
    con.close()
    return path


def serve(cube: Path, *, predicate_short_by: int = 0):
    """A stand-in warehouse read API.

    `predicate_short_by` makes a filtered read return fewer rows than a scan, which is what an
    index short of its table does and what no honest query can be made to do on demand.
    """
    connection = duckdb.connect(str(cube), read_only=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self.send_error(401)
            sql = parse_qs(parsed.query)["sql"][0]
            result = connection.execute(sql)
            names = [column[0] for column in result.description]
            rows = [dict(zip(names, row)) for row in result.fetchall()]
            if (predicate_short_by and "OFFSET 0" not in sql and "WHERE terminal_sha256 IN" in sql
                    and "gov_terminal_metric" in sql and rows and "n" in rows[0]):
                rows = [{"n": rows[0]["n"] - predicate_short_by}]
            payload = json.dumps({"rows": rows}, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def run(url: str, tmp_path: Path, monkeypatch, *extra):
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", TOKEN)
    state = tmp_path / "state"
    code = watch.main(["--service-url", url, "--schema", "main", "--state", str(state),
                       "--json", *extra])
    return code, json.loads((state / "LATEST.json").read_text(encoding="utf-8")), state


# --- what it must see ----------------------------------------------------------------------

def test_a_clean_cube_raises_no_alert(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb"))
    try:
        code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert code == 0
    assert report["alert"] is False and report["findings"] == []
    assert report["terminals"] == 2
    assert all(entry["predicate_vs_scan"]["agrees"] for entry in report["relations"].values())


def test_a_row_present_twice_is_reported(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb", duplicate=True))
    try:
        code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert code == 1 and report["alert"] is True
    kinds = {finding["kind"] for finding in report["findings"]}
    assert kinds == {"MULTIPLICITY"}
    metric = report["relations"]["gov_terminal_metric"]
    assert metric["rows_present_more_than_once"] == 1
    assert metric["duplicate_groups"][0]["n"] == 2


def test_a_filter_and_a_scan_that_disagree_are_reported(tmp_path, monkeypatch):
    """The fault a filtered read cannot show you, because it is the filtered read that lies."""
    server, url = serve(build(tmp_path / "c.duckdb"), predicate_short_by=1)
    try:
        code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert code == 1 and report["alert"] is True
    finding = next(f for f in report["findings"] if f["kind"] == "PREDICATE_VS_SCAN")
    assert finding["relation"] == "gov_terminal_metric"
    agreement = report["relations"]["gov_terminal_metric"]["predicate_vs_scan"]
    assert agreement["by_predicate"] == 1 and agreement["by_scan"] == 2
    assert agreement["agrees"] is False


def test_both_faults_at_once_are_both_reported(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb", duplicate=True), predicate_short_by=1)
    try:
        _code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert {finding["kind"] for finding in report["findings"]} == {"MULTIPLICITY",
                                                                  "PREDICATE_VS_SCAN"}


# --- what it must cost, and how often it may run -------------------------------------------

def test_the_report_states_its_own_measured_cost(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb"))
    try:
        _code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    cost = report["cost"]
    assert cost["queries"] > 0
    assert cost["query_seconds"] >= 0 and cost["wall_seconds"] >= cost["query_seconds"]
    assert cost["duty_cycle"] == watch.DEFAULT_DUTY_CYCLE


def test_the_query_count_does_not_grow_with_the_number_of_terminals(tmp_path, monkeypatch):
    """Two queries per relation, not two per terminal: the cadence has to stay affordable."""
    small = build(tmp_path / "small.duckdb")
    server, url = serve(small)
    try:
        _code, first, _state = run(url, tmp_path / "a", monkeypatch)
    finally:
        server.shutdown()
    big = build(tmp_path / "big.duckdb")
    con = duckdb.connect(str(big))
    for index in range(200):
        digest = f"{index:064x}"
        con.execute("INSERT INTO main.gov_terminal VALUES (?)", [digest])
        con.execute("INSERT INTO main.gov_terminal_metric VALUES (?, 'm', 1.0)", [digest])
    con.close()
    server, url = serve(big)
    try:
        _code, second, _state = run(url, tmp_path / "b", monkeypatch)
    finally:
        server.shutdown()
    assert second["terminals"] == 202
    assert second["cost"]["queries"] == first["cost"]["queries"]


def test_the_cadence_follows_the_measured_cost():
    assert watch.recommended_interval(0.001, 0.01) == watch.MIN_INTERVAL_SECONDS
    assert watch.recommended_interval(60.0, 0.01) == 6000
    assert watch.recommended_interval(100000.0, 0.01) == watch.MAX_INTERVAL_SECONDS
    assert watch.recommended_interval(1.0, 0) == watch.MAX_INTERVAL_SECONDS
    cheap = watch.recommended_interval(2.0, 0.01)
    dear = watch.recommended_interval(20.0, 0.01)
    assert dear > cheap


def test_the_report_recommends_an_interval_within_the_bounds(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb"))
    try:
        _code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    interval = report["recommended_interval_seconds"]
    assert watch.MIN_INTERVAL_SECONDS <= interval <= watch.MAX_INTERVAL_SECONDS


# --- what it must never do -----------------------------------------------------------------

def code_without_prose() -> str:
    """The tool's source with every docstring removed.

    The claim being tested is about what the code can DO, and a docstring that says the tool
    never deletes anything would otherwise fail a search for the word "delete". Prose is not
    behaviour, in either direction.
    """
    import ast

    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    lines = TOOL.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for number, line in enumerate(lines, 1) if number not in skip)


def test_the_watch_has_no_way_to_write_anything():
    body = code_without_prose()
    for forbidden in ("DELETE ", "UPDATE ", "INSERT ", "DROP ", "CREATE INDEX", "CHECKPOINT",
                      "ALTER ", "--repair", "repair_surplus", "reindex_relation"):
        assert forbidden not in body, forbidden
    # one way in, and it is a read
    assert body.count("urlopen") == 1
    assert "def query" in body
    assert "data=" not in body and "method=" not in body


def test_no_finding_causes_an_action(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb", duplicate=True), predicate_short_by=1)
    try:
        _code, report, _state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert report["actions_taken"] == []
    assert "not a governed campaign" in report["note"]


def test_the_cube_is_unchanged_by_being_watched(tmp_path, monkeypatch):
    cube = build(tmp_path / "c.duckdb", duplicate=True)
    before = cube.read_bytes()
    server, url = serve(cube)
    try:
        run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert cube.read_bytes() == before


def test_the_token_is_never_an_argument(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_GOV_LAKE_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="is not set"):
        watch.main(["--service-url", "http://127.0.0.1:1", "--schema", "main"])


# --- what it must keep ---------------------------------------------------------------------

def test_every_observation_is_appended_and_none_replaced(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb"))
    try:
        run(url, tmp_path, monkeypatch)
        run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    lines = (tmp_path / "state" / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert {"generated_utc", "terminals", "findings", "alert", "cost",
                "recommended_interval_seconds"} <= set(entry)


def test_a_discrepancy_is_kept_in_the_log_not_only_in_the_latest_report(tmp_path, monkeypatch):
    server, url = serve(build(tmp_path / "c.duckdb", duplicate=True))
    try:
        run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    server, url = serve(build(tmp_path / "clean.duckdb"))
    try:
        _code, latest, state = run(url, tmp_path, monkeypatch)
    finally:
        server.shutdown()
    assert latest["alert"] is False
    lines = [json.loads(line) for line in
             (state / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["alert"] is True and lines[1]["alert"] is False
