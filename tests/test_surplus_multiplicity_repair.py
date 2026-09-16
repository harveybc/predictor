"""Surplus multiplicity in the live cube: detect it, remove exactly it, and never more.

I1-I2 of `docs/handoffs/MUSASHI_H1_H3_LIVE_REVIEW_AND_I1_I3_2026_09_16.md`.

Musashi read the running warehouse through its own API and found what my H1 report did not
describe: two terminals hold **each** of `bytes_delivered` and `delivery_from_cache` twice
where the accepted payload expects them once. Four surplus metric rows. My H1 reconciliation
was of a boundary-held snapshot at a moment that had already passed, and a terminal count of
55 cannot see a child-level duplication at all.

The rules below are the reviewer's finding frozen before anything is edited:

* an exact duplicate child is a **difference**, with its multiplicity named;
* expected multiplicity comes from the accepted payload, so a contract that legitimately
  carries the same row twice keeps both — there is no `SELECT DISTINCT` anywhere;
* a row whose VALUE differs is not surplus and is never removed automatically;
* the correction is atomic, preserves before-images, and a second invocation changes nothing;
* the deployed writer must not be able to append a child that is already present.

Everything here builds its own accounting database and its own cube under `tmp_path`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))
sys.path.insert(0, str(REPO / "olap" / "duckdb_store" / "src"))

TOOL = REPO / "tools" / "incident_evidence_reconcile.py"
spec = importlib.util.spec_from_file_location("reconcile_for_surplus_tests", TOOL)
reconcile = importlib.util.module_from_spec(spec)
sys.modules["reconcile_for_surplus_tests"] = reconcile
spec.loader.exec_module(reconcile)

duckdb = pytest.importorskip("duckdb")

MIGRATE = REPO / "tools" / "olap_duckdb_migrate.py"
_spec = importlib.util.spec_from_file_location("migrate_for_surplus_tests", MIGRATE)
migrate = importlib.util.module_from_spec(_spec)
sys.modules["migrate_for_surplus_tests"] = migrate
_spec.loader.exec_module(migrate)


# --- a cube built by the DEPLOYED writer, from payloads we choose --------------------------

def terminal_payload(index: int, *, metrics: list[dict]) -> dict:
    """A contract-valid payload whose digest is its own canonical body, with chosen metrics."""
    from predictor_olap_store.query import canonical_text

    body = {
        "schema": "governed_terminal.v1", "campaign_sha256": "c" * 64,
        "campaign_key": f"surplus-{index}", "unit_id": f"unit-{index}", "generation": 1,
        "actor": "a", "project": "p", "classification": "NON_GOVERNING",
        "status": "COMPLETED", "reason": None, "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z", "terminal_lake": "olap_cube",
        "config_sha256": "e" * 64, "code_identity": {"kind": "git_commit", "value": "d" * 40},
        "costs": {"wall_seconds": 1.0}, "tags": {}, "synthetic_spec_sha256": None,
        "deliveries": [], "metrics": metrics, "artifacts": [], "verified_datasets": [],
    }
    body["terminal_sha256"] = hashlib.sha256(
        canonical_text(body).encode("ascii")).hexdigest()
    return body


def metric(name: str, value: float, unit: str = "bytes") -> dict:
    return {"metric": name, "split": "test", "horizon": 0, "unit": unit, "value": value,
            "std_dev": None, "min_value": None, "max_value": None}


#: The two metrics the live duplication is on, with the values the reviewer observed.
LIVE_SHAPE = [metric("bytes_delivered", 228801.0), metric("delivery_from_cache", 1.0, "bool")]


def store_at(path: Path):
    from predictor_duckdb_store.provider import PredictorDuckdbStore

    store = PredictorDuckdbStore()
    store.set_params(duckdb_path=str(path), schema="main", memory_limit="1GB", threads=2,
                     min_free_bytes=1)
    store.engine()
    return store


def accounting_with(path: Path, payloads: list[dict]) -> Path:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE governed_terminals (terminal_sha256 TEXT PRIMARY KEY,"
                " campaign_sha256 TEXT, unit_id TEXT, generation INTEGER, status TEXT,"
                " terminal_lake TEXT, body_json TEXT, created_at TEXT)")
    for body in payloads:
        con.execute("INSERT INTO governed_terminals VALUES (?,?,?,?,?,?,?,?)",
                    (body["terminal_sha256"], body["campaign_sha256"], body["unit_id"],
                     body["generation"], body["status"], "olap_cube",
                     json.dumps(body, sort_keys=True), "2026-01-01T00:00:00Z"))
    con.commit()
    con.close()
    return path


def duplicate_rows(cube: Path, digest: str, names: tuple) -> None:
    """Append an exact second copy of the named metric rows: the live condition, reproduced."""
    con = duckdb.connect(str(cube))
    try:
        placeholders = ", ".join("?" for _ in names)
        con.execute(
            'INSERT INTO "main"."gov_terminal_metric"'
            " SELECT * FROM \"main\".\"gov_terminal_metric\""
            f" WHERE terminal_sha256 = ? AND metric IN ({placeholders})",
            [digest, *names])
    finally:
        con.close()


def metric_multiset(cube: Path, digest: str) -> dict:
    from collections import Counter

    con = duckdb.connect(str(cube), read_only=True)
    try:
        rows = con.execute(
            'SELECT metric, split, horizon, unit, value, std_dev, min_value, max_value'
            ' FROM "main"."gov_terminal_metric" WHERE terminal_sha256 = ?', [digest]).fetchall()
    finally:
        con.close()
    return dict(Counter(rows))


@pytest.fixture
def live_shape(tmp_path):
    """Two terminals written by the deployed writer; one then duplicated as the cube shows."""
    cube = tmp_path / "cube.duckdb"
    payloads = [terminal_payload(0, metrics=LIVE_SHAPE), terminal_payload(1, metrics=LIVE_SHAPE)]
    store = store_at(cube)
    for body in payloads:
        store.write_terminal(body)
    store.engine().dispose()
    accounting = accounting_with(tmp_path / "accounting.db", payloads)
    duplicate_rows(cube, payloads[0]["terminal_sha256"],
                   ("bytes_delivered", "delivery_from_cache"))
    return {"cube": cube, "accounting": accounting, "tmp": tmp_path,
            "affected": payloads[0]["terminal_sha256"],
            "untouched": payloads[1]["terminal_sha256"], "payloads": payloads}


def run(scene, *extra, name="report.json"):
    out = scene["tmp"] / name
    code = reconcile.main(["--accounting", str(scene["accounting"]),
                           "--cube", str(scene["cube"]), "--schema", "main",
                           "--out", str(out), *extra])
    return code, json.loads(out.read_text())


# --- the reviewer's finding, frozen -------------------------------------------------------

def test_an_exact_duplicate_child_is_a_difference_with_its_multiplicity(live_shape):
    code, report = run(live_shape)
    assert code == 1
    assert report["counts"]["content_differs"] == 1
    differing = report["content_differs"][0]
    assert differing["terminal_sha256"] == live_shape["affected"]
    surplus = differing["differences"]["children"]["gov_terminal_metric"]
    assert (surplus["missing"], surplus["extra"]) == (0, 2)


def test_a_terminal_count_alone_cannot_see_the_duplication(live_shape):
    _code, report = run(live_shape)
    assert report["counts"]["terminals_in_cube"] == 2
    assert report["counts"]["accepted_by_governance"] == 2
    assert report["counts"]["content_matches"] == 1


def test_the_additive_repair_never_removes_a_row(live_shape):
    """`--repair` adds what is missing. Facing surplus it must refuse, not reinterpret."""
    before = metric_multiset(live_shape["cube"], live_shape["affected"])
    _code, report = run(live_shape, "--repair", name="additive.json")
    assert report["counts"]["repaired"] == 0
    assert metric_multiset(live_shape["cube"], live_shape["affected"]) == before


# --- the bounded correction ---------------------------------------------------------------

def test_surplus_repair_ends_at_exactly_the_accepted_multiset(live_shape):
    evidence = live_shape["tmp"] / "surplus_evidence.json"
    code, report = run(live_shape, "--repair-surplus", "--evidence", str(evidence),
                       name="repair.json")
    assert code == 0
    assert report["counts"]["surplus_rows_removed"] == 2
    observed = metric_multiset(live_shape["cube"], live_shape["affected"])
    assert sorted(observed.values()) == [1, 1]


def test_surplus_repair_preserves_a_duplicate_the_contract_declares(tmp_path):
    """Two identical metric rows in the ACCEPTED payload are two rows, not one.

    A global de-duplication rule, or `SELECT DISTINCT`, would silently rewrite this terminal's
    accepted content. Expected multiplicity is read from the payload or it is not expected.
    """
    cube = tmp_path / "cube.duckdb"
    body = terminal_payload(0, metrics=[metric("bytes_delivered", 228801.0),
                                        metric("bytes_delivered", 228801.0)])
    store = store_at(cube)
    store.write_terminal(body)
    store.engine().dispose()
    scene = {"cube": cube, "accounting": accounting_with(tmp_path / "a.db", [body]),
             "tmp": tmp_path}
    code, report = run(scene, "--repair-surplus", "--evidence", str(tmp_path / "e.json"))
    assert code == 0
    assert report["counts"]["surplus_rows_removed"] == 0
    assert metric_multiset(cube, body["terminal_sha256"]) == {
        ("bytes_delivered", "test", 0, "bytes", 228801.0, None, None, None): 2}


def test_surplus_repair_refuses_a_row_whose_value_conflicts(live_shape):
    """Missing-and-extra is a changed value, not a surplus copy. Removing either is a rewrite."""
    con = duckdb.connect(str(live_shape["cube"]))
    con.execute('UPDATE "main"."gov_terminal_metric" SET value = 999999'
                " WHERE terminal_sha256 = ? AND metric = 'bytes_delivered'",
                [live_shape["untouched"]])
    con.close()
    before = metric_multiset(live_shape["cube"], live_shape["untouched"])
    code, report = run(live_shape, "--repair-surplus", "--evidence",
                       str(live_shape["tmp"] / "e.json"), name="conflict.json")
    assert code == 1
    conflicted = [row for row in report["content_differs"]
                  if row["terminal_sha256"] == live_shape["untouched"]]
    assert conflicted and conflicted[0]["repair"] == "REFUSED_NOT_PURE_SURPLUS"
    assert metric_multiset(live_shape["cube"], live_shape["untouched"]) == before


def test_surplus_repair_is_idempotent(live_shape):
    run(live_shape, "--repair-surplus", "--evidence", str(live_shape["tmp"] / "e1.json"),
        name="first.json")
    after_first = metric_multiset(live_shape["cube"], live_shape["affected"])
    code, report = run(live_shape, "--repair-surplus", "--evidence",
                       str(live_shape["tmp"] / "e2.json"), name="second.json")
    assert code == 0
    assert report["counts"]["surplus_rows_removed"] == 0
    assert metric_multiset(live_shape["cube"], live_shape["affected"]) == after_first


def test_surplus_repair_leaves_every_unaffected_row_untouched(live_shape):
    before = metric_multiset(live_shape["cube"], live_shape["untouched"])
    run(live_shape, "--repair-surplus", "--evidence", str(live_shape["tmp"] / "e.json"))
    assert metric_multiset(live_shape["cube"], live_shape["untouched"]) == before


def test_the_removed_rows_are_preserved_as_evidence_before_they_are_removed(live_shape):
    evidence = live_shape["tmp"] / "surplus_evidence.json"
    run(live_shape, "--repair-surplus", "--evidence", str(evidence))
    body = json.loads(evidence.read_text())
    removed = body["removed"]
    assert len(removed) == 2
    assert {row["relation"] for row in removed} == {"gov_terminal_metric"}
    assert {row["values"]["metric"] for row in removed} == {"bytes_delivered",
                                                            "delivery_from_cache"}
    assert all(row["terminal_sha256"] == live_shape["affected"] for row in removed)


def test_surplus_repair_without_an_evidence_path_is_refused(live_shape):
    """Removing a row without first preserving it is not a repair, whatever it is called."""
    before = metric_multiset(live_shape["cube"], live_shape["affected"])
    with pytest.raises(SystemExit):
        run(live_shape, "--repair-surplus", name="noevidence.json")
    assert metric_multiset(live_shape["cube"], live_shape["affected"]) == before


def test_an_interrupted_surplus_repair_removes_nothing(live_shape, monkeypatch):
    """Atomic: a failure part-way through leaves the multiset exactly as it was."""
    before = metric_multiset(live_shape["cube"], live_shape["affected"])
    original = reconcile.delete_by_rowid
    calls = {"n": 0}

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(reconcile, "delete_by_rowid", explode)
    _code, report = run(live_shape, "--repair-surplus", "--evidence",
                        str(live_shape["tmp"] / "e.json"), name="interrupted.json")
    assert metric_multiset(live_shape["cube"], live_shape["affected"]) == before
    assert report["content_differs"][0]["repair"].startswith("FAILED")


# --- which path can append an already present child ---------------------------------------

def test_the_deployed_writer_cannot_append_an_already_present_child(tmp_path):
    """Normal ingestion, twice. The second delivery must add no child row at all."""
    cube = tmp_path / "cube.duckdb"
    body = terminal_payload(0, metrics=LIVE_SHAPE)
    store = store_at(cube)
    first = store.write_terminal(body)
    second = store.write_terminal(body)
    store.engine().dispose()
    assert first["stored"] is True and second["already_stored"] is True
    assert sum(metric_multiset(cube, body["terminal_sha256"]).values()) == 2


def test_the_additive_repair_cannot_append_an_already_present_child(live_shape):
    """The G1 path, re-run against a cube that already holds every accepted row."""
    complete = live_shape["untouched"]
    before = metric_multiset(live_shape["cube"], complete)
    run(live_shape, "--repair", name="reapply.json")
    assert metric_multiset(live_shape["cube"], complete) == before


def test_overlapping_additive_repairs_cannot_duplicate_a_child(tmp_path):
    """Two repairs that both decided a row was missing, the second committing after the first.

    This is the shape an operator retry actually takes: the decision is made from a read, and
    the write happens later. The second writer must re-read inside its own transaction, or the
    retry itself becomes the duplication.
    """
    cube = tmp_path / "cube.duckdb"
    body = terminal_payload(0, metrics=LIVE_SHAPE)
    store = store_at(cube)
    store.write_terminal(body)
    store.engine().dispose()
    digest = body["terminal_sha256"]
    con = duckdb.connect(str(cube))
    con.execute('DELETE FROM "main"."gov_terminal_metric" WHERE terminal_sha256 = ?', [digest])
    con.close()

    scene = {"cube": cube, "accounting": accounting_with(tmp_path / "a.db", [body]),
             "tmp": tmp_path}
    run(scene, "--repair", name="retry-a.json")
    run(scene, "--repair", name="retry-b.json")
    assert sum(metric_multiset(cube, digest).values()) == 2


def test_an_interrupted_surplus_repair_can_be_retried_to_the_accepted_multiset(live_shape,
                                                                               monkeypatch):
    """Interruption then retry converges: the first attempt removed nothing, the second all."""
    original = reconcile.delete_by_rowid
    calls = {"n": 0}

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(reconcile, "delete_by_rowid", explode)
    run(live_shape, "--repair-surplus", "--evidence", str(live_shape["tmp"] / "e1.json"),
        name="attempt1.json")
    monkeypatch.setattr(reconcile, "delete_by_rowid", original)
    code, report = run(live_shape, "--repair-surplus", "--evidence",
                       str(live_shape["tmp"] / "e2.json"), name="attempt2.json")
    assert code == 0
    assert report["counts"]["surplus_rows_removed"] == 2
    assert sorted(metric_multiset(live_shape["cube"], live_shape["affected"]).values()) == [1, 1]


# --- reading the cube through the running service -----------------------------------------

def serve(cube: Path, schema: str = "main", page_cap: int | None = None):
    """A stand-in for the warehouse's read API, backed by the same cube file.

    The point under test is the reconciler's side of the contract: pages, the separate count
    check, and the bearer token coming from the environment. `page_cap` truncates every page,
    which is the failure the count check exists to catch.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    connection = duckdb.connect(str(cube), read_only=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if self.headers.get("Authorization") != "Bearer test-token":
                return self.send_error(401)
            if parsed.path == "/api/v1/schema":
                relation = params["relation"][0]
                names = [row[1] for row in connection.execute(
                    f'PRAGMA table_info("{schema}"."{relation}")').fetchall()]
                body = {"columns": [{"name": name} for name in names]}
            else:
                sql = params["sql"][0]
                if page_cap is not None and " LIMIT " in sql:
                    head, _, _tail = sql.rpartition(" LIMIT ")
                    sql = f"{head} LIMIT {page_cap}"
                result = connection.execute(sql)
                names = [column[0] for column in result.description]
                body = {"rows": [dict(zip(names, row)) for row in result.fetchall()]}
            payload = json.dumps(body, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_the_service_route_reports_the_same_contents_as_the_file(live_shape, monkeypatch):
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    _code, by_file = run(live_shape, name="byfile.json")
    server, url = serve(live_shape["cube"])
    try:
        out = live_shape["tmp"] / "byservice.json"
        code = reconcile.main(["--accounting", str(live_shape["accounting"]),
                               "--service-url", url, "--schema", "main", "--out", str(out)])
    finally:
        server.shutdown()
    by_service = json.loads(out.read_text())
    assert code == 1
    assert by_service["counts"] == by_file["counts"]
    assert by_service["content_differs"] == by_file["content_differs"]
    assert by_service["cube"].endswith("(live service)")


def test_a_truncated_service_page_is_refused_not_reported_as_a_population(live_shape,
                                                                         monkeypatch):
    """A short page that is counted as the whole population is how a count-based claim lies."""
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    server, url = serve(live_shape["cube"], page_cap=1)
    try:
        with pytest.raises(RuntimeError, match="the service counts"):
            reconcile.main(["--accounting", str(live_shape["accounting"]),
                            "--service-url", url, "--schema", "main",
                            "--out", str(live_shape["tmp"] / "truncated.json")])
    finally:
        server.shutdown()


def test_the_service_route_refuses_to_repair(live_shape, monkeypatch):
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    with pytest.raises(SystemExit):
        reconcile.main(["--accounting", str(live_shape["accounting"]),
                        "--service-url", "http://127.0.0.1:1", "--out",
                        str(live_shape["tmp"] / "x.json"), "--repair-surplus",
                        "--evidence", str(live_shape["tmp"] / "e.json")])


def test_the_service_token_is_never_an_argument(live_shape, monkeypatch):
    monkeypatch.delenv("DATA_GOV_LAKE_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="is not set"):
        reconcile.main(["--accounting", str(live_shape["accounting"]),
                        "--service-url", "http://127.0.0.1:1", "--schema", "main",
                        "--out", str(live_shape["tmp"] / "x.json")])


# --- the path that CAN append an already present child ------------------------------------

def test_a_write_ahead_log_replayed_onto_a_checkpointed_base_doubles_its_rows(tmp_path):
    """The mechanism, reproduced: this is how identical child rows come to exist twice.

    A writer that commits and dies leaves a log. The next opener replays it and folds it into
    the file. If that same log is then put back beside the file — restored from a copy, or
    simply kept — it is replayed a second time onto a base that already contains it, and every
    row it carries appears twice. No writer ran twice; nothing was ingested twice.
    """
    import shutil
    import subprocess

    cube = tmp_path / "c.duckdb"
    log = Path(str(cube) + ".wal")
    con = duckdb.connect(str(cube))
    con.execute("CREATE TABLE m (terminal TEXT, metric TEXT, value DOUBLE)")
    con.close()

    writer = tmp_path / "writer.py"
    writer.write_text(
        "import duckdb, os, signal\n"
        f"con = duckdb.connect({str(cube)!r})\n"
        "con.execute(\"INSERT INTO m VALUES ('t','bytes_delivered',228801.0)\")\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n")
    subprocess.run([sys.executable, str(writer)], check=False)
    if not log.exists():
        pytest.skip("this build does not leave a replayable log after an unclean exit")
    kept = tmp_path / "kept.wal"
    shutil.copy2(log, kept)

    con = duckdb.connect(str(cube))
    assert con.execute("SELECT count(*) FROM m").fetchone()[0] == 1
    con.close()

    shutil.copy2(kept, log)
    con = duckdb.connect(str(cube))
    doubled = con.execute("SELECT count(*) FROM m").fetchone()[0]
    con.close()
    assert doubled == 2


def test_a_repair_folds_its_own_log_into_the_cube_before_it_returns(tmp_path):
    """So the repair cannot become the input to that mechanism."""
    cube = tmp_path / "cube.duckdb"
    body = terminal_payload(0, metrics=LIVE_SHAPE)
    store = store_at(cube)
    store.write_terminal(body)
    store.engine().dispose()
    digest = body["terminal_sha256"]
    con = duckdb.connect(str(cube))
    con.execute('DELETE FROM "main"."gov_terminal_metric" WHERE terminal_sha256 = ?', [digest])
    con.close()

    scene = {"cube": cube, "accounting": accounting_with(tmp_path / "a.db", [body]),
             "tmp": tmp_path}
    _code, report = run(scene, "--repair", name="checkpointed.json")
    assert report["counts"]["repaired"] == 1
    assert report["write_ahead_log_bytes_after_checkpoint"] == 0


def test_a_read_only_reconciliation_reports_no_log_measurement(live_shape):
    """Nothing is checkpointed on a read, so nothing is claimed about the log."""
    _code, report = run(live_shape, name="readonly.json")
    assert report["write_ahead_log_bytes_after_checkpoint"] is None


# --- a receipt tied to the bytes it describes ---------------------------------------------

def test_the_report_states_the_content_it_was_computed_from(live_shape):
    _code, report = run(live_shape, name="stated.json")
    content = report["source_content"]
    assert content["gov_terminal"]["rows"] == 2
    assert content["gov_terminal_metric"]["rows"] == 6      # 4 accepted + 2 surplus copies
    assert len(content["gov_terminal_metric"]["md5"]) == 32


def test_the_stated_content_moves_when_a_single_row_is_duplicated(live_shape):
    _code, before = run(live_shape, name="before.json")
    duplicate_rows(live_shape["cube"], live_shape["untouched"], ("bytes_delivered",))
    _code, after = run(live_shape, name="after.json")
    assert (after["source_content"]["gov_terminal_metric"]["md5"]
            != before["source_content"]["gov_terminal_metric"]["md5"])
    assert after["source_content"]["gov_terminal_metric"]["rows"] == 7


def test_the_stated_content_survives_the_service_route(live_shape, monkeypatch):
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    _code, by_file = run(live_shape, name="f.json")
    server, url = serve(live_shape["cube"])
    try:
        out = live_shape["tmp"] / "s.json"
        reconcile.main(["--accounting", str(live_shape["accounting"]), "--service-url", url,
                        "--schema", "main", "--out", str(out)])
    finally:
        server.shutdown()
    assert json.loads(out.read_text())["source_content"] == by_file["source_content"]


def test_a_repair_receipt_states_the_content_it_left_behind(live_shape):
    _code, report = run(live_shape, "--repair-surplus", "--evidence",
                        str(live_shape["tmp"] / "e.json"), name="after.json")
    before = report["source_content"]["gov_terminal_metric"]
    after = report["content_after_repair"]["gov_terminal_metric"]
    assert before["rows"] == 6 and after["rows"] == 4
    assert before["md5"] != after["md5"]
    for relation in ("gov_terminal", "gov_terminal_artifact", "gov_terminal_dataset"):
        assert (report["source_content"][relation]["md5"]
                == report["content_after_repair"][relation]["md5"])


# --- freezing the live evidence without an outage -----------------------------------------

FREEZE = REPO / "tools" / "olap_freeze_live_evidence.py"
_fspec = importlib.util.spec_from_file_location("freeze_for_surplus_tests", FREEZE)
freeze = importlib.util.module_from_spec(_fspec)
sys.modules["freeze_for_surplus_tests"] = freeze
_fspec.loader.exec_module(freeze)


def test_a_live_read_is_named_for_what_it_proves(live_shape, monkeypatch):
    """`CONSISTENT_LIVE_READ`: the content did not move while it was read. Nothing stronger."""
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    server, url = serve(live_shape["cube"])
    try:
        out = live_shape["tmp"] / "freeze.json"
        code = freeze.main(["--service-url", url, "--schema", "main",
                            "--target", str(live_shape["tmp"] / "evidence.duckdb"),
                            "--out", str(out)])
    finally:
        server.shutdown()
    receipt = json.loads(out.read_text())
    assert code == 0
    assert receipt["kind"] == "CONSISTENT_LIVE_READ"
    assert receipt["counts"]["gov_terminal_metric"] == 6
    assert "not a transactional snapshot" in receipt["limit"]


def test_the_evidence_copy_reproduces_the_services_own_digests(live_shape, monkeypatch):
    """Three agreements, and the copy's digest is recomputed rather than carried over."""
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    server, url = serve(live_shape["cube"])
    try:
        out = live_shape["tmp"] / "freeze.json"
        freeze.main(["--service-url", url, "--schema", "main",
                     "--target", str(live_shape["tmp"] / "evidence.duckdb"),
                     "--out", str(out)])
    finally:
        server.shutdown()
    receipt = json.loads(out.read_text())
    for relation in ("gov_terminal", "gov_terminal_metric", "gov_terminal_dataset"):
        assert (receipt["service_digests_before_read"][relation]
                == receipt["service_digests_after_read"][relation]
                == receipt["evidence_copy_digests"][relation])


def test_a_truncated_read_never_becomes_an_evidence_copy(live_shape, monkeypatch):
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    server, url = serve(live_shape["cube"], page_cap=1)
    try:
        with pytest.raises(RuntimeError, match="the service counts"):
            freeze.main(["--service-url", url, "--schema", "main",
                         "--target", str(live_shape["tmp"] / "evidence.duckdb"),
                         "--out", str(live_shape["tmp"] / "freeze.json")])
    finally:
        server.shutdown()


def test_an_evidence_copy_is_never_written_over(live_shape, monkeypatch):
    monkeypatch.setenv("DATA_GOV_LAKE_TOKEN", "test-token")
    target = live_shape["tmp"] / "evidence.duckdb"
    target.write_bytes(b"")
    with pytest.raises(SystemExit, match="exists"):
        freeze.main(["--service-url", "http://127.0.0.1:1", "--schema", "main",
                     "--target", str(target), "--out", str(live_shape["tmp"] / "f.json")])


def test_both_repairs_in_one_invocation_never_act_on_a_stale_report(live_shape):
    """`--repair` changes the cube, so the surplus decision must not run off the same read."""
    con = duckdb.connect(str(live_shape["cube"]))
    con.execute('DELETE FROM "main"."gov_terminal_metric" WHERE terminal_sha256 = ?',
                [live_shape["untouched"]])
    con.close()
    before_surplus = metric_multiset(live_shape["cube"], live_shape["affected"])
    _code, report = run(live_shape, "--repair", "--repair-surplus", "--evidence",
                        str(live_shape["tmp"] / "e.json"), name="both.json")
    restored = [row for row in report["repaired"]
                if row["terminal_sha256"] == live_shape["untouched"]]
    assert restored and restored[0]["repair"] == "RESTORED"
    # the additively repaired terminal is not then "de-duplicated" off the same stale read
    assert sum(metric_multiset(live_shape["cube"], live_shape["untouched"]).values()) == 2
    # and the genuinely surplus terminal is still corrected
    assert sorted(metric_multiset(live_shape["cube"], live_shape["affected"]).values()) == [1, 1]
    assert before_surplus != metric_multiset(live_shape["cube"], live_shape["affected"])
