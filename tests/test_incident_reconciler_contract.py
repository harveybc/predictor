"""The reconciler must compare survivors against independently retained EXPECTED content.

G1 of `docs/handoffs/MUSASHI_F1_F5_REVIEW_AND_G1_G3_2026_09_16.md`.

The defect is mine and it is the worst kind: my "no loss at child level" evidence hashed the
recovered cube and compared it with **itself**. Musashi ran the shipped `main()` three times and
got exit 0 every time — with the original fixture, with a metric silently changed to 999999,
and with all three child tables emptied. A verifier that cannot fail cannot verify.

What makes it checkable is that `data-gov`'s accounting retains the FULL canonical terminal
payload it accepted (`governed_terminals.body_json`: metrics, artifacts, verified_datasets,
identity, costs, tags) in a database the incident never touched. That is the expected content,
and it is not derived from the rows under examination.

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
TOOL = REPO / "tools" / "incident_evidence_reconcile.py"

spec = importlib.util.spec_from_file_location("incident_evidence_reconcile_under_test", TOOL)
reconcile = importlib.util.module_from_spec(spec)
sys.modules["incident_evidence_reconcile_under_test"] = reconcile
spec.loader.exec_module(reconcile)

duckdb = pytest.importorskip("duckdb")

MIGRATE = REPO / "tools" / "olap_duckdb_migrate.py"
_spec = importlib.util.spec_from_file_location("migrate_for_reconciler_tests", MIGRATE)
migrate = importlib.util.module_from_spec(_spec)
sys.modules["migrate_for_reconciler_tests"] = migrate
_spec.loader.exec_module(migrate)


def accounting_with(path: Path, terminals: list[dict]) -> Path:
    """An accounting database holding the canonical payloads, as data-gov retains them."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE governed_terminals (terminal_sha256 TEXT PRIMARY KEY,"
                " campaign_sha256 TEXT, unit_id TEXT, generation INTEGER, status TEXT,"
                " terminal_lake TEXT, body_json TEXT, created_at TEXT)")
    for entry in terminals:
        con.execute("INSERT INTO governed_terminals VALUES (?,?,?,?,?,?,?,?)",
                    (entry["terminal_sha256"], entry["campaign_sha256"], entry["unit_id"],
                     entry["generation"], entry["status"], "olap_cube",
                     json.dumps(entry["body"], sort_keys=True), "2026-01-01T00:00:00Z"))
    con.commit()
    con.close()
    return path


def payloads_written() -> list:
    """The payloads the fixture builder actually wrote.

    Taken from the builder rather than rebuilt beside it: a hand-copied expectation drifts from
    the fixture the moment either changes, and an expectation that drifts is the defect being
    corrected. It also has to carry the SAME availability contract digest the builder used.
    """
    return list(migrate.build_fixture_cube.last_payloads)


@pytest.fixture
def scene(tmp_path):
    cube = str(tmp_path / "cube.duckdb")
    migrate.build_fixture_cube(cube, terminals=1, with_children=True, with_contract=True)
    payload = payloads_written()[0]
    digest = payload["terminal_sha256"]
    accounting = accounting_with(tmp_path / "accounting.db", [{
        "terminal_sha256": digest, "campaign_sha256": payload["campaign_sha256"],
        "unit_id": payload["unit_id"], "generation": 1, "status": "COMPLETED",
        "body": payload}])
    return {"cube": cube, "accounting": accounting, "digest": digest, "tmp": tmp_path}


def run(scene, name="report.json"):
    out = scene["tmp"] / name
    code = reconcile.main(["--accounting", str(scene["accounting"]), "--cube", scene["cube"],
                           "--schema", "main", "--out", str(out)])
    return code, json.loads(out.read_text())


# --- the reviewer's three cases -----------------------------------------------------------

def test_the_untouched_fixture_reconciles(scene):
    code, report = run(scene)
    assert code == 0
    assert report["counts"]["content_matches"] == 1
    assert report["counts"]["content_differs"] == 0


def test_a_silently_changed_metric_is_caught(scene):
    """999999 hashes perfectly well. The point is that it is not what was accepted."""
    con = duckdb.connect(scene["cube"])
    con.execute("UPDATE main.gov_terminal_metric SET value = 999999")
    con.close()

    code, report = run(scene, "changed.json")

    assert code != 0, "a verifier that cannot fail cannot verify"
    assert report["counts"]["content_differs"] == 1
    differing = report["content_differs"][0]
    assert "gov_terminal_metric" in json.dumps(differing)


def test_emptied_children_are_reported_as_loss_not_as_absence(scene):
    con = duckdb.connect(scene["cube"])
    for child in ("gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact"):
        con.execute(f"DELETE FROM main.{child}")
    con.close()

    code, report = run(scene, "emptied.json")

    assert code != 0
    assert report["counts"]["content_differs"] == 1
    body = json.dumps(report["content_differs"][0])
    assert "missing" in body.lower()


# --- identity, not only presence ----------------------------------------------------------

def test_a_changed_parent_field_is_caught(scene):
    con = duckdb.connect(scene["cube"])
    con.execute("UPDATE main.gov_terminal SET actor = 'somebody-else'")
    con.close()
    code, report = run(scene, "actor.json")
    assert code != 0
    assert any("actor" in json.dumps(entry) for entry in report["content_differs"])


def test_a_wrong_generation_is_caught(scene):
    con = duckdb.connect(scene["cube"])
    con.execute("UPDATE main.gov_terminal SET generation = 7")
    con.close()
    code, report = run(scene, "generation.json")
    assert code != 0


# --- legitimate empties and unverifiable populations --------------------------------------

def test_a_refused_terminal_with_no_children_is_legitimate(tmp_path):
    """A REFUSED unit has no deliveries or metrics BY CONSTRUCTION. That is not loss."""
    cube = str(tmp_path / "refused.duckdb")
    migrate.build_fixture_cube(cube, terminals=1)
    payload = dict(payloads_written()[0])
    payload["status"] = "REFUSED"
    import sys as _sys
    _sys.path.insert(0, str(REPO / "olap" / "store" / "src"))
    from predictor_olap_store.query import canonical_text
    payload.pop("terminal_sha256")
    payload["terminal_sha256"] = hashlib.sha256(
        canonical_text(payload).encode("ascii")).hexdigest()
    digest = payload["terminal_sha256"]
    accounting = accounting_with(tmp_path / "acc.db", [{
        "terminal_sha256": digest, "campaign_sha256": payload["campaign_sha256"],
        "unit_id": payload["unit_id"], "generation": 1, "status": "COMPLETED",
        "body": payload}])
    con = duckdb.connect(cube)
    con.execute("UPDATE main.gov_terminal SET terminal_sha256 = ?, status = 'REFUSED'",
                [digest])
    con.close()
    out = tmp_path / "r.json"
    code = reconcile.main(["--accounting", str(accounting), "--cube", cube,
                           "--schema", "main", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["counts"]["content_matches"] == 1
    assert code == 0


def test_a_terminal_without_a_retained_payload_is_unverifiable_not_preserved(tmp_path):
    cube = str(tmp_path / "c.duckdb")
    migrate.build_fixture_cube(cube, terminals=1, with_children=True)
    digest = payloads_written()[0]["terminal_sha256"]
    accounting = accounting_with(tmp_path / "acc.db", [{
        "terminal_sha256": digest, "campaign_sha256": "c" * 64, "unit_id": "unit-0",
        "generation": 1, "status": "COMPLETED", "body": {}}])      # no payload retained
    out = tmp_path / "r.json"
    code = reconcile.main(["--accounting", str(accounting), "--cube", cube,
                           "--schema", "main", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["counts"]["content_unverifiable"] == 1
    assert report["counts"]["content_matches"] == 0, (
        "an absent expectation is not a match; it is an unverifiable population")
    assert code != 0


def test_an_empty_accounting_never_produces_a_blanket_no_loss(tmp_path):
    """Nothing to compare against is not evidence of preservation."""
    cube = str(tmp_path / "c.duckdb")
    migrate.build_fixture_cube(cube, terminals=2, with_children=True)
    accounting = accounting_with(tmp_path / "acc.db", [])
    out = tmp_path / "r.json"
    code = reconcile.main(["--accounting", str(accounting), "--cube", cube,
                           "--schema", "main", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code != 0
    assert report["counts"]["accepted_by_governance"] == 0
    assert report["counts"]["cube_rows_without_an_accepted_record"] == 2
    assert report["verdict"] != "NO_LOSS"


# --- H1: every PERSISTED field, not a handpicked list ---------------------------------------

def mutate(cube: str, sql: str) -> None:
    con = duckdb.connect(cube)
    con.execute(sql)
    con.close()


def test_a_changed_cost_is_caught(scene):
    """The reviewer's probe: costs_json altered to wall_seconds=999999, still NO_LOSS before."""
    mutate(scene["cube"], "UPDATE main.gov_terminal SET costs_json = "
                          "'{\"wall_seconds\":999999.0}'")
    code, report = run(scene, "costs.json")
    assert code != 0
    assert any("costs" in json.dumps(entry["differences"]) for entry in report["content_differs"])


def test_a_changed_availability_contract_link_is_caught(scene):
    """The reviewer's second probe: the stored contract digest set to 64 zeroes."""
    mutate(scene["cube"], "UPDATE main.gov_terminal_dataset SET "
                          "availability_contract_sha256 = repeat('0', 64)")
    code, report = run(scene, "contract.json")
    assert code != 0
    assert any("availability_contract_sha256" in json.dumps(entry["differences"])
               for entry in report["content_differs"])


@pytest.mark.parametrize("column,value", [
    ("code_identity_json", "'{\"kind\":\"git_commit\",\"value\":\"deadbeef\"}'"),
    ("tags_json", "'{\"tampered\":\"yes\"}'"),
    ("started_at", "'1999-01-01T00:00:00Z'"),
    ("finished_at", "'1999-01-01T00:00:01Z'"),
    ("terminal_lake", "'somewhere_else'"),
    ("campaign_key", "'a-different-campaign'"),
    ("reason", "'invented'"),
    ("config_sha256", "repeat('9', 64)"),
])
def test_every_persisted_parent_field_is_compared(scene, column, value):
    """One rule per persisted column: a list nobody checks is the defect being corrected."""
    mutate(scene["cube"], f"UPDATE main.gov_terminal SET {column} = {value}")
    code, report = run(scene, f"{column}.json")
    assert code != 0, f"{column} is persisted and was not compared"


@pytest.mark.parametrize("column,value", [
    ("source_sha256", "repeat('7', 64)"),
    ("delivery_kind", "'REWRITTEN'"),
    ("range_from", "'1999-01-01'"),
    ("time_column", "'not_the_column'"),
    ("bytes", "424242"),
])
def test_every_persisted_dataset_field_is_compared(scene, column, value):
    mutate(scene["cube"], f"UPDATE main.gov_terminal_dataset SET {column} = {value}")
    code, report = run(scene, f"ds_{column}.json")
    assert code != 0, f"gov_terminal_dataset.{column} is persisted and was not compared"


def test_the_mapping_names_every_persisted_column(scene):
    """The coverage claim is checkable: each stored column is compared, excused or reported."""
    report_code, report = run(scene, "coverage.json")
    coverage = report["field_coverage"]
    con = duckdb.connect(scene["cube"], read_only=True)
    try:
        for relation in ("gov_terminal", "gov_terminal_metric", "gov_terminal_dataset",
                         "gov_terminal_artifact"):
            columns = {row[1] for row in con.execute(
                f'PRAGMA table_info("main"."{relation}")').fetchall()}
            described = set(coverage[relation]["compared"]) | set(
                coverage[relation]["not_persisted_from_payload"])
            assert columns <= described, (
                f"{relation}: {sorted(columns - described)} is stored and neither compared "
                "nor explained")
    finally:
        con.close()


def test_a_payload_that_does_not_match_its_digest_is_unverifiable(tmp_path):
    """A retained expectation must itself be valid, or it cannot be an expectation."""
    cube = str(tmp_path / "c.duckdb")
    migrate.build_fixture_cube(cube, terminals=1, with_children=True)
    payload = dict(payloads_written()[0])
    digest = payload["terminal_sha256"]
    payload["campaign_key"] = "tampered-after-acceptance"   # no longer hashes to its digest
    accounting = accounting_with(tmp_path / "acc.db", [{
        "terminal_sha256": digest, "campaign_sha256": "c" * 64, "unit_id": "unit-0",
        "generation": 1, "status": "COMPLETED", "body": payload}])
    out = tmp_path / "r.json"
    code = reconcile.main(["--accounting", str(accounting), "--cube", cube,
                           "--schema", "main", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["counts"]["content_unverifiable"] == 1
    assert code != 0
    assert "digest" in json.dumps(report["content_unverifiable"]).lower()
