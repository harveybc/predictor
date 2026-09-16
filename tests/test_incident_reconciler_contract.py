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


def fixture_payload(digest: str) -> dict:
    """The payload the fixture cube's one terminal corresponds to."""
    return {
        "terminal_sha256": digest, "campaign_sha256": "c" * 64, "unit_id": "unit-0",
        "generation": 1, "status": "COMPLETED", "actor": "a", "project": "p",
        "classification": "NON_GOVERNING", "config_sha256": "e" * 64,
        "metrics": [{"metric": "wall_seconds", "split": "test", "horizon": 0, "unit": "s",
                     "value": 1.0, "std_dev": None, "min_value": None, "max_value": None}],
        "artifacts": [{"role": "log", "sha256": "a" * 64, "bytes": 10}],
        "verified_datasets": [{"delivery_id": "delivery-0", "lake_id": "l",
                               "resource_id": "r", "role": "input", "sha256": "b" * 64,
                               "bytes": 1, "state": "VERIFIED_TRANSFER"}],
    }


@pytest.fixture
def scene(tmp_path):
    cube = str(tmp_path / "cube.duckdb")
    migrate.build_fixture_cube(cube, terminals=1, with_children=True, with_contract=True)
    digest = hashlib.sha256(b"terminal-0").hexdigest()
    accounting = accounting_with(tmp_path / "accounting.db", [{
        "terminal_sha256": digest, "campaign_sha256": "c" * 64, "unit_id": "unit-0",
        "generation": 1, "status": "COMPLETED", "body": fixture_payload(digest)}])
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
    digest = hashlib.sha256(b"terminal-0").hexdigest()
    payload = fixture_payload(digest)
    payload.update(status="REFUSED", metrics=[], artifacts=[], verified_datasets=[])
    accounting = accounting_with(tmp_path / "acc.db", [{
        "terminal_sha256": digest, "campaign_sha256": "c" * 64, "unit_id": "unit-0",
        "generation": 1, "status": "COMPLETED", "body": payload}])
    con = duckdb.connect(cube)
    con.execute("UPDATE main.gov_terminal SET status = 'REFUSED'")
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
    digest = hashlib.sha256(b"terminal-0").hexdigest()
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
