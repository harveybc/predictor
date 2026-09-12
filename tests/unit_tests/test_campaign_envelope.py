"""P6: the campaign envelope, tested against a real throwaway
database.

The order's rules are the tests: nothing invented, result classes
never mixed, identity preserved, idempotent ingestion, a mutated
artifact rejected, and counts compared before and after. The
populated cube is never touched — every test that needs
PostgreSQL creates and drops its own database.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import campaign_envelope as ce  # noqa: E402

ENVELOPES = REPO / "examples/research/envelopes"

PGHOST = os.getenv("PGHOST", "127.0.0.1")
PGPORT = os.getenv("PGPORT", "5432")
PGUSER = os.getenv("PGUSER", "metabase")
PGPASSWORD = os.getenv("PGPASSWORD", "metabase_pass")


def _dsn(db):
    return (f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:"
            f"{PGPORT}/{db}")


def _minimal(**over):
    doc = dict(
        campaign_key="c1", producer="p",
        result_class="DEVELOPMENT",
        identity={"run_id": "r", "code_identity": "c",
                  "design_sha256": "d"},
        data_consumed={"datasets": [], "variables": [],
                       "operators": []},
        partitions={"exposure": "dev", "splits": "s"},
        budget={"device": "cpu", "wall_seconds": 1.0,
                "cost_units": "seconds"},
        terminal={"state": "COMPLETE",
                  "adjudication": "NONE"},
        artifacts={"a": "b"},
        units=[{"cell_key": "cell", "candidate_key": "cand",
                "metric_name": "m", "metric_value": 1.0,
                "terminal_state": "OK"}])
    doc.update(over)
    return ce.build_envelope(**doc)


# ==============================================================
# nothing is invented
# ==============================================================

def test_a_null_anywhere_refuses():
    with pytest.raises(SystemExit, match="never null"):
        ce.build_envelope(
            campaign_key="c", producer="p",
            result_class="DEVELOPMENT",
            identity={"run_id": None, "code_identity": "c",
                      "design_sha256": "d"},
            data_consumed={}, partitions={"exposure": "e",
                                          "splits": "s"},
            budget={"device": "cpu", "wall_seconds": 1.0,
                    "cost_units": "s"},
            terminal={"state": "X", "adjudication": "Y"},
            artifacts={})


def test_a_missing_field_refuses_rather_than_defaulting():
    for group, field in (("identity", "design_sha256"),
                         ("partitions", "exposure"),
                         ("budget", "device"),
                         ("terminal", "adjudication")):
        doc = _minimal()
        del doc[group][field]
        doc["envelope_sha256"] = ce._sha(doc, "envelope_sha256")
        with pytest.raises(SystemExit, match="is missing"):
            ce.validate_envelope(doc)


def test_unknown_is_written_unavailable_and_accepted():
    doc = _minimal(budget={"device": ce.UNAVAILABLE,
                           "wall_seconds": ce.UNAVAILABLE,
                           "cost_units": ce.UNAVAILABLE})
    ce.validate_envelope(doc)
    assert doc["budget"]["device"] == "UNAVAILABLE"


# ==============================================================
# result classes never mix
# ==============================================================

def test_unknown_result_class_refuses():
    with pytest.raises(SystemExit, match="unknown result class"):
        _minimal(result_class="PROBABLY_FINE")


def test_the_five_classes_are_distinct_and_complete():
    assert ce.RESULT_CLASSES == (
        "NON_GOVERNING", "MECHANICAL", "DEVELOPMENT",
        "CALIBRATION", "CONFIRMATION")


# ==============================================================
# identity survives
# ==============================================================

def test_mutating_an_envelope_refuses():
    doc = _minimal()
    doc["terminal"]["adjudication"] = "ADVANCE"
    with pytest.raises(SystemExit, match="mutated after"):
        ce.validate_envelope(doc)


def test_committed_envelopes_verify_and_keep_their_class():
    if not ENVELOPES.is_dir():
        pytest.skip("envelopes not built")
    seen = {}
    for p in sorted(ENVELOPES.glob("envelope-*.json")):
        doc = json.loads(p.read_text())
        ce.validate_envelope(doc)
        seen[doc["campaign_key"]] = doc["result_class"]
    assert seen, "no envelopes committed"
    # the three producers the order names for backfill, each with
    # the class its own evidence supports
    assert seen.get("t2_resource_successor_v1_20260909") == \
        "CONFIRMATION"
    assert seen.get("m4_v5_calibration_attempt3_20260909") == \
        "CALIBRATION"
    assert seen.get("b4_campaign_generation_v7_20260908") == \
        "NON_GOVERNING"


def test_quarantined_campaign_invents_no_metric():
    if not ENVELOPES.is_dir():
        pytest.skip("envelopes not built")
    p = ENVELOPES / ("envelope-b4_campaign_generation_v7_"
                     "20260908.json")
    if not p.is_file():
        pytest.skip("b4 envelope not built")
    doc = json.loads(p.read_text())
    assert doc["units"] == []
    assert doc["terminal"]["state"] == \
        "QUARANTINED_RUNTIME_STALL"
    assert doc["budget"]["wall_seconds"] == ce.UNAVAILABLE


# ==============================================================
# ingestion: throwaway database only
# ==============================================================

@pytest.fixture()
def throwaway_db():
    sa = pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, text
    name = f"predictor_olap_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(_dsn("postgres"), future=True,
                          isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"no PostgreSQL available: "
                    f"{exc.__class__.__name__}")
    engine = create_engine(_dsn(name), future=True)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def _counts(engine):
    from sqlalchemy import text
    out = {}
    with engine.connect() as c:
        for t in ("dim_campaign", "fact_campaign_unit",
                  "fact_campaign_consumption"):
            out[t] = c.execute(
                text(f"select count(*) from {t}")).scalar()
    return out


def test_ddl_is_additive_and_repeatable(throwaway_db):
    ce.ensure_envelope_tables(throwaway_db)
    ce.ensure_envelope_tables(throwaway_db)
    assert _counts(throwaway_db) == {
        "dim_campaign": 0, "fact_campaign_unit": 0,
        "fact_campaign_consumption": 0}


def test_ingestion_is_idempotent(throwaway_db):
    doc = _minimal(data_consumed={
        "datasets": [{"id": "d1", "digest": "x",
                      "eligibility_state": "DISCOVERED"}],
        "variables": [{"id": "v1", "digest": "y",
                       "eligibility_state": "DISCOVERED"}],
        "operators": []})
    ce.ensure_envelope_tables(throwaway_db)
    before = _counts(throwaway_db)
    first = ce.load_envelope(throwaway_db, doc)
    after_one = _counts(throwaway_db)
    second = ce.load_envelope(throwaway_db, doc)
    after_two = _counts(throwaway_db)
    assert before["fact_campaign_unit"] == 0
    assert after_one == after_two, (
        "a second load changed the cube")
    assert first["units"] == 1 and second["units"] == 0
    assert after_one["fact_campaign_consumption"] == 2


def test_a_mutated_artifact_is_rejected_before_any_write(
        throwaway_db):
    doc = _minimal()
    ce.ensure_envelope_tables(throwaway_db)
    before = _counts(throwaway_db)
    doc["units"][0]["metric_value"] = 999.0
    with pytest.raises(SystemExit, match="mutated after"):
        ce.load_envelope(throwaway_db, doc)
    assert _counts(throwaway_db) == before, (
        "the refusal came after a write")


def test_classes_stay_separable_in_the_cube(throwaway_db):
    from sqlalchemy import text
    ce.ensure_envelope_tables(throwaway_db)
    for cls in ("CONFIRMATION", "CALIBRATION", "DEVELOPMENT"):
        ce.load_envelope(throwaway_db, _minimal(
            campaign_key=f"c_{cls}", result_class=cls))
    with throwaway_db.connect() as c:
        rows = dict(c.execute(text(
            "select result_class, count(*) from "
            "fact_campaign_unit group by result_class")).all())
    assert rows == {"CONFIRMATION": 1, "CALIBRATION": 1,
                    "DEVELOPMENT": 1}
    with throwaway_db.connect() as c:
        conf = c.execute(text(
            "select count(*) from fact_campaign_unit "
            "where result_class = 'CONFIRMATION'")).scalar()
    assert conf == 1


def test_real_envelopes_load_into_a_throwaway_cube(throwaway_db):
    if not ENVELOPES.is_dir():
        pytest.skip("envelopes not built")
    ce.ensure_envelope_tables(throwaway_db)
    before = _counts(throwaway_db)
    total_units = 0
    for p in sorted(ENVELOPES.glob("envelope-*.json")):
        doc = json.loads(p.read_text())
        counts = ce.load_envelope(throwaway_db, doc)
        total_units += counts["units"]
        rec = ce.ingestion_receipt(doc, loaded_counts=counts,
                                   as_of="2026-09-10T00:00:00Z")
        assert rec["envelope_sha256"] == doc["envelope_sha256"]
        assert "grants_nothing" in rec
    after = _counts(throwaway_db)
    assert after["dim_campaign"] == 3
    assert after["fact_campaign_unit"] == total_units > 0
    assert before["fact_campaign_unit"] == 0
    # reloading everything changes nothing
    for p in sorted(ENVELOPES.glob("envelope-*.json")):
        ce.load_envelope(throwaway_db,
                         json.loads(p.read_text()))
    assert _counts(throwaway_db) == after


def test_module_has_no_delete_path():
    src = (REPO / "olap/campaign_envelope.py").read_text().upper()
    for forbidden in ("DROP TABLE", "TRUNCATE", "DELETE FROM"):
        assert forbidden not in src, (
            f"the envelope module contains {forbidden}")
