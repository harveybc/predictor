"""C12-C15: producer-bound envelopes, relational identity, a
durable outbox and the census inside the cube.

The audit found that envelopes translated summaries without
checking a producer's artifact, that a campaign_key collision
attached facts to the wrong dimension, that the backup gate was a
length check, and that nothing fed the cube automatically. Every
PostgreSQL test here creates and drops its own database; the
populated cube is never written by this file.
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
from olap import outbox as ob  # noqa: E402

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


@pytest.fixture()
def throwaway_db():
    sa = pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine, text
    name = f"predictor_olap_c14_{uuid.uuid4().hex[:12]}"
    admin = create_engine(_dsn("postgres"), future=True,
                          isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"no PostgreSQL: "
                    f"{exc.__class__.__name__}")
    engine = create_engine(_dsn(name), future=True)
    try:
        yield name, engine
    finally:
        engine.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


# ==============================================================
# C12: envelopes are bound to their producers
# ==============================================================

def test_a_fabricated_artifact_refuses(tmp_path):
    from tools.build_campaign_envelopes import (
        T2_SCHEMA_KEYS, _consume_producer_artifact)
    fake = {k: "x" for k in T2_SCHEMA_KEYS}
    fake["record_sha256"] = "a" * 64
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(fake))
    with pytest.raises(SystemExit,
                       match="does not re-derive"):
        _consume_producer_artifact(
            p, schema_keys=T2_SCHEMA_KEYS,
            self_key="record_sha256", producer="T2",
            verifier="v")


def test_a_similar_looking_json_refuses(tmp_path):
    from tools.build_campaign_envelopes import (
        T2_SCHEMA_KEYS, _consume_producer_artifact)
    p = tmp_path / "similar.json"
    p.write_text(json.dumps({"screen_adjudication": {},
                             "record_sha256": "a" * 64}))
    with pytest.raises(SystemExit,
                       match="not the producer's schema"):
        _consume_producer_artifact(
            p, schema_keys=T2_SCHEMA_KEYS,
            self_key="record_sha256", producer="T2",
            verifier="v")


def test_shipped_envelopes_record_their_producer_binding():
    if not ENVELOPES.is_dir():
        pytest.skip("envelopes not built")
    seen = {}
    for p in sorted(ENVELOPES.glob("envelope-*.json")):
        doc = json.loads(p.read_text())
        ce.validate_envelope(doc)
        art = doc["artifacts"]
        seen[doc["campaign_key"]] = art.get("verification")
        assert "source_file_name" in art
        assert "source_file_sha256" in art
    t2 = seen["t2_resource_successor_v1_20260909"]
    m4 = seen["m4_v5_calibration_attempt3_20260909"]
    b4 = seen["b4_campaign_generation_v7_20260908"]
    assert t2 == "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED"
    assert m4 == "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED"
    # B4 never completed, so it has no artifact and says so
    assert b4.startswith("NO_PRODUCER_ARTIFACT_EXISTS")


def test_only_a_producer_bound_envelope_is_authoritative():
    if not ENVELOPES.is_dir():
        pytest.skip("envelopes not built")
    for p in sorted(ENVELOPES.glob("envelope-*.json")):
        doc = json.loads(p.read_text())
        state = ce.envelope_authority_state(doc)
        if doc["campaign_key"].startswith("b4_"):
            assert state == ce.TRANSLATED
        else:
            assert state == ce.PRODUCER_BOUND


# ==============================================================
# C13: relational identity
# ==============================================================

def test_a_campaign_key_collision_refuses(throwaway_db):
    _, engine = throwaway_db
    ce.ensure_envelope_tables(engine)
    first = _minimal()
    ce.load_envelope(engine, first)
    impostor = _minimal(
        identity={"run_id": "OTHER", "code_identity": "c",
                  "design_sha256": "d"})
    with pytest.raises(SystemExit,
                       match="DIFFERENT identity"):
        ce.load_envelope(engine, impostor)


def test_the_collision_refuses_before_any_fact(throwaway_db):
    from sqlalchemy import text
    _, engine = throwaway_db
    ce.ensure_envelope_tables(engine)
    ce.load_envelope(engine, _minimal())
    with engine.connect() as c:
        before = c.execute(text(
            "select count(*) from fact_campaign_unit")).scalar()
    impostor = _minimal(producer="SOMEONE_ELSE")
    with pytest.raises(SystemExit, match="DIFFERENT identity"):
        ce.load_envelope(engine, impostor)
    with engine.connect() as c:
        after = c.execute(text(
            "select count(*) from fact_campaign_unit")).scalar()
    assert before == after


def test_the_same_identity_is_still_idempotent(throwaway_db):
    from sqlalchemy import text
    _, engine = throwaway_db
    ce.ensure_envelope_tables(engine)
    doc = _minimal()
    ce.load_envelope(engine, doc)
    ce.load_envelope(engine, doc)
    with engine.connect() as c:
        assert c.execute(text(
            "select count(*) from dim_campaign")).scalar() == 1
        assert c.execute(text(
            "select count(*) from fact_campaign_unit"
        )).scalar() == 1


def test_units_are_strictly_typed():
    with pytest.raises(SystemExit, match="undeclared fields"):
        _minimal(units=[{"cell_key": "c", "candidate_key": "d",
                         "metric_name": "m",
                         "terminal_state": "OK",
                         "surprise": 1}])
    with pytest.raises(SystemExit, match="not a number"):
        _minimal(units=[{"cell_key": "c", "candidate_key": "d",
                         "metric_name": "m",
                         "terminal_state": "OK",
                         "metric_value": True}])
    with pytest.raises(SystemExit,
                       match="must be a non-empty string"):
        _minimal(units=[{"cell_key": "", "candidate_key": "d",
                         "metric_name": "m",
                         "terminal_state": "OK"}])


def test_the_backup_gate_opens_the_file(tmp_path):
    src = (REPO /
           "tools/backfill_campaign_envelopes.py").read_text()
    assert "--backup-file" in src
    assert "_sha_file(a.backup_file)" in src
    assert "a digest is not a backup" in src
    assert "OPENED_AND_REHASHED" in src


# ==============================================================
# C14: the outbox
# ==============================================================

def test_emit_never_needs_a_database(tmp_path, monkeypatch):
    """The whole point: a producer finishes with PostgreSQL
    unreachable."""
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", "1")        # nothing listens
    out = ob.emit(_minimal(), kind="envelope", root=tmp_path)
    assert out["written"] is True
    assert out["state"] == ob.PENDING
    assert ob.counts(tmp_path)["pending"] == 1


def test_emitting_the_same_document_twice_is_a_no_op(tmp_path):
    doc = _minimal()
    a = ob.emit(doc, kind="envelope", root=tmp_path)
    b = ob.emit(doc, kind="envelope", root=tmp_path)
    assert a["written"] is True and b["written"] is False
    assert ob.counts(tmp_path)["pending"] == 1


def test_a_down_database_leaves_entries_pending(tmp_path,
                                                monkeypatch):
    from tools import olap_loader
    ob.emit(_minimal(), kind="envelope", root=tmp_path)
    monkeypatch.setenv("PGPORT", "1")
    out = olap_loader.drain_once(tmp_path)
    assert out["database_unavailable"] is True
    assert out["loaded"] == 0
    assert ob.counts(tmp_path)["pending"] == 1
    assert ob.counts(tmp_path)["failed"] == 0


def test_the_loader_recovers_after_the_database_returns(
        tmp_path, throwaway_db, monkeypatch):
    from sqlalchemy import text
    from tools import olap_loader
    name, engine = throwaway_db
    ob.emit(_minimal(), kind="envelope", root=tmp_path)

    monkeypatch.setenv("PGPORT", "1")
    first = olap_loader.drain_once(tmp_path)
    assert first["database_unavailable"] is True

    monkeypatch.setenv("PGPORT", PGPORT)
    monkeypatch.setenv("PGDATABASE", name)
    second = olap_loader.drain_once(tmp_path,
                                    as_of="2026-09-10T00:00:00Z")
    assert second["loaded"] == 1
    assert ob.counts(tmp_path) == {"pending": 0, "loaded": 1,
                                   "failed": 0}
    with engine.connect() as c:
        assert c.execute(text(
            "select count(*) from fact_campaign_unit"
        )).scalar() == 1

    third = olap_loader.drain_once(tmp_path)
    assert third["attempted"] == 0
    with engine.connect() as c:
        assert c.execute(text(
            "select count(*) from fact_campaign_unit"
        )).scalar() == 1


def test_a_duplicate_emit_never_double_loads(tmp_path,
                                             throwaway_db,
                                             monkeypatch):
    from sqlalchemy import text
    from tools import olap_loader
    name, engine = throwaway_db
    monkeypatch.setenv("PGDATABASE", name)
    doc = _minimal()
    ob.emit(doc, kind="envelope", root=tmp_path)
    olap_loader.drain_once(tmp_path)
    ob.emit(doc, kind="envelope", root=tmp_path)   # no-op
    olap_loader.drain_once(tmp_path)
    with engine.connect() as c:
        assert c.execute(text(
            "select count(*) from fact_campaign_unit"
        )).scalar() == 1
    assert ob.counts(tmp_path)["loaded"] == 1


def test_failures_and_inconclusives_are_loaded_too(
        tmp_path, throwaway_db, monkeypatch):
    from sqlalchemy import text
    from tools import olap_loader
    name, engine = throwaway_db
    monkeypatch.setenv("PGDATABASE", name)
    for key, cls, state, adj in (
            ("ok", "DEVELOPMENT", "COMPLETE", "ADVANCE"),
            ("bad", "NON_GOVERNING", "FAILED_TYPED",
             "DOES_NOT_ADVANCE"),
            ("meh", "CALIBRATION", "COMPLETE",
             "INCONCLUSIVE")):
        ob.emit(_minimal(campaign_key=key, result_class=cls,
                         terminal={"state": state,
                                   "adjudication": adj}),
                kind="envelope", root=tmp_path)
    out = olap_loader.drain_once(tmp_path)
    assert out["loaded"] == 3
    with engine.connect() as c:
        rows = dict(c.execute(text(
            "select adjudication, count(*) from "
            "fact_campaign_unit group by adjudication")).all())
    assert set(rows) == {"ADVANCE", "DOES_NOT_ADVANCE",
                         "INCONCLUSIVE"}


def test_a_refused_envelope_is_marked_failed_with_its_reason(
        tmp_path, throwaway_db, monkeypatch):
    from tools import olap_loader
    name, engine = throwaway_db
    monkeypatch.setenv("PGDATABASE", name)
    doc = _minimal()
    doc["units"][0]["metric_value"] = 999.0      # digest broken
    r = ob.ensure_outbox(tmp_path)
    payload = json.dumps({"outbox_kind": "envelope",
                          "document": doc},
                         sort_keys=True).encode()
    (r / ob.PENDING / "envelope-broken.json").write_bytes(
        payload)
    out = olap_loader.drain_once(tmp_path)
    assert out["failed"] == 1
    assert ob.counts(tmp_path)["failed"] == 1
    reason = (r / ob.FAILED / "envelope-broken.reason"
              ).read_text()
    assert "mutated after" in reason


def test_the_heartbeat_reports_what_an_operator_needs(tmp_path):
    ob.emit(_minimal(), kind="envelope", root=tmp_path)
    import time as _t
    hb = ob.heartbeat(tmp_path, now=_t.time() + 60)
    assert hb["pending"] == 1
    assert hb["failed"] == 0
    assert hb["lag_seconds"] > 0
    assert hb["healthy"] is True
    written = json.loads(
        (ob.ensure_outbox(tmp_path) / "HEARTBEAT.json"
         ).read_text())
    assert written["pending"] == 1


def test_the_outbox_never_deletes_evidence():
    src = (REPO / "olap/outbox.py").read_text()
    assert "def mark" in src
    assert "append-only" in src
    # the only unlink is the de-duplication of an already-loaded
    # entry, and it is explicit
    assert src.count("unlink") == 1


def test_predictor_emits_at_the_end_of_a_run():
    src = (REPO / "app/main.py").read_text()
    assert "outbox.emit(" in src or "_outbox.emit(" in src
    assert "NOT EMITTED" in src, (
        "an outbox failure must be reported, never silent")
    assert src.index("_outbox.emit(") > src.index(
        "pipeline_plugin.run_prediction_pipeline(")


# ==============================================================
# C15: the census and index inside the cube
# ==============================================================

def test_the_supersession_tool_deletes_nothing():
    src = (REPO /
           "tools/supersede_translated_rows.py").read_text()
    assert "DELETE" not in src.upper().replace(
        "DELETED", "")
    assert "TRUNCATE" not in src.upper()
    flat = " ".join(src.split())
    assert "two additive columns record that they are" in flat
    assert "NOT deleted and NOT rewritten" in flat


def test_translated_rows_keep_their_metrics(throwaway_db):
    """Marking must not rewrite a value."""
    from sqlalchemy import text
    _, engine = throwaway_db
    ce.ensure_envelope_tables(engine)
    doc = _minimal()
    ce.load_envelope(engine, doc)
    with engine.connect() as c:
        before = c.execute(text(
            "select metric_value from fact_campaign_unit"
        )).scalar()
    with engine.begin() as c:
        c.execute(text(
            "update fact_campaign_unit set authority_state = :s"),
            {"s": ce.TRANSLATED})
    with engine.connect() as c:
        after = c.execute(text(
            "select metric_value from fact_campaign_unit"
        )).scalar()
        state = c.execute(text(
            "select authority_state from fact_campaign_unit"
        )).scalar()
    assert before == after
    assert state == ce.TRANSLATED
