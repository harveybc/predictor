"""C35 + C37 (order 2026-09-11): runs are history, health is the service.

C35 — `dim_campaign` froze result_class, design_sha256, code_identity
and run_id against `campaign_key`, and the loader refused any
difference. A campaign is a QUESTION; those four describe an
EXECUTION. So a second legitimate attempt — after a code revision, a
design revision, or simply a failure followed by a success — collided
with its own first attempt and went to dead-letter. The execution
identity now lives on `dim_campaign_run`, keyed by (campaign, run).

C37 — `healthy` was `failed == 0`. Since the outbox never deletes, one
permanent refusal made the alarm red forever, and it was the same red a
dead loader would show. Health now answers only the service question;
dead-letters are counted separately, adjudicated rather than deleted,
and may be requeued only after someone has ruled on them.

The database tests need PostgreSQL and run against a THROWAWAY
database, created and dropped by the test. The real cube is never
touched.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import outbox as ob                                # noqa: E402
from olap.campaign_envelope import (EnvelopeRefusal,          # noqa: E402
                                    SCHEMA, build_envelope,
                                    load_envelope)


# ============================================================ C37
def make_outbox(tmp_path: Path) -> Path:
    root = tmp_path / "outbox"
    ob.ensure_outbox(root)
    return root


def fail_one(root: Path, doc: dict, reason: str) -> str:
    entry = ob.emit(doc, kind="event", root=root)
    path = next(iter(ob.pending_entries(root)))
    ob.mark(path, ob.FAILED, reason=reason, root=root)
    return entry["outbox_entry"]


def test_a_dead_letter_does_not_make_the_service_unhealthy(tmp_path):
    root = make_outbox(tmp_path)
    fail_one(root, {"x": 1}, "REFUSED: synthetic")
    hb = ob.heartbeat(root)
    assert hb["healthy"] is True, (
        "a refusal on file is not an outage; it was the same red as a "
        "dead loader and that is what made the alarm useless")
    assert hb["dead_letters_unadjudicated"] == 1
    assert hb["attention_required"] is True


def test_a_stale_heartbeat_is_unhealthy(tmp_path):
    root = make_outbox(tmp_path)
    ob.heartbeat(root)
    fresh = ob.health(root)
    stale = ob.health(root, now=time.time() + 10_000)
    assert fresh["healthy"] is True and fresh["process_fresh"] is True
    assert stale["healthy"] is False and stale["process_fresh"] is False


def test_overdue_backlog_is_unhealthy_but_young_backlog_is_not(tmp_path):
    root = make_outbox(tmp_path)
    ob.emit({"x": 2}, kind="event", root=root)
    ob.heartbeat(root)
    assert ob.health(root)["healthy"] is True
    overdue = ob.health(root, now=time.time() + 5_000,
                        max_heartbeat_age_s=1e9)
    assert overdue["backlog_overdue"] is True
    assert overdue["healthy"] is False


def test_adjudication_never_deletes_the_evidence(tmp_path):
    root = make_outbox(tmp_path)
    name = fail_one(root, {"x": 3}, "REFUSED: synthetic")
    before = sorted(p.name for p in (root / ob.FAILED).iterdir())
    ob.adjudicate(name, disposition=ob.ADJUDICATED,
                  reason="examined; cause understood", root=root)
    after = sorted(p.name for p in (root / ob.FAILED).iterdir())
    assert set(before) < set(after), "the entry and reason must survive"
    dl = ob.dead_letters(root)
    assert len(dl) == 1 and dl[0]["state"] == ob.ADJUDICATED
    assert ob.counts(root)["failed"] == 1, (
        "the adjudication sidecar is a note about an envelope, not "
        "another envelope")


def test_adjudication_refuses_a_reasonless_disposition(tmp_path):
    root = make_outbox(tmp_path)
    name = fail_one(root, {"x": 4}, "REFUSED: synthetic")
    with pytest.raises(SystemExit, match="without a reason"):
        ob.adjudicate(name, disposition=ob.ADJUDICATED, reason="   ",
                      root=root)
    with pytest.raises(SystemExit, match="unknown dead-letter"):
        ob.adjudicate(name, disposition="GREEN_PLEASE", reason="x",
                      root=root)
    with pytest.raises(SystemExit, match="must name the entry"):
        ob.adjudicate(name, disposition=ob.SUPERSEDED, reason="x",
                      root=root)


def test_requeue_requires_an_adjudication(tmp_path):
    root = make_outbox(tmp_path)
    name = fail_one(root, {"x": 5}, "REFUSED: synthetic")
    with pytest.raises(SystemExit, match="nobody has ruled on"):
        ob.requeue(name, root=root)
    ob.adjudicate(name, disposition=ob.SUPERSEDED, reason="carried later",
                  superseded_by="envelope-other.json", root=root)
    with pytest.raises(SystemExit, match="not a disposition that permits"):
        ob.requeue(name, root=root)
    ob.adjudicate(name, disposition=ob.ADJUDICATED, reason="cause removed",
                  root=root)
    out = ob.requeue(name, root=root)
    assert out["requeued"] is True
    assert (root / ob.FAILED / name).is_file(), (
        "requeueing must not erase the fact that it failed")
    assert (root / ob.PENDING / name).is_file()


def test_requeue_is_idempotent(tmp_path):
    root = make_outbox(tmp_path)
    name = fail_one(root, {"x": 6}, "REFUSED: synthetic")
    ob.adjudicate(name, disposition=ob.ADJUDICATED, reason="ok", root=root)
    ob.requeue(name, root=root)
    again = ob.requeue(name, root=root)
    assert again["requeued"] is False
    assert len(ob.pending_entries(root)) == 1


def test_the_first_heartbeat_is_not_born_stale(tmp_path):
    """It reads its own beat, not the previous file that does not
    exist yet."""
    root = make_outbox(tmp_path)
    hb = ob.heartbeat(root)
    assert hb["process_fresh"] is True
    assert hb["heartbeat_age_seconds"] == 0.0


# ============================================================ C35
def _pg_env() -> dict | None:
    env = Path.home() / ".config/crispdm/olap-loader.env"
    if not env.is_file():
        return None
    out = dict(os.environ)
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    if not all(out.get(k) for k in ("PGUSER", "PGPASSWORD")):
        return None
    return out


PG = _pg_env()
requires_pg = pytest.mark.skipif(
    PG is None, reason="no PostgreSQL credentials available")


@pytest.fixture
def throwaway_engine():
    """A database created and dropped by this test. The real cube is
    never opened here."""
    from sqlalchemy import create_engine, text
    name = f"crispdm_test_{uuid.uuid4().hex[:12]}"
    admin_url = (f"postgresql+psycopg2://{PG['PGUSER']}:{PG['PGPASSWORD']}"
                 f"@{PG.get('PGHOST', 'localhost')}:"
                 f"{PG.get('PGPORT', '5432')}/postgres")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    url = admin_url.rsplit("/", 1)[0] + "/" + name
    engine = create_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def envelope(*, campaign="q::alpha", run="run-1", producer="predictor",
             result_class="DEVELOPMENT", code="c" * 64, design="d" * 64,
             state="COMPLETE", cell="cell-1", candidate="ann"):
    return build_envelope(
        campaign_key=campaign, producer=producer,
        result_class=result_class,
        identity={"run_id": run, "code_identity": code,
                  "design_sha256": design},
        data_consumed={"variables": [], "operators": [], "datasets": []},
        partitions={"exposure": "DEV", "splits": "train/test"},
        budget={"device": "cpu", "wall_seconds": 1.0,
                "cost_units": "wall_seconds"},
        terminal={"state": state, "adjudication": "NONE"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL"},
        units=[{"cell_key": cell, "candidate_key": candidate,
                "metric_name": "m", "metric_value": 1.0,
                "terminal_state": state}])


@requires_pg
def test_a_failed_run_and_a_later_complete_run_both_land(throwaway_engine):
    from sqlalchemy import text
    load_envelope(throwaway_engine, envelope(run="run-1", state="FAILED"))
    load_envelope(throwaway_engine, envelope(run="run-2", state="COMPLETE",
                                             cell="cell-2"))
    with throwaway_engine.connect() as c:
        runs = [dict(r) for r in c.execute(text(
            f"SELECT run_id, terminal_state FROM {SCHEMA}.dim_campaign_run "
            "WHERE campaign_key='q::alpha' ORDER BY run_id")).mappings()]
        campaigns = c.execute(text(
            f"SELECT count(*) FROM {SCHEMA}.dim_campaign")).scalar()
    assert [r["terminal_state"] for r in runs] == ["FAILED", "COMPLETE"]
    assert campaigns == 1, "one question, two attempts"


@requires_pg
def test_a_code_revision_is_a_new_run_not_a_conflict(throwaway_engine):
    from sqlalchemy import text
    load_envelope(throwaway_engine, envelope(run="r1", code="a" * 64))
    load_envelope(throwaway_engine,
                  envelope(run="r2", code="b" * 64, cell="cell-2"))
    with throwaway_engine.connect() as c:
        row = dict(c.execute(text(
            f"SELECT * FROM {SCHEMA}.v_campaign_attempts "
            "WHERE campaign_key='q::alpha'")).mappings().first())
    assert row["attempts_total"] == 2
    assert row["code_revisions"] == 2


@requires_pg
def test_a_design_revision_is_a_new_run(throwaway_engine):
    from sqlalchemy import text
    load_envelope(throwaway_engine, envelope(run="r1", design="1" * 64))
    load_envelope(throwaway_engine,
                  envelope(run="r2", design="2" * 64, cell="cell-2"))
    with throwaway_engine.connect() as c:
        row = dict(c.execute(text(
            f"SELECT * FROM {SCHEMA}.v_campaign_attempts "
            "WHERE campaign_key='q::alpha'")).mappings().first())
    assert row["design_revisions"] == 2


@requires_pg
def test_reusing_a_run_id_for_a_different_execution_still_refuses(
        throwaway_engine):
    """A run is ONE execution. New history needs a new run_id."""
    load_envelope(throwaway_engine, envelope(run="r1", code="a" * 64))
    with pytest.raises(EnvelopeRefusal, match="DIFFERENT identity"):
        load_envelope(throwaway_engine,
                      envelope(run="r1", code="b" * 64, cell="cell-2"))


@requires_pg
def test_a_foreign_producer_cannot_attach_to_a_campaign(throwaway_engine):
    load_envelope(throwaway_engine, envelope(run="r1"))
    with pytest.raises(EnvelopeRefusal, match="belongs to producer"):
        load_envelope(throwaway_engine,
                      envelope(run="r2", producer="someone-else"))


@requires_pg
def test_reloading_the_same_envelope_changes_nothing(throwaway_engine):
    from sqlalchemy import text
    doc = envelope(run="r1")
    first = load_envelope(throwaway_engine, doc)
    second = load_envelope(throwaway_engine, doc)
    with throwaway_engine.connect() as c:
        units = c.execute(text(
            f"SELECT count(*) FROM {SCHEMA}.fact_campaign_unit")).scalar()
        runs = c.execute(text(
            f"SELECT count(*) FROM {SCHEMA}.dim_campaign_run")).scalar()
    assert first["units"] == 1 and second["units"] == 0
    assert units == 1 and runs == 1


@requires_pg
def test_every_fact_row_reaches_a_run_dimension(throwaway_engine):
    from sqlalchemy import text
    load_envelope(throwaway_engine, envelope(run="r1"))
    load_envelope(throwaway_engine, envelope(run="r2", cell="cell-2"))
    with throwaway_engine.connect() as c:
        orphans = c.execute(text(f"""
            SELECT count(*) FROM {SCHEMA}.fact_campaign_unit f
             LEFT JOIN {SCHEMA}.dim_campaign_run d
                    ON d.campaign_key=f.campaign_key
                   AND d.run_id=f.run_id
             WHERE d.run_id IS NULL""")).scalar()
    assert orphans == 0


@requires_pg
def test_the_migration_conserves_every_count(throwaway_engine):
    from olap.migrate_campaign_run import migrate
    load_envelope(throwaway_engine, envelope(run="r1"))
    load_envelope(throwaway_engine, envelope(run="r2", cell="cell-2"))
    report = migrate(throwaway_engine, apply=True)
    assert report["conserved"] is True
    assert report["before"] == report["after"]
    assert report["orphan_runs"] == 0


@requires_pg
def test_the_migration_adopts_a_fact_left_without_a_run(throwaway_engine):
    """The exact gap a pre-C35 loader process leaves behind."""
    from sqlalchemy import text
    from olap.migrate_campaign_run import migrate
    load_envelope(throwaway_engine, envelope(run="r1"))
    with throwaway_engine.begin() as c:
        c.execute(text(
            f"UPDATE {SCHEMA}.fact_campaign_unit SET run_id = 'ghost'"))
        c.execute(text(f"DELETE FROM {SCHEMA}.dim_campaign_run"))
        c.execute(text(
            f"UPDATE {SCHEMA}.dim_campaign SET run_id = 'ghost'"))
    report = migrate(throwaway_engine, apply=True)
    assert report["orphan_runs"] == 0, (
        "re-running the migration must ADOPT the orphan, not ignore it")
    with throwaway_engine.connect() as c:
        assert c.execute(text(
            f"SELECT count(*) FROM {SCHEMA}.dim_campaign_run "
            "WHERE run_id='ghost'")).scalar() == 1
