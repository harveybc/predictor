"""C73 on THROWAWAY databases only: additive load, idempotence,
conflict refusal, and a view that names the evidence layer of every
historical row without touching it."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text  # noqa: E402
import load_terminal_verification_v3 as L  # noqa: E402

SQL = (REPO / "olap/c73_terminal_verification_layers.sql").read_text()


def dsn(db):
    return "postgresql+psycopg2://{u}:{p}@{h}:{P}/{d}".format(
        u=os.environ["PGUSER"], p=os.environ["PGPASSWORD"],
        h=os.environ.get("PGHOST", "localhost"),
        P=os.environ.get("PGPORT", "5432"), d=db)


@pytest.fixture()
def db():
    if "PGUSER" not in os.environ:
        pytest.skip("no PG credentials")
    name = "c73_throwaway_" + uuid.uuid4().hex[:12]
    admin = create_engine(dsn("postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    e = create_engine(dsn(name))
    with e.begin() as c:
        c.execute(text("""CREATE TABLE public.fact_variable_characterization (
            variable_id TEXT, partition_key TEXT, descriptor TEXT,
            value DOUBLE PRECISION, value_text TEXT, identifiable BOOLEAN,
            terminal_attempt TEXT, source_sha256 TEXT)"""))
        for stmt in SQL.split(";\n"):
            if stmt.strip() and not stmt.strip().startswith("--") or \
                    "CREATE" in stmt:
                c.execute(text(stmt))
        for vid, d, v in (("var_a", "mean", 2.5), ("var_a", "p99", 9.0),
                          ("var_b", "mean", 1.0)):
            c.execute(text("INSERT INTO public.fact_variable_characterization"
                           " VALUES (:v,'p',:d,:x,NULL,true,'att','s')"),
                      {"v": vid, "d": d, "x": v})
    try:
        yield e, dsn(name)
    finally:
        e.dispose()
        admin = create_engine(dsn("postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


INDEX = {"verification_sha256": "v" * 64, "v3_terminals_written": 1,
         "population_verdict": "TERMINAL_POPULATION_AND_SOURCE_BINDING_VERIFIED",
         "recomputation_verdict": "INDEPENDENT_RECOMPUTATION_NO_DIVERGENCE"}
REPORT = {"schema": "crispdm.lake_terminal_verification.v2",
          "population": {"divergence_count": 0},
          "recomputation": {"variables_by_layer": {"INDEPENDENTLY_RECOMPUTED": 1},
                            "published_attempt": "att"}}
BODY = {"variable_id": "var_a", "verification_sha256": "v" * 64,
        "producer_declared_outcome": "PRODUCER_DECLARED_MEASURED",
        "layer": "INDEPENDENTLY_RECOMPUTED", "recomputation": "AGREES",
        "source": {"sha256": "s" * 64}, "terminal_sha256": "t" * 64,
        "descriptors": {
            "mean": {"layer": "INDEPENDENTLY_RECOMPUTED", "state": "AGREES",
                     "specificity": "FULLY_SPECIFIED", "reason": None,
                     "published": 2.5, "recomputed": 2.5},
            "p99": {"layer": "SOURCE_BOUND",
                    "state": "NOT_INDEPENDENTLY_VERIFIABLE",
                    "specificity": "UNDERSPECIFIED",
                    "reason": "interpolation", "published": 9.0,
                    "recomputed": 9.0}}}


def layers(e):
    with e.connect() as c:
        return {(r.variable_id, r.descriptor): (r.evidence_layer,
                                                r.verification_state)
                for r in c.execute(text(
                    "SELECT * FROM public.v_characterization_evidence_layer"))}


def test_before_any_verification_every_row_is_producer_declared(db):
    e, _ = db
    assert set(v[0] for v in layers(e).values()) == {"PRODUCER_DECLARED"}


def test_a_load_names_each_rows_layer_and_leaves_history_alone(db):
    e, d = db
    w = L.load(d, INDEX, [BODY], REPORT, "2026-09-12T00:00:00Z")
    assert w == {"runs": 1, "variables": 1, "descriptors": 2}
    lay = layers(e)
    assert lay[("var_a", "mean")] == ("INDEPENDENTLY_RECOMPUTED", "AGREES")
    assert lay[("var_a", "p99")] == ("SOURCE_BOUND",
                                      "NOT_INDEPENDENTLY_VERIFIABLE")
    assert lay[("var_b", "mean")] == ("PRODUCER_DECLARED", "NO_VERIFICATION")
    with e.connect() as c:
        assert c.execute(text("SELECT count(*) FROM "
                              "public.fact_variable_characterization")
                         ).scalar() == 3


def test_a_second_load_writes_nothing(db):
    _, d = db
    L.load(d, INDEX, [BODY], REPORT, "2026-09-12T00:00:00Z")
    assert L.load(d, INDEX, [BODY], REPORT, "2026-09-12T00:00:00Z") == \
        {"runs": 0, "variables": 0, "descriptors": 0}


def test_a_conflicting_row_for_the_same_key_refuses(db):
    _, d = db
    L.load(d, INDEX, [BODY], REPORT, "2026-09-12T00:00:00Z")
    changed = json.loads(json.dumps(BODY))
    changed["descriptors"]["mean"]["state"] = "DIVERGES"
    changed["descriptors"]["mean"]["layer"] = "SOURCE_BOUND"
    with pytest.raises(L.LoadRefusal):
        L.load(d, INDEX, [changed], REPORT, "2026-09-12T00:00:00Z")


def test_a_run_reloaded_with_different_verdicts_refuses(db):
    _, d = db
    L.load(d, INDEX, [BODY], REPORT, "2026-09-12T00:00:00Z")
    idx = dict(INDEX, population_verdict="X")
    with pytest.raises(L.LoadRefusal):
        L.load(d, idx, [BODY], REPORT, "2026-09-12T00:00:00Z")


def test_two_runs_at_the_same_instant_declare_the_tie(db):
    e, d = db
    L.load(d, INDEX, [BODY], REPORT, "2026-09-12T00:00:00Z")
    idx2 = dict(INDEX, verification_sha256="w" * 64)
    body2 = dict(BODY, verification_sha256="w" * 64)
    L.load(d, idx2, [body2], REPORT, "2026-09-12T00:00:00Z")
    with e.connect() as c:
        states = {r[0] for r in c.execute(text(
            "SELECT currency_state FROM public.v_characterization_evidence_layer"))}
    assert states == {"AMBIGUOUS_TIE"}
