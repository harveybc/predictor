"""C92 on THROWAWAY databases only."""
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
import load_terminal_verification_v4 as L  # noqa: E402

SQL73 = (REPO / "olap/c73_terminal_verification_layers.sql").read_text()
SQL92 = (REPO / "olap/c92_terminal_verification_layers_v2.sql").read_text()


def dsn(db):
    return "postgresql+psycopg2://{u}:{p}@{h}:{P}/{d}".format(
        u=os.environ["PGUSER"], p=os.environ["PGPASSWORD"],
        h=os.environ.get("PGHOST", "localhost"), P=os.environ.get("PGPORT", "5432"), d=db)


def run_sql(c, sql):
    for stmt in sql.split(";\n"):
        body = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--")).strip()
        if body:
            c.execute(text(body))


@pytest.fixture()
def db():
    if "PGUSER" not in os.environ:
        pytest.skip("no PG credentials")
    name = "c92_throwaway_" + uuid.uuid4().hex[:12]
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
        run_sql(c, SQL73)
        run_sql(c, SQL92)
        for vid, d, v in (("var_num", "mean", 2.5), ("var_num", "p99", 9.0),
                          ("var_sent", "mean", -9.2e18), ("var_other", "mean", 1.0)):
            c.execute(text("INSERT INTO public.fact_variable_characterization VALUES "
                           "(:v,'p',:d,:x,NULL,true,'att','s')"), {"v": vid, "d": d, "x": v})
    try:
        yield e, dsn(name)
    finally:
        e.dispose()
        admin = create_engine(dsn("postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


IDX = {"verification_sha256": "v" * 64, "v4_written": 2}
REPORT = {"schema": "crispdm.lake_terminal_verification.v3",
          "census_identity": {"recomputed_canonical": "c" * 64},
          "population": {"verdict": "TERMINAL_POPULATION_AND_CANONICAL_CENSUS_VERIFIED", "divergence_count": 0},
          "layers": {"INDEPENDENTLY_RECOMPUTED": 1, "SEMANTICALLY_UNRESOLVED": 1},
          "semantic_sweep": {"by_state": {"NUMERIC_MEASURABLE": 1, "SEMANTIC_TYPE_UNRESOLVED": 1}}}
BODIES = [
    {"variable_id": "var_num", "verification_sha256": "v" * 64, "terminal_sha256": "t" * 64,
     "producer_declared_outcome": "PRODUCER_DECLARED_MEASURED", "layer": "INDEPENDENTLY_RECOMPUTED",
     "appearances": {"app": {"semantic": {"state": "NUMERIC_MEASURABLE", "arrow_type": "double"},
                             "recomputation": {"descriptors": {
                                 "mean": {"state": "AGREES", "specificity": "FULLY_SPECIFIED"},
                                 "p99": {"state": "NOT_INDEPENDENTLY_VERIFIABLE", "specificity": "UNDERSPECIFIED"}}}}}},
    {"variable_id": "var_sent", "verification_sha256": "v" * 64, "terminal_sha256": "u" * 64,
     "producer_declared_outcome": "PRODUCER_DECLARED_MEASURED", "layer": "SEMANTICALLY_UNRESOLVED",
     "published_numeric_descriptors_withdrawn": ["mean"],
     "appearances": {"app": {"semantic": {"state": "SEMANTIC_TYPE_UNRESOLVED", "arrow_type": "int64"}}}},
]


def layers(e):
    with e.connect() as c:
        return {(r.variable_id, r.descriptor): (r.evidence_layer, r.published_numeric_withdrawn)
                for r in c.execute(text("SELECT * FROM public.v_characterization_evidence_layer_v2"))}


def test_six_layer_view_names_each_row(db):
    e, d = db
    assert L.load(d, IDX, BODIES, REPORT, "2026-09-12T00:00:00Z") == {"runs": 1, "variables": 2, "descriptors": 2}
    lay = layers(e)
    assert lay[("var_num", "mean")] == ("INDEPENDENTLY_RECOMPUTED", False)
    assert lay[("var_num", "p99")] == ("PHYSICALLY_TYPED", False)
    assert lay[("var_sent", "mean")] == ("SEMANTICALLY_UNRESOLVED", True)
    assert lay[("var_other", "mean")][0] == "PRODUCER_DECLARED"
    with e.connect() as c:
        assert c.execute(text("SELECT count(*) FROM public.fact_variable_characterization")).scalar() == 4


def test_second_load_is_idempotent_and_conflicts_refuse(db):
    e, d = db
    L.load(d, IDX, BODIES, REPORT, "2026-09-12T00:00:00Z")
    assert L.load(d, IDX, BODIES, REPORT, "2026-09-12T00:00:00Z") == {"runs": 0, "variables": 0, "descriptors": 0}
    bad = json.loads(json.dumps(BODIES))
    bad[0]["layer"] = "DIVERGES"
    with pytest.raises(L.LoadRefusal):
        L.load(d, IDX, bad, REPORT, "2026-09-12T00:00:00Z")


def test_c73_tables_are_untouched(db):
    e, d = db
    L.load(d, IDX, BODIES, REPORT, "2026-09-12T00:00:00Z")
    with e.connect() as c:
        assert c.execute(text("SELECT count(*) FROM public.dim_terminal_verification")).scalar() == 0


def test_the_layer_check_rejects_an_unknown_layer(db):
    e, d = db
    bad = json.loads(json.dumps(BODIES))
    bad[0]["layer"] = "PROMOTED"
    with pytest.raises(Exception):
        L.load(d, IDX, bad, REPORT, "2026-09-12T00:00:00Z")
