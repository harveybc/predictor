"""C139 on THROWAWAY databases only: additive, idempotent, strict, and no
eligibility grant can be stored."""
from __future__ import annotations

import importlib.util
import os
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("load_data_foundation", REPO / "tools/load_data_foundation.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
SHA = "c" * 64


def test_sql_file_is_the_generated_ddl():
    import re
    assert L.SQL_FILE.read_text() == L.ddl()
    # Statements only: the header comment may say what the schema never does.
    statements = "\n".join(ln for ln in L.ddl().splitlines() if not ln.strip().startswith("--"))
    assert not re.search(r"\b(ALTER|DROP|DELETE|TRUNCATE|UPDATE)\b", statements, re.IGNORECASE)
    assert statements.count("CREATE TABLE IF NOT EXISTS") == len(L.TABLES)


def profile_row(**over):
    r = {"run_id": "run-1", "dataset_id": "d", "content_sha256": SHA, "partition": "train",
         "metric": "acf_lag_1", "estimator": "acf", "estimator_params": {"lag": 1}, "value": 0.7,
         "value_text": None, "status": "COMPLETED", "reason": "", "code_sha256": SHA, "cpu_seconds": 0.01,
         "variable_id": "v"}
    r.update(over)
    return r


@pytest.mark.parametrize("over, needle", [
    ({"status": "COMPLETED", "value": None}, "needs a value"),
    ({"status": "FAILED", "value": 1.0}, "only a COMPLETED"),
    ({"status": "FAILED", "value": None, "reason": ""}, "needs a reason"),
    ({"status": "DONE"}, "not in"),
    ({"value": float("nan")}, "finite number"),
    ({"value": True}, "finite number"),
    ({"value_text": "PUBLICLY_ELIGIBLE", "value": None}, "never carries"),
    ({"code_sha256": "x"}, "sha256"),
    ({"extra": 1}, "keys differ"),
])
def test_row_validation_refuses(over, needle):
    assert any(needle in p for p in L.validate_row("df_fact_variable_profile", profile_row(**over)))


def test_lab_decisions_are_only_laboratory_states_and_unreviewed():
    row = {"run_id": "r", "operator_kind": "ewma", "operator_params": {"alpha": 0.3}, "spec_sha256": SHA,
           "regime": {"family": "sine"}, "decision": "LAB_CALIBRATED", "rule_sha256": SHA, "evidence": {},
           "failure_regions": [], "externally_reviewed": False, "code_sha256": SHA}
    assert L.validate_row("df_fact_lab_decision", row) == []
    assert L.validate_row("df_fact_lab_decision", dict(row, decision="PUBLICLY_ELIGIBLE"))
    assert L.validate_row("df_fact_lab_decision", dict(row, externally_reviewed=True))


def test_metric_row_adapter_maps_text_values():
    module_row = {"dataset_id": "d", "partition": "train", "metric": "group", "value": "GROUP_NOT_IDENTIFIED",
                  "estimator": {"name": "hclust", "params": {"k": 2}, "assumptions": {}},
                  "status": "COMPLETED", "reason": "", "code_sha256": SHA, "cpu_seconds": 0.1}
    r = L.metric_row(module_row, run_id="r", content_sha256=SHA, group_id="g", members=["a", "b"])
    assert r["value"] is None and r["value_text"] == "GROUP_NOT_IDENTIFIED"
    assert L.validate_row("df_fact_group_relation", r) == []


# ------------------------------------------------------------ throwaway DB
def dsn(db):
    return "postgresql+psycopg2://{u}:{p}@{h}:{P}/{d}".format(
        u=os.environ["PGUSER"], p=os.environ["PGPASSWORD"],
        h=os.environ.get("PGHOST", "localhost"), P=os.environ.get("PGPORT", "5432"), d=db)


@pytest.fixture()
def engine():
    if "PGUSER" not in os.environ or "PGPASSWORD" not in os.environ:
        pytest.skip("no PG credentials")
    sa = pytest.importorskip("sqlalchemy")
    name = "c139_throwaway_" + uuid.uuid4().hex[:12]
    admin = sa.create_engine(dsn("postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(sa.text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    e = sa.create_engine(dsn(name))
    try:
        yield e
    finally:
        e.dispose()
        admin = sa.create_engine(dsn("postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_schema_is_additive_and_idempotent(engine):
    from sqlalchemy import text
    with engine.begin() as c:
        c.execute(text("CREATE TABLE public.fact_variable_characterization (variable_id TEXT, value DOUBLE PRECISION)"))
        c.execute(text("INSERT INTO public.fact_variable_characterization VALUES ('old', 1.5)"))
    L.ensure_schema(engine)
    L.ensure_schema(engine)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*), sum(value) FROM public.fact_variable_characterization")).one() == (1, 1.5)
        names = {r[0] for r in c.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))}
    assert set(L.TABLES) <= names


def test_load_is_idempotent_keeps_every_outcome_and_writes_a_receipt(engine):
    from sqlalchemy import text
    L.ensure_schema(engine)
    rows = [profile_row(), profile_row(metric="adf_p", status="INCONCLUSIVE", value=None, reason="too short"),
            profile_row(metric="kpss_p", status="FAILED", value=None, reason="solver error"),
            profile_row(metric="bad", status="COMPLETED", value=None)]
    first = L.load(engine, "df_fact_variable_profile", rows, "run-1")
    second = L.load(engine, "df_fact_variable_profile", rows, "run-1")
    assert (first["rows_inserted"], first["rows_refused"]) == (3, 1)
    assert (second["rows_inserted"], second["rows_already_present"]) == (0, 3)
    with engine.connect() as c:
        states = sorted(r[0] for r in c.execute(text("SELECT status FROM public.df_fact_variable_profile")))
        receipts = c.execute(text("SELECT count(*) FROM public.df_fact_load_receipt")).scalar()
    assert states == ["COMPLETED", "FAILED", "INCONCLUSIVE"] and receipts == 2


def test_the_database_itself_refuses_an_eligibility_grant(engine):
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError
    L.ensure_schema(engine)
    with pytest.raises(IntegrityError):
        with engine.begin() as c:
            c.execute(text(
                "INSERT INTO public.df_fact_lab_decision (row_sha256, run_id, operator_kind, operator_params, "
                "spec_sha256, regime, decision, rule_sha256, evidence, failure_regions, externally_reviewed, code_sha256) "
                "VALUES ('x', 'r', 'ewma', '{}', 's', '{}', 'PUBLICLY_ELIGIBLE', 's', '{}', '[]', false, 's')"))
