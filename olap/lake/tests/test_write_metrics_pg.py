"""write_metrics on PostgreSQL. Runs only when DATA_GOV_TEST_PG_URL names a
throwaway database (sqlalchemy URL); the gov_* objects there are dropped
and recreated by the test."""

import os

import pytest

PG_URL = os.getenv("DATA_GOV_TEST_PG_URL")

pytestmark = pytest.mark.skipif(not PG_URL, reason="DATA_GOV_TEST_PG_URL not set")


def _pg_env(monkeypatch):
    from sqlalchemy.engine import make_url

    url = make_url(PG_URL)
    monkeypatch.setenv("PGHOST", url.host or "127.0.0.1")
    monkeypatch.setenv("PGPORT", str(url.port or 5432))
    monkeypatch.setenv("PGDATABASE", url.database or "")
    monkeypatch.setenv("PGUSER", url.username or "")
    monkeypatch.setenv("PGPASSWORD", url.password or "")
    monkeypatch.delenv("PGUSER_WRITE", raising=False)
    monkeypatch.delenv("PGPASSWORD_WRITE", raising=False)


def test_pg_ddl_store_already_stored_and_view(monkeypatch, make_report):
    from sqlalchemy import create_engine, text

    from query_plugins.sql_query import Plugin

    _pg_env(monkeypatch)
    admin = create_engine(PG_URL)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("DROP VIEW IF EXISTS public.gov_metric_current"))
        for table in ("gov_metric", "gov_dataset", "gov_report"):
            conn.execute(text(f"DROP TABLE IF EXISTS public.{table}"))

    q = Plugin()
    q.set_params(sqlite_path=None, schema="public", holdout_start=None)
    q.engine()
    assert q._schema_error is None
    # DDL is idempotent and additive: a second engine sees the objects
    q.set_params()
    q.engine()
    assert q._schema_error is None

    older = make_report(received_at="2026-09-13T00:00:00.000000+00:00")
    newer = make_report(
        received_at="2026-09-13T00:00:01.000000+00:00",
        metrics=[{"metric": "MAE", "value": 0.001, "split": "train", "horizon": 24}],
    )
    assert q.write_metrics(older) == {
        "stored": True, "already_stored": False, "lineage": "VERIFIED",
    }
    assert q.write_metrics(newer)["stored"]
    again = dict(older, lineage="UNVERIFIED")
    assert q.write_metrics(again) == {
        "stored": False, "already_stored": True, "lineage": "VERIFIED",
    }

    with admin.connect() as conn:
        n_reports = conn.execute(text("SELECT COUNT(*) FROM public.gov_report")).scalar()
        n_metrics = conn.execute(text("SELECT COUNT(*) FROM public.gov_metric")).scalar()
        dataset = conn.execute(
            text("SELECT lineage, event_id, source_sha256 FROM public.gov_dataset LIMIT 1")
        ).one()
        current = conn.execute(
            text("SELECT report_sha256, value FROM public.gov_metric_current")
        ).all()
    assert (n_reports, n_metrics) == (2, 2)
    assert tuple(dataset) == ("VERIFIED", 17, "2" * 64)
    assert current == [(newer["report_sha256"], 0.001)]
