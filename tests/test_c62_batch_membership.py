"""C62 battery: exact batch membership and an honest current view.

Every database test runs against a THROWAWAY database created and
dropped by the test. The populated cube is never written to, never
truncated and never restarted.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import load_batch_membership as L  # noqa: E402

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text  # noqa: E402

SCHEMA_SQL = (ROOT / "olap/c62_batch_membership.sql").read_text()


def admin_dsn(db: str) -> str:
    return ("postgresql+psycopg2://{u}:{p}@{h}:{P}/{d}".format(
        u=os.environ["PGUSER"], p=os.environ["PGPASSWORD"],
        h=os.environ.get("PGHOST", "localhost"),
        P=os.environ.get("PGPORT", "5432"), d=db))


@pytest.fixture()
def throwaway():
    if "PGUSER" not in os.environ:
        pytest.skip("no PG* credentials in the environment")
    name = "c62_throwaway_" + uuid.uuid4().hex[:12]
    admin = create_engine(admin_dsn("postgres"),
                          isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = create_engine(admin_dsn(name))
    try:
        with engine.begin() as c:
            c.execute(text("""
                CREATE TABLE public.fact_variable_characterization (
                  variable_id TEXT, partition_key TEXT,
                  bank_authority TEXT, descriptor TEXT,
                  value DOUBLE PRECISION, value_text TEXT,
                  descriptor_contract TEXT, identifiable BOOLEAN,
                  cost_seconds DOUBLE PRECISION,
                  observation_sha256 TEXT, measured_at TEXT,
                  loaded_at TIMESTAMPTZ DEFAULT now(),
                  source_id TEXT, source_sha256 TEXT,
                  window_sha256 TEXT, window_contract TEXT,
                  code_identity TEXT, protocol_version TEXT,
                  side TEXT, contract_role TEXT, units TEXT,
                  terminal_attempt TEXT, measurement_sha256 TEXT,
                  binding_state TEXT)"""))
            for stmt in SCHEMA_SQL.split(";\n"):
                if stmt.strip():
                    c.execute(text(stmt))
        yield engine, admin_dsn(name)
    finally:
        engine.dispose()
        admin = create_engine(admin_dsn("postgres"),
                              isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(
                f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def insert_obs(engine, vid, measured_at, obs_sha, value):
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO public.fact_variable_characterization "
            "(variable_id, partition_key, descriptor, value, "
            " measured_at, observation_sha256) "
            "VALUES (:v,'p','mean',:x,:m,:o)"),
            {"v": vid, "x": value, "m": measured_at, "o": obs_sha})


def current(engine):
    with engine.connect() as c:
        return [dict(r) for r in c.execute(text(
            "SELECT variable_id, value, currency_state, observations "
            "FROM public.v_variable_characterization_current_v2 "
            "ORDER BY variable_id")).mappings()]


# ---------------------------------------------------- the current view
def test_a_single_observation_is_unambiguous(throwaway):
    engine, _ = throwaway
    insert_obs(engine, "v1", "2026-09-12T00:00:00Z", "a" * 64, 1.0)
    (row,) = current(engine)
    assert row["currency_state"] == "UNAMBIGUOUS"
    assert row["observations"] == 1


def test_a_clean_succession_is_unambiguous_and_takes_the_newest(
        throwaway):
    engine, _ = throwaway
    insert_obs(engine, "v1", "2026-09-11T00:00:00Z", "a" * 64, 1.0)
    insert_obs(engine, "v1", "2026-09-12T00:00:00Z", "b" * 64, 2.0)
    (row,) = current(engine)
    assert row["value"] == 2.0
    assert row["currency_state"] == "UNAMBIGUOUS"
    assert row["observations"] == 2


def test_two_observations_at_the_same_instant_declare_the_tie(
        throwaway):
    """The defect the v1 view hides: it picks one deterministically and
    never says a coin was flipped."""
    engine, _ = throwaway
    insert_obs(engine, "v1", "2026-09-12T00:00:00Z", "a" * 64, 1.0)
    insert_obs(engine, "v1", "2026-09-12T00:00:00Z", "b" * 64, 2.0)
    (row,) = current(engine)
    assert row["currency_state"] == "AMBIGUOUS_TIE"
    assert row["value"] == 2.0, "still deterministic: digest order"


def test_an_older_tie_does_not_make_the_newest_ambiguous(throwaway):
    engine, _ = throwaway
    insert_obs(engine, "v1", "2026-09-10T00:00:00Z", "a" * 64, 1.0)
    insert_obs(engine, "v1", "2026-09-10T00:00:00Z", "b" * 64, 2.0)
    insert_obs(engine, "v1", "2026-09-12T00:00:00Z", "c" * 64, 3.0)
    (row,) = current(engine)
    assert row["currency_state"] == "UNAMBIGUOUS"
    assert row["value"] == 3.0


def test_the_v2_view_returns_exactly_one_row_per_grain(throwaway):
    engine, _ = throwaway
    for i in range(5):
        insert_obs(engine, "v1", f"2026-09-0{i+1}T00:00:00Z",
                   chr(97 + i) * 64, float(i))
    insert_obs(engine, "v2", "2026-09-01T00:00:00Z", "z" * 64, 9.0)
    rows = current(engine)
    assert len(rows) == 2
    assert rows[0]["observations"] == 5


# -------------------------------------------------- batch membership
BATCH = {"batch_00000": [
    {"variable_id": "var_a", "outcome": "MEASURED",
     "terminal_sha256": "t" * 64, "source_sha256": "s" * 64},
    {"variable_id": "var_b", "outcome": "NOT_IDENTIFIABLE",
     "terminal_sha256": "u" * 64, "source_sha256": None}]}


def test_membership_is_loaded_exactly_and_is_idempotent(throwaway):
    _, dsn = throwaway
    first = L.load(dsn, "att", BATCH, "2026-09-12T00:00:00Z",
                   dry_run=False)
    assert first["batches_written"] == 1
    assert [r["state"] for r in first["integrity"]] == ["EXACT"]
    second = L.load(dsn, "att", BATCH, "2026-09-12T00:00:00Z",
                    dry_run=False)
    assert second["batches_written"] == 0
    assert second["batches_unchanged"] == 1
    assert second["refused"] == []


def test_a_changed_membership_is_refused_not_merged(throwaway):
    _, dsn = throwaway
    L.load(dsn, "att", BATCH, "2026-09-12T00:00:00Z", dry_run=False)
    changed = {"batch_00000": BATCH["batch_00000"] + [
        {"variable_id": "var_c", "outcome": "MEASURED",
         "terminal_sha256": "w" * 64, "source_sha256": None}]}
    out = L.load(dsn, "att", changed, "2026-09-12T00:00:00Z",
                 dry_run=False)
    assert out["batches_written"] == 0
    assert len(out["refused"]) == 1
    assert "not silently merged" in out["refused"][0]["why"]
    assert out["integrity"][0]["variables_present"] == 2


def test_a_removed_variable_also_changes_the_membership_digest():
    full = BATCH["batch_00000"]
    assert L.membership_digest(full) != L.membership_digest(full[:1])


def test_the_membership_digest_ignores_row_order():
    full = BATCH["batch_00000"]
    assert L.membership_digest(full) == \
        L.membership_digest(list(reversed(full)))


def test_the_exact_members_of_a_batch_are_rows_not_a_count(throwaway):
    engine, dsn = throwaway
    L.load(dsn, "att", BATCH, "2026-09-12T00:00:00Z", dry_run=False)
    with engine.connect() as c:
        members = [r[0] for r in c.execute(text(
            "SELECT variable_id FROM public.bridge_batch_variable "
            "WHERE batch_key='batch_00000' ORDER BY variable_id"))]
    assert members == ["var_a", "var_b"]


def test_a_batch_declared_larger_than_its_rows_is_named(throwaway):
    engine, dsn = throwaway
    L.load(dsn, "att", BATCH, "2026-09-12T00:00:00Z", dry_run=False)
    with engine.begin() as c:
        c.execute(text(
            "DELETE FROM public.bridge_batch_variable "
            "WHERE variable_id='var_b'"))
        state = c.execute(text(
            "SELECT state, shortfall FROM "
            "public.v_batch_membership_integrity")).first()
    assert state[0] == "MEMBERSHIP_DIVERGES"
    assert state[1] == 1
