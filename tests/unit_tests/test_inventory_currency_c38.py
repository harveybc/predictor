"""C38 (order 2026-09-11): current means latest OBSERVED, not latest LOADED.

The `v_*_current` views ordered by `loaded_at DESC`. Re-importing an
old census today therefore made the old observation current, silently
superseding a newer one. The existing test only ever loaded
old-then-new, so it never saw the defect — which is the more important
lesson: a test that exercises one direction of an ordering has tested
nothing about the ordering.

Four cases, per the order: old->new, new->old, a repetition, and two
incomparable branches. The last must not be settled silently; it is
reported as AMBIGUOUS_TIE.

Runs against a THROWAWAY database, created and dropped here.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import inventory_rows as inv                       # noqa: E402


def _pg_env():
    env = Path.home() / ".config/crispdm/olap-loader.env"
    if not env.is_file():
        return None
    out = dict(os.environ)
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out if out.get("PGUSER") and out.get("PGPASSWORD") else None


PG = _pg_env()
requires_pg = pytest.mark.skipif(PG is None,
                                 reason="no PostgreSQL credentials")


@pytest.fixture
def engine():
    from sqlalchemy import create_engine, text
    name = f"crispdm_c38_{uuid.uuid4().hex[:12]}"
    admin_url = (f"postgresql+psycopg2://{PG['PGUSER']}:{PG['PGPASSWORD']}"
                 f"@{PG.get('PGHOST', 'localhost')}:"
                 f"{PG.get('PGPORT', '5432')}/postgres")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    eng = create_engine(admin_url.rsplit("/", 1)[0] + "/" + name)
    inv.ensure_inventory_tables(eng)
    try:
        yield eng
    finally:
        eng.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def put(engine, *, variable_id="v1", observed_at, digest,
        supersedes=None, unit="usd"):
    """One observation of one conceptual variable."""
    from sqlalchemy import text
    obs = inv.observation_sha256({"digest": digest, "unit": unit})
    with engine.begin() as c:
        c.execute(text(f"""
            INSERT INTO {inv.SCHEMA}.dim_lake_variable
              (variable_id, entity, concept_name, source_class, unit,
               event_time, available_time, semantics_declared,
               appearance_count, authority_class, census_sha256,
               observation_sha256, observed_at, observed_at_source,
               supersedes_sha256)
            VALUES (:v, 'e', 'c', 'sc', :u, 'et', 'at', true, 1,
                    'FINANCIAL_DOMAIN_DEVELOPMENT_ONLY', :d, :o, :oa,
                    'ARTIFACT_INDEXED_AT', :sup)
            ON CONFLICT (variable_id, observation_sha256) DO NOTHING
        """), {"v": variable_id, "u": unit, "d": digest, "o": obs,
               "oa": observed_at, "sup": supersedes})
    return obs


def current(engine, variable_id="v1"):
    from sqlalchemy import text
    with engine.connect() as c:
        return dict(c.execute(text(
            f"SELECT * FROM {inv.SCHEMA}.v_lake_variable_current "
            "WHERE variable_id = :v"), {"v": variable_id})
            .mappings().first())


def versions(engine, variable_id="v1") -> int:
    from sqlalchemy import text
    with engine.connect() as c:
        return c.execute(text(
            f"SELECT count(*) FROM {inv.SCHEMA}.dim_lake_variable "
            "WHERE variable_id = :v"), {"v": variable_id}).scalar()


# ------------------------------------------------------------ the four
@requires_pg
def test_old_then_new(engine):
    put(engine, observed_at="2026-01-01T00:00:00Z", digest="a" * 64)
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="b" * 64)
    row = current(engine)
    assert row["observed_at"] == "2026-06-01T00:00:00Z"
    assert row["currency_state"] == "UNAMBIGUOUS"
    assert versions(engine) == 2


@requires_pg
def test_new_then_old_does_not_resurrect_the_old_one(engine):
    """The direction the previous test never tried."""
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="b" * 64)
    put(engine, observed_at="2026-01-01T00:00:00Z", digest="a" * 64)
    row = current(engine)
    assert row["observed_at"] == "2026-06-01T00:00:00Z", (
        "re-importing an older census must not make it current")
    assert row["census_sha256"] == "b" * 64
    assert versions(engine) == 2, "and the old one is still kept"


@requires_pg
def test_repeating_an_observation_is_idempotent(engine):
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="b" * 64)
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="b" * 64)
    assert versions(engine) == 1
    assert current(engine)["currency_state"] == "UNAMBIGUOUS"


@requires_pg
def test_two_incomparable_branches_are_declared_ambiguous(engine):
    from sqlalchemy import text
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="b" * 64)
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="c" * 64,
        unit="eur")
    row = current(engine)
    assert row["currency_state"] == "AMBIGUOUS_TIE", (
        "two observations of the same instant with no supersession "
        "link between them cannot be ordered; saying so is the answer")
    with engine.connect() as c:
        amb = [dict(r) for r in c.execute(text(
            f"SELECT * FROM {inv.SCHEMA}.v_lake_variable_ambiguous"))
            .mappings()]
    assert len(amb) == 1 and amb[0]["branches"] == 2


# ------------------------------------------------------- supersession
@requires_pg
def test_an_explicit_supersession_beats_the_clock(engine):
    """A producer asserting an order outranks inference from dates."""
    older = put(engine, observed_at="2026-06-01T00:00:00Z",
                digest="b" * 64)
    put(engine, observed_at="2026-01-01T00:00:00Z", digest="a" * 64,
        supersedes=older)
    row = current(engine)
    assert row["census_sha256"] == "a" * 64
    assert row["observed_at"] == "2026-01-01T00:00:00Z"
    assert versions(engine) == 2


@requires_pg
def test_a_supersession_resolves_an_otherwise_ambiguous_tie(engine):
    first = put(engine, observed_at="2026-06-01T00:00:00Z",
                digest="b" * 64)
    put(engine, observed_at="2026-06-01T00:00:00Z", digest="c" * 64,
        unit="eur", supersedes=first)
    row = current(engine)
    assert row["census_sha256"] == "c" * 64
    assert row["currency_state"] == "UNAMBIGUOUS", (
        "the tie is only a tie when nothing links the branches")


@requires_pg
def test_the_views_never_order_by_load_time(engine):
    from sqlalchemy import text
    with engine.connect() as c:
        for name in ("v_lake_appearance_current", "v_lake_variable_current",
                     "v_public_series_current",
                     "v_synthetic_generator_current"):
            body = c.execute(text(
                "SELECT pg_get_viewdef(CAST(:v AS regclass), true)"),
                {"v": f"{inv.SCHEMA}.{name}"}).scalar()
            assert "loaded_at DESC" not in body, name
            assert "observed_at DESC" in body, name


# ----------------------------------------------------- the producer
def test_an_index_without_its_own_chronology_refuses(monkeypatch):
    """No `indexed_at` means currency would fall back to load order."""
    class FakeEngine:
        def begin(self):                                 # pragma: no cover
            raise AssertionError("must refuse before touching the db")
    with pytest.raises(SystemExit, match="carries no `indexed_at`"):
        inv.load_index(FakeEngine(), {"schema": "crispdm.bank_index.v1",
                                      "index_sha256": "0" * 64,
                                      "banks": {}})
