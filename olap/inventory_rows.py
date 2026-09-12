"""C15: the census and the common index, inside the cube.

Additive tables for the rows the index now carries — physical
appearances, conceptual variables, public series, synthetic
generators and eligibility decisions. They are kept apart from
the 99 historical variable profiles already in the cube, because
those describe two locally registered datasets and these describe
a 1,965-variable lake: mixing them would make one look like the
other.

`event_time` and `available_time` stay separate columns, the
authority class travels with every row, and the metadata state is
explicit — so a query can always ask "what is actually declared
here?" and get an honest answer.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "public"


def observation_sha256(payload: dict) -> str:
    """C25: the identity of ONE OBSERVATION of a logical id.

    Two loads of the same observation collide on this digest and
    are idempotent; a changed digest, availability, semantics or
    membership produces a DIFFERENT observation, which is stored
    as a new version beside the old one instead of being dropped.
    """
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()

INVENTORY_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_lake_appearance (
  appearance_id     TEXT NOT NULL,
  entity            TEXT NOT NULL,
  source_class      TEXT NOT NULL,
  frequency         TEXT NOT NULL,
  period_start      TEXT NOT NULL,
  period_end        TEXT NOT NULL,
  physical_sha256   TEXT NOT NULL,
  digest_state      TEXT NOT NULL,
  authority_class   TEXT NOT NULL,
  census_sha256     TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (appearance_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_lake_variable (
  variable_id       TEXT NOT NULL,
  entity            TEXT NOT NULL,
  concept_name      TEXT NOT NULL,
  source_class      TEXT NOT NULL,
  unit              TEXT NOT NULL,
  event_time        TEXT NOT NULL,
  available_time    TEXT NOT NULL,
  semantics_declared BOOLEAN NOT NULL,
  appearance_count  INTEGER NOT NULL,
  authority_class   TEXT NOT NULL,
  census_sha256     TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (variable_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_public_series (
  series_id         TEXT NOT NULL,
  family            TEXT NOT NULL,
  dataset_id        TEXT NOT NULL,
  frequency         TEXT NOT NULL,
  digest            TEXT NOT NULL,
  authority_class   TEXT NOT NULL,
  index_sha256      TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (series_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_synthetic_generator (
  generator_id      TEXT NOT NULL,
  producer          TEXT NOT NULL,
  family            TEXT NOT NULL,
  role              TEXT NOT NULL,
  mechanism_json    JSONB NOT NULL,
  authority_class   TEXT NOT NULL,
  index_sha256      TEXT NOT NULL,
  observation_sha256 TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (generator_id, observation_sha256)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_eligibility_decision (
  subject_id        TEXT NOT NULL,
  subject_kind      TEXT NOT NULL,
  scope             TEXT NOT NULL,
  decision          TEXT NOT NULL,
  decision_reason   TEXT NOT NULL,
  reviewer          TEXT NOT NULL,
  reviewed_at       TEXT NOT NULL,
  manifest_sha256   TEXT NOT NULL,
  authority_class   TEXT NOT NULL,
  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (subject_id, scope, manifest_sha256)
);

-- C38 (order 2026-09-11): CURRENT means the latest SCIENTIFIC
-- observation, not the latest LOAD.
--
-- The views ordered by loaded_at DESC. Re-importing an old census
-- today therefore made the OLD observation current, silently
-- superseding a newer one — and the existing test only ever loaded
-- old-then-new, so it never saw it. Load time is a fact about this
-- database; it says nothing about when the world was observed.
--
-- Three things decide currency now, in order:
--
--   1. SUPERSESSION. An observation another observation explicitly
--      supersedes is never current, whatever its date. An explicit
--      chain is the producer asserting an order and beats inference;
--   2. OBSERVED TIME. The artifact's own chronology — when the census
--      or index was produced — not when this database heard about it;
--   3. the observation digest, purely so two readers agree.
--
-- When the newest observed_at is shared by observations with no
-- supersession link between them, the two branches are INCOMPARABLE.
-- The view still returns one row deterministically, but it says so:
-- currency_state = AMBIGUOUS_TIE. Silently picking a winner among
-- incomparable branches is the failure mode this replaces.
ALTER TABLE {SCHEMA}.dim_lake_appearance
  ADD COLUMN IF NOT EXISTS observed_at TEXT;
ALTER TABLE {SCHEMA}.dim_lake_appearance
  ADD COLUMN IF NOT EXISTS observed_at_source TEXT;
ALTER TABLE {SCHEMA}.dim_lake_appearance
  ADD COLUMN IF NOT EXISTS supersedes_sha256 TEXT;
ALTER TABLE {SCHEMA}.dim_lake_variable
  ADD COLUMN IF NOT EXISTS observed_at TEXT;
ALTER TABLE {SCHEMA}.dim_lake_variable
  ADD COLUMN IF NOT EXISTS observed_at_source TEXT;
ALTER TABLE {SCHEMA}.dim_lake_variable
  ADD COLUMN IF NOT EXISTS supersedes_sha256 TEXT;
ALTER TABLE {SCHEMA}.dim_public_series
  ADD COLUMN IF NOT EXISTS observed_at TEXT;
ALTER TABLE {SCHEMA}.dim_public_series
  ADD COLUMN IF NOT EXISTS observed_at_source TEXT;
ALTER TABLE {SCHEMA}.dim_public_series
  ADD COLUMN IF NOT EXISTS supersedes_sha256 TEXT;
ALTER TABLE {SCHEMA}.dim_synthetic_generator
  ADD COLUMN IF NOT EXISTS observed_at TEXT;
ALTER TABLE {SCHEMA}.dim_synthetic_generator
  ADD COLUMN IF NOT EXISTS observed_at_source TEXT;
ALTER TABLE {SCHEMA}.dim_synthetic_generator
  ADD COLUMN IF NOT EXISTS supersedes_sha256 TEXT;

-- Rows that predate C38 have no artifact chronology to recover. They
-- keep their load time as a STAND-IN, and the stand-in is labelled so
-- nobody reads it as the producer's own date.
UPDATE {SCHEMA}.dim_lake_appearance
   SET observed_at = to_char(loaded_at AT TIME ZONE 'UTC',
                             'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
       observed_at_source = 'BACKFILLED_FROM_LOAD_TIME'
 WHERE observed_at IS NULL;
UPDATE {SCHEMA}.dim_lake_variable
   SET observed_at = to_char(loaded_at AT TIME ZONE 'UTC',
                             'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
       observed_at_source = 'BACKFILLED_FROM_LOAD_TIME'
 WHERE observed_at IS NULL;
UPDATE {SCHEMA}.dim_public_series
   SET observed_at = to_char(loaded_at AT TIME ZONE 'UTC',
                             'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
       observed_at_source = 'BACKFILLED_FROM_LOAD_TIME'
 WHERE observed_at IS NULL;
UPDATE {SCHEMA}.dim_synthetic_generator
   SET observed_at = to_char(loaded_at AT TIME ZONE 'UTC',
                             'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
       observed_at_source = 'BACKFILLED_FROM_LOAD_TIME'
 WHERE observed_at IS NULL;

DROP VIEW IF EXISTS {SCHEMA}.v_lake_appearance_current;
CREATE VIEW {SCHEMA}.v_lake_appearance_current AS
WITH base AS (
  SELECT t.*,
         EXISTS (SELECT 1 FROM {SCHEMA}.dim_lake_appearance s
                  WHERE s.appearance_id = t.appearance_id
                    AND s.supersedes_sha256 = t.observation_sha256)
           AS is_superseded
    FROM {SCHEMA}.dim_lake_appearance t
), ranked AS (
  SELECT b.*,
         count(*) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY appearance_id, observed_at) AS live_at_observed,
         max(observed_at) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY appearance_id)             AS newest_live
    FROM base b
)
SELECT DISTINCT ON (appearance_id) ranked.*,
       CASE WHEN NOT is_superseded
             AND observed_at = newest_live
             AND live_at_observed > 1
            THEN 'AMBIGUOUS_TIE' ELSE 'UNAMBIGUOUS'
       END AS currency_state
  FROM ranked
 ORDER BY appearance_id, is_superseded ASC, observed_at DESC,
          observation_sha256 DESC;

DROP VIEW IF EXISTS {SCHEMA}.v_lake_appearance_ambiguous;
CREATE VIEW {SCHEMA}.v_lake_appearance_ambiguous AS
SELECT appearance_id, observed_at, count(*) AS branches
  FROM {SCHEMA}.dim_lake_appearance t
 WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.dim_lake_appearance s
                    WHERE s.appearance_id = t.appearance_id
                      AND s.supersedes_sha256 = t.observation_sha256)
 GROUP BY appearance_id, observed_at
HAVING count(*) > 1;

DROP VIEW IF EXISTS {SCHEMA}.v_lake_variable_current;
CREATE VIEW {SCHEMA}.v_lake_variable_current AS
WITH base AS (
  SELECT t.*,
         EXISTS (SELECT 1 FROM {SCHEMA}.dim_lake_variable s
                  WHERE s.variable_id = t.variable_id
                    AND s.supersedes_sha256 = t.observation_sha256)
           AS is_superseded
    FROM {SCHEMA}.dim_lake_variable t
), ranked AS (
  SELECT b.*,
         count(*) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY variable_id, observed_at) AS live_at_observed,
         max(observed_at) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY variable_id)             AS newest_live
    FROM base b
)
SELECT DISTINCT ON (variable_id) ranked.*,
       CASE WHEN NOT is_superseded
             AND observed_at = newest_live
             AND live_at_observed > 1
            THEN 'AMBIGUOUS_TIE' ELSE 'UNAMBIGUOUS'
       END AS currency_state
  FROM ranked
 ORDER BY variable_id, is_superseded ASC, observed_at DESC,
          observation_sha256 DESC;

DROP VIEW IF EXISTS {SCHEMA}.v_lake_variable_ambiguous;
CREATE VIEW {SCHEMA}.v_lake_variable_ambiguous AS
SELECT variable_id, observed_at, count(*) AS branches
  FROM {SCHEMA}.dim_lake_variable t
 WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.dim_lake_variable s
                    WHERE s.variable_id = t.variable_id
                      AND s.supersedes_sha256 = t.observation_sha256)
 GROUP BY variable_id, observed_at
HAVING count(*) > 1;

DROP VIEW IF EXISTS {SCHEMA}.v_public_series_current;
CREATE VIEW {SCHEMA}.v_public_series_current AS
WITH base AS (
  SELECT t.*,
         EXISTS (SELECT 1 FROM {SCHEMA}.dim_public_series s
                  WHERE s.series_id = t.series_id
                    AND s.supersedes_sha256 = t.observation_sha256)
           AS is_superseded
    FROM {SCHEMA}.dim_public_series t
), ranked AS (
  SELECT b.*,
         count(*) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY series_id, observed_at) AS live_at_observed,
         max(observed_at) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY series_id)             AS newest_live
    FROM base b
)
SELECT DISTINCT ON (series_id) ranked.*,
       CASE WHEN NOT is_superseded
             AND observed_at = newest_live
             AND live_at_observed > 1
            THEN 'AMBIGUOUS_TIE' ELSE 'UNAMBIGUOUS'
       END AS currency_state
  FROM ranked
 ORDER BY series_id, is_superseded ASC, observed_at DESC,
          observation_sha256 DESC;

DROP VIEW IF EXISTS {SCHEMA}.v_public_series_ambiguous;
CREATE VIEW {SCHEMA}.v_public_series_ambiguous AS
SELECT series_id, observed_at, count(*) AS branches
  FROM {SCHEMA}.dim_public_series t
 WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.dim_public_series s
                    WHERE s.series_id = t.series_id
                      AND s.supersedes_sha256 = t.observation_sha256)
 GROUP BY series_id, observed_at
HAVING count(*) > 1;

DROP VIEW IF EXISTS {SCHEMA}.v_synthetic_generator_current;
CREATE VIEW {SCHEMA}.v_synthetic_generator_current AS
WITH base AS (
  SELECT t.*,
         EXISTS (SELECT 1 FROM {SCHEMA}.dim_synthetic_generator s
                  WHERE s.generator_id = t.generator_id
                    AND s.supersedes_sha256 = t.observation_sha256)
           AS is_superseded
    FROM {SCHEMA}.dim_synthetic_generator t
), ranked AS (
  SELECT b.*,
         count(*) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY generator_id, observed_at) AS live_at_observed,
         max(observed_at) FILTER (WHERE NOT is_superseded)
           OVER (PARTITION BY generator_id)             AS newest_live
    FROM base b
)
SELECT DISTINCT ON (generator_id) ranked.*,
       CASE WHEN NOT is_superseded
             AND observed_at = newest_live
             AND live_at_observed > 1
            THEN 'AMBIGUOUS_TIE' ELSE 'UNAMBIGUOUS'
       END AS currency_state
  FROM ranked
 ORDER BY generator_id, is_superseded ASC, observed_at DESC,
          observation_sha256 DESC;

DROP VIEW IF EXISTS {SCHEMA}.v_synthetic_generator_ambiguous;
CREATE VIEW {SCHEMA}.v_synthetic_generator_ambiguous AS
SELECT generator_id, observed_at, count(*) AS branches
  FROM {SCHEMA}.dim_synthetic_generator t
 WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.dim_synthetic_generator s
                    WHERE s.generator_id = t.generator_id
                      AND s.supersedes_sha256 = t.observation_sha256)
 GROUP BY generator_id, observed_at
HAVING count(*) > 1;


CREATE INDEX IF NOT EXISTS idx_dim_lake_variable_entity
  ON {SCHEMA}.dim_lake_variable (entity);
CREATE INDEX IF NOT EXISTS idx_dim_lake_variable_available
  ON {SCHEMA}.dim_lake_variable (available_time);
CREATE INDEX IF NOT EXISTS idx_dim_lake_appearance_entity
  ON {SCHEMA}.dim_lake_appearance (entity);
CREATE INDEX IF NOT EXISTS idx_dim_public_series_family
  ON {SCHEMA}.dim_public_series (family);
"""

# The historical tables this loader must never touch.
HISTORICAL_TABLES = ("dim_experiment", "fact_performance",
                     "fact_results_summary", "dim_dataset",
                     "dim_series", "dim_variable",
                     "fact_dataset_inventory",
                     "fact_variable_profile")

NEW_TABLES = ("dim_lake_appearance", "dim_lake_variable",
              "dim_public_series", "dim_synthetic_generator",
              "fact_eligibility_decision")


class InventoryLoadRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def ensure_inventory_tables(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(INVENTORY_DDL)


def load_index(engine, index: dict) -> dict:
    """Load the common index additively. Idempotent by id."""
    from sqlalchemy import text

    if index.get("schema") != "crispdm.bank_index.v1":
        raise InventoryLoadRefusal(
            f"not a bank index: {index.get('schema')!r}")
    index_sha = index["index_sha256"]
    # C38: the artifact's OWN chronology decides currency. `indexed_at`
    # is when the index was built; `loaded_at` is when this database
    # heard about it, and re-importing an old index today must not make
    # it current.
    observed_at = index.get("indexed_at")
    observed_source = "ARTIFACT_INDEXED_AT"
    if not isinstance(observed_at, str) or not observed_at.strip():
        raise InventoryLoadRefusal(
            "the index carries no `indexed_at`; without the artifact's "
            "own chronology, currency would silently fall back to load "
            "order — which is the defect C38 exists to remove")
    banks = index["banks"]
    counts = {t: 0 for t in NEW_TABLES}
    with engine.begin() as conn:
        conn.exec_driver_sql(INVENTORY_DDL)
        fin = banks.get("financial_domain", {})
        census_sha = fin.get("binding", {}).get(
            "census_document_sha256", "UNAVAILABLE")
        auth_fin = fin.get("authority_class", "UNAVAILABLE")
        for a in fin.get("appearances", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_lake_appearance
                  (appearance_id, entity, source_class,
                   frequency, period_start, period_end,
                   physical_sha256, digest_state,
                   authority_class, census_sha256,
                   observation_sha256, observed_at,
                   observed_at_source)
                VALUES (:i, :e, :sc, :f, :ps, :pe, :d, :ds, :a,
                        :c, :o, :oa, :os)
                ON CONFLICT (appearance_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "entity": a["entity"],
                       "source_class": a["source_class"],
                       "frequency": a["frequency"],
                       "period_start": str(a["period_start"]),
                       "period_end": str(a["period_end"]),
                       "physical_sha256": a["physical_sha256"],
                       "digest_state": a["digest_state"],
                       "authority_class": auth_fin,
                       "census_sha256": census_sha}),
                   "i": a["appearance_id"], "e": a["entity"],
                   "sc": a["source_class"],
                   "f": a["frequency"],
                   "ps": str(a["period_start"]),
                   "pe": str(a["period_end"]),
                   "d": a["physical_sha256"],
                   "ds": a["digest_state"], "a": auth_fin,
                   "c": census_sha, "oa": observed_at,
                   "os": observed_source})
            counts["dim_lake_appearance"] += r.rowcount or 0
        for v in fin.get("variables", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_lake_variable
                  (variable_id, entity, concept_name,
                   source_class, unit, event_time,
                   available_time, semantics_declared,
                   appearance_count, authority_class,
                   census_sha256, observation_sha256,
                   observed_at, observed_at_source)
                VALUES (:i, :e, :cn, :sc, :u, :et, :at, :sd,
                        :ac, :a, :c, :o, :oa, :os)
                ON CONFLICT (variable_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "entity": v["entity"],
                       "concept_name": v["concept_name"],
                       "source_class": v["source_class"],
                       "unit": v["unit"],
                       "event_time": v["event_time"],
                       "available_time": v["available_time"],
                       "semantics_declared":
                           bool(v["semantics_declared"]),
                       "appearance_count":
                           int(v["appearance_count"]),
                       "authority_class": auth_fin,
                       "census_sha256": census_sha}),
                   "i": v["variable_id"], "e": v["entity"],
                   "cn": v["concept_name"],
                   "sc": v["source_class"], "u": v["unit"],
                   "et": v["event_time"],
                   "at": v["available_time"],
                   "sd": bool(v["semantics_declared"]),
                   "ac": int(v["appearance_count"]),
                   "a": auth_fin, "c": census_sha,
                   "oa": observed_at, "os": observed_source})
            counts["dim_lake_variable"] += r.rowcount or 0

        pub = banks.get("public_forecasting", {})
        auth_pub = pub.get("authority_class", "UNAVAILABLE")
        for s in pub.get("series", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_public_series
                  (series_id, family, dataset_id, frequency,
                   digest, authority_class, index_sha256,
                   observation_sha256, observed_at,
                   observed_at_source)
                VALUES (:i, :f, :d, :fr, :dg, :a, :x, :o, :oa,
                        :os)
                ON CONFLICT (series_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "family": s["family"],
                       "dataset_id": s["dataset_id"],
                       "frequency": str(s.get("frequency",
                                              "UNKNOWN")),
                       "digest": s.get("digest", "UNAVAILABLE"),
                       "authority_class": auth_pub}),
                   "i": s["series_id"], "f": s["family"],
                   "d": s["dataset_id"],
                   "fr": str(s.get("frequency", "UNKNOWN")),
                   "dg": s.get("digest", "UNAVAILABLE"),
                   "a": auth_pub, "x": index_sha,
                   "oa": observed_at, "os": observed_source})
            counts["dim_public_series"] += r.rowcount or 0

        syn = banks.get("synthetic_known_mechanism", {})
        auth_syn = syn.get("authority_class", "UNAVAILABLE")
        for g in syn.get("generators", []):
            r = conn.execute(text(f"""
                INSERT INTO {SCHEMA}.dim_synthetic_generator
                  (generator_id, producer, family, role,
                   mechanism_json, authority_class,
                   index_sha256, observation_sha256,
                   observed_at, observed_at_source)
                VALUES (:i, :p, :f, :r, CAST(:m AS JSONB), :a,
                        :x, :o, :oa, :os)
                ON CONFLICT (generator_id, observation_sha256)
                DO NOTHING
            """), {"o": observation_sha256({
                       "producer": g["producer"],
                       "mechanism": g["mechanism"],
                       "role": g["role"],
                       "authority_class": auth_syn}),
                   "i": g["generator_id"], "p": g["producer"],
                   "f": str(g["mechanism"].get("family",
                                               "UNKNOWN")),
                   "r": g["role"],
                   "m": json.dumps(g["mechanism"],
                                   sort_keys=True),
                   "a": auth_syn, "x": index_sha,
                   "oa": observed_at, "os": observed_source})
            counts["dim_synthetic_generator"] += r.rowcount or 0
    return counts


def load_eligibility_decisions(engine, manifest: dict,
                               *, authority_class: str) -> dict:
    """Eligibility decisions are facts about subjects, and belong
    in the cube like any other decision."""
    from sqlalchemy import text
    loaded = 0
    with engine.begin() as conn:
        conn.exec_driver_sql(INVENTORY_DDL)
        for e in manifest.get("entries", []):
            scopes = e["decision_scope"]
            if isinstance(scopes, str):
                scopes = [scopes]
            for scope in scopes:
                r = conn.execute(text(f"""
                    INSERT INTO
                      {SCHEMA}.fact_eligibility_decision
                      (subject_id, subject_kind, scope,
                       decision, decision_reason, reviewer,
                       reviewed_at, manifest_sha256,
                       authority_class)
                    VALUES (:i, :k, :s, :d, :r, :rv, :ra, :m,
                            :a)
                    ON CONFLICT (subject_id, scope,
                                 manifest_sha256) DO NOTHING
                """), {"i": e["subject_id"],
                       "k": e["subject_kind"], "s": scope,
                       "d": e["decision"],
                       "r": e["decision_reason"],
                       "rv": e["reviewer"],
                       "ra": e["reviewed_at"],
                       "m": manifest["manifest_sha256"],
                       "a": authority_class})
                loaded += r.rowcount or 0
    return {"fact_eligibility_decision": loaded}
