-- C92 (order 2026-09-12): six evidence layers, added beside C73's three.
--
-- C73's tables keep their three-layer CHECK and their rows; nothing here
-- alters or deletes them. A v3 verification is recorded in new tables whose
-- layer vocabulary adds PHYSICALLY_TYPED, SEMANTICALLY_UNRESOLVED and
-- DIVERGES, and a new view names, for every historical characterization
-- row, the newest layer across BOTH generations.

CREATE TABLE IF NOT EXISTS public.dim_terminal_verification_v2 (
    verification_sha256   TEXT PRIMARY KEY,
    report_schema         TEXT NOT NULL,
    census_recomputed     TEXT NOT NULL,
    population_verdict    TEXT NOT NULL,
    divergence_count      INTEGER NOT NULL,
    layers                JSONB NOT NULL,
    semantic_by_state     JSONB NOT NULL,
    observed_at           TIMESTAMPTZ NOT NULL,
    loaded_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.fact_terminal_verification_variable_v2 (
    verification_sha256        TEXT NOT NULL
        REFERENCES public.dim_terminal_verification_v2 (verification_sha256),
    variable_id                TEXT NOT NULL,
    producer_declared_outcome  TEXT NOT NULL,
    layer                      TEXT NOT NULL CHECK (layer IN (
        'PRODUCER_DECLARED', 'SOURCE_BOUND', 'PHYSICALLY_TYPED',
        'INDEPENDENTLY_RECOMPUTED', 'SEMANTICALLY_UNRESOLVED', 'DIVERGES')),
    semantic_state             TEXT,
    arrow_type                 TEXT,
    withdrawn_descriptors      JSONB NOT NULL DEFAULT '[]'::jsonb,
    terminal_sha256            TEXT NOT NULL,
    PRIMARY KEY (verification_sha256, variable_id)
);

CREATE TABLE IF NOT EXISTS public.fact_terminal_verification_descriptor_v2 (
    verification_sha256   TEXT NOT NULL
        REFERENCES public.dim_terminal_verification_v2 (verification_sha256),
    variable_id           TEXT NOT NULL,
    descriptor            TEXT NOT NULL,
    state                 TEXT NOT NULL,
    specificity           TEXT NOT NULL,
    PRIMARY KEY (verification_sha256, variable_id, descriptor)
);

CREATE OR REPLACE VIEW public.v_characterization_evidence_layer_v2 AS
WITH newest AS (
    SELECT verification_sha256,
           count(*) OVER (PARTITION BY observed_at) AS same_instant,
           row_number() OVER (ORDER BY observed_at DESC,
                                       verification_sha256 DESC) AS rn
      FROM public.dim_terminal_verification_v2
)
SELECT f.variable_id, f.partition_key, f.descriptor, f.value,
       f.value_text, f.identifiable, f.terminal_attempt,
       CASE
         WHEN v.layer IS NULL THEN coalesce(old.evidence_layer, 'PRODUCER_DECLARED')
         WHEN v.layer = 'SEMANTICALLY_UNRESOLVED' AND f.value IS NOT NULL
           THEN 'SEMANTICALLY_UNRESOLVED'
         WHEN d.state = 'AGREES' THEN 'INDEPENDENTLY_RECOMPUTED'
         WHEN d.state = 'DIVERGES' THEN 'DIVERGES'
         WHEN d.state = 'NOT_INDEPENDENTLY_VERIFIABLE' THEN 'PHYSICALLY_TYPED'
         ELSE v.layer
       END AS evidence_layer,
       v.layer AS variable_layer,
       v.semantic_state,
       (v.layer = 'SEMANTICALLY_UNRESOLVED' AND f.value IS NOT NULL)
           AS published_numeric_withdrawn,
       n.verification_sha256,
       CASE WHEN n.same_instant > 1 THEN 'AMBIGUOUS_TIE'
            WHEN n.verification_sha256 IS NULL THEN 'NO_V3_VERIFICATION'
            ELSE 'UNAMBIGUOUS' END AS currency_state
  FROM public.fact_variable_characterization f
  LEFT JOIN newest n ON n.rn = 1
  LEFT JOIN public.fact_terminal_verification_variable_v2 v
         ON v.verification_sha256 = n.verification_sha256
        AND v.variable_id = f.variable_id
  LEFT JOIN public.fact_terminal_verification_descriptor_v2 d
         ON d.verification_sha256 = n.verification_sha256
        AND d.variable_id = f.variable_id AND d.descriptor = f.descriptor
  LEFT JOIN public.v_characterization_evidence_layer old
         ON old.variable_id = f.variable_id AND old.descriptor = f.descriptor
        AND old.partition_key IS NOT DISTINCT FROM f.partition_key
        AND old.terminal_attempt IS NOT DISTINCT FROM f.terminal_attempt;
