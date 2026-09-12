-- C73 (order 2026-09-12): the evidence layer of every published
-- characterization row, added beside the history and never over it.
--
-- fact_variable_characterization keeps every historical row untouched.
-- A verification run is recorded once (dim_terminal_verification), its
-- per-variable outcome once per variable, and its per-descriptor state
-- once per row it compared. The view names, for every historical row,
-- which of three layers the newest verification reached:
--
--   PRODUCER_DECLARED         no independent check reached this row
--   SOURCE_BOUND              bound to verified source bytes, but not
--                             independently recomputed (underspecified
--                             contract, or a divergence)
--   INDEPENDENTLY_RECOMPUTED  recomputed from source bytes and agreeing

CREATE TABLE IF NOT EXISTS public.dim_terminal_verification (
    verification_sha256     TEXT PRIMARY KEY,
    report_schema           TEXT NOT NULL,
    population_verdict      TEXT NOT NULL,
    recomputation_verdict   TEXT NOT NULL,
    divergence_count        INTEGER NOT NULL,
    variables_by_layer      JSONB NOT NULL,
    published_attempt       TEXT,
    observed_at             TIMESTAMPTZ NOT NULL,
    loaded_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.fact_terminal_verification_variable (
    verification_sha256        TEXT NOT NULL
        REFERENCES public.dim_terminal_verification (verification_sha256),
    variable_id                TEXT NOT NULL,
    producer_declared_outcome  TEXT NOT NULL,
    layer                      TEXT NOT NULL CHECK (layer IN (
        'PRODUCER_DECLARED', 'SOURCE_BOUND', 'INDEPENDENTLY_RECOMPUTED')),
    recomputation              TEXT NOT NULL,
    source_sha256              TEXT,
    terminal_sha256            TEXT NOT NULL,
    PRIMARY KEY (verification_sha256, variable_id)
);

CREATE TABLE IF NOT EXISTS public.fact_terminal_verification_descriptor (
    verification_sha256     TEXT NOT NULL
        REFERENCES public.dim_terminal_verification (verification_sha256),
    variable_id             TEXT NOT NULL,
    descriptor              TEXT NOT NULL,
    layer                   TEXT NOT NULL CHECK (layer IN (
        'PRODUCER_DECLARED', 'SOURCE_BOUND', 'INDEPENDENTLY_RECOMPUTED')),
    state                   TEXT NOT NULL,
    specificity             TEXT NOT NULL,
    reason                  TEXT,
    published_value         DOUBLE PRECISION,
    recomputed_value        DOUBLE PRECISION,
    recomputed_note         TEXT,
    PRIMARY KEY (verification_sha256, variable_id, descriptor)
);

CREATE OR REPLACE VIEW public.v_characterization_evidence_layer AS
WITH newest AS (
    SELECT verification_sha256, observed_at,
           count(*) OVER (PARTITION BY observed_at) AS same_instant,
           row_number() OVER (ORDER BY observed_at DESC,
                                       verification_sha256 DESC) AS rn
      FROM public.dim_terminal_verification
)
SELECT f.variable_id, f.partition_key, f.descriptor, f.value,
       f.value_text, f.identifiable, f.terminal_attempt,
       f.source_sha256 AS published_source_sha256,
       coalesce(d.layer, v.layer, 'PRODUCER_DECLARED') AS evidence_layer,
       coalesce(d.state, CASE WHEN v.layer IS NOT NULL
                              THEN 'NOT_COMPARED' END,
                'NO_VERIFICATION') AS verification_state,
       d.specificity, d.reason,
       n.verification_sha256,
       CASE WHEN n.same_instant > 1 THEN 'AMBIGUOUS_TIE'
            WHEN n.verification_sha256 IS NULL THEN 'NO_VERIFICATION'
            ELSE 'UNAMBIGUOUS' END AS currency_state
  FROM public.fact_variable_characterization f
  LEFT JOIN newest n ON n.rn = 1
  LEFT JOIN public.fact_terminal_verification_variable v
         ON v.verification_sha256 = n.verification_sha256
        AND v.variable_id = f.variable_id
  LEFT JOIN public.fact_terminal_verification_descriptor d
         ON d.verification_sha256 = n.verification_sha256
        AND d.variable_id = f.variable_id
        AND d.descriptor = f.descriptor;
