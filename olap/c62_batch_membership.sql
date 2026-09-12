-- C62 (order 2026-09-12): exact batch membership, and a current view
-- that refuses to break a tie in silence.
--
-- Two gaps the audit is entitled to name:
--
--  1. The cube recorded `terminal_attempt` — WHICH RUN produced a row
--     — and nothing about which BATCH a variable belonged to. A run of
--     1,965 variables in 13 batches was, to the cube, one opaque
--     attempt. "Which variables were in batch_00007, exactly?" had no
--     answer, so a batch could neither be re-run nor audited.
--
--  2. v_variable_characterization_current orders by measured_at DESC
--     and then observation_sha256 DESC. The second key makes the view
--     deterministic, which is necessary, but it also makes a genuine
--     TIE — two observations of one variable at the same instant —
--     indistinguishable from a clean succession. A reader cannot tell
--     that a coin was flipped on their behalf.
--
-- Both are additive. No existing table is altered, no existing view is
-- dropped, and no row is deleted.

CREATE TABLE IF NOT EXISTS public.dim_characterization_batch (
    terminal_attempt     TEXT NOT NULL,
    batch_key            TEXT NOT NULL,
    variables_declared   INTEGER NOT NULL,
    membership_sha256    TEXT NOT NULL,
    outcomes_json        JSONB NOT NULL,
    observed_at          TIMESTAMPTZ NOT NULL,
    loaded_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (terminal_attempt, batch_key)
);

COMMENT ON COLUMN public.dim_characterization_batch.membership_sha256 IS
 'sha256 over the SORTED variable_id list of this batch. Two loads of '
 'the same batch agree; one added or removed variable does not.';

CREATE TABLE IF NOT EXISTS public.bridge_batch_variable (
    terminal_attempt  TEXT NOT NULL,
    batch_key         TEXT NOT NULL,
    variable_id       TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    terminal_sha256   TEXT NOT NULL,
    source_sha256     TEXT,
    PRIMARY KEY (terminal_attempt, batch_key, variable_id)
);

COMMENT ON TABLE public.bridge_batch_variable IS
 'EXACT membership: one row per (attempt, batch, variable). The '
 'question "which variables were in batch_00007" is answered by rows, '
 'not by a count.';

CREATE INDEX IF NOT EXISTS bridge_batch_variable_variable_idx
    ON public.bridge_batch_variable (variable_id);

-- A batch whose declared count disagrees with its rows is a defect,
-- not a rounding difference. The view names it rather than hiding it.
CREATE OR REPLACE VIEW public.v_batch_membership_integrity AS
SELECT d.terminal_attempt,
       d.batch_key,
       d.variables_declared,
       count(b.variable_id)                       AS variables_present,
       d.variables_declared - count(b.variable_id) AS shortfall,
       CASE WHEN d.variables_declared = count(b.variable_id)
            THEN 'EXACT' ELSE 'MEMBERSHIP_DIVERGES' END AS state
  FROM public.dim_characterization_batch d
  LEFT JOIN public.bridge_batch_variable b
         ON b.terminal_attempt = d.terminal_attempt
        AND b.batch_key = d.batch_key
 GROUP BY d.terminal_attempt, d.batch_key, d.variables_declared;

-- C62 current view, superseding rather than replacing. It keeps the
-- deterministic order AND publishes whether the choice was forced.
CREATE OR REPLACE VIEW public.v_variable_characterization_current_v2 AS
WITH ranked AS (
    SELECT f.*,
           count(*) OVER w                    AS observations,
           count(*) FILTER (WHERE TRUE) OVER (
               PARTITION BY f.variable_id, f.partition_key,
                            f.descriptor, f.measured_at) AS at_newest,
           row_number() OVER (
               PARTITION BY f.variable_id, f.partition_key, f.descriptor
               ORDER BY f.measured_at DESC,
                        f.observation_sha256 DESC)       AS rn,
           max(f.measured_at) OVER w                     AS newest
      FROM public.fact_variable_characterization f
    WINDOW w AS (PARTITION BY f.variable_id, f.partition_key,
                              f.descriptor)
)
SELECT variable_id, partition_key, bank_authority, descriptor,
       value, value_text, descriptor_contract, identifiable,
       cost_seconds, observation_sha256, measured_at, loaded_at,
       source_id, source_sha256, window_sha256, window_contract,
       code_identity, protocol_version, side, contract_role, units,
       terminal_attempt, measurement_sha256, binding_state,
       observations,
       CASE WHEN measured_at = newest AND at_newest > 1
            THEN 'AMBIGUOUS_TIE' ELSE 'UNAMBIGUOUS' END AS currency_state
  FROM ranked
 WHERE rn = 1;

COMMENT ON VIEW public.v_variable_characterization_current_v2 IS
 'Supersedes v_variable_characterization_current. Same deterministic '
 'row, plus currency_state: AMBIGUOUS_TIE means two observations share '
 'the newest measured_at and the tie was broken by digest order. The '
 'v1 view still exists and is not dropped.';
