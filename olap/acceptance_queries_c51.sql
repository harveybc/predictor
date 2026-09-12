-- C51 (order 2026-09-12): the five queries a reviewer runs.
--
-- Every one of them answers a question the order asks, from the cube
-- itself. None of them writes.

-- 1. Every conceptual variable has exactly one terminal disposition.
--    The denominator is the census, not the subjects that succeeded.
-- Counted on the DISPOSITION descriptor, which exists once per census
-- variable. Counting every bound row instead would add the earlier
-- 107-subject run, whose variable ids live in a different namespace
-- (`dataset::column`, not a lake `variable_id`) — 2,072 is the sum of
-- two populations, not the coverage of one.
SELECT 'census_variables_with_a_terminal_disposition' AS fact,
       count(DISTINCT variable_id)                    AS value
  FROM public.fact_variable_characterization
 WHERE descriptor = 'characterization_disposition'
UNION ALL
SELECT 'disposition_' || lower(value_text), count(*)
  FROM public.fact_variable_characterization
 WHERE descriptor = 'characterization_disposition'
 GROUP BY value_text;

-- 2. Coverage by bank and by state. A NOT_IDENTIFIABLE row is a
--    result, not a gap: absence is reported, never imputed.
SELECT bank_authority,
       identifiable,
       count(*)                       AS rows,
       count(DISTINCT variable_id)    AS variables
  FROM public.fact_variable_characterization
 GROUP BY bank_authority, identifiable
 ORDER BY bank_authority, identifiable;

-- 3. Physical appearances are a DIFFERENT population from conceptual
--    variables and are never multiplied together.
SELECT 'conceptual_variables' AS population,
       count(*)               AS value
  FROM public.v_lake_variable_current
UNION ALL
SELECT 'physical_appearances', count(*)
  FROM public.v_lake_appearance_current
UNION ALL
SELECT 'census_variables_dispositioned',
       count(DISTINCT variable_id)
  FROM public.fact_variable_characterization
 WHERE descriptor = 'characterization_disposition'
UNION ALL
SELECT 'earlier_run_subjects_other_namespace',
       count(DISTINCT variable_id)
  FROM public.fact_variable_characterization
 WHERE descriptor <> 'characterization_disposition'
   AND variable_id NOT IN (
        SELECT variable_id FROM public.fact_variable_characterization
         WHERE descriptor = 'characterization_disposition');

-- 4. No selection was emitted. A descriptor that ranked, scored or
--    promoted anything would appear here; the expected result is zero
--    rows.
SELECT descriptor, count(*) AS rows
  FROM public.fact_variable_characterization
 WHERE descriptor ~* '(rank|select|chosen|score|importance|promot|eligib)'
 GROUP BY descriptor;

-- 5. Re-ingestion is idempotent: one row per
--    (variable, partition, descriptor, observation). A repeated load
--    adds nothing; a CHANGED observation is a new version beside the
--    old one, never an overwrite. The expected result is zero rows.
SELECT variable_id, partition_key, descriptor, observation_sha256,
       count(*) AS duplicates
  FROM public.fact_variable_characterization
 GROUP BY 1, 2, 3, 4
HAVING count(*) > 1;

-- 6. The audited counts are conserved. Every number below must be at
--    least what the 2026-09-12 audit recorded.
SELECT 'dim_experiment' AS table_name, count(*) AS value,
       39 AS audited_floor FROM public.dim_experiment
UNION ALL SELECT 'fact_performance', count(*), 1404
  FROM public.fact_performance
UNION ALL SELECT 'dim_campaign_run', count(*), 8
  FROM public.dim_campaign_run
UNION ALL SELECT 'fact_campaign_unit', count(*), 127
  FROM public.fact_campaign_unit
UNION ALL SELECT 'fact_variable_characterization', count(*), 2988
  FROM public.fact_variable_characterization;
