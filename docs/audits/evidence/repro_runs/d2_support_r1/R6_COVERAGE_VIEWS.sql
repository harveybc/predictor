-- D2-R6: coverage views with EXPLICIT version and code selection (proposed, not applied
-- to the production cube; rehearsed on a throwaway database, see R6_THROWAWAY_TEST.out).
--
-- Facts in the cube (2026-09-14): df_fact_coverage (v1) holds ONE run id
-- c140_17e79fa33a11f298c70180ec with TWO code digests (31e3376d… and b4c9157c…),
-- 220,347 rows each — the two v1 matrices under one identifier; df_fact_coverage_v2
-- holds run c170_17e79fa33a11f298c70180ec, code b4c9157c…, 633,189 rows;
-- df_fact_coverage_v1_v2_map links c170 -> c140. Nothing is deleted or deduplicated:
-- the current view selects one (table, run_id, code_sha256) declared in a small
-- selection table; the history view keeps every version with its provenance.

CREATE TABLE IF NOT EXISTS df_coverage_version_selection (
    selected_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    table_name    TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    code_sha256   TEXT NOT NULL,
    reason        TEXT NOT NULL,
    PRIMARY KEY (selected_at, table_name)
);

-- the selection is a governed write (adoption route), never an implicit default
INSERT INTO df_coverage_version_selection (table_name, run_id, code_sha256, reason)
VALUES ('df_fact_coverage_v2', 'c170_17e79fa33a11f298c70180ec',
        'b4c9157c206dc914953c0ca52d6ae60fa79780d38aaef318ba2eb3cf1d53cb86',
        'C170 coverage v2 (9 states + applicability); supersedes v1 c140, which carries two code digests under one run id');

CREATE OR REPLACE VIEW df_coverage_current AS
SELECT v2.*
FROM df_fact_coverage_v2 v2
JOIN (
    SELECT run_id, code_sha256
    FROM df_coverage_version_selection
    WHERE table_name = 'df_fact_coverage_v2'
    ORDER BY selected_at DESC
    LIMIT 1
) sel ON sel.run_id = v2.run_id AND sel.code_sha256 = v2.code_sha256;

CREATE OR REPLACE VIEW df_coverage_history AS
SELECT 'v1'::TEXT AS version, run_id, code_sha256, dataset_id, variable_id, metric, operator, state,
       NULL::TEXT AS partition, NULL::TEXT AS policy, NULL::TEXT AS applicability, loaded_at
FROM df_fact_coverage
UNION ALL
SELECT 'v2', run_id, code_sha256, dataset_id, variable_id, metric, operator, state,
       partition, policy, applicability, loaded_at
FROM df_fact_coverage_v2;

-- the denominator of the current view is one matrix: one row per
-- (dataset, variable, partition, metric, operator) for the selected version
CREATE OR REPLACE VIEW df_coverage_current_denominator AS
SELECT dataset_id, COUNT(*) AS cells, COUNT(DISTINCT (variable_id, partition, metric, operator)) AS distinct_cells
FROM df_coverage_current
GROUP BY dataset_id;
