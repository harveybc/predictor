-- olap_schema_patch_experiment_key.sql

-- Ensure experiment_key is not null
ALTER TABLE olap.dim_experiment
    ALTER COLUMN experiment_key SET NOT NULL;

-- Example generated columns from config JSON
ALTER TABLE olap.dim_experiment
    ADD COLUMN IF NOT EXISTS activation_gen TEXT
        GENERATED ALWAYS AS (config->>'activation') STORED;

ALTER TABLE olap.dim_experiment
    ADD COLUMN IF NOT EXISTS learning_rate_gen DOUBLE PRECISION
        GENERATED ALWAYS AS ((config->>'learning_rate')::DOUBLE PRECISION) STORED;

ALTER TABLE olap.dim_experiment
    ADD COLUMN IF NOT EXISTS window_size_gen INT
        GENERATED ALWAYS AS ((config->>'window_size')::INT) STORED;
