
-- olap_schema_postgres.sql
CREATE SCHEMA IF NOT EXISTS olap;

-- Projects
CREATE TABLE IF NOT EXISTS olap.dim_project (
    project_id SERIAL PRIMARY KEY,
    project_key TEXT UNIQUE NOT NULL
);

-- Phases
CREATE TABLE IF NOT EXISTS olap.dim_phase (
    phase_id SERIAL PRIMARY KEY,
    project_id INT NOT NULL REFERENCES olap.dim_project(project_id),
    phase_key TEXT NOT NULL,
    UNIQUE(project_id, phase_key)
);

-- Experiments
CREATE TABLE IF NOT EXISTS olap.dim_experiment (
    experiment_id SERIAL PRIMARY KEY,
    phase_id INT NOT NULL REFERENCES olap.dim_phase(phase_id),
    experiment_key TEXT,
    config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(phase_id, experiment_key)
);

-- Horizon (prediction steps ahead)
CREATE TABLE IF NOT EXISTS olap.dim_horizon (
    horizon_id SERIAL PRIMARY KEY,
    horizon_key TEXT UNIQUE NOT NULL
);

-- Dataset (train, validation, test)
CREATE TABLE IF NOT EXISTS olap.dim_dataset (
    dataset_id SERIAL PRIMARY KEY,
    dataset_key TEXT UNIQUE NOT NULL
);

-- Metrics (MAE, R2, etc.)
CREATE TABLE IF NOT EXISTS olap.fact_experiment_metrics (
    metric_id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES olap.dim_experiment(experiment_id),
    dataset_id INT NOT NULL REFERENCES olap.dim_dataset(dataset_id),
    horizon_id INT NOT NULL REFERENCES olap.dim_horizon(horizon_id),
    mae DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION,
    snr DOUBLE PRECISION,
    naive_error DOUBLE PRECISION
);

-- Predictions (time series)
CREATE TABLE IF NOT EXISTS olap.fact_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES olap.dim_experiment(experiment_id),
    horizon_id INT NOT NULL REFERENCES olap.dim_horizon(horizon_id),
    ts TIMESTAMP NOT NULL,
    test_close DOUBLE PRECISION,
    target_value DOUBLE PRECISION,
    prediction_value DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION
);

-- Trade metrics (aggregated strategy results)
CREATE TABLE IF NOT EXISTS olap.fact_trade_metrics (
    trade_metrics_id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES olap.dim_experiment(experiment_id),
    horizon_id INT REFERENCES olap.dim_horizon(horizon_id),
    total_profit DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    sortino DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    avg_trade_return DOUBLE PRECISION,
    trades_count INT
);

-- Strategy parameters (JSON configs for strategies)
CREATE TABLE IF NOT EXISTS olap.fact_strategy_params (
    strategy_params_id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES olap.dim_experiment(experiment_id),
    strategy_name TEXT,
    freq TEXT,
    params JSONB
);
