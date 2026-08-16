# AGENTS.md — predictor

Guidance for AI coding agents working in this repository. See [agents.md](https://agents.md).

## Project overview

predictor trains, evaluates and optimizes Keras/TensorFlow models for financial
time-series forecasting and classification. Model families (ANN, CNN, LSTM,
Transformer, TCN, TFT, N-BEATS, MIMO, plus binary and direction classifier
variants) are selected by name from flat JSON configs, as are the pipeline,
preprocessor, target and optimizer components — everything resolves through
setuptools entry points declared in `setup.py`. Experiments are organized as
numbered phases under `examples/config/`, and each run writes predictions,
per-horizon metrics, plots and the fully merged effective config.

It does **not** serve predictions (that is `prediction_provider`), generate
features or labels (`feature-eng`, `feature-extractor`), run trading
environments or RL agents (`gym-fx`, `agent-multi`), or provide the distributed
optimization runtime (`doin-node`). All training and evaluation is offline on
historical CSV files; nothing here executes trades.

## Agent quickstart (install → run → show the user results)

Verified end to end on 2026-08-16 with Python 3.12.13 and TensorFlow 2.21.0.

### 1. Environment

No `python_requires` is declared; the platform is exercised on Python 3.12.

```bash
conda create -n predictor python=3.12 -y && conda activate predictor
# or: python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`setup.py` declares a deliberately minimal `install_requires`; treat
`requirements.txt` as authoritative. Installing without it yields a
non-functional environment.

> Unverified: the install was not re-executed from a clean environment for this
> document. Everything below was verified in an existing Python 3.12.13
> environment that already satisfied `requirements.txt`.

### 2. Smoke test (fastest proof the code works)

```bash
PYTHONPATH=. python app/main.py --help
PYTHONPATH=. python -c "from app.plugin_loader import load_plugin; print('plugin_loader OK')"
```

Both verified. The pytest suite is **stale** and is not a usable smoke test:

```bash
python -m pytest tests --collect-only -q
# observed: "3 tests collected, 8 errors in 3.24s"
```

Most committed tests predate the current plugin architecture and fail at import
(`app.autoencoder_manager`, `load_encoder_decoder_plugins`, `merge_config` no
longer exist). Do not treat a red suite here as a regression you introduced.

### 3. Representative run

`examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json` — a small daily
ANN whose data files are present in the checkout. The config ships
`epochs: 10000`, so override on the command line for a fast run:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python app/main.py \
  --load_config examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json \
  --epochs 2 --max_steps_train 300 --max_steps_test 300 --mc_samples 2
```

Verified: exit 0, ~12 s on CPU. Writes into `examples/results/phase_1_daily/`:
`*_results.csv` (per-horizon metrics), `*_prediction.csv`, `*_uncertanties.csv`,
`*_loss_plot.png`, `*_predictions_plot.png`.

Three cautions, all verified:

- **`CUDA_VISIBLE_DEVICES=""` is deliberate.** This machine may be running GPU
  training jobs. Force CPU unless the user explicitly asks for the GPU.
- **This overwrites committed sample outputs.** Restore with
  `git checkout -- examples/results/phase_1_daily/`.
- **Use long-form CLI flags only.** `app/config_merger.py` merges CLI values by
  scanning `sys.argv` for tokens starting with `--`, so short flags are parsed
  by argparse and then silently dropped during the merge: `--epochs 2` overrides
  the config file, `-e 2` does not.

Drop the overrides to reproduce the real experiment (long: `epochs: 10000` with
`early_patience: 100`).

Not every config is runnable: of 224 configs referencing data files, **137 have
all their inputs present and 87 do not**. The `examples/config/phase_1/` (1h)
configs point at `examples/data/phase_1/normalized_d*.csv`, which are absent
from the checkout — only `base_d*.csv` ship. Check the referenced paths exist
before picking a config.

### 4. OLAP: load results into the cube

The analytics layer is PostgreSQL. `olap/` holds an idempotent schema
initializer, an ETL that loads a run's config + metrics CSV, and a truncate
helper. Connection comes from the standard `PG*` environment variables
(`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`); defaults are in
`olap/init_db.py`.

```bash
# 4a. Postgres (skip if one is already running)
docker run -d --name predictor-pg -p 5432:5432 \
  -e POSTGRES_USER="$PGUSER" -e POSTGRES_PASSWORD="$PGPASSWORD" \
  -e POSTGRES_DB=predictor_olap postgres:16

# 4b. Create the star schema and seed dimensions (idempotent)
python olap/init_db.py

# 4c. KNOWN BUG — apply before the first ETL run on a fresh database
psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" \
  -c "ALTER TABLE public.fact_performance ADD COLUMN IF NOT EXISTS metric_value DOUBLE PRECISION;"

# 4d. Load one experiment
python olap/etl_migrate_v2.py \
  --project-key predictor \
  --phase-key phase_1_daily \
  --experiment-key ann_1575_1d \
  --experiment-config examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json \
  --results-csv examples/results/phase_1_daily/phase_1_ann_1575_1d_results.csv
```

Verified end to end against a throwaway database: 90 rows written, 0 skipped.

**About step 4c.** `olap/etl_migrate_v2.py` fails on a database freshly created
by `olap/init_db.py`. Its `ensure_schema_and_tables()` runs a legacy backfill,
`UPDATE public.fact_performance SET metric_value = avg_value ...`, inside the
same transaction without a savepoint. `init_db.py` never creates a
`metric_value` column, so the statement raises `UndefinedColumn`, which aborts
the whole Postgres transaction; the surrounding `try/except` swallows the Python
exception but cannot un-abort the transaction, so every following statement dies
with `InFailedSqlTransaction` and the ETL exits non-zero. Adding the legacy
column first makes the backfill a no-op and the ETL succeeds. The proper fix is
to wrap those migration statements in `conn.begin_nested()` savepoints, as the
row loader already does.

The ETL parses the `Metric` column of the results CSV with the regex
`^(Train|Validation|Test)\s+(.+?)\s+H(\d+)$`, so rows must look like
`Train MAE H9`. Runs of this repo produce exactly that format.

Schema loaded by `init_db.py` / `etl_migrate_v2.py`, all in schema `public`:

| Table | Grain |
|---|---|
| `dim_project` | project |
| `dim_phase` | phase, FK to project |
| `dim_experiment` | experiment; full `config_json` JSONB plus ~30 extracted columns (`predictor_plugin`, `learning_rate`, `window_size`, `max_steps_train`, …) |
| `dim_dataset_split` | `train` / `validation` / `test` |
| `dim_horizon` | horizon number |
| `dim_metric` | metric key, type, and whether higher is better |
| `fact_performance` | experiment × phase × split × horizon × metric → avg/std/min/max |
| `fact_results_summary` | experiment × metric → avg/std/min/max |

> **Two incompatible schemas live in `olap/`.** `init_db.py`, `etl_migrate_v2.py`
> and `reset_olap.py` target schema **`public`** with the tables above.
> `olap_schema_postgres.sql`, `olap_metabase_helpers.sql` and
> `olap_schema_patch_experiment_key.sql` define a *different*, unused design in
> schema **`olap`** (`dim_dataset`, `fact_experiment_metrics`,
> `fact_predictions`, `fact_trade_metrics`, `fact_strategy_params`, with
> `experiment_id` surrogate keys and a `config` column). Verified: only the
> `public` schema is deployed. Do not apply the `.sql` files expecting the
> helper view `olap.v_experiment_config_kv` to work against the ETL's output —
> it references columns the ETL does not create.

### 5. Metabase

```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```

Open `http://localhost:3000`, complete the first-run setup wizard (creates the
admin account), then **Add a database → PostgreSQL** pointing at the
`predictor_olap` database with the same `PG*` values.

Networking caveat: if Postgres also runs in Docker, `localhost` inside the
Metabase container refers to the container itself. Put both on one user-defined
Docker network and use the Postgres **container name** as the host, or start
Metabase with `--network host`. Verified on this machine: both containers share
a user-defined network and Metabase reaches Postgres by container name.

### 6. Final message to give the user

> Results are in two places. Raw run outputs are under
> `examples/results/phase_1_daily/` — `phase_1_ann_1575_1d_results.csv` has the
> per-horizon MAE / R2 / SNR / Uncertainty table, and
> `phase_1_ann_1575_1d_predictions_plot.png` plots predictions against actuals.
> The aggregated cube is in Metabase at **http://localhost:3000** (database
> `predictor_olap`).
>
> First query to try — rank model architectures by validation AUC-ROC across
> training-set sizes:
>
> ```sql
> SELECT e.predictor_plugin,
>        e.max_steps_train                   AS train_bars,
>        round(avg(f.avg_value)::numeric, 4) AS mean_val_auc_roc,
>        count(*)                            AS n_experiments
> FROM fact_performance f
> JOIN dim_experiment  e USING (experiment_key)
> WHERE f.split_key  = 'validation'
>   AND f.metric_key = 'AUC_ROC'
> GROUP BY e.predictor_plugin, e.max_steps_train
> ORDER BY mean_val_auc_roc DESC;
> ```
>
> Swap `'AUC_ROC'` for `'MAE'` or `'R2'` for the regression phases, and add
> `f.horizon_key` to the SELECT and GROUP BY to see how accuracy decays with
> forecast horizon.

That query was executed against the populated cube and returns a ranked
architecture table. Note it only returns rows for classification phases, which
are the ones that emit `AUC_ROC`; regression phases emit `MAE`, `R2`, `SNR`,
`Uncertainty`, `Naive_MAE`.

## Build, test and lint commands

```bash
pip install -r requirements.txt          # runtime dependencies (authoritative)
pip install -e .                         # installs the `predictor` console script
python -m pytest tests --collect-only -q  # 3 collected, 8 errors — stale suite
PYTHONPATH=. python app/main.py --help    # CLI surface
sh predictor.sh --load_config <config>    # prepends checkout to PYTHONPATH, runs app/main.py
```

There is no configured linter, formatter, type checker or CI workflow in this
repository. Do not claim the code is lint- or type-clean.

## Layout

| Path | Purpose |
|---|---|
| `app/` | CLI, config defaults/merge, plugin loader, `main.py` entry point |
| `predictor_plugins/` | Keras model plugins; `binary/` and `direction/` subpackages hold classifier variants |
| `pipeline_plugins/` | Run orchestration: `default`, `stl`, `binary`, `direction` |
| `preprocessor_plugins/` | Sliding windows, normalization, STL decomposition |
| `target_plugins/` | Regression, binary and direction target construction |
| `optimizer_plugins/` | DEAP genetic search and NEAT hyperparameter optimization |
| `examples/config/` | Phased experiment configs — the stable interface |
| `examples/data/`, `examples/data_downsampled/` | Training CSVs by phase |
| `examples/results/` | Committed sample outputs per phase |
| `examples/scripts/` | Batch drivers for sweeps |
| `olap/` | Postgres star schema, ETL, reset helper, Metabase SQL |
| `tests/` | Legacy pytest suite, mostly stale |
| `tools/` | One-off utilities (`oom_smoke_test.py`) |

## Conventions and constraints

- **Config-driven.** A run is a flat JSON file merged over `app/config.py`
  defaults. Precedence, from `app/config_merger.py`: plugin params → defaults →
  file config → CLI args. CLI wins, but only for long-form `--flags`.
- **Plugin architecture.** Components resolve by name through entry-point groups
  declared in `setup.py`: `predictor.plugins`, `optimizer.plugins`,
  `pipeline.plugins`, `preprocessor.plugins`, `target.plugins`. Adding a model
  means adding a plugin module and an entry point, then reinstalling with
  `pip install -e .` — not editing the pipeline.
- **Shared entry-point namespace.** `preprocessor.plugins` is also used by
  `gym-fx` and the standalone `preprocessor` app, which register their own
  `default_preprocessor`. Co-installing those packages mixes the group. Use one
  environment per application.
- **Top-level package names** (`app`, `*_plugins`) are shared conventions across
  sibling repositories. Run from the checkout root, as `predictor.sh` does, so
  local packages win.
- **Reproducibility.** Every run writes its fully merged effective config to
  `save_config` (default `./config_out.json`); a run can be replayed from it.
- **No credentials.** Training reads local CSVs only. The CLI retains optional
  `--username` / `--password` / `--remote_log` flags — never commit real values.

## Do not touch

- **Running processes.** GPU training workers and the Docker Postgres/Metabase
  containers may be live. Never start, stop or restart them, and never launch a
  training sweep. Force CPU with `CUDA_VISIBLE_DEVICES=""` for verification runs.
- **The populated OLAP database.** `olap/reset_olap.py --yes` truncates every
  dimension and fact table. Do not run it against a database holding real
  results; create a throwaway database instead.
- **Committed sample outputs** under `examples/results/`. If a verification run
  overwrites them, restore with `git checkout -- examples/results/`.
- **Generated artifacts** at the repo root — `predictor_model.keras`,
  `pretrained_model.keras`, `*_metadata.json`, `config_out.json`, `*.log`,
  `confusion_matrix.png`, `roc_pr_curves.png`. These are gitignored working
  files of ongoing campaigns; do not commit or curate them.
- **Per-host sweep scripts** at the repo root are operator-specific working
  files pinned to particular machines. Do not run them.
- **Sibling repositories.** Changes to `prediction_provider`, `feature-eng`,
  `doin-node` and friends belong in those repositories.
- **Secrets in a public repo.** Never write account identifiers, broker
  credentials, private IP addresses or machine host names into files here. Use
  placeholders such as `<your-host>`.
