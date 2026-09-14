# predictor

Configurable deep-learning experiments for time-series forecasting and classification. predictor
trains, evaluates and optimizes Keras/TensorFlow forecasting and
classification models — ANN, CNN, LSTM, Transformer, TCN, TFT, N-BEATS, MIMO
and binary/direction classifier variants — through a plugin architecture in
which predictor, optimizer, pipeline, preprocessor and target-calculation
plugins are selected by name from JSON configs. Experiments are organized as
numbered phases under [`examples/config/`](examples/config/), each phase a
reproducible sweep over architectures, dataset sizes and horizons.

## Research and current development

This project supports a data-centric research program: characterize the inputs,
test temporal preprocessing, and compare learned representations before scaling
model search. The doctoral proposal investigates modular temporal representations
for forecasting and reinforcement learning. It is a proposal, not a claim that
its hypotheses have already been confirmed.

- **[Current doctoral proposal (PDF)](docs/propuesta_doctoral_representaciones_temporales_modulares.pdf)**
  and [editable LaTeX](docs/propuesta_doctoral_representaciones_temporales_modulares.tex).
- [Research repository map](docs/RESEARCH_STACK.md): data, preprocessing,
  feature engineering, representation learning, optimization and evaluation.
- [Related proposals and earlier formulations](docs/tres_temas_entrevista/README.md).
- [Data governance](https://github.com/harveybc/data-gov): dataset receipts,
  experiment provenance and results in an analytical warehouse.

**Branch scope (2026-09-14):** `master` contains the standalone trainer and
the documents linked above. The larger data-characterization and governed-run
implementation remains on the
[published research snapshot](https://github.com/harveybc/predictor/tree/174cb8869848691e8e720e7f57c1fe938a3976d1).
Its [governed-run guide](https://github.com/harveybc/predictor/blob/174cb8869848691e8e720e7f57c1fe938a3976d1/docs/GOVERNED_RUN.md)
applies to that revision, not automatically to this branch. Research snapshots
are linked by commit so that development cannot silently change a cited implementation.

## Status

**Research software under active development.** predictor is the offline
model-training component. Model serving belongs to
[prediction_provider](https://github.com/harveybc/prediction_provider).
Example results are not evidence of out-of-sample trading profitability.

> **Disclaimer:** all training and evaluation happens offline on historical
> data (simulation/backtest). Model outputs are research artifacts, not
> trading signals; nothing in this repository is financial advice, and no
> real-capital execution happens here.

## Role and non-responsibilities

**Role:** own model definition, training, evaluation and hyperparameter
optimization for time-series prediction, plus the phased experiment configs
and their results.

**Not responsible for:**

- Serving predictions — [prediction_provider](https://github.com/harveybc/prediction_provider)
  hosts trained models behind a FastAPI service.
- Feature/label engineering — [feature-eng](https://github.com/harveybc/feature-eng)
  generates technical-indicator features and direction/oracle labels;
  [feature-extractor](https://github.com/harveybc/feature-extractor) trains
  the autoencoder encoders referenced by some configs.
- Generic CSV preprocessing — the standalone
  [preprocessor](https://github.com/harveybc/preprocessor) application.
- Trading environments or RL agents —
  [gym-fx](https://github.com/harveybc/gym-fx) and
  [agent-multi](https://github.com/harveybc/agent-multi).
- Decentralized optimization infrastructure —
  [doin-node](https://github.com/harveybc/doin-node) (see below).

## Architecture

```
JSON config (examples/config/phase_*/...)
        │
        ▼
app/main.py ── app/cli.py / app/config.py / app/config_handler.py
        │
        ▼
pipeline plugin (pipeline_plugins/)
  ├─ preprocessor plugin (preprocessor_plugins/)  sliding windows, STL, features
  ├─ target plugin       (target_plugins/)        regression / binary / direction targets
  ├─ predictor plugin    (predictor_plugins/)     Keras model build/train/predict
  └─ optimizer plugin    (optimizer_plugins/)     DEAP GA / NEAT hyperparameter search
```

### Phased experiment structure

- [`examples/config/phase_1/`](examples/config/phase_1/) — ANN/CNN/LSTM/
  Transformer sweeps at 1h over dataset sizes from 1 575 to 50 400 bars, with
  an `optimization/` subdirectory; `phase_1_daily/` repeats the sweep at 1d
  and adds TCN+NEAT optimization configs.
- [`examples/config/phase_1b_binary/`](examples/config/phase_1b_binary/) —
  binary entry/exit classifiers (buy/sell × entry/exit) per architecture,
  plus champion inference configs.
- [`examples/config/phase_1c_direction/`](examples/config/phase_1c_direction/)
  — direction classifiers.
- `phase_2` … `phase_4_3` (with `_daily` variants) — progressively deeper
  experiments culminating in Transformer configs at multiple horizons.

Trained champions are kept as `.keras` models with JSON metadata (e.g. a
generated, gitignored `predictor_model_metadata.json` at the repository
root; committed examples live under
[`examples/results/`](examples/results/)) so prediction_provider can load
them.

## Prerequisites

Runtime dependencies are listed in [`requirements.txt`](requirements.txt)
(TensorFlow, tf-keras, tensorflow-probability, numpy, pandas, scipy, DEAP,
pmdarima, PyWavelets, matplotlib, psycopg2-binary, ...).
[`setup.py`](setup.py) intentionally declares only a minimal
`install_requires`; treat `requirements.txt` as authoritative. No
`python_requires` is declared; the platform is exercised in practice on
Python 3.12 (verified below with Python 3.12.13, TensorFlow 2.21.0). A CUDA
GPU is optional but strongly recommended for training.

## Installation

```bash
git clone https://github.com/harveybc/predictor.git
cd predictor
pip install -r requirements.txt
pip install -e .        # installs the `predictor` console script
```

*Unverified in a clean environment* — the commands above are the standard
install; they were not re-executed from scratch for this README. The imports
and CLI below were verified in an existing Python 3.12.13 environment.
The 2026-09-14 publication check used an existing environment and generated
local entry-point metadata with `python setup.py egg_info`; it did not install
or upgrade packages in a shared environment.

## Smallest working example

Verified (cheap) — the CLI parses and prints its full usage:

```bash
PYTHONPATH=. python app/main.py --help
# observed: "usage: main.py [-h] [--x_train_file X_TRAIN_FILE] ..." with the
# full flag list (plugin, epochs, iterations, load/save config, horizons, ...)
```

Bounded CPU example using the bundled daily dataset (completed with exit 0 in
the 2026-09-14 publication check):

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python app/main.py \
  --load_config examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json \
  --epochs 2 --max_steps_train 300 --max_steps_test 300 --mc_samples 2
```

This is a mechanics demonstration, not a validated trading experiment. It writes
the configured sample outputs; use a separate clone and fresh output paths for
your own work. Use long-form `--flags`; short forms do not consistently override
config values. Several 1h configs reference datasets absent from a fresh clone.

[`predictor.sh`](predictor.sh) simply prepends the checkout to `PYTHONPATH`
and runs `python app/main.py`. Training data ship under
[`examples/data/`](examples/data/) (organized by phase) and results are
written under [`examples/results/`](examples/results/); batch drivers live in
[`examples/scripts/`](examples/scripts/).

## Distributed / DOIN usage

predictor is a DOIN *domain*: the external
[doin-plugins](https://github.com/harveybc/doin-plugins) package registers
`predictor` and `binary_predictor` optimization/inference entry points that
wrap this repository, and [doin-node](https://github.com/harveybc/doin-node)
— the unified participant runtime — runs them collaboratively (candidate
leasing, deduplication, champion migration and blockchain persistence are
doin-node's responsibility). predictor always works locally first; DOIN
extends its optimizers, it does not absorb them. The retired
`doin-optimizer`/`doin-evaluator` services are not required. OLAP/ETL helpers
for analyzing experiment databases live under [`olap/`](olap/).

## Configuration and plugins

Configuration is a flat JSON merged over defaults in
[`app/config.py`](app/config.py); every key can also be passed as a CLI flag
([`app/cli.py`](app/cli.py)). Plugins resolve via
[`app/plugin_loader.py`](app/plugin_loader.py) from entry points declared in
[`setup.py`](setup.py):

| Entry-point group | Plugins (this package) |
|---|---|
| `predictor.plugins` | `ann`, `cnn`, `lstm`, `transformer`, `tcn`, `tft`, `n_beats`, `mimo`, plus `binary_*` and `direction_*` variants of each and `binary_logistic`/`direction_logistic` ([`predictor_plugins/`](predictor_plugins/)) |
| `optimizer.plugins` | `default_optimizer` (DEAP GA), `neat_optimizer` ([`optimizer_plugins/`](optimizer_plugins/)) |
| `pipeline.plugins` | `default_pipeline`, `stl_pipeline`, `binary_pipeline`, `direction_pipeline` ([`pipeline_plugins/`](pipeline_plugins/)) |
| `preprocessor.plugins` | `default_preprocessor`, `stl_preprocessor` ([`preprocessor_plugins/`](preprocessor_plugins/)) |
| `target.plugins` | `default_target`, `stl_target`, `binary_target`, `direction_target` ([`target_plugins/`](target_plugins/)) |

**Note on `preprocessor.plugins`:** the preprocessors under
[`preprocessor_plugins/`](preprocessor_plugins/) are local to this repository
but registered into the *shared* `preprocessor.plugins` entry-point group
also used by [gym-fx](https://github.com/harveybc/gym-fx) (which registers
its own `default_preprocessor`) and the standalone
[preprocessor](https://github.com/harveybc/preprocessor) app. Co-installing
those packages mixes the group's contents, so prefer one environment per
application.

## Tests

```bash
python -m pytest tests --collect-only -q
# observed: "3 tests collected, 8 errors in 3.21s"
```

Known limitation of this standalone branch: much of the committed suite predates the
current plugin architecture and fails at import (e.g.
`app.autoencoder_manager`, `load_encoder_decoder_plugins`, `merge_config` no
longer exist). The count above is a recorded baseline observation, not a test
count for the newer research branch. Verified sanity check:

```bash
PYTHONPATH=. python -c "from app.plugin_loader import load_plugin; print('plugin_loader OK')"
# observed: "plugin_loader OK"
```

## Outputs and reproducibility

A run writes: predictions CSV (`output_file`), aggregated metrics
(`results_file`), training/validation loss plot, optional model plot, the
trained model (`save_model`, `.keras`) with metadata JSON, and the fully
merged effective config (`save_config`) from which the run can be reproduced.
Optimization runs additionally write per-generation statistics, best
hyperparameters and a resumable population state under
[`examples/results/`](examples/results/).

## Safety and credentials

Training operates on local CSV files and requires no credentials. The CLI
retains optional remote config/logging flags (`--username`, `--password`,
`--remote_log`); never embed real credentials in configs or commit them.
Database helpers under [`olap/`](olap/) connect to locally provisioned
databases only. Predictions are historical-data research output — not
financial advice and not a live trading system.

## Limitations and migration notes

- The legacy pytest suite is stale (see Tests) — collection errors are
  expected until it is rewritten.
- `setup.py` `install_requires` is minimal; installing without
  `requirements.txt` yields a non-functional environment.
- No `python_requires` or dependency version pins are declared.
- Top-level package names (`app`, `*_plugins`) are shared conventions across
  sibling repositories; use a dedicated environment or run from the checkout
  root (as `predictor.sh` does) so local packages win.
- Some root-level artifacts (champion models, sweep scripts, logs) are
  working files of ongoing campaigns; treat directories under
  [`examples/`](examples/) as the stable interface.

## Related repositories

- [prediction_provider](https://github.com/harveybc/prediction_provider) — FastAPI service serving models trained here
- [feature-eng](https://github.com/harveybc/feature-eng) — feature/label generation upstream of training
- [feature-extractor](https://github.com/harveybc/feature-extractor) — autoencoder encoder/decoder training
- [preprocessor](https://github.com/harveybc/preprocessor) — standalone CSV preprocessing app
- [doin-node](https://github.com/harveybc/doin-node) / [doin-plugins](https://github.com/harveybc/doin-plugins) — distributed collaborative optimization around this domain
- [agent-multi](https://github.com/harveybc/agent-multi) / [gym-fx](https://github.com/harveybc/gym-fx) — RL trading side of the stack

## License

MIT — see [`LICENSE.txt`](LICENSE.txt).
