# Research software and evaluation guide

Harvey Demian Bastidas Caicedo's repositories separate data preparation, model
learning, distributed optimization and execution. Each component can be
inspected independently; cloning one repository does not provision all datasets,
trained models, services or dependencies of the others.

## Start here

- [Current doctoral proposal: modular temporal representations](propuesta_doctoral_representaciones_temporales_modulares.pdf).
- [Editable source](propuesta_doctoral_representaciones_temporales_modulares.tex).
- [Alternative proposals and historical formulations](tres_temas_entrevista/README.md).
- [Original master's research and DOIN history](https://github.com/harveybc/doin-core/blob/master/docs/THESIS.md).

The doctoral work is a proposed experimental program. Existing code is a
starting point and a source of reproducible engineering evidence, not proof
of the proposed hypotheses. Negative results and unresolved limitations are
part of that evidence.

## Components

The repositories below are public except `doin-domains`, which requires
repository access. Its restricted visibility is intentional and was not changed
by this publication pass; evaluators can inspect the public `doin-plugins`
interfaces without access to it.

| Stage | Repository | Responsibility |
|---|---|---|
| Data inventory | [financial-data](https://github.com/harveybc/financial-data) | Data dictionaries, provenance, inventory and file-lake adapter; restricted datasets are not redistributed |
| Governance | [data-gov](https://github.com/harveybc/data-gov) | One policy and accounting system for lakes and analytical warehouses; governed dataset deliveries and experiment outcomes |
| Signal preparation | [preprocessor](https://github.com/harveybc/preprocessor) | Configured transforms, train-only fitting and dataset partitions; causal-transform research is versioned separately |
| Feature engineering | [feature-eng](https://github.com/harveybc/feature-eng) | Technical indicators and supervised labels; target construction is distinct from an input feature |
| Learned features | [feature-extractor](https://github.com/harveybc/feature-extractor) | Autoencoder training and encoder/decoder artifacts |
| Forecasting | [predictor](https://github.com/harveybc/predictor) | Configurable Keras/TensorFlow training and evaluation |
| RL experiments | [agent-multi](https://github.com/harveybc/agent-multi) / [gym-fx](https://github.com/harveybc/gym-fx) | Agents, optimizers, environments and offline evaluation |
| Distributed research | [doin-core](https://github.com/harveybc/doin-core) / [doin-node](https://github.com/harveybc/doin-node) | Shared protocol and participant runtime |
| Domain adapters | [doin-plugins](https://github.com/harveybc/doin-plugins) / [doin-domains (private)](https://github.com/harveybc/doin-domains) | Integration plugins and non-financial benchmark domains |
| Serving | [prediction_provider](https://github.com/harveybc/prediction_provider) | Prediction API and model adapters, separate from training |
| Simulation and execution | [heuristic-strategy](https://github.com/harveybc/heuristic-strategy) / [lts](https://github.com/harveybc/lts) | Backtests and configured execution adapters; inspection does not require broker access |
| Synthetic controls | [synthetic-datagen](https://github.com/harveybc/synthetic-datagen) / [timeseries-gan](https://github.com/harveybc/timeseries-gan) | Controlled signal generation and generative-model experiments |

## What an evaluation should distinguish

Current infrastructure work: [mandatory creation and adoption of the reusable
lake and warehouse hosts](integracion_workplan_2026_09_10/09_ADOPCION_DATA_LAKE_DATA_WAREHOUSE_2026_09_14.md).
These new repositories are assigned for implementation, not presented as
already deployed. Existing services remain in use during the tested transition.

1. **Implemented software:** source, interfaces, dependencies and tests.
2. **Demonstrated integration:** a recorded run on a named code/data version.
3. **Scientific evidence:** a declared hypothesis, suitable comparisons and
   genuinely held-out evaluation. Successful integration does not imply this.
4. **Deployment:** the exact version running in an environment, not whichever
   branch currently appears on GitHub.

Use isolated Python environments because several applications expose the same
top-level `app` package or plugin namespaces. Start with CLI help and small
offline fixtures. Do not launch a sweep, connect to a broker, reset a database
or treat historical sample predictions as trading recommendations.

Feature engineering, preprocessing and representation learning have different
roles. A useful transform must preserve the prediction-time information boundary;
centered filters, future-derived labels and full-series normalization cannot
silently become real-time inputs. Input hashes establish provenance, not
causality, dataset rights or predictive usefulness.

## Publication scope

This documentation update makes the project and its research material visible
on default branches. It does not bulk-merge experimental campaign branches.
Each repository README distinguishes published components from linked research
snapshots. For an exact reproduction, use the cited commit and its own setup
instructions, data contract and test report. See the
[publication record](PUBLICATION_2026_09_14.md) for this integration pass.
