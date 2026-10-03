# Retsu order: execute PS4 profiles for the ten causal transform features

This is one literal CPU execution assignment. It must produce new measured
profile rows. It is not another inventory, join, design, or status-only task.
Do not touch the running PS3-R process on gamma, any GPU, any service, any
credential, external validation, external test, or live trading.

## 1. Immutable start

1. Fetch `origin`.
2. Create a new worktree and branch from
   `origin/satoshi/canonical-exec-20261003`.
3. Verify that the base contains commit `61c8ea9d` and that
   `docs/audits/evidence/canonical_20261003/ps4_transform_join/REPORT.json`
   says `emitted_features=10`, `ps1_metric_cells=110`, and
   `ps4_expanded_state=PENDING`.
4. Write only under
   `docs/audits/evidence/canonical_20261003/ps4_transform_profile/`.

## 2. Source bytes and custody

The source host is `gamma`. Copy, read-only, these retained files:

- `~/.local/state/canonical_20261003/ps1/batch_003/features_train.parquet`
- `~/.local/state/canonical_20261003/ps1/batch_003/admissible_features.json`
- `~/.local/state/canonical_20261003/ps1/batch_003/digests.json`
- `~/.local/state/canonical_20261003/ps1/batch_003/READY`
- `~/.local/state/canonical_20261003/ps1/batch_001/folds.json`

The required SHA-256 of `features_train.parquet` is
`fd0b4423db991cfb05cd4f9357a378648579d4c11552a1cac413ed711c3520bc`.
Re-hash every copied file on both sides and retain the source/destination
digests. A mismatch stops before profiling. Do not copy run directories or
predictions.

Use `dragon` for the measured execution. The transfer destination is a new
directory under
`~/.local/state/canonical_20261003/ps4_transform_profile/input/`; never write
inside the retained PS1 directories. The transfer is approximately 13 MB.

## 3. Exact measured population

Read the ten non-empty `emitted_feature_id` values from
`ps4_transform_join/transform_feature_join.csv`; do not hard-code a second
list. Require exactly ten unique identifiers and require every identifier to
be a column in the authenticated parquet file.

Use all five folds from the authenticated `folds.json`. For each feature and
fold, the only population is `train_rows=[0, train_end)`. Do not read or emit a
validation block. No target is an input to this task. Expected denominator:
`10 features x 5 folds = 50 feature-fold units`.

Read one parquet column at a time. For every unit call the existing pinned
implementation in `tools/df_profile_information.py` through
`variable_rows(dataset_id, feature_id, x_train, [("train", 0, n_train)])`.
Do not reimplement its estimators and do not call a global-fit shortcut.

Retain all rows it emits, including `COMPLETED`, `INCONCLUSIVE`, `UNAVAILABLE`,
`NOT_RUN`, and `FAILED`. Required families include:

- discrete entropy and lag-1 conditional redundancy/surprisal;
- permutation entropy orders 3, 4 and 5;
- trailing-window spectral entropy;
- zlib/lzma compression descriptors and temporal-structure gain.

Effective rank is a matrix metric and is outside this ten-univariate unit. Do
not fabricate it per feature.

## 4. Implementation and tests

Create only:

- `run_ps4_transform_profiles.py`
- `test_run_ps4_transform_profiles.py`
- generated `profile_rows.jsonl`
- generated `REPORT.json`
- generated `input_digests.json`

All live under the output directory named in section 1.

Tests must fail before implementation for, then cover:

1. wrong parquet digest;
2. absent or duplicated emitted feature;
3. a feature absent from parquet;
4. absent, overlapping, nonchronological, or out-of-range fold boundaries;
5. any partition name other than literal `train`;
6. mutation of rows at or after a fold's `train_end` changing that fold;
7. duplicate `(feature, fold, metric, estimator)` output identities;
8. missing one of the 50 expected units;
9. non-finite `COMPLETED` values;
10. deterministic replay producing byte-identical outputs.

Each output row must bind feature, fold, train row interval, source digests,
profiler code digest, metric, estimator, status, reason, value and CPU seconds.
The report must count every status and prove the 50-unit denominator. It must
say `NO_TARGET_READ`, `NO_OUTER_VALIDATION_READ`, `NO_TEST_READ`, and
`NO_FEATURE_SELECTION_DECISION`.

## 5. Execute, do not merely prepare

Run focused tests first. Then execute the real 50-unit job on dragon under one
CPU-only governed scope:

- `CUDA_VISIBLE_DEVICES=""`
- memory ceiling: 4 GiB;
- wall ceiling: 7,200 seconds;
- one process, one BLAS thread;
- no retry at a lower evidence population.

If the full job exceeds the wall limit, retain completed units atomically and
resume only missing units with the same identities. Never recompute a completed
unit. A resource stop is a measured cost result, not permission to shrink rows,
metrics, folds, or features.

Run the generator a second time after completion and require byte-identical
`profile_rows.jsonl`, `REPORT.json`, and `input_digests.json` except that runtime
measurements must live in a separate attempt receipt and therefore cannot alter
scientific output bytes.

## 6. Return

Commit and push the new branch. Return:

- branch and commit;
- focused test count;
- 50/50 completion or explicit missing unit identities;
- metric rows and counts by status;
- CPU/wall/RSS cost from the governed scope;
- exact input/output digests;
- confirmation that gamma PS3-R was untouched;
- `NO_NEW_MODEL_MEASUREMENT` and `NO_FEATURE_SELECTION_DECISION`.

Do not edit the old PS1 files or the prior PS4 join. This is a successor
measurement that names them by digest.
