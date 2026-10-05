# VALIDATION_2024 materialisation receipt (lane A, PS1 producer)

Produced 2026-10-05 for FS-CLOSE's declared closure rule (`fs_closure/fs_close/closure_rule.json`,
rule sha256 `8f7a1fc1…`), which scores the frozen K=24 arms once on EXTERNAL VALIDATION (calendar
2024). TEST (2025) was not read: the producer's READ_END for this split is 2025-01-01T00:00Z and
`guard_rows` refuses any decision row at or after it (test: 2025 bytes poisoned and never read).

## Producer

- feature-eng `satoshi/eurusd-ps0-ps1-20261003` @ `cef8b6e` — `tools/eurusd_ps/run_batch.py`,
  `run_batch_cov.py`, `run_batch3.py` with `--split validation_2024`
  (`contract.configure_split`). TRAIN_END stays 2024-01-01; no scaler is fitted by PS1; windows,
  EWMA sigma (halflife 24 bars) and the Kalman q (fitted on TRAIN 2012-05..2023-12) keep their
  TRAIN definitions. Nothing was refit on 2024. A validation run refuses to write outside a
  `validation_2024` directory, so the TRAIN artifacts under `ps1/batch_00N/` were not touched.
- Tests: 13/13 on worker_b (`tests/test_eurusd_ps_laneA.py`), including the new split test.
- Compute: worker_b CPU under `~/.local/bin/crispdm-run`, sequential; probe first, then cap =
  1.25 x measured probe peak under a distinct job name (657M, 325M, 573M). Inputs were relayed
  worker_a -> coordinator -> worker_b with sha256 checked at both ends (2,827 files; EURUSD 5m
  `c746f344…`, archive `5172b322…`, FXMacroData `d8dd8c13…`, regime module `6241ee3e…`).

## Location on worker_b (role name only)

`~/.local/state/canonical_20261003/validation_2024/ps1/batch_00{1,2,3}/` — exactly the paths and
file names `tools/fs_close_manifest.py::launch_closure` probes:
`batch_00{1,2,3}/features_train.parquet` + `batch_001/targets_train.parquet`, each batch with
`READY` and `digests.json`. The three batches were produced in a staging directory and moved into
place atomically only after the schema gate below passed, so the follower never saw a partial set.

## Rows and schema

| item | value |
|---|---|
| decision rows | **6,243** hourly UTC bar ends, 2024-01-01 23:00 .. 2024-12-31 22:00 (market-open hours only) |
| row_id | 0..6242, identical across the three batches; `t_decision_utc` carried in every file |
| features batch_001 / 002 / 003 | 86 / 300 / 24 columns incl. `row_id`, `t_decision_utc` — column names, order and dtypes EQUAL to FS-CLOSE's TRAIN inputs (`fs_pred/input/ps1/…`) |
| targets | same schema as TRAIN: Y_s 1..6 h, Y_l 24..144 h (+staleness), sigma_t, Y_b_s6 / Y_b_l144 with state and time-to-touch |
| censoring | 95 rows whose 144 h support crosses 2025-01-01 are CENSORED (Y_l_144h NaN, Y_b_l144 state CENSORED), never filled |
| Y_b_l144 states | TIMEOUT 3,388 · TP 1,398 · SL 1,362 · CENSORED 95 |
| all-NaN model-input columns in 2024 | 9: `fred.fx_indices.dtwexb.*`, `fred.fx_indices.dtwexm.*`, `fred.stress.tedrate.*` — discontinued FRED series (ended 2019/2020/2022); NaN is the honest value, no imputation by the producer |
| selector-episode-source columns | 37 (batch_001), all NaN in 2024 (archive ends 2021-04); not model inputs |

## Digests (sha256, computed on worker_b at the destination)

```
1ccd6129f3a3846cb7d3745d5b84f9719f0fe1cf1da23417a71d7a570f559a64  batch_001/features_train.parquet
8620b917552bf4c49c49497ef59628a7d16ff5eab13ecd08b207e120fe40f696  batch_002/features_train.parquet
fb14ceeb5e96014cdc017d759d7a287fbe5eddf13b16abefdcb5eccb99a6071c  batch_003/features_train.parquet
9f6827bf5b90ffd4b52c6ad204bb00b6aa25bf7f30eeccb0c3d213f8c7454989  batch_001/targets_train.parquet
b2aac49285776554498c1adf64656d0911b7d1f3582603f97edca1ae666b3d64  batch_001/digests.json
2b7d41c4ca725aeecab0632a6c2ae0845fcd113ca5b5c8be3b6a447df09f2c3f  batch_002/digests.json
7b41b92be917f4b95f3ca3b392e96f9442c055668a73159a79946eb8e5f07df5  batch_003/digests.json
```

The small tables of each batch are committed beside this receipt (`ps1/batch_00N/`); the parquet
arrays stay on worker_b and are bound by `digests.json`.

## PS1 metric matrix on 2024 (catalog only; split=validation_2024; never used for selection)

| batch | features | cells | MEASURED | NOT_APPLICABLE | FAILED | PENDING |
|---|---|---|---|---|---|---|
| batch_001 | 84 (46 model-input, 37 selector-source, 1 quality) | 924 | 583 | 341 | 0 | 0 |
| batch_002 | 298 | 3,278 | 3,193 | 85 | 0 | 0 |
| batch_003 | 22 | 242 | 234 | 8 | 0 | 0 |

NOT_APPLICABLE = no finite 2024 values (selector sources, discontinued FRED series) or fewer than 3
distinct values. Every `profile_cells.csv` row carries `split = validation_2024`.

## Cost (worker_b, CPU only)

| batch | peak RSS | wall | cap used |
|---|---|---|---|
| batch_001 | 525 MiB | 13.7 s | 657M |
| batch_002 | 261 MiB | 15.4 s | 325M |
| batch_003 | 464 MiB | 75.8 s | 573M |

Runtime FS01 (future perturbation / truncation on real bytes): PASS in all three batches.

## Follower pickup

`tools/fs_close_manifest.py` probes the validation files only inside `worker_dispatch` once the
paired refit reaches COMPLETE (`cov["complete"] >= cov["planned"]`); at 05:33Z the refit was
RUNNING (1,073 cells done, plan 143 sets x 14 targets x 5 folds). The follower's exact probe
command, run by hand against worker_b at 05:36Z, returns `OK`. Until the refit completes,
STATUS.json keeps `VALIDATION_2024_FEATURES_AND_TARGETS` in `missing_objects`; that is the
follower's design, not an absence of the inputs. STATUS/MASTER files were not edited.
