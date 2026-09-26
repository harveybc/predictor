### N1. The block as sealed

* design `47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea`, schema `df_e1_block_design.v1`, block `Q2_CONTEXT_BOUNDED`, state at seal `SEALED_NOT_EXECUTED`, phase `DEVELOPMENT`
* tier: TIER1: RP66-RP73 blocks: patience 3 events (600 non-improving updates)
* prepared data `43c93db381de4e4281b0b00ae1b577e3f860ad29a477b13ac6dc86e2bbed5350`, panel rows 1410981..1462761 (pad 1380), common evaluation **10020 origins** (panel rows 41700..51719), sigma_evaluation 0.9125164391265214 kW
* train population `COMMON_INTERSECTION`: **38700 origins, identical for every arm**; subset of the source run's train origins: True; the 28 d baseline enumeration reproduces the source's train origins: True; the common evaluation equals the source's: True
* recipe: mae loss, adam, lr 0.003, batch 64, ceiling 4000 updates, validation every 200 observed updates, patience 3 events, restore_best True, min_delta 0.0
* scaler rule: COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels mean 0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd
* common evaluation rule: the intersection over the block's arms of admissible validation origins, with a finite label and a finite daily lookup, derived at prepare BEFORE any score; every arm scores on it

| arm | window | features | dilations | crop | role | parameters | per-arm train admissible before the intersection |
|---|---:|---|---|---:|---|---:|---:|
| `modular_w60` | 60 | base | — | — | ARM | 8127 | 40080 |
| `daily_lag` | 60 | daily_lag | — | — | ARM | 8208 | 40020 |
| `long_window_crop60` | 1440 | base | — | 60 | EXACT_INFORMATION_NULL: the raw input is cropped to its last 60 rows before the extractor | 8127 | 38700 |
| `short_window_deep_core` | 60 | base | [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] | — | ARM | 12047 | 40080 |

**Reading rules, verbatim from the sealed design:** *three seeds on one task are development evidence* · *a fit that reached the update ceiling is CENSORED wherever its best checkpoint fell* · *no cell is removed after its score is seen* · *a published number under another protocol never enters the comparison column*

**The block's own question, as sealed:** the part of the Q2_CONTEXT question a 30 GiB host can fit: does a causal daily-lag channel, or a deeper dilated core at W60, change error against the W60 baseline and against the W1440 exact-crop information null? The two W1440 FULL-DEPTH arms of Q2_CONTEXT (long_window_own_depth, long_window_local_support_67) are NOT in this block, so this block does NOT separate context from depth and does NOT answer the Q2_CONTEXT question: it measures the three arms whose retained cost pilots fit this host, and leaves the long-window treatment unmeasured

**Why the two W1440 full-depth arms of Q2_CONTEXT are not in this block, verbatim from the sealed design:** RESTRICTED BEFORE ANY SCORE from the arms of Q2_CONTEXT v1 (design 6d1aaecaf27c581c709a745a4f976c2e9dcc05594815b2ee1a9747595f4398b1, state BUDGET_LIMITED_BEFORE_ANY_OUTCOME, zero cells fitted), on the two RETAINED cost pilots' own measurements and on nothing else: long_window_own_depth 4.165 CPU s per update and 8 458 399 744 B peak RSS, long_window_local_support_67 4.443 CPU s per update and 10 279 276 544 B peak RSS, i.e. 17 507 s and 18 406 s per cell at the 4 000-update ceiling and a resident set that would take this host's memory away from its owner; the three arms kept cost 287 s, 312 s and 488 s per cell at the ceiling with peak RSS under 1 GiB. The restriction is a resource declaration made before any score of any cell existed, never a removal after a score was seen (reading rule 3); no arm, seed, recipe, scaler, row, cadence or ceiling of Q2_CONTEXT v1 is otherwise changed

### N2. The cost projection the block executed on

* four cost pilots spent 82.4 CPU s; projection at the 4 000-update ceiling 2654.3 CPU s, with 25 % headroom 3317.9 CPU s; campaign ceiling 14400 CPU s, closure reserve 2000 CPU s
* decision **EXECUTE** (`fits_the_ceiling`: True) — contrast with Q2_CONTEXT v1's own decision, `BUDGET_LIMITED_BEFORE_ANY_OUTCOME`

| arm | CPU s per update (pilot) | peak RSS GiB (pilot) | projected CPU s per cell at the ceiling |
|---|---:|---:|---:|
| `modular_w60` | 0.0390 | 0.87 | 179.8 |
| `daily_lag` | 0.0392 | 0.87 | 180.9 |
| `long_window_crop60` | 0.0410 | 0.87 | 197.8 |
| `short_window_deep_core` | 0.0710 | 0.90 | 326.3 |

### N3. Every fit, as it landed

Errors recomputed here from each cell's retained `arrays.npz`, each one cross-checked against the value the cell record stored (a disagreement above 1e-12 refuses the whole table).

| cell | arm | seed | MAE_z | MAE kW | naive MAE kW, same rows | skill vs naive | worse than naive | stop | censoring | updates | best update | CPU s | peak RSS GiB | reload max err | fresh-process replay |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| `modular_w60_s1` | `modular_w60` | 1 | 0.552467 | 0.504135 | 0.617372 | 0.183418 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 54.1 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s1` | `daily_lag` | 1 | 0.539528 | 0.492328 | 0.617372 | 0.202542 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1600 | 1000 | 50.6 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s1` | `long_window_crop60` | 1 | 0.552467 | 0.504135 | 0.617372 | 0.183418 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 62.3 | 0.95 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s1` | `short_window_deep_core` | 1 | 0.552868 | 0.504501 | 0.617372 | 0.182824 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1400 | 800 | 93.7 | 0.99 | 0.00e+00 | allclose(1e-6) PASS |
| `modular_w60_s2` | `modular_w60` | 2 | 0.537390 | 0.490377 | 0.617372 | 0.205702 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 3000 | 2400 | 82.5 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s2` | `daily_lag` | 2 | 0.537977 | 0.490913 | 0.617372 | 0.204834 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 2000 | 1400 | 62.6 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s2` | `long_window_crop60` | 2 | 0.537390 | 0.490377 | 0.617372 | 0.205702 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 3000 | 2400 | 95.9 | 0.95 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s2` | `short_window_deep_core` | 2 | 0.548908 | 0.500888 | 0.617372 | 0.188677 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 120.1 | 0.99 | 0.00e+00 | allclose(1e-6) PASS |
| `modular_w60_s3` | `modular_w60` | 3 | 0.550991 | 0.502789 | 0.617372 | 0.185599 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 59.0 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s3` | `daily_lag` | 3 | 0.547895 | 0.499963 | 0.617372 | 0.190176 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 60.7 | 0.94 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s3` | `long_window_crop60` | 3 | 0.550991 | 0.502789 | 0.617372 | 0.185599 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 66.3 | 0.95 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s3` | `short_window_deep_core` | 3 | 0.549280 | 0.501227 | 0.617372 | 0.188128 | no | EARLY_STOPPING | STOPPED_ON_VALIDATION | 1800 | 1200 | 122.4 | 0.99 | 0.00e+00 | allclose(1e-6) PASS |

Censoring across the twelve fits: `STOPPED_ON_VALIDATION`. Initial-weight digests, per seed: seed 1: `modular_w60` b9caefebdc3c…, `daily_lag` 0f6b46da2c89…, `long_window_crop60` b9caefebdc3c…, `short_window_deep_core` 9d7edff6f248… · seed 2: `modular_w60` 84f67bc65cb5…, `daily_lag` f9c474649f56…, `long_window_crop60` 84f67bc65cb5…, `short_window_deep_core` 1a6308265c88… · seed 3: `modular_w60` e0e2765cded8…, `daily_lag` 7fc47852dea4…, `long_window_crop60` e0e2765cded8…, `short_window_deep_core` 59e492f0e56f…

### N4. Per arm, and the paired difference against the baseline arm

| arm | mean MAE_z | sd (ddof 1) | mean MAE kW | mean skill vs naive | paired Δ MAE_z vs `modular_w60`, per seed | mean Δ | signs (+ / −) |
|---|---:|---:|---:|---:|---|---:|---|
| `modular_w60` | 0.546949 | 0.008311 | 0.499100 | 0.191573 | — (this is the baseline arm) | — | — |
| `daily_lag` | 0.541800 | 0.005335 | 0.494401 | 0.199184 | -0.012938 · +0.000587 · -0.003097 | -0.005149 | 1 / 2 |
| `long_window_crop60` | 0.546949 | 0.008311 | 0.499100 | 0.191573 | +0.000000 · +0.000000 · +0.000000 | 0.000000 | 0 / 0 |
| `short_window_deep_core` | 0.550352 | 0.002187 | 0.502205 | 0.186543 | +0.000402 · +0.011518 · -0.001711 | 0.003403 | 2 / 1 |

Negative Δ = smaller error than the baseline arm. Three paired seeds on one task, one previously inspected DEV validation week: **development evidence**, as the sealed reading rule says. Both signs are printed and **no interval is claimed from n = 3**. A difference between two forecast errors is not a verified causal effect: no causal claim here is verified against a retained-row error, because a causal claim does not predict a retained row.

**The exact-information null, measured.** `long_window_crop60` is the W1440 input cropped to its last 60 rows before the extractor: by RP87 it is the SAME computation as `modular_w60` on the same origins. Measured here rather than assumed: identical initial-weight digest in 3 of 3 seeds, and MAE equal to the baseline's to the last bit in 3 of 3 seeds. A null that reproduces its treatment exactly is the block's own positive control on its plumbing.

### N5. The three declared references, on the same rows

Computed by `df_e1_block.baselines` on the block's 10020 common evaluation origins. The block's closure suppresses these when it fails, so they are published separately.

| reference | definition | MAE kW | MAE_z | skill vs persistence |
|---|---|---:|---:|---:|
| `persistence` | y(t) | 0.617372 | 0.676560 | 0.000000 |
| `daily_seasonal` | y(t+h-1440) | 0.731659 | 0.801804 | -0.185119 |
| `train_constant` | mean of the train labels of the 28 d tier; computed on the common evaluation set at closure, no fit, no terminal | 0.709950 | 0.778013 | -0.149955 |

Published exactly as they landed: `daily_seasonal`, `train_constant` are **worse** than persistence on these rows.

### N6. The owner closure table as it landed

`owner_closure_table.v2` from `tools/df_closure_table.py`, generated 2026-09-26T12:24:14Z: **12 rows, 0 verified, 0 preserved with a qualified scope**; custody classes {"UNCHECKED": 12}; preparation classes {"PREPARATION_LOCAL_ONLY": 12}.

**This is the load-bearing fact about this run, and it is printed before any number of mine.** The verifier's policy is that a score with **no accepted terminal receipt is not reported as a model error at all** — not as a qualified one. This host holds no data-gov service key, so no terminal was ever accepted, so every error column below is `null` and every custody is `UNCHECKED`. Published exactly as it landed.

| unit | task / horizon / split | metric and scale | model error | naive error | skill | literature value + source | placed in the comparison column | comparability | custody | binding |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| `modular_w60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |

**Why NOT_COMPARABLE, verbatim:** unknown identity fields cannot match as proof: ['target_transform']

**Planned matched comparison, verbatim:** read the primary source (or its code) and fill the field; a placeholder is not a protocol

**Every problem the table recorded, in full:**

* Q2_CONTEXT_BOUNDED_20260926/daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_BOUNDED_20260926/short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt

### N7. The unanchored measurement table

`unanchored_measurement_table.v1`. It supplies the five columns the owner's rule names, for a run the owner closure table can only print as `null`. Each error is recomputed from the cell's own retained arrays and cross-checked against the record's stored score; the contract columns are taken from the landed owner table above. **It is not an `owner_closure_table.v2` row, it is never verified, and its custody is `UNANCHORED_NO_ACCEPTED_TERMINAL` in every row.** Nothing may promote, select or rank on it.

| unit | metric and scale | model error | naive error, SAME rows | skill | rows (model / naive) | horizon (model / naive) | scale (model / naive) | literature value + source | comparability | custody |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `modular_w60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.504135 | 0.617372 | 0.183418 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.492328 | 0.617372 | 0.202542 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.504135 | 0.617372 | 0.183418 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.504501 | 0.617372 | 0.182824 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.490377 | 0.617372 | 0.205702 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.490913 | 0.617372 | 0.204834 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.490377 | 0.617372 | 0.205702 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.500888 | 0.617372 | 0.188677 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.502789 | 0.617372 | 0.185599 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.499963 | 0.617372 | 0.190176 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.502789 | 0.617372 | 0.185599 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.501227 | 0.617372 | 0.188128 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |

**Fits that landed worse than their naive reference: 0 of 12.** Every fit's skill is printed above whatever its sign.

### N8. The closure as it landed

* `df_e1_block_report.v3`, design `47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea`, block `Q2_CONTEXT_BOUNDED`
* **`verified`: False**
* common evaluation rows 10020; sigma_evaluation 0.9125164391265214; spent CPU 1012.7 s
* closure code drift: `{"df_e1_block.py": {"sealed": "24fdb67fca3c98781de1c1b56f69ff3dce02d8469f48e3de5c39991d1440d97f", "now": "0708a56f71e9e997d52490cd3fa14a487fae44a5172017b5788809455c575a99"}}`
* scope: DEVELOPMENT; paired seeds within host blocks; one previously inspected DEV validation week; no test rows read
* disposition: `{"disposition": "HISTORICAL_DEV_ONLY", "task_id": "uci_235.W60_h60.DEV_28d_7d", "policy": "docs/tres_temas_entrevista/program_v3/SOTA_FIRST_2026_09_21.md", "why": "previous exploratory pilot (household task / adapted models): preserved with its receipts and failures, excluded from active selection, ranking and recommendations"}`; active_selection: `null`
* `summary`, `paired` and `baselines` are all `None`: a failed closure emits no verified comparator, no paired contrast and no selected arm. The per-arm means and the paired differences in N4, and the references in N5, are therefore published OUTSIDE the closure, recomputed from the arrays.

**Every problem the closure recorded, in full:**

* daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt
* closure without a warehouse read: no accepted custody, nothing is verified

### N9. Governance, stated as it is

* classification **NON_GOVERNING**
* data_gov_acquisition: **ABSENT**
* accepted_terminal: **ABSENT**
* receipt: **ABSENT**
* warehouse_read: **ABSENT**
* why: **no data-gov service key is held on this host; the governed runner refuses before opening data, so this driver ran instead and its results promote nothing**
* every closure row's custody is UNCHECKED: no accepted payload anchors the score
* the preparation's custody is PREPARATION_LOCAL_ONLY
* the closure's `verified` flag is False for that reason and for no other unless it names one
* these numbers are DEVELOPMENT measurements and cannot select, rank or promote anything
* data custody `BYTES_IDENTITY_ONLY`, panel sha256 `b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db`, 10890295 bytes — the file's sha256 equals the design's source_run.panel_sha256, which a previous GOVERNED acquisition recorded as VERIFIED_TRANSFER; this driver re-verified the BYTES, not the transfer, and holds no delivery id, no availability contract and no acceptance
* interpreter: Python 3.12.13 (anaconda env `trading-stack`)

