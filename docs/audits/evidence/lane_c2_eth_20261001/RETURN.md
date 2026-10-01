# Lane C2 return: PS3-C over the ETH 4h population, PS3-R support -- DEVELOPMENT

Satoshi's lane C2 (front C of `SATOSHI_AUTONOMOUS_PARALLEL_EXECUTION_2026_10_01.md`). 2026-10-01. Every number below is
**DEVELOPMENT**: the population is a git-pinned local file with undeclared timestamp semantics and no governed
availability contract. Nothing is confirmatory. Validation (2024) and test (2025) rows were never read.

## 0. What was bound (digests first)

| object | binding | digest |
|---|---|---|
| population | predictor `b1f8a74f` `examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`, 18,085 rows, 90 columns | `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f` |
| features | variant A, 83 features in `all_admissible_control` order of M03's declaration | declaration `f3c0beca…`; manifest file `fdff0c85…` (feature-eng `b90c4b3`), canonical `30b78078…` |
| **split** | **M07's `SPLIT_eth4h_l24_h6_v1.json`** (predictor `satoshi/f2-eth-forecast-20261001` `13ef175f`): declared calendar split TRAIN rows [0, 13699), VALIDATION [13699, 15895), TEST [15895, 18085) never read; window 24, horizons 1..6, purge h_max = 6; **train origins [23, 13669], 13,415 windows, 232 gap-excluded** | split file `116a5b645fe08138c49a558c206dd970528128bcd8370bb9ec8cd62b20a61e25`; train row-ids sha `3903daae…` **reproduced** by `tools/c2_eth_population.py` from M07's `windows_for` rule; mu `0.000146097`, sigma `0.0194728` reproduced to 1e-6 |
| target | M07's definition: `Y_h = sum_{k=1..h} z(log_return_1[t+k])`, z fitted on TRAIN rows only; raw `log(CLOSE[t+h]/CLOSE[t])` kept beside for log-return units | — |
| producer of the features (for the leak probe) | financial-data `_scripts/workers/stage22_trading_features_worker.py` @ `19fe375a`, named by FEATURE_DAG.v3 (`ac2e8072`, dag sha `e15017ce…`), vendored verbatim under `tools/c2_vendor/` | `7495a0d9…` |
| lane B references | `eth_f1_probe.json` @ `678665e` (`ad7feba4…`), `ETH_VARIANT_D_FAMILIES_PROBE.v1.json` @ `2840e52` (`cfc77827…`), `LANE_B_PROBE_SUMMARY.v1.json` @ `980e087` (`b898077d…`) | copied as `LANE_B_*_REFERENCE*.json` |

The binding is refused, never patched, when any digest or M07's origin count / row-id digest disagrees
(`PopulationRefusal`, tested).

Units: **MAE_z** = M07 z-units of the cumulative standardized 1-bar log return; **MAE_log_return** = raw log return
of CLOSE. All held-out evaluations are inside TRAIN: five contiguous time blocks of the scored origins (`blocks5`,
fit on the other blocks outside an embargo of `6 + h` rows on each side) and lane B's three expanding inner folds
(`laneB`, purge 60) for comparability.

## 1. Deliverable 1: per-feature predictive contribution (DONE; 152 s CPU, observed scope peak 295 MB)

`tools/c2_feature_contribution.py` -> `contribution/contribution_records.csv` (14,336 records; every record carries
population, split, fit rows, eval rows, horizon, n) and `contribution_summary.json`.

### 1.1 Reference rows and the all-83 ridge on the identical rows (blocks5, mean of 5 blocks, MAE_z)

| h bars / hours | n eval | naive zero | naive last-return | naive seasonal 24 h | naive train-mean | ridge all-83 (alpha selected) | ridge all-83 (alpha = 1, lane B) | HGB all-83 |
|---|---|---|---|---|---|---|---|---|
| 1 / 4 | 13415 | **0.63323** | 0.95261 | 0.96986 | 0.63327 | 0.63386 | 0.64680 | 0.65664 |
| 2 / 8 | 13415 | **0.90460** | 1.33501 | 1.38247 | 0.90474 | 0.90949 | 0.94100 | 0.97497 |
| 3 / 12 | 13415 | **1.12731** | 1.61810 | 1.71443 | 1.12778 | 1.13504 | 1.18647 | 1.22246 |
| 4 / 16 | 13415 | **1.32750** | 1.92242 | 1.99723 | 1.32800 | 1.33738 | 1.40799 | 1.51681 |
| 5 / 20 | 13415 | **1.51131** | 2.22084 | 2.26161 | 1.51225 | 1.52284 | 1.61324 | 1.70370 |
| 6 / 24 | 13415 | **1.67674** | 2.48560 | 2.48560 | 1.67767 | 1.68857 | 1.79532 | 1.86013 |
| 36 / 144 | 13181 | **4.36517** | 6.24954 | 6.24954 | 4.37126 | 4.43123 | 5.15689 | 5.73271 |

In log-return units (blocks5, from the same records, `contribution/blocks5_log_return_units.json`): h = 1 naive zero 0.012331,
ridge all-83 0.012343, HGB 0.012787; h = 6 naive zero 0.032651, ridge 0.032881, HGB 0.036222; h = 36 naive zero 0.085002,
ridge 0.086289, HGB 0.111632. **No model arm beats the zero-return naive at any horizon.** The
selected ridge alpha is 1e3-1e4 on most blocks: the best linear model is almost the train mean.

### 1.2 Comparability with lane B (laneB protocol; ours on M07's scored origins, lane B on its rows)

| target | n eval ours / lane B | naive zero ours / lane B (log-return) | ridge all-83 alpha = 1 ours / lane B |
|---|---|---|---|
| Y_s@4h (h=1) | 6134 / 6162 | 0.01087 / 0.01088 | 0.01301 / 0.01296 |
| Y_l@24h (h=6) | 6124 / 6147 | 0.02853 / 0.02853 | 0.04726 / 0.04632 |
| Y_l@144h (h=36) | 6057 / 6057 | 0.07270 / 0.07270 | 0.15711 / 0.15574 |

Lane B's result is reproduced independently (naives to four decimals; ridge within 2 %, the difference being the
232 gap windows M07 excludes). Lane B's 32 ETH rows of `LANE_B_PROBE_SUMMARY.v1` are all FAIL against the naive; its
PS4 families (range 0.010834 vs naive 0.010881 on Y_s@4h; wavelet/z-score/realized-vol worse) are copied as
reference rows in `recommendation/RECOMMENDATION_TABLE.md`.

### 1.3 Conditional incremental utility and its stability

Incremental utility delta = MAE_z(without f) - MAE_z(all 83), held out, per block. Across the 83 x 7 cells:

* **rank agreement across the five blocks** (mean pairwise Spearman of per-block incremental-utility ranks):
  h1 0.116, h2 0.049, h3 0.051, h4 0.029, h5 -0.023, h6 0.001, h36 0.035 (laneB folds: 0.101, 0.038, 0.053, 0.074,
  0.071, 0.041, -0.165). **The per-block rankings are essentially uncorrelated: there is no stable ranking.**
* 93 cells have a positive median delta with sign agreement >= 4/5, but the largest such median is
  **0.00035 MAE_z** (cci_14, h2) against a naive MAE_z of 0.905 (0.04 %), and 0.0093 MAE_z at h36 (obv, 0.2 % of
  4.365). The HGB permutation deltas agree in magnitude (noise level).
* Does ANY single feature beat the zero-return naive on its own? **Only at the 0.1 % level**: `log_return_1` /
  `statistical__log_return_1` / `return_1` alone at h=1, MAE_z 0.63229 vs naive 0.63323 (+0.147 %), in 5 of 5 blocks;
  `realized_var_48` at h36 (+0.128 %, 3 of 5); `hist_vol_10` at h4 (+0.048 %, 4 of 5). Everything else is below
  0.05 % or not stable. Lane B's zero-return naive stays the gate.
* Top 20 by the stable score (mean over h=1..6 of the median relative delta): obv, hurst_proxy_200, realized_var_48,
  roll_skew_ret_252, cci_14, log_return_5, zscore_close_100, return_5, sqret_autocorr_lag1_100, close_sma_ratio_10, …
  (full list in the table). The best score is 0.00059 (obv) with mean sign agreement 0.77.

### 1.4 Causal-ordering (leak) probe (DONE; 2.7 s CPU, observed peak 29 MB) -- `leak_probe/leak_probe.json`

RL01 template (lane G's `causal_timestamp_probe`): the vendored producer is replayed on the view's own OHLCV; a stored
column is certified only when (i) the replay reproduces it after a 1,500-row burn-in (max |diff| / std < 1e-3; obv
compared in first differences) and (ii) moving every raw row strictly after the probe step (+1000 on O/H/L/C, x3 on
volume, at steps 3000, 6000, 9000, 12000, 13600) leaves the recomputed column at rows <= step unchanged while moving
the lookback rows changes it. Result: **83 of 83 `CAUSAL_BY_RECOMPUTATION`**, 0 producer mismatches, 0 future-row
influence; statistical flag (|Spearman| with the next-bar return > 0.2): **0 suspect**. The 45 columns the DAG v3 left
`UNRESOLVED` (no producer located) are reproduced by the same stage22 functions, so their producer is now located by
replay, not by text.

## 2. Deliverable 2: causal dossiers (`dossiers/`, one `causal_dossier.v1` per feature x horizon)

TBD_D2

## 3. Deliverable 3: recommendation table to M03 and M07 -- `recommendation/RECOMMENDATION_TABLE.{csv,md}`

1,162 rows (83 features x 7 horizons x 2 protocols), each with population, split, rows, horizon, the four naives
and the three full models on the same rows, the leak verdict, and the dossier's support state and control verdicts.

Recommendation, stated plainly:

* **M03 (selection):** no feature of variant A earns inclusion on predictive grounds against the zero-return naive at
  any horizon 1..6 bars; the only reproducible single-feature margin is the last 1-bar return at h=1 (+0.15 %). Rank
  the screen by the table's `stable_incremental_utility_score` if a bounded candidate list is needed, but the
  per-block rank agreement (<= 0.12) says that order will not survive a new block.
* **M07 (campaign):** the paired gate on the same rows is the zero-return naive (table 1.1); the all-83 linear
  model loses to it by 0.1 % (h1) to 1.5 % (h36); a non-linear tree loses by 3.7-31 %.

## 4. Deliverable 4: PS3-R representation probe rerun under M01's tooling

TBD_D4

## 5. What was NOT verified

* The timestamp semantics of the view (bar open / close / publication) -- undeclared in FEATURE_DAG.v3; bar-close
  assumed, not certified. Point-in-time availability of any column: not established (manifest blocker B1).
* That the model-ready view was materialized by the exact revision `19fe375a` of the producer: the replay reproduces
  every column numerically, which certifies the formula, not the provenance of the bytes.
* The partially linear model of the dossiers against a non-linear alternative; the HAC lag rule (h + 6) against
  alternatives; no permutation multiplicity correction (581 cells; the reported p-values are raw).
* Lane B's probe rows were reproduced on M07's origins, not re-executed from lane B's code.
* Nothing on validation or test rows; nothing governed; no financial performance.

## 6. CPU, memory and placement

| job | host | cap declared | observed scope peak | CPU |
|---|---|---|---|---|
| tests (11 passed, 1 skipped on worker_b; 44 passed incl. schema tests on the coordinator under a 1G cap) | worker_b | 2G | 29 MB | 2.5 s |
| contribution pilot (10 features, h 1 and 6) | worker_b | 2G | 200 MB | 4 s |
| contribution full | worker_b | 1G (above 1.25 x pilot; re-declared 400M for successors) | 295 MB | 152 s |
| leak probe | worker_b | 1G | 29 MB | 2.7 s |
| dossiers (498 cells) | worker_b | 1G | TBD_PEAK | TBD_CPU |
| PS3-R rerun (M01 runner) | worker_b | 3G (1.25 x M01's 2.157 GB measured child peak) | child RSS 2.036 GB (contrastive record) | stopped at 6/240 by the one-CPU-job rule; resume cap 2.6G |

Incidents against me: (1) the first dossier run (superseded by the control-criterion fix) was stopped by signalling
its crispdm-run wrapper instead of its python process (`pgrep -f` matched the wrapper); the wrapper's own TERM trap
forwarded the signal and released the lease; nothing outside my job was touched. (2) Three CPU leases of mine were
live at once on worker_b before the coordinator's one-job rule; corrected within 10 minutes.

## 7. Files

`tools/c2_eth_population.py`, `tools/c2_feature_contribution.py`, `tools/c2_leak_probe.py`, `tools/c2_causal_dossier.py`,
`tools/c2_recommendation_table.py`, `tools/c2_progress_png.py`, `tools/c2_vendor/` (producer + PROVENANCE.json),
`tests/test_c2_eth_causal_tools.py`, `docs/contracts/causal_dossier.v1.schema.json` (new `CONTRACTED_MODEL_READY_VIEW`
slot, which can never carry an identified effect), and this directory: `STATUS.json`, `PROGRESS.png`, `RESULTS.json`,
`RESULTS.csv`, `contribution/`, `leak_probe/`, `dossiers/`, `recommendation/`, the copied manifest and split.
