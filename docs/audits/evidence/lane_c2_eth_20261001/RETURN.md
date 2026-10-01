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

Tool `tools/c2_causal_dossier.py`; 498 dossiers (83 features x h=1..6), 498 cells, 948 s CPU on worker_b
(observed scope peak 270 MB as measured by M06: the 1G cap was 3.98x; successors declare 340M). **All 498 validate against
`causal_dossier.v1`** (Draft 2020-12, on the coordinator); 24 carried a residual-variance share above 1 (out-of-fold
nuisance worse than the mean) and were clipped to the contract bound with the raw value kept in `rung2.sensitivity`
and a limitation line. The 20 delivered features are the top-20 of deliverable 1 (`delivered_top20` in the index); the
other 63 x 6 stay as evidence.

Identifying assumptions, declared in every dossier so they can be attacked: the lineage DAG v3 names the producer of
X_j and of W from the same price history; partially linear effect; additive noise; **no unmeasured confounding given W
is declared FALSE** (latent market state); **timestamp = bar close is declared FALSE** (undeclared in the view); no
physical intervention on a derived feature exists. Consequently **every rung 2 is `NOT_IDENTIFIED`** and every rung 3 is
`NOT_IDENTIFIED` with label `MODEL_BASED_COUNTERFACTUAL`; the dossier's rung 1 is `ASSOCIATION_REPORTED` and
`selection.causal_evidence_level = ASSOCIATION`, `cf_eligible = false`. The estimate is reported in `rung2.sensitivity`
as a DEVELOPMENT number only (`rung2.estimate` is null, as the contract's identified-only reading requires).

Estimator: cross-fitted partially linear DML (ridge nuisances, five time blocks with embargo, HAC Newey-West lag h+6),
contrast do(X_j = q75) - do(X_j = q25) in M07 z-units and log-return units; support screen = residual variance share
of X_j after adjustment on ALL 82 other features (floor 0.05).

Results over the 498 cells:

* **support**: 300 cells `TREATMENT_PREDICTED_BY_CONTROLS` (median share 0.004: a feature of
  variant A is, in general, a deterministic transform the other 82 reproduce), 198 cells `SUPPORTED` over
  33 features (atr_14, autocorr_lag1_100, autocorr_lag5_100, bb_width, cci_14, hist_vol_10, hurst_proxy_200, mfi_14, mom_10, mom_20, natr_14, obv, obv_delta_20, realized_var_12, realized_var_48, roll_kurt_ret_20, roll_kurt_ret_252, roll_kurt_ret_60, roll_mean_ret_252, roll_skew_ret_20, roll_skew_ret_252, roll_skew_ret_60, roll_std_ret_252, sqret_autocorr_lag1_100, stoch_d, trend_slope_50, trend_strength_50, vol_regime_high, vol_regime_low, volume_ratio_20, volume_sma_10, volume_sma_20, zscore_close_100).
* **controls that MUST fail, actually run** (`rung2.placebo.tests`): future-shifted feature on the RL01 shift template
  fired in **498/498**; scrambled label left the one-feature ridge with nothing to add in **498/498**;
  the noise treatment's HAC interval covered zero in 498/498. **But the scrambled-label HAC interval excluded zero in
  90/498 = 18 % of cells against a nominal 5 % (25 expected)**: the estimator's interval is
  anti-conservative on this population (persistent regressors x volatility-clustered labels), so `battery_verdict =
  BATTERY_SUSPECT` and no theta interval below may be read at face value.
* on the real labels the same interval excluded zero in 76/498 = 15 % of cells -- **no higher than under the
  scrambled-label null**. There is no cell whose effect stands out from what the control produces with no signal.
* largest contrasts belong to screened cells (macd_hist: effect -42 z at h3 with theta -9.9 +- 15.1, share 0.0000):
  collinearity artefacts, flagged by the support screen, not effects.
* rung 3 under the PLM: delta_e = theta (x_e - x0) exactly; `prediction.model_based` is reported beside the
  counterfactual mean so the two are never confused; `emission.operational_use = RETROSPECTIVE_ONLY`,
  `emittable_from` = last TRAIN bar + h.

Top-20 dossiers at h = 1 and h = 6 (full set in `dossiers/DOSSIER_INDEX.json`):

| feature | h | n | theta (z per unit) +- HAC se | effect q25->q75 (z / log-return) | share (full W) | support | controls failed as required |
|---|---|---|---|---|---|---|---|
| log_return_1 | 1 | 13415 | -3.6191 +- 1.1524 | -0.0528 / -0.00103 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| log_return_1 | 6 | 13415 | -1.4738 +- 1.6393 | -0.0215 / -0.00042 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| return_5 | 1 | 13415 | +5.7868 +- 0.8999 | +0.2244 / +0.00437 | 0.002 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| return_5 | 6 | 13415 | -1.5354 +- 1.4981 | -0.0596 / -0.00116 | 0.002 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| log_return_5 | 1 | 13415 | +5.9282 +- 0.9480 | +0.2295 / +0.00447 | 0.002 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| log_return_5 | 6 | 13415 | -1.7448 +- 1.6364 | -0.0675 / -0.00132 | 0.002 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| close_sma_ratio_10 | 1 | 13415 | -10.3709 +- 2.8201 | -0.2918 / -0.00568 | 0.023 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| close_sma_ratio_10 | 6 | 13415 | -16.7139 +- 9.5190 | -0.4702 / -0.00916 | 0.023 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| close_sma_ratio_200 | 1 | 13415 | -0.1579 +- 0.3715 | -0.0317 / -0.00062 | 0.035 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| close_sma_ratio_200 | 6 | 13415 | -1.3447 +- 1.7735 | -0.2700 / -0.00526 | 0.035 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| cci_14 | 1 | 13415 | +0.0012 +- 0.0004 | +0.1377 / +0.00268 | 0.071 | SUPPORTED | yes |
| cci_14 | 6 | 13415 | +0.0025 +- 0.0014 | +0.2991 / +0.00583 | 0.071 | SUPPORTED | no (scrambled null rejected) |
| bb_width | 1 | 13415 | -0.1126 +- 0.3712 | -0.0103 / -0.00020 | 0.122 | SUPPORTED | yes |
| bb_width | 6 | 13415 | -1.6981 +- 1.8250 | -0.1555 / -0.00303 | 0.122 | SUPPORTED | yes |
| hist_vol_10 | 1 | 13415 | +2.2163 +- 1.0607 | +0.0800 / +0.00156 | 0.127 | SUPPORTED | yes |
| hist_vol_10 | 6 | 13415 | +10.9924 +- 3.5184 | +0.3969 / +0.00773 | 0.127 | SUPPORTED | no (scrambled null rejected) |
| hist_vol_60 | 1 | 13415 | -0.1797 +- 0.4459 | -0.0131 / -0.00025 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| hist_vol_60 | 6 | 13415 | -0.5414 +- 2.3865 | -0.0394 / -0.00077 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| obv | 1 | 13415 | +0.0000 +- 0.0000 | +0.0537 / +0.00105 | 0.353 | SUPPORTED | no (scrambled null rejected) |
| obv | 6 | 13415 | +0.0000 +- 0.0000 | +0.3969 / +0.00773 | 0.353 | SUPPORTED | yes |
| mfi_14 | 1 | 13415 | -0.0024 +- 0.0010 | -0.0588 / -0.00115 | 0.244 | SUPPORTED | yes |
| mfi_14 | 6 | 13415 | -0.0101 +- 0.0046 | -0.2522 / -0.00491 | 0.244 | SUPPORTED | yes |
| statistical__log_return_1 | 1 | 13415 | -3.6191 +- 1.1524 | -0.0528 / -0.00103 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| statistical__log_return_1 | 6 | 13415 | -1.4738 +- 1.6393 | -0.0215 / -0.00042 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| roll_std_ret_60 | 1 | 13415 | -1.3921 +- 3.4540 | -0.0131 / -0.00025 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| roll_std_ret_60 | 6 | 13415 | -4.1936 +- 18.4857 | -0.0394 / -0.00077 | 0.000 | TREATMENT_PREDICTED_BY_CONTROLS | yes |
| roll_kurt_ret_60 | 1 | 13415 | -0.0025 +- 0.0019 | -0.0081 / -0.00016 | 0.782 | SUPPORTED | yes |
| roll_kurt_ret_60 | 6 | 13415 | -0.0115 +- 0.0102 | -0.0381 / -0.00074 | 0.782 | SUPPORTED | yes |
| roll_skew_ret_252 | 1 | 13415 | -0.0145 +- 0.0071 | -0.0126 / -0.00024 | 1.271 | SUPPORTED | yes |
| roll_skew_ret_252 | 6 | 13415 | -0.0793 +- 0.0377 | -0.0689 / -0.00134 | 1.273 | SUPPORTED | yes |
| realized_var_48 | 1 | 13415 | +0.9966 +- 1.2192 | +0.0149 / +0.00029 | 0.254 | SUPPORTED | yes |
| realized_var_48 | 6 | 13415 | +6.8917 +- 6.0810 | +0.1027 / +0.00200 | 0.254 | SUPPORTED | no (scrambled null rejected) |
| autocorr_lag1_100 | 1 | 13415 | +0.0959 +- 0.1217 | +0.0125 / +0.00024 | 0.636 | SUPPORTED | yes |
| autocorr_lag1_100 | 6 | 13415 | +0.9864 +- 0.6731 | +0.1281 / +0.00249 | 0.635 | SUPPORTED | yes |
| sqret_autocorr_lag1_100 | 1 | 13415 | -0.1161 +- 0.0772 | -0.0174 / -0.00034 | 1.057 | SUPPORTED | yes |
| sqret_autocorr_lag1_100 | 6 | 13415 | -0.5600 +- 0.4224 | -0.0841 / -0.00164 | 1.058 | SUPPORTED | yes |
| hurst_proxy_200 | 1 | 13415 | -0.5888 +- 0.2956 | -0.0292 / -0.00057 | 0.620 | SUPPORTED | yes |
| hurst_proxy_200 | 6 | 13415 | -3.6467 +- 1.5924 | -0.1806 / -0.00352 | 0.620 | SUPPORTED | yes |
| zscore_close_100 | 1 | 13415 | +0.0399 +- 0.0242 | +0.0872 / +0.00170 | 0.091 | SUPPORTED | yes |
| zscore_close_100 | 6 | 13415 | +0.2432 +- 0.1261 | +0.5315 / +0.01035 | 0.091 | SUPPORTED | no (scrambled null rejected) |


### 2.1 Interval calibration (coordinator ruling 2026-10-01) -- `calibration/BATTERY_CALIBRATION.json`, `tools/c2_battery_calibration.py`

Measured integrated autocorrelation length (lags until rho < 0.05) of the target and of each feature on the TRAIN origins; block
length L = max(tau_Y, tau_X, h + 6): **min 7, median 74, max 1287** (median tau_Y 3.5,
median tau_X 74; the features, not the labels, set L). Tests first: a planted persistent null (AR(1) 0.97 regressor, AR(1) 0.9
outcome) where the lag-7 HAC rejects above 10 % and the block interval at the measured L does not (`test_block_interval_is_calibrated_...`).

| rejection rate over 498 cells (nominal 0.05; tolerance 0.0695 = nominal + 2 binomial sd) | scrambled label | real label |
|---|---|---|
| BEFORE: HAC lag h + 6 | **0.181** (90) | 0.153 (76) |
| AFTER: HAC bandwidth L | 0.054 (27) | 0.161 (80) |
| AFTER: circular block bootstrap, block L, B = 200 | 0.056 (28) | 0.165 (82) |

Both calibrated intervals bring the scrambled-label rate inside the tolerance (battery verdict `CONTROLS_FAIL_AS_REQUIRED`; the future-shift
template fired in 498/498 and the noise treatment rejected 0/498). The dossiers' scrambled-label verdict now reads the block bootstrap,
and each dossier carries its L, tau_Y, tau_X and both calibrated SEs. **Identification is still not claimed**: the real-label rate stays
at 16 % (about 3x nominal), which the calibrated null says is not interval error. It is association the declared model
cannot place (non-stationary association between deterministic price transforms and returns), and rung 2 stays `NOT_IDENTIFIED` in
498/498 for the reasons already declared. 498/498 re-validated against the contract after the patch.
Cost: 196 s CPU, measured peak RSS 277 MB (cap declared 400M, 1.38x; the 0.28 GB pilot sampler reading was a 23 MB undercount on a seconds-long job).

### 2.2 Findings (not effects): where the real-label rejection lives, and the seasonal reference -- `findings/REAL_REJECTION_FINDINGS.json`, `tools/c2_real_rejection.py`

498 cells, calibrated block-bootstrap interval (L as in 2.1), 889 s CPU, measured peak 298 MB (cap 376 MB = 1.25x the pilot's 300 MB). Label **FINDING, not an
effect**; every cell keeps rung 2 `NOT_IDENTIFIED`.

**Real-label rejection (0.165 base).** 82 cells reject; 64 of them have the same partial-association sign in at least 4 of 5 time blocks. By horizon
(base -> after regime conditioning): h1 0.217 -> 0.133, h2 0.181 -> 0.169, h3 0.145 -> 0.108, h4 0.145 -> 0.145, h5 0.205 -> 0.120, h6 0.096 -> 0.036.
Conditioning on the naive's own error history (a trailing mean |return| over 60 bars as a regime scale, the label divided by it, and the
trailing scale and the signed h-bar return added to the controls; rows <= t only): scaled label 0.096, controls added 0.165 (unchanged), both 0.118.
Per cell: **42 SURVIVE_REGIME_CONDITIONING, 40 REGIME_ARTEFACT_CANDIDATE**, 416 not rejected. So about half of the rejections vanish once the volatility regime is
removed (they are volatility-state features: hist_vol_20 6/6 -> 0/6, roll_std_ret_20 6/6 -> 0/6, realized_var_48 3 -> 0, mfi_14 6 -> 1, hurst_proxy_200 6 -> 1);
the rest persists in features that carry sign-stable association with the return label: `obv` 6/6 -> 6/6, `roll_skew_ret_252` 6/6 -> 6/6, `cci_14` 5 -> 4,
and `log_return_1` (3 -> 2). The unchanged rate with only controls added says the rejection is not the persistence/regime level in W but the scale of the label.
Reading: the 0.16 is partly a persistent-volatility-regime artefact and partly sign-stable association of two slow-moving features; the
latter does not improve out-of-sample error against the zero-return naive (section 1), and obv carries the sentinel-like minimum flagged in the manifest.

**Seasonal reference (partial association of Y_h with the h-bar return one 24 h period earlier, beyond all 83 features, calibrated):**

| h | theta (per sd of the reference) | block-bootstrap se | t | rejects | sign stable (blocks of 5) | corr with label | L |
|---|---|---|---|---|---|---|---|
| 1 | -0.0501 | 0.0164 | -3.05 | yes | 0.8 | -0.081 | 7 |
| 2 | -0.1510 | 0.0246 | -6.15 | yes | 0.8 | -0.089 | 8 |
| 3 | -0.2021 | 0.0348 | -5.80 | yes | 0.8 | -0.078 | 9 |
| 4 | -0.2401 | 0.0445 | -5.40 | yes | 1.0 | -0.060 | 10 |
| 5 | -0.2674 | 0.0551 | -4.85 | yes | 1.0 | -0.042 | 11 |
| 6 | -0.1842 | 0.0720 | -2.56 | yes | 0.8 | -0.032 | 12 |

The seasonal lag carries a real, calibrated, mostly sign-stable association with the label on ETH, but its sign is NEGATIVE (the label tends to
reverse the return of the same hours the day before) with a small correlation (|r| <= 0.09). That is the opposite of what the 24 h seasonal naive
assumes (it predicts +y(t - 6)), which is why that naive is far worse than the zero-return naive (section 1.1: 2.4856 vs 1.67674 at h6). The association
does not by itself beat the zero-return naive: the intercept-only and zero models remain the gate. Persistence reference (h-bar return over (t - h, t]): rejects only
at h3 (+0.26, t 4.5), h4 (+0.16, t 2.7) and h6 (-0.18, t -2.6, identical to the seasonal row at h6 because the lags coincide); at h1 and h5 its theta is degenerate (-1445, -715)
because that reference is almost collinear with `log_return_1` and the other return features (residual variance ~0): treat those two cells as not estimable.

### 2.3 Applicable interval rule, evidence classes, time semantics (owner order 4.C)

* `tools/c2_interval_rule.py` + `test_interval_rule_threshold_and_committed_eth_evidence`: block length L = max(tau_Y, tau_X, h+6), block bootstrap B >= 200 (or HAC
  bandwidth L), reject when |theta| > 1.96 se, thresholds scrambled-label and noise rates <= 0.05 + 2 sqrt(0.05*0.95/n_cells) (0.0695 at 498), future-shift template in 100 %
  of cells; the rule refuses the uncalibrated HAC h+6 (0.181) and never moves a rung. Applies to ETH 4h variant A with the committed calibration. EURUSD 1h: NOT_MEASURED.
* Schema: `evidence_classes` (association / observed natural interventions / paired counterfactuals, each with state, assumptions and support, assumptions required when reported)
  and `time_semantics` (event time, availability time = UNDECLARED, population). 498/498 dossiers carry them and validate (worker_b). Observed natural interventions is
  `NOT_AVAILABLE` in all 498 (no quasi-experiment exists for a derived feature).
* Constraint recorded for selection: PS3-R is not read as selection. Extension (PS3-R/PS5) only on candidates that survive lane B's eligible ledger, with original vs transformed
  vs destroyed-signal control; on ETH 4h TRAIN no variant-A feature clears the zero-return naive, so no candidate survives that gate today.

### 2.4 First EURUSD 1h measurement under the calibrated rule (DEVELOPMENT) -- `eurusd_1h/{fxA_full,fxR_full}/POPULATION_MEASUREMENT.json`

Binding: `PopulationSpec` (tools/c2_eth_population.py; ETH stays the default). Population: heuristic-strategy `939f5e6e` `tests/data/eurusd_hour_2005_2020.csv`
(sha `72b8271d…`, 93,084 rows), lane B's frozen manifests (variant A file sha `d36cb91e…`, declaration `6f8fed41…`; range family vD1h file sha `2985f589…`),
split declared by the manifest (TRAIN rows [0, 65158), 2005-05-02 to 2015-10-16; validation and test never read). Scored origins [0, 65151], **61,796**
(3,356 excluded for weekend gaps: label at exactly h x 3600 s), purge 6. Target: M07's cumulative standardized `log_return_1` (derived causally from CLOSE).
Intervals: the rule of `c2_interval_rule` (block bootstrap, B = 100, S = 10 scrambled and 10 noise draws per cell). 24 h reference = 24 bars.

**Variant A (OPEN, LOW, HIGH, CLOSE) is not estimable**: the four price levels are one series (every pair above the 0.95 redundancy bound), so all 24 cells are
`TREATMENT_COLLINEAR_WITH_ALL_CONTROLS` and the rule verdict is `NOT_ESTIMABLE_ALL_CELLS_COLLINEAR_WITH_THEIR_CONTROLS`. The model references still hold on the
same 61,796 rows (blocks5, MAE_z): zero-return naive 0.639962 (h1) ... 1.644951 (h6); intercept-only 0.639991 ... 1.645053; 24 h seasonal 0.927777 ... 2.375595; ridge on the
four levels 0.639717 (h1, 0.04 % below zero), 0.914701 (h2, above), 1.646114 (h6, above): no gain beyond the noise, as lane B found.

**Range family (log_high_low, close_location, log_close_open; lane B vD1h, valid for h = 1 only per lane B)** -- 3 features x h = 1..6 = 18 cells x 10 draws = 180 null draws:
block length L min 7, median 11, max 692 (log_high_low carries tau_X = 692: volatility persistence). Scrambled-label rejection: HAC h+6 0.039, HAC bandwidth L 0.039,
**block bootstrap 0.061** (tolerance 0.0825 for 180 draws); noise treatment 13/180 = 0.072; future-shift template fired 180/180. **Rule verdict CONTROLS_FAIL_AS_REQUIRED**, so these
intervals may be read (the stricter n here is small: 180 draws, tolerance 0.0825). `close_location` rejects at h = 1, 2, 3, 4, 6 with a negative theta (-0.10 z per unit at h1,
se 0.021, calibrated; sign-stable across blocks was not computed in this run); `log_high_low` and `log_close_open` do not reject at any horizon. Real-label rate 5/18 = 0.278.
Model arms against the controls (blocks5, MAE_z): ridge on the three range features **0.639027 vs zero 0.639962 and intercept-only 0.639991 at h1 (0.146 % below zero)**, 0.913878 vs 0.914482
at h2 (0.066 %), 1.131535 vs 1.132082 at h3 (0.048 %), 1.321143 vs 1.321527 at h4, 1.489645 vs 1.489797 at h5, 1.644719 vs 1.644951 at h6 (0.014 %). The gain is smallest where lane B says the
family is not valid (h > 1) and is of the order of the ETH `log_return_1` margin; it is a finding about predictive association, not an effect, and not a financial result.
`close_location` filled with the neutral 0.5 on 4 rows where HIGH == LOW (declared deviation from lane B's NaN). Cost: variant A 3 s CPU (peak 210 MB, cap declared 750 MB without a pilot:
over-declared, my error); range pilot 255 MB -> cap 319 MB; range full 73 s CPU, peak 308 MB.

### 2.5 EURUSD lake S1/S2 and per-block sign stability (DEVELOPMENT) -- `eurusd_1h/{lakeR_S1,lakeR_S2,lakeA_S1,lakeA_S2,fxR2_full}/`

Population: lane B's lake derivative `eurusd_1h_from_lake_5m.csv` (sha `ab0ada28…`, 129,873 rows; frozen manifests v1 file sha `ab5ba3c8…`, vDh `963d0236…`), bound through
`PopulationSpec(split_variant=...)`. **S1** (70/15/15): TRAIN rows [0, 90911), 86,093 scored origins. **S2 (prospective reserve)**: TRAIN rows [0, 93924), 88,944 scored origins, validation
[93924, 117405), **reserve [117405, 129873) PROTECTED_NEVER_READ -- no row of it entered any statistic here** (bind_population only records its row bounds; tested). Horizons 1..4 (the vDh valid set).
Rule (c2_interval_rule): lake range S1 verdict CONTROLS_FAIL_AS_REQUIRED (scrambled rejection block bootstrap 0.050, tolerance 0.0898 over 120 draws), S2 CONTROLS_FAIL_AS_REQUIRED (0.033). Variant A on the lake
is again 4 collinear price levels: NOT_ESTIMABLE in both splits. Reference rows (blocks5, MAE_z, identical rows): S1 h1 zero 0.637173, intercept 0.637255, ridge-on-range 0.636484 (0.108 % below zero);
h4 1.314782 / 1.314922 / 1.314400. S2 h1 0.634755 / 0.634843 / 0.634098 (0.103 %); h4 1.310077 / 1.310233 / 1.309704.

Per-block sign stability of the real-label partial association (5 time blocks; theta per unit of the standardized feature; stable = same sign in >= 4 of 5 blocks):
* `close_location`: **rejects at every horizon in S1 and S2 (h1-h4) and sign-stable (1.0 at h1-h3, 0.8 at h4), always NEGATIVE** (theta about -0.12 to -0.14 z per unit in both splits, block values -0.004 to -0.31); on the
  git-pinned file it rejects at h1-h4 and h6, stable 1.0, 1.0, 0.8, 0.8, (h5 not rejected, 0.8), 0.8. The same sign and size across three overlapping populations (git-pinned 2005-2015, lake S1, lake S2) is the only
  association on EURUSD that is replicated, and it is not independent evidence (the populations share years).
* `log_close_open`: rejects only on the lake at h1, h2 (theta +15.8 / +18.4 S1, +15.6 / +18.1 S2; sign stable 0.8) with block values from -15 to +65 (a heavy-tailed block 4): unstable in size, not rejected on the git-pinned file. Treat as not established.
* `log_high_low`: never rejects at any horizon, in any population; its block signs flip (0.6): L = 692-885 shows its tau_X is the volatility-clustering length.
Label: FINDING (predictive association under a declared model), not an effect; rung 2 stays NOT_IDENTIFIED. Cost: 70-74 s CPU each, peaks 298-329 MB (lake pilot 296 MB -> cap 370 MB; the git-pinned rerun
at cap 400 MB).

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
| dossiers (498 cells) | worker_b | 1G (3.98x the observed 270 MB per M06; successors 340M) | 270 MB (M06 measurement) | 948 s |
| PS3-R rerun (M01 runner) | worker_b | 3G (1.25 x M01's 2.157 GB measured child peak) | child RSS 2.036 GB (contrastive record) | stopped at 6/240 by the one-CPU-job rule; resume cap 2.6G |

Incidents against me: (0) four tiny assembly jobs (schema validation, final table, RESULTS, a clip post-fix; 2-4 s each) ran under 1G caps on the coordinator between 08:17:58Z and 08:18:44Z, contrary to the zero-batch rule; the ledger copy `COORDINATOR_ADMISSION_LEDGER_c2.txt` proves each lease released and none queued; all further assembly runs on worker_b. (1) the first dossier run (superseded by the control-criterion fix) was stopped by signalling
its crispdm-run wrapper instead of its python process (`pgrep -f` matched the wrapper); the wrapper's own TERM trap
forwarded the signal and released the lease; nothing outside my job was touched. (2) Three CPU leases of mine were
live at once on worker_b before the coordinator's one-job rule; corrected within 10 minutes.

## 7. Files

`tools/c2_eth_population.py`, `tools/c2_feature_contribution.py`, `tools/c2_leak_probe.py`, `tools/c2_causal_dossier.py`,
`tools/c2_recommendation_table.py`, `tools/c2_progress_png.py`, `tools/c2_vendor/` (producer + PROVENANCE.json),
`tests/test_c2_eth_causal_tools.py`, `docs/contracts/causal_dossier.v1.schema.json` (new `CONTRACTED_MODEL_READY_VIEW`
slot, which can never carry an identified effect), and this directory: `STATUS.json`, `PROGRESS.png`, `RESULTS.json`,
`RESULTS.csv`, `contribution/`, `leak_probe/`, `dossiers/`, `recommendation/`, the copied manifest and split.
