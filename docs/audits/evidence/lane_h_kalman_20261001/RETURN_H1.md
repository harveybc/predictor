# Lane H (causal Kalman family): return of H1

Label: DEVELOPMENT. Branch `satoshi/h-kalman-20261001` of predictor. Every table below is generated from the evidence files in
this folder (`worker_a/`, `worker_b/`, `gate/`, `pack/`); nothing is typed by hand except this text.

## 1. Contract and historical operator located

| item | path | sha256 (first 16) |
|---|---|---|
| causal operator contract (C135) and bank | predictor `tools/df_operators.py` (last change 1c810dcf) | 669d5cdb3a49e7cd |
| snapshot boundary the contract binds to | `tools/df_snapshot.py` | d0f82d492af90a92 |
| causality battery / reference | `tools/df_causal_battery.py`, `tests/df_causal_reference.py` | d8371ca843556 / 6580e713626e3a6c |
| historical `local_level_kalman`, `local_linear_trend_kalman` | `tools/df_operators.py` (`_ll_loglik`, `_llt_loglik`, `_fit_kind`) | same file |
| not-portable finding and policy | `docs/audits/evidence/repro_runs/d2_support_r1/r4/R4_PORTABILITY_POLICY.md`, `docs/integracion_workplan_2026_09_10/10_DIAGNOSTICO_KALMAN_NO_REPRODUCIBLE_2026_09_14.md` | 6cc2bc7533fee9de / 5a59681aadb7264d |

The historical operator fits Q/R by concentrated MLE (`scipy.optimize`); on one of three CPUs 83 of 1,716 cells deviated
beyond 1e-9 dB, largest 1.099 dB, cause not isolated. It is not used here for any decision.

## 2. The deterministic successor (`tools/df_kalman_family.py`, 37 tests of its own)

* Models: local level and local linear trend, per feature, independent. Multivariate: not started (the spec makes it
  conditional on a pilot and a cost of these two; both now exist, see section 8).
* Q/R: closed-form second moments of the differences, summed with `math.fsum` (autocovariances g0,g1 for level;
  g0,g1,g2 of the second difference for level+slope), fitted on TRAIN rows only (`fit` refuses any other role), or a
  pre-declared ratio q/r with r from the closed-form innovation power. No optimiser, no iteration budget needed.
  Negative estimates are clipped to a declared floor and typed (`r_clipped`, `q_clipped`, `qs_clipped`).
* Filter: scalar IEEE recursion, `+ - * /` and `sqrt` only, no BLAS, float64, one thread, environment recorded beside (not
  inside) the sealed digest. Initial state and covariance (`level = first finite observation`, `P0 = p0_scale * r`) recorded.
* Contract reuse: typed reasons AVAILABLE / WARMUP / MISSING_INPUT, sealed canonical-JSON artifact, `fit`,
  `transform_batch`, `init_state`, `step`, `transform_chunk`, `save_state`, `load_state`, delay 0 never compensated.
  Difference, stated: the D2 `FitSnapshot` objects are tied to the D2 dataset contracts, so this lane binds to the lane F2
  split artefact (role, row range, split digest, recomputed matrix digest) instead.
* Outputs per feature: obs, level, slope (trend), innov, zinnov, state_var (and slope_var), with units, availability and
  warm-up recorded; `state_var` is documented as a model variance, never a probability of being right.
* The backward RTS smoother exists only as `smoother_control` returning a `NonCausalControlOutput`; `validate_spec`
  refuses its kind, `eligible_matrix` refuses its output, a forged eligible flag is refused (tests).

Successor against the historical MLE in the declared regime (local level, q/r = 0.1, T = 20,000; `pack/SUCCESSOR_VS_HISTORICAL.json`):
ratio 0.104400 (closed form) vs 0.104391 (MLE), max |level difference| / obs std = 3.9e-06. Outside that regime (near-pure random
walk) they are not interchangeable: the MLE abstains or sits on its bound, the successor returns floor-clipped typed parameters.

## 3. Parity, restart, leak, replay (real data: ETH 4h variant A, 29 declared features, 15,895 rows)

`tools/h_kalman_parity_proof.py`, run on both workers (`worker_a/PARITY_PROOF.eth.worker_a.json`,
`worker_b/PARITY_PROOF.eth.worker_b.json`):

| check | local level (13 cols) | local trend (16 cols) |
|---|---|---|
| batch == tick-by-tick == restart from a durable blob in a SEPARATE process == chunked (output SHA-256) | equal, e2836562... | equal, eb818024... |
| end state after restart equals the end state of the batch | true | true |
| future-leak probe: rows after t shifted / replaced by noise / NaN, 12 probes | max move before t = 0.0 | 0.0 |
| the same probe on the non-causal smoother | moves (probe can fail) | moves |
| smoother refused as an input | true | true |
| typed counts | AVAILABLE 206,505, WARMUP 130 | AVAILABLE 254,160, WARMUP 160 |

Replay across the two workers (AMD on worker_a, Intel on worker_b): the artifact digests (85b00495... level, 6c2181bf...
trend), the fitted-state digests, the output digests and the `digest_summary` of the proof are IDENTICAL. The four-variant
grid (8 variant x group Kalman digest sets, 28 eligible arm input matrices), the 24-lag arms (float32 matrices) and the
EURUSD run compare equal on the exact layer (`tools/h_kalman_replay_compare.py`). The only differing digest is the
non-causal smoother control matrix (LAPACK 2x2 algebra, a rejected control) and, as expected, the ridge predictions: they
agree by value (max |dMAE| 3.0e-13, max |dMSE| 7.0e-13 over 96 horizon rows) but not bitwise (eigendecomposition by LAPACK).
The MLP arms were run on worker_b only (peak 2.26 GB; the coordinator reserved worker_a), so the MLP layer is NOT replayed.
Verdict: the successor reproduces across workers on the exact layer; it is NOT the historical estimator and no result of the
historical operator is restated.

Tests: 88 passed on worker_b (all lane H files, `pack/pytest_all.txt`), 84 passed on worker_a (same files without the TensorFlow
MLP test file, peak 152.7 MB).

## 4. Cost (measured, one thread)

| item | value |
|---|---|
| fit + transform, 13 level + 16 trend columns, 15,895 rows | 0.78 s CPU (worker_b) |
| batch cost | 1.04 us (level) / 1.54 us (trend) per row per column |
| tick-by-tick latency per full row, median / p99 | 0.30 ms / 0.56 ms (13 cols); 0.49 ms / 0.65 ms (16 cols), artifact verification included in every call |
| state size | 2-6 floats per column (JSON blob) |
| light run (3 arms + 5 controls, lag 1) | cgroup peak 645 MB, 15-54 s CPU depending on host |
| 24-lag arms (ridge, float32 matrices 13,415 x 5,160) | cgroup peak 1.71 GB (worker_b), 1.81 GB (worker_a), 81-181 s CPU |
| MLP arms, 3 seeds x 6 arms | cgroup peak 2.26 GB, 293 s CPU, 174 s wall |
| lane B gate replication | 2.4 s CPU, peak 643 MB |

## 5. Three arms, ETH 4h variant A, validation 2024 (2,190 rows), MAE and MSE in train-standardized units

Split `SPLIT_eth4h_l24_h6_v1` (sha 116a5b64...), TRAIN [0,13699), VALIDATION [13699,15895), purge 6, TEST 2025 never read.
Same-row naives reproduce lane F2's table (max difference 1.6e-9). Ridge, alpha chosen on an inner chronological hold-out of TRAIN.
Features: lane B's declared groups (13 local-level, 16 level+slope; the two binary `ema_cross_*` flags excluded).

| h | zero-return MAE / MSE | persistence MAE / MSE | seasonal-6 MAE / MSE | train-mean MAE | A | B | C |
|---|---|---|---|---|---|---|---|
| 1 | 0.4648 / 0.4856 | 0.6919 / 1.0039 | 0.7038 / 1.0287 | 0.4647 | 0.4647 / 0.4855 | 0.4647 / 0.4855 | 0.4647 / 0.4855 |
| 2 | 0.6574 / 0.9384 | 0.8347 / 1.4129 | 0.9848 / 1.9886 | 0.6572 | 0.6572 / 0.9380 | 0.6572 / 0.9380 | 0.6572 / 0.9380 |
| 3 | 0.8281 / 1.4348 | 0.9806 / 1.9205 | 1.2355 / 2.9848 | 0.8275 | 0.8275 / 1.4341 | 0.8275 / 1.4341 | 0.8275 / 1.4341 |
| 4 | 0.9560 / 1.9200 | 1.0924 / 2.3801 | 1.4337 / 3.9684 | 0.9551 | 0.9551 / 1.9186 | 0.9551 / 1.9186 | 0.9551 / 1.9186 |
| 5 | 1.0891 / 2.4294 | 1.2075 / 2.8920 | 1.6102 / 4.9437 | 1.0879 | 1.0879 / 2.4275 | 1.0879 / 2.4274 | 1.0879 / 2.4274 |
| 6 | 1.2003 / 2.9357 | 1.3276 / 3.4581 | 1.7661 / 5.9181 | 1.1981 | 1.1981 / 2.9329 | 1.1981 / 2.9328 | 1.1981 / 2.9328 |

Reading. On every arm the inner hold-out chose the largest alpha of the grid (1e8): the learner collapses to the TRAIN mean,
which IS the train-mean naive. The three arms are therefore the same constant to four decimals. The differences between arms
are at most 7e-6 in MAE (0.0015 percent of the naive; block-bootstrap interval of B minus A at h1 [-1.1e-5, -3e-6], quarter range
1e-6), which the report generator labels NEGLIGIBLE. This is not a Kalman effect.

To compare the arms at a regularisation that lets the features act, the same arms were also fitted at fixed alpha (mean MAE
over h1..h6, ETH, lag 1, alpha 1e3; `RESULTS.json` field `fixed_alpha_validation`):

| variant | A | B | C | C with causal EWMA | B with permuted outputs | B with noise | smoother (non-causal) |
|---|---|---|---|---|---|---|---|
| moments_train | 0.8976 | 0.9103 | 0.9024 | 0.9078 | 0.8946 | 0.9061 | 0.7333 |
| declared 1e-3 | 0.8976 | 0.9170 | 0.9082 | 0.9001 | 0.8948 | 0.9061 | 0.7707 |
| declared 1e-2 | 0.8976 | 0.9220 | 0.9133 | 0.9065 | 0.8945 | 0.9061 | 0.7728 |
| declared 1e-1 | 0.8976 | 0.9184 | 0.9094 | 0.9064 | 0.8953 | 0.9061 | 0.8373 |

The Kalman arms are worse than A at fixed alpha, and B is worse than its equal-capacity permutation control. Nothing to select.
The non-causal smoother "improves" h1 from 0.4647 to 0.2485 (and 0.7333 mean): that is the leak the control is meant to show,
it is flagged ineligible and never selected.

24-lag windows (ridge, moments_train; alpha 1e5 fixed, MAE h1..h6): A 0.4650/0.6624/0.8363/0.9730/1.1116/1.2269,
B 0.4685/0.6681/0.8430/0.9825/1.1190/1.2421, C 0.4678/0.6662/0.8397/0.9767/1.1120/1.2322, C with causal EWMA
0.4661/0.6635/0.8361/0.9726/1.1116/1.2281 (zero-return 0.4648/0.6574/0.8281/0.9560/1.0891/1.2003). Chosen alpha 1e8 for all.

Flatten+MLP control learner (hidden 44-44, MAE loss, AdamW, batch 64, patience 5, early stopping on an inner TRAIN hold-out; 24
lags; 3 seeds 2021-2023; mean MAE over seeds, seed std in parentheses):

| h | zero-return | A | B | C | C EWMA | B permuted | B noise |
|---|---|---|---|---|---|---|---|
| 1 | 0.4648 | 0.4856 (0.009) | 0.5177 (0.009) | 0.5167 (0.019) | 0.5254 (0.042) | 0.5257 (0.034) | 0.5110 (0.016) |
| 2 | 0.6574 | 0.7196 (0.073) | 0.7286 (0.043) | 0.6991 (0.009) | 0.6950 (0.021) | 0.7206 (0.050) | 0.7027 (0.029) |
| 3 | 0.8281 | 0.8609 (0.022) | 0.8752 (0.013) | 0.8676 (0.019) | 0.8714 (0.029) | 0.8751 (0.029) | 0.8480 (0.016) |
| 4 | 0.9560 | 0.9712 (0.010) | 1.0120 (0.012) | 1.0021 (0.018) | 0.9951 (0.032) | 1.0130 (0.036) | 0.9946 (0.009) |
| 5 | 1.0891 | 1.1368 (0.038) | 1.1553 (0.040) | 1.1394 (0.015) | 1.1290 (0.036) | 1.1240 (0.019) | 1.0976 (0.005) |
| 6 | 1.2003 | 1.2503 (0.079) | 1.2738 (0.074) | 1.2477 (0.025) | 1.2472 (0.076) | 1.2478 (0.034) | 1.2346 (0.010) |

The MLP stops at epoch 1-3 (it overfits immediately) and is worse than the zero-return naive at every horizon in every arm. Paired
against A (same rows, same seeds; mean over seeds of B minus A, with the range over the three seeds): B is WORSE than A by
+0.032 at h1 (seed range 0.030) and +0.041 at h4 (range 0.009), so on those two horizons the gap exceeds the seed spread, in the
unfavourable direction; C is worse by +0.031 at h4 (range 0.021); at every other horizon and arm the mean difference is inside
the seed range (C is lower than A by 0.021 at h2 against a range of 0.136, inside the spread). No arm is better than A by more
than its seed spread, so no improvement is reported. Lane B's gate (below) and lane C2's finding ("no feature or linear model beats
the zero-return naive on ETH 4h TRAIN") agree.

EURUSD 1h (git-pinned OHLC file 72b8271d..., lane B manifest 326aee0c, TRAIN 48,890 origins, VALIDATION 10,415, elapsed-second targets, windows
with a missing hour excluded: 16,101 train / 3,404 validation; 4 local-level features): zero-return MAE h1..h6 0.5345/0.7669/0.9505/1.1178/1.2685/1.4092,
persistence 0.7810/0.9713/1.1308/1.2758/1.4066/1.5380, seasonal-6 0.7850/1.1223/1.3790/1.6169/1.8347/2.0391, train-mean
0.5345/0.7669/0.9505/1.1179/1.2686/1.4093. A, B and C at the selected alpha (1e8) are the constant to four decimals; at fixed alpha
1e3 the mean MAE over horizons is A 1.0084, B 1.0082, C 1.0082, EWMA 1.0082, permuted 1.0092 (moments_train): differences of 2e-4,
below any interval worth reading. Two of the four price levels (LOW, HIGH) hit the closed-form floor (r clipped, K=1: pass-through).

## 6. Lane B's gate, run against the same recipe (`gate/GATE_LANE_B_ETH.worker_b.json`)

Local level on log close, q and r from fold-TRAIN moments, causal filter, deviation/innovation/level-change, redundancy filter
(only `deviation` survives), ridge alpha 1.0, 3 expanding inner folds, MAE AND MSE strictly below the zero-return naive AND the
intercept-only control. Mean model/zero MAE ratio at h1: 0.996545 (lane B reported 0.99654 on 0.0108431/0.0108808): a faithful
replication. Per fold at h1: fold 1 passes; fold 2 MSE 0.811686 vs zero 0.811338 (fails); fold 3 MSE 0.242743 vs intercept 0.242721
(fails). No horizon h1..h6 passes all folds. The gate result is REJECTED, as in lane B's ledger.

## 7. Diagnostics (moments_train unless stated)

* Floors: 8 of 13 level columns and 15 of 16 trend columns hit a closed-form floor (rolling-window features such as SMA/EMA/BB
  have difference autocorrelation from window overlap, lane B's caveat (a)): K is about 1 and the "filter" is a pass-through
  (median level retention 1.0, lag 0). The closed form cannot identify a slope variance far below its own standard error
  (tested and typed). With declared ratios nothing is clipped but the standardized innovation is strongly autocorrelated
  (Ljung-Box Q10 median 9,536 / 13,315 on validation): the model is mis-specified for these features.
* Innovation stability, train -> validation (std of zinnov, median over columns): level group 0.90 -> 0.68, trend group 0.93 -> 1.34
  (moments_train); 1.00 -> 0.77 and 1.00 -> 1.49 (declared 1e-2).
* Phase: median lag 0 bars (moments_train); 3 bars for the trend group at declared 1e-2. Extremes: excursions above 3 train std
  keep 100 percent of their amplitude in the level under moments_train (pass-through), 70 percent under declared 1e-2, and the
  innovation exceeds 2 sigma on 91 percent of them there.

## 8. DOIN hand-off (design + executable declaration)

`DOIN_KALMAN_SEARCH_SPACE_DESIGN.md`, `pack/DOIN_KALMAN_SEARCH_SPACE.v1.json` (sha256 db0a2bee94c2bfb9...,
design_sha256 88274bea...), `tools/h_kalman_search_space.py`, `tests/test_h_kalman_search_space.py` (8 tests): six flat parameters,
every one except `kalman.active` conditional; 77 conditional configurations instead of 360 Cartesian. Phase 1 = {inactive, one
promoted variant (moments_train, group both, append)}. Because the pilot shows no utility, the recommendation is to run phase 1 only as
a probe inside the differentiated model against the same model without it, and to open phase 2 only if that effect exceeds the seed
spread. Outputs enter the engine as named columns `<f>__kf_level|slope|innov|zinnov|logvar` (M01 contract, f0a850c8). Multivariate
Kalman was NOT started: the pilot did not justify widening.

## 9. Not verified

* No second-worker replay of the MLP arms and of the lane B gate (worker_a was reserved for GPU work).
* Ridge predictions are value-reproducible across workers (<= 3e-13) but not bitwise; the claim of bitwise replay is confined to the
  Kalman artifacts, outputs and eligible input matrices.
* Point-in-time availability of the inputs is the bar-close convention of lane B (DEVELOPMENT, uncertified).
* Utility inside the differentiated DOIN model, any trading effect, the TEST rows (never read), the lake EURUSD and GBPUSD variants.
* The first `lag24` arms use float32 matrices (7 digits) to bound memory; arithmetic downstream is float64.
* One deviation from the dispatch rules: a 0.15 s pytest run of the search-space tests on worker_b was started once directly
  (not through crispdm-run) before I noticed; every other worker command went through crispdm-run with a cap from a measured peak or,
  for the first pilot of each new workload, an a priori cap that was then replaced by 1.25 x the measured peak.

## 10. Reproduction

`/tmp`-free commands used (worker scratch, role names only): `crispdm-run -n <name> -m <1.25 x measured peak> -- python tools/h_kalman_eth_run.py
--view <ETH view csv> --feature-manifest inputs/SELECTED_FEATURE_MANIFEST.eth_4h.v1.FROZEN_DEVELOPMENT.json --candidates
inputs/KALMAN_CANDIDATE_FEATURES.v1.json --naive-table inputs/NAIVE_TABLE_validation_eth4h_l24_h6_v1.json --out <dir> --role <worker_x>`;
`tools/h_kalman_parity_proof.py`, `tools/h_kalman_eurusd_run.py`, `tools/h_kalman_eth_mlp_run.py`, `tools/h_kalman_gate_eth_run.py`
with the same pattern.

## 11. ECL testbed (coordinator order after the first return; worker_b, CPU, DEVELOPMENT)

Inputs, read-only, M04 v2: `data_ecl_l24_h24_v1` MANIFEST 16a3e535..., train.npz 9215b099..., validation.npz e3712565...
(digests checked at every run). 321 channels, L 24, H 1..24, z_train; TRAIN rows [0, 18412), validation origins 18411..21019
(2,609 origins x 321 channels), test rows from 21044 never present. The row panel is reconstructed exactly from the windows and
targets (every overlapping cell checked bitwise; rows sha 9e0b3c26...). Same-row seasonal-24 naive mean MAE 0.2479664,
equal to M04's 0.247966. Learners are channel-independent and pooled over the 321 channels, target = seasonal residual
y(t+h) - y(t+h-24), forecast = seasonal + predicted residual. Kalman local level per channel, fitted on TRAIN rows only, two
declared variants: `moments_train` (closed form: 299 of 321 channels clip r to the floor, i.e. near pass-through) and
`declared ratio 1e-1` (nothing clipped). Tools: `tools/h_kalman_ecl.py`, `tools/h_kalman_ecl_run.py`, tests `tests/test_h_kalman_ecl.py` (6).

Ridge (alpha chosen on an inner chronological hold-out of TRAIN; `ecl/RESULTS_ECL_ridge2.worker_b.json`), validation MAE z_train:

| h | seasonal-24 | persistence | A | B (ratio 1e-1) | C (ratio 1e-1) | C EWMA (ratio 1e-1) | B permuted | B noise | B (moments) | C (moments) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2476 | 0.2623 | 0.1966 | 0.1787 | 0.1799 | 0.1849 | 0.1963 | 0.1966 | 0.1676 | 0.1702 |
| 2 | 0.2477 | 0.4321 | 0.2338 | 0.2143 | 0.2140 | 0.2159 | 0.2333 | 0.2338 | 0.2094 | 0.2099 |
| 3 | 0.2477 | 0.5817 | 0.2458 | 0.2308 | 0.2304 | 0.2319 | 0.2452 | 0.2458 | 0.2287 | 0.2290 |
| 4 | 0.2478 | 0.7151 | 0.2524 | 0.2410 | 0.2405 | 0.2416 | 0.2516 | 0.2524 | 0.2403 | 0.2405 |
| 5 | 0.2478 | 0.8315 | 0.2563 | 0.2478 | 0.2473 | 0.2482 | 0.2555 | 0.2563 | 0.2478 | 0.2479 |
| 6 | 0.2478 | 0.9310 | 0.2587 | 0.2523 | 0.2518 | 0.2525 | 0.2578 | 0.2587 | 0.2528 | 0.2529 |
| 7 | 0.2478 | 1.0148 | 0.2604 | 0.2555 | 0.2551 | 0.2557 | 0.2594 | 0.2604 | 0.2564 | 0.2564 |
| 8 | 0.2477 | 1.0797 | 0.2611 | 0.2575 | 0.2572 | 0.2577 | 0.2601 | 0.2611 | 0.2585 | 0.2585 |
| 9 | 0.2477 | 1.1275 | 0.2614 | 0.2587 | 0.2584 | 0.2589 | 0.2604 | 0.2614 | 0.2597 | 0.2597 |
| 10 | 0.2477 | 1.1608 | 0.2618 | 0.2597 | 0.2595 | 0.2599 | 0.2609 | 0.2618 | 0.2606 | 0.2607 |
| 11 | 0.2477 | 1.1814 | 0.2621 | 0.2604 | 0.2602 | 0.2606 | 0.2614 | 0.2621 | 0.2613 | 0.2614 |
| 12 | 0.2477 | 1.1886 | 0.2622 | 0.2609 | 0.2606 | 0.2610 | 0.2617 | 0.2622 | 0.2619 | 0.2619 |
| 13 | 0.2477 | 1.1850 | 0.2626 | 0.2614 | 0.2612 | 0.2616 | 0.2623 | 0.2626 | 0.2625 | 0.2624 |
| 14 | 0.2478 | 1.1681 | 0.2628 | 0.2618 | 0.2616 | 0.2620 | 0.2629 | 0.2628 | 0.2629 | 0.2629 |
| 15 | 0.2479 | 1.1386 | 0.2629 | 0.2621 | 0.2619 | 0.2622 | 0.2633 | 0.2629 | 0.2633 | 0.2633 |
| 16 | 0.2480 | 1.0945 | 0.2631 | 0.2623 | 0.2620 | 0.2623 | 0.2638 | 0.2631 | 0.2636 | 0.2636 |
| 17 | 0.2481 | 1.0335 | 0.2631 | 0.2621 | 0.2618 | 0.2621 | 0.2643 | 0.2631 | 0.2637 | 0.2637 |
| 18 | 0.2481 | 0.9547 | 0.2631 | 0.2618 | 0.2615 | 0.2617 | 0.2648 | 0.2631 | 0.2638 | 0.2638 |
| 19 | 0.2482 | 0.8612 | 0.2631 | 0.2614 | 0.2611 | 0.2611 | 0.2652 | 0.2631 | 0.2638 | 0.2639 |
| 20 | 0.2483 | 0.7524 | 0.2631 | 0.2607 | 0.2603 | 0.2602 | 0.2654 | 0.2632 | 0.2638 | 0.2639 |
| 21 | 0.2484 | 0.6303 | 0.2632 | 0.2596 | 0.2592 | 0.2590 | 0.2656 | 0.2632 | 0.2637 | 0.2638 |
| 22 | 0.2486 | 0.4984 | 0.2632 | 0.2584 | 0.2579 | 0.2577 | 0.2657 | 0.2632 | 0.2632 | 0.2633 |
| 23 | 0.2487 | 0.3619 | 0.2630 | 0.2575 | 0.2569 | 0.2568 | 0.2655 | 0.2630 | 0.2609 | 0.2609 |
| 24 | 0.2488 | 0.2488 | 0.2604 | 0.2578 | 0.2572 | 0.2565 | 0.2628 | 0.2604 | 0.2567 | 0.2567 |
| mean | 0.2480 | 0.8514 | 0.2569 | 0.2519 | 0.2516 | 0.2522 | 0.2573 | 0.2569 | 0.2524 | 0.2525 |

alphas chosen on the inner hold-out: {'declared_1e-1': {'A': 100000.0, 'B': 1000000.0, 'B_NOISE': 100000.0, 'B_PERMUTED': 100000.0, 'C': 1000000.0, 'C_EWMA': 1000000.0, 'C_SMOOTHER_NONCAUSAL': 100.0}, 'moments_train': {'A': 100000.0, 'B': 1.0, 'B_NOISE': 100000.0, 'B_PERMUTED': 100000.0, 'C': 1.0, 'C_EWMA': 1.0, 'C_SMOOTHER_NONCAUSAL': 100000.0}}
smoother (non-causal, never eligible) mean MAE: {'declared_1e-1': 0.23809, 'moments_train': 0.25689}

Paired against A (block bootstrap over origins, L = 48, plus the range of four contiguous-quarter deltas):
* ratio 1e-1: B and C are lower than A at all 24 horizons beyond both the interval and the quarter range; C minus A is
  -0.0167 at h1 [-0.0177, -0.0154], -0.0069 at h6, -0.0033 at h24; mean over horizons -0.0054 (2.1 percent of A).
* moments_train: C minus A -0.0264 at h1 [-0.0286, -0.0240], beyond spread at 12 of 24 horizons, unfavourable beyond spread at 5.
* Controls: B with permuted Kalman blocks moves A by -0.0003 at h1 and +0.0023 at h24; B with noise blocks equals A (weight zero).
  The non-causal smoother is lower still (mean 0.2381 at ratio 1e-1): the leak, never eligible.
* Against the comparable causal EWMA (same bandwidth), C is lower at h1..h14 beyond spread at ratio 1e-1 (-0.0050 at h1,
  shrinking to -0.0003 at h14) and higher at h24 (+0.0007); at moments_train lower at h1..h11 and h22..h24.
* Against the seasonal naive, the ridge arms are lower only at h1..h4/h5 (C 0.1799 vs 0.2476 at h1) and higher from h6 on:
  pooled ridge is a weak learner here (A mean 0.2569 > seasonal 0.2480).
* Same-host rerun: predictions bitwise identical (two runs, 28 prediction digests).

Flatten+MLP control learner (hidden 64-64, MAE, AdamW, batch 256, 30 epochs max, patience 5 on an inner TRAIN hold-out;
moments_train; 150,000 sampled TRAIN (origin, channel) rows; seeds 2021-2024; `ecl/RESULTS_ECL_mlp.worker_b.json`),
mean MAE over the four seeds:

| h | seasonal | A | B | C | C EWMA | B permuted |
|---|---|---|---|---|---|---|
| 1 | 0.2476 | 0.1697 | 0.1582 | 0.1582 | 0.1571 | 0.1850 |
| 2 | 0.2477 | 0.2026 | 0.1933 | 0.1919 | 0.1908 | 0.2133 |
| 3 | 0.2477 | 0.2170 | 0.2089 | 0.2095 | 0.2078 | 0.2255 |
| 4 | 0.2478 | 0.2245 | 0.2194 | 0.2185 | 0.2178 | 0.2317 |
| 5 | 0.2478 | 0.2289 | 0.2248 | 0.2245 | 0.2248 | 0.2357 |
| 6 | 0.2478 | 0.2326 | 0.2294 | 0.2297 | 0.2303 | 0.2385 |
| 12 | 0.2477 | 0.2433 | 0.2447 | 0.2449 | 0.2442 | 0.2465 |
| 18 | 0.2481 | 0.2456 | 0.2475 | 0.2489 | 0.2464 | 0.2499 |
| 24 | 0.2488 | 0.2428 | 0.2412 | 0.2416 | 0.2404 | 0.2477 |
| mean h1..h24 | 0.2480 | 0.2355 | 0.2346 | 0.2347 | 0.2333 | 0.2411 |

(all 24 horizons are in the JSON). A beats the seasonal naive at every horizon. B and C are lower than A at h1..h6 for every
seed, by more than the seed range (h1: -0.0115, seed range of A 0.0019); at h9..h22 they are higher than A. The comparable
causal EWMA arm is at least as good as C (C minus EWMA at h1: +0.0023, +0.0008, +0.0005, +0.0007 over the four seeds) and has the
lowest mean (0.2333). The permuted-capacity control is worse than A (0.2411): the gain of B and C is not extra width.

Reading. On ECL the coordinator's condition is met at the short horizons: B and C beat A by more than the seed and fold spread
at h1..h6 (MLP, every seed) and h1..h24 (ridge, ratio 1e-1). The condition is NOT met for anything Kalman-specific with the MLP:
the comparable causal EWMA gives the same or a larger gain. With ridge, C beats the EWMA at h1..h14 by 0.0003-0.0050. So the
effect is "smoothed level / innovation channels help at short horizons", and the Kalman filter is one of two ways to produce
them. Recommendation for the DOIN phase-1 probe (after the financial cells, GPU): pair three arms, without operator, Kalman
(moments_train, group both, append) and the causal EWMA at the Kalman gain, on M04's ECL rows and seeds, and judge h1..h6
separately from h7..h24.

Not verified on ECL: no replay on worker_a (the ridge peak is 1.67 GB and the MLP peak 2.90 GB, above the 300 MB that worker_a
allows for this lane); the Kalman layer itself was shown bit-portable on ETH/EURUSD. Costs: ridge 695-719 s CPU per full run
(2 variants x 7 arms), cgroup peak 1.66-1.67 GB; MLP 1,147 s CPU, cgroup peak 2.90 GB. Three earlier MLP/ridge attempts were stopped by
me (memory.high stalls or a cap kill) and replaced by memory-reduced successors at the same caps; none of their numbers are used.
