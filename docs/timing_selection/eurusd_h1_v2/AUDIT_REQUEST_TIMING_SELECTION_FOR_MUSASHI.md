# Audit request for Musashi: TIMING_PROXY_SELECTION results (EURUSD H1, DEVELOPMENT)

From: the SELECTION lane (Satoshi). To: Musashi (auditor role). Date: 2026-10-19.
Answers: your plan audit REVISE_BEFORE_FINAL_SEAL (predictor `codex/data-coverage-repair-20261018` @ed5b1acc).
Sealed object: `SELECTION_MANIFEST.json` in this directory, schema `feature_selection.manifest.v1`,
`selection_name: TIMING_PROXY_SELECTION`, status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`,
`final_selection: false`, `proxy_only: true`, manifest sha256
`4fd6710e5a1fa746d6b1303ca871dbbe3c81fc516638e04eb352300c6bcbf2d0`. Every file here is listed in
`SHA256SUMS`. No claim of RL value is made. The `feature_selection.decision.v1` record is yours to write
or refuse (decision id `TIMING_PROXY_SELECTION:timing-selection:2c6689c8f93232d7:decision`, decider role
`auditor`).

## 1. Purpose

Select inputs for expert-guided (expert-demonstration / hindsight-guided) learning. The expert is the
optimized ZigZag signal (Depth 3 / Deviation 36.3 pips / Backstep 2). Its peaks and valleys are the close
tick of one order and the open tick of the next. This is a TIMING task, not price prediction and not
human-preference RLHF. The result is usefulness for the pivot-classification proxy and its trivial
reversal strategy only. The fixed-budget SAC comparison of the 15 inputs against this subset is a later,
separately authorized integration.

## 2. Targets and their construction

- **Expert:** heuristic-strategy `app/short_horizon/zigzag.py`, sha256 `b30788e6fd591c6f49fb7661e2eb54373166131d9734b46efba7ebbffcb4d364`
  (branch `satoshi/short-horizon-strategy-20261018` @b9b4f9c), loaded by path with the digest verified.
- **Lake:** `market_data/forex/g10/eurusd/1h.parquet` sha256 `c6628375b2872f16dcdbd6b1b90ed8e68bc5bdd852efdef01f2515b8712d4e2e`.
  It is loaded 2019-12-01 to 2025-01-01 exclusive (30,854 bars) and TEST is refused at read.
- **Pivots:** 6,371 final pivots, of which 6,191 are confirmed by the expert's own replay rule and 180 never are.
- **Semantics:** `timing_targets.v2`, target identity `TARGETS.json:identity`.

| Target | Definition | Role |
|---|---|---|
| T1 `timing_pivot_k2` (+ `timing_valley_k2`, `timing_peak_k2`) | earliest final pivot in (t, t+2] (t excluded, t+2 included; k counts OBSERVED market bars); 1 valley, 2 peak, 0 none; a same-bar tie is refused; a second opposite pivot in the window is flagged and does not change the class; every hourly origin kept | PRIMARY, anticipatory |
| T2 `timing_action_confirmed` | +1 / -1 on the bar where the expert's confirmation locks a final valley / peak, 0 elsewhere | reactive comparator, reported separately |
| T3 `timing_expert_leg_remaining_pips` | for an agent HOLDING the expert's current leg: remaining gross pips from close(t) to the next final pivot level; no costs, no SL/TP, not full-trade P&L | state-conditional comparator |

Label support (F01) is recorded per row in `LABEL_LEDGER.parquet` (in the snapshot, not in Git). Each row
carries origin time, origin availability (bar end), event time, support end and label availability per
target, `expert_trade_id`, `label_group_id` and status.

- **Support rule:** the confirmation bar of the first final pivot after t+k (T1); the next pivot's
  confirmation for T2/T3.
- **Verification:** truncation and seeded future-perturbation replays at every cut: every row declared
  final is unchanged (`TARGETS.json:finality`). Rows not declared final do change, so the rule is not vacuous.
- **Censoring:** rows whose certifying pivot never confirms are CENSORED, never NONE or zero.
- **Purging:** rows whose support reaches a cut are PURGED. Cuts: 2021-01-01, 2022-01-01, 2023-01-01,
  2023-07-01, 2024-01-01, 2025-01-01.

Every hourly origin is accounted for:

| Period | Origins | T1 labelled | censored | purged | window crosses a gap |
|---|---:|---:|---:|---:|---:|
| TRAIN 2020-2022 | 18,688 | 18,070 | 579 | 39 | 324 |
| INNER_A 2023H1 | 2,445 | 2,381 | 54 | 10 | 756 |
| INNER_B 2023H2 | 2,964 | 2,874 | 75 | 15 | 207 |
| 2024 | 6,243 | 6,089 | 154 | 0 | 114 |

## 3. Candidate pool and provenance (F04)

The ordered inventory has 30 inventory + 47 price pool + 12 fractal = 89 columns (`INVENTORY.json`,
columns digest inside). The plan's "88" was my miscount of the price pool (47, not 46). Eight exact alias
groups were found by phase 2, e.g. `cal.hod_sin` = `hod_sin` and `px.bar_body` = `RAW_BAR_LOG_RETURN`.

| Source | Provenance |
|---|---|
| Inventory (30) | the admitted-core candidate code, loaded by path: financial-data `_scripts/core_selection_bundle.py` @41f5af170 (sha256 `f7bf19f8...`) and feature-eng `normalize_phase1b.py` @d081d0f (`17a33c44...`). It is computed on the LAKE bars, not on the repaired admitted root (see section 9) |
| Price pool (47) | predictor `tools/timing_price_pool.py`: returns, ranges, close location, efficiency ratio and realized vol at 1/2/4/6/12/24/48 observed bars; ATR and ATR ratio; range ratio; bar body; elapsed hours; hour, weekday and session encodings on the LAKE LABEL CLOCK (no UTC claim, no hand shift) |
| Fractal (12) | the fractal lane's store, manifest sha256 `906578e1c96ddd1aa2024a455467db91db623c6583a351ea2468f6d57cb2740f` (producer feature-eng `app/plugins/causal_fractal.py` @6c028aa). Keyed by (EURUSD, timestamp, lake bar convention, lake bytes): the store manifest names the same lake sha256, and its close equals the lake close on all 30,340 joined rows |
| Baseline | your admitted-core 15 (identity `553fc047...`), unchanged and scored on identical rows; its members also compete in the pool |
| 19-feature manifest `2ff34fa2...` | NOT_COMPARABLE: its INPUT.npz exists only in a private deployment directory and cannot be regenerated from the lake |

Quality and causality checks (`QUALITY_PROFILES.json`, `PERTURBATION.json`):

- **Quality:** every column profiled for coverage, duplicates, finite share, warm-up, interior non-finite
  values, extrema, distinct values, constants and window unit.
- **Existing profiler:** the existing causal-inference profiler `profile_series` was run on all 59
  new-family columns.
- **Prefix invariance and future perturbation:** PASSED for the inventory generator, the price pool, the
  fractal producer (windows 48/96/200) and the evaluator's TRAIN-only scaler.

## 4. Stages, engines and what needed code

| Stage | Engine (file@commit) | Unchanged / new |
|---|---|---|
| population manifest, phases 2 and 3 | predictor `fs_phase23_manifest`, `feature_pairwise_campaign`, `feature_pairwise_worker`, `feature_filter_selection` @15893493 (nine methods) | unchanged; warehouse = `fs_phase23_warehouse_adapter` sqlite3 backend (duckdb is not installed here), readback reconciled |
| phase-1 profiler | causal-inference `feature_selection_profile.profile_series` @a4cb9ea | unchanged |
| targets, ledger, finality, pool, checks, screens, recipe, DAG, manifest, publication, snapshots | predictor `tools/timing_pivot_targets.py`, `tools/timing_price_pool.py`, `tools/timing_selection.py` (branch `satoshi/timing-selection-20261019`) | new |
| causal stage | causal-inference `fs_timing_pack.py` (branch `satoshi/timing-target-pack-20261019`), reusing `fs_causal.hac_partial_test`, `event_impact.match_controls` and tigramite 5.2.7.0 `PCMCI.run_pcmciplus` + `RegressionCI` | wrappers new, estimators unchanged; `feature_selection_worker.py` extended append-only (two literal lines) |
| predictive validation | heuristic-strategy `tools/short_horizon_timing_eval.py` (branch `satoshi/timing-selection-eval-20261019` @0eb4cc4), reusing `Ctx.run` / `execution.run_backtest` and the pivot-detector helpers | evaluator new, adapter unchanged |

Selection decisions (all on TRAIN 2020-2022, 18,688 rows, before any evaluator read):

- **Recipe (T1 task only, per your F05):** the MRMR K=12 union over T1 and its two direction indicators
  gives 19 features.
- **Screen:** 12 of the 19 fail the T1 Kruskal-Wallis screen at BH q 0.1.
- **Cluster pruning:** one representative per TRAIN Spearman 0.9 cluster pruned nothing.
- **Result:** 7 selected (frozen in `FROZEN_SUBSET.json`).
- **Comparators:** the T2 and T3 MRMR K=12 subsets are reported in `REPORT.json` and were not used.

Fold stability: the rank Spearman of the screen versus TRAIN is 0.809 (prefix before 2021) and 0.945
(before 2022).

## 5. The selected manifest

Selected, in order: `RAW_BAR_LOG_RETURN`, `cal.hod_sin`, `cal.session_europe`, `px.atr_ratio_14_120`,
`px.close_loc_2`, `px.logret_24`, `variogram_hurst_w96`. All 89 features carry a state and reason codes.
Gate check on this manifest (`GATE_CHECK.json`):

- Refused without a decision record (`MISSING_DECISION_RECORD`).
- Admitted with a SYNTHETIC independent-role record. This proves conformance only and is not a decision.
- Refused when a rejected feature is consumed.

## 6. E0 imitability: the 7 versus the 15 on identical rows

**Rows.** Rows are identical for both sets and every arm: a full finite 48-bar window over the union of
the 22 compared columns, plus a labelled T1. That gives TRAIN 18,023, INNER_A 2,381, INNER_B 2,874 and
2024 6,089. The 15 lost 47 TRAIN rows to the union's longest warm-up (fractal w96); no other period lost
rows.

**Decision rule.** INNER_A (2023H1) chose the early-stopping checkpoint, the temperature, the
regularization cell and the threshold. INNER_B (2023H2) was scored with all of these frozen. 2024 was
read once after `FROZEN.json` (`freeze_sha256` `50952d2b...`) and is PREVIOUSLY OBSERVED. 2023 is JOINT
development, because the expert was chosen with it.

**Metric definitions.** Seed means; per-direction rates include NONE in the denominator. Sharpe =
`SHARPE_DEFINITION` in `eval/SEAL.json`: daily calendar returns including closures, times sqrt 365.

| Set | Arm (chosen cell) | Period | PR-AUC valley / peak (base 0.19) | ROC-AUC valley / peak | log-loss | precision long / short | Sharpe | net USD | MDD % | entries |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|
| 7 | m1 (dropout 0, wd 1e-4) | INNER_B | 0.327 / 0.356 | 0.700 / 0.728 | 0.996 | 0.360 / 0.369 | -2.67 | -5,015 | 55.0 | 405 |
| 15 | m1 (dropout 0, wd 1e-4) | INNER_B | 0.312 / 0.331 | 0.686 / 0.704 | 1.005 | 0.359 / 0.344 | -1.49 | -3,932 | 46.4 | 300 |
| 7 | m1 | 2024 | 0.348 / 0.347 | 0.714 / 0.718 | 0.990 | 0.405 / 0.402 | -1.51 | -4,203 | 56.8 | 803 |
| 15 | m1 | 2024 | 0.330 / 0.319 | 0.694 / 0.692 | 1.002 | 0.417 / 0.383 | -0.96 | -3,678 | 55.1 | 575 |
| 7 | m2 (dropout 0.2, wd 1e-4) | INNER_B | 0.354 / 0.383 | 0.716 / 0.738 | 1.003 | 0.403 / 0.417 | -1.46 | -3,448 | 51.0 | 402 |
| 15 | m2 (dropout 0.1, wd 1e-3) | INNER_B | 0.325 / 0.350 | 0.702 / 0.720 | 0.998 | 0.381 / 0.396 | -2.35 | -4,767 | 54.7 | 334 |
| 7 | m2 | 2024 | 0.366 / 0.363 | 0.719 / 0.724 | 1.001 | 0.421 / 0.424 | -2.08 | -5,916 | 66.7 | 783 |
| 15 | m2 | 2024 | 0.339 / 0.319 | 0.699 / 0.692 | 1.003 | 0.442 / 0.384 | -1.57 | -5,124 | 61.6 | 638 |
| 7 | m3 (l2 0) | INNER_B | 0.331 / 0.375 | 0.702 / 0.734 | 0.980 | 0.392 / 0.476 | -0.57 | -1,985 | 27.2 | 135 |
| 15 | m3 (l2 0) | INNER_B | 0.336 / 0.356 | 0.702 / 0.720 | 0.970 | 0.491 / 0.437 | -0.55 | -1,790 | 39.3 | 108 |
| 7 | m3 | 2024 | 0.353 / 0.344 | 0.715 / 0.716 | 0.983 | 0.436 / 0.397 | -0.59 | -2,928 | 49.5 | 245 |
| 15 | m3 | 2024 | 0.342 / 0.328 | 0.707 / 0.698 | 0.978 | 0.427 / 0.452 | -0.12 | -1,482 | 49.5 | 208 |
| no-trade comparator | - | every period | - | - | - | - | 0.00 | 0 | 0 | 0 |

Paired differences (7 minus 15), seed-mean predictions and returns, ISO-week block bootstrap with 2,000
replicates (`eval/UNCERTAINTY.json`):

| Arm | Period | delta log-loss [95 %] | delta Sharpe [95 %] |
|---|---|---|---|
| m1 | INNER_B | -0.0083 [-0.0159, 0.0000] | [-4.08, 1.50] |
| m1 | 2024 | -0.0108 [-0.0170, -0.0046] | [-3.17, 2.07] |
| m2 | INNER_B | +0.0069 [-0.0004, 0.0141] | [-1.21, 4.76] |
| m2 | 2024 | +0.0007 [-0.0044, 0.0059] | [-2.37, 1.51] |
| m3 | INNER_B | +0.0096 [-0.0016, 0.0215] | [-3.66, 3.76] |
| m3 | 2024 | +0.0050 [-0.0041, 0.0138] | [-2.83, 1.98] |

Reading:

1. Both sets classify the pivot above the base rates: PR-AUC about 0.31-0.37 against 0.19, ROC-AUC
   about 0.69-0.74. This is unlike the earlier detector on the 19-feature arms (AUC 0.50-0.55), although
   those were different inputs and different origins.
2. With 7 features instead of 15, the selection matches the baseline on calibrated log-loss. The one
   interval that excludes zero favours the 7 (m1, 2024); the others include zero.
3. Every operating point of every set and arm loses money in the trivial reversal strategy, with Sharpe
   below the no-trade comparator. The Sharpe differences between sets are not resolved, and the DSR
   supplement is 0.00 for every cell (12 trials per set and arm).
4. Classification usefulness for the proxy is established. Strategy usefulness is not.

Regularization grid (owner decision, declared; cell rule: lowest mean INNER_A loss, ties by gap):

- **m1:** dropout 0 replaced the primary (0.1, 1e-4) in both sets.
- **m2:** (0.2, 1e-4) won for the 7 and (0.1, 1e-3) for the 15.
- **m3:** l2 0 kept; every m3 fit ran to its 400-iteration budget, with a TRAIN-INNER gap of about 0.35.

The per-cell tables, per-epoch losses and stop causes are in `eval/FROZEN.json` and in the DEVELOP
records (snapshot). In the trial ledger, m3 counts as one fit per cell: without subsampling its three
seeds are bit-identical.

## 7. Redundancy, stability and within-day variance

Full tables are in `TABLES.md`.

- **VIF:** VIF of the selected 7 is at most 1.79 (`RAW_BAR_LOG_RETURN`). The largest pairwise |Spearman|
  inside the selected 7 is 0.79.
- **Fractal versus realized volatility:** Spearman of the fractal columns against the realized-vol family
  (px.rv_*, ATR, px.atr_14, rolling_std_24) is |rho| <= 0.30, the largest being `variogram_hurst_w48`
  against `rolling_std_24`.
- **Constant within a day:** `cal.dow_cos`, `cal.dow_sin`, `dow_cos`, `dow_sin` (none selected).
  `variogram_hurst_w96` has a within-day variance ratio of 0.06: it is slow.
- **Multi-member clusters:** 12 TRAIN Spearman clusters have more than one member, e.g. [CLOSE, HIGH,
  LOW, OPEN] and [ATR, ATR_ratio, px.atr_14].

## 8. Causal stage: CONDITIONAL_PREDICTIVE_ASSOCIATION, not intervention (F07)

PCMCI+ with tigramite 5.2.7.0 RegressionCI, tau_max 2, pc_alpha 0.05, H = [px.logret_24, px.rv_24]:

- **Typing:** T1 is ONE categorical variable; binary candidates are typed discrete; continuous columns are
  standardized (an affine transform, invariant for the test).
- **Lag-0 mapping:** lag 0 = the completed bar t against the label of (t, t+2], which is time-ordered
  x -> y.
- **Scope:** 42 candidates (the T1 FDR-pass set plus the selected) and the two history members, each
  tested against H without itself: 44 records.
- **Known controls:** the positive control (a lag-1 label leak) is detected as LAGGED_PARENT; the
  negative noise and circular-shift controls show NO_LINK.
- **Temporal halves:** 36 of 45 verdicts (42 candidates + 3 controls) agree across the two TRAIN halves.
- **Verdicts:** LAG0_TIME_ORDERED_LINK 26, NO_LINK 13, CONFLICTING_OR_REVERSED 4, LAGGED_PARENT 1
  (`px.rv_1`).

| Selected feature | mixed CI verdict (best link) | halves | HAC lagged parent (peak / valley) | matched-control excess pivot rate |
|---|---|---|---|---|
| RAW_BAR_LOG_RETURN | LAG0 link, o-o, p 1.9e-51 | agree | yes / yes | +0.033, significant |
| cal.hod_sin | NO_LINK | agree | no / yes | too few matched events |
| cal.session_europe | NO_LINK | agree | no / no | too few events |
| px.atr_ratio_14_120 | NO_LINK | agree | no / yes | +0.003, not significant |
| px.close_loc_2 | LAG0 link, o-o, p 1.8e-109 | disagree (LAGGED_PARENT / LAG0) | no / no | +0.084, significant |
| px.logret_24 | CONFLICTING_OR_REVERSED (lag 0, <--, p 1.5e-25) | not run (history member) | not run (history) | not run (history) |
| variogram_hurst_w96 | NO_LINK | agree | no / yes | +0.064, not significant |

No feature was rejected for an unresolved causal status.

## 9. Limitations and what I suspect

- **The 2023 lake is incomplete.** 2023H1 has 559 intra-week gaps (633 missing hours) and 2023H2 has
  148 missing hours, against at most 38 in any other half-year. INNER_A, where every evaluator choice is
  made, therefore runs on a gappy grid: 756 of its 2,445 origins have a k-window that crosses a gap, and
  the expert's pivots there are computed on incomplete bars. This is the known data-integrity hold; the
  repaired admitted root is not on this host.
- **The repaired admitted root is not used.** The inventory columns, including your 15, are recomputed on
  the lake bytes, so the 15 here are not byte-identical to the admitted-core inputs. Calendar encodings
  are on the lake label clock (Sunday 17:00 weekly open), not on certified UTC.
- **Fill latency.** The pinned adapter fills two bars after the signal bar (`signal_blocks` starts at b+1
  and `RealisticProfile.entry_delay_bars = 1`). With k = 2, a pivot at b+1 has already passed at the
  fill. The strategy therefore tests a later entry than the label anticipates, which may explain part of
  the negative Sharpe.
- **The 12 h no-reopen rule is missing from `zigzag.signal_blocks`** (test AC02). The evaluator adds a
  guard; the adapter itself is unchanged.
- **The screen.** The T1 Kruskal-Wallis screen dropped 12 of the 19 union features, including all four
  efficiency ratios and five fractal columns. A location test may be the wrong gate for a timing target.
- **The selected 7 contains two calendar features** on an uncertified clock, and one slow fractal column
  that has no association beyond HAC in the causal screens.
- **m3 never stops under the monitor** (budget-exhausted in every cell), so the capacity grid did not
  bind. Its seeds are not independent.
- **Several engines did not run with their pinned dependencies.** Phase 2/3 used the sqlite adapter
  instead of DuckDB. Neural fits ran on a GPU with TF 2.21 and were reloaded on CPU with Keras 3.15.1
  (load verified).
- **Technical failures of my own,** recorded in `eval/COMPUTE_LEDGER.json` and `READ_LEDGER_ADDENDUM.json`:
  - Model files collided (`Path.with_suffix`). The four CPU neural shards were superseded and rerun on
    the GPU; the fix has test TE09.
  - The first freeze failed on a file-name mismatch after writing `FROZEN.json`. The rerun reproduced an
    identical `freeze_sha256`.
  - The first reconfirm failed at import (missing pydantic) before opening any data.
- **The second snapshot copy is on another partition of the same physical disk.** An independent physical
  copy is a missing object (section 11).

## 10. What to attack first (my weakest points)

1. INNER_A sits on the 2023 coverage gap. Every evaluator choice (cell, temperature, threshold) was made
   on a grid missing about a fifth of its hours.
2. The fill latency against k = 2: the reversal strategy cannot exploit the anticipated pivot. A negative
   Sharpe may say more about the order path than about the features.
3. The Kruskal-Wallis screen as the gate of the MRMR union, and the one-representative-per-cluster rule
   that kept two calendar encodings.
4. T1's support rule depends on the expert's confirmation replay. 180 final pivots never confirm and
   their rows are censored, which removes rows that are not missing at random.
5. The causal stage's thin H and the CONFLICTING_OR_REVERSED lag-0 orientation of `px.logret_24`, a
   selected feature that is also in H.
6. Equal treatment of the two sets on the regularization grid. The 7 and the 15 ended up with different
   m2 cells chosen on INNER_A.

## 11. What was NOT done

- No independent decision record; no RL training or gym-fx run; no SAC comparison.
- Not rerun with the NSGA-II rerun's expert parameters, with another k, or on another instrument.
- Not built on the repaired admitted root (missing object). Next command, once the bundle exists:
  `python tools/timing_selection.py run --config <CFG with candidates.inventory pointing at the admitted bundle> --out <NEW_OUT> --reuse-from <THIS_OUT>`.
- No independent physical second copy (missing object). Next command:
  `rsync -a --checksum <SNAPSHOT>/ <secondary-worker>:<path>/ && ssh <secondary-worker> 'cd <path> && sha256sum -c SHA256SUMS'`.
- No causal ladder rungs 2-3 (episodes, SCM counterfactuals) for the timing targets; three screens only.
- No dataset_selection `upstream.lock.json` repin (the owners merge first).
- The 19-feature manifest comparison (NOT_COMPARABLE, section 3).
