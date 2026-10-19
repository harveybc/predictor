# Audit request for Musashi (v3): TIMING_PROXY_SELECTION with the ZigZag v2 expert (EURUSD H1, DEVELOPMENT)

From: the SELECTION lane (Satoshi). To: Musashi (auditor role). Date: 2026-10-19.
Supersedes the v2 request in `docs/timing_selection/eurusd_h1_v2/` (kept, marked `SUPERSEDED.json`).

Sealed object: `SELECTION_MANIFEST.json` in this directory, schema `feature_selection.manifest.v1`,
`selection_name: TIMING_PROXY_SELECTION`, status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`,
`final_selection: false`, `proxy_only: true`, manifest sha256
`b4da1012c53fd21003ec36f1d0c9f0e79354db1661444013168f86592fc2cced`.

The decision record is yours to write or refuse: decision id
`TIMING_PROXY_SELECTION:timing-selection:47d1e3fca7d01d2a:decision`, decider role `auditor`.
The gate refuses the manifest without it (`GATE_CHECK.json`). Purpose, terminology and the proxy-only scope
are unchanged from v2: expert-demonstration / hindsight-guided learning for TIMING, not price prediction
and not human-preference RLHF. No RL value is claimed.

## 1. What changed since v2 and why

| Change | Why | Effect |
|---|---|---|
| **Expert v2** (heuristic-strategy `satoshi/short-horizon-strategy-20261018` @abbfaa6, `ZIGZAG_EXPERT_STUDY.md` section 8, FINAL.json `A4_wide`): Depth 3 / Deviation 40.7 pips (407 points) / Backstep 0, chosen at cap 4/day by 2023 Sharpe; module sha256 `fc24c6f2...`; demonstrations v2 `38125b0b...` (v1 `352ddf6a` kept) | the full NSGA-II replaced the 16 x 15 search whose choice sat on a bound | new labels: 7,038 final pivots (v2 run: 6,371) |
| **Confirmation definition** `confirmation_bars_fast` (first opposite candidate + Backstep, brute-force verified by its owner), configurable in the targets module and part of the target identity | the v1 rule waited for the next FINAL pivot | 1 of 7,038 pivots unconfirmed (v2 run: 180 of 6,371); censoring nearly disappears. Finality holds under the new definition: every row declared final is unchanged under truncation and future perturbation at every cut (`TARGETS.json:finality`; real-module test `test_a03_finality_holds_with_the_v2_confirmation_definition`) |
| **Executable fill path** in the evaluator: the owner's `zigzag_opt.Evaluator.run(..., executable=True)` with `executable_profile()`: decision after the completed bar b, fill at the open of b + 1, entries before the operational start (12 h after a reopen) refused by the adapter; cap 4/day | resolves the v2 finding that the pinned path filled at b + 2 (with k = 2 the anticipated pivot had already passed) | test AC10; every score row carries `fill_path: EXECUTABLE_B_PLUS_1` and the adapter's refusal count |
| **Period labels**: 2023 JOINT development, 2024 PREVIOUSLY OBSERVED development | owner relabel; unchanged meaning | `READ_LEDGER*.json` |
| **DAG rerun (F08)**: `run --reuse-from v2`; REUSE of pool_features, quality_profiles, perturbation_checks and within_day_variance (the pool parquet is byte-identical to v2; within-day variance recomputed and identical); RECOMPUTE of every target-dependent stage | the expert hash and semantics changed | `DAG.json` plan |
| **Independent physical copy** of the v2 and v3 snapshots on the secondary worker (`rsync --checksum`, then `sha256sum -c` there: ALL_OK 31 and 32 files) | missing object in v2 | `PHYSICAL_COPY.json` |
| **Admitted root found** (read-only, on another worker): `ADMISSION.json` (schema `financial_data.core_price_inventory.v1`, ADMITTED, EURUSD `eurusd.parquet` sha256 `c80847bb...`, 0 missing required operating hours, UTC bar start, availability +1 h). Copied and verified against its receipt | the v2 request asked for it | witness only (section 3); the join is still a missing object |

Code: predictor `satoshi/timing-selection-20261019`; heuristic-strategy `satoshi/timing-selection-eval-20261019`
(merges abbfaa6); causal-inference unchanged since v2 (a379a67).

## 2. Targets (v3)

The semantics are unchanged (`timing_targets.v2`), now with confirmation `confirmation_bars_fast`. T1 is
primary and anticipatory: the earliest final pivot in (t, t+2], k in observed bars, tie refused. T2 and
T3 are comparators, as in v2.

| Period | Origins | T1 labelled | censored | purged | two kinds in window |
|---|---:|---:|---:|---:|---:|
| TRAIN 2020-2022 | 18,688 | 18,659 | 0 | 29 | 528 |
| INNER_A 2023H1 | 2,445 | 2,440 | 0 | 5 | 76 |
| INNER_B 2023H2 | 2,964 | 2,952 | 0 | 12 | 96 |
| 2024 | 6,243 | 6,236 | 7 | 0 | 190 |

## 3. The clock and the admitted root (F04): measured, not applied

`CLOCK_AND_COVERAGE_WITNESS.json`: rows before 2025 only, filtered at read; no selection input was
shifted.

- **Offset scan.** Comparing the lake label + h hours with the admitted UTC bar start for h in -8..+8,
  the lake close equals the admitted bid close exactly at **+4 h** (median absolute difference 0.0,
  return correlation 0.55). +5 h is nearly exact (0.00011, 0.47). Every other offset is near zero
  correlation.
- **What it means.** The lake labels look like New York local time with daylight saving (EDT +4, EST +5)
  written as UTC. No single shift is right all year, which is exactly why a hand shift is forbidden. A
  join needs a pinned, DST-aware clock identity from the data owner.
- **Coverage.** The admitted root has 3,111 rows in 2023H1 against the lake's 2,445; the 2023 coverage
  gap is the lake's.
- **Why the rebuild is not done.** The pinned strategy adapter (`Ctx`, `load_inputs`) is bound to the
  lake file's digest and to the lake-derived session calendar, so a selection rebuilt on the admitted root
  also needs an adapter and calendar for the admitted source. That remains a missing object (section 9).

## 4. Pool, stages, selection

The pool is the same 89 columns as v2 (reused byte-identical). The engines ran unchanged, as in v2.

| Step | Features |
|---|---|
| Recipe (T1 task only): MRMR K = 12 union | 16 |
| Dropped by the T1 FDR screen | 10: RAW_LOG_RANGE, box_counting_dimension_w48, cal.dow_cos, hurst_rs_w200, hurst_rs_w48, hurst_spectrum_width_w200, hurst_spectrum_width_w96, px.er_1, px.er_12, px.er_4 |
| Pruned as redundant | 0 |
| **Selected (6)** | **RAW_BAR_LOG_RETURN, RAW_ELAPSED_HOURS, cal.hod_cos, px.atr_ratio_14_120, px.close_loc_2, variogram_hurst_w96** |

- **Fold stability:** 0.805 / 0.925.
- **VIF:** at most 1.77.
- **Redundancy:** the maximum pairwise |Spearman| inside the 6 is 0.79; fractal versus realized
  volatility is |rho| at most 0.30; 12 multi-member clusters.
- **Comparator subsets:** the T2 and T3 MRMR K=12 subsets are in `REPORT.json`.

## 5. E0 imitability: the 6 versus the 15 on identical rows

**Rows.** Identical for both sets and every arm: TRAIN 18,612, INNER_A 2,440, INNER_B 2,952, 2024 6,236.
The 15 lost 47 TRAIN rows to the union's longest warm-up.

**Decision rule.** INNER_A chose every operating choice; INNER_B was scored frozen; 2024 was read once
after `FROZEN.json` (`freeze_sha256` `03fe9b2d...`). Freeze and reconfirm each ran once.

**Strategy.** Executable fill at b+1; TP 2 sigma / SL 2 sigma; cap 4/day; realistic costs.

**Metric definitions.** Seed means; PR-AUC base rate about 0.20-0.21, with NONE in the denominator.

| Set | Arm (chosen cell, threshold) | Period | PR-AUC valley / peak | ROC-AUC valley / peak | log-loss | Sharpe | net USD | MDD % | entries |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| 6 | m1 (dropout 0, wd 1e-3, 0.5) | INNER_B | 0.385 / 0.394 | 0.734 / 0.728 | 0.997 | -1.20 | -3,492 | 45.7 | 436 |
| 15 | m1 (dropout 0, wd 1e-3, 0.5) | INNER_B | 0.363 / 0.350 | 0.709 / 0.706 | 1.011 | -2.56 | -5,190 | 61.8 | 338 |
| 6 | m1 | 2024 | 0.379 / 0.393 | 0.725 / 0.725 | 1.000 | -1.41 | -5,133 | 62.3 | 875 |
| 15 | m1 | 2024 | 0.359 / 0.353 | 0.701 / 0.692 | 1.013 | -1.34 | -5,063 | 68.1 | 635 |
| 6 | m2 (dropout 0, wd 1e-4, 0.4) | INNER_B | 0.403 / 0.389 | 0.734 / 0.731 | 1.001 | -1.89 | -2,943 | 39.1 | 504 |
| 15 | m2 (dropout 0.1, wd 1e-3, 0.5) | INNER_B | 0.379 / 0.371 | 0.727 / 0.714 | 1.012 | -1.96 | -4,550 | 52.2 | 297 |
| 6 | m2 | 2024 | 0.388 / 0.380 | 0.722 / 0.717 | 1.007 | -3.97 | -7,406 | 79.4 | 1,037 |
| 15 | m2 | 2024 | 0.366 / 0.348 | 0.705 / 0.696 | 1.023 | -0.93 | -4,166 | 65.1 | 520 |
| 6 | m3 (l2 10, 0.6) | INNER_B | 0.392 / 0.401 | 0.732 / 0.730 | 0.982 | **+0.84** | +1,705 | 29.0 | 153 |
| 15 | m3 (l2 10, 0.6) | INNER_B | 0.370 / 0.367 | 0.726 / 0.715 | 0.981 | -0.22 | -1,213 | 32.2 | 120 |
| 6 | m3 | 2024 | 0.386 / 0.383 | 0.723 / 0.720 | 0.990 | -1.73 | -6,114 | 74.4 | 309 |
| 15 | m3 | 2024 | 0.372 / 0.347 | 0.712 / 0.692 | 0.994 | **+0.44** | +1,002 | 49.2 | 223 |
| no-trade comparator | - | every period | - | - | - | 0.00 | 0 | 0 | 0 |

Paired differences (6 minus 15), seed-mean predictions and returns, ISO-week block bootstrap with 2,000
replicates (`eval/UNCERTAINTY.json`):

| Arm | Period | delta log-loss [95 %] | Sharpe 6 / 15 (seed-mean returns) | delta Sharpe [95 %] |
|---|---|---|---|---|
| m1 | INNER_B | **-0.0113 [-0.0185, -0.0033]** | -1.51 / -3.45 | [-0.14, 4.14] |
| m1 | 2024 | **-0.0105 [-0.0154, -0.0054]** | -2.18 / -1.81 | [-2.77, 2.08] |
| m2 | INNER_B | **-0.0125 [-0.0224, -0.0029]** | -2.12 / -2.62 | [-2.78, 3.49] |
| m2 | 2024 | **-0.0163 [-0.0212, -0.0119]** | -4.76 / -1.25 | **[-6.05, -0.92]** |
| m3 | INNER_B | +0.0008 [-0.0131, 0.0140] | 0.84 / -0.22 | [-2.41, 4.51] |
| m3 | 2024 | -0.0040 [-0.0110, 0.0028] | -1.73 / 0.44 | [-5.15, 0.69] |

Reading:

1. With the v2 expert both sets classify better than in v2: PR-AUC 0.36-0.40 against base rates of
   0.20-0.21.
2. The 6 beat the 15 on calibrated log-loss for both neural arms in both periods, with intervals that
   exclude zero. For m3 the two sets are indistinguishable.
3. The strategy is not resolved:
   - Most operating points still lose money. On the trading metric, m2 in 2024 shows the 6 significantly
     worse than the 15.
   - The two positive m3 cells (the 6 in INNER_B, the 15 in 2024) do not survive the other period.
   - DSR is at most 0.057.
4. Classification usefulness of the 6 for the proxy is established. Strategy usefulness is not, and on
   one arm it is negative.

**Regularization.** The pre-declared rule replaced the primary in every shard:

- **m1:** dropout 0, weight decay 1e-3 (both sets).
- **m2:** (0, 1e-4) for the 6 and (0.1, 1e-3) for the 15.
- **m3:** l2 10 (both sets).

Per-cell tables are in `eval/FROZEN.json`. m3 seeds are still bit-identical.

## 6. Causal stage (CONDITIONAL_PREDICTIVE_ASSOCIATION; method unchanged since v2)

- **Records:** 50 (48 candidates plus the 2 history members).
- **Verdicts:** LAG0_TIME_ORDERED_LINK 22, NO_LINK 12, LAGGED_PARENT 9 (RSI, px.close_loc_2/4/6/48,
  px.logret_2/6, px.range_ratio_1_24, px.rv_1), CONFLICTING_OR_REVERSED 7.
- **Controls:** the positive control is detected; the noise and circular-shift negatives show NO_LINK.
- **Temporal halves:** 34 of 51 verdicts agree (v2 run: 36 of 45).

| Selected feature | mixed CI verdict | halves | HAC lagged parent (peak / valley) | matched-control excess pivot rate |
|---|---|---|---|---|
| RAW_BAR_LOG_RETURN | LAG0 link (o-o, p 3.0e-76) | LAG0 / LAGGED_PARENT | yes / yes | +0.038, significant |
| RAW_ELAPSED_HOURS | CONFLICTING_OR_REVERSED (lag 0, <--, p 0.02) | NO_LINK / NO_LINK | no / no | too few matched events |
| cal.hod_cos | NO_LINK | agree | no / yes | too few matched events |
| px.atr_ratio_14_120 | NO_LINK | agree | yes / no | -0.015, not significant |
| px.close_loc_2 | LAGGED_PARENT | agree | yes / yes | +0.073, significant |
| variogram_hurst_w96 | NO_LINK | NO_LINK / CONFLICTING | no / yes | +0.076, not significant |

## 7. F-table delta (v2 → v3)

| ID | v2 disposition | v3 change |
|---|---|---|
| F01 | ledger, support from the confirmation, finality replays, censoring, purges | support ends now use the v2 confirmation definition; finality re-verified under it; censored rows 579 → 0 in TRAIN |
| F02 | 2023 joint, 2024 previously observed, freeze before read | unchanged rules; freeze and reconfirm each executed once in v3 (v2 needed two freezes, identical digest) |
| F03 | adapter pinned; b+2 fill and the missing 12 h reopen rule reported; evaluator guard | **resolved**: evaluator uses the owner's executable path (b+1, reopen refusal in the adapter, cap 4/day); AC10 |
| F04 | inventory 89, profiles, perturbation, fractal keyed; repaired root not available | the admitted root was found and verified; clock and coverage measured (+4/+5 h DST signature, 2023 gap is the lake's); the join stays a missing object, now with evidence |
| F05 | T1 primary, T2/T3 comparators, recipe on T1, 15 as baseline | unchanged |
| F06 | trial ledger, monitor fix, blocks, bootstrap, DSR, declared grid | unchanged; the grid's winners changed (section 5) |
| F07 | RegressionCI with controls, labelled association | unchanged; rerun on the v3 targets |
| F08 | DAG with content identities | **exercised**: expert change → reuse of the 4 target-independent stages, recompute of the rest (`DAG.json`) |
| F09 | TIMING_PROXY_SELECTION | unchanged |
| A12 | second copy on the same disk only | **independent physical copy done** (secondary worker, sha256sum -c ALL_OK) |

## 8. What to attack first

1. **RAW_ELAPSED_HOURS is selected.** It is 1 almost everywhere and jumps after closures and missing
   hours, so its value may partly encode the lake's own coverage defects, including the 2023 gaps. Its
   causal verdict is reversed at lag 0.
2. **INNER_A still sits on the lake's 2023 coverage gap** (633 missing intra-week hours). Every
   operating choice was made there. The admitted root shows the hours exist.
3. **Calendar encodings (cal.hod_cos) are on the lake label clock,** which the witness shows is
   DST-dependent local time, so the same feature value means different UTC hours in summer and winter.
4. **The m2 2024 Sharpe is significantly worse for the 6** although its log-loss is significantly better.
   The classifier gain does not translate into this order path, and its threshold (0.4) was the most
   aggressive on INNER_A.
5. **The 2 sigma TP / SL rule** is still the detector harness's, not optimized for the v2 expert. It was
   kept to isolate the expert change; the expert's own (B) study found its confirmed version loses too.
6. **A thin H and the 34 / 51 half agreement** in the causal stage.

## 9. Not done / missing objects

- **No independent decision record;** no RL, SAC or gym-fx run; no other k or instrument; no causal
  ladder rungs 2-3.
- **The repaired-root rebuild (missing object).** It needs a pinned DST-aware clock identity for the lake
  (or a rebuild on the admitted root alone) plus a strategy adapter and calendar for the admitted source.
  Next command, once those exist:
  `python tools/timing_selection.py run --config <CFG: lake -> admitted eurusd.parquet with its availability semantics, adapter -> admitted calendar> --out <NEW_OUT>`.
- **No repin** of `upstream.lock.json` in feature_selector.
