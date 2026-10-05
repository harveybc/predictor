# FS-CAUSAL closure report

Generated 2026-10-05T08:32:02Z from the worker_b run directory (CPU only, crispdm-run capped, seed 1729, TEST never read).
Code: causal-inference `bd483de` (`fs_causal.py`, `fs_causal_batch.py`, `fs_causal_discovery.py`).

## Denominators

- candidates: **366** = batch_001 46 + batch_002 298 + batch_003 22 (episode-source overlay columns excluded: 37; they are locators, never candidates)
- targets: 14 (Y_s_1h, Y_s_2h, Y_s_3h, Y_s_4h, Y_s_5h, Y_s_6h, Y_l_24h, Y_l_48h, Y_l_72h, Y_l_96h, Y_l_120h, Y_l_144h, Y_b_s6, Y_b_l144)
- cells: **5124** feature x target, three rungs each; chunks: 47 (block 8)
- multiplicity families: {"rung1": "per target: HAC partial-test p over all candidates", "rung2": "per target: p_linear over all cells with an identified estimate", "sypi_condition1": "per target: condition-1 p over all candidates"}

## Progress

- stage: **FINALIZED**; chunks 47/47; cells 5124/5124; candidates 366/366; failed features 0
- median chunk 267 s; 32.1 s per candidate; ETA 2026-10-05T08:32:02Z
- provisional raw states (before family BH): {"rung1_raw": {"ASSOCIATION_REPORTED": 5124}, "rung2_raw": {"NOT_IDENTIFIED": 4576, "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS": 128, "NOT_EVALUATED": 420}, "rung3_raw": {"NOT_IDENTIFIED": 4576, "COUNTERFACTUAL_UNDER_DECLARED_SCM": 128, "NOT_EVALUATED": 420}}

## Discovery comparator (PCMCI+, screening only)

- pcmci_plus: candidates {'done': 0, 'total': 366}, verdicts {}, ETA None, pin {"library": "tigramite", "version": "5.2.10.1", "license": "GNU General Public License v3.0", "method": "PCMCI+ (run_pcmciplus) with ParCorr", "reference": "Runge (2020) UAI; Runge et al. (2019) Sci. Adv."}

## Final states per rung (all targets)

| rung | SUPPORTED | CONTRADICTED | NOT_IDENTIFIED |
|---|---:|---:|---:|
| rung1 | 34 | 0 | 5090 |
| rung2 | 0 | 0 | 5124 |
| rung3 | 0 | 0 | 5124 |

## States per rung and target

| target | rung1 S/C/N | rung2 S/C/N | rung3 S/C/N |
|---|---|---|---|
| Y_s_1h | 3/0/363 | 0/0/366 | 0/0/366 |
| Y_s_2h | 3/0/363 | 0/0/366 | 0/0/366 |
| Y_s_3h | 3/0/363 | 0/0/366 | 0/0/366 |
| Y_s_4h | 4/0/362 | 0/0/366 | 0/0/366 |
| Y_s_5h | 4/0/362 | 0/0/366 | 0/0/366 |
| Y_s_6h | 5/0/361 | 0/0/366 | 0/0/366 |
| Y_l_24h | 6/0/360 | 0/0/366 | 0/0/366 |
| Y_l_48h | 0/0/366 | 0/0/366 | 0/0/366 |
| Y_l_72h | 1/0/365 | 0/0/366 | 0/0/366 |
| Y_l_96h | 0/0/366 | 0/0/366 | 0/0/366 |
| Y_l_120h | 1/0/365 | 0/0/366 | 0/0/366 |
| Y_l_144h | 1/0/365 | 0/0/366 | 0/0/366 |
| Y_b_s6 | 1/0/365 | 0/0/366 | 0/0/366 |
| Y_b_l144 | 2/0/364 | 0/0/366 | 0/0/366 |

## SUPPORTED / CONTRADICTED cells

Rung 2 (identified historical intervention; estimand: ATE of a first available TRAIN-q80 crossing vs staying below, both from the pre-row band [q60,q80), AIPW with declared W): 0 cells

_none_

Rung 1 (association only, HAC partial test + OOF gain + BH per target): 34 cells; first 60 by |t|:

| feature_id | target | rung1_state | rung1_robust | rung1_reason | r1_coef | r1_t_hac | r1_q | r1_oof_gain | r1_oof_positive_folds | r1_best_extra_lag | r1_mss | sypi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| px.close_loc | Y_s_1h | SUPPORTED | True |  | -0.0001011 | -8.153 | 1.625e-13 | 0.0008239 | 4 | 1 |  | SYPI_CONDITION2_FAILED |
| px.close_loc | Y_s_2h | SUPPORTED | True |  | -0.0001271 | -7.362 | 6.648e-11 | 0.000744 | 4 | 1 |  | SYPI_CONDITION2_FAILED |
| px.close_loc | Y_s_3h | SUPPORTED | True |  | -0.0001414 | -6.853 | 2.653e-09 | 0.000612 | 3 | 1 |  | SYPI_CONDITION2_FAILED |
| yh.usdtwd_x.logret_1d | Y_b_l144 | SUPPORTED | False |  | -0.3412 | -6.575 | 1.778e-08 | 3.44e-05 | 3 | 24 |  | SYPI_CONDITION2_FAILED |
| px.close_loc | Y_s_4h | SUPPORTED | False |  | -0.0001491 | -6.335 | 8.689e-08 | 0.0004392 | 3 | 1 |  | SYPI_CONDITION2_FAILED |
| yh.usdtwd_x.logret_1d | Y_l_144h | SUPPORTED | True |  | -0.006566 | -6.191 | 1.094e-07 | 3.726e-05 | 3 | 3 |  | SYPI_CONDITION2_FAILED |
| px.close_loc | Y_s_5h | SUPPORTED | False |  | -0.0001626 | -6.12 | 3.43e-07 | 0.0003504 | 3 | 1 |  | SYPI_CONDITION2_FAILED |
| yh.usdtwd_x.logret_1d | Y_l_120h | SUPPORTED | False |  | -0.005513 | -6.01 | 6.784e-07 | 1.31e-05 | 3 | 3 |  | SYPI_CONDITION2_FAILED |
| yh.usdclp_x.logret_1d | Y_b_l144 | SUPPORTED | False |  | 0.06713 | 5.933 | 5.445e-07 | 1.548e-05 | 4 | 2 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| px.close_loc | Y_s_6h | SUPPORTED | False |  | -0.0001567 | -5.494 | 1.434e-05 | 0.0001178 | 3 | 1 |  | SYPI_CONDITION2_FAILED |
| cal.hour_sin | Y_s_6h | SUPPORTED | True |  | -0.0001246 | -4.421 | 0.001728 | 0.0005633 | 3 | 3 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| cal.hour_sin | Y_s_5h | SUPPORTED | True |  | -0.0001039 | -4.339 | 0.002167 | 0.0003409 | 3 | 6 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| cal.hour_sin | Y_s_4h | SUPPORTED | True |  | -8.09e-05 | -4.157 | 0.002944 | 0.0001449 | 3 | 6 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| cal.hour_sin | Y_b_s6 | SUPPORTED | True |  | -0.02738 | -4.04 | 0.01959 | 0.0004251 | 3 | 1 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.usdtwd_x.logret_1d | Y_l_72h | SUPPORTED | False |  | -0.002892 | -3.981 | 0.0251 | 1.116e-05 | 3 | 24 |  | SYPI_CONDITION2_FAILED |
| yh.si_f.logret_1d | Y_l_24h | SUPPORTED | True |  | 0.01447 | 3.9 | 0.01173 | 0.0041 | 3 | 24 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlc.logret_1d | Y_s_1h | SUPPORTED | False |  | 0.001716 | 3.806 | 0.008623 | 8.832e-05 | 3 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlc.logret_1d | Y_s_2h | SUPPORTED | False |  | 0.003309 | 3.783 | 0.009463 | 0.0002466 | 3 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.iyr.logret_1d | Y_l_24h | SUPPORTED | True |  | 0.02162 | 3.773 | 0.01381 | 0.003805 | 3 | 24 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_l_24h | SUPPORTED | True |  | 0.01508 | 3.734 | 0.01381 | 0.004103 | 3 | 24 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlc.logret_1d | Y_s_3h | SUPPORTED | False |  | 0.004787 | 3.685 | 0.01196 | 0.0003018 | 3 | 6 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.vnq.logret_1d | Y_l_24h | SUPPORTED | True |  | 0.02061 | 3.635 | 0.01406 | 0.003755 | 3 | 24 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlre.logret_1d | Y_l_24h | SUPPORTED | True |  | 0.02289 | 3.609 | 0.01406 | 0.004031 | 3 | 1 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlc.logret_1d | Y_s_4h | SUPPORTED | False |  | 0.006152 | 3.585 | 0.01652 | 0.0003873 | 3 | 6 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlc.logret_1d | Y_s_5h | SUPPORTED | False |  | 0.007435 | 3.498 | 0.02038 | 0.0004265 | 3 | 1 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.xlc.logret_1d | Y_s_6h | SUPPORTED | False |  | 0.008734 | 3.467 | 0.02137 | 0.0005548 | 3 | 3 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_s_3h | SUPPORTED | True |  | 0.002646 | 3.306 | 0.03146 | 0.001089 | 5 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_s_1h | SUPPORTED | True |  | 0.0009093 | 3.3 | 0.03936 | 0.0003813 | 5 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_s_2h | SUPPORTED | True |  | 0.001784 | 3.298 | 0.03724 | 0.0007226 | 5 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_s_5h | SUPPORTED | True |  | 0.004264 | 3.294 | 0.03289 | 0.001682 | 5 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_s_6h | SUPPORTED | True |  | 0.005039 | 3.286 | 0.03387 | 0.001949 | 5 | 24 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.slv.logret_1d | Y_s_4h | SUPPORTED | True |  | 0.003448 | 3.277 | 0.03195 | 0.001372 | 5 | 12 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |
| yh.gc_f.logret_1d | Y_l_24h | SUPPORTED | True |  | 0.02099 | 3.218 | 0.04727 | 0.003849 | 4 | 24 |  | SYPI_CONDITION1_FAILED |
| rg.roc_12 | Y_s_6h | SUPPORTED | True |  | -0.0002907 | -3.191 | 0.03702 | 0.0003946 | 3 | 24 |  | SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS |

## Abstention reasons

```json
{
 "rung1": {
  "NO_CONDITIONAL_DEPENDENCE_AT_FDR_0.05": 5026,
  "NO_OOF_GAIN_MAJORITY": 4125,
  "SIGN_NOT_STABLE_ACROSS_FOLDS": 2315
 },
 "rung2": {
  "OVERLAP_SCREEN_FAILED": 2976,
  "EFFECT_NOT_SIGNIFICANT_AT_FAMILY_FDR_0.05": 128,
  "INTERVAL_INCLUDES_ZERO_AND_NOT_PRECISE_NULL": 117,
  "NONLINEAR_CONFIRMATION_OVERLAP_SCREEN_FAILED": 128,
  "IMBALANCE": 30,
  "NO_COMMON_SUPPORT": 1570,
  "THRESHOLD_BAND_DEGENERATE": 238,
  "PLACEBO_FAILED": 5,
  "KNOWN_IN_ADVANCE_CALENDAR_NOT_AN_OBSERVED_INTERVENTION": 140,
  "ASSUMED_PUBLICATION_CLOCK": 3584,
  "EMPTY_TREATMENT_STRATUM": 244,
  "NO_EPISODES": 42
 },
 "rung3": {
  "RUNG2_NOT_IDENTIFIED": 4704,
  "THRESHOLD_BAND_DEGENERATE": 238,
  "KNOWN_IN_ADVANCE_CALENDAR_NOT_AN_OBSERVED_INTERVENTION": 140,
  "NO_EPISODES": 42
 }
}
```

- robust CONTRADICTED features (the only ones that may weigh against selection): []
- features SUPPORTED at rung 2: []
- features SUPPORTED at rung 3: []
- features SUPPORTED at any rung: 12
- SyPI screen verdicts: {"SYPI_CONDITION1_FAILED": 4948, "SYPI_CONDITION2_FAILED": 19, "SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS": 157}; PCMCI+: {"PENDING": 366}
- clock distribution of candidates: {"OBSERVED": 81, "KNOWN_IN_ADVANCE": 10, "ASSUMED": 275}

## Rules honoured

- TRAIN only: thresholds, bands, regimes, folds, propensities, outcome models and SCMs fitted inside the lane-A TRAIN rows; external validation and sealed test never read.
- Calendar columns are episode locators / W only; nothing feeds a predictor (I11 deferred).
- Endogenous indicators: treatment = predeclared historical transition (first available crossing of the TRAIN q80 threshold from the pre-row band), never do(indicator=value); upstream mechanism = cross-fitted propensity on W recorded per episode set.
- Repaired fail-closed gate (causal-inference 48ae17c) unchanged: declared DAG + back-door check, support, overlap without trimming, balance <= 0.1, placebo battery, four assumptions declared with evidence references (CAUSAL_SUFFICIENCY is DECLARED_WITH_SENSITIVITY_ONLY).
- NOT_IDENTIFIED never eliminates; only robust CONTRADICTED weighs against a feature; the 1,076 historical rung-2 estimates were not reused.
- One seed (1729); compute on worker_b CPU under crispdm-run with the cap bound to the measured pilot peak.

## Digests

```json
{
 "progress.json": "e7307786c8b98e2ff0c46535c26b55bb62c045b226dc92a0b7fac9517e4059e6",
 "plan.json": "b5f4825c6b53ab4697f49029700b677cad661e9dc70053b1344828c3fd131a95",
 "final_summary.json": "ece610f614d95cfa6ea3a3c43b64ccb3515fbddcb04e6381d18c1d0ddc607f6f",
 "cells_summary.csv": "a1b0ff6225c8e4944a13ae46b44aac491a31de88ecee0af31d928a80df433fcc",
 "feature_summary.csv": "e2b0421e4e763ca806a795701c92406d49c039ea6bb34752a5b5e8dc162c5871",
 "supported_contradicted_cells.csv": "e93c272598141f2ec1f07b12c14484dc5575fb0be56edf168e19d4b49c56cd0a",
 "digests.json": "f6c157cb1d6128aa45a2c349c5af8010d5104519e2afe4b2c6e5b697f4d62cd9",
 "READY": "bcf93c8217c16846d3de94e7952354880e65f320eb3a7f207ae1a6b148414484",
 "progress_discovery.json": "e7a21bc7a0c234e223f6eff6edecacc14dcb1d74c6b53130380edaa81ed0804a"
}
```
