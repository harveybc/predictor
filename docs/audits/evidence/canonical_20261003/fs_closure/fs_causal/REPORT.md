# FS-CAUSAL interim progress report

Generated 2026-10-05T05:19:28Z from the worker_b run directory (CPU only, crispdm-run capped, seed 1729, TEST never read).
Code: causal-inference `bd483de` (`fs_causal.py`, `fs_causal_batch.py`, `fs_causal_discovery.py`).

## Denominators

- candidates: **366** = batch_001 46 + batch_002 298 + batch_003 22 (episode-source overlay columns excluded: 37; they are locators, never candidates)
- targets: 14 (Y_s_1h, Y_s_2h, Y_s_3h, Y_s_4h, Y_s_5h, Y_s_6h, Y_l_24h, Y_l_48h, Y_l_72h, Y_l_96h, Y_l_120h, Y_l_144h, Y_b_s6, Y_b_l144)
- cells: **5124** feature x target, three rungs each; chunks: 47 (block 8)
- multiplicity families: {"rung1": "per target: HAC partial-test p over all candidates", "rung2": "per target: p_linear over all cells with an identified estimate", "sypi_condition1": "per target: condition-1 p over all candidates"}

## Progress

- stage: **RUNNING**; chunks 1/47; cells 112/5124; candidates 8/366; failed features 0
- median chunk 265 s; 33.1 s per candidate; ETA 2026-10-05T08:37:11Z
- provisional raw states (before family BH): {"rung1_raw": {"ASSOCIATION_REPORTED": 112}, "rung2_raw": {"NOT_IDENTIFIED": 102, "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS": 10}, "rung3_raw": {"NOT_IDENTIFIED": 102, "COUNTERFACTUAL_UNDER_DECLARED_SCM": 10}}

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
 "progress.json": "69704658c2620fcdfe2e861d3036aeba5bf6de84c4b09b531ffd0a6577909b6b",
 "plan.json": "b5f4825c6b53ab4697f49029700b677cad661e9dc70053b1344828c3fd131a95"
}
```
