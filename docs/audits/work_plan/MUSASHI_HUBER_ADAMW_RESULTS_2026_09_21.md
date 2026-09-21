# Huber and AdamW: measured comparison before further E1 work

## Result

All 12 DEVELOPMENT fits ran. All 12 terminals were accepted and reconciled;
live warehouse metric contents, costs and tags matched exactly (not just counts
or rounded scores). Three saved artifacts per fit are bound to the terminal.
The earlier failed dispatcher left one completed control, retained separately
and never pooled as a fourth seed. No reserved test, GPU or trading run was used.

| Training recipe | Validation MAE, kW (mean +/- seed SD) | Mean RMSE, kW | MAE reduction vs naive |
|---|---:|---:|---:|
| Persistence naive | 0.617372 | 0.966455 | 0% |
| MAE + Adam | 0.496382 +/- 0.004487 | 0.795722 | 19.598% |
| MAE + AdamW | 0.495073 +/- 0.004303 | 0.793072 | 19.810% |
| Huber + Adam | 0.520662 +/- 0.001674 | 0.781151 | 15.665% |
| Huber + AdamW | 0.522479 +/- 0.005178 | 0.778648 | 15.371% |

These are means of three seed-level metrics, not a claim that seed dispersion
measures sampling uncertainty. Relative improvements refer to the same naive
on the same validation rows, not percentage prediction accuracy.

Huber+AdamW has the lowest mean RMSE, but not the lowest MAE. Relative to
MAE+AdamW, its MAE is 0.027405 kW higher and its mean RMSE is 0.014424 kW lower.
It improves RMSE in two of three paired seeds, not every seed. AdamW by itself
does not consistently improve MAE: its paired differences change sign.

| Paired contrast, first minus second | MAE differences, seeds 1 / 2 / 3 (kW) | Mean |
|---|---|---:|
| Huber+Adam minus MAE+Adam | +0.021702 / +0.020989 / +0.030150 | +0.024280 |
| Huber+AdamW minus MAE+AdamW | +0.017394 / +0.028619 / +0.036203 | +0.027405 |
| MAE+AdamW minus MAE+Adam | +0.002447 / -0.005650 / -0.000723 | -0.001308 |
| Huber+AdamW minus Huber+Adam | -0.001861 / +0.001980 / +0.005331 | +0.001817 |

## Exactly what was compared

- Original E1 modular ARCH-A detector/adapters, sequence fusion and dilated
  temporal core; 8,127 parameters, random trainable detector, no pretraining.
- Same household electricity slice: 40,080 training origins and 10,020 common
  validation origins; seven channels, 60-sample minute window, horizon 60 minutes.
- Same train-only z-score; target mean 0.92613963 kW, SD 0.91251644 kW.
  Predictions were inverted before scoring. No log1p or new decomposition.
- Seeds 1/2/3; the four arms have identical initial-weight digests within each
  seed. Every restored model reproduces its saved predictions with max error 0.
- Learning rate .003, batch 64, beta1 .9, beta2 .999, epsilon 1e-7.
  Huber delta=1 in standardized units; AdamW weight decay=.004. These are
  declared initial settings, not a reconstruction of the owner's old optimum.
- All arms select on validation MAE, patience 3 epochs, restore best weights,
  ceiling 4,000 observed optimizer updates. No post-result tuning.
- Seven fits reached that ceiling. Their best epochs are retained in the table
  artifact. Potential performance at a larger budget is unknown; no convergence
  or global recipe ranking is claimed. The finite-budget comparison is measured.
- All four arms ran on one CPU host under the same environment, four concurrent
  children maximum, two intra-op threads each. This was not a three-host run.

The MAE control was rerun, rather than pooling the earlier phase-1 scores under
a different execution context. The initial dispatcher control and its v2 repeat
are exactly equal at 0.4970333449520751 kW.

## Audit and traceability

Frozen v2 design: `be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965`.
Execution revision: `73f3bab`. Campaign prefix: `musashi-huber-adamw-20260921-v2`.
Private run root: `~/.local/state/crispdm-data-foundation/huber_adamw_v2`.
Full public table: [REPORT.json](../evidence/HUBER_ADAMW_2026_09_21/REPORT.json).
Protocol: [HUBER_ADAMW_PROTOCOL_2026_09_21.md](HUBER_ADAMW_PROTOCOL_2026_09_21.md).

The verifier re-reads the prediction arrays, matches their origins/targets/naive
against source DATA, recomputes the scores, checks paired initialization, then
queries the live warehouse by campaign and compares metrics and artifacts.
A supplementary exact check verified all metric fields without rounding, costs,
tags, weight-file hashes, and fresh accounting reconciliation: all 12 passed,
no missing/accounting-only/lake-only units. The cube contains 60 metric rows and
36 artifact bindings for these 12 cells. Historical rows were not rewritten.

[verify_inputs.py](../evidence/HUBER_ADAMW_2026_09_21/verify_inputs.py) independently
re-materialized the inputs from the governed delivery: scaler mean/SD, raw
labels, standardized inputs and train/evaluation origins all match exactly.
This closes the cache-to-source link; it does not claim a new independent
causality audit of all inherited preprocessing.

Focused verification: 35 loss/optimizer/baseline tests plus 21 work-plan tests,
all passing. Documentary validator: PASS, scientific approval false as expected.
The entire legacy repository suite was not rerun for this scoped addition.

Measured v2 service cost: 4,279.179 CPU seconds, 17m07s wall, approximately
2.1 GiB peak and zero swap. Child-record CPU sum: 4,257.172 seconds; service cost
also includes orchestration/import overhead. The first dispatch consumed another
402.085 CPU seconds. Total measured service CPU: 4,681.264 s, below 14,400 s.

Implementation fault, not hidden: the first dispatch raced the legacy dynamic
module loader. One child completed; another could not register. Imports now
finish before worker threads start, with a regression test. The original root
and terminal remain intact. The scientific recipe was not changed by the fix.

## Disposition and corrections to RP57-RP64 interpretation

1. Preserve Huber as a useful alternative, particularly for the RMSE tradeoff;
   do not assert that it beats MAE on this task, or that it cannot work better
   with another delta, learning rate, target transform, architecture or domain.
2. AdamW's small mean MAE advantage here is not consistent across seeds. Keep
   MAE+Adam as the continuity reference for the next isolated information
   diagnostic; record MAE+AdamW as a measured candidate, not a universal winner.
3. RP63 changed both loss and early-stopping monitor. Its 0.049 kW improvement
   is a recipe contrast, not proof that loss alone caused the entire difference.
4. A single shuffled-label run is a negative control, not a quantitative bound
   on all learnable information or all attainable improvements.
5. The old financial phase-3 result remains OWNER_REPORTED/UNBOUND. This public
   household experiment neither reproduces nor invalidates that historical run.
6. Input information and then volume remain next, with explicit receptive-field,
   sample-size, stopping and baseline controls. No return to R0/R1/R2 yet.

## Owner disposition, 2026-09-21 (additive)

The continuity reference above applies only to the household diagnostic.
No default loss is selected for trading. The owner requires the independent
[FIN-LOSS-OPT comparison](../../tres_temas_entrevista/program_v3/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md)
with fine tuning and explicit measurement of marginal normalized improvements.
The interim suggestion to generalize MAE+AdamW is superseded. These measured
results and their limitations remain unchanged. No historical OLAP search or
heuristic-strategy rerun is a prerequisite of that comparison.
