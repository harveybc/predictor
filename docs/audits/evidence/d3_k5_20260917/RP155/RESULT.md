# The doctoral R0/R1/R2 contrast: measured, replayed and closed

Design `b5d5eee1b5fce981`. Ran 2026-09-24 on WORKER_A, finished 06:45:06Z, **11,415 CPU seconds of the declared 14,400** and
8,058 s wall. Twelve cells: three auto-encoders and nine regime fits, three paired seeds.

## The run's own gates, all satisfied

`regime_checks` returns **COMPLETE** over the exact expected population, computed only from typed evidence:

| check | result |
|---|---|
| R1's detector unchanged by its fit | true, in all three seeds |
| R2's detector changed by its fit | true, in all three seeds |
| R1 and R2 share their seed's donor | true |
| the same update allowance | true |
| the same OBSERVED updates | true, 35,910 optimizer iterations in every one of the nine fits |

Every cell selected a checkpoint on the declared monitor, restored it, and matched the restored loss to the selection.

## Replay and scoring

Each of the nine cells was replayed **in a separate process** from its saved selected weights, and all nine model digests
reconcile with the record. The reduction is over the complete outer validation population, with matched persistence on the
**same rows** in the same pass. The outer test split was not read.

| regime | population | MAE mean (SD, n=3) | MSE mean (SD) | persistence MAE, same rows | skill |
|---|---|---|---|---|---|
| R0, no pre-training | complete validation, 2,537 windows | 0.371964 (0.001081) | 0.289503 (0.001598) | 0.887945 | 0.5811 |
| R1, frozen pre-trained detector | complete validation | 0.374912 (0.001040) | 0.294932 (0.001585) | 0.887945 | 0.5778 |
| R2, pre-trained and adjustable | complete validation | 0.370234 (0.000156) | 0.287712 (0.000247) | 0.887945 | 0.5830 |
| R0 | never used for selection, 1,897 windows | 0.371383 (0.001599) | 0.285562 (0.002366) | 0.869638 | 0.5729 |
| R1 | never used for selection | 0.374612 (0.000619) | 0.290918 (0.000870) | 0.869638 | 0.5692 |
| R2 | never used for selection | 0.368980 (0.000425) | 0.282995 (0.000753) | 0.869638 | 0.5757 |

## What this does and does not establish

It establishes that the path works end to end on the matched ECL task: the regimes are provably what they claim, the budget was
identical and observed, the checkpoints are restorable, and every number here was produced by a fresh process from a saved
model over a stated population with its own matched baseline.

It does **not** decide the doctoral hypothesis. Three seeds at a development budget separate R2 from R0 by 0.0024 MAE on the
selection-free complement, with seed standard deviations between 0.0004 and 0.0016; R1 is worse than R0 by 0.0032. The
direction is that freezing the detector costs accuracy and adjusting it recovers slightly more than training from scratch, and
that is a development observation, not a result. **No H1 claim is made.**

Two scope limits travel with the table. The scored split is the outer VALIDATION split, which also supplied the 640-window
checkpoint-selection monitor, so the complete-population row is not an independent estimate; the 1,897-window complement never
used for selection is reported beside it for that reason. And the outer TEST split remains untouched for a later confirmation
that this round does not attempt.
