# Re-scored on the corrected target support, both reductions, closure gated

No retraining. The nine selected checkpoints of `ecl-modular-contrast-20260924b` were replayed again, each in a fresh process.

## The three corrections

**The disjoint subset is computed, not assumed.** A window at origin `o` predicts rows `[o + 96, o + 192)`. The monitor's 640
windows therefore reach target row 830, so the first origin that shares no target row with it is **735**, leaving **1,802**
windows and **95** that overlapped. This reproduces the review's count exactly and is now derived by
`label_disjoint_origins`, which states the rule: sharing an input context is not sharing a label. The earlier
"never used for selection" label is withdrawn at target-support scope; the earlier numbers stand as descriptive.

**The closure refuses incomplete evidence.** The expected population comes from the REGISTERED design rather than from whatever
survives in the run directory, a checkpoint whose digest does not match its record is refused **before** it is loaded or
scored, and a run that is missing a cell or carries an identity failure produces `INCOMPLETE_EVIDENCE` with per-cell
diagnostics and **no** regime summary. This run: 9 expected, 9 scored, no identity failures, no problems, status COMPLETE.

**Both reductions are named and separate.** The author's float32 metric now comes from the existing bounded exact reducer, with
an independent float64 diagnostic beside it. On this population they differ by between 1e-9 and 4e-8 MAE, so the precision
question changes nothing here and does not explain any regime difference.

## The corrected table, 1,802 label-disjoint windows

| regime | author float32 MAE, mean (SD, n=3) | float64 MAE | matched persistence, same rows | skill |
|---|---|---|---|---|
| R0, no pre-training | 0.371174 (0.001534) | 0.371174 | 0.868283 | 0.5725 |
| R1, frozen pre-trained detector | 0.374584 (0.000733) | 0.374583 | 0.868283 | 0.5686 |
| R2, pre-trained and adjustable | 0.368596 (0.000649) | 0.368596 | 0.868283 | 0.5755 |

Paired against R0, per seed:

| seed | R1 − R0 | R2 − R0 |
|---|---|---|
| 2021 | +0.004801 | −0.001712 |
| 2022 | +0.002874 | −0.001702 |
| 2023 | +0.002554 | −0.004319 |
| mean | **+0.003410 (+0.919 %)** | **−0.002577 (−0.694 %)** |

**R1 is worse than R0 in three of three seeds and R2 is better than R0 in three of three.** On the review's contaminated
subset the R2 advantage was −0.002403 MAE and −0.647 %; on the corrected one it is −0.002577 and −0.694 %, so the descriptive
signal survives the correction and is marginally larger.

## What this is not

Three seeds at a development budget on the outer VALIDATION split. Disjoint labels are not statistical independence: these
windows are adjacent in time to the monitor's. The outer TEST split was not read and no hypothesis is decided. **No H1 claim.**
