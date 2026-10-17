# Exact Weekly Short Plus 18 Hours

## Scope And Frozen Parent

Owner order 2026-10-17: repeat the complete weekly short procedure that won
at six horizons, adding only the 18-hour head. No sweep, static substitute,
new feature selection, donor, DOIN change or TEST access is authorized.
Parent runtime: e0626cd9b2ac491a4508487883d04f40692d798c. Parent campaign:
i6dmo-continuation-41c3a7698150. Preserve every declared weekly period,
including explicit empty-period dispositions, and the exact rolling four
calendar years of FIT ending at that week's original market-open cutoff.

The selected parent is Conv1D4(k3) per feature, weightless fusion with adjacent
elapsed-hour/24 and closure context, Conv1D2(k3) core, and one LSTM4 -> Dense8
-> Dense1 head per horizon. Nineteen RAW inputs, 13 observations spaced two
open-market hours including t; no target smoothing. Append h18 after
[2,3,4,5,6,12]. Shared/old-head initialization must be byte-identical.

Training is the original AdamW, learning_rate .001, weight_decay .0001,
batch 256, seed 6102026. Preserve the original MeanMonitorStopper exactly:
0.5*(TRAIN loss + the current week's DEVELOPMENT loss), equal-weight means
over supported heads of per-head FIT-naive-normalized MAE, patience 10,
absolute min_delta 1e-8, max_epochs 200, restore lowest monitored first tie.
Adding a head necessarily changes the joint loss and can change gradients,
checkpoint and stop epoch. Do not promise identical short predictions.

## Data And Evidence Gates

- W18-01: source trainer, data, layers and continuation function stay byte
  identical to parent. Only the model's allowed-horizon catalogue expands.
- W18-02: all original FIT/VALIDATION arrays, row order, old labels/masks,
  scales, periods and six naives remain exact. No all-head intersection.
- W18-03: h18 is exact elapsed t+18 with the original C1 MANAGEMENT calendar,
  force-flat and split-maturity rules. No interpolation or weekend padding.
- W18-04: all seven heads use their own naive on exactly their scored rows,
  original price scale. Annual MAE uses sums/counts across the whole year.
- W18-05: source and output digests, effective model/training/scales, original
  period receipt, source-file identity, seed, weights and predictions persist.
- W18-06: one heavy fit at a time on the original GPU class, a fresh resource
  check, 4352 MiB admission (at least 1.25 times the parent's measured maximum
  of 3637268480 bytes) and 6 GiB outer cap, no concurrent shared-RAM GPU.
- W18-07: immutable terminals, no automatic refit after interrupted STARTED,
  bounded CPU sync/publication/readback, verified local and second-host copies.
- W18-08: status by campaign/period includes epoch, stale counter and measured
  remaining-time estimate. Campaign closes its own timers automatically.

## Test Design And Execution

Before implementation: unit tests for projection equality, h18 availability,
maturity, separate naive, model-only-horizon delta, parent source identity,
per-period preservation, pooling/duplicate rejection and terminal corruption.
Integration: all real periods are prepared with six-column projection checks;
GPU fit-free initialization compares shared and old-head weights and forwards.
Launch only after these pass. No independent scientific replicate is needed.

## Interpretation

Exactly like the parent, weekly DEVELOPMENT 2024 is used for checkpoint
selection. This is retrospective development, not an independent holdout,
TEST result or proof of trading profit. Preserve the original result table
next to this contrast, without attributing any difference solely to cadence.

Status: `python -m tools.i6dmo.weekly18 status --spec <private-run-json>`.
Top-level automation: bootstrap -> prepare -> preflight -> install -> worker
and tick timers -> annual report -> OLAP readback -> two mirrors -> close.
