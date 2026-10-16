# Autonomous Predictive Continuation

Owner directive 2026-10-16: the closed 19-fit resolution batch is retained,
but is NOT the end of the approved predictive investigation. This finite
successor owns all the following stages without another chat authorization.
No previous scientific identity is reclassified or overwritten.

## Data And Stopping Contract

Static TRAIN is 2020-2023, all four years for gradients. VALIDATION is all
supported 2024. There is NO extra eight-week holdout. At each epoch evaluate
the same weights on TRAIN and VALIDATION, stop on their mean normalized masked
MAE with patience 10, min_delta 1e-8, max_epochs 200, and restore the lowest
monitored checkpoint. Scalers and naive loss scales fit TRAIN only. The exact
per-horizon original-price naive still uses the same evaluation pairs as each
model. TEST 2025 is not loaded, scanned, trained or scored.

This makes 2024 an explicitly reused DEVELOPMENT/TUNING period, not an
independent out-of-sample estimate. Cadence models also use their evaluation
period as the requested VALIDATION checkpoint monitor: their 2024 comparison
is retrospective development, NOT a certified prospective trading simulation.
Gradient training and scaling still use only the preceding four years with
mature labels. A deployable cadence must subsequently freeze its checkpoint
selection policy before held-out evaluation; the future TEST gate cannot be
opened just because a tuned 2024 score is positive.

Keep the existing 19 RAW features, source qualifications, hourly issuance,
exact elapsed targets, calendar business masks, 12-hour operational offsets,
actual past observed market-hour endpoints including t, and no smoothing or
fabricated closed-market ticks. Freeze common support across all planned
6/9/13/18/24-tick windows and DS2/DS4. No feature selection or vendor re-audit.

Conv branches remain minimal (4 filters, k3). No additive sinusoidal code is
allowed in the new base; actual elapsed intervals and closure flags are kept
as explicit context. Heads are either the historical Dense control, a private
causal Conv1D(4,k3) followed by Dense(8)/Dense(1), or a private LSTM(4) followed
by Dense(8)/Dense(1). No Transformer, core pretraining or imported donor.

## Automatic Stages

1. Twelve DMO fits: short 2/3/4/5/6/12 h and long 18/24/36/48/60/72 h,
   DS2/DS4, 13 ticks, each of Dense/Conv/LSTM private head types. Compare on
   identical support and retain deterministic minimum equal-head annual
   relative MAE, with complexity and predeclared case order breaking ties.
2. Fifteen h72 fits at the selected long spacing: 6/9/13/18/24 ticks, linear,
   adapted DLinear and Conv1D with the selected private head type.
3. Two hundred forty representative joint configurations: short h4 and long h72, both
   spacings, five windows, and core filters [2], [4], [4,2], [8,4], ordered
   from lower to higher capacity. The window and core are compared jointly,
   not greedily fixed in separate sweeps. This is a bounded grid, NOT DOIN or
   a claim that these ranges contain the global optimum. All three private
   head types are crossed jointly, not fixed greedily from stage 1. Simple
   h72 controls remain candidates for stage 4.
4. Transfer the selected short and long recipes to the full six-head DMO
   families, two logical fits. Exact existing scientific identities are
   adopted by verified artifacts; completed identical fits are not repeated.
5. For each family compare quarterly rolling-four-year retraining with the
   selected static model on ALL 2024 evaluation pairs. Monthly follows only
   if quarterly lowers the equal-head pooled annual relative error by >1e-8;
   weekly follows only if monthly improves on quarterly by >1e-8. Otherwise
   emit explicit NOT_BENEFICIAL predecessor dispositions, not idle approval
   waits. All consecutive periods, including partial year-end support, are
   aggregated by error sums/counts per horizon; never average one week as a
   substitute for the full year. Cadence boundaries use the existing market
   weeks and actual market-open cutoff, not arbitrary Monday UTC.
6. Emit final per-head/pair eligibility, numeric result ledger, costs,
   warehouse readback, two verified physical mirrors and automatic closure.
   No incomplete or naive-failing head is promoted. Trading/TEST gates remain
   explicit; no fabricated performance or silent TEST access.

There are 269 static logical cells before cadence (12+15+240+2), potentially
fewer actual fits through exact adoption. Cadence counts come from the retained
calendar; quarterly/monthly/weekly have all actual periods, not assumed 48/52
row counts. No statistical seed repeats. Seed 6102026 everywhere.

## Execution And Acceptance

One admitted 5090 process at a time, no concurrent 5070 Ti. The coordinator
handles bounded status/outbox/mirroring; secondary host retains a mirror and
may prepare independent CPU data only if needed. Existing admission, training
kernel, serialization and outbox are reused; no consensus changes.

Persist the entire finite graph and decisions before scientific execution.
Owned timers resume and adopt terminals after restarts, never repeat an
interrupted fit silently, never wait for chat between stages, and never stop
for scientific negative results. Status must show stage denominators,
measured/unknown ETA, actual cell heartbeat, skips, failures and delivery.
All metrics/parameters/costs/dispositions go to OLAP with exact readback.
Code and small ledgers go to git; no arrays/weights/database uploads.

Before launch test: full TRAIN/no inner, validation actually used by stopping,
no TEST, future perturbation, calendar and fold boundaries, train-only scaling,
private head graphs/reload, no additive encoding, exact naive pairing,
conditional transitions, cache adoption, blocked dependencies, restart,
delivery retries, memory/GPU admission and verified automatic closure.
