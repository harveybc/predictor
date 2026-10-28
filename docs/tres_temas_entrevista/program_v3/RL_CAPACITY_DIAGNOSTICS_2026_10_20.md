# RL Generalization: Capacity And Chronological Diagnostics

Authority: owner approved continuation on2026-10-20. This is the current
RL2 generalization subtask, not a return to forecasting, DOIN or temporal fits.

## Closed Regularization Result

Four critic-only ablations completed with150/150 warehouse reports read back,
pending0,mismatch0. Selected full-period return percentages:

| Arm | TRAIN2020-23 | VALIDATION2024 | VALIDATION maxDD |
| --- | ---: | ---: | ---: |
| Reference | +361.07594 | +0.75188 | 10.63613 |
| Dropout0.05 | +85.40826 | -7.96999 | 11.33558 |
| L2 1e-6 | +124.97927 | -11.94570 | 13.71679 |
| Dropout+L2 | +126.55252 | -17.75835 | 19.86430 |
| BatchNorm alone | +0.33042 | -0.46870 | 0.79253 |

BatchNorm selected5000, stopped55000 by patience10; other treatments reached
100000. Smaller gap from suppressing TRAIN/activity is not better validation
profitability. One strength/seed does not exclude a regularizer family; enabled
early stopping does not establish that every overtraining cause is excluded.

## Completed D0

All91 original checkpoint pairs were authenticated and their selected policies
retained. New quarterly panels and paired moving-block bootstrap5/10/20
observed-day sensitivity settings (1000 repetitions,seed7) are published in
five reports,5/5 read back; a second physical checksum comparison is identical.
The CPU run used98.5MiB,12.253s. Reference quarterly VALIDATION returns:
Q1 +1.82524%,Q2 -3.18103%,Q3 +1.13833%,Q4 +1.04655%.
All five arms lost in Q2, a clue about time dependence, not proof of drift.
The reference's descriptive95% block10 interval versus no-trade includes0:
[-0.000593810,+0.000631361] mean log growth per observed NY date. Do not
interpret311 observed calendar dates (including partial sessions) as252 full
trading days or the reused development interval as independent confirmation.
Trade-level concentration and historical critic/gradient/replay telemetry
remain unavailable, explicitly, not fabricated from bar PnL.

## Automatic Capacity Sequence

Fresh SAC actor and twin-critic hidden sizes [8] -> [16] -> [16,16], compared
with the completed [32,32] reference. No regularizer; no architecture change
outside these hidden sizes. All other common scientific config fields are
verified identical: selected15 features,13 observations/stride2 observed bars,
seed7, full4-year TRAIN/full1-year VALIDATION, costs, leverage50, SL/TP,
fractional sizing, one entry/day, forced-flat12h and reopen12h restrictions.
No refit, TEST, new temporal fit or ZigZag selection integration.

Unchanged monitor: every5000 transitions, patience10, absolute1e-8, cap100000;
minimize -0.5*(mean daily log growth TRAIN + VALIDATION). Keep all trading
metrics separately by role; do not compare four-year raw profit with one-year
raw profit as an overfitting ratio. No validation-only replacement selector.

Runtime `agent-multi@dad7294c`:132 focused tests passed in the worker environment;
fresh GPU readiness and synthetic [8] update/reload passed. At05:49UTC
(00:49Bogota), the real [8] task had5000 transitions and was evaluating its
first paired VALIDATION; other arms WAIT_PREDECESSOR. No capacity results yet.
Contracts:bb2a9f3c9072,d443f9dfd52b,edf777576a9c. One heavy5090 process;
the shared-RAM5070Ti is not used as another slot. Timers own all three workers,
publication/readback, post-closure panels and second-machine model/metrics
collection. No owner approval between these declared arms, no LLM polling.

Provisional budget2-3h/arm to cap,6-9h total from prior matched runs, shorter
if patience triggers. Full paired inference, not gradient time, dominates it.

## Remaining Causes And Next Actions

Keep the19-hypothesis register in
[generalization diagnostics](RL_GENERALIZATION_DIAGNOSTICS_2026_10_19.md).
Capacity is being tested, not assumed. Business-state observability/action
mapping, input/regime shift, experience diversity, reward/cost alignment,
critic/replay/optimization behavior and seed variability remain candidates.
Use frozen-artifact checks before new telemetry or targeted interventions.
Do not choose a different VALIDATION year or promise that regularization,
smaller models or early stopping guarantee profits. Temporal SAC stays paused.

Implementation, traceability and one-read status command:
`agent-multi/docs/work_plan/49_SAC_CAPACITY_AND_GENERALIZATION.md`.
