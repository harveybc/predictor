# Exact Weekly Twelve-Horizon Contrast

Owner order 2026-10-17: append the remaining long horizons to the weekly
seven-head procedure, changing nothing else scientifically. Parent runtime
bd21fc2fc22a6e8607df14d68412223dc97a64d5; parent inputs and folds descend from
e0626cd9. Preserve [2,3,4,5,6,12,18] in that order and append [24,36,48,60,72].
The currently running seven-head campaign must finish without interference.

## Invariants And Test Design

- WA01: original trainer, continuation fit, data, business masks and layer
  builders unchanged. Broaden only the allowed model-horizon catalogue.
- WA02: keep all 53 calendar periods, the same empty first period, rolling
  four calendar years, 19 RAW features, DS2 and 13 slots including t. Preserve
  every original seven-head row, label, mask, input scaler and naive exactly.
- WA03: Conv1D4 per feature, Conv1D2 core, LSTM4 -> Dense8 -> Dense1 per head;
  same seed 6102026, batch256 and AdamW .001/.0001. Preserve initial shared
  and seven-head weights/forward. Each new head necessarily adds parameters
  and a loss term; gradients, stopping epoch and predictions may change.
- WA04: unchanged mean TRAIN/current-week VALIDATION patience metric, patience
  10, absolute min_delta 1e-8, max200 and lowest monitored first-tie checkpoint.
  Never substitute ANY_HEAD or optimize architecture/window in this contrast.
- WA05: each new exact elapsed t+h target obeys the original C1 MANAGEMENT
  calendar, strict force-flat and split maturity. Do not intersect all heads,
  fill weekend bars or compress target horizons into market time.
- WA06: own naive per head on exact scored rows, original price scale, full
  year pooled sums/counts. Unsupported period/head is explicit, not zero skill.
- WA07: same admitted single-heavy-job GPU class, caps, source pins, immutable
  terminals, OLAP readback and two physical mirrors; retain failures, no refits.
- WA08: an owned CPU successor waits for successful published/mirrored parent
  closure, then performs frozen preparation, GPU preflight and installs the
  new campaign's finite timers. No agent polling or human launch approval.

Unit design: source/AST identity, horizon-only config delta, seven-column
projection/mutation, exact targets/maturity/force-flat for all five horizons,
individual naives, annual pooling, successor wait/refusal/idempotency and
unchanged original monitor. Integration: actual completed parent period,
GPU fit-free old/new initialization and all real periods before release.

The successor must leave a typed status if a prerequisite or resource fails.
It may retry operational preparation/delivery, not a started scientific fit.
The twelve-head worker's operational wall ceiling is 3600 seconds instead of
1800, so the extra heads do not prematurely truncate the unchanged max200
epoch protocol. The parent worker and every scientific parameter stay intact.
Status: `python -m tools.i6dmo.weekly_all status --spec <private-spec-json>`.

## Interpretation

The promising seven-head numbers are partial DEVELOPMENT 2024, not final
annual or TEST evidence. This extension is not conditional on maintaining a
favorable partial score. Like the parent, weekly evaluation data select the
checkpoint; no independent OOS or trading-profit claim. TEST stays closed.
