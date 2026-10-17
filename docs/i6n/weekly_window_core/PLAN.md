# Weekly Window And Core Contrast

Owner approval 2026-10-17. Independent single-heavy-process rtx4090 lane;
leave the active rtx5090 twelve-head experiment untouched. Parent recipe:
bd21fc2f weekly seven heads [2,3,4,5,6,12,18], nineteen RAW features,
Conv1D4 branches, adjacent elapsed-hour/24 plus closure context, Conv1D2
core, LSTM4 -> Dense8 -> Dense1 heads, seed6102026 and AdamW .001/.0001.
Preserve original mean TRAIN/current-week VALIDATION monitor, patience10,
min_delta1e-8, max200 and batch256. Each weekly FIT uses the same rolling
four calendar years; evaluation is the complete retained 2024 calendar.

## Finite Sequence

1. Windows: 13 control, then 7,25,49 observations every two expected open
   market hours, including t (24,12,48,96 market-hour spans respectively).
2. Rank complete annual window arms by equal-weight mean of seven pooled
   relative errors. Ties within 1e-8 prefer fewer observations. Never rank
   partial years or compare different per-head supports.
3. On the two best windows, test cores [4] and [4,2]; reuse their completed
   [2] arms without retraining. Other predeclared core arms are typed skips.

There are twelve predeclared arms (four windows, eight conditional core
arms). Eight can train: 416 actual weekly fits and eight empty-period skips;
four unselected arms become 212 explicit conditional skips. No seed replicas.
The 13-slot control is necessary on the SAME device as the treatments, not
an attempted bitwise scientific reproduction across GPU classes.

## Data And Tests

- WC01: exact timestamp source, calendar and parent preparation authentication;
  no TEST, future window observation, synthetic closed-market tick or smoothing.
- WC02: build window slots relative to each t on the expected market-hour grid;
  skip genuine absent observations, not known calendar closures. Include t.
- WC03: freeze a common origin population across four windows independently
  for FIT and VALIDATION, retaining the original per-head business masks.
  Never intersect all seven target masks. Input mean/std stay parent TRAIN-only;
  recompute one shared FIT-naive loss scale on the paired FIT population.
- WC04: each arm/head has its exact own naive, same scored rows and price units;
  cache naives before fitting, independently recompute/check and annual-pool sums.
- WC05: original trainer/fit/layers unchanged; expand only allowed slot catalogue.
  Window-only arms all have 1901 parameters and identical seeded initial weights.
  Context length is the only changed model field; core contrasts change only core.
- WC06: automatic dependency resolution, same-host resource checks, immutable
  terminals, no automatic repeat after interrupted fit, full outbox/readback,
  two verified copies and scoped timer shutdown. Operational wall7200 seconds
  allows the unchanged max200 epochs with long windows, not extra epochs.
- WC07: complete per-arm reports, best-recipe provenance, ETA by current fit and
  whole queue; no agents polling. Failed coarse arms block conditional ranking.

Tests precede implementation: source/AST identity, origin anchoring, missing
observations vs closure, shared-support projection, seven distinct naive
scales, complete-year ranking/tie, conditional skips, corruption and interruption.
Fit-free actual-data/GPU checks precede launch. Historical input source and
point-in-time qualifications remain exactly as declared by the parent.

## Trading And Demo Rule

Each new complete improvement is a trigger for paired strategy simulation
using the best currently retained compatible short/long forecasts. Retain the
exact per-horizon model/checkpoint hashes, masks, calendar, data/support and
pair identity. Save initial capital10000, net profit/return, Sharpe, maximum
drawdown, trades, costs and MAE/relative-error jointly in OLAP for meta-analysis.
Do not silently assemble incompatible supports or stale/future checkpoints.

OWNER CLARIFICATION: the destination is PAPER/DEMO only, never real money.
Once a governed simulation has positive NET profit with the configured
realistic costs, automatically serve it through prediction_provider and the
LTS pipeline's MT5-demo or Alpaca-paper backend. TEST is NOT a prerequisite
for demo and no further human approval is required. Verify account mode and
broker endpoint automatically; refuse any real-money account. Keep weekend
rules, one concurrent position per asset, configured limits and rollback.
Development evidence may authorize demo, not an independent performance claim.
TEST remains closed for research and future independent evaluation.
