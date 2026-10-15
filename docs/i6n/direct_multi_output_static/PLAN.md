# I6-DMO: static direct multi-output business experiment

Owner amendment: 2026-10-14; next-design decision: 2026-10-15.
Status: exact-DS2 short CLOSED, six heads NOT_ELIGIBLE; market-time successor
APPROVED_FOR_IMPLEMENTATION, not yet implemented, sealed or running.
The governing successor is [MARKET_TIME_PAIR_PLAN.md](MARKET_TIME_PAIR_PLAN.md).
Sections below retain the measured exact-DS2 recipe and shared business rules;
their elapsed-time input grids do not authorize another DS2 fit or an empty
DS12 fit. The successor changes input sampling and its time encoding only,
under new identities. This dispatch supersedes automatic H4 -> H72 searches,
per-horizon sweeps and mandatory weekly fits for this experiment. Historical
chains, fits, metrics and conclusions remain immutable.

## 1. Priorities and sequence

1. Point-in-time inputs and business-masked targets, including the current tick.
2. A complete short/long prediction pair and its historical trading utility.
3. The incremental value and cost of more frequent retraining.

The finite sequence is:

`SHORT_DS_STATIC_VALIDATION -> LONG_DS_STATIC_VALIDATION -> PAIR_VALIDATION
-> FINAL_PROCEDURE_FROZEN -> TEST_READINESS -> FINAL_PAIR_TEST -> CLOSED`.

A typed failure or lack of predictive eligibility is a terminal scientific
disposition, not permission to search TEST or launch unbounded alternatives.
The orchestrator advances eligible stages without an agent or a new approval.
First implement and run the short stage; do not run twelve separate searches.
Hourly controls and NONE/QUARTERLY/MONTHLY/WEEKLY comparisons are later
validation-only experiments, not prerequisites for this first static baseline.
They cannot reuse TEST as a new independent holdout after its results are seen.

## 2. Fixed first experiment

For the closed exact-DS2 measurement, `EXPERIMENT.short.json` is its authoritative design, not an immediately
runnable legacy predictor config. A typed adapter must produce an effective
config, reject unknown fields and preserve every design field in its identity.

- Asset EURUSD; selected RAW feature values, target never denoised.
- Short physical horizons: `[2, 3, 4, 5, 6, 12]` hours, six independent heads.
- Input span 24 elapsed hours inclusive: `t-24, t-22, ..., t-2, t`.
- Exactly 13 observations, spaced two hours, issued at every hourly origin.
- No smoothing, interpolation, row-count horizons or coarse-output relabelling.
- Horizons 3 and 5 remain exact hourly endpoints; input spacing does not
  constrain direct-output horizons to its multiples.
- One fit, not one fit per head/week/quarter, for the full validation year.
- Seed 6102026; one repetition. Adopt authenticated completed units after restart.
- TRAIN `[2020-01-01, 2024-01-01)` UTC, VALIDATION
  `[2024-01-01, 2025-01-01)` UTC; TEST `[2025-01-01, 2026-01-01)` stays closed.
  Preflight must certify complete coverage, not silently shorten these periods.
- The chronological last eight weeks of TRAIN are inner validation for
  checkpoint selection. Purge fit labels whose availability reaches the inner
  boundary. Both fit and inner belong to the declared four-year corpus.
- Training checkpoint selection uses only fit/inner, never outer validation.

The 19-feature identity is the active availability-gated H4 manifest, excluding
STLFSI4. The set was developed using 2024: report outer VALIDATION as development,
not an untouched selection holdout. Recover the seven unresolved producers and
availability clocks before confirmatory TEST. Missing evidence is not zero skill.
STLFSI4 cannot return through a renamed config or cached artifact.

## 3. Minimal modular model

Each feature is its own branch, `(batch, 13, 1)`:

1. `Conv1D(filters=4, kernel_size=3, strides=1, padding='causal', relu)`.
2. Concatenate branch outputs on the filter axis only.
3. Explicit positional encoding immediately after fusion, with physical
   offsets available to the encoder; keep the current-origin position.
4. Core: causal Conv1D 8 filters/kernel 3/stride 1, then causal Conv1D
   4 filters/kernel 3/stride 1. No Transformer or LSTM in this minimal identity.
5. Each head reads the full core sequence via Flatten, Dense(8, relu),
   Dense(1, linear). Only the heads collapse the temporal axis.

Reuse hierarchical plugins/facade and serialization. Do not replace the
representation with a whole-input MLP or modify historical plugin defaults.
First identity is R0, explicitly random initialization, to keep the first
baseline independent of donor recovery. R1/R2 remain supported; subsequent R2
loads only donors certified for these features, scalers, physical support,
spacing, shapes and code. Do not silently load hourly/lagged donors into this
two-hour representation. Core pretraining, NEAT and uncertainty remain later.

Heads have independent weights, not six separate full backbones. All six
targets are trained jointly. Invalid labels must be removed safely before loss
calculation, not multiplied as NaN times zero. Retain a row with any valid head.
Each head's loss is its masked mean MAE divided by its fit-only naive MAE;
average the supported heads equally. A zero scale or unsupported head is a
typed disposition, not a hidden epsilon-based success. Report inverse-scaled
original-price MAE and naive regardless of the training coordinate.

Initial training: AdamW, learning rate 0.001, weight decay 0.0001, batch 256,
maximum 200 epochs, patience 10, minimum monitored improvement 1e-8. Early
stopping monitors `0.5 * (fit_loss + inner_loss)` evaluated at the same
end-of-epoch weights, restores the best weights and records both losses.
The ceiling is not a demand to execute every epoch. Do not tune these values
from TEST. Measure wall time, CPU, GPU use and full cgroup memory peak.

No DEAP search in this first fixed-model measurement. A later search must
jointly optimize input span/spacing, branch/core capacity and head capacity;
keep established DOIN local population >=20, generations >=100, patience >=10,
exact relative improvement 1e-8, shared chain and canonical work-proof rules.
Never restart the retired single-horizon campaign as an implicit fallback.

## 4. Temporal and business contract

Reuse the audited C1/C2 components and the effective strategy interface.
Separate forecasting support S_h from actual strategy-use support U_h.

- Origin is the last completed observed hourly bar at t, including its value.
- Every input must have available_time <= t; changing data after t cannot
  change that input or prediction. Retain known past observations outside the
  order-opening interval; the mask applies to decisions/targets, not deletion
  of all weekend-adjacent input history.
- Strategy opening begins 12 h after market reopening. All positions must
  be filled closed before the force-flat boundary 12 h before market close.
- Retain the versioned Thursday entry cutoff and instrument calendar from the
  business contract; distinguish entry and management masks.
- For each head require an exact observed target at t+h, within the permitted
  operational interval, strictly before force-flat and within its split.
  Never bridge weekend gaps using the next observed row. Do not cross into
  2025 when scoring 2024. Missing inputs/targets get counted reasons.
- Use a versioned point-in-time calendar and source/bar-label convention.
  An assumed Forex calendar is labelled as such, never a verified broker one.

Normalization fits only fit rows. The naive for horizon h is p(t) compared
with p(t+h), with MAE `sum(abs(p(t+h)-p(t)))/n_h`, on exactly the same rows as
the model. Relative error is `(MAE_h-naive_h)/naive_h`; negative is better.
Compute annual pooled sums/counts and a separate unweighted weekly diagnostic;
do not average unequal weekly MAEs and call it pooled annual MAE.

## 5. Long model, strategy and final TEST

The existing six-long contract is `[18, 24, 36, 48, 60, 72]` physical hours,
as recorded in `docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json`. Do not revert to
unsupported 96/120/144 h endpoints. Prepare a separate long design after short
closure: initial 144-hour inclusive span, input spacing 12 h, 13 observations,
same small modular family. This elapsed DS12 grid proved to have empty support
and must not be fitted. The successor uses MG12 as defined in the linked plan,
not padding or substitution into this sealed recipe.
Every hourly origin still gets exact endpoint heads.
Freeze its support/cost design before fitting; it is not twelve-hour issuance.

Six declared heads need not all be available near the end of the market week.
The strategy consumes masks, never substitutes current price for missing heads.
Use the existing contract's minimum valid long-head requirement for entry.
Opening evaluates long forecasts for buy/sell TP/SL and expected profit/risk;
management combines short and long forecasts in temporal order to detect SL
before TP. Verify the TP-at-2h/SL-at-12h versus reversed-order counterexample.
Do not claim continuous-path first-passage certainty from sparse endpoints.
One concurrent position per asset; forced flat and entry guards are mandatory.

Before trading, per-horizon annual validation eligibility is MAE_h < naive_h
on the matching support. No failed head is hidden by an aggregate score. If
eligible masked heads are insufficient for the full strategy, return
PAIR_NOT_ELIGIBLE; do not synthesize heads or silently change the strategy.
No trading run with only the short model is a full-strategy result.

Before TEST freeze both models' architectures, feature policy, costs, heads,
calendar, strategy version, cadence, seed and final-fit procedure. Under the
previously accepted final-refit policy, fit once on `[2021-01-01, 2025-01-01)`
using the sealed inner/checkpoint rule, then freeze weights throughout TEST.
This is one scheduled pre-TEST refit, not quarterly/weekly adaptation. It is
distinct from evaluating the original 2020-2023 validation checkpoint; never
switch policies silently. TEST can open automatically only after all data
provenance, eligibility, software and procedure gates have verifiable receipts.
If a gate fails, return a typed terminal and missing object; no repeated prompts.

On the same full TEST year, report forecasts plus the frozen pair's backtest:
initial equity 10000; net PnL/return, closed/open trades, marked-to-market equity,
maximum drawdown, annualized Sharpe with declared return clock, Sortino,
profit factor, exposure/turnover and all commission/spread/slippage/swap costs.
Zero-trade and undefined-denominator metrics are explicit, not invented zeros.
Keep a cash reference and business-matched passive reference; label conventional
buy-and-hold separately if it violates the no-weekend-position restriction.

For static and future cadence arms, matching short/long model hashes identify
every forecast and trade. Never choose a new pair or thresholds from TEST.
Later cadence comparisons use the same validation origins and initial recipe;
quarterly/monthly/weekly count every expected segment, with four-year rolling
windows, mature labels and continuous equity. No assumption of monotonic gains.

## 6. Automation, resources and evidence

Satoshi implements the CLI contract in `ACCEPTANCE.md` and follows the handoff.
All metrics, failures, exclusions, model/genome identities and compute costs go
to the existing OLAP with idempotent write plus exact readback. Durable local
outbox delivery retries must not retrain. Create verified physical analytical
copies on two hosts; Git stores scripts, small configs, summaries and digests,
not datasets, weights, secrets or database binaries.

Preferred first scientific fit: 5090. Its host runs only one heavy GPU process;
5070 Ti is not a parallel slot. The secondary handles CPU preparation, producer/clock
recovery and later independent admitted stages; the coordinator runs bounded orchestration
and warehouse delivery, protecting desktop memory. One fixed fit cannot occupy
all GPUs without duplication; utilization is not an acceptance criterion.

Use a full-workload first fit as the measurement and adopt its terminal; do
not rerun it as a pilot. Initial governed cap <=7936 MiB on the preferred host
only with fresh slice/host admission and reserve for other processes. Subsequent
production cap is 1.25 times measured full-tree peak, only if it fits. Never
trim the margin after a refusal; route an eligible job to the secondary host.
No CPU fallback for scientific GPU fits, no blind cache drop or machine reboot.

Durable exclusive claims, heartbeats, expiry/recovery, atomic terminals and
identity-based reuse must survive power loss. No chat agent monitors logs.
Every level offers status/ETA: unit, stage, campaign, outbox and fleet health.
ETA is unmeasured until real timings exist and includes fitting, scoring and
publication separately. Publish complete expected/finished/failed denominators.
Produce a linked result index with <=one paragraph of numeric findings per
substage, and an M1-M8 progress plot generated from evidence without made-up
percentages or marking paused work complete.
