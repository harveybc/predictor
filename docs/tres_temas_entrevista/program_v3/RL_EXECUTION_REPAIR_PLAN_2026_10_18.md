# RL execution repair and data handoff

Owner approval: continue autonomously, plan first, adapt to evidence. Musashi
owns RL; Satoshi's concurrent data work must not be duplicated or overwritten.
This plan refines RL_FIRST, not a new forecasting or DOIN campaign.

## Current continuation, 2026-10-19

EURUSD core admission is now complete: 37309 observed H1 bars, all 72 months,
zero missing required hours on the declared conservative operating calendar.
37306 rows replay exactly from M30; three verified native fallback rows retain
their provenance. This admits the independent raw-price control, not the full
inventory or final feature selection. Its pipeline uses four-year TRAIN and
one-year VALIDATION without refitting, 13 observations at stride two, a
single 5090 job, four mechanical controls then flat/compact-temporal SAC.
The financial-data raw-input renderer binds prices, calendar, availability,
transforms and explicit initial warmup; the engine must first pass real-data
completed-bar/fill/session acceptance. Nonzero analytical trading-cost
assumptions are frozen and labelled, not claimed broker measurements.
The full catalog and selection rebuilding continue under their own admission.
No further owner approval is required for these already authorized actions.

| Step | Work and acceptance | State |
|---|---|---|
| 1 | Read generated recovery and selection receipts; verify input bytes, timestamps/availability, business calendar and complete split support | CHECKED: neither data admission nor rebuilt selection complete |
| 2 | Real-engine action at completed bar t executes at next open; reward, fills and observation describe that same transition; terminal/cancel/forced-flat behavior preserved | IMPLEMENTED: next_open_v2; synthetic acceptance, broader release acceptance pending |
| 3 | Costs, financing, undefined fitness, continuous risk and business rules tested with no-trade/buy/sell/scripted controls | PARTIAL: commission and leveraged entry-bar settlement, finite fitness and explicit analytical financing repaired; broker costs and full risk acceptance pending |
| 4 | Accepted selection and data feed the same static SAC flat/Conv1D comparison; frozen evaluation, metrics/checkpoints/outbox, bounded unattended execution and status | PENDING_DEPENDENCIES |

## Immediate test design

On synthetic actual-engine episodes: reset exposes completed t; first and every
later step advances exactly one observed bar; BUY fills at the next open and
its commission/reward is returned with that BUY, not the next HOLD. Perturb
unavailable future bars and compare earlier transitions. Include strict
13-observation/stride-2 warmup, final-bar termination, interrupted reset,
protective exits, session wind-down, forced flatten, quota and cancellation.
Do not accept the same-bar repeated observation or hide a lag in a plot.

Review financing and real fill costs separately; unknown costs cannot become
zeros. Repair defects at their owning simulator/agent interfaces and test each
change. No artificial price interpolation, calendar guesses or restored old
feature lists to make an input pass.

## Handoff and execution

Use generated admission/selection manifests, not a verbal 'done'. Record exact
missing artifacts and proceed with independent CPU engineering meanwhile.
No financial fits until those artifacts and simulator checks pass. Once ready,
use the existing governed publisher and physical mirrors. Initial static
comparison keeps one entry/day, one position, 12-hour close/open exclusions,
13 observations at stride 2, small causal Conv1D branches/core and fixed splits.
Daily entry caps 2/4 remain a later controlled contrast. No real-money trading.

Persist progress here and in the program/engineering method states. Do not
require another approval for already scoped repairs or completed dependencies.

## Execution result, 2026-10-18

No financial fit. The new explicit next-open path fixes delayed action/fill/reward
attribution and a calendar index that could refer to the next row. Real-engine
synthetic cases cover long/short, exact one-bar advance, 13x2 strict warmup,
future perturbation, reset, terminal/error propagation and 12-hour business
buffers across a calendar closure. Entry-bar stop settlement now includes both
commissions in net trade PnL and correctly returns leveraged entry capital.
Financing uses Backtrader's existing annual-notional daily debit only when that
analytical model and rate are explicitly configured; this is NOT verified broker
FX rollover. Missing/nonfinite return or risk cannot select a checkpoint.

Verification: gym-fx focused suite 256 passed; agent-multi focused suite 65
passed; acceptance-runner unit tests 7 passed. The bounded CPU runner is
`agent-multi/tools/verify_rl_static_engineering.py`, with durable status, per-stage
logs and JUnit counts. It does not start financial jobs or grant admission.

Data receipt at this check: 612/636 weekly requests available, 24 unavailable;
84/110 requested alternative dates available, 26 unavailable. These denominators
include calendar/source-candidate questions, not certified missing-trading-hour
counts. Both admission and rebuilt feature selection remain false. Latest remote
Satoshi data return remains 9c371fe88; additional unpublished work is not assumed.

Next independently executable task: finish environment risk/accounting acceptance
and broker-cost contract; accept Satoshi's additional source/calendar evidence
when published, recover real missing observations, admit replacements, rebuild
preprocessing/selection, then bind the compact static comparison to those bytes.
No need for another owner approval to continue these already scoped tasks.

Implementation pins: gym-fx `codex/rl-static-inputs-20261018` @5c97750;
agent-multi `codex/rl-static-temporal-20261018` @3cf82c12. No deployment or merge
into an active scientific executor is implied by these engineering branches.

## Subsequent verified repairs

The next-open direct settlement flattened cash/position but had left the native
Backtrader Trade open. Four repeated-close regressions reproduced the missing
second trade before repair. Native lifecycle and analyzer notification now close
once, without a second broker order; net PnL zero also stays zero. Nine new tests
cover both directions, leverage, repeated closes and a nonzero-slippage/fee/
financing hand ledger. Full focused CPU runner: 256 gym + 65 agent passes.

Forexite's primary MetaStock documentation explicitly specifies CET in winter,
CEST in summer. The provider-clock contract and normalization code are in
financial-data @f7d051754. Separate candidate output: 86 files, 206,858 rows;
no interpolation, price modification or admission. Quote side and minute-label
semantics still need resolution. The file cohort includes diagnostic dates and
must not be confused with the 84/110 requested-date recovery denominator.
One new clock-only warehouse report (6 metrics) was read back successfully;
no pending/mismatched delivery. A 1,585,152-byte DuckDB snapshot has matching
copies on two machines, SHA-256
`739f3bfd5ef8a71378c99b2c2c23be3c56ea37c8d9412808547eb39263e2ef1a`.
These quality measurements do not grant data or feature-selection admission.

## Additional execution, same data-first order

Recovered and authenticated 40 missing FXCM boundary hours per pair from the
2019/53 source archive; new candidate files preserve native prices and timestamp
availability, with atomic completion and read-only status. Data code 5866040b6,
25 parser/integrity tests passed. No source overwrite or interpolation.

Downloaded four additional annual HistData archives for both pairs, 2024/2025.
Raw-source clock comparisons now cover 2023-2025: 6 warehouse reports, 22,890
metrics, readback without pending/mismatch and verified copies on two machines.
These are source diagnostics, not experiment performance. Freeze the empirically
supported time convention before using replacement observations; do not silently
carry the conflicting provider-declared clock into the admitted lake.

Feature-selector 2e90ee2 now executes a resumable passage through the existing
PROFILE_ONLY worker, pairwise metrics and nine filters. Synthetic tests include
real engines, row readback, restart and target availability at TRAIN/fold ends.
It does not confer financial-data admission, causal evidence, final selection
or remote publication. Deployment still needs the actual admitted input bundle,
resource acceptance, remote OLAP/mirrors and the remaining selection stages.
This preparation runs independently of source recovery, with no financial fits.
