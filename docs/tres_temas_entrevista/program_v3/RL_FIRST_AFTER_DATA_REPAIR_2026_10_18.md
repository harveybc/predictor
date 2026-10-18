# RL First After Data Repair

Authority: owner's 2026-10-18 change of priority. Status: approved direction,
not an implemented or launched campaign. This overrides the former requirement
to finish forecasting experiments and all representation studies before RL.

## 1. Execution Order

1. Recover observed data; resolve clocks, calendars, availability and actual
   monthly support. Admit the required raw and derived inventory to the lake.
2. Recompute preprocessing and feature selection on the admitted population.
   Reuse tested code, not deleted rankings, weights or numerical results.
3. RL0: certify the existing gym-fx environment against the business contract
   and run synthetic, CPU-only accounting/causality tests.
4. RL1: replay no-trade and passive/rule-based controls on common support.
5. RL2: train/evaluate the existing SAC plugin, first a compact control and
   then the existing modular temporal representation on the same selected
   information. The compact control does not replace the modular experiment.
   Owner clarification: this FIRST contrast is STATIC, refit cadence NONE.
   Validation/test evaluation performs inference without optimizer/replay
   updates. Periodic adaptation is deferred to RL4, not silently enabled.
6. RL3: DQN for a separately declared discrete action space; PPO/MaskablePPO
   and a modern continuous-control challenger only after the action interface
   and first SAC result make the comparison meaningful. No algorithm is
   assumed to be the universally best trading method.
7. RL4: controlled cadence/history experiments; net trading evaluation and
   automatic DEMO/PAPER integration under the existing no-real-money rule.
8. Deferred final work: supervised short/long forecasting and its heuristics,
   prediction-cadence/early-stopping studies, then NEAT and the other optional
   extensions. An approved finite reproduction for literature remains distinct.

The RL path is not conditional on beating a price-persistence MAE. Predictive
selectors may still use supervised probes, but cannot silently relaunch the
deferred predictor campaign or require all forecast heads to beat naive before
an RL policy is eligible to be evaluated.

## 2. Refit Insight, Not A Retained Performance Claim

The owner observed that frequent warm-start adjustment appeared far more useful
than static fitting, especially with separate short/long multi-output models
and a compact 13-observation input sampled every two open-market hours. Keep
this as a qualitative hypothesis and prospective recipe idea only. The old
affected experiments are not donors, baselines, champions or evidence of OOS
skill; this note does not restore their metrics or authorize keeping artifacts.

HYP-R1: adaptation cadence and history length may interact with representation,
checkpoint monitoring and regime change. More frequent updates are not assumed
to improve returns or prediction error monotonically. Predict first, seal the
decision, observe the matured outcome, score, then update for later decisions.
Training on a week's observations and scoring that same week with the updated
weights is a fitted/development score, not an unseen-week estimate.

HYP-R2: per-tick adaptation resembles online learning, but does not turn a
supervised predictor into an RL agent. RL additionally learns a policy from
actions/rewards/dynamics. Both can overfit, and replay cannot reveal a future
transition/reward before it becomes available.

## 3. Business Contract, Required Before RL Fits

- EURUSD first; ETH only under a separate 24x7 calendar/instrument contract.
- One concurrent position per instrument; BUY, SELL and FLAT semantics explicit.
- At most one entry per business day (initial timezone America/New_York).
  Protective/closing orders do not consume or obey the entry quota. Count both
  submissions and filled entry parents, including delayed and partial fills;
  cancellation does not restore the day's submission allowance. Expose quota
  availability in the policy observation, and report entry and closed-trade
  counts separately. A reversal closes first, with no same-action new entry.
- Versioned source/broker session calendar and timezone, DST and holidays.
  Force flat 12 hours before close; prevent reopening until 12 hours after open.
  Entry cutoff/wind-down is configured, never guessed from a weekday name.
- Inputs may contain observed prior-week history. Closed hours add no invented
  observations; unexplained open-session gaps cannot become adjacent ticks.
- The environment must enforce the rules for every agent, including random
  controls, rather than rely on a learned policy obeying them.
- An input for decision t is available by t, including the latest completed
  bar. Execution uses an explicitly available quote with declared latency,
  never that bar's earlier close at an impossible decision time. No intrabar
  TP/SL priority can be inferred from an hourly OHLC without an ambiguity rule.
- Equity, margin, notional sizing, leverage, liquidation, commission, spread,
  slippage, financing and forced-close costs belong to gym-fx/instrument
  contracts. A bid candle plus an unsynchronized ask endpoint is not a measured
  executable spread. Assumed costs must be labeled and sensitivity-tested.
- Initial capital USD10,000 for comparable simulations; capital and exposure
  limits remain explicit configuration, not implicit library defaults.
- WEEKLY warm-start updates during closure are the business default. NONE,
  QUARTERLY, MONTHLY, DAILY and per-tick are separately identified experiments,
  not a reason to discard the default or an automatic path to more updates.
  For the first approved RL comparison, NONE explicitly overrides that default.
- TRAIN2020-2023, VALIDATION2024 and TEST2025 are the declared time ranges.
  TEST2025 has already been examined in the prior initiative; repaired bytes
  do not make it a pristine final holdout. Declare this and reserve genuinely
  untouched future/forward-paper observations for independent confirmation.

PDT is not a universal EURUSD restriction. FINRA Notice26-10 replaced the old
day-trading margin requirements effective2026-06-04 with a transition through
2027-10-20. Applicability and a broker's adoption depend on instrument/account;
do not disable a live restriction on the strength of a political statement.
Record broker/account rule state through the existing private configuration,
without placing identifiers or credentials in public repositories.

## 4. Reuse And Ownership

`financial-data`: acquisition/inventory; `feature-eng` and `preprocessor`:
causal transformations; `feature-selector`: selection; `feature-extractor`:
reusable representation code; `gym-fx`: simulator/accounting; `agent-multi`:
agents, fitting, evaluation and campaigns; `doin-*`: established collaborative
optimization; warehouse: all metrics and candidate genomes. Predictor does not
own RL training. Do not reimplement SAC, PPO, DQN or the DOIN protocol.

SAC requires continuous actions and DQN discrete actions in the existing SB3
implementations. Define a genuine continuous control (for example a bounded
target exposure within one net position) for SAC rather than pretending a
rounded discrete action is an equivalent algorithm comparison. Use a suitable
discrete implementation for explicit hold/open/close choices. Compare economic
constraints fairly and state any remaining difference in action expressivity.

Reusing accepted feature-generation/representation code is preferred. Learned
transforms require admitted data and fold-causal training; no retired encoder
weights or causal scores may re-enter. Reconstruction loss is not policy value.
RL-specific selection must retain direction/event/risk relevance and not reject
a feature merely because a price-regression probe did not beat persistence.

## 5. Metrics And Acceptance

Primary business measurements: net profit and return, end equity, maximum
drawdown, Sharpe/Sortino with declared sampling/annualization, trades, turnover,
exposure, profit factor, transaction-cost breakdown, liquidation/constraint
violations, and weekly/yearly coverage. Undefined ratios stay undefined.
Use an equity-return series including flat periods, not only closed trades.
Winning-trade percentage uses closed trades and net PnL, not submitted orders.

The owner subsequently asked about caps of two or four entries/day. Keep one
as the accepted control and prepare a proposed 1/2/4 contrast; this is a maximum,
never a minimum activity objective. Compare each policy trained under its own
cap on the same support, costs, seed protocol and daily risk budget. Do not
infer a higher allowable leverage or daily loss budget from more entries.
No live broker rule is disabled. The caps are experiment parameters, not a
representation of FINRA's former rolling-five-day PDT count.

Engineering specification and test mapping now live in agent-multi:
`docs/work_plan/43_STATIC_RL_CURATED_COMPARISON.md`. Initial temporal branch
filters=4, core filters=8, kernels=3, no temporal downsampling inside the
encoder; flatten only at the actor/critic head boundary. The per-feature
convolutions remain trainable with the agent. These are hypotheses, not fitted
optimal sizes or transferred weights. Starting history=13 observations with
input stride=2 admitted hourly bars, including the current completed bar.

Report TRAIN and VALIDATION separately, including fixed-policy diagnostics and
prospective walk-forward scores as distinct populations. Online updates may
use only matured past transitions. Selection/checkpoint monitoring uses TRAIN
and development validation, never future evaluation rewards. Offline replay
does not demonstrate execution profitability. No price-MAE gate for RL.

Baselines: NO_TRADE; passive BUY and SELL restricted to the same tradable window,
forced flat and exposure/cost rules; optional unrestricted buy-and-hold only as
an explicitly non-business-matched reference. Use a random-action negative
control and existing simple rules. Drawdown is a metric, not itself a strategy.
Candidate progress compares actual immutable policy/config/data identities.

Before a campaign, freeze reward, action mapping, episode/reset rules, model
sizes, update schedule, stopping monitor/patience/tolerance/budget, resource
limits and an exact manifest of runs. SAC entropy-augmented training return is
not substituted for net evaluation PnL. A profitable training score is not a
deployment decision. Use temporal validation and cost perturbations, then the
existing DEMO/PAPER procedure; no real capital.

Every result goes to the existing warehouse with exact readback, genome/config,
seed, data/calendar/business/reward/model/code identity and measured cost.
Retain a closed physical database copy and second-host hash verification; no
large data or metric database in Git. Config/code/regeneration commands do go
to Git. Expose status/ETA at job/stage/campaign level; unavailable ETA is not0.

## 6. Required Acceptance Tests And Automation

| ID | Required behavioral evidence before the corresponding stage |
|---|---|
| RL00 | Actual admitted inventory, monthly prepared support, selection manifest; changed input rejects reuse |
| RL01 | Future-price perturbation does not change observations/actions at t; fills/rewards obey availability |
| RL02 | Long/short round trips and flat account hand-calculated; each cost charged once; equity/PnL reconcile |
| RL03 | Forced flatten/reopen/entry-cutoff, DST and holidays; all agents obey; no fabricated weekend fills |
| RL04 | Intrabar ambiguity, gaps, terminal/truncated episodes, hidden-state/replay reset and leakage controls |
| RL05 | Frozen weights at scoring; checkpoint/prediction before reward/update; future week cannot change prior bytes |
| RL06 | Baselines share actual support, capital, exposure and execution costs; undefined metrics/cash periods counted |
| RL07 | L1 stopping monitor, restores best checkpoint, reason/steps; L2 stopping distinct, no silent tiny search |
| RL08 | Durable job DAG, idempotent restart, typed retry/cooldown, bounded RAM/VRAM/disk and restart after power loss |
| RL09 | All metrics/genomes outbox/readback, physical mirrors and a reproducible final policy artifact |
| RL10 | Paper endpoint/account verified; real capital rejected; no policy substitution after validation |

CPU synthetic verification can run while data are repaired, with no scientific
performance claim. GPU priority after acceptance:5090 first,4090 for independent
admitted runs; no simultaneous heavy5090/5070Ti workloads until actual host RAM
supports them. DOIN only when a frozen search is required: existing node/plugin
JSONs, same experiment/seed/commits/chain, normal broadcast verification and
migration protocol, no new champion/consensus mechanism. Do not start it now.

## 7. Research Sources, Consulted2026-10-18

- [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html):
  action-space compatibility, evaluation and iterative environment validation.
- [SB3 Contrib](https://sb3-contrib.readthedocs.io/): maintained TQC, RecurrentPPO,
  MaskablePPO and QR-DQN implementations; alternatives, not promised FX winners.
- [TQC original paper](https://arxiv.org/abs/2005.04269): critic overestimation
  control in continuous RL; trading relevance remains to be demonstrated.
- [Online Forex RL](https://arxiv.org/abs/2106.03035): transaction costs and
  adaptation are directly relevant; not evidence our configuration is profitable.
- [FINRA Notice26-10](https://www.finra.org/rules-guidance/notices/26-10):
  effective and transition dates, rather than a blanket assumption PDT vanished.

Next: finish clock/value/calendar and inventory admission; Satoshi can perform
the bounded independent tasks in SATOSHI_DATA_AND_RL_ORDER_2026_10_18.md without
touching live data services or launching any scientific run.
