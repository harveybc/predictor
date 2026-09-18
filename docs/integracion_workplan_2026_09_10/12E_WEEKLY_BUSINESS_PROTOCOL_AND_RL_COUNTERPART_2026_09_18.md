# 12E — Weekly-trading business protocol and the RL counterpart (S4, design only)

**Dependency correction, 2026-09-18:** [master v3](https://github.com/harveybc/predictor/blob/master/docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md)
makes this application mandatory and distinct from P-L2. Prepare its business/data
contract now; it does not wait for a successful sinusoid run, DOIN or a positive
H1. Historical/point-in-time eligibility and economic execution assumptions still
must hold for its own evaluation. The old next-action references to 12D below do
not govern dispatch. No RL experiment is marked executed by this correction.

Order: `MUSASHI_UTILITY_STATUS_AND_WEEKLY_TRADING_REQUIREMENTS_2026_09_18.md` (mandatory
requirements) and S4 of the adequacy order. **Nothing here is executed**: no financial scores,
no reserve, no trading, no GPU, no running service touched. Every requirement lists its gap and
owner; none is claimed to hold.

## Requirement → test → evidence table (business track)

| # | requirement | executable check | evidence today | status | owner | next action |
|---|---|---|---|---|---|---|
| 1 | business decision and weekly schedule: asset universe, decision times, horizons, training cutoff, release time, frozen interval; measured weekly cost | a written schedule with timestamps; a dry run of the weekly cycle with recorded durations | none in this repository; the consumers live in `prediction_provider` / `agent-multi` | NOT_TESTED | owner + Satoshi | read the deployed configs; write the schedule; measure one dry cycle |
| 2 | input information available at decision time: provenance, semantics, missingness, lookback, publication/receipt, revisions | per-feature availability ledger; point-in-time evidence, never inferred from bar close | data-gov lineage and column roles exist for governed inputs (P1–P7); no point-in-time receipts for feeds | NOT_TESTED | Satoshi | availability ledger per feature from feed receipts |
| 3 | enough data, empirically: independent weeks/episodes, regimes, usable samples after warm-up and purge, label balance; chronological learning curves at several lengths | predeclared learning curves over weekly evaluation units | synthetic learning curves designed (12D), not measured; no financial curves | NOT_TESTED | Satoshi | after 12D successor, the same design on governed financial inputs under an owner order |
| 4 | representative models: ridge/naive as baselines; forecasting/RL consumers derived from executable configs; optimizer updates, convergence, early stopping, tuning budget | instantiate the configured consumer models; record graph, updates, curves | plugins exist (`predictor_plugin_cnn/lstm`: composite, bidirectional, Bayesian heads — not plain Conv1D/LSTM); the adequacy learners are explicit stand-ins | NOT_TESTED | Satoshi | instantiate the deployed configs and record their graphs |
| 5 | fair representation comparison: raw/transformed/augmented with matched decision times and labels; history, dimensionality, capacity separated; train-only transformations; selection repeated inside weekly folds | the harness's pairs and capacity control, extended with information-span control | utility harness (H_T/H_A) exists; weekly-fold repetition not implemented | PARTIAL | Satoshi | add weekly-fold selection to the harness under a new order |
| 6 | temporal realism: rolling train → inner validation → next-week untouched evaluation; purge/embargo from consumed support; prefix, padding, reconstruction, missingness, restart on the deployed path (wavelets) | tests on the deployed preprocessing path | adequacy tests cover purge/forward/perturbation/restart on the synthetic path; D3 mechanics cover operator causality; deployed path not covered | PARTIAL | Satoshi | run the mechanics battery on the deployed preprocessing path |
| 7 | business outcomes: net outcomes, drawdown, turnover, exposure, slippage/fees, latency; policy effects in the actual simulator with baseline policies; uncertainty across weeks | simulator evaluation with fixed cost assumptions | none | NOT_TESTED | owner + Satoshi | fix cost assumptions; simulator run under an owner order |
| 8 | traceability and preregistered acceptance: every attempt with data/code/config identities, seeds, compute and metrics in data-gov and the warehouse; requirement→test→evidence table | as the D3/utility campaigns already do | governance in place for synthetic campaigns; no financial campaign registered | PARTIAL | Satoshi | reuse the governed runners for the financial protocol |

## RL counterpart (design)

* **Environment and agent**: the intended weekly-retrained agent lives in `agent-multi` (DQN
  configurations such as `dqn_btc_1h_twelve_atr.json`, `dqn_eth_1h_twelve_atr.json`) over the
  `gym-fx` environment. Nothing was instantiated here; the graphs, observation histories and
  reward definitions must be read from those executable configurations, not assumed.
* **Comparable representation arms**: raw observation history vs transformed vs augmented, with
  the same decision times, the same action/cost timing and the same reward; history length and
  dimensionality separated as in H_T/H_A; every transformation fitted on training weeks only.
* **Weekly cycle**: training cutoff → inner validation → frozen policy for the next week →
  untouched evaluation of that week; costs, turnover, exposure and drawdown recorded per week;
  baseline policies (flat, buy-and-hold, persistence signal) alongside.
* **What counts as evidence**: an RL smoke run is not an RL utility experiment; policy effects are
  read in the actual simulator under its declared assumptions with uncertainty across weeks and
  regimes. **No RL training or scoring is part of this CPU diagnostic order.**

## Boundaries

The adequacy pilot (12D) answers questions about learners on synthetic sinusoids only; it is not
evidence for weekly-retrained trading. The reserve stays untouched until the complete protocol
above is reviewed. Prior D2 candidate calibrations are regime-specific synthetic findings.
