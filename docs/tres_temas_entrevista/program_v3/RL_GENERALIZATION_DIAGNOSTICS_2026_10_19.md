# SAC Generalization Diagnostic Plan

Executed narrow checkpoint analysis: [GD18 result](RL_CHECKPOINT_DIAGNOSTIC_RESULT_2026_10_19.md).
Other diagnostic measurements below remain planned unless separately evidenced.

Date: 2026-10-19. Owner request: distinguish every suspected cause of poor
generalization using measured evidence and targeted interventions. This is a
diagnostic specification, not a claim that new telemetry has been implemented.
The current sealed regularization sequence continues unchanged. No new temporal
fit is dispatched by this document; the existing pause remains in force.

## Decision Contract

The outcome of interest is better VALIDATION net trading performance at
comparable risk, costs, support and business constraints. A smaller TRAIN to
VALIDATION gap caused only by degrading TRAIN is not success. No-trade is an
explicit zero-return control, with zero activity reported honestly.

Validation is development data used repeatedly for checkpoint/model selection.
It is not an untouched holdout. TEST remains outside this diagnostic program.
An apparent failure of a regularizer at one strength and one seed does not rule
out its family. An apparent gap does not distinguish memorization from drift.
Early stopping being enabled does not establish that overtraining was avoided.

Every hypothesis has an evidence state: NOT_MEASURED, INCONCLUSIVE,
SUPPORTED_ON_THIS_CONTRAST, or NOT_SUPPORTED_ON_THIS_CONTRAST. A failed integrity
invariant is instead DEFECT_CONFIRMED. A signal is not proof of a cause; only a
controlled intervention supports the corresponding remedy, within its scope.

## Common Measurements

For each frozen checkpoint, use the existing full inference traces, with no
updates, and the exact business support. Let r_d = log(E_d / E_previous_d),
using observed NY business-day closing equity and initial equity for the first
day. Keep zero-return eligible days; do not insert synthetic weekend bars.

- g_T, g_V: mean daily log growth for TRAIN and VALIDATION, separately.
- G = g_T - g_V: descriptive generalization gap in the same daily units.
  Do not divide by TRAIN profit or compare four-year raw profit to one year.
- Delta_V = g_V(treatment) - g_V(reference), paired on the same validation days.
- Gross/net PnL, return, maximum drawdown, trade count, win rate, profit factor,
  exposure, turnover, commission, financing and execution-cost assumptions.
  Separate recorded debits from slippage embedded in fill prices.
- Per-year TRAIN and per-quarter TRAIN/VALIDATION panels; retain full-period
  results. Mark a panel in-sample when its data influenced that checkpoint.
- Sharpe frequency, risk-free convention and any annualization factor explicit;
  do not relabel the existing nonannualized daily Sharpe as annualized.
- Return concentration: largest winning trade and top-five winning trades as a
  fraction of total positive trade PnL, not of possibly near-zero net PnL.
- Learning trajectory: transitions, updates, selected step, stop reason,
  patience, and both role metrics at every completed checkpoint pair.

Uncertainty: paired contiguous-block bootstrap of daily return differences,
with block lengths 5, 10 and 20 observed business days as sensitivity settings;
these are not three independent confirmations. Retain chronological quarterly
panels because bootstrap does not remove market drift. Separate across-seed
variation from time-series uncertainty. Initial finalist replication uses three
predeclared shared seeds including seed7, with more only if the comparison stays
unresolved. Three seeds are a budgeted start, not a certificate of sufficiency.
Intervals after adaptive model selection remain descriptive development evidence.

## Hypothesis Register

| ID | Hypothesis | Measurements / evidence | Targeted response and interpretation |
| --- | --- | --- | --- |
| GD01 | Excess capacity | Actor, critic and encoder parameter counts; g_T/g_V curves versus capacity; per-seed variability | Compare [8], [16], [16,16], [32,32] on identical inputs/budget, then isolate actor vs critic if justified. Lower capacity helping V supports this contrast, not a universal size law. |
| GD02 | Inadequate temporal representation | Paired Delta_V at comparable total parameter budgets; temporal-order sensitivity on separate diagnostic inputs | Candidate small causal Conv1D representation vs flat input, same window. Permutation is a stress test that can be out of distribution, not causal proof. Temporal dispatch remains paused. |
| GD03 | Excessive compression | Latent variance/effective rank on standardized fixed TRAIN probes; current-tick perturbation sensitivity; Delta_V across bottlenecks | Compare two small latent widths and a declared current-state bypass. Low latent rank alone is not failure; useful states can be low-dimensional. |
| GD04 | Wrong context/resolution | Delta_V, exposure and cost across window sizes at fixed stride; then stride at matched time span | Separate window and sampling tests. Record observed-bar span and elapsed hours, including closure offsets. Keep current-bar inclusion exact. |
| GD05 | Insufficient stable information | Group ablation; blocked out-of-period auxiliary predictive probes; lagged/shifted negative controls with purged label support | Compare simple candidate signals before adding large networks. Failure of finitely many probes cannot prove absence of exploitable information. Selection scores alone do not establish trading value. |
| GD06 | Few independent experiences | Unique timestamps, episodes, traded days and trades; return autocorrelation; concentration and block-bootstrap width | Increase diversity within allowed TRAIN periods, or balanced chronological episode coverage. More gradient steps or overlapping windows are not more independent data. |
| GD07 | Regime/distribution shift | Per-quarter outcomes; feature quantile shifts and normalized Wasserstein distance; volatility/session/exposure distributions | Compare shifts against TRAIN chronological block-to-block reference variation. Consider regime conditioning or history-length/adaptation contrasts later; covariate drift alone does not prove conditional drift. |
| GD08 | Scaling or saturation | Finite/clip rates per feature and split; quantiles; activation saturation; normalizer identity/availability | Correct a demonstrated transform defect. Otherwise test a separately sealed normalization variant. No fit to future validation observations for a causal inference transform. |
| GD09 | Missing business state | Observation inventory for position, entry allowance, SL/TP distance, time to mandatory close; identical-observation cases with distinct legal actions | Controlled state-field ablation/addition. A representation unable to distinguish different constraints can make the problem partially observed. |
| GD10 | Reward/business mismatch | Stepwise reward-to-ledger reconciliation; gross/net return; exposure and drawdown; directional edge vs sizing contribution | Correct arithmetic first; otherwise compare a separately declared reward or fixed-size diagnostic. Preserve SL/TP and business rules in every arm. |
| GD11 | Critic overestimation/divergence | Q1/Q2 disagreement, Q magnitude, held-out TD residual, action-conditioned calibration against compatible rollout returns | Use fixed probes, recorded gamma, entropy alpha, termination/truncation semantics. A soft-Q target includes future entropy and must not be compared directly to undiscounted cash PnL. A complete calibration may require new frozen-policy rollouts; mark unavailable rather than fabricate. |
| GD12 | Replay imbalance / optimizer / plasticity | Updates per transition, replay age/time coverage, sample reuse, actor/critic gradient and update norms, loss trends, inactive activations | Change one of replay scheduling, learning rate or update ratio with fixed data budget. Different update ratios require reporting both transitions and updates. Loss/reset experiments only after evidence. |
| GD13 | Exploration or action mapping | Entropy/alpha, proposed/executed actions, rejected-action reasons, action saturation, fraction mapped to identical execution | Diagnose the continuous-to-order adapter and compare exploration settings. Deterministic inference remains the common evaluation convention; entropy alone is not trading quality. |
| GD14 | Misplaced/too-strong regularization | Exact affected components; train/eval modes; BN running statistics; gradient/activation changes; matched Delta_V | Finish the four critic-only arms. Distinguish actor regularization from critic regularization. Do not combine BN/dropout or alter an active scientific identity. |
| GD15 | Simulator/accounting optimism | Ledger conservation residuals; per-fill commission/financing; SL/TP collision and forced-close cases; actual spread/slippage support | Independent replay of fills and adverse cases; broker-calibrated cost sensitivity as a separate protocol. Passing accounting tests does not certify market execution realism. |
| GD16 | Seed luck / adaptive winner selection | Across-seed Delta_V, block intervals, profitable-quarter fraction, trial ledger and PnL concentration | Replicate finalists with common seed schedule; retain all arms and uncertainty. Do not select the best seed or manufacture an independent claim from reused VALIDATION. |
| GD17 | Evaluation/causal/software defects | Checkpoint/config/data hashes; exact paired support; business violations; future perturbations; terminal/bootstrap and recurrent-state handling | Zero violations required for exact invariants; use predefined numerical tolerances for numeric comparisons. A demonstrated defect blocks affected comparisons and triggers scoped repair. |
| GD18 | Overtraining / checkpoint objective imbalance | Count checkpoint improvements of M accompanied by worsening g_V; patience resets; curves of g_T, g_V and G | Report the trade-off under the unchanged mean objective. A validation-only or constrained selector is a future explicit design change, never silently substituted. Cap termination is not evidence that overtraining was excluded. |
| GD19 | Underfitting / failed optimization | Poor T and V relative to controls, plateaued returns, gradients, replay coverage, critic calibration; response to modest capacity or optimization changes | Distinguish insufficient capacity from poor exploration/optimization and weak signal. Underfitting of useful structure can coexist with memorization of incidental patterns. |

## Evidence Collection Order

1. D0, postprocessing without fits: read authenticated completed pairs and final
   traces from the existing sequence; compute common metrics, quarterly panels,
   gap trajectory, concentration, objective-disagreement counts and uncertainty.
   Extract only metrics supported by retained fields. No edits inside closed
   campaign directories, no changes to active workers, no repeated log polling.
2. D1, frozen-checkpoint diagnostics: inventory actual observation/action/normalizer
   paths; independent accounting, future-perturbation and policy replay checks.
   Bound CPU/RAM and use completed copies. Reuse existing verified evidence with
   its scope; do not call metadata alone an independent accounting replay.
3. D2, instrumentation for new identities: gradient/replay/entropy/action telemetry
   and fixed-probe Q diagnostics, only where not already retained. Tests precede
   deployment; observational probes preserve optimizer, RNG, replay and BN state.
   Never retrofit missing historical measurements as if they were collected.
4. D3, targeted small contrasts: rank hypotheses by observed evidence, expected
   diagnostic value and cost. Small capacity is the first architectural candidate;
   temporal representation remains a documented candidate pending release of its
   existing pause. Do not create an exhaustive cross-product of interventions.
5. D4, replication: compare final candidates on the identical development support
   and predeclared seeds. Later data may be required to resolve stability; retain
   INCONCLUSIVE rather than increase searches until a desired result appears.

All stages need finite resumable scripts, machine-readable status and estimated
remaining work where measured. D0/D1 on CPU while the sealed GPU queue proceeds;
new GPU work only after predecessor closure and admission. Prefer the 5090;
one heavy process on its shared-RAM host. No extra training just to keep a GPU busy.
These stages are planned work, not deployed timers or completed diagnoses.

## Decision Rules

For exact integrity requirements, expected violations = 0. For numerical
reconciliation, specify absolute/relative tolerances and units before evaluating.
For statistical indicators, use within-TRAIN chronological reference variation
and matched treatment differences; do not invent universal thresholds such as
"PSI > 0.2 proves failure" or "small loss proves a good critic".

A treatment with positive Delta_V and a paired interval excluding zero is a
promising development contrast only if risk/activity/business constraints remain
acceptable. An interval spanning zero is INCONCLUSIVE, not a forced failure.
Compare no-trade as well: improving from a large loss to a small loss is useful
engineering evidence but not profitable trading. Retain the full return/risk
trade-off instead of selecting by gap alone. No new maximum-DD threshold is
invented here; any hard risk promotion rule must come from the business contract.

## Persistence And Acceptance

Use the existing governed warehouse outbox/readback and physical backup/mirror
path, with a new diagnostic protocol identity. No parallel database or blockchain.
Each metric carries campaign/contract/checkpoint/data/feature/config/code digests,
seed, split and period, support count, units, metric definition, diagnostic version,
reference/treatment identities and input artifact hashes. Unavailable metrics have
null value plus a typed reason; never zero-fill. Bootstrap settings/seed are pinned.
Hypothesis status and evidence links are separate from numerical metric records.

Acceptance tests for the future runner: unequal-duration normalization; constant
equity/no-trade; known ledger and objective-disagreement examples; zero/negative
equity refusal; cumulative fee non-double-counting; paired-support mismatch;
deterministic contiguous-block bootstrap; closed-directory immutability; missing
telemetry typed as unavailable; restart/idempotent publication/readback; exact
checkpoint replay without changing RNG, replay, optimizer or BN statistics.

## Sources And Limits

- Nauman et al. (2024), overestimation, overfitting and plasticity in actor-critic:
  https://proceedings.mlr.press/v235/nauman24a.html
- Hussing et al. (2024), value divergence at high update ratios:
  https://rlj.org/papers/hussing2024dissecting
- Agarwal et al. (2021), uncertainty in few-run RL evaluation:
  https://arxiv.org/abs/2108.13264

These motivate diagnostics, not a financial-profitability guarantee. The block
lengths, capacity candidates and diagnostic ordering above are our prospective
engineering choices, not findings attributed to these papers.
