# Utility status and weekly trading evaluation requirements

Owner request: explain useful findings, present stage and next step; prevent inadequate
data sizes or diagnostic models being presented as evidence for weekly-retrained trading.
Read: Satoshi return at `500f3c6`, utility R1-R4 and current 12C coverage map.

## Status and limits

We are in D3, development validation of the per-variable utility instrument, following
data profiling, noise/denoising work and operator mechanics. This is not completion of
the thirteen-step signal-processing plan, financial feature selection or trading validation.
The current experiment uses synthetic series, n=2048, one variable, horizon one, ridge,
four blocks and limited operator/regime coverage. It cannot establish sample adequacy,
model adequacy, profitability or weekly retraining feasibility for the business.

The preserved development experiment proposed no pair: 18 contrasts did not advance and
18 were inconclusive. The new aligned positive control detects 10/12 H_T effects, with
a wide interval reported as 0.52-0.98; null, information-loss and future-leak controls
meet their finite criteria. H_A remains inconclusive through calibration. These controls
validate a narrow instrument behavior, not every operator, noise regime or future-leak route.

Useful deliverables include governed input/result lineage, measured numerical/causal
mechanics, preservation of unsuccessful outcomes, and code-bound calibration reuse with
recovery tests. None is itself evidence of a profitable representation. Prior D2 candidate
calibrations are regime-specific synthetic findings, not a universal financial denoiser.

## Next step: design adequacy before more selection

The proposed 190 simulations are the minimum allowing ONE false advance under the stated
bound, not a guarantee of support. Do not repeatedly increase n or try fresh seeds until
passing. Before a successor, specify the acceptable false-advance rate, confidence,
probability of obtaining a decision at a declared alternative rate, fixed sample size or
valid sequential rule, budget and stopping disposition. Preserve all previous outcomes.
Resolve the instrument's calibration scope before launching the six remaining operators.
Operator-specific positive controls remain required; an aligned MAD control does not validate
wavelets, quantizers or learned representations. Any follow-on retains a DEVELOPMENT label.

In parallel, prepare the business protocol below from actual deployed/configured consumers
and available data. This document adds required work; it does not authorize new financial
scores, reserves, trading, GPU sweeps or changes to running services.

## Required business evidence before financial model selection

1. **Business decision and weekly schedule.** Document actual asset universe, decision
   times, horizons, weekly training cutoff, model release time and frozen interval of use.
   Include data collection, preprocessing, selection, training, validation and deployment
   in measured weekly cost. Do not assume a model trained after the cutoff was usable before it.
2. **Input information available then.** Per feature: provenance, semantics, missingness,
   lookback, publication/receipt or explicitly bounded availability, revisions and cost.
   Retrospective archives without point-in-time evidence cannot establish live feasibility.
   Calendar, labels and auxiliary inputs follow the same rule. Never infer arrival from bar close.
3. **Enough data, justified empirically.** Report effective independent weeks/episodes,
   regimes, market coverage, usable samples after warmup and label purging, and label balance.
   Use predeclared chronological learning curves at several training lengths and intervals
   over weekly evaluation units. Neither 2048 rows nor millions of overlapping windows
   establish adequacy. Declare imprecision or unsaturated learning curves honestly.
4. **Representative models.** Keep ridge/naive models as baselines and diagnostic probes,
   not as the sole basis for rejecting representations. Derive the intended forecasting/RL
   consumers from executable configurations, verify actual optimizer updates, convergence,
   early-stopping behavior and feasible tuning budget. Include the representative business
   model in later comparisons; a larger model is not automatically a better experiment.
5. **Fair representation comparison.** Compare raw, transformed and augmented inputs with
   matched decision times and labels; separate effects of history length, dimensionality,
   model capacity and representation. Fit scalers, feature selection and all transformations
   on training history only. Repeat selection inside each weekly fold when it would run weekly.
6. **Temporal realism.** Rolling train -> inner validation -> next-week untouched evaluation;
   purge/embargo derived from consumed feature and label support, no random time splits.
   Test prefix perturbation, padding, reconstruction, missingness and batch/restart behavior
   on the exact deployed preprocessing path, especially wavelets. A delayed representation
   must carry its real availability; causal does not mean zero-delay or perfect denoising.
7. **Business outcomes, not MAE alone.** Fix cost assumptions and compare net outcomes,
   drawdown, turnover, exposure, slippage/fees and latency, together with predictive metrics.
   Evaluate policy effects in the actual simulator under its assumptions, include baseline
   policies and report uncertainty across weeks/regimes. No profitability claim from synthetic MAE.
8. **Traceability and preregistered acceptance.** Every candidate and failed/inconclusive
   attempt records data/code/config identities, row coverage, seeds, observed compute and
   metrics in data-gov and the warehouse. Keep a requirement -> test -> evidence table with
   PASS/FAIL/NOT_TESTED. Reserve remains untouched until the complete protocol is reviewed.

## Responsibility

The user need not inspect every test or discover omitted data/model requirements. Engineering
owns the coverage and independent review. No universal correctness guarantee is possible:
green software tests, statistical adequacy and business validity are separate claims, each
requiring its own evidence. Return the business protocol and gaps with exact owners, not a
claim that all requirements already hold. No new user authorization is required to design it.

Independent check this turn: 30 tests passed in 35.61 seconds (instrument runner, calibration
cache, reverify and development closure). The full reported suite was not rerun; production
warehouse content was not freshly queried. This is a scoped review, not comprehensive acceptance.
