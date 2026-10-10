# Predictive information and window diagnostic

Status: PLANNED, NOT_RUN. Accepted 2026-10-10. No new training allocation.

## Placement and scope

This diagnostic follows the active modular architecture contrast and reuses
retained feature profiles, paired predictions and BUSINESS_WEEKLY_WALK_FORWARD
contracts first. It does not replace I6-B/I7, reopen feature selection or launch
literature replications with weekly/quarterly forecast horizons. No current
worker or campaign configuration is changed by this document.

## Hypothesis and assumptions

For price P, define future movement U(t,h) = P(t+h) - P(t) and a past-only
window X(t,W). Persistence predicts U = 0; its MAE is mean(abs(U)) on exactly
the scored origins. That MAE is a movement scale, not proof of irreducible noise.

Under independent zero-mean Gaussian increments of standard deviation sigma,
naive MAE(h) = sigma * sqrt(2*h/pi), and past observations cannot improve the
population-optimal forecast. This is a reference assumption, not an EURUSD fact.

Under jointly Gaussian zero-mean movement and inputs, with constant conditional
variance, predictive mutual information in bits is
I = 0.5 * log2(Var(U) / Var(U|X)). The ideal conditional-mean/median MAE divided
by zero-movement naive MAE is 2**(-I). Outside these assumptions this identity
must not be reported as a measured bound or a universal forecasting law.

The empirical question is whether additional past observations add useful
predictive information at each horizon. W=h and W=2*h are candidate windows,
not a Nyquist guarantee. Information about volatility alone can be useful
without improving a point forecast's MAE.

## Ordered execution

1. Read retained contracts and TRAIN-only profiles; authenticate feature order,
   timestamps, target units, availability, physical lookback and artifact hashes.
   Reuse existing results where identities match; missing metrics remain missing.
2. For each actually declared strategy horizon, prepare unique candidate
   windows {h, 2*h, 24} in hourly observations. Require exact timestamp support;
   never substitute row counts across market closures. Record exclusions.
3. Inspect TRAIN lag dependence and incremental predictive value using blocked
   inner folds. Fit scaling and any tuning only on inner TRAIN. Use time-aware
   null controls; unconditional price-level correlation is not predictive skill.
   Any estimated information score is an estimator diagnostic, not oracle truth.
4. Only after the active architecture contrast closes, cost a small paired
   window ablation using the selected modular architecture and existing runner.
   Hold features, heads, regime, donor policy, seed and early-stop rule fixed.
   Preserve the required four-year rolling training history and weekend refits.
   Cost every changed shape/donor requirement before scheduling it.
5. Evaluate every declared validation week, with identical scored origins across
   windows for each horizon. Report exclusions and coverage alongside skill.
   TEST remains unopened; no strategy evaluation without the existing naive gate.

## Evidence and automation contract

Use one saved seed initially; no duplicate experiment identities. At most three
seeds only when a justified paired stability comparison requires them. Replays
that need no training must reuse retained checkpoints and predictions.

A future executor must support plan/run/status/resume, exclusive claims,
atomic terminals, bounded resources and idempotent warehouse submission. Status
must separate completed/failed/pending work and estimate ETA from measured
workloads; this document does not claim that executor already exists.

Persist per horizon/window/week: row population digest, paired model and naive
MAE/MSE, annual pooled errors, equal-week summaries, skill, dependency diagnostics,
assumptions, actual updates, seed, code/config/input/checkpoint identities and
measured cost. Annual pooled skill is the strategy gate; weekly skill is a
diagnostic, not an automatic per-week exclusion. Keep metrics in the existing
OLAP path, with local backup, and compact phase summaries in the results index.

No result from a finite set of trained models proves that beating persistence
is theoretically impossible. A negative result limits only the tested inputs,
windows, models and protocol.

## Horizon-adapted input resolution contrast

The locked first experiment is
[I6-E](I6E_ORIGIN_ANCHORED_RESOLUTION_WORK_PLAN.md); its exact arms and training
protocol take precedence over the candidate suggestions below.

Accepted extension 2026-10-10; PLANNED, NOT_RUN. On Gamma schedule only the
5090, one admitted workload at a time. Do not schedule training on its 5070 Ti
or assume two GPU devices imply enough host RAM for parallel workloads.

Separate input spacing delta, input sample count n, physical context and forecast
horizon h. Start with n in {2,4,8}. For h=72 hours, delta=36 hours makes h two
coarse forecast steps; n point samples ending at origin span (n-1)*36 hours.
Changing n does not change the forecast horizon. Generalize delta=h/2 only when
it is an integer multiple of the available 1-hour sampling grid. For odd-hour
horizons choose a declared supported spacing/divisor, not invented subhour data.
Hourly data do not establish that h>=2 is predictably beatable.

Retain the original unsmoothed price target P(origin+h), origin price and paired
naive at every arm. Use physical timestamps and the intersection of eligible
origins, excluding missing required bars rather than spanning closures by row
count. The origin price remains available even if transformed inputs are filtered.

Distinguish coarse endpoint selection from causal aggregation/anti-alias filtering.
Endpoint selection is an explicit control, not assumed alias-free. Filtered arms
must use trailing support only, declare lag and pass future-perturbation and
prefix-invariance tests. Never filter the full train/validation series with a
two-sided zero-phase routine. Filter history is part of physical context and
resource accounting. Downsampling may destroy useful paths, extrema and timing.

First compare hourly and coarse inputs at matched physical context and fixed
model family; report residual sample/count/parameter differences. Then compare
small n within the coarse arm. Preserve weekly four-year refits and all annual
validation weeks. No broad resolution search or repetition of completed fits.

Statistical controls: ARIMA on coarse endpoint prices, forecast h/delta steps
where integer, fitted on the full declared training history, not n observations.
AR order and recent input count are distinct from estimation sample size; MA
states can also depend on older history and must be disclosed. A direct ARX
control with n declared lags is preferable when exactly matching predictor
input support. ARIMAX must not receive realized future market covariates: use
only values known at origin (such as calendar variables), separately forecast
exogenous paths, or explicitly lagged direct regressors. Record the chosen
policy; no silent future-exogenous oracle.

Cost a bounded TRAIN-only pilot before assigning these arms. Reduced sequence
length suggests lower cost, not guaranteed skill or a theoretical sufficiency
threshold. No workers were changed or new fits launched by this extension.

## Primary theoretical reference

Guo, Shamai and Verdu, "Mutual Information and Minimum Mean-square Error in
Gaussian Channels": https://arxiv.org/abs/cs/0412108. The forecasting-specific
Gaussian ratio above is a derivation under the stated assumptions, not an
application of the channel theorem without qualification.
