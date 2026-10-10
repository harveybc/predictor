# I6-E: origin-anchored resolution experiment

Status: SPECIFIED / NOT_IMPLEMENTED / NOT_RUN. Accepted 2026-10-10.
Follows the active I6-B/I7 contrast; CPU preparation may proceed independently.
No current campaign, target definition, worker or checkpoint is modified.

## Question and locked first experiment

On EURUSD, forecast the unsmoothed 72-hour log return
log(P(t+72h)/P(t)), issuing a forecast at every eligible hourly origin t.
Reconstruct predicted price as P(t)*exp(predicted_return). Report errors and
paired persistence separately in log-return and original price units. Never
compare values across these units. Primary decision metric is annual pooled
price MAE, paired on the same origins; weekly means are secondary summaries.

Freeze the 20 ordered RAW features of the retained I6-D weekly design, including
their availability masks. This deliberately transfers a fixed feature set; it
does not claim these are the optimal selected features for 72 hours. Record
the parent design digest and the feature manifest digest in the new design.
Do not clean targets or perform feature selection on outer validation.

## Eighteen primary neural arms

| Arm | Input spacing | Samples | Oldest-to-current sample span |
| --- | --- | --- | --- |
| C2 | 72 hours | 2 | 72 hours |
| H73 | 1 hour | 73 | 72 hours |
| C4 | 72 hours | 4 | 216 hours |
| H217 | 1 hour | 217 | 216 hours |
| C8 | 72 hours | 8 | 504 hours |
| H505 | 1 hour | 505 | 504 hours |
| C2_HALF | 36 hours | 2 | 36 hours |
| H37 | 1 hour | 37 | 36 hours |
| C4_HALF | 36 hours | 4 | 108 hours |
| H109 | 1 hour | 109 | 108 hours |
| C8_HALF | 36 hours | 8 | 252 hours |
| H253 | 1 hour | 253 | 252 hours |
| C2_QUARTER | 18 hours | 2 | 18 hours |
| H19 | 1 hour | 19 | 18 hours |
| C4_QUARTER | 18 hours | 4 | 54 hours |
| H55 | 1 hour | 55 | 54 hours |
| C8_QUARTER | 18 hours | 8 | 126 hours |
| H127 | 1 hour | 127 | 126 hours |

Every window ends at t; coarse samples are t-j*delta, with delta in {72h,36h,18h},
never a fixed calendar
resampling grid. At t+1h shift every required timestamp by one hour. Spacing
changes only input support, not prediction issuance or the 72-hour target.
The statistical coarse model forecasts one, two or four steps respectively. The
neural head predicts the 72-hour target directly in both cases; it is not an
unannounced recursive rollout or a change of target. This directly tests the user's prior
observation that horizon-adapted one-step modeling improved an ARIMA forecast
on another dataset. It is an empirical hypothesis, not a guaranteed consequence
of renaming the horizon: resampling changes the process and its fitted model.
All three resolutions belong to the primary experiment, crossed with n in {2,4,8}.
Do not infer a pure resolution effect from unequal-context n=2 comparisons.
The primary matched hourly pairs already isolate resolution at fixed context.
Use exact original endpoint values, without smoothing or filtering, by explicit
user decision. Do not introduce averaging, low-pass filters, denoisers or
interpolation into this experiment or its horizon extension. Existing TRAIN-fit
scaling remains allowed and is not temporal smoothing. Preserve sampled extrema;
hourly issuance shifts the sampled endpoints but does not recover all omitted
intrawindow extrema or guarantee absence of aliasing. No automatic filtered
follow-up is scheduled by this plan.

Use a sealed common origin intersection across all eighteen arms and the statistical
controls, requiring all timestamp support and the exact future price. Preserve
availability masks for source-missing values under the parent policy. Never
fill nonexistent market bars or treat 72 rows as 72 hours. Publish excluded
origins by reason and weekly coverage. This exact-support experiment does not
promise a prediction during closures or missing-data periods.

## Shared model and training

R0 only: randomly initialized trainable causal Conv1D per feature, channels 16,
kernel 3, as in the parent CONV branch. Concatenate branches along channels;
preserve time; positional encoding; two causal Transformer blocks, d_model=64,
4 heads, ff_dim=128, dropout=0. Progressive causal Conv1D channel reduction
32 -> 16 -> 8 with kernel 3 and stride 1 at every stage. Keep n temporal steps.
Predict from the final time position with a linear scalar head. No flattened
branch, temporal averaging, artificial sample repetition or time-axis padding
to disguise a shorter window. Ordinary causal convolution boundary padding is
declared and allowed. This is a new resolution-control configuration, not the
unchanged I6-D core or a reuse of incompatible hourly donor weights.

Parameter shapes must match across all eighteen arms; positional encoding must be
nontrainable and support each length. Copy initial weights across arms within
each week. Record actual parameter counts and receptive support. Use the
parent optimizer, scaling, loss and early-stop policy without silently changing
them; save the complete inherited effective config and its digest.

Seed 6102026, one repetition. Full weekly refits from the same initial weights,
preceding four calendar years, max epochs 50, patience 5, batch 256. Early stop
uses the parent TRAIN-internal validation protocol and declared joint train/
validation criterion, restores best weights and never reads the outer week.
Purge training and inner-validation labels by their actual 72-hour support:
all fitted labels must be available before the weekly cutoff; no split overlap.

Evaluate all 52 retained 2024 validation weeks with their actual boundaries;
describe coverage outside those boundaries rather than claiming a full calendar
year if the parent schedule excludes days. No single-week quality conclusion.
TEST stays closed. There are 936 primary week-arm fits, not eighteen total fits.

## Two statistical controls

Use statsmodels ARIMA(1,1,0) without drift on coarse endpoint price history and
ARIMAX(1,1,0) without drift with the same declared feature values/masks. Estimate
parameters weekly on the full four-year history, not two/eight observations.
No automatic order search. Report differencing and full state-history support;
these controls do not have the same finite receptive field as neural arms.

For 72-hour spacing there are 72 UTC-hour phase streams; for 36-hour spacing
there are 36; for 18-hour spacing there are 18. Fit one model per
phase each week; at origin t use its phase stream, update state from only data
available at t with parameters fixed, and forecast one/two/four steps respectively.
That is 6,552 phase fits per control over 52 weeks across all three resolutions;
cost it explicitly, never refit hourly. Do not replicate these fits for each
neural input count: these ARIMA controls do not use n as their history length.

For ARIMAX, use regressors delayed by 72 physical hours on the coarse stream.
Thus regressors for the forecast at t+72h come from t and are known at origin.
At 36-hour resolution the intermediate t+36h regressors come from t-36h.
At 18-hour resolution regressors for t+18h, t+36h, t+54h and t+72h come
from t-54h, t-36h, t-18h and t, respectively.
No realized future exogenous values.
Fit preprocessing inside weekly training only. Record convergence, singular
design and forecast failures; never substitute naive as a successful model.

## Execution, acceptance and automatic continuation

Implement through existing predictor preprocessing/model plugin contracts and
weekly runner helpers; no standalone duplicate orchestration framework. Before
implementation write focused tests for shifted origins, exact-hour gaps,
future perturbation, masks, label purging, phase/state routing, shared weight
initialization, 52-week completeness and paired-naive arithmetic.

The executor must expose plan, cost, run, status, resume and close. Use durable
exclusive cell claims, atomic terminals, validated digests and idempotent OLAP
submission/readback. Resume only unfinished cells, never repeat completed fits.
Status includes expected/completed/failed cells per arm, heartbeat, host/device,
measured updates/time/peak and ETA from measured throughput with uncertainty.

Measure a TRAIN-only cost pilot for C8/H505 and the two phase controls before
dispatch. Retain reusable work where identities match. Gamma: one job only on
5090, never 5070 Ti. Dragon: 4090 independently if host-memory admission fits.
Coordinator: lightweight CPU orchestration; statistical fits on workers with
bounded threads/memory. Production reservation is 1.25 times measured peak
only when it fits; do not claim wall-clock ETA before measuring cost.

Close only with all expected terminals and identical per-horizon populations.
Persist per-week predictions, actuals, naive, MAE/MSE, direction, parameter and
update counts, cost, convergence and identities in the existing OLAP path.
Keep local physical metric backup and one-paragraph phase summaries; Git stores
code, configs, manifests and compact reports, not a large database binary.

Report paired C2-H73, C4-H217, C8-H505, C2_HALF-H37, C4_HALF-H109,
C8_HALF-H253, C2_QUARTER-H19, C4_QUARTER-H55, C8_QUARTER-H127 and all
arms versus naive. Report the annual pooled validation price MAE of every arm.
A best validation arm is exploratory, not confirmatory.
Positive skill permits the existing strategy gate review but is not profit.
Negative skill does not establish theoretical impossibility.

## Extension to the strategy's declared horizons

Read the short/long horizon lists from the authenticated strategy/target
configuration; never guess whether the longest horizon is 120 or 144 hours.
Publish that list and its digest before creating a task. For each h compare
delta=h, delta=h/2 and delta=h/4, crossed with n in {2,4,8}, where spacing is supported
by the existing hourly timestamps. For h=1 only delta=1h is supported; for odd
integer-hour h the half-spacing arm is NOT_APPLICABLE; quarter-spacing requires
h divisible by four. Unsupported spacing is not interpolated data.

To avoid a full annual grid at every horizon, choose (delta,n) within TRAIN
using three chronological inner folds, each a four-week evaluation block with
weekly refits and preceding four-year histories. Derive nonoverlapping block
dates from the retained TRAIN contract, seal dates/support/populations before
fitting, and purge labels across each cutoff. These are distinct temporal
folds, not repeated random-seed fits. Selection objective is pooled paired
price MAE across all inner blocks; ties choose lower measured compute cost,
then fewer samples, then the finer spacing. Record every tried configuration.

Freeze the selected configuration for each horizon before outer validation.
Evaluate that configuration and its matched hourly-context neural control
over all 52 weeks, plus statistical controls and naive; one saved seed. Do not
reselect windows using the outer week or exclude weeks with negative skill.
This extension estimates which windows work among the declared candidates,
not a universally optimal window. Feature IDs remain the fixed transfer set
unless a separate TRAIN-only target-specific selection design is registered.
R1/R2 arms are separate follow-ups; no reuse of incompatible donors. Filtered
inputs are outside this accepted experiment and are not automatic follow-ups.

## Business decision at closure

Produce per-horizon eligibility and a two-function table: early close (short)
and entry/TP/SL (long). Use annual pooled same-row naive skill, not weekly
exclusion. Audit required horizon arrays against the actual strategy plugin
before declaring a reduced horizon subset executable. Only eligible complete
configurations proceed to existing costed strategy evaluation; forecast skill
is not a profit guarantee. If neither an eligible short/long combination nor
an explicit compatible subset exists, retire further optimization of this
tested formulation and prioritize existing SAC/DQN and separately designed
event/barrier/action objectives. Do not claim theoretical impossibility.
