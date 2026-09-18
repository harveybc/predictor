# Context and model adequacy: audit and S1-S4 order

Owner accepted continuation and explicitly requested adequate context and temporal models.
This order supersedes automatic expansion to the six remaining operators: adequacy first.
Preserve utildev-v1 and utilinst-v1 outcomes. No claim of failed sinusoid prediction follows
from a non-improving representation compared with raw inputs.

## What the inspected experiment actually did

Inspected `df_utility_harness.features`, `fit_predict`, `label`, `blocks`, the generator
`df_synthetic_bank._clean_univariate`, and the two historical UNIT.json metadata records.
The raw probe was ridge, lambda=1, four raw values; raw_wide had eight. Its target is
observed[t+1] - observed[t], not clean signal or trading profit. Four expanding folds evaluate
the second half of usable rows; 2048 is the series length, not training length per fold.

| selection/replication seed | period in samples | 4/P | 8/P | total 2048/P | clean increment amplitude | white-noise SD |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 41.438459 | 0.096529 | 0.193057 | 49.422687 | 0.222213 | 0.328259 |
| 13 | 78.154908 | 0.051180 | 0.102361 | 26.204368 | 0.069740 | 0.193549 |

Source unit ids: `sinusoid__white__snr10__none__n2048__v1__seed12` and seed13, bank
`synthetic_bank_c128_v1`. Values above are a read-only metadata/arithmetic audit, not new
scores or a fresh raw-array integrity verification. Increment amplitude is 2*A*sin(pi/P).
Noise SD is the generator's declared train-scaled value. Noise in the future observed
sample is unpredictable from past independent noise; this is not a clean-wave extrapolation
task. Do not convert these numbers into a measured optimal prediction-error bound.

Four/eight raw values do NOT cover two periods. Counts W/P above are nominal coverage;
actual timestamp span is (W-1)/P for unit spacing. Transformed branches have additional
operator-specific support and sometimes state; their input count is not total raw history.

Ridge has no hierarchy of nonlinear features, but a fixed-frequency clean sinusoid obeys
x[t+1] = 2*cos(2*pi/P)*x[t] - x[t-1]. Thus two exact lags and the correct coefficient can
predict that special case. Two periods are a useful experimental context scale, not a
universal mathematical requirement. Noise, unknown/multiple/drifting frequencies and
nonlinear patterns make adequacy a separate empirical question. Test it; do not assume it.

CNN/LSTM plugins exist, but are not the ridge utility probe. The current CNN plugin includes
convolutions plus bidirectional LSTM and Bayesian heads; the LSTM plugin is also composite.
Do not label either a plain Conv1D/LSTM without inspecting the instantiated graph. Bidirection
inside a wholly past window is not itself future leakage; assigning intermediate outputs
earlier timestamps, or processing future-containing sequences, can be. Test consumed support.

## S1 - Write tests and a bounded factorial design before training

Separate questions: (a) can the learner predict the raw task, (b) how much history does it need,
(c) does preprocessing/representation improve it, (d) does the gain transfer to prediction/RL
and weekly trading? Existing tests answer only a limited portion of (c).

First declare acceptance tests: analytic clean-sine recurrence; label/row/horizon identity;
train-only scaling; true forward held-out predictions; future perturbation and restart parity;
noise-only control; actual optimizer updates; convergence/underfit diagnostics; complete failures
and costs. Truth may be used as a named synthetic oracle/target, not leaked to a normal learner.

Freeze a DEVELOPMENT pilot over clean and noisy single-frequency signals first. Context factors
must include historical W=4/8 and longer W=128/256, covering over two periods of the inspected
signals. Report W/P and actual consumed timestamp span, plus effective receptive field. More
samples and context are hypotheses, not automatic improvements. Horizons: one step first;
prepare separately one-period and multi-period tasks with explicit direct/recursive semantics.
Do not conflate these with the original one-step result.

Label clean-next-level, clean-increment and observed-increment as DISTINCT tasks, each with its
own baseline and loss denominator. For noisy observed targets, report limitations from future
noise; never demand zero error. Separate fitting a known frequency (oracle diagnostic) from
estimating it from training data. Truth-derived windows are diagnostic metadata, not an allowed
procedure on public or financial evaluation data.

Freeze training lengths, train/inner-validation/test boundaries, seeds, early-stopping rule,
maximum updates, small tuning allowance and meaningful error criteria before running. Use
learning curves rather than arbitrary claims that n=2048 is enough. Additional lengths are
a staged proposal with cost pilot, not an unbounded grid. Preserve the independent final test.

## S2 - Real temporal learners and baselines, not names

Use proven installed framework layers through existing plugin patterns. Add isolated adapters
only where current composite plugins cannot express the intended comparison. Models: ridge
baseline, explicit multilayer causal Conv1D/TCN, explicit LSTM; retain a persistence/zero-change
baseline appropriate to the task and an analytic known-frequency clean oracle. Do not replace
ridge; put its limits in context. Record model graph, layers, parameters, receptive field,
input support, optimizer, observed updates and training/validation curves.

Check CNN receptive field from kernel/dilation/stride and the actual output head; a long input
with a last-position head and tiny receptive field still sees little history. LSTM state must
reset or continue under a declared policy, without carrying validation/test into training.
Test real forward/backward calls and reload predictions. Verify all compared models receive
the same allowed rows and labels; separately report information-history and dimensionality
controls. A larger model or more layers are not evidence of adequate training by themselves.

## S3 - Governed diagnostic execution, no selection claims

After S1/S2 tests, register the pilot in data-gov and execute through bounded children. CPU only,
maximum 7200 aggregate CPU seconds for new diagnostic runs, failures included; cost pilot first,
existing memory limits, no service restart. Independent jobs may use all healthy workers with
their own identities; do not duplicate runs just to occupy machines. If the projected complete
design does not fit, stop after the cost pilot with the frozen design and exact resource need;
do not quietly remove the difficult cells. Full test-suite costs are reported separately.

Persist predictions, observations, clean diagnostic truth with its role, exact training/scoring
ids, errors by horizon, curves, coverage, observed costs and negative/inconclusive outcomes.
Independently recompute losses from arrays and compare terminal/accounting/warehouse contents.
No calibration from ridge/W=4 is transferred to a CNN, LSTM or new context: this pilot is
descriptive model/context adequacy, not ADVANCES, public eligibility or financial promotion.

If the raw task itself fails, diagnose label/scale/context/optimization before interpreting
representation losses. Once raw-task adequacy is evidenced, propose the next representation
comparison with its own null/sensitivity calibration. The existing MAD-positive generator
cannot validate all other operators or every neural model. No repeated tuning until passing.

## S4 - Maintain prediction and RL as two required tracks

Update 12C/09_ADOPCION with a matrix of model x context x task x preprocessing: IMPLEMENTED,
EXECUTED, VERIFIED and NOT_TESTED separately. Include observed shortcomings, not only winners.
Prepare the RL counterpart with the actual environment and intended weekly-retrained agent,
observation histories, rewards, action/cost timing and comparable representation arms. An RL
smoke run is not an RL utility experiment; no new financial/RL training in this CPU diagnostic
order. The weekly-business requirements in the previous handoff remain mandatory.

Return PRE/POST, frozen design, real model evidence, governed results and a concrete diagnosis
of whether context/model/target noise explain the pilot behavior. No owner authorization is
missing. Finish all implementable blocks; report a scientific failure as a result, not a reason
to request permission at every step. Do not reopen old campaigns, reserves, live or broad sweeps.
