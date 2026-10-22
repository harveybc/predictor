# Feature Profiles, Encodings And SAC Input Policy

Owner clarification 2026-10-22. This is a measured-state explanation and input
policy, not a claim of completed extractor selection or a deployed SAC fit.

## Baseline Authority

- Policy inputs: formally selected, identified feature columns only. No agent
  feedback, account/position/reward/SL/TP state concatenated to actor or critic
  observations. Replay actions/rewards remain required SAC learning inputs;
  this does not authorize appending them as market observations.
- The simulator retains true account/position/risk/calendar state and enforces
  the existing business rules. A state-free policy can be partially observable;
  report that limitation, never silently restore auxiliary input slots.
- Fixed per-feature TRAIN mean/std for eligible continuous columns. No rolling
  normalization, clipping, winsorization, outlier deletion, NaN-to-zero,
  smoothing, differencing or nonlinear transform silently added by the adapter.
- Native sin/cos pairs, probabilities and binary indicators keep their declared
  coordinates. Preserve valid extremes. A finite constant feature needs an
  explicit scaler rule; missing/nonfinite observations need source correction
  or a declared support exclusion, not invented values.
- Existing upstream transformed columns are NOT untransformed observations.
  Each selected @diff/@asinh/relative column requires its own explicit recipe
  and audit disposition. Do not rename it as a raw parent or silently replace
  it with one. New optional transforms/encoders require a declared contrast,
  not an unannounced baseline change.
- The approved initial stride2 sampling itself selects observations; it is not
  lossless. No additional unapproved information reduction is implied.

## What The Existing Groups Actually Mean

The selection engine contains 57 source/semantic families, not 57 clusters
learned from numerical signal profiles. They include raw/inventory technicals,
price statistics at several scales, candle geometry, clocks, cross-asset and
daily/COT/factor inputs, activity, and named decomposition/regime families.
Results: 41 retained, 2 interaction rescues, 14 not retained under this evidence.
Consumer candidates: core15, strict125, shortlist173, strict+encodings137.
The original selection counts126/174 remain a different artifact identity.

The catalog has369 rows with ADF/KPSS/skew/kurtosis fields plus one pending row.
Rows include alternate stores and derived columns; this is not369 independent
selected inputs. It also records mean/std/min/max/zero share, monthly coverage,
warmup, missingness, availability, producer and fit identity.

Actual diagnostic rules in retained producer e04b496a:

- ADF p>0.05 AND KPSS p<0.05 proposes DIFF, except known bounded/calendar fields.
- Nonnegative data with |skew|>2 proposes ASINH only if transformed |skew| is
  less than half its original value. See fit-scope defect below.
- Outlier share counts values outside median +/-6*(1.4826*MAD), fitted on
  catalog TRAIN2020-2022; this is a diagnostic, not permission to clip/delete.
- These thresholds propose preprocessing, not groups or an optimal model.
  Contradictory or inconclusive stationarity tests do not prove stationarity.
  A fixed z-score does not require differencing as a mathematical prerequisite.

Examples read from the catalog: CLOSE ADF0.763736/KPSS0.01 -> DIFF proposal;
px.logret_1 ADF0/KPSS0.1/skew0.06479/excess-kurtosis14.9131 -> NONE;
hod_sin -> native bounded calendar encoding. Heavy tails alone do not license
removing real extremes. Diagnostics here use2020-2022, not a replacement for
the principal four-year TRAIN contract.

Selection is a separate task-conditional test. The implemented lower bound
named lo90 is mean(gain)-1.645*sample_std(gain)/sqrt(n); positive bounds retain
evidence from retrained contrasts or grouped block permutation. The coefficient
1.645 is a normal-approximation coefficient; the inherited name is not proof
of calibrated coverage across dependent folds or multiple comparisons. Strict
lists count at least two evidence records; these are not guaranteed independent
replications. No scalar profile threshold currently selects a Conv1D or LSTM.

## Measured Encodings And Candidate Extractors

| Profile/family | Measured representation | Candidate extractor, NOT a measured optimum |
| --- | --- | --- |
| Return/range/candle/momentum series | Selected parent/component columns, with declared upstream transforms | Small causal Conv1D, kernel3; start with4 or8 filters, compare8/16 explicitly |
| Calendar and event clocks | Hour/weekday sin+cos remain; session flags retained; event encoding kept | Identity or pointwise projection preserving the time axis; compare temporal branch only as an explicit alternative |
| Hilbert phase | Return-phase sin/cos subgroup retained | Joint phase-pair branch; do not split the circular coordinates or use a wrapped raw angle |
| Wavelet/strength scales | Wavelet parents+components; selected strength components | Small per-scale causal branches then fusion, with exact producer support retained |
| GMM/HMM regimes | Probabilities/entropy retained with parents; GMM one-hot added in encoded candidate; HMM base kept | Pointwise projection or small temporal Conv1D over probabilities, not silent argmax replacement |
| Slow daily/COT/factor inputs | Selected published values on the admitted grid | Identity/pointwise baseline; test long-memory temporal model only with evidence and explicit scope |

Winning encoding cells, not universal benefits:

| Encoding contrast | Model/task | Mean validation loss gain | Implemented lower bound |
| --- | --- | --- | --- |
| sessions vs hourly sin/cos | temporal/hit | 0.0149622460 | 0.0001903839 |
| GMM one-hot vs base | temporal/hit | 0.0007159511 | 0.0002299686 |
| range rel24 vs base | HGB/hit | 0.0005271490 | 0.0000680981 |
| difference+rel480 vs difference | temporal/hit | 0.0060620904 | 0.0027026013 |

These are selection-probe losses, NOT profit, Sharpe or SAC improvements.
The encoded137 list retains the original cyclic fields/probabilities and adds
12 columns:8 range-relative,1 close-relative and3 GMM one-hot. Session flags
already occur in its base subset; there is no silent removal of hour sin/cos.
Explicit relatives and one-hot are experimental candidate inputs, not default
scaling rules. Upstream history lengths24/480 are feature construction support,
not the policy's downsampled observation count.

## Window Versus Receptive Field

Initial SAC control: all time-series inputs have the SAME13 observed instants
at stride2, oldest-to-current, including t. Missing weekend hours are not ticks.
This spans24 observed hourly intervals, not necessarily24 elapsed wall hours.
No feature-specific window is currently measured as optimal.

Three distinct supports must be logged:

1. The feature producer's support, e.g. trailing wavelet/fractal/relative window
   or a filtered regime state. Some causal filtered states retain recursive
   history rather than a finite hard window.
2. The13 sampled feature instants delivered to the actor/critic encoder.
3. The local receptive field of each convolutional output and the final head.

For sequential convolutions, start R=1,j=1; each layer updates
R <- R+(kernel-1)*dilation*j, j <- j*stride. The current candidate uses kernel3,
stride1/dilation1, one per-feature convolution then one core convolution:
R=5 sampled instants per final temporal location; an additional core gives7.
At input stride2 these are8 or12 observed hourly intervals, respectively.
The head currently flattens ALL temporal outputs, so its total input support
still covers13 instants. Do not claim a5-point local field means the policy
ignores its older8 observations, or that flattening alone discards time order.

Window size is not filter count. To select feature-specific supports later,
measure lag/conditional predictive utility and stability on TRAIN/development,
then compare explicit bounded window/dilation candidates with matched support.
The existing catalog has no completed per-feature ACF/conditional-lag/spectral
profile-to-extractor assignment table. Do not fabricate such assignments or
thresholds to describe the present state.

## Narrow Fit-Scope Finding For Satoshi

Retained layer1 producer sha256
9d613f18dcf9b9226f888a875f7d1d89c4875e16e820a9615c4ab9c59b96cf9d
matches the catalog producer. main computes diagnostics(x[train]) but calls
decide(name,x,diagnostics) with the FULL series. ASINH acceptance then uses
nanmin(x) and skew(arcsinh(x)) on that full series. Thus TRAIN-only diagnostics
do not make the transform selection TRAIN-only. This is selection leakage,
not a centered-filter dependency within each transformed sample.

CPU synthetic probe executed the exact retained decide AST: RNG20261022,
TRAIN10000 positive lognormal(0,1.5) values, suffix100 zeros. Hold diagnostics
fixed (ADF0,KPSS0.1, TRAIN skew17.93660678394807). Original decision ASINH;
changing only the last suffix value to1e200 produces NONE. TRAIN is byte-identical.
This is a finite synthetic dependency witness, not a measured real-data effect.

Minimal correction: choose the transform from TRAIN values only, freeze that
choice, then apply its pointwise/difference recipe to the complete causal
series. Audit every caller (layer1, layer2, catalog), not just the scaler.
Retain a regression in which later values cannot change that decision, then
compare old/new decisions on the actual selected columns to bound the impact.
Only affected generated columns/selection/consumers require recomputation.
Do not silently alter Satoshi's active queue or claim all its results are
invalid based on a synthetic witness alone. Musashi SAC must not adopt an
unreviewed automatic transform under the new input policy.
