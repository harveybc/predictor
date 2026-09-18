# Requirements-first redesign: stop automatic inherited-period expansion

Priority: latest owner direction. Supersedes permission in T3 to launch the inherited
216-cell adequacy factorial and its larger CPU allowance. Existing evidence is preserved.
This order permits design, code tests and read-only business/data discovery, NOT another
scientific training campaign. Do not launch more cells of the superseded design. Record
the actual dispatch state; do not claim existing processes were stopped by publishing this file.
Follow existing orderly checkpoint/stop procedures if campaign work has already begun;
do not kill shared services or erase attempts.

## Why 41.438459 is not defended

The generator draws period uniformly from [16,128). Reproducibility explains the number,
not scientific adequacy. No business derivation was found for that interval in this front.
Reusing these units to inspect mechanics is legitimate at limited scope; using them to
dictate larger context/model/resource decisions without requirements was not justified.
Do not invent a retrospective rationale or assert that every prior result is invalid.

There is no universally optimal period in samples. P = physical_period / sampling_interval;
its relevance depends on forecast horizon, available history, bandwidth, noise and the
question. Weekly retraining does not by itself imply weekly signal periodicity.

## First deliverable: actual business question and observed scales

Read actual executable forecasting and agent configurations, decision cadence, target horizon,
weekly cutoff/release cycle and existing characterization summaries. Record exact sources,
declared vs executed status, and missing facts. No owner decision is needed to inventory them.
Do not silently infer intended configurations from filenames. Separate forecasting and RL.

Use existing TRAIN-only characterization where adequate; design any missing autocorrelation,
spectral/time-frequency or drift analysis before execution. Report uncertainty, nonstationarity,
sampling gaps and absence of a clear cycle. Do not force a period onto nonperiodic financial
data. No held-out data, full-series dominant frequency or generator truth may select production
windows. A new financial period/window requires this evidence, not the diagnostic number below.

## Optional cheap diagnostic reference: P=16, not a market assumption

This is an explicit engineering convention for a small software/learning sanity fixture,
NOT an estimated financial period or an optimized value. The deliberately chosen requirements
are: unambiguous quarter-cycle horizon at h=4; a two-cycle context within W=32; diagnostic
short/full/two-cycle windows W=4/16/32; and a compact train history with 32 full cycles.
These choices fix P=16, L=512. Choosing those limits is discretionary and must be declared,
not disguised as a theorem. They make the experiment cheap and interpretable rather than
representative of a market. P=8 or P=32 are deterministic scale checks only if a named test
requires them; they are not an automatic factorial or an invitation to pick the best outcome.

Clean generator: x[t]=sin(2*pi*t/16+phi), amplitude 1, offset 0. Phase conditions are declared
explicitly (e.g. 0, pi/2), not searched until passing and not counted as independent statistical
replicates. Known phase/period are oracle metadata only, absent from learner features.
W values count samples; report true elapsed support (W-1)/P. To cover TWO FULL elapsed periods
at unit spacing needs W=33, not 32. Therefore use W=4/17/33 when the requirement means full
elapsed cycles, and describe that distinction consistently throughout tests and design.

First exercise the exact linear recurrence with two lags as an analytic unit test, without
training a neural net. A single clean sine does not require nonlinear abstraction; use it to
catch data/label/fit problems. If the objective is nonlinear model capacity, first define a
different identifiable nonlinear predictive relation and a justified linear-model limitation.
Do not enlarge a neural architecture merely because the fixture is called temporal.

Before any learning pilot, derive shared train/validation/evaluation rows for maximum W and
horizon; keep held-out labels inaccessible to cost pilots. Derive n from required training
cycles, validation/evaluation coverage and consumed support, not a convenient power of two.
The prior 2048-series bank is not modified. No noise stage until clean-task mechanics are
correct; noise strength later follows a declared measurement question and target, not a
default SNR value. Observed-increment and clean prediction remain separate tasks.

## Deliverables and review

Satoshi: finish the useful oracle, shared-boundary and input-integrity fixes already in progress
and their tests; do not throw them away. Then publish (1) business/config evidence table,
(2) dimensionless context/horizon/noise requirements, (3) tiny diagnostic design with each
constant justified and discretionary choices labeled, (4) proposed learning comparison with
cost and stopping rule. Mark work-plan selection expansion SUPERSEDED_PENDING_REDESIGN.

Musashi: review those ML requirements before authorizing a new campaign. No repeated
calibration, 216-cell factorial, GPU, financial scores, RL training or service restart now.
All persisted experimental results still require governance; unit-test fixtures remain tests.
Owner need not grant more permission for these deliverables. No claim that a publication
constitutes runtime dispatch or acknowledgement by another agent.
