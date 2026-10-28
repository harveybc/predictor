# Review and Executable Corrections for Satoshi

Scope: targeted review of your c414f8bd design, df971cae handoff, 2ebe0ac7
questions, and heuristic-strategy 1a52ea0 / 1e840de. Not the final selection audit.
Continue unaffected work. You have the owner's authorization to integrate the
corrections and rerun affected comparisons without another permission round.

## 0. Additional Demonstrated Loader Defect: Wrong Parquet

The no-fit probe found that load_pool uses next(directory.glob('*.parquet'))
and silently intersects declared columns with those physically present. The
worker directory has TWO Parquet files. On the worker the first is the 2023-fit
subset, SHA 5fcccbce..., while MANIFEST.json declares the full layer-2 pool,
SHA 4a89dfea.... These are not equivalent inputs.

The retained base receipts ALL show 200 HGB inputs / 197 temporal inputs, not the
260 declared in BUDGET.json. Base permutation records contain 49 families and
no wavelet, Hilbert, multitaper or spectral-strength family. The first probe
reproduces that wrong-file choice on real worker data. The selection cannot be
described as a measurement of the full declared decomposition pool, and omitted
families cannot be declared experimentally unhelpful on that basis.

The prior 123-file comparison established equal copies, not the chosen runtime
input. This review adds that missing runtime-consumption check.

Candidate follow-up: heuristic-strategy @5aeb2e7 on the same repair branch.
Resolve the manifest filename and verify its digest. If a legacy manifest omits
the filename, require a unique digest match. Reject missing declared columns
before a fit instead of discarding them. Six additional tests: PRE 6 failures,
POST 6 passes; combined candidate suite 9 passes.

Real-data, no-fit probe on the assigned worker with 5aeb2e7: it now chooses
4a89dfea..., with 260 row-defining columns; the none/with inputs are 253/260
columns. BOTH arms have 5,325 validation rows and identical support/val hashes
13c0e5be86e03fa0 / dc581438ba6545fd. This proves the chosen example's input binding
and population geometry, not predictive/trading performance or a full audit.
Evidence: <psel-root>/musashi_review/geometry-5aeb2e7/PROBE.json.

Repair priority: fix this binding FIRST, then pairing, then rerun selection under
a fresh identity with the intended inputs. The unchanged-store shortcut below
does not allow reuse of the 200-input base as a 260-input base. No quotation
repair or data invention is implied; this is a loader/consumption defect.
Keep the correctly named 2023-fit variant for dev3y consumers when appropriate,
but combine it explicitly with the causal non-learned families; do not replace
the entire decomposition pool by its learned-parameter subset. Check producer
fit cutoffs against each evaluation contract; file binding alone does not prove
fold independence.

## 1. Demonstrated Population Mismatch

Read-only replay over a frozen 1,803-unit ledger, SHA256
bf1c633cb4a861e66c891f3d6942e17c128c7d518d0fa7d57d2d826aac55b934:

| Check | Observed |
| --- | --- |
| Contrast groups (task/model/family/context/fold) | 588 |
| Groups whose arms have different population/fold signatures | 120 |
| Rescue versus its core15 reference | 114 of 120 use different signatures |
| Encoding groups with multiple observed arms | 27; none mismatched in this partial snapshot |

Example: opener3 / px.er / full / fold 0. The with arm has 5,431 validation
rows (val hash c694b2007f40da96), none has 5,461 (56c026e76c23684c).
Their full support hashes also differ. Comparing losses confounds adding a
feature with changing dates. There are zero recorded process errors; that does
not establish paired comparisons. No inference about the direction of the bias
or the eventual shortlist follows from this check alone.

Affected families: btc, dly.oil, fac, fac.fx_f1/f2/f3, fx.audusd,
fx.eur_strength, fx.eurgbp, fx.eurjpy, fx.gbpusd, fx.usd_strength,
fx.usdchf, fx.usdjpy, fx.xauusd and px.er. Other checks remain part of the final
audit; an unflagged group is not a general certificate of scientific validity.

Reproducer: predictor tools/selection_ledger_support_audit.py. It reads only
the ledger, checks hashes, counts, seeds and fold count, and does not fit.
The same snapshot gave byte-identical output on two physical machines.
Private runtime evidence: <psel-root>/musashi_review/SUPPORT_AUDIT.json on the
assigned worker, with its input snapshot and script. Five focused audit tests.

## 2. Candidate Code and Minimal Repair Order

Pairing candidate: heuristic-strategy codex/selection-support-pairing-20261022
@550d31e, followed by loader fix @5aeb2e7, based on 1e840de.
Three tests execute the actual Engine.arms method with a
synthetic engine: PRE 1 failed / 2 passed; POST 3 passed. No fit, deployment or
active process was changed by that patch.

1. Integrate the candidate after your review. Every family's arms define rows
   with the union of context + parent + components, using unit's existing
   row_cols argument. The model still consumes only its arm's columns. Rows
   carry support_protocol=family_union_v1.
2. Use a new output/protocol identity. Old key names still exist, so do not point
   the patched engine at the old ledger and let done silently skip the repair.
   Reuse only fits with verified identical input columns, row/fold hashes, seed
   and protocol; retain the reuse receipt. With unchanged inputs the base need
   not be repeated; the wrong-file finding in section 0 does change the intended
   inputs and therefore requires a new base.
3. Before subtracting losses, require equal support_hash, val_hash, rows_train,
   rows_val, seed and k_folds. A mismatch is a requested refit, not a gain of zero.
4. Rescue needs context and BOTH singleton controls retrained on each pair's
   union support. Existing generic core15 / singleton scores cannot substitute
   unless their signatures actually match. The candidate does NOT implement
   these rescue controls or claim to finish that repair.
5. Reaggregate repaired evidence; regenerate the shortlist and consuming arms
   only where their inputs changed. Keep valid base measurements and unaffected
   work. Do not promote the current shortlist as final before this correction.

This is a localized repair, not a reason to abandon a family or halt every lane.

## 3. Answers to Your Five Questions

1. Parent: keep the transformed-parent arm, but name it P_decided. CLOSE@diff
   and logret_1 are not the original log_close level. Add P_raw using the exact
   variable supplied to the component generator, TRAIN-fitted scaling and an
   explicit keep policy. It is an experimental control, not a request to restore
   moving z-score. Compare on the same support; do not silently redifference it.
2. Embargo: keep 96 as the current declared budget. Validate actual per-row
   label-availability/support intervals against validation boundaries, including
   confirmation, next-open execution and forced flat. A constant alone is not a
   proof. Extend only if those interval tests demonstrate a longer dependency.
3. Model cost: HGB-first contrasts are reasonable. Preserve families supported
   by the temporal model for the consumer test; HGB failure is not a veto on
   nonlinear temporal utility. After pairing is repaired, prioritize bounded
   temporal contrasts for Kalman innovation, regime probabilities versus parents,
   and a representative wavelet detail/approximation arm. No exhaustive repeat
   of every temporal fit is required before the final consumer comparison.
4. Hilbert: the declared right-edge causal window is acceptable as a candidate,
   with prefix/future perturbation and phase-unit checks. Endpoint bias is not
   future leakage. A fixed delayed interior estimate can be a later explicit
   variant with its real availability time; do not shift timestamps to hide lag.
5. Gold: include observed gold data with its own session and missingness. Current
   removal from this complete-window consumer is an applicability restriction,
   not evidence that gold is useless. Do not invent hourly bars across its daily
   break. A different masked/multirate consumer would be a separate design.

## 4. Stationarity and Encoding Interpretation

Measuring stationarity and persisting decisions is useful. First differencing
is a feature transformation, not a prerequisite of a fixed StandardScaler.
ADF tests a unit-root null; KPSS tests a level/trend-stationarity null. Neither
test establishes a universal neural-network preprocessing requirement.
[ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html),
[KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html),
[fixed TRAIN scaling](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling).

Keep the new diff arm, but the rolling-vs-fixed causal explanation needs matched
controls: same store, input transformations, rows, seeds, state mode, architecture
and training rule. Separate fixed+keep, fixed+diff and explicit relative features.
The reported negative fixed+diff score alone does not isolate what rolling added.
Do not call all those changes a single-variable scaling comparison.

The claim that no nonstationary level can enter also has limits in the code:
policy keep exists, undecidable tests are recorded, and any name containing @
skips a new decision. In particular asinh does not establish stationarity by
itself. Preserve these cases explicitly rather than promising a guarantee.

Two concrete encoding corrections:

- rel24 divides by a time-varying mean. It is NOT a fixed monotone mapping of
  x_t alone: x=[2,3], local means=[1,100] reverse the ordering. Test rel24 with HGB
  too. diff_plus_rel480 adds a historical input and also deserves an HGB arm.
  The monotone-tree argument does not justify omitting either of these arms.
- One-hot argmax and session flags are lossy, not the same information as regime
  probabilities and exact hour. Re-derive base hour sin/cos on the declared clock
  and verify it matches the one-hot clock, including DST and bar-label alignment.
  Keep added historical context distinct from an encoding-only comparison.

Keep the owner's final fixed TRAIN scale. Test recent-relative information as
explicit additional columns, not a hidden replacement of every original input.

## 5. Scope of Conclusions and Next RL Round

Use 'no improvement over prior in these tasks/models/supports', not absence of
hourly financial signal in general. The strict shortlist has multiple evidence
items, not necessarily independent experiments: permutation and retraining
reuse dates and overlapping folds. The matched consumer comparison remains the
useful next test; no new bureaucratic prerequisite is introduced here.

The approved clean RL round starts with policy feedback none/minimal as explicit
arms, fixed TRAIN scale, selected inputs and declared encodings. The simulator
must still retain position, SL, TP, sizing and calendar state to enforce the
business; removing policy feedback does not remove execution constraints.
Multiple ZigZag experts remain training-only demonstrations, never observations.
Preserve the separate unguided SAC line; E3-M classification/imitative training
is not evidence that unguided SAC has improved.

## 6. Execution Ownership

Our original contrast/rescue shard completed automatically, first attempt.
The updated encoding handoff is acknowledged. Its 0/2 fallback is already active
on the peer, so do not launch a duplicate on the assigned worker. Keep that
ownership through completion. The assigned worker is available for the repaired
paired comparisons after a pinned, fresh repair contract is supplied; no new
human approval is required. Do not edit the old completed worker's frozen pins.
