# I6-DMO: controlled sampling; RAW priority supersedes R2-first dispatch

OWNER CORRECTION, 2026-10-15: the active order is
[RAW_PRIORITY_20261015.md](RAW_PRIORITY_20261015.md). Short/long RAW prediction
and eligible strategy evaluation precede R2. The earlier ordered questions
below explain how R2 was dispatched; they no longer authorize R2 ahead of the
unfinished RAW long control. Musashi executes; no new Satoshi assignment.

Owner accepted this sequence on 2026-10-15. Status: APPROVED_FOR_DEFINITION_AND_IMPLEMENTATION.
The MG2/MG12 campaign is CLOSED / PAIR_NOT_ELIGIBLE and must not be reopened.
This document does not claim the successor runtime is implemented or running.
Existing baseline receipts and percentages remain unchanged.

## 1. Ordered questions

1. Does a short hourly-input recipe improve on downsampling when both are
   fitted and evaluated on the same population?
2. On the selected recipe, do compatible pretrained branches, fine-tuned in
   R2, improve on the matched R0 reference?
3. Does the promising recipe also help the six long heads? Transfer is a new
   measurement, not an assumption that short and long need the same context.
4. What is the incremental benefit/cost of quarterly versus static retraining?
   Monthly and weekly follow only as separately defined cost/benefit studies.

Do not start a large DOIN search, RL, NEAT, core pretraining or a new feature
selection campaign as an implicit substitute for these questions. Actual
strategy use remains downstream of an eligible short/long pair. Its masks and
the source-provenance gates continue to govern TEST.

## 2. Next executable scope: short MG1 versus MG2

Create TWO new, independently sealed R0 treatment identities:

| Item | MG1 control | MG2 matched reference |
|---|---|---|
| Real observations, t included | 25 | 13 |
| Slot spacing | 1 expected open-market hour | 2 expected open-market hours |
| Context span | 24 open-market hours | 24 open-market hours |
| Issuance | each supported observed hourly t | same |
| Exact elapsed output horizons | 2,3,4,5,6,12 h | same |
| Features | same retained 19 RAW features | same |
| Branch/core | causal Conv1D 4; core 8 then 4, k=3, stride=1 | same |
| Heads | six independent Flatten -> Dense(8) -> Dense(1) | same |
| Time inputs | actual UTC offsets + adjacent elapsed/closure flags | same |
| Initialization | R0, seed 6102026 | same |
| Cadence | static | static |

The existing EXPERIMENT.short_hourly.json describes 25 observations in ELAPSED
hours and is not this market-time MG1 control. Do not relabel it, substitute it,
silently extend its accepted schema, or overwrite its seal. Generate new
market-time designs. No smoothing, interpolation or next-observed-row targets.
Keep exact t+h and the audited per-output business masks.

### Common population, fitted before either scientific training

For FIT, INNER and full DEVELOPMENT-2024 separately, freeze the intersection
of valid (origin row_id, horizon) pairs under BOTH treatments. Retain only
origins with at least one common valid head. Both trainers, checkpoint
monitors, annual scorers and weekly diagnostics must consume those masks.
Freeze the source/design/calendar identities, ordered row IDs, per-head
counts, entry/management masks and exclusions BEFORE either fit.

Use TRAIN 2020-2023 with the chronological last eight weeks as INNER and
label-maturity purging. No TEST data, metadata scan or TEST worker. FIT-only
normalization is shared: calculate feature statistics on the union of MG1/MG2
input-source rows for the common FIT origins, save the exact row digest and
use those same statistics in both arms. Neither INNER nor DEVELOPMENT can
change those statistics. Target loss scales are the same FIT-only naive per
horizon on the common FIT mask. Unsupported heads are typed, never filled.

The completed MG2 baseline was trained on its own larger population. It can
provide descriptive subset scores, but CANNOT replace the newly matched MG2
fit. These two new fits answer a new controlled question, not a seed repeat.
No other completed baseline is retrained or relabelled.

### Capacity disclosure and decision rule

Flatten makes the MG1 heads larger: (25-13)*4*8*6 = 2304 additional trainable
parameters. Record full per-component parameter counts. This is a comparison
of practical input/readout recipes, not proof that additional signal alone
caused any gain. A capacity-matched readout contrast is future work only if
needed; do not silently change the head or add a third arm here.

Retain the baseline AdamW recipe (lr .001, weight decay .0001, batch 256,
max epochs 200, patience 10, min_delta 1e-8, restore best). Stop on the mean
of end-of-epoch FIT and INNER at the SAME weights, never DEVELOPMENT.

Primary descriptive selection score is the equal-head mean of annual pooled
relative MAE across the six heads on the frozen common DEVELOPMENT rows.
Every head still reports its own MAE, naive and relative error; an aggregate
does not make a failed head trading-eligible. Choose the lower score for the
next representation contrast; within 1e-8 retain cheaper MG2. If any required
head or arm fails technically, do not silently select from incomplete heads:
close INCOMPLETE_COMPARISON with the missing evidence. One seed establishes
this recipe decision, not statistical superiority or an independent holdout.

## 3. Subsequent representations, long family and cadence

R2 uses the selected RAW input recipe, not reconstructed/denoised target or
replacement input series. Pretrain each feature encoder with matching support,
calendar/time inputs, scaler and split; import compatible encoder weights and
keep them trainable. Reuse the exact matched R0 receipt if every training and
evaluation identity component still agrees. No hourly/lagged donor is accepted
merely because its tensor shape matches. A new donor/recipe requires a new
identity. Freeze its reconstruction and predictive evaluation before fitting.

Keep core pretraining downstream of branch-pretraining validation, and never
credit R2 gains to sampling if both changed. R2 pretraining loss, fit/inner
diagnostics and compute costs are separate OLAP facts; reconstruction alone
does not prove predictive utility or noise removal.

The long-family design retains exact elapsed heads 18,24,36,48,60,72 h and
requires a fresh support/memory check. Do not assume the short 24-open-hour
context is suitable for 72 h. Design the long contrast after short evidence,
without TEST and without a broad horizon sweep. Quarterly cadence is a later
separate comparison on matched validation support, with mature labels and
four-year rolling training; no guaranteed monotonic benefit is assumed.

## 4. Automation contract and resources

Implement a finite contrast runner reusing the existing queue, governed worker,
terminal verifier, outbox, publication readback and physical mirrors. It must
advance: design/support seal -> engineering/review -> fit/report both arms ->
paired comparison -> publish/readback -> mirror -> CLOSE. A scientific negative
is a completed result. A delivery retry never causes another fit. Carry out the
CPU eligibility/strategy/auto-close fixes in the recovery handoff independently.

Prefer the 5090; only one heavy GPU process on its host, no parallel 5070 Ti.
Independent arms may use the 4090 after fresh host/slice admission; do not
duplicate a fit to populate a GPU. Coordinator: bounded CPU orchestration,
warehouse and light tests only. Reserve memory for the desktop. Measure the
first full fit; subsequent reservation is 1.25 times its full-cgroup peak if
admissible, never trim that safety margin to pass a gate. A 25-slot model has
no measured peak yet: do not call a 13-slot peak its measured requirement.

Live SQLite/WAL, locks, queue, private configs and worktrees stay on ext4.
Archive large immutable authenticated artifacts under EXPERIMENT_STORAGE_ROOT,
with mount/writability/free-space checks and digest verification. Git holds
code, small configs, summaries and hashes, not arrays/weights/database binaries.

Status at experiment/arm/unit levels must include state, count/denominator,
actual live child, progress time, failure cause and measured/unknown ETA. No
agent polls logs. Services resume after power loss and adopt verified terminals.
Each closure publishes all metrics, exclusions, dispositions and costs to the
existing OLAP with exact readback, two physical mirrors, a short indexed numeric
result paragraph, and a refreshed M1-M8 plot without invented percentages.

## 5. Requirement-to-test design (not execution evidence)

| ID | Mandatory observable acceptance |
|---|---|
| SC01 | MG1/MG2 share the exact common masks for FIT/INNER/DEVELOPMENT; reject a changed row, head or calendar digest |
| SC02 | Last input is t; perturb >t changes neither input; exact t+h changes only its label |
| SC03 | Weekend/DST slots use expected open hours and actual offsets; missing open-session bars remain exclusions |
| SC04 | Shared scalers and naive loss scales are FIT-only; perturb INNER/DEVELOPMENT and verify identical fitted statistics |
| SC05 | Model and naive for each horizon use the same rows, mask and original-price units; manual pooled replay agrees |
| SC06 | Both real fit/report paths consume the frozen common population; reject cached full-population MG2 as a matched fit |
| SC07 | Record the 2304 head-parameter difference; prohibit an unreported capacity change |
| SC08 | Negative and positive fixture dispositions close automatically with publication/readback and mirror proofs; no OHLC required for a negative pair decision |
| SC09 | Crash/restart, duplicate claims, unavailable host and outbox retry never duplicate a completed fit |
| SC10 | No TEST unit exists; changed scientific design/source/donor cannot reuse an old identity |
| SC11 | CPU and bounded GPU engineering replay preserve serialization, origin support and declared numerical policy |
| SC12 | Operator can request status at every level without launching/altering a fit; failed/stale children cannot appear RUNNING indefinitely |

Before release retain PRE/POST, exact code pins, traceability and one independent
bounded review of the actual fit/report adapters, not injected mock arrays.
No additional chat approval between stages once these machine-verifiable gates
pass. Until implemented/reviewed, label the successor PREPARING, never RUNNING.
