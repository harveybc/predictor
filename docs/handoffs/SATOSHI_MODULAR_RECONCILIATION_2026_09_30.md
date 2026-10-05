# Satoshi: reconcile the architecture and continue independent lanes

> **HISTORICAL, NOT OPERATIONAL.** Superseded on 2026-10-03 by the consolidated
> master plan and SATOSHI_CANONICAL_SELECTION_FIRST_EXECUTION_2026_10_03.md.
> Retained only as audit evidence.

Written for Satoshi (execution), Musashi (review), and the owner.
Observed 2026-09-30 23:55Z through 2026-10-01 00:03Z. This is an additive
correction to SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md, not a new campaign.
Do not restart completed literature cells or suspend independent work.

## Findings requiring action

1. M01 at `8f38bfeb` still contains the monolithic modular_temporal.py,
   branch_steps=12, and branch/core _compress calls. This is not merely an
   ancestry difference: the inspected implementation retains the superseded
   architecture. The owner-corrected package at `da4ce7b4` and master handoff
   `1a53a49e` require full-window branch outputs and residual Conv1D core stages.
2. The M04 default_r0_model_for_donors.json inspected on the execution host
   has window=24, branch_steps=12 and core time_factors=[2,1,1]. M02's retained
   synthetic evidence likewise uses a 12-step fused representation. Those
   artifacts demonstrate their old implementation only, not the current design.
3. M04's retained PILOT_TERMINATED.json says STOPPED_BY_M04_IN_BUILD,
   no_fit_no_update=true, peak 5,801,222,144 bytes under a 6 GiB cap. The report
   describes MemoryHigh pressure and oom_kill=0. It is neither a completed cost
   pilot nor proof that GPU training itself needs that amount. Its timing fields
   also need reconciliation against admission/termination and heartbeat clocks.
4. Direct compute-process queries found no GPU compute processes on the three
   hosts at this inspection. The refreshed STATUS still describes the old M04
   pilot as running in a note and queued work with misparsed names/memory
   (for example an SSH alias treated as a job and memory venv). Do not report
   these as fits.

## Immediate parallel orders

### M01 + M02: integrate once, retain working additions

M01 owns assembly integration. Consume da4ce7b4's package split and architecture,
then port the useful component versions, effective-parameter donor binding,
installed facade, entry points and shared flat/nested grammar onto it. Do not
overwrite those improvements or maintain two independent model engines. M02
ports its early-stop, bounded materialization and atomic resume additions onto
that same integrated revision; M04 consumes M01's parameter grammar.

Required shapes for the hourly default:

- Each configured feature branch: (batch,24,1) -> (batch,24,16).
- Channel concatenation: (batch,24,16 * configured_branch_count).
- Positional encoding immediately after fusion; two complete causal Transformer
  blocks with attention and FFN residual connections.
- Three residual Conv1D stages: time 24 -> 12 -> 6 -> 6, channels 32 -> 16 -> 8.
- No temporal flattening in branches, fusion or core. Forecast head may flatten
  the completed latent. Dense applied to the last axis of a rank-three tensor
  does not itself collapse time and remains valid for Transformer projections.

FeatureSelect is fixed column routing via tf.gather(axis=-1), not learned
feature selection. Document it and use an unambiguous display label in diagrams;
retain serialization compatibility. Do not insert learned gating as a silent
replacement. Any learned selector is a separately declared ablation.

Before additional donor/candidate training, identify queued or active jobs using
the old configuration. Cancel only unstarted obsolete successor requests. For
active old-design work, retain its state and stop through its normal safe
checkpoint boundary when continuation would only produce unusable donors; do
not edit its source/configuration in place or relabel its outputs. Independent
reference/profiling/adapter work continues.

Run installed component, evaluator, pretraining and legacy-compatibility tests
against the integrated commit. Include shape/time-grid behavior, all three
regimes, save/reload equality, upstream donor mismatch, and optimizer mapping
parity. Existing tests of the old engine do not certify the new one. Regenerate
candidate, donor and fused-materialization identities; remeasure disk and memory
because retaining 24 fused steps changes materialization size.

### M04 + M06: finish a real pilot and start the finite queue

In parallel with integration, diagnose the measured build stage with bounded
CPU profiling. Separate graph construction, serialization/manifests, data load,
first optimizer step and steady training cost. Do not shrink scientific inputs
to turn an admission failure into a pass; smaller probes are explicitly
engineering diagnostics and cannot price the full candidate by assumption.

Refresh actual reservations and host state. Prefer the external 5090 for the
corrected pilot, with the verified TensorFlow library recipe and a real device
operation checked inside the child. Use another eligible host if its RAM is the
limiting resource. A memory request must follow measured evidence and preserve
desktop headroom; do not unilaterally raise system ceilings or restart services.
Any installation/privileged operation actually required is named precisely;
that does not pause independent tasks.

Once the corrected complete pilot passes, execute the already-authorized finite
DOIN queue on the declared train/validation population, with independent
checkpoint scoring (not retraining), then paired R1/R2 as real donors complete.
No additional owner "continue" is needed within the existing allocation.
Retain early stopping at every fitting stage and record updated cost/ETA.

M06 must replace stale free-text running/blocked notes with reconciled process,
lease, heartbeat and terminal facts. Parse launcher argument vectors explicitly,
not token-position guesses across ssh wrappers. Refresh the plan revision and
agent completion states. Distinguish idle, queued, building, fitting and failed.

### M03 + M05: continue without waiting for the fit

M03's 4ddcce48 reports 3,455/15,256 profiled inventory rows, including 1,227 new
full-TRAIN profiles. Integrate the declarations now; do not rediscover them.
Keep reused-profile metric gaps visible and clarify the distinct-column
denominator (15,228). Preserve the author's preprocessing in exact literature
reproductions; sentinel cleaning and alternative grouping are separate variants.
Continue train-only grouping/selection work without reading held-out targets.

M05's 329d9a1/80757c4c adapter replay is an engineering result, not a promoted
financial model. Run installed adapter tests against the integrated package and
keep building the existing shadow/paper path. No real-money promotion follows
from a synthetic donor or a forecasting metric alone.

## Retained result and next report

Traffic L96/H96 has three recorded seeds. Retained normalized float32 means:
MSE 0.3751992683, MAE 0.2511426806; published reference values recorded in the
closure are 0.375/0.251; paired persistence MSE 2.7144524181, MAE 1.0772232192.
This inspection read the closure and did not independently replay all three
cells. Preserve that distinction and do not repeat their training.

Within 15 minutes of receipt, return the integrated-source owner and revision,
disposition of old-design requests, actual device/job table, and next executable
candidate. Continue the existing heartbeat <=60 s, atomic STATUS <=5 min and
30-minute report cadence. Each ETA cites measured rate and remaining work, or
names the precise missing measurement. Return tests, engineering pilots and
scientific scores in separate sections. Update the progress diagram from this
same state. No fabricated utilization, completion percentages or performance.
