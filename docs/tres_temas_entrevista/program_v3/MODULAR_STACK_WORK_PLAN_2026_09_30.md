# Modular temporal stack: implementation and experimental plan

Owner instruction: 2026-09-30. This subplan belongs to the master work plan and
runs concurrently with reference reproduction, profiling and product work.

## Goal and boundaries

Build one reusable temporal representation for forecasting and future RL heads:
typed feature selection -> independent branch encoders -> common time grid ->
fusion -> positional encoding -> full Transformer blocks -> progressive learned
compression -> task head. Support random, frozen-pretrained and fine-tuned
initialization for every branch. Core pretraining is a later, separate experiment
after architecture and branch regime selection. DOIN searches
the declared candidate space using validation objectives. predictor owns offline
forecast fitting, feature-eng owns descriptive feature profiles, feature-extractor
owns reusable representation producers, and agent-multi/gym-fx own policy fitting.

The new assembly is a usable implementation, not evidence that it beats the
published reference. Existing TimeFilter cells keep their exact author recipe.
No scientific conclusion is based on a small plumbing fixture.

Public reuse entry point: [feature_selector](https://github.com/harveybc/feature_selector).
Its synthetic CPU example and portability guide document existing selection
engines. The universal selection-plugin API and a second real nonfinancial
adapter are follow-up packaging work after I5; they do not add a prerequisite
to the running feature-selection campaign. Scientific engines retain their
current repository owners.

## Requirements and acceptance matrix

| ID | Requirement | Observable acceptance |
|---|---|---|
| MS01 | Configurable branch, fusion, core and head plugins | Named built-ins and external entry points resolve deterministically; unknown names fail. |
| MS02 | At least 24 hours of context | Physical sampling period plus window length checked; timestamps/gaps checked by data preparation. |
| MS03 | Preserve time through fusion and core | Rank-three tensors, equal right-edge time grid, no flatten/pooling across all time before task head. |
| MS04 | Different branch architectures | Per-branch plugin and parameters; no constraint that all branches use one type. |
| MS05 | Full Transformer core | Positional encoding, multihead attention, FFN, residuals and normalization; configurable blocks/heads/width/dropout. |
| MS06 | Learned progressive bottleneck | Three or four configurable stages, every input position participates, no silent trimming or fabricated observations. |
| MS07 | Independent R0/R1/R2 | R0 uses random weights, R1 verified donor and frozen weights, R2 same donor initialization and trainable weights; check actual updates. |
| MS08 | Fail on explicit bad donors | Missing file, digest mismatch, feature order, input/time shape or upstream core identity mismatch rejected before fit. |
| MS09 | Early stopping in each training stage | Validation monitor, patience/min_delta/max_epochs, selected checkpoint restore, observed update count and stop reason. |
| MS10 | Branch pretraining | Train-only AE, separate internal validation, decoder reconstruction, encoder export and reload parity. |
| MS11 | Core pretraining | Fixed branch donors -> fused materialized train representations -> core AE; producer identities carried to core donor. |
| MS12 | Fair forecast metrics | Same rows/targets/scaler/horizons for model and persistence; MAE/MSE, skill, per-horizon and aggregate values. |
| MS13 | DOIN objective | Candidate and data digests, explicit validation metric/direction, retained weights and replayable result. |
| MS14 | Profiling coverage | One inventory row per dataset/feature and explicit measured/missing/excluded status; no blanket coverage claim. |
| MS15 | Representation for RL | Encoder exports rank-three bottleneck; policy adapter and its reward/early-stop validation are separate owned integrations. |
| MS16 | Strong per-feature Dense controls after final selection | Historical flattened-window ANN is identified and replayed faithfully or marked NOT_REPRODUCIBLE; a separately named causal-window Dense branch is compared with Conv1D under the same temporal fusion/core/head, rows and budget. Neither equates dense units with timestamps. |

## Default shapes and contracts

Default hourly input: `(batch, 24, F)`; one branch per feature initially. The
sampling period is explicit, so 24 minute observations do not count as 24 hours.
Each default branch uses causal Conv1D and produces `(batch, 24, branch_width)`.
It preserves every input time step. Every branch output represents the same
input interval and right-edge grid. Concatenation is on channels, producing
`(batch, 24, sum(branch_width))`.

The core begins with positional encoding after fusion, projects each time step
to width 64, uses two full causal Transformer blocks with four heads, then uses
three residual Conv1D stages to reach `(batch, 6, 8)`. Default time factors are
`[2, 2, 1]`, with channel widths `[32, 16, 8]`. Widths, time factors, depth and
attention heads are candidate parameters. A four-stage schedule is also
permitted if dimensions remain compatible. The default is a starting candidate,
not a discovered optimum.

Branches do not reduce time. In the core, each strided Conv1D stage consumes
complete adjacent windows with valid padding and a matched residual projection.
Temporal factors must divide the current length exactly; incompatible
configurations fail. This maps every sample to a right-edge output without
dropping a tail. Downsampling is inherently potentially lossy; the residual
path and learned filters do not guarantee information preservation. Measure
reconstruction and downstream skill at each bottleneck.

The four feature channels shown in the Keras example diagram are illustrative.
The model uses each feature/group passed in its configuration; no inventory-wide
four-branch limit is implied. Large inventories require explicit feature
selection or grouping with coverage and exclusions reported.

External plugins must declare input/output contracts and preserve their grid.
Equal output shapes alone do not establish aligned times. Inputs with different
frequencies require a separate causal alignment transform before assembly.
The historical per-feature ANN is a deliberately **non-temporal control** and
is exempt from MS03 only as a separate whole-model comparator. Its branches
flatten each feature's input window, apply their own Dense layers and concatenate
learned vectors. Matching the length of a temporal axis never turns those units
into chronological positions; a `Reshape` alone cannot make it an MS03 plugin.

## Feature characterization, grouping and selection

Approved implementation sequence and causal/representation study design:
[progressive selection subplan](FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md).
Its PS0-PS7 increments and FS01-FS20 criteria extend this section. Broad basic
profiling continues while ready batches enter reversible prioritization and
parallel representation/causal studies. Neither VAE nor successful reconstruction
is a universal eligibility requirement. New objectives do not retroactively
change existing AE experiments or their donor identities.

Profiles are fitted exclusively on the designated training period. Each report
binds source bytes, columns, time interval, sampling frequency, transforms and
profile settings. The coverage index distinguishes discovered files from
governed resources and measured columns from queued or excluded columns.

Record missingness, constants, robust scale and tails, volatility, trend,
autocorrelation by declared lag, dominant spectral periods and spectral entropy.
Stationarity diagnostics include ADF and KPSS with hypotheses, lag settings,
finite-sample limitations and failure states. Stationarity and seasonality are
separate axes. STL strength or Hilbert-derived quantities are optional diagnostic
transforms; any feature fed to a model must pass its own point-in-time test.

Start with one selected feature per branch. This makes channel ownership and
ablations explicit. Then compare: domain groups; clustering of standardized
training-only profiles; and groups based on causal lag dependence/redundancy.
Choose clustering count and thresholds using inner validation. ADF, volatility
or an oscillator label cannot by themselves prove that Conv1D beats LSTM or a
Transformer; they generate candidate families to test under equal budgets.

Selection sequence: reject unavailable/noncausal inputs; report constant and
excess-missing columns; evaluate redundant groups and feature ablations on
inner chronological validation; freeze a candidate before outer validation/test.
No importance, grouping or target-based selection is fitted on the test period.
Keep an all-admissible-feature control and the reasons for every exclusion.

## Pretraining and training sequence

1. Freeze split/time/target/scaler contracts. Partition train internally for AE
   selection with purging appropriate to window/target supports.
2. For the existing reconstructive control, train one branch AE per feature/group,
   each with its own optimizer, loss and early stopping. Export selected encoder
   plus manifest and reload check. Other objectives follow the progressive
   subplan with their own validation criterion; no decoder is required for a
   contrastive or latent-prediction encoder. Keep architecture and objective
   separate in configs and evidence.
3. Fit forecasting heads under branch regimes. Baseline R0 has no donor;
   R1 freezes the specified donor; R2 starts from the same donor and fine-tunes.
4. Select the architecture and branch regime on the declared validation support.
   Fix one identical branch prefix for the H-CORE experiment.
5. Freeze that prefix during H-CORE materialization; encode train and internal
   validation in bounded batches and fuse with the downstream plugin/grid.
6. Train the core AE control on fused sequences, early stop on its own internal
   validation and export it with upstream branch/fusion identities.
7. Test transfer of that core in a separate random/frozen/fine-tuned comparison.
   Do not call these arms a redefinition of the branch R0/R1/R2 experiment.
8. Restore selected weights and replay predictions after serialization. Persist
   actual parameter counts, updates, stop reason, monitor values and identities.
9. Evaluate on fixed validation rows with paired persistence. Select candidates
   by the declared objective across paired seeds, not their best seed.
10. Confirm frozen finalists under the declared evaluation mode. In
   `LITERATURE_STATIC`, fit and score exactly as the paper. In
   `BUSINESS_WEEKLY_WALK_FORWARD`, traverse the untouched test year exactly once
   with the frozen update procedure, producing one point-in-time checkpoint per
   eligible week and never selecting from test metrics. Export eligible
   candidates to the LTS paper adapter only after the applicable result.

AdamW with MAE, Huber and MSE are explicit candidate choices. Huber delta is in
the target's declared scale; tune on training/inner-validation only. The metric
used for comparison need not be the training loss. Every candidate has finite
epoch/update/wall limits as well as early stopping. A budget stop is recorded
separately from convergence or a patience stop.

RL uses held-out evaluation episodes, costs and the existing policy objective,
with episode/checkpoint cadence, patience and best-policy restore. Forecasting
MAE does not become an RL stopping metric. Implement this in agent-multi/gym-fx;
the predictor encoder export alone does not implement RL training.

## Experiments and concurrent execution

GPU priority after already-running cells: cost pilot of the full modular model,
then a finite optimization batch of the most promising validation-eligible
configuration. Continue candidates automatically while independent CPU workers
profile, verify metrics and prepare adapters. Benchmark tasks use CPU where
feasible; a faithful reference that genuinely requires GPU gets an explicit slot
without monopolizing the optimization lane. The external 5090 remains preferred
when host RAM and thermal admission allow it.

Run the following matched ablations, with equal search budgets and paired seeds:
published reference; modular R0; branch-pretrained R1/R2; then, after selecting
the prefix, core transfer arms; one-feature vs grouped branches; selected vs all admissible
features; bottleneck width/time/depth; MAE vs tuned Huber. Report configuration
and optimization cost, including AE cost, with every model metric.

**I6-D, after the I5 final feature manifest, alongside I6-B:** run two distinct
Dense comparisons. First, recover the historical ANN's actual source revision,
effective configuration, inputs, preprocessing and checkpoint/result identities.
Replay it on its original feature set and protocol. A refit on the finally
selected features is a new whole-model control, never a thesis replication. If
the original recipe cannot be authenticated, mark the historical replay
NOT_REPRODUCIBLE rather than claiming a thesis replication. Second, implement
a new time-aligned
causal-window Dense branch: at each output time it reads only its declared
past receptive field, shares weights over time and emits a rank-three sequence.
Swap only this branch family against Conv1D while keeping fusion, positional
encoding, core and heads identical. This second arm tests branch choice but is
not the historical ANN. Do not silently route flattened Dense vectors into a
temporal core or compare unlike downstream architectures as a branch-only test.

The new controls use the same final feature IDs/order, physical lookback, target
and horizon definitions, availability masks, normalization fitted on TRAIN,
BUSINESS_WEEKLY_WALK_FORWARD weeks, same-row target naive, seed, early-stop
rule and finite search budget. Match parameter count and measured compute as
closely as feasible; report residual differences and pretraining cost rather
than hiding them. Start with one seed; use at most three if a paired result
needs stability evidence. Keep LITERATURE_STATIC reproductions untouched.
Rank by held-out weekly target skill and cost; strategy tests remain behind the
strict same-row naive gate. Dense winning is an accepted scientific outcome.

DOIN optimizes the declared validation objective. Candidate validity includes
time divisibility, width/head compatibility, donor identity, parameter bounds
and resource limits. A dry-run contract is not an optimization result. Keep the
incumbent, candidate history and independent verification result in the existing
warehouse path when that adapter is exercised; local smoke metrics remain local.

## Predictive information diagnostic

After the active architecture contrast, follow
[Predictive Information and Window Diagnostic](PREDICTIVE_INFORMATION_WINDOW_DIAGNOSTIC.md).
Reuse retained TRAIN diagnostics first; no new fit is launched by this addition.
It tests horizon/window information saturation, not a universal Nyquist bound.

**I6-E:** the first concrete resolution experiment is specified in
[Origin-Anchored Resolution Work Plan](I6E_ORIGIN_ANCHORED_RESOLUTION_WORK_PLAN.md).
It compares hourly, 72-hour, 36-hour and 18-hour inputs for hourly-issued 72-hour predictions,
with a shared modular R0 model, weekly refits and statistical controls. The
extension selects spacing and input count within TRAIN for each authenticated
strategy horizon, then evaluates frozen choices over all validation weeks. Prepare
on CPU alongside existing work; GPU fits follow the active I6-B/I7 contrast.

## Test order and release evidence

Write behavior tests before implementation: shape/time coverage; prefix
invariance for causal branches; future perturbation; frozen vs trainable weight
updates; same-donor initialization; full-model save/load; corrupted donor and
feature-order rejection; explicit validation early stop and best restore;
same-row naive arithmetic; test-data exclusion. Small synthetic cases prove
mechanics. A bounded real-train pilot measures cost before a full fit.

The release report must state what ran, exact command/environment, per-stage
loss, actual updates, weights path, memory/time, and remaining integrations.
No promise of flawlessness: defects found in these tests are fixed and retained
as regressions. Scientific efficacy requires the matched experiments above.
