# Modular temporal stack: implementation and experimental plan

Owner instruction: 2026-09-30. This subplan belongs to the master work plan and
runs concurrently with reference reproduction, profiling and product work.

## Goal and boundaries

Build one reusable temporal representation for forecasting and future RL heads:
typed feature selection -> independent branch encoders -> common time grid ->
fusion -> positional encoding -> full Transformer blocks -> progressive learned
compression -> task head. Support random, frozen-pretrained and fine-tuned
initialization independently for every branch and for the core. DOIN searches
the declared candidate space using validation objectives. predictor owns offline
forecast fitting, feature-eng owns descriptive feature profiles, feature-extractor
owns reusable representation producers, and agent-multi/gym-fx own policy fitting.

The new assembly is a usable implementation, not evidence that it beats the
published reference. Existing TimeFilter cells keep their exact author recipe.
No scientific conclusion is based on a small plumbing fixture.

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
3. Freeze those branch encoders during materialization; encode train and internal
   validation in bounded batches. Fuse with the same plugin/grid used downstream.
4. For the reconstructive control, train the core AE on the fused sequences; early stop on its own
   internal validation. Export the core with upstream branch/fusion identities.
5. Fit forecasting heads under branch/core regimes. Baseline R0 has no donor;
   R1 freezes the specified donor; R2 starts from the same donor and fine-tunes.
   Mixed per-component regimes are allowed and recorded, not conflated with the
   original three-arm scientific comparison.
6. Restore selected weights and replay predictions after serialization. Persist
   actual parameter counts, updates, stop reason, monitor values and identities.
7. Evaluate on fixed validation rows with paired persistence. Select candidates
   by the declared objective across paired seeds, not their best seed.
8. Confirm frozen finalists on the untouched test once under its declared design.
   Export eligible candidates to the LTS paper adapter only after this result.

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
published reference; modular R0; branch-pretrained R1/R2; core-pretrained R1/R2;
joint pretraining; one-feature vs grouped branches; selected vs all admissible
features; bottleneck width/time/depth; MAE vs tuned Huber. Report configuration
and optimization cost, including AE cost, with every model metric.

DOIN optimizes the declared validation objective. Candidate validity includes
time divisibility, width/head compatibility, donor identity, parameter bounds
and resource limits. A dry-run contract is not an optimization result. Keep the
incumbent, candidate history and independent verification result in the existing
warehouse path when that adapter is exercised; local smoke metrics remain local.

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
