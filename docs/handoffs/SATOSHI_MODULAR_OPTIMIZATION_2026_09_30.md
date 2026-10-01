# Satoshi: concurrent modular optimization and integration

Written for Satoshi (execution), Musashi (independent review), and the owner.
Owner approved this continuation on 2026-09-30. Begin immediately from observed
state; do not wait for another "continue" after a completed unit.

**Latest inspection and immediate integration orders:**
[architecture reconciliation](SATOSHI_MODULAR_RECONCILIATION_2026_09_30.md).
The delivered M01/M02/M04 paths still used the superseded 12-step branches at
inspection; integrate the correction below before successor donor/candidate fits.

## 1. Objective and governing plan

Deliver a working, backward-compatible hierarchical predictor and a real DOIN
optimization campaign using it. The temporal stack is branches -> common time
grid -> fusion -> positional encoding -> Transformer core -> learned compression
-> forecasting or downstream policy head. Each branch and the core support
R0/R1/R2 with selected pretrained weights. The detailed design, acceptance matrix
and experiments are in
[MODULAR_STACK_WORK_PLAN](../tres_temas_entrevista/program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md).

**Owner architecture correction, 2026-09-30:** the first candidate implementation
violated the intended design by reducing time inside each branch before fusion.
That design is withdrawn. The corrected candidate is on predictor branch
`codex/modular-stack-20260930`, commit `da4ce7b4`; inspect that tip before fitting
or adapting M04's search space. Branches preserve the full window: hourly
`(batch,24,1) -> (batch,24,16)`. Fusion yields `(batch,24,16 * branch_count)`.
Positional encoding follows fusion. Two standard causal Transformer blocks each
use attention residual Add/normalization and FFN residual Add/normalization.
Three residual Conv1D stages then reduce time 24 -> 12 -> 6 -> 6 and channels
to `[32,16,8]`. There is no reshape/dense pooling in branches or core. The
forecast task head alone flattens the finished `(6,8)` latent to emit horizons.
The four-feature diagram is illustrative; branch count comes from configured
features/groups and the inventory still requires explicit measured coverage.

The corrected package has a CPU construction smoke only. Run its adjusted
focused component and pretraining suites before any GPU candidate fit. Discard
or regenerate candidate identities that contain the earlier branch reduction.
Updated source, Keras diagrams, Sphinx/Napoleon API docs and detailed decisions
are on the linked branch. This correction supersedes the earlier 12-step branch
default below and in older handoff snapshots.

The master plan links that subplan. Use it as the current modular specification;
older returns remain evidence of their own date. Implement any missing integration
needed by these orders rather than returning a list of engineering work for the
owner. A software boundary, model artifact or parameter mapping is our work.

## 2. Take over without duplicate work

In the first 15 minutes, inspect live services, GPU UUIDs, thermals, memory, disk,
reservations and all inherited agent/worktree states. Adopt useful active jobs.
Do not start a duplicate service or pull code beneath an active child. Re-read
unit state even if its description says queued: the child may already be training.

At handoff, Traffic h96 seed 2021 was training on the 4070 coordinator and seed
2023 on the 4090 worker. Seed 2022 completed in 3906 seconds: official normalized
MSE/MAE 0.3753611147/0.2512390912, published 0.375/0.251, paired persistence
MAE 1.0772232192. This is one cell; do not call it the three-seed result.
Preserve its checkpoints/receipt and finish the already-started cells.

The external 5090 remains first choice for new fitting. At the last read its host
had only about 1.9 GiB available RAM and 5.7 GiB unreclaimable slab; its GPU was
idle. Diagnose the actual host condition and use eligible alternatives while
doing so. Coordinator GPU clock was capped at 1500 MHz after thermal slowdown;
an existing reset service restores it after its current Traffic cell. The 4090
also reported thermal slowdown. Recheck physical cooling and observed limits.

After these active cells, assign the first admissible GPU slot to the complete
modular candidate cost pilot and optimization lane. CPU workers continue
profiles, tests, metrics/replay and reference checks where practical. Faithful
reference fits needing GPU get explicit slots after optimization has a working
incumbent/candidate queue. Neither heavy CPU work nor extra agents may starve
an experiment or the desktop. Do not fill hardware with irrelevant work.

## 3. Agents and ownership

Use at least four independent roles when capacity allows, up to six initially.
Hermes instances or available subagents are acceptable; record actual agent/task
IDs and acknowledgements. A proposed assignment is not a dispatch. Close or
reuse completed agents, do not accumulate unbounded sessions. Each writer gets
a separate worktree and disjoint ownership. You own integration and dispatch.

| Lane | Ownership and required deliverable | Can run alongside |
|---|---|---|
| M01 model assembly | Complete hierarchical components, predictor facade and save/load; shape/time/donor/regime behavioral tests | All CPU/data lanes |
| M02 pretraining | Integrate branch AE -> bounded fused representation -> core AE -> matched regimes; early stopping at every stage | Profiling and adapter development |
| M03 feature inventory | Enumerate dataset/column coverage, train-only profiles, grouping candidates and feature selection protocol | Fits and integration |
| M04 DOIN execution | Connect real trainer/evaluator to existing optimizer plugin API; persistent finite candidate queue and incumbent | CPU literature checks and M03 |
| M05 paper adapter | Implement modular inference/feature/action adapter for existing LTS routes; replay recorded inputs, then eligible paper smoke | Model fitting; promotion waits for model evidence |
| M06 evidence/resources | Collect completed cells, compare exact metrics/naive/literature, track ETA and capacity; review outputs independently | All work |

Do not make every lane wait for every other lane. M04 can prepare and test its
contract before M01/M02 finish, then consume their real implementation. M05 can
build and test inference without a scientifically promoted trading model.
Missing model weights or an adapter are not missing owner permission.

## 4. Compatibility and plugin architecture

Keep the current `predictor.plugins` facade and legacy flat config behavior.
Add one opt-in modular predictor; do not replace existing predictor plugins or
silently route their names to new models. Inside it, resolve independent branch,
fusion, core and head plugins through entry points, each with its own version,
parameters, tensor/time contract and serialization. Keep preprocessing separate.

Use a versioned nested modular config with deterministic serialization. Adapt
it to the optimizer's existing parameter interface explicitly, for example
`branches.price.params.channels` and `core.params.blocks`, with reversible
mapping. Conditional parameters and invalid combinations must fail before fit.
Changing plugin hierarchy must not break old JSON configs, result schemas,
entry-point loading, prediction_provider, or current LTS routes. Add behavior
regressions for those boundaries, and run them in installed isolated environments.

Defaults: at least 24 physical hours, one feature per branch, causal Conv1D
preserving the full time grid, concatenation on channels, positional encoding,
two full Transformer blocks, then three configurable residual Conv1D reduction
stages to 6 time steps x 8 channels. The head's forecast horizons/target count are
independent of this latent shape. No temporal collapse before fusion/core.
Compression cannot guarantee zero information loss; quantify reconstruction and
downstream utility. Shape equality alone does not establish time alignment.

R0 is fresh trainable initialization. R1 loads a verified donor and freezes it.
R2 loads the same donor and fine-tunes it. Each component can have its own regime;
the controlled scientific three-arm comparison uses the declared common regime.
An explicitly requested bad donor is an error, not permission to initialize a
different model. Core donors bind the actual upstream branch/fusion identity.

Every fit has configurable patience, min_delta, monitor cadence, hard limits,
best-checkpoint restore and observed optimizer updates. Apply to branch AEs,
core AE, forecasting and later RL fits. Exact literature reproductions retain
the author's recipe; new early-stop variants get separate experimental identity.
RL uses declared validation episodes and reward/cost criteria, implemented in
agent-multi/gym-fx; predictor's forecast early stop is not an RL implementation.

## 5. Feature metrics and selection

Produce an inventory coverage index with every known dataset and column, split
identity, status, profile path and missing reason. Report denominator and covered
count. Discovery of a file is not permission to profile its holdout. Reuse
existing profiles after verifying their bytes, split and implementation identity.

Compute train-only profiles in bounded CPU batches: missingness/constants,
distribution/scale/tails, volatility/trend, ACF, spectral summaries/periods,
stationarity diagnostics (ADF/KPSS with settings/failure states), and declared
seasonality diagnostics. Keep absent/unsupported metrics visible. Start with
one feature per branch; then compare domain groups, metric-space clusters and
redundancy/lag-informed groups using inner validation. Metrics propose candidate
extractors; they do not prove which architecture wins. Selection is fitted
inside training, preserves exclusion reasons and includes an all-admissible
feature control. No target-based selection on the external test.

Use existing data-gov/data-lake resource identities and warehouse metric grain
for governed campaigns. Local component tests are labeled local; do not issue
authority from fixtures or a JSON manifest alone.

## 6. Actual optimization and scientific order

First run a complete real-model cost pilot. Then execute a finite, persisted
DOIN candidate batch on the full declared train/validation populations. Optimize
the most promising *validation-eligible* model; architectural complexity alone
does not make an incumbent better. Begin with the full requested modular R0 and
paired pretrained R1/R2 candidates when donors are ready. Include both branch
and core pretraining costs. Keep the full model and data when testing resources.

Tune admissible branch types/widths/kernels/dilations, groupings, shared time
resolution, Transformer blocks/heads/width/dropout, compression depths/widths,
learning rate/decay, loss and Huber delta, plus stopping parameters within fixed
search bounds. Huber/MAE comparisons use the same final metric, population and
search budget. Persist candidates before execution, observed cost, failures,
selected weights, paired seeds and incumbent changes. Resume the remaining
candidates instead of repeating completed ones. This must reach the real trainer
and independent evaluator; `config_validated_no_training` is not completion.

Keep the matched ECL TimeFilter comparison as its own lane. Its nine retained
modular weights are missing; obtain a verified backup or train a new sealed
successor with new identities. A real CPU pilot measured 0.887 GiB; the previous
16.64 GiB refusal was aggregate reservation pressure, not model footprint.
Do not let the obsolete memory interpretation hold all fitting.

After validation selection, freeze the finalist and perform the declared test
comparison. LTS paper consumes a versioned inference/feature/action contract
and measured candidate. DOIN remains transversal; it is not the RL policy.
Do not infer trading profitability from forecasting MAE. Real-money deployment
is a separate existing account/risk mandate, not part of this handoff.

## 7. Inherited implementation locations

Inspect each branch tip and working tree before adoption; some are still being
completed by inherited agents. Integrate completed changes incrementally.

- Predictor orchestration branch: `codex/modular-stack-20260930`; plan plus
  staged pretraining helper/tests. Treat helper as integration work until its
  real engine/evaluator test passes.
- Engine: `feat/modular-temporal-engine`; agent Ohm owns branch/core/head/donor
  code and behavioral tests.
- Predictor evaluator: `codex/modular-candidate-evaluator-20260930`; agent Rawls
  owns validation objective and reusable early-stop loop.
- Feature profiling: agent Gibbs owns a separate feature-eng worktree; adopt
  its returned commit and coverage index, not a guessed completion state.
- DOIN bridge: agent Archimedes owns a separate doin-node worktree based on
  `c986c26`; it must invoke the real predictor evaluator.
- Prior DOIN contract: core `66b6d99`, node `c986c26`; dry run only.
- Prior LTS preflight: `99184e2`; current paper runner lacks modular adapter.
- Branch AE prototype: feature-extractor `1e4d7a9`; core AE pilot: predictor
  `b8fde409`; ECL matched fit path: predictor `9c3321d9`.

Current agents at handoff: Ohm `01a0f46b-5990-71d0-a6b4-50b540eb6f73`, Gibbs
`01a0f46b-59ba-71f0-94f2-83e9972f662a`, Rawls
`01a0f46b-59e5-7c50-9908-4c4e65f25d68`, Archimedes
`01a0f472-dd62-7ec1-ac16-cfd6f0d7ba27`. These are local subagent IDs, not Hermes
task IDs. Their completion receipts may supersede this snapshot; inspect before
spawning replacements.

## 8. Reporting: owner view, review and machine-readable state

### Completion addendum (supersedes the in-progress snapshot above)

All four inherited agents have returned. Do not wait on them. The orchestration
branch includes engine commit `1b76981a` (from `caa0b8bb`) and evaluator
`1b9daf6c` (from `4371ad90`). Their focused suites passed 24 and 30 tests
respectively. The integrated `tools/modular_pretrain.py` suite passed six tests
in 13.65 seconds on CPU, including actual branch/core training, donor saving,
and R1/R2 loading. These are synthetic component checks, not forecasting results.

Feature-eng branch `codex/feature-metrics-audit-20260930`, tip `dce037b`, has
15 passing tests and a bounded train-only profile: 512 rows, 23 features and
five exclusions. Its inventory reports 198/1680 financial appearances matched;
this is not full coverage. Start with `docs/feature_metrics/HANDOFF.md`.

DOIN node commit `7411e5bf` adds the isolated predictor bridge with 74 targeted
tests. Read `docs/PREDICTOR_BRIDGE.md` before integration. Its real-engine
acceptance is still pending; its `evaluate` currently retrains and MUST NOT be
presented as independent consensus verification. Add checkpoint-based independent
scoring before relying on that path. No live chain was changed.

Next integration priorities are the installed predictor facade/legacy compatibility,
real DOIN-to-evaluator acceptance, train-only domain inputs and staged donors,
then an admitted finite optimization campaign. The tests above do not establish
these integrations or authorize promoting an unmeasured model to trading.

Within 15 minutes publish one acknowledgement with actual assignments, observed
jobs, adopted tips and immediate candidate queue. Every live experimental child
writes a heartbeat at most 60 seconds apart: stage, epoch/update progress,
last-completed checkpoint, resources and ETA basis. Fix buffered-only logging in
future runner snapshots; don't hot-edit the currently executing source.

Update `STATUS.json` atomically at least every five minutes and on job state
changes. Every 30 minutes, or earlier on a completed scientific result/incident,
give the owner a short result-first report. Routine unit tests don't need chat
messages. Do not wait for all agents to finish to publish a completed cell.

At each coherent delivery publish:

1. `RETURN.md`: what became executable; new numerical results; what failed;
   exact remaining dependencies; next dispatched work; commits and test scope.
2. `STATUS.json`: use
   [the report schema](../contracts/modular_status.schema.json), including
   per-device running/queued/idle reason, candidate progress and ETA assumptions.
3. `RESULTS.json` or CSV: task/split/rows, model/config/seed, exact metric space
   and reduction, model error, same-row naive/skill, published comparator and
   comparability, training and pretraining costs, weights and evidence identity.
4. `PROGRESS.png`: generated from the same status, showing dated scientific and
   integration milestones, present task and ETA. Report completion criteria and
   fixed weights if giving a weighted completion percentage. Do not turn tests,
   prototype count or arbitrary visual progress into measured scientific progress.

One request for Musashi review per coherent packet. Its pending review holds
only claims/actions that depend on it; other implementation, profiling and
eligible candidates continue. Report software compatibility, actual changes in
R1/R2 weights, serialized-model replay, data support and candidate comparisons
as independently inspectable evidence. No repeated owner's approval for already
authorized offline engineering or bounded experiments.

Stop and report the affected job on nonfinite outputs, target/data identity
failure, memory/thermal breach, or exhausted allocation. State the concrete
object needed and dispatch an independent eligible task. Do not promise all
GPUs will be active regardless of useful work and physical capacity.
