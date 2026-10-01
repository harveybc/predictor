# Satoshi: next dispatch after the corrected R0 batch

Written for Satoshi, Musashi and the owner. Snapshot from STATUS at approximately
03:44 UTC, 2026-10-01, plus current receipts and read-only device checks.
Continue the current agents and valid jobs. This is a delta to `256c61a6`, not
a restart of the program. The owner accepted proceeding with the reviewed
admission repair on the preferred worker; no physical restart was requested.

## 1. What has changed: do not repeat completed work

- Integrated engine `3ecdb256`, evidence head `26c13fda`: M01 reports installed
  tests and preserved legacy boundaries. Default branches now preserve 24 steps.
  The integration task in the previous orders is no longer an unmet prerequisite.
- Corrected campaign pin `df9ae31c`: 16 verified R0 cells, eight paired-seed
  configurations; eight per-feature cells and twelve R1/R2 remain in this
  36-cell batch. Keep 72 historical superseded rows outside that denominator.
- Recomputed from the queue and 16 replay receipts: best paired configuration
  `corrected_draw3_R0_mae`, MAE_z 0.3910575164106178; naive 0.8514061497227646.
  Validation ECL L24/H1..24 only, NOT_COMPARABLE to the paper's L96/test rows.
  h1/h23/h24 skill negative in 16/16; h2 in 2/16. Preserve these warnings.
- Corrected donors on worker_b: 181/321 branch records in direct read at 03:43
  UTC, CPU, core pending. M06's throughput estimate at 03:42 was 04:16-04:35
  UTC for branches ONLY (23:16-23:35 Colombia, September 30). Update the interval
  from live progress; do not reuse it for the core or R1/R2.
- Source expansion now has 22 providers, 5,273 discovered files and 77 transform
  ledger rows; no evaluated/selected transform rows in that new ledger. The
  3,538/15,228 column-profile index is a separate denominator.

## 2. Immediate resource action: preferred worker, not a reboot

Deploy the reviewed admission repair at `775c5545` on worker_a only, using the
retained DEPLOY_ADM_DEADCACHE.sh procedure and rollback. Verify current old/new
hashes, no conflicting deployment, and the script's exact scope before running.
Do not deploy `0928dc06`, which subtracted live-scope cache. No changed memory
ceilings, global drop_caches, swap changes, persistence mode, host reboot or
force-push. Do not extend this authorization to the other worker automatically.

Read back installed hashes, admission accounting and live reservations; run
the bounded smoke. Then check one actual CUDA operation in the existing pinned
environment on the 5090 UUID, not merely nvidia-smi enumeration. The recent
driver NV_ERR_NO_MEMORY is recorded separately from the admission defect; if a
new CUDA operation fails, retain its error and diagnose it rather than claiming
the bookkeeping repair proves GPU health. Independent CPU work continues.

The current 5090 is detected at 37 C with no compute process; host pressure was
zero and available memory approximately 11 GiB. Those are observations, not a
permanent reservation. Reissue only your own queued old-launcher requests after
checking their identities. Never duplicate a candidate already running/verified.

## 3. Parallel execution and integration

| Owner | Execute now | Completion evidence |
| --- | --- | --- |
| M06 + M04 | Repair worker_a admission; place the next eligible pilot/candidate on the external 5090. Consider moving the per-feature pilot only if its full measured RAM/VRAM/materialization demand fits there. | Installed hashes, fresh lease, in-child CUDA facts, heartbeat, result/terminal and cost. |
| M02 | Continue corrected branch pretraining, export/reload selected weights, materialize fusion in bounded batches, then core pretraining. | Branch/core donor identities, upstream grid, internal-validation early stop, actual updates and memory; no old-grid donor relabeling. |
| M04 | Finish the finite corrected queue. R1/R2 consume verified compatible donors as they arrive; retain paired seeds and full per-horizon scores. | No repeated completed R0 cells; pretraining and search costs included, saved checkpoints independently rescored. |
| M01 | Complete the representation-cost pilot with an honest cap; fix missing progress heartbeat. Do not run simultaneous memory probes that pressure-stop donors. | Measured stage cost/peak, engineering label, ETA, installed contract tests; not a new forecasting score. |
| M03 + C | Source entitlement/availability reconciliation, native/proxy transform repair, regime fitting-period audit, PS2 reversible selection and PS3 causal studies. | Explicit source/recipe/feature coverage and prefix/fold tests; unknown entitlement is not fabricated or silently excluded. |
| M05 + RL owner | Continue real runner consumption in shadow/paper; implement the SAC/DQN x with/without differentiated temporal representation matrix in agent-multi/gym-fx. Satoshi assigns the next available agent as RL owner. | Installed end-to-end consumer, no broker order outside existing mandate; RL design/fixtures distinguished from financial performance. |
| M06 | Close small receipt/table jobs on an admissible CPU host rather than waiting behind large fits on one host. | Corrected architecture tables and warehouse readback, separate from old batch; current PNG and status. |

Reuse materialized features and all completed reference cells. A transfer or
evaluation stage is legitimate activity, but is not a training job. Keep the
next useful job queued; no synthetic busywork to show GPU utilization.
The 5090/internal card share host RAM. Preserve coordinator desktop headroom.

## 4. Resolve technical questions without another owner approval loop

These are implementations of the existing approved plan, not new science or
financial permissions:

1. RL owner: Satoshi assigns an agent under M05 coordination; agent-multi/gym-fx
   owns policy training, predictor exports the temporal encoder, DOIN optimizes
   across them. FS13/14 remain unaccepted until actually demonstrated.
2. Causal-study home: causal-inference's maintained provider package, extending
   its tested interfaces; no automatic adoption of legacy experimental code.
3. B supplies C the available contracted resource manifest. Prepare missing
   contracts from real source provenance; a missing contract is a local task,
   not permission to invent availability or stop all other datasets.
4. Keep the predeclared strict-minimum validation incumbent rule. Retain full
   precision and paired uncertainty; do not add an arbitrary tolerance that
   deletes small improvements or claim every lower number is a proven advantage.
5. Report three denominators: distinct column profiles, metric-family cells,
   and source/recipe catalogue coverage. None substitutes for the others.
6. Add required provenance/OPERATIONAL versus SYNTHETIC_OFFLINE fields with a
   versioned compatibility migration. Unknown historical provenance stays
   unknown; do not silently invent safe defaults for learned corpus identity.
7. Pin `[2,2,1]` as the default; retain other valid time factors as declared,
   optimizable variants. Tests must distinguish a default from an invariant.
8. Y_s/Y_l/Y_b are financial targets. Author-protocol TSL tasks retain their own
   targets, scales and splits; mark financial-only criteria NOT_APPLICABLE there.
9. FS08/09 are optional-generation criteria: show them separately as deferred;
   the 20-item catalogue remains visible, without pretending all 20 are required
   for the first operational release.
10. Purge predictive labels by their target support; purge strategy evaluation
    additionally by actual trade-support/holding rules from the versioned plugin.
    Existing read reserves are development evidence, not fresh confirmation.
11. Preserve each dataset's declared split. Do not impose a financial 4y/1y/1y
    recipe on exact TSL reproduction; do not treat row offsets as elapsed hours
    across gaps. Inner folds derive from the actual task's supports.
12. Cross-lane owners are explicit: FS16 B+A, FS19 B+F, FS18 D+B. Send artefacts
    directly on completion and integrate incrementally.

Genuinely external facts remain narrowly scoped: exact Yahoo/Alpaca paid-product
entitlements, execution of C07 remote code under its license, and a new reserved
confirmation period. Prepare one concise consolidated request for those facts,
not seventeen implementation decisions. Existing source access asserted by the
owner is a discovery lead, not invented API coverage. Check nonsecret product
metadata and existing connector configuration first. No key values in reports.

## 5. Diagnostic follow-up on the horizon pattern

**Owner addition, binding for the next strategy evaluations:** do not run the
heuristic strategy with learned predictions that have not beaten persistence.
This is an eligibility rule, not a universal claim that worse MAE implies a loss.

- Freeze the forecast metric before selection: MAE is the current primary metric;
  report MSE alongside it. Do not switch metrics after seeing which passes.
- On the strategy's declared asset, period, scale and identical valid rows,
  require finite model MAE < naive MAE for every consumed short/long horizon.
  Check the exact configuration of the strategy, not an invented 1h/24h pair.
  A missing result or a tie does not pass. With naive zero, strict improvement
  is impossible; reject rather than divide by zero or call it infinite skill.
- Use held-out validation / chronological out-of-fold evidence available before
  the trading evaluation. Never peek at each future target at trade time to
  decide which predictions to execute, and never use the reserved trading test
  to select model eligibility. Do not repeatedly tune against the gate's holdout.
- Both prediction families must pass. A favorable overall mean cannot mask a
  failed consumed horizon. Do not silently remove a failing horizon: a reduced
  strategy-input configuration is a separately declared experiment.
- On failure, record SKIPPED_NOT_BETTER_THAN_NAIVE with forecast scores, paired
  baselines, horizon/population identities and reason. Do not start the strategy
  or generate/store trade trajectories for that rejected candidate. Continue
  independent forecast training/diagnosis; preserve historical evidence.
- Do not launch new below-naive strategy/noise arms automatically. Any future
  diagnostic exception must be explicit, not slipped into the business queue.
- Every displayed/saved MAE and MSE must include its corresponding baseline on
  the same dataset/split/rows/horizon/normalization/reduction, and skill or delta.
  If no legitimate baseline exists, say NOT_AVAILABLE and why, not zero or a
  value borrowed from another task. Training losses are labelled as losses,
  not silently substituted for these evaluation metrics.

Required tests before enabling this gate: one failing short horizon, one failing
long horizon, favorable macro mean with a failing member, equality, zero naive,
missing/NaN metric, mismatched rows/scaler, and a genuinely passing configuration.
Failure must produce zero strategy invocations; test this at the actual runner.
M05 owns consumption, M04 supplies the frozen forecast evidence, M06 reporting.

In CPU, without retuning on test: inspect same-row per-horizon target scale,
variance, baseline seasonality, residual mean and prediction dispersion. Plot
the naive and model curves and compare against existing calendar/seasonal
controls. Confirm the 24-hour endpoints are not indexing mistakes. A positive
aggregate does not justify suppressing the losing horizons; the pattern alone
does not prove a bug or financial uselessness. Any objective/horizon-weight
change is a new TRAIN/validation ablation, not a rewrite of this batch.

## 6. Owner addition: parallel SAC and DQN experiment lane

Follow the [RL subplan](../tres_temas_entrevista/program_v3/RL_TEMPORAL_COMPARISON_WORK_PLAN_2026_10_01.md).
The four primary arms are SAC baseline, SAC modular temporal, DQN baseline and
DQN modular temporal. This is an experiment instruction, not a claim that these
four integrations or fits already exist. The forecast-versus-naive gate above
does not apply to RL policy admission.

Prepare adapters, environment tests, configs, accounting and monitoring NOW in
parallel. Start real-data cost pilots as soon as a versioned selected-feature
batch and its required point-in-time datasets are registered and readable in
the existing lake. Do not wait for all inventory-wide selection, all causal
studies, or forecast/Traffic completion. Selection must be frozen for this task;
it cannot be silently recomputed from policy evaluation or future data.

Preserve paired seeds and the same financial environment within each algorithm's
representation contrast. Standard DQN uses discrete actions; continuous SAC and
DQN are not an architecture-only comparison. Freeze and disclose the action
mapping and use only a tested discrete-SAC implementation for a shared discrete
comparison. No silent conversion or new RL engine from scratch.

Use the compatible temporal encoder in the actor and critics of SAC, and the
online/target Q networks of DQN, with explicit weight-sharing and frozen/tunable
donor contracts. Test actual optimizer updates, target synchronization and
checkpoint replay. The no-modular baseline must be identified, not mislabeled
as our branching architecture. Match information, tuning budget and episode
population; report parameter/compute differences and pretraining cost.

Queue bounded, monitored GPU jobs after a measured cost pilot using the existing
resource envelopes. Prefer the external 5090; other admitted GPUs may run
independent cells. Every fit has validation-episode early stopping, best-policy
restore, hard limits and heartbeat. Reward, costs, slippage, margin, execution
timing and terminal positions stay explicit. Report net return, Sharpe with its
sampling convention, drawdown, turnover/trades and baselines on identical
episodes; no claim of financial utility from fixtures. No real-money deployment
is implied. Feed tested finalists into the existing shadow/paper LTS lane.

## 7. Report and ETA

Ack this delta within 15 minutes of actual receipt. Return deployment outcome,
actual GPU/CPU jobs, remaining finite queue, and next measurable result. Keep
heartbeats <=60 s and STATUS <=5 min; report every 30 min while active and at
each terminal. A child without progress reporting gets a heartbeat at its next
safe successor, not a fictional ETA.

The current donor ETA is stage-specific. Measure the core pilot before giving
its duration; price R1/R2 from compatible measured fits. For queued work state
both execution-duration estimate and waiting-time uncertainty. No fixed global
finish date from a one-branch ETA. Include tasks whose estimates expired.

M06 should regenerate the progress PNG and JSON from the same state. Show actual
completed/declared units and measured scores; no invented weighted percentage
of the entire doctoral/business program. Link both old and corrected campaign
identities, model/naive/literature where comparable, memory/cost and exclusions.
