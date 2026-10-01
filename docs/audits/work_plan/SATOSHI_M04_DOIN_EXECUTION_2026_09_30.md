# M04: DOIN execution of the modular predictor (MS13)

Satoshi, successor technical lead. 2026-09-30 (work ran into 2026-10-01 UTC).
Lane M04 of `docs/handoffs/SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md`, section 6.
Evidence for this document is in `docs/audits/evidence/m04_doin_20260930/`.

> **Correction 2026-10-01 (lane D).** Every result in this document is evidence of the
> OLD architecture (branch_steps=12 default, core time factors [2,1,1], compress-based
> branches/core). It is not evidence of the owner-corrected full-grid design. These rows
> are recorded as `SUPERSEDED_OLD_ARCH` in campaign `d_ecl_l24_h24_corrected_r0_v1`.
> No M04 pilot is queued or running. See `docs/audits/evidence/m04_doin_20260930/STATUS_CORRECTION_2026_10_01.json`.

## Result first

A candidate taken from the persisted DOIN queue reached the real trainer. An
independent process then scored its saved checkpoint and confirmed the result.
All 16 runnable R0 candidates of batch 1 (campaign v3, deterministic) were
trained through `doin_node.predictor_bridge`. Every one was rescored in a
separate process from its saved `best.keras` and matched the training receipt
bit for bit (`exact_match: true`). That covers 8 configurations, each with seeds
2021 and 2022.

The incumbent is `cc4235d7…` ("draw1", Huber). Its mean validation MAE is
**0.397174** in z_train space over 2609 windows x 24 horizons x 321 channels.
Persistence on the same rows scores 0.851406, so the aggregate skill_MAE is
0.527 / 0.541. Two caveats follow below: per-horizon skill is negative at h1 and
h24, and the task is not comparable to the published L96/H96 rows.

| Item | Measured |
|---|---|
| Cost pilot (grouped32 R0, worker_b RTX 4090, full train 18,341 / validation 2,609, 2 epochs) | Whole-cgroup peak **4,303,351,808 B** (load-dominated: 4.07 GB at load); TF device peak 229.9 MB; 574 updates; **0.140 s/update** including validation; wall 102.5 s |
| Per-feature default R0 (321 branches, width 5136) | **COST-01**: ended by its own 6.44 GB scope in the build/first-step stage with zero updates. Lower bound of need 5.80 GB. A CPU probe shows build alone at 0.39 GB; the first-step trace/compile adds 3.52 GB and takes 130 s |
| Batch caps | train 5130M = 1.25 x pilot peak. Verify 4G = 1.25 x the 3.03 GB lower bound measured on worker_a |
| v3 attempt peaks | train 4.09-4.62 GB; verify 2.42-3.14 GB |
| v3 seconds/update (elapsed/updates) | worker_a RTX 5090: 0.0104-0.0165. worker_b RTX 4090: 0.0191-0.0648 |
| Queue | 36 candidates persisted before execution. 16 VERIFIED. 8 per-feature held as `HOLD:BLOCKED_COST_01`. 12 R1/R2 blocked on donors |

## Dispatch block (as issued)

| Field | Value |
|---|---|
| Worktrees | predictor `satoshi/m04-doin-execution-20260930` (off `556c5f3e`); doin-node `satoshi/m04-doin-execution-20260930` (off `7411e5bf`); doin-core base `66b6d99` published as `satoshi/m04-doin-execution-20260930` |
| Source tips | predictor `8f724b12` (campaign pin); doin-node `bd61d4b`; doin-core `66b6d99` |
| First command | `tools/modular_doin_ecl_npz.py` under `crispdm-run -m 4G` (NPZ build, measured peak 2.69 GiB), then the cost pilot |
| Acceptance test | `tests/test_modular_doin_m04.py` (31 tests) plus the evaluator and engine suites: 85 passed. doin-node `test_predictor_bridge.py` plus handoff and CLI tests: 76 passed |
| Budget | Per attempt: train cap 5130M and 2 h wall; verify cap 4G and 20 min wall. Orchestration runner capped at 512M |
| Blocker | R1/R2 need real ECL donors (M02) under M01's effective-params identity, re-manifested (v4). The per-feature default needs an 8G solo slot (COST-01) |

## Task declaration (data and objective)

- **Source.** Data-gov `sota_benchmarks` ECL CSV, sha256 `7e45845d…`.
  Admissible inputs are bound to M03's final declaration `ca1098ed…`
  (feature-eng `9c36720`): 321 of 322 columns, with only the timestamp excluded.
- **Split.** The author's 7/1/2 borders. The StandardScaler (ddof 0) is fitted
  on train rows [0, 18412) only.
- **Purge.** 24 train windows were purged. The last train target row is 18387,
  before the first validation input row 18388.
- **Test rows.** Rows from 21044 on are dropped before any statistic. No test
  window exists.
- **Shapes.** Window 24 h, horizons 1..24, all 321 channels as inputs and as
  targets. Train has 18,341 windows (`9215b099…`); validation has 2,609
  (`e3712565…`).
- **Objective.** Validation MAE in z_train space; lower is better. The same-row
  last-value persistence baseline is computed alongside.
- **Comparability.** L24/H1..24 is not the published L96/H96 protocol, so this is
  `NOT_COMPARABLE` to TimeFilter's rows. The matched ECL comparison is its own
  lane.

## What was built

### predictor

**`tools/modular_search_space.py`.** The explicit, reversible mapping between the
optimizer's flat parameters and the versioned nested candidate
`modular.candidate.v1`. It is pure Python.

- Every invalid combination fails before TensorFlow is imported:
  - head and width divisibility;
  - strictly decreasing compression channels;
  - time factors that multiply to `branch_steps/output_steps`;
  - window and branch-step divisibility;
  - bounds;
  - capability-gated branch dilation.
- Conditional parameters fail too. `train.huber_delta` is active only when the
  loss is huber; an inactive or missing parameter is refused.
- An R1/R2 regime without a declared donor blocks the candidate.
- M01's ruling applies: the trainer is wired only through `from_flat`'s
  `nested["model"]`, never through raw flat keys.

**`optimizer_plugins/modular_doin_optimizer.py`.** An opt-in
`optimizer.plugins` entry with the same surface as `default_optimizer`.

- Its `optimize()` returns the incumbent's flat hyperparameters.
- Legacy plugins and configs are untouched.

**`tools/modular_doin_campaign.py`.** The persistent finite queue (SQLite, WAL,
synchronous FULL).

- Every candidate, crossed with every paired seed, is inserted before any
  execution.
- **Atomic claim.** `BEGIN IMMEDIATE` selects the next item, inserts the attempt
  row and marks it running, all in one transaction. Verification is claimable
  only by the host that holds the checkpoint.
- **Recorded per attempt:** cost (whole-cgroup peak, wall, updates,
  seconds/update), stop reason, selected epoch, model and weights digests, data
  digests, and failures.
- **Resume.** Completed, verified, refuted and failed candidates never re-run.
  An interrupted attempt returns to the queue under a new attempt number. A
  completed attempt that lacks verification resumes at verification only.
- **Holds** (`HOLD:…`) block a candidate without ever dropping it.
- **Placement.** Host-bound admissibility rules route work to a host that admits
  it.
- **Incumbent.** Only configurations whose declared paired seeds are all verified
  are eligible. They are ranked by mean objective, and every change is appended
  to `incumbent_changes`.
- **Remote execution.** One queue runs on the orchestrating host, with one light
  runner per worker role. The ssh alias comes from the environment at run time
  and is never written to a file.

**`tools/modular_checkpoint_scorer.py`.** The independent verifier.

- It runs in a fresh process and re-hashes the checkpoint and the validation NPZ.
- It loads the model with `compile=False, safe_mode=True`, so there is no
  optimizer and no fit path.
- It rescores at the receipt's own inference batch and recomputes every metric,
  including persistence and per-horizon values, in float64.
- It reports the verdict, the problems found, `exact_match`, the batch size,
  library versions and its own cgroup peak.

**Heartbeat and pilot tools.**

- `tools/modular_heartbeat.py` writes an unbuffered heartbeat at most every 30 s
  (cap 60 s). Each record holds the stage, epoch and update progress, the last
  validated checkpoint, cgroup and process memory, and an ETA with its basis.
- Its `run_request` asserts the three GPU facts inside the child: the driver
  exposes the UUID, TensorFlow registers a GPU, and a matmul runs on `/GPU:0`.
  Otherwise it raises `GPU_REQUEST_FELL_BACK_TO_CPU`.
- `tools/modular_doin_cost_pilot.py` records per-stage cgroup peaks with a
  per-fd reset of `memory.peak`, plus the TF device peak.
- `tools/modular_doin_build_probe.py` measures build and first-step cost.

**`tools/modular_candidate_evaluator.py`.** Gained only an optional `progress`
callback, used to feed the heartbeat. Results are unchanged.

**`tools/modular_doin_ecl_npz.py`.** Builds the identified NPZ inputs described
above.

### doin-node (`bd61d4b`)

- `evaluate()` **never retrains.** It locates the accepted receipt for the
  candidate identity (under `output_dir` or `receipt_roots`) and calls
  `verify_checkpoint`.
- `verify_checkpoint` runs the pinned predictor scorer in a fresh, isolated
  process. If no receipt exists it raises an error and does not train.
- The CLI gains `--verify`.
- Device visibility and threads are now configurable. When the pinned checkout
  provides the heartbeat, the worker uses it.
- Tests:
  - an evaluation without a receipt is refused;
  - a tampered checkpoint is refused;
  - settings are validated.

## Evidence of each acceptance point (MS13)

| MS13 element | Evidence |
|---|---|
| Candidate digest | Receipt `candidate.cid` = canonical sha of the nested config. The queue cid is equal (checked on every attempt) |
| Data digests | Each receipt binds train `9215b099…` and validation `e3712565…`. The queue fails an attempt whose digests differ from the declaration (tested) |
| Explicit validation metric and direction | `objective = {MAE, validation, higher_is_better false, z_train}` in the declaration, the candidate and every receipt |
| Retained weights | `best.keras` per attempt under `campaign_batch1_v3/attempts/<cid>/train-1/bridge/<id>/artifacts/`, with model and weights sha in the receipt |
| Replayable | Under `TF_DETERMINISTIC_OPS=1`, two independent fits of the same candidate gave identical weights sha `18c794a0…` and MAE `0.4520497731630392`. Every v3 rescoring is bitwise equal to its receipt |
| Real trainer and independent evaluator | 16/16 v3 attempts went DOIN bridge → pinned evaluator → separate scorer process, all VERIFIED with `exact_match: true` |

## Batch 1 v3 results (validation, z_train, persistence 0.851406 on the same rows)

| Config (label) | Shape | Seed 2021 | Seed 2022 | Mean |
|---|---|---|---|---|
| draw1 Huber `cc4235d7` **incumbent** | 1 branch x 321 features, d_model 128, 3x4 latent, δ 0.211 | 0.403170 | 0.391178 | **0.397174** |
| draw2 MAE `4c14e945` | 41 branches (grouping 8), 3 blocks, 8 heads | 0.406981 | 0.391440 | 0.399210 |
| draw2 Huber `8fa7d94d` | same draw, δ 0.160 | 0.408565 | 0.393390 | 0.400977 |
| grouped32 MAE `e323775b` (first incumbent) | 11 branches, width 176, default core | 0.403343 | 0.401216 | 0.402279 |
| draw1 MAE `331188a4` | as draw1 | 0.386604 | 0.421962 | 0.404283 |
| draw3 MAE `4ebc89be` | 1 branch x 321, 1 block | 0.422334 | 0.394921 | 0.408627 |
| draw3 Huber `468b6e1b` | δ 1.252 | 0.409110 | 0.439963 | 0.424537 |
| grouped32 Huber `fe256dab` | δ 1.0 | 0.422055 | 0.441446 | 0.431750 |

**Incumbent changes.** (1) `e323775b` at 0.402279 was the first paired-seed
verified configuration. (2) `cc4235d7` at 0.397174 replaced it with a lower mean
objective.

**Huber versus MAE.** Both losses ran on the same architecture and training
draw, with the same metric, population and budget. Huber won 1 of 4 pairs and
MAE won 3. The mean over pairs is 0.4136 for Huber and 0.4036 for MAE. Seed
spread within a configuration reaches 0.035 (draw1 MAE), which is larger than
most between-config gaps. Two seeds cannot separate these configurations, so
**no loss or architecture claim is made.**

**Per-horizon caveat.** For both incumbent seeds, skill_MAE is **negative at h1
(-0.49 / -0.44) and h24 (-0.58 / -0.53)**. It is positive from h2 to h12 (h6
+0.56/+0.58, h12 +0.67). Last-value persistence wins at the next hour and at the
daily lag. The aggregate skill hides this.

## Incidents and corrections (all recorded, none edited away)

1. **COST-01.** The per-feature default R0 needs more than 6.44 GB before its
   first update. The cause is the first-step trace/compile of the 321-branch
   graph, not graph construction. Its 8 candidates are held. Its pilot needs an
   8G solo slot on worker_b (coordinator ruling). A later SIGTERM sent by M04
   found no process; the scope had already ended at 23:48:26Z.
2. **Dead slice cache.** Admission on both workers charges dead clean page cache
   to the batch slice. This kept the worker_a pilot queued. No cap was lowered
   and no cache was cleared; the matter is escalated to the owner and M06.
3. **REFUTED in v2 (c52af198, 721a5b8b).**
   - Cross-process GPU float32 inference is not reproducible without
     deterministic kernels. Two fresh rescorings at the same batch already
     differ by 5e-7 relative, and the trainer by up to 1.9e-5.
   - The checkpoint bytes match the receipt, and in-process reload parity was
     exact.
   - Fix: `TF_DETERMINISTIC_OPS=1` for training and verification, plus the
     receipt's batch size. A deterministic campaign is VERIFIED only on exact
     match; a tolerance-only pass becomes `FINDING_NOT_EXACT`.
   - v2 is retained untouched as `superseded_nondeterministic`. v3 re-ran the
     same 36 candidate ids.
4. **Verify cap on worker_a.** Two verifications were stopped by
   SUSTAINED_ABOVE_RESPOND at their own 3G cap. Tree peaks were 3.03 GB, and the
   RTX 5090 (CC 12.0) PTX-JIT-compiles its kernels. The cap was raised to 4G from
   that measurement (amendment 2 in v2). `CUDA_CACHE_MAXSIZE=2147483648` is set
   on worker_a; its JIT cache stood at 186-200 MB.
5. **draw2 on worker_a.** In v2, three draw2 attempts failed at the 5.38 GB cap
   on worker_a (tree peaks about 5.03 GB, lower bounds). v3 runs draw2 only on
   worker_b through a host-bound admissibility rule; neither the cap nor the
   candidate changed. Under determinism, draw2 peaked at 4.09-4.13 GB on
   worker_b.
6. **Keras split.** The `tensorflow` env (Keras 3.13.2) cannot load Keras
   3.14/3.15 donors (`use_gate`). The campaign env is pinned to `tensorflow` for
   training, verification and donors, and every receipt records the versions.
7. **Donor identity.** Engine manifests compare params literally: `{}` is not
   equal to explicit defaults. A synthetic R1/R2 plumbing run on M02 donors
   (`PLUMBING_NOT_A_RESULT`) showed:
   - the BLOCKED → runnable transition;
   - R1 frozen weights byte-equal to the donor, R2 weights changed;
   - VERIFIED checkpoint rescoring;
   - `evaluate()` refusing to retrain.

   M01's effective-params identity (`64a91a74`) and its re-manifest tool
   (`41955f50`) are the v4 path.

## Not done, refused or not measured

- **R1/R2 (paired pretrained).** Blocked in v3. They wait for M02's real ECL
  donors (RECEIPT and core) and for the integration of M01 `64a91a74` +
  `41955f50` with a re-pin, the parity test and re-manifested donors. That is a
  separate v4 declaration.
- **Branch and core pretraining cost.** Not yet attached to any candidate;
  M02's donor run is still in progress.
- **The per-feature default R0.** Not trained (COST-01).
- **Statistics.** Two seeds per configuration give no uncertainty interval.
- **Test split and literature.** No test number exists. No literature comparison
  is licensed.
- **Warehouse.** No warehouse write. The metrics are local campaign receipts;
  grain mapping to `gov_*` is pending.
- **Determinism cost.** Seconds per update did not degrade visibly, measured
  before (v2) vs after (v3) on the same configs: worker_a 0.009-0.026 vs
  0.010-0.017; worker_b 0.025-0.069 vs 0.019-0.065. These are elapsed/update
  figures that include validation, so this is not a controlled benchmark.

## Locations

- **Queue and declarations.** `~/.local/state/crispdm-data-foundation/m04_doin_20260930/campaign_batch1_v3/` on the orchestrating host. Identical `CAMPAIGN.json` (sha `53bce359…`) on both workers.
- **Attempt directories.** On the worker that ran each attempt, under the same path.
- **Exported state.** `QUEUE_batch1_v3.json` and `QUEUE_batch1_v2.json`, both in this evidence directory.
- **Code.** predictor `8f724b12` (pushed); doin-node `bd61d4b` (pushed); doin-core `66b6d99`.

```
M04 — DOIN execution (MS13)
repo/branch/tip: predictor satoshi/m04-doin-execution-20260930 8f724b12 · doin-node satoshi/m04-doin-execution-20260930 bd61d4b · doin-core 66b6d99
files: tools/modular_{search_space,doin_campaign,checkpoint_scorer,heartbeat,doin_cost_pilot,doin_build_probe,doin_ecl_npz}.py, optimizer_plugins/modular_doin_optimizer.py, tests/test_modular_doin_m04.py; doin-node src/doin_node/predictor_{bridge,worker}.py
suites: predictor M04+evaluator+engine 85 passed · doin-node bridge+handoff+cli 76 passed
acceptance: 16/16 v3 candidates trained via DOIN bridge and VERIFIED exact_match by a separate checkpoint scorer; incumbent cc4235d7 mean 0.397174 (2 seeds)
what is NOT done / refused / not measured: R1/R2 (v4), per-feature default (COST-01), pretraining cost, test split, warehouse grain
```

Satoshi, successor technical lead, 2026-09-30.
