# Retsu: continuation on the M1-M8 critical path

This order supersedes the unexecuted PS4-only order only by adding parallel
resource lanes. It does not change its scientific population. Execute it
literally. Do not create a new plan, inventory, reconciliation-only return, or
status schema.

## 0. Immutable start and authority

Run:

```bash
git fetch origin
git rev-parse origin/satoshi/canonical-exec-20261003
git merge-base --is-ancestor 61c8ea9d origin/satoshi/canonical-exec-20261003
git merge-base --is-ancestor ca167efc origin/satoshi/canonical-exec-20261003
```

All four commands must succeed. Create fresh worktrees from that exact remote
tip. The governing documents are:

1. `docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md`;
2. `docs/audits/evidence/canonical_20261003/MASTER_MILESTONE_STATUS.json`;
3. this order;
4. `docs/handoffs/RETSU_PS4_TRANSFORM_PROFILE_EXECUTION_2026_10_03.md` for the exact PS4 unit.

The spearhead is M1 -> M2 -> M3 -> M4 -> M5 -> M6/M7 -> M8. Optional market
regimes, economic-calendar model inputs, synthetic generation, and NEAT do not
enter this dispatch. The calendar may be used only to identify historical
treatment/control episodes inside M2.

## 1. Parallel lanes

Start lanes A, B and C independently. A failure in one does not stop the others.

### Lane A: PS4 transform profiles on CPU

Execute `RETSU_PS4_TRANSFORM_PROFILE_EXECUTION_2026_10_03.md`, sections 1-6,
without converting it into another join. Placement is omega, CPU only, one
process and one BLAS thread, under:

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 7200s -n ps4-transform-profile -- <exact runner command>
```

Read one parquet column at a time. Preserve all 50 feature-fold units. If 2 GiB
is insufficient, return the measured stop; do not shrink features, folds, rows,
or metric families. The output must contain real PS4 metric rows, not merely
PS1 links.

### Lane B: recover and continue PS3-R on gamma 5090

Before changing admission state, prove both:

```bash
pgrep -af 'app.univariate_temporal_pilot'
pgrep -af 'laneE_.*crispdm-run|crispdm-run.*laneE_'
```

If a model child is alive, do not cancel or duplicate it. If there is no child
and the exact `laneE_fred_rates_dgs30_level` request is still only queued, cancel
that queued admission by its exact request identity, record the cancellation
receipt, and relaunch the same scientific cell once at **7936 MB**, the approved
maximum below the 8 GiB slice:

```bash
mapfile -t WAIT_PIDS < <(pgrep -f '^bash /home/harveybc/.local/bin/crispdm-run .* -n laneE_fred_rates_dgs30_level ')
test "${#WAIT_PIDS[@]}" -eq 1
python3 "$HOME/.local/libexec/crispdm/crispdm_admission.py" cancel \
  --name laneE_fred_rates_dgs30_level --witness-pid "${WAIT_PIDS[0]}"

crispdm-run -q -W 86400 -m 7936M -t 6h \
  -n laneE_fred_rates_dgs30_level -- \
  /home/harveybc/anaconda3/envs/tensorflow/bin/python \
  -m app.univariate_temporal_pilot \
  --batch_dir "$HOME/.local/state/canonical_20261003/ps2/batch_002" \
  --out_dir "$HOME/.local/state/canonical_20261003/ps3r/E/runs/batch_002/fred.rates.dgs30.level" \
  --features fred.rates.dgs30.level --window 168 --latent_dim 8 --seed 0 \
  --families identity,random,ae,dae --max_fit_windows 16384
```

Use the pinned feature-extractor checkout already recorded by lane E. Verify its
commit before launch; do not run from the predictor checkout. If the 7936 MB
attempt is stopped for memory, do not retry lower. Record `BLOCKED_BY_SLICE`.
After every terminal cell, run the existing lane-E aggregator and publish the
fresh completed/pending counts; the stale 16/137 summary is not authoritative.

### Lane C: disjoint PS3-R work on dragon 4090

Copy the authenticated PS2 input needed for one unclaimed batch-002 feature
from gamma to a new read-only dragon directory. Re-hash source and destination.
Create an exclusive claim before launch with this identity:

```text
(batch_id, feature_id, stage=PS3-R, families=identity/random/ae/dae,
 seed=0, input_digest, code_digest)
```

The feature must have no terminal result, no live child, no queued request, and
no claim in lanes E/F. Select the first such feature after
`fred.rates.dgs30.level` in the canonical queue. Run the identical scientific
arguments on the 4090 under the retained 8,000 MB cap only if dragon admission
accepts it without reducing the population. Do not run two PS3-R cells on the
same host concurrently. Return the claim and input/output digests.

## 2. Selection work after the three lanes start

Use CPU capacity not needed by lane A to prepare, but not fabricate, the final
M2 comparison:

1. denominator is 366 candidates, not the ten transformed outputs;
2. PS3-C `NOT_IDENTIFIED` is neutral, never causal rejection;
3. compare common-K sets on the same TRAIN-only weekly folds: predictive and
   redundancy baseline, plus causal evidence, plus extractibility evidence;
4. include one random and one all-admissible control where computationally
   feasible;
5. use one seed by default and at most three total only if required;
6. no feature is selected from reconstruction alone;
7. output must name selected, rejected and pending candidates with reasons.

Do not issue the final feature manifest until PS3-R and the required PS4 rows
for every evaluated survivor are present. Do not start H-CORE, NEAT, strategy
evaluation or real-data RL from a provisional manifest.

## 3. Independent nonblocking work

Literature rows may be consolidated on CPU while A-C run. Preserve
`LITERATURE_STATIC` exactly and keep it separate from `BUSINESS_WEEKLY` and
`BUSINESS_MONTHLY`. Every reported MAE/MSE must carry its same-row naive.

Do not retrain a completed literature cell. Do not create trading output from a
predictor that fails its naive. Do not run optional calendar-input, regime or
NEAT work.

## 4. Return cadence and required format

Send a compact heartbeat when a lane starts, when a cell finishes, or every 30
minutes while work is live. Each heartbeat must contain:

```text
UTC | host/GPU | lane | exact cell | RUNNING/QUEUED/DONE/FAILED |
completed/denominator | measured ETA basis | next cell
```

The final return must begin with new measured results. Then provide:

- branch and commit per lane;
- PS4 units and metric rows by status;
- current PS3-R completed/denominator after a fresh aggregation;
- exact gamma and dragon cells, seeds, digests, time, peak RAM/VRAM;
- feature-selection denominator and remaining evidence, without declaring a
  final selection early;
- M1-M8 status changes and ETA changes for
  `MASTER_MILESTONE_STATUS.json`;
- failures and the exact next executable action.

`NO_NEW_MODEL_MEASUREMENT` is acceptable only for lane A. It is not an adequate
return for lanes B/C unless both produce explicit terminal resource failures.
