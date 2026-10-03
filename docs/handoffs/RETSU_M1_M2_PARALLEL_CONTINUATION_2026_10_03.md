# Retsu: literal M1/M2 parallel continuation

This order supersedes the Gamma retry paragraph in
`RETSU_CRITICAL_PATH_PARALLEL_EXECUTION_2026_10_03.md`. It does not supersede
the master plan. Execute lanes A, B, C and D independently; a blocked lane must
not stop the others.

## 0. Start and invariants

Fetch `origin/satoshi/canonical-exec-20261003` and create one fresh worktree per
code lane from its current tip. Verify that `1b832e9b`, `1f86accf`,
`8987c573` in feature-extractor, and `48ae17c` in causal-inference are reachable
from their declared repositories before use.

Never run NEAT, H-CORE, real-data RL, strategy evaluation, or calendar as a
model input in this dispatch. Use seed 0 for existing PS3-R contracts. Do not
repeat a completed cell. Every MAE/MSE must carry its same-row naive; a model
that does not strictly beat the configured naive is not strategy-eligible.

## A. M1 weekly scorer on omega CPU

Execute `docs/handoffs/RETSU_M1_WEEKLY_SCORE_GATE_2026_10_03.md` literally in a
fresh branch. This is engineering and focused tests only. It must not open real
test data or train a model. If the business task still lacks a primary error
metric, retain `NOT_CONFIGURED`; do not choose one silently.

## B. Gamma: use the 5090 for the workload that fits

1. Prove there is no `app.univariate_temporal_pilot` child and identify the one
   queued `laneE_fred_rates_dprime_logret_5d` waiter.
2. Cancel that queued waiter by exact request identity. Preserve its receipt.
   Do not cancel a child or lease if the state changed and real work started.
3. Do not resubmit `dgs30` or `dprime` below 8,000 MiB on gamma. Their existing
   refusal/queue records are terminal placement evidence, not a timer.
4. Resume lane F on the external RTX 5090, not the internal 5070 Ti. Use the
   existing plan and scientific arguments, select the first missing terminal
   unit, and change only physical placement to `CUDA_VISIBLE_DEVICES=1`.
5. `masked_temporal_ae` keeps its retained 5,940 MiB cap;
   `past_to_current_siamese` keeps 6,670 MiB. One gamma child total. Before
   launch verify the physical UUID inside the child is the external 5090.
6. After each terminal unit, continue with the next missing unit. Never infer
   completion from the driver log alone; require the run manifest and result
   digest.

Do not run lane E and F concurrently on gamma. The goal is useful 5090 work,
not two processes competing for the same 14 GiB host.

## C. Dragon: continuous heavy PS3-R baseline queue

Dragon owns the 8,000 MiB identity/random/AE/DAE cells while gamma runs lane F.
Use the authenticated batch-002 bytes already present. Before every launch,
re-hash `series.npz`, `targets.npz`, the batch manifest and code, and create an
exclusive claim checked against gamma and dragon terminal results, children,
waiters and claims.

Run exactly one cell at a time on the RTX 4090:

1. `fred.rates.dgs30.level`, if still globally unclaimed and without terminal;
2. `fred.rates.dprime.logret_5d`, under the same conditions;
3. the next canonical unclaimed PS2-priority feature.

Use the same scientific command as the accepted vix cell: window 168, latent 8,
seed 0, families `identity,random,ae,dae`, max fit windows 16,384, early stopping
from the pinned runner, and an 8,000 MiB governed cap. No smaller population or
cap retry. If a cell cannot be admitted on dragon, record the terminal resource
failure and advance only to a scientifically independent cell that fits.

Each result must report the 20 fold-family rows, 280 probes, 140 deltas and one
summary or explicitly explain why that schema does not apply. Report trained
versus random and trained versus raw by target and horizon; reconstruction alone
never selects a feature.

## D. CPU selection readiness and audited PS4 integration

Start after A is dispatched; it may run concurrently with B and C.

1. Integrate the accepted PS4 transform subpopulation from
   `docs/audits/evidence/canonical_20261003/ps4_transform_profile/` into the
   coverage ledger by content digest. Change only the ten corresponding
   transform statuses from `PENDING_PROFILE` to measured. Do not label all PS4
   complete.
2. Produce one deterministic readiness row for every one of the 366 candidates
   with PS0/PS1, PS2, PS3-C, PS3-R, PS4 and PS5 status, exact evidence locator,
   digest and missing next action. `NOT_IDENTIFIED` in PS3-C is neutral.
3. Fail if the denominator is not exactly 366, any candidate is absent or
   duplicated, a terminal status lacks evidence, or a missing status is cast to
   zero/rejected.
4. For every newly completed PS3-R feature, add its readiness evidence and
   schedule only the PS4 metrics required by the plan for prioritized
   survivors/exploration. Do not compute an unbounded product and do not issue
   a final feature manifest.
5. Prepare the common-K PS5 comparison inputs (`K=24`, sealed sensitivity
   `8,16,24,32,48`) on identical inner TRAIN weekly folds, including predictive
   baseline, +causal, +extractibility, random-K and all-admissible controls.
   Preparation is not permission to train before readiness gates pass.

Add test-first counterexamples for a missing candidate, duplicate candidate,
false causal rejection, stale PS4 digest and PS3-R reconstruction presented as
selection utility.

## Return format

Heartbeat only on start, terminal cell, or every 30 minutes while work is live:

```text
UTC | host/resource | lane | exact unit | RUNNING/QUEUED/DONE/FAILED |
completed/denominator | ETA basis | next unit
```

Return measured results first, then one section per lane with branch, commit,
tests, digests, exact denominators and remaining work. Update M1/M2 progress only
from completed acceptance criteria. Explicitly distinguish queued admission
from live GPU work and list GPU UUID, utilization and memory for every live
child. Commit and push each lane; do not wait for all lanes before publishing a
completed independent lane.
