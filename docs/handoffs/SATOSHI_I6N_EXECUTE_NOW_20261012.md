# Satoshi: integrate, deploy, and hand over to automation

Issued 2026-10-12 00:30 UTC. This order continues the approved I6-N plan; it
does not replace the scientific contract, reopen TEST, or authorize live trades.
Read this while your current work continues. Reconcile existing work instead
of starting duplicate agents, integrations, queues, or fits.

## 1. Verified starting point

- Your integration worktree already contains B's correction `f7a05838`.
  Do not report B as undelivered. A.3 and C.3 are its ancestors.
- Musashi's published branch is `codex/i6n-launch-integration-20261011`,
  tip `2101d97e`. Read `docs/i6n/swarm/MUSASHI_LAUNCH_RETURN.md` there.
- Three distinct TRAIN cost pilots completed: h2 minimal on secondary,
  h12 S1 and h36 S2 on preferred. They have 72 OLAP reports / 558 metric
  rows with readback verification. They are NOT annual validation results.
- Corrected S3 is running on preferred: 106 updates, epoch 2 at the last
  retained observation. Root `/var/tmp/i6n-launch-cost-S3-corrected-20261011/`;
  unit `i6n-cost-preferred-2df243fc.service`. Refresh status once before
  placement. Do not duplicate, stop, or replace this fit.
- Secondary reboot was verified healthy; its cost pilot is complete.
  Fresh admission is still required for the next workload, not another reboot.
- `i6n-launch-cost-collect.timer` automatically mirrors and publishes pilots.
  Preserve it. Its roots and private configurations are local, not repo assets.

## 2. Dispatch up to three independent, finite tasks

Use available subagents or Hermes for bounded work, not resident monitoring.
Reuse an agent already doing the same task. Each owns a separate worktree.
Keep at most one CPU test job per lane; obey aggregate memory admission.
Do not add token-consuming polling agents.

### A. Integration owner: launcher and real runtime

1. Start from your current integrated branch. Adopt `5bff3b21` or its exact
   tested behavior: the real admitted launcher loses inherited stdin, so the
   evaluator request must be relayed from a private digest-checked file INSIDE
   admission. Preserve fencing, orphan handling, and resource limits.
2. Adopt `75e4f749`'s trajectory-aware retained-result verification and
   all-head publication. Do not replace current B/C interfaces with old code.
3. Reconcile `e242c846` with B's E-B1 repair; there must be one v2 wiring path,
   not two competing config grammars. Use the current sealed E plan and C
   plan by digest. The old cost pilot's C plan is not the production pin.
4. Run focused launcher, bridge, E-B review, optimizer-wrapper, and publication
   regressions under existing CPU admission. Retain exact commands and pins.
5. Exercise two real DOIN nodes in a bounded isolated smoke, with synthetic
   evaluators explicitly labelled as such, no production fit and no TEST.
   Verify population exchange, claim exclusion, terminal adoption and failure
   isolation. Two Bridge objects are not this smoke.
6. Verify real cross-host input/binding/receipt availability. Identical paths
   do not prove identical bytes. Read each required pin on the consuming host.

Deliver one pinned integration, test evidence and a machine-readable readiness
result to the deployment owner. Do not wait for the S3 fit to perform this work.

### B. Deployment owner: automatic start and successor scheduling

Prepare configuration and scripts concurrently with A; deploy only its passing
pin. Read `docs/i6n/swarm/README_B.md` and `INTERFACE_B.json`. Use existing CLI:

```bash
python -m tools.i6n_swarm_campaign plan --config <private.json> --output <campaign-root>
python -m tools.i6n_swarm_campaign preflight --root <campaign-root> --json
python -m tools.i6n_swarm_campaign install --root <campaign-root> --role coordinator
python -m tools.i6n_swarm_campaign start --root <campaign-root> --role coordinator
# After verified code, inputs and private configuration are deployed on a worker:
python -m tools.i6n_swarm_campaign install --root <campaign-root> --role secondary
python -m tools.i6n_swarm_campaign start --root <campaign-root> --role secondary
python -m tools.i6n_swarm_campaign status --root <campaign-root> --json
```

These are CLI templates, not permission to leave placeholders in the launcher.
Resolve paths and roles from the existing private inventory/configuration;
validate every substituted argument and write the effective private config.
Create one idempotent top-level deployment command with `--status`, using these
existing components, if the current integration does not already provide it.

- Start admitted small-class production on secondary as soon as integration
  and its workload envelope pass. Do not wait for S3 or all horizons to close.
- Preferred joins automatically after its S3 lease releases and its result is
  verified. Use a timer/state dependency, not an agent watching logs or a sleep
  loop in your conversation. If S3 fails, record the cause and keep independently
  admitted smaller workloads runnable; do not treat S3 failure as global pause.
- One GPU fit per host. Do not co-schedule the other GPU on preferred's host
  without measured aggregate RAM proof. Coordinator prioritizes services and
  bounded CPU work, not desktop-destabilizing training.
- Pilot peaks certify those exact workloads, not every model in a size class.
  Use the declared resource model plus measured coverage for class admission;
  preserve the 1.25 safety margin. Never lower a cap to evade refusal or retry
  an OOM. Check materialization and batch limits before any large fit, including
  the explicit S3 preflight/resource fix in `2df243fc`.
- The DAG must advance search -> full weekly annual evaluation -> supported
  multi-output arms automatically. Keep any genuinely missing arm capability
  explicitly pending; do not mark that milestone complete or block unrelated
  horizons. No waiting for two unrelated agent returns as an artificial gate.
- On reboot, reconcile leases and adopt verified terminals before new claims.
  No duplicate training; only the existing bounded recovery policy applies.

### C. Evidence owner: archive, OLAP, and reporting

1. Consume D's current integration. Verify durable off-chain payload storage,
   hash verification, chain reference ingestion and idempotent OLAP readback
   through the actual deployed path. Keep each stage's state separate.
2. Every horizon retains its own naive, rows, units, support and exclusions.
   Relative error is `(MAE_h - naive_h) / naive_h`; negative is better.
3. Do not claim blockchain finality from OLAP success: pilot records explicitly
   retain `chain_finalized=false`. Archive failures remain visible/retryable
   without refitting an already verified completed model.
4. Preserve local physical copies and the agreed second-host copy. Commit
   scripts, manifests and compact summaries, not large data/database binaries,
   credentials, private addresses or machine identifiers.
5. Ensure generated status distinguishes queued, claimed, actually updating,
   completed, archived and readback-verified. ETA must identify measured versus
   projected, expose its sample count, and remain unknown when unsupported.

## 3. Scientific invariants: no substitutions

Use the sealed business contract: short horizons 2/3/4/5/6/12 hours; long
18/24/36/48/60/72 hours; 20 RAW features with their declared selection history;
weekly refitting with four-year rolling training and purged inner evaluation;
all eligible validation weeks, not one week as a substitute. TEST stays closed.
Preserve calendar masks, weekend closure and entry restrictions; every naive
uses the same eligible rows as its horizon's model error. Do not upgrade a
nominal calendar to broker-verified without evidence.

Use DOIN/DEAP joint optimization of history, sampling and core architecture,
with the minimal seed and sealed search budgets. Preserve the current tick in
the window, separate input/output sampling, and use no smoothing. No NEAT,
replacement MLP, shortened training history or reduced annual evaluation.
Preserve configured early stopping on mean TRAIN/inner performance, restored
best weights, recorded seed and the no-unnecessary-repetitions policy.

## 4. Reports and stopping your interactive work

- Send one ACK naming owners and existing work reused; no request for another
  authorization for this already-approved integration and deployment.
- Send one startup return after the first real production optimizer updates
  and a successful heartbeat/lease verification. Name the active workload,
  node/host role, pins, seed, status path, OLAP state and honest remaining work.
  A generated config, active service or queued request is not a training start.
- Write `RETURN.md` and generated `STATUS.json` at the campaign root. Include
  the exact status command and pending DAG nodes with concrete reasons.
- Then end the agent turn. Do not periodically read logs, sleep/wake in chat,
  or spawn monitoring agents. Timers own progression and alerting. Musashi and
  the user can query status without becoming a prerequisite for the next task.
- Escalate only a concrete external requirement or a measured failure the
  implementation cannot resolve. Do not bypass tool/security denials, kill
  unrelated services, or silently weaken scientific/resource requirements.
