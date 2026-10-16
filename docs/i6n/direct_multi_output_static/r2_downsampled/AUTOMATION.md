# Finite R0/R2 Downsampled Sequence

## Operator Entry Point

Use an isolated checkout of the accepted release and the coordinator Python.
The private spec is not committed. It contains exactly:

```json
{
  "schema": "i6dmo.r2_downsampled_sequence.v1",
  "work_root": "/absolute/new/sequence-root",
  "source_deployment": "/absolute/existing/private-deployment.json",
  "short_source_campaign": "/absolute/original/r2-source-campaign",
  "short_source_pins": {
    "effective_sha256": "<SOURCE_PINS.json pins.effective_sha256>",
    "plan_sha256": "<SOURCE_PINS.json pins.plan_sha256>",
    "code_commit": "<SOURCE_PINS.json pins.code_commit>"
  },
  "long_source_work_root": "/absolute/closed/raw-long-work-root",
  "input_dir": "/absolute/retained/input-directory"
}
```

The deployment file supplies existing role aliases, pinned interpreters,
storage roots, mirrors and warehouse EnvironmentFile. Never print credentials.
The original source campaign must be quiescent, with its immutable queue and
outbox backups, publication readback and two original physical custody copies.
Do not create a fictional CLOSE for its failed R2 unit.

```bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2
python -m tools.i6dmo.r2_downsampled_campaign install --spec /absolute/private-spec.json
python -m tools.i6dmo.r2_downsampled_campaign run --spec /absolute/private-spec.json
```

`install` arms one bounded continuation timer. `run` is idempotent and starts
the next eligible family; subsequent progress needs no chat approval or agent
polling. A clean committed checkout, hash-bound independent release acceptance,
fresh resource/runtime checks and authenticated source custody are mandatory.
Neither command bypasses admission, grants credentials or modifies DOIN.

## Status At Any Level

```bash
python -m tools.i6dmo.r2_downsampled_campaign status --root /absolute/sequence-root
python -m tools.i6dmo r2 status --root /absolute/sequence-root/short/campaign --level campaign
python -m tools.i6dmo r2 status --root /absolute/sequence-root/short/campaign --level stage
python -m tools.i6dmo r2 status --root /absolute/sequence-root/short/campaign --level unit
python -m tools.i6dmo r2 status --root /absolute/sequence-root/short/campaign --level publication
python -m tools.i6dmo r2 status --root /absolute/sequence-root/long/campaign --level campaign
```

Query a family only after its PLAN exists. Status reads bounded generated state;
it does not tail logs or launch work. ETA is unmeasured until that phase has a
completed compatible timing; an epoch-ceiling estimate is an upper bound, not
a promise that early stopping will reach that ceiling.

## Sequence And Recovery

1. Authenticate original short source and both custody copies, retain remote
   SHA readback, seal recovery and adopt the 19 original donor identities.
2. Fit the compatible fresh R0 reference only when its existing strict reuse
   predicate requires it; fit R2 once with transported authenticated donors.
   Both predictive fits use the same preferred GPU class. No short donor fit
   may fall back from ADOPT_ONLY to training.
3. Score complete paired DEVELOPMENT rows, report own naive per head, import,
   cost and adoption facts, read back OLAP, verify both mirrors and CLOSE.
4. Only after that valid CLOSE, prepare long MG12m, pretrain its 19 own branch
   donors on FIT/purged INNER, fit R0/R2, report and close identically. Independent
   long donors can use the secondary GPU; do not run the auxiliary preferred-host
   GPU concurrently.
5. Stop only this sequence's timer and each campaign's own units at terminal.

A scientific negative advances to long. A technical failure is never relabelled
as a scientific negative or silently retrained. Resource waits retain their
cause; a changed source/release/configuration is refused. Restart resumes the
sealed queue and verified terminals. Publication retries do not perform fits.

Mutable SQLite/WAL, locks and Git worktrees remain on a local POSIX filesystem.
Large immutable outputs use the configured storage adapter. Git contains code,
contracts, digests and small evidence, never input arrays or model weights.
The original short donor cost remains in its source campaign; adoption is a
separate count/identity fact, not newly performed pretraining.

This release does not start quarterly retraining, optimization, core
pretraining, trading, RL, NEAT or TEST. Those require their separately frozen
subplans. Unknown source publication clocks, assumed calendar and strategy
costs remain explicit limitations.
