# I6-N swarm deployment (deployment owner, orders 2026-10-12 section 2.B)

One idempotent top-level command, `tools/i6n_swarm_deploy.py`, drives lane B's
existing campaign CLI. The private deploy spec is never committed; hosts,
addresses, tokens and absolute home paths stay out of git.

```bash
# from a checkout of the deploy branch, on the coordinator
python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json> --dry-run    # derive + validate the config only
python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json> --replicate  # copy + sha256-verify on workers
python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json>              # full idempotent deploy
python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json> --status     # read-only
python -m tools.i6n_swarm.crosshost_verify --pin <predictor deploy commit> # integration owner's byte check
```

## What the full deploy does

1. **Code export:** exports the pinned trees (predictor deploy commit,
   doin-core, doin-node, doin-plugins, data-lake, data-warehouse) with
   `git archive` into one code directory, plus a sha256 manifest. Each host
   gets a python wrapper that puts the pinned trees first on PYTHONPATH (so
   live editable installs are not used) and sets `I6N_DATA_ROOT`.
2. **Effective private campaign config** (lane B schema), derived from the
   private worker inventory, the pilot receipts and C's sealed plan. Every
   substituted argument is validated: no placeholder, pin digests, ports,
   seed and mode.
   - Capabilities: VALIDATION_OUTER_WEEK and DIRECT_MULTI_OUTPUT_ARM.
   - Class admission requires a COMPLETED pilot at the class corner (samples
     and width maxima) and every reachable workload inside the cell guards.
   - Caps are `ceil(1.25 x (measured peak + declared coverage adjustment))`.
     The adjustment is the largest positive declared materialization
     (rows x row bytes) above the pilot's own, over C's sealed support. Its
     derivation is pinned by digest.
3. **Coordinator:** lane B's `plan`, then `preflight` (which must be READY),
   then `install` and `start` for the coordinator (ledger service and DAG tick
   timer).
4. **Each worker, secondary first:**
   - copy the code, data root, campaign root and ledger token to the same
     absolute paths with a single rsync;
   - verify every byte remotely with `sha256sum -c`; any difference refuses
     the step;
   - make the predictor commit present in the worker's own repo as a git
     bundle (refs only, no checkout change);
   - `install` the worker's units.
5. **Start policy:** the secondary starts at once. The preferred is installed,
   and a coordinator timer (`i6n-swarm-<id>-join-preferred.timer`, every
   5 min) runs `--join preferred`:
   - It waits while the running pilot unit is active.
   - If the pilot COMPLETED, it starts the preferred only after the pilot's
     RESULT sha256 is verified on that host.
   - If the pilot FAILED, it records the cause and starts the preferred anyway
     for its admitted smaller classes.
   - It never stops, edits or duplicates the pilot.

## Invariants kept

- One GPU fit per host. The coordinator runs services and bounded CPU only,
  and its environment hides the GPU.
- No cap is lowered and no OOM is retried.
- No TEST data is used.
- The DAG advances in the coordinator tick, not in a conversation.
