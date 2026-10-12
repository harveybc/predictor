# Lane B: I6-N swarm campaign (execution and status)

Owner: lane B. Code: `tools/i6n_swarm/`, `tools/i6n_swarm_campaign.py` (predictor);
`src/doin_plugins/predictor/optimizer.py` (doin-plugins, `i6n_swarm` mode);
`src/doin_node/unified.py` + `cli.py` (doin-node, shared-population runtime).
Interface: `INTERFACE_B.json` (B.2). Nothing here has been installed or started
on any host; no fit has run under this code.

## What runs where

| Host role | Units (systemd **user**, all named `i6n-swarm-<campaign>-*`) | Does |
| --- | --- | --- |
| coordinator | `-ledger.service`, `-tick.timer` → `-tick.service` | Ledger authority over HTTP (bearer token, private bind); DAG tick every 5 min: lease recovery, successor activation, finalists, `status/STATUS.json` |
| each GPU worker | `-<role>-node.service`, `-<role>-agent.timer` → `-<role>-agent.service` | DOIN node running the shared wrapper in `i6n_swarm` mode; agent: boot recovery, healthcheck, A `freeze` units (CPU, data only), annual/multi-output fits, DOIN node-config sync |

Every unit carries `X-I6N-Swarm-Campaign=<id>`; `install` refuses to overwrite a
file of the same name without that marker and never stops, disables or edits
any other unit. Services use `Restart=on-failure`; timers are `Persistent`.
Start at boot needs user lingering (`preflight` reports it; it is not changed).

## Flow (DAG)

`freeze_folds:h` and `freeze_weeks:h` (A `freeze`, weeks with `--score-outer`)
→ `search:h` (DOIN swarm, 4 folds per candidate, 24 new candidates, population ≤ 8,
3 generations, patience 3, tolerance 1e-4, seed 6102026, minimization)
→ `annual:h` (BEST + MINIMAL_CONTROL or SMALL_ALTERNATIVE × every READY week,
weekly refit) → `multi_output:short|long` (needs an arms manifest; until then
`WAIT_DEPENDENCY MULTI_OUTPUT_ARMS_UNDEFINED`). A failed node blocks only its
dependants; the others keep running. Activation happens in the tick, not in a
conversation.

## Operator commands

```bash
# on the coordinator, from the predictor checkout (private config is never committed)
python -m tools.i6n_swarm_campaign plan --config <private.json> --output <campaign-root>
python -m tools.i6n_swarm_campaign preflight --root <campaign-root> --json
python -m tools.i6n_swarm_campaign install --root <campaign-root> --role coordinator
python -m tools.i6n_swarm_campaign start   --root <campaign-root> --role coordinator
# copy <campaign-root>/PLAN.json and private/ (0700) to the SAME absolute path on
# each worker (A's bindings and contexts carry absolute paths), then on that worker:
python -m tools.i6n_swarm_campaign install --root <campaign-root> --role preferred
python -m tools.i6n_swarm_campaign start   --root <campaign-root> --role preferred
# any time, anywhere the root is readable:
python -m tools.i6n_swarm_campaign status  --root <campaign-root> --json
python -m tools.i6n_swarm_campaign collect --root <campaign-root>
```

`start --role none` initializes the ledger and DAG without touching systemd.
`start` is idempotent (it starts only inactive campaign units).

## Status semantics

`status` reads `PLAN.json`, the ledger (SQLite read-only) and `status/TICK.json`;
it imports no TensorFlow and launches nothing. Levels: `swarm`, `horizon`
(one entry per DAG node), `candidate`, `fold`, `publication`. States: QUEUED,
RUNNING (with owner, host key, boot id, fencing token, lease expiry), COMPLETE,
WAIT_RESOURCE, INTERRUPTED, FAILED, WAIT_DEPENDENCY, each with a concrete
cause. ETA kinds: `measured` (A's epoch ETA of a running fold, or 0 for a
completed node), `projected` (remaining units × mean measured duration of
COMPLETE production attempts of that stage, p25/p75 interval, divided by hosts
currently running), `unknown` (no production measurement yet). Fixture
attempts (`cpu_fixture`) never feed an ETA.

## Recovery policy (ledger)

- A verified terminal receipt is adopted, never refit (any node, any directory).
- Lease expiry or a new boot id turns a RUNNING attempt into INTERRUPTED.
- Successor: RESUME only with a complete checkpoint (optimizer, rng, epoch,
  stopping); otherwise one linked RECOVERY attempt carrying A's parent STATUS
  digest. At most 3 physical attempts per fit_id. OOM, timeout and evaluator
  refusals are terminal FAILED: no retry, no lower cap, no CPU fallback.
- The runner renews the lease and stops its own child before the lease can
  expire if renewal fails (self-fencing); the orphan guard stops the child if
  the node process dies. A late receipt of the same fit is accepted only if no
  terminal exists; a later one is discarded, never published.
- The wrapper persists the search state of every generation (immutable) and
  seeds reproduction with 6102026 + generation + 1, so a restarted or peer node
  proposes the same offspring.

## Placement

`tools/i6n_swarm/placement.py`: 5090 preferred, one GPU job per host; 4090 only
with a fresh healthcheck of the current boot (no NV_ERR_NO_MEMORY, bounded
slab, passing probe); 4070 small pilots only with 8 GiB RAM and 2 GiB VRAM
left; 5070 Ti only with an aggregate proof from measured peaks and drained
under pressure. Caps come only from admitted pilot receipts (1.25 × peak).

## Known limits (see B_RETURN.md for the full list)

No DOIN node, unit or fit was started; E's `score_folds`, C's rollup and D's
publisher are not merged into this branch (typed WAIT_DEPENDENCY or a
same-signature reference in tests); cross-host recovery depends on A
accepting a parent STATUS digest that only the lost host holds.
