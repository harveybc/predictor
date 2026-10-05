# Phase-1 inventory orchestrator

This component drains the canonical feature inventory without per-column code.
It is intentionally CPU-only and cannot authorize phase 2.

## Boundary

The orchestrator owns lifecycle concerns:

- freezes the exact inventory denominator and row identities;
- estimates cost from row and byte counts;
- assigns the smallest series to the small host and balances the rest across
  the large hosts;
- permits one series process per host through atomic claims;
- recovers abandoned claims after their heartbeat becomes stale;
- retains one immutable terminal per feature;
- submits each worker envelope to the configured warehouse API;
- exposes read-only JSON status and ETA;
- invokes the population finalizer only after every inventory member has a
  closure-eligible `COMPLETED` or explicit `UNAVAILABLE` terminal;
- retains every failed attempt and retries it only up to the configured bound;
- requires warehouse receipts and an authenticated full-population readback;
- creates `PHASE_1_COMPLETE.json` only when the finalizer accepts the complete
  denominator.

The configured worker owns scientific calculations. It may wrap the existing
PS1 profiler and causal-ladder implementation. Population-wide correction,
including BH/FDR, belongs in the configured finalizer because it cannot be
decided honestly from one column in isolation.

## Independent populations

The EURUSD template reads the three retained
`laneA/batch_*/admissible_features.csv` files and includes only rows where
`role=feature` and `admissibility=ADMISSIBLE`. That filter yields exactly 366
unique features: 46 in batch 1, 298 in batch 2 and 22 in batch 3. A different
count is refused before any worker starts.

ETH is a separate 83-feature population with its own config, plan, state root,
target pack, terminals, receipts, reconciliation and completion gate. EURUSD
and ETH must never be combined into a 449-row denominator or share a state
root. The ETH template intentionally leaves its governed inventory path as a
deployment placeholder rather than pretending that the EURUSD files contain
ETH.

Machine addresses, credentials and deployment paths do not belong in this
public repository. Copy `canonical_config.template.json` (EURUSD) or
`eth_config.template.json` outside the checkout,
replace the bracketed host and path placeholders, and keep API credentials in
the environment variable named by `warehouse.token_env`.

## Worker contract

The only supported transport is `stdio-json-v1`. The coordinator sends one
`phase1.column_request.v1` JSON document on standard input and requires exactly
one canonical `phase1.column_result.v1` JSON document on standard output. This
works unchanged through an SSH command prefix: no coordinator-local input or
output path is passed to the remote process.

Every command token is passed without a shell. Only identity placeholders are
available:

| Placeholder | Meaning |
|---|---|
| `{feature_id}` | Canonical feature ID |
| `{feature_key}` | SHA-256 key used for paths |
| `{host_id}` | Logical assigned host |
| `{population_id}` | Independent population identity |
| `{target_pack}` | Target contract for that population |

The worker must print:

```json
{
  "schema": "phase1.column_result.v1",
  "feature_id": "fx.example.logret_1h",
  "state": "COMPLETED",
  "envelope": {
    "schema": "feature_selection_envelope.v1"
  }
}
```

`FAILED`, timeout, OOM, admission refusal and `rc75` are attempts, not science.
They are retained under `attempts/<feature-key>/`, retried up to
`retries.max_attempts`, and then recorded under `failures/`; they can never
satisfy closure. Only `COMPLETED` and an explicit `UNAVAILABLE` result produce
closure terminals. Every subprocess receives `CUDA_VISIBLE_DEVICES=""`,
`NVIDIA_VISIBLE_DEVICES=void`,
`PHASE1_ONLY=1`, and `PHASE2_FORBIDDEN=1`.

## Finalizer contract

The finalizer receives the terminal directory and must validate the complete
population, apply global corrections and write:

```json
{
  "schema": "phase1.finalizer_result.v1",
  "state": "PHASE_1_COMPLETE",
  "terminal_count": 366
}
```

Anything else leaves the phase-2 gate closed. Consumers must call
`gate-phase2`; checking that a file merely exists is not sufficient because the
gate digest, plan identity, terminal set, per-envelope warehouse receipts and
authenticated warehouse reconciliation are recomputed.

The submission endpoint must return a JSON receipt with `accepted: true`
(directly, or in the body of the HTTP adapter response). A transport success or
an explicit rejection is not a receipt. The reconciliation endpoint must bind
the complete expected identity set, count, plan and configured authentication
profile and return its canonical digest.

## Commands

```bash
python tools/phase1_inventory_orchestrator.py --config <deployment.json> plan
python tools/phase1_inventory_orchestrator.py --config <deployment.json> run-host --host <logical-host>
python tools/phase1_inventory_orchestrator.py --config <deployment.json> run
python tools/phase1_inventory_orchestrator.py --config <deployment.json> finalize
python tools/phase1_inventory_orchestrator.py --config <deployment.json> gate-phase2
python tools/phase1_inventory_status.py --config <deployment.json>
```

`run-host` executes at most one series. `run` drains all declared hosts in
parallel while preserving the one-process-per-host rule. Re-running either is
idempotent: existing terminals are not recomputed, active claims are not
stolen, stale claims are archived, missing warehouse receipts are retried and
failed attempts remain available for diagnosis.

## Status

The status document reports `total`, `completed`, `failed`, `unavailable`,
`pending`, `running`, `current` by host, `closure_eligible`, `eta_seconds`,
warehouse receipts, warehouse reconciliation and integrity errors. Its
denominator must reconcile before it can be used for an operational progress
display.

No live jobs or services are started by installation or import.
