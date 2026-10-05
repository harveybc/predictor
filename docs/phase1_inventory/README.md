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
  `COMPLETED`, `FAILED`, or `UNAVAILABLE` terminal;
- creates `PHASE_1_COMPLETE.json` only when the finalizer accepts the complete
  denominator.

The configured worker owns scientific calculations. It may wrap the existing
PS1 profiler and causal-ladder implementation. Population-wide correction,
including BH/FDR, belongs in the configured finalizer because it cannot be
decided honestly from one column in isolation.

## Canonical denominator

The checked template reads the three retained
`laneA/batch_*/admissible_features.csv` files and includes only rows where
`role=feature` and `admissibility=ADMISSIBLE`. That filter yields exactly 366
unique features: 46 in batch 1, 298 in batch 2 and 22 in batch 3. A different
count is refused before any worker starts.

Machine addresses, credentials and deployment paths do not belong in this
public repository. Copy `canonical_config.template.json` outside the checkout,
replace the bracketed host and path placeholders, and keep API credentials in
the environment variable named by `warehouse.token_env`.

## Worker contract

Every command token is passed without a shell. These placeholders are
available:

| Placeholder | Meaning |
|---|---|
| `{feature_id}` | Canonical feature ID |
| `{feature_key}` | SHA-256 key used for paths |
| `{host_id}` | Logical assigned host |
| `{inventory_row_path}` | JSON copy of the authenticated inventory row |
| `{worker_output_path}` | Unique output path for this attempt |
| `{state_root}` | Shared orchestration state |
| `{plan_path}` | Frozen `PLAN.json` |

The worker must write:

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

`FAILED` and `UNAVAILABLE` are valid explicit terminals. A configured
unavailable exit code also creates an `UNAVAILABLE` terminal. Every subprocess
receives `CUDA_VISIBLE_DEVICES=""`, `NVIDIA_VISIBLE_DEVICES=void`,
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
gate digest, plan identity and terminal-set identity are recomputed.

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
stolen, stale claims are archived, and missing warehouse receipts are retried.

## Status

The status document reports `total`, `completed`, `failed`, `unavailable`,
`pending`, `running`, `current` by host, `eta_seconds`, warehouse receipts and
integrity errors. Its denominator must reconcile before it can be used for an
operational progress display.

No live jobs or services are started by installation or import.
