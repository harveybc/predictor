# Requirement To Evidence

| Contract | Evidence | Acceptance |
|---|---|---|
| WC01 | source_identity, authenticate, corrupted-terminal test | fixed parent and pristine trainer; no TEST |
| WC02 | window_arrays, future perturbation and closure/missing-row tests | t included; no >t; actual missing rows excluded |
| WC03 | prepare, GPU UNPAIRED_PERIOD_DATA check | identical origin masks, truth and shared scales across all arms |
| WC04 | per-period cached naive, worker recomputation, annual support hashes | own horizon; paired same units and full-year sums |
| WC05 | frozen sources and exact catalogue test; GPU initialization proofs | all window-only arms 1901 params with same seeded weights |
| WC06 | reused weekly worker, terminal identity, conditional disposition tests | finite queue; no refit after interruption; bounded resources |
| WC07 | full-year ranking, missing/failed-year tests, multiarm registry test | no partial winner or mixed recipe; readback and mirrors |

The host-specific runtime, preparation manifest, measured preflight and live
STATUS remain outside git. Inspect one generated status, not an agent loop.
No result of a scientific fit is claimed by these engineering tests.

## Operator Commands

Set SPEC to the private JSON path allocated by the operator. Code and configs
are versioned; data, tokens and host addresses never belong in this document.

```bash
python -B -m tools.i6dmo.weekly_window_core prepare --spec "$SPEC"
python -B -m tools.i6dmo.weekly_window_core launch --spec "$SPEC"
python -B -m tools.i6dmo.weekly_window_core status --spec "$RUN_SPEC"
```

Launch performs the actual-device fit-free checks and installs finite worker
and publisher timers. Its effective RUN.private.json is RUN_SPEC. Do not run
the preparation, launch or status commands repeatedly to monitor training.
Timers collect status, complete the queue, publish/read back, verify two
mirrors and shut down only their own campaign. If a fit is interrupted, retain
FAILED_INTERRUPTED; never silently retry or lower a refused memory ceiling.

Paper/demo deployment is separately verified integration, not a consequence
of a forecasting unit-test pass. The owner's rule requires net-positive
simulation, configured costs, matched predictions and automated demo-account
and endpoint verification. TEST is not its gate; real money is forbidden.
