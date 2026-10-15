# Automation repair: one finite end-to-end receipt (CPU, 2026-10-15)

Machine receipt: `E2E_RECEIPT.json` in this directory
(sha256 `e2e1780068564827b30adcf9c3b40108a86e56142006c3a9959f144bd63b8a7e`).

## Command

Run from the repository root at commit `3ba826ae`, with a clean `tools/i6dmo` and
`tests/i6dmo` (`worktree_dirty_under_tools_or_tests: false`). Earlier runs at `cec74baa` and
`332d50b8` also passed; they did not have the timer-stop step.

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-repair-e2e -- \
  python -m tests.i6dmo.e2e_automation_receipt --scratch "$SCRATCH" \
  --out docs/i6n/direct_multi_output_static/automation_repair/E2E_RECEIPT.json
```

`$SCRATCH` was a new throwaway directory under `/var/tmp`. It was not a live campaign.
Result: exit code `0`, verdict `PASS`, 40 s wall (38.9 s in-process).

## What ran

| Step | Code path | Result |
|---|---|---|
| Designs | Real sealed MG2/MG12 designs | Seals `4b64b4ee…` / `97541764…`, unchanged |
| Data | Real market bundles built by the data lane from the synthetic fixture | No TEST row |
| Deployment record | `deploy --no-start`, units rendered into `$SCRATCH/systemd-units` | Injected recorder, nothing enabled |
| Fits | Prediction fixture; it trains nothing | Exactly 2 invocations: `short.fit`, `long.fit` |
| Reports | Real worker → governed `unit_runner --kind report` → `reporting.run_unit` with the default loader | COMPLETE ×2 |
| pair.eligibility | Real worker → `unit_runner --kind pair` → stored-terminal verification | `PAIR_NOT_ELIGIBLE`: short heads 2,3,4,5,6,12 eligible, long heads none |
| strategy.validation | Tick gate | BLOCKED / `PAIR_NOT_ELIGIBLE`, never claimed |
| Strategy OHLC pins | — | Absent |
| Publication | Outbox → local test sink `FakeWarehouse` (not the production OLAP) | 109/109 reports, 3991 metric rows, exact readback, 0 pending, 0 mismatch |
| Mirrors | Two scratch mirror directories | VERIFIED; CLOSE, snapshots and timer receipt all mirrored |
| CLOSE | Automatic, inside `publish` | `trigger: AUTOMATIC`, disposition `PAIR_NOT_ELIGIBLE` |
| Timer stop | This campaign's units from its own `DEPLOY.json`, through an injected `systemctl --user` recorder | `STOPPED`, trigger `CLOSED:PAIR_NOT_ELIGIBLE` |
| Idempotence | Re-tick, re-publish and a worker run after CLOSE | `ALREADY_CLOSED` / `ALREADY_CLOSED_AND_MIRRORED` / `CLOSED`; all 91 files unchanged; 0 further systemctl calls |

## Units stopped

All 10 units are this campaign's own, and every name carries its id `i6dmo-0108077fb798`:

| Unit | Action |
|---|---|
| `tick.timer`, `publish.timer`, `preferred-worker.timer`, `secondary-worker.timer`, `coordinator-worker.timer` | `systemctl --user disable --now <unit>`, rc 0 |
| `tick.service`, `preferred-worker.service`, `secondary-worker.service`, `coordinator-worker.service` | `systemctl --user stop <unit>`, rc 0 |
| `publish.service` | `SELF_NOT_STOPPED`: it invoked the close; the oneshot exits by itself and its timer is disabled |

## Hashes

| Item | sha256 |
|---|---|
| `CLOSE.json` | `88cf4553194c7290a97127315101b27138d9c2fde2b4e731467d29408a7f723e` |
| `TIMERS_STOPPED.json` | `e0953686339d31eb6d435b8100b1809a2062c8584fbe0524e4617df7aae086bb` (its `close_sha256` equals the above) |
| Publication (closure input) | `8c4c33a52aff696d901fac7d63f5ba5961f55b8f8efe9b8016c1b575ea7ef7b0` |
| Mirrors (closure input) | `d51eaf211a6cc61e1b0b2f3afdf15848e779d93e5852fc023f8dee2b33b1dd56` |
| Outbox readback (closure input) | `812b9adf1cc3c6bfa52e59737b864a14ed4f86184066440bb2635982f130f68d` |
| Reports (closure input) | `dc284d41cdbf5d202d71e91c23cd3bc568cff5ed84f0d8bc9b4eccb5579b7b75` |
| Root tree after CLOSE and stop | `7912f628eb92faaab690998915d7cf2f08f7984698fb24b1ba2d5b4010194579` |

## Scope

- The scratch root was a fixture campaign, `i6dmo-0108077fb798`.
- The closed `i6dmo-9e1ebad89ff2` and the retained `i6dmo-72809b6c88a6` were not touched.
- The real user systemd was never called; the runner was an injected recorder.
- No GPU was used and no scientific fit ran.
