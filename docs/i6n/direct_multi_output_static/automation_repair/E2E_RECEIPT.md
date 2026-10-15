# Automation repair: one finite end-to-end receipt (CPU, 2026-10-15)

Machine receipt: `E2E_RECEIPT.json` in this directory
(sha256 `1b2b1ccc17fbd7791079eb77f2c7f4ee33a595ca42a07e80173fa01412fc6bd0`).

## Command

Run from the repository root at commit `5e9e6428`, with a clean `tools/i6dmo` and
`tests/i6dmo` (`worktree_dirty_under_tools_or_tests: false`). Earlier runs at `cec74baa` and
`332d50b8` also passed but did not have the timer-stop step; a run at `3ba826ae` (receipt `e2e17800…`) had it and also passed.

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-repair-e2e -- \
  python -m tests.i6dmo.e2e_automation_receipt --scratch "$SCRATCH" \
  --out docs/i6n/direct_multi_output_static/automation_repair/E2E_RECEIPT.json
```

`$SCRATCH` was a new throwaway directory under `/var/tmp`. It was not a live campaign.
Result: exit code `0`, verdict `PASS`, 45 s wall (41.9 s in-process).

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

All 10 units are this campaign's own, and every name carries its id `i6dmo-d562422a8500`:

| Unit | Action |
|---|---|
| `tick.timer`, `publish.timer`, `preferred-worker.timer`, `secondary-worker.timer`, `coordinator-worker.timer` | `systemctl --user disable --now <unit>`, rc 0 |
| `tick.service`, `preferred-worker.service`, `secondary-worker.service`, `coordinator-worker.service` | `systemctl --user stop <unit>`, rc 0 |
| `publish.service` | `SELF_NOT_STOPPED`: it invoked the close; the oneshot exits by itself and its timer is disabled |

## Hashes

| Item | sha256 |
|---|---|
| `CLOSE.json` | `abc0a9c42fa8f659c795aef55af414b85aa5ca23e67165e7eb68df339bc5f1c4` |
| `TIMERS_STOPPED.json` | `a9a617f1dd03877e4fa78a2350367aa7b5414129aebb888178e98969df094850` (its `close_sha256` equals the above) |
| Publication (closure input) | `8c4c33a52aff696d901fac7d63f5ba5961f55b8f8efe9b8016c1b575ea7ef7b0` |
| Mirrors (closure input) | `4e92508891c1ecd0e9d2489cfbc1479eabe159aa44c3a677fd48569b8d15cbf2` |
| Outbox readback (closure input) | `ecec45bcb7d0245d98b09cc96ff94576a10e030898453b279d4853007b1835e1` |
| Reports (closure input) | `aa99b3f5091f9ea9c9cdad4198a10a25419b51f46ef4a8d7a41f62de1680747e` |
| Root tree after CLOSE and stop | `c0540163ba83596a331aa628c3d9dafcc53522b6332ba7ffd78e796caf256d73` |

## Scope

- The scratch root was a fixture campaign, `i6dmo-d562422a8500`.
- The closed `i6dmo-9e1ebad89ff2` and the retained `i6dmo-72809b6c88a6` were not touched.
- The real user systemd was never called; the runner was an injected recorder.
- No GPU was used and no scientific fit ran.
