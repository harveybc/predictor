# Automation repair: one finite end-to-end receipt (CPU, 2026-10-15)

Machine receipt: `E2E_RECEIPT.json` in this directory
(sha256 `8d000fdce8aff8a164d14a1cb0aa4f4dd7798fc7f4c68ae1fc590c50cdbf55c3`).

## Command

Run from the repository root at commit `332d50b8` (final repair + matched-runner tip; first run at `cec74baa` also PASS), with a clean `tools/i6dmo` and
`tests/i6dmo` (the receipt records `worktree_dirty_under_tools_or_tests: false`):

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-repair-e2e -- \
  python -m tests.i6dmo.e2e_automation_receipt --scratch "$SCRATCH" \
  --out docs/i6n/direct_multi_output_static/automation_repair/E2E_RECEIPT.json
```

`$SCRATCH` was a new throwaway directory under `/var/tmp`. It was not a live campaign.
Result: exit code `0`, verdict `PASS`, 40 s wall (38.4 s in-process).

## What ran

| Step | Code path | Result |
|---|---|---|
| Designs | Real sealed MG2/MG12 designs | Seals `4b64b4ee…` / `97541764…`, unchanged |
| Data | Real market bundles built by the data lane from the synthetic fixture | No TEST row |
| Fits | Prediction fixture; it trains nothing | Exactly 2 invocations: `short.fit`, `long.fit` |
| Reports | Real worker → governed `unit_runner --kind report` → `reporting.run_unit` with the default loader | COMPLETE ×2 |
| pair.eligibility | Real worker → `unit_runner --kind pair` → stored-terminal verification | `PAIR_NOT_ELIGIBLE`: short heads 2,3,4,5,6,12 eligible, long heads none, `LONG_ELIGIBLE_HEADS_0_LT_2` |
| strategy.validation | Tick gate | BLOCKED / `PAIR_NOT_ELIGIBLE`, never claimed |
| Strategy OHLC pins | — | Absent: `strategy_pins_present: false` |
| Publication | Outbox → local test sink `FakeWarehouse` (not the production OLAP) | 109/109 reports, 3991 metric rows, exact readback, 0 pending, 0 mismatch |
| Mirrors | Two scratch mirror directories | VERIFIED, 40 files each, `CLOSE.json` identical, snapshots present |
| CLOSE | Automatic, inside `publish` | `trigger: AUTOMATIC`, disposition `PAIR_NOT_ELIGIBLE` |
| Idempotence | Re-tick, re-publish and a worker run after CLOSE | `ALREADY_CLOSED` / `ALREADY_CLOSED_AND_MIRRORED` / `CLOSED`; all 89 files unchanged |

## Hashes

| Item | sha256 |
|---|---|
| `CLOSE.json` | `c6d2d3180240112fa6560f42bf441e6e7c5dbf2e737e0d976310b42cd257bc11` |
| Publication (closure input) | `8c4c33a52aff696d901fac7d63f5ba5961f55b8f8efe9b8016c1b575ea7ef7b0` |
| Mirrors (closure input) | `5829b35c0c69e4d4eba95981889c08bf8d7ff22a030c52a1478d1dcb856e54d8` |
| Outbox readback (closure input) | `a713042e90c50ebabd4974a3597b506ca80c4eba48a5e0cd5cf9e5cc015af6cd` |
| Reports (closure input) | `390e27f63d7dafe6e5961b1a53f68e40c345c936a2d819192c682221a6881daa` |
| Root tree after CLOSE | `5148bdbc75145d2c99fdafa7416b6641b681ccb61fb7d795a4b518c9685bae1c` |

The scratch root was a fixture campaign, `i6dmo-d6924e1b79bb`. The closed
`i6dmo-9e1ebad89ff2` and the retained `i6dmo-72809b6c88a6` were not touched. No timer or service
was installed or enabled. No GPU was used and no scientific fit ran.
