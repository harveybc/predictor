# Automation repair: one finite end-to-end receipt (CPU, 2026-10-15)

Machine receipt: `E2E_RECEIPT.json` in this directory
(sha256 `b78728f51a2444c4f20ccd82085daac232f69f787fa7393bb8443ab5fa3db5d3`).

## Command

Run from the repository root at commit `cec74baa`, with a clean `tools/i6dmo` and
`tests/i6dmo` (the receipt records `worktree_dirty_under_tools_or_tests: false`):

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-repair-e2e -- \
  python -m tests.i6dmo.e2e_automation_receipt --scratch "$SCRATCH" \
  --out docs/i6n/direct_multi_output_static/automation_repair/E2E_RECEIPT.json
```

`$SCRATCH` was a new throwaway directory under `/var/tmp`. It was not a live campaign.
Result: exit code `0`, verdict `PASS`, 44 s wall.

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
| `CLOSE.json` | `18f5c6600df8c3ea684727243acbde345113ac82536f512c0e22314c8769cb7a` |
| Publication (closure input) | `8c4c33a52aff696d901fac7d63f5ba5961f55b8f8efe9b8016c1b575ea7ef7b0` |
| Mirrors (closure input) | `ac423c2884eb02942df1664d91e135dbede5c600aa33e27d3f0f6a2a3ec0f9a1` |
| Outbox readback (closure input) | `a824e1a52ab319c775665052cfcee1ada4230b32c90db51bce49539647c6356d` |
| Reports (closure input) | `9a584bfe8bdd2e9298c79647544d12683ff3fa12fd8e89d334b934d610d1e26c` |
| Root tree after CLOSE | `eb1c8dfa330908fb13d328dcfd029f16713ed2b7ab8bb56f4318452c7fad22cc` |

The scratch root was a fixture campaign, `i6dmo-8c5badd0cb19`. The closed
`i6dmo-9e1ebad89ff2` and the retained `i6dmo-72809b6c88a6` were not touched. No timer or service
was installed or enabled. No GPU was used and no scientific fit ran.
