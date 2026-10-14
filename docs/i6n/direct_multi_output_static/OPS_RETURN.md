# I6-DMO ops lane return

Branch `satoshi/i6dmo-ops-20261014`, based on `2a440265`. Nothing has been installed, enabled or
started for this campaign. Deployment belongs to Satoshi's integration step.

## DMO traceability (ops-owned IDs)

Tests are under `tests/i6dmo/`. They use fake trainers and a fake warehouse, run on CPU with no GPU,
no network, no data and no TEST rows.

PRE: the tests were written first. All 27 failed at import because `tools.i6dmo.campaign` did not
exist yet. POST: 38 of 38 passed. That count covers the 29 DMO tests below plus 9 engineering tests.
The POST suite was also run once on a scratch merge of the ops, model, strategy and data branches,
and all 38 passed there too.

| DMO | Test(s) | PRE | POST |
|---|---|---|---|
| DMO07 | `test_ops_dmo07_units.py`: one fit per family; reports never fit; a report entry equal to the fit entry is refused | FAIL (import) | PASS |
| DMO11 | `test_ops_dmo11_identity.py`: an identical identity is adopted with no refit; a changed code, data or config cannot adopt; a tampered prior terminal is rejected; an interrupted attempt stays visible and is followed by RECOVERY | FAIL | PASS |
| DMO12 | `test_ops_dmo12_outbox.py`: retries and restarts produce no duplicate facts; a mismatched readback never closes delivery; a forged digest is refused; publish retries never retrain | FAIL | PASS |
| DMO13 | `test_ops_dmo13_test_closed.py`: init refuses an open-TEST design; init and deploy never open TEST, and a tampered seal is refused; readiness reports BLOCKED with the missing artifact and its owner; final units appear only when READY; PAIR_NOT_ELIGIBLE is terminal; the pair waits on owner rulings without failing | FAIL | PASS |
| DMO17 | `test_ops_dmo17_host_lease.py`: a heavy host lease blocks the 5070 Ti; a foreign heavy GPU process refuses admission; the 5090 is used first at a 7936 MiB cap; a refused cap is never shrunk and the unit routes to the secondary | FAIL | PASS |
| DMO19 | `test_ops_dmo19_status.py`: progress and heartbeat age come from the child; the in-flight ETA is only an upper bound; a stale or failed unit is never COMPLETE; the ETA becomes MEASURED only after a completed fit; close refuses an incomplete campaign | FAIL | PASS |
| DMO20 | `test_ops_dmo20_cadence.py`: the four cadences have distinct identities; a static terminal is not adoptable by a cadence campaign; a gate built from TEST reads is refused; selection ignores TEST terminals | FAIL | PASS |

The 9 engineering tests are in `test_ops_deploy_runner.py`. They cover: idempotent and foreign-safe
deploy; `--no-start` and `--render-to` never enabling anything; private bootstrap rendering; typed
preflight blockers; the conversion of the model lane's RECEIPT into a TERMINAL (exit codes 0, 3, 2
and 137, plus a bundle-pin mismatch); a missing hook or bundle producing WAIT_ARTIFACT without an
attempt; and verified, self-repairing mirrors. These tests were written after the code, so they
have no PRE record.

## Commands

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 300s -n i6dmo-ops -- python -m pytest -q tests/i6dmo/test_ops_*.py
```

The exact deploy sequence is in `INTERFACE_OPS.json` under `deploy_sequence`.

## Interface and integration gaps (precise owners)

- **Report unit:** `tools.i6dmo.reporting.run_unit` does not exist yet. Until it does, `short.report`
  stays in WAIT_ARTIFACT, and so the first fit runs but the report unit waits. Owner: the data lane
  (`naive.paired_metrics` plus `weekly_diagnostics` on `predictions/validation.npz`) or the Satoshi
  integration.
- **Pair unit:** `tools.i6dmo.strategy_adapter.run_unit` is missing; the pair unit should call
  `pair_eligibility` and then `run_backtest`. Owner: the strategy lane.
- **Owner rulings:** the pair unit is also gated by a typed `STRATEGY_OWNER_RULINGS` gate. It is
  shown as WAIT_OWNER_RULING, not as a failure.
- **Long design:** the model lane's `design.py` fixes the design to the short family. A long design
  therefore cannot be sealed until the model lane accepts it, and `long.fit` stays in WAIT_ARTIFACT.
- **Data bundles:** `data_pins.short_bundle` (BUNDLE.json relpath and sha256) must come from the
  data lane's `build_frozen_bundle`. Without it, `short.fit` is WAIT_ARTIFACT BUNDLE_PIN_MISSING.
  The final refits need `final_bundles` inside the DATA_PROVENANCE evidence.

## Risks

- The remote executor (`ssh <alias> systemd-run --user`, rsync) has not been exercised against a
  real host. Its re-attach logic is covered only for the local executor.
- `cgroup_memory_peak_bytes` may be null in the model receipt. In that case no measured peak
  exists and the cap stays at the declared initial value.
- No owner rulings have been recorded, so TEST cannot open.
- During a scratch CLI check I ran `deploy --no-start`. The old behaviour of that flag enabled the
  scratch campaign's timers. None of those units ever started. I disabled and removed them within
  the same minute and confirmed zero `i6dmo` units remain. The flag is now fixed: `--no-start`
  enables nothing, and the fix is covered by a test.
