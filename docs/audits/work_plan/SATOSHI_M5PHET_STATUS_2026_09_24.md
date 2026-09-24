# M5PHET implementation: delivery status

Assignment [SATOSHI_M5PHET_IMPLEMENTATION_2026_09_24](../../handoffs/SATOSHI_M5PHET_IMPLEMENTATION_2026_09_24.md) at
`27868f5d`, against design revision `4508b98`. Implementation branch `satoshi/runtime-p02-20260924` in
[M5PHET](https://github.com/harveybc/M5PHET), commit `94562ed`.

## Running work reconciled first, as the orders require

| job | outcome |
|---|---|
| RP135 protocol B closure | **FINISHED**, twelve of twelve cells verified, every replay bit-exact, 2,156 CPU s; the continuation allocation stands at 16,167 of 24,000 |
| R0/R1/R2 development contrast | **RUNNING** on WORKER_A, 2,684 CPU s of its 14,400, seed 2021 in progress |

Nothing completed in RP144–RP151 was restarted and no sealed configuration was touched.

## Per delivery

| delivery | state | what it needs |
|---|---|---|
| 1. M5PHET runtime (P02) | **IMPLEMENTED and VERIFIED** | 34 new behavioral tests, 66 in the suite, PRE failure frozen |
| 2. Typed decisions beyond choice | NOT STARTED | the news-signal adapter and the pinned Laya SDK |
| 3. Economic calendar (CAL01–CAL12) | NOT STARTED | the governed resource mapped first |
| 4. Domain providers | NOT STARTED | thin adapters over existing engines |
| 5. Composition and packaging | NOT STARTED | depends on 2 to 4 |

## What delivery 1 actually establishes

The tests were written **before** the implementation, from `INTERFACES.md` and `IMPLEMENTATION_PLAN.md`, and their
pre-implementation failure is frozen beside them. Each one states a behaviour a caller can observe:

- **A refusal happens before the model is loaded.** The provider double records every call it receives, so the tests assert
  that `load()` was never reached for an unsupported family, an unsupported output kind, an operation the provider does not
  offer, an unknown fitted state or a missing one. `infer` without a fitted state is MODEL_NOT_FITTED, never implicit training.
- **Inference cannot move the fitted state.** The provider is handed a copy and mutates it on purpose; the binding's state
  digest is unchanged on the next run.
- **Nothing collapses into one VERIFIED boolean.** The envelope carries schema validity, capability checking, replay,
  calibration binding, governance acceptance and application eligibility as six separate facts, and
  `execution_authorized` is false in the payload itself.
- **A refused output carries no invented score.** An ABSTAINED answer's payload is dropped. An omitted question is
  INVALID_INPUT and never counts as success; only an explicitly declared `partial_results` turns a mixed result into PARTIAL.
  An answer to a question nobody asked, and an uncertainty method the provider never declared, are both invalid. A provider
  that raises is reported with what it said rather than replaced by a substitute result.
- **External providers register through the designed `m5phet.providers` entry-point group.** Discovery reports a provider
  whose import fails by name and keeps the others; a second provider claiming a registered name is refused rather than
  replacing the incumbent.

What it does **not** establish: no real engine has been wired to this runtime yet, so no family beyond the shipped
classification contract is implemented, and passing these tests says nothing about calibration, causality or profit.

## Test command

```
pip install -e . && python -m pytest tests -q        # 66 passed
```
