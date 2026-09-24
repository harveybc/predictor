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
| 1. M5PHET runtime (P02) | **IMPLEMENTED and VERIFIED** | 34 behavioral tests, PRE failure frozen |
| 1b. Local-first evidence and optional OLAP (INT01, INT03, INT05, INT11, INT12) | **IMPLEMENTED and VERIFIED** | 9 behavioral tests, PRE failure frozen, 75 in the package suite |
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


## Update, 2026-09-24 04:35Z, after `fd437d6` and `d46d3363`

**The contrast was corrected and restarted.** Musashi's F1 to F4 were reproduced through the real runner first and their nine
failing tests are frozen. The previous run persisted only the initial weights and the donor, so it could not supply a
replayable checkpoint comparison: it is classified a **diagnostic at its actual scope**, preserved unchanged under
`ecl-modular-contrast-20260924-diagnostic-superseded`, and was stopped at a seed boundary rather than spending the rest of the
allocation on further non-replayable successors. The corrected successor started at 04:30:46Z from `f777698f`, with checkpoint
selection and restoration, durable per-cell models, observed optimizer updates, CPU and wall tracked apart, an allocation that
refuses rather than rounds up, and every completed cell persisted immediately.

**INT delivery.** The local-first evidence layer is implemented on the same M5PHET branch at `adba176`: atomic manifest,
durable attempt and metric records with unique event identities, typed metrics that state unit, scale, aggregation and
population, artifact references, interrupted-write recovery, retry as a new attempt, a distinct governed authority whose
denied delivery refuses before any work with no local fallback, an optional rebuildable DuckDB projection that the core path
never imports, and a historical import that creates no authority it never had.

| INT | state |
|---|---|
| INT01, INT03, INT05, INT11, INT12 | **IMPLEMENTED and VERIFIED** |
| INT02, INT04 | NOT STARTED: need the governed adapter against the real services |
| INT06 to INT10 | NOT STARTED: need the DOIN plugin round trip and the remaining families |

## Update, 2026-09-24 05:20Z, after the integration review `5cb79c06`

**PRE and POST.** The review's probe was run unchanged first and all twelve counterexamples reproduced
([PRE](../evidence/d3_k5_20260917/RP153/PRE_probe_output.json)). Each scenario is now a rejection or recovery regression, and
the probe was left unedited, so running it against the repaired code stops at its first scenario because the API refuses where
it used to record a value ([POST and the scenario map](../evidence/d3_k5_20260917/RP153/README.md)).

| finding | repaired |
|---|---|
| 1, resume misattributes work and loses attempts | identity and profile validated on resume; open attempts derived from durable events; interior corruption quarantined and refused, only a torn tail dropped |
| 2, governed authority from truthiness | an accepted delivery is a validated contract, and a profile is not changed by resuming in either direction |
| 3, the runtime certifies what it has not checked | a null payload under OK is invalid, an unanswerable request refuses before load and infer, a foreign or future calibration does not bind, and a provider exception keeps its class |
| 4, records not typed or idempotent | unknown attempts, contradictory finishes, non-finite or boolean values and boolean populations all refuse; metric events have stable identities and are idempotent |
| 5, missing R2 evidence became a positive finding | checks are computed only from typed evidence and are None otherwise; an unexpected cell is an error, not extra support |
| 6, class entry points cannot load | a class entry point is instantiated |
| analytics gaps | grain preserved, summed population named as such beside a distinct attempt count, an unreadable record refuses instead of being skipped, rebuilds published atomically |

**The corrected contrast is running and its first seed is complete**, with the declared contract actually executed:

| cell | selected epoch | restored matches selection | detector moved by its fit | observed updates |
|---|---|---|---|---|
| R0_s2021 | 58 | yes | yes | 35,910 |
| R1_s2021 | 55 | yes | **no** | 35,910 |
| R2_s2021 | 51 | yes | yes | 35,910 |

The equal update allowance is now an observation rather than an arithmetic claim, and R1's detector is measurably frozen while
R0's and R2's move. Seeds 2022 and 2023 remain. **No score is reported from this run**: the validation figures are the declared
selection monitor, not the reference metric, and there is no H1 claim.

Suites: predictor reproduction 142 passed at `b62fd5f6`; M5PHET 107 passed at `8b78114`.
