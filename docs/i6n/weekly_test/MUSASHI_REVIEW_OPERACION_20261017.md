# Operational candidate review

Reviewed candidate: `satoshi/prequential-ops-20261017`, code `6e96be14`,
return `c274595b`, exact base `9beb31b6`.

## Decision

The original candidate required corrections before deployment. The corrected
candidate on `codex/prequential-ops-review-20261017` passes the focused CPU
regressions below. This is not GPU acceptance, a real annual-close certificate,
or authorization to overwrite a running source-pinned campaign.

The active campaign remains on its original runtime, with its scientific identity,
predictions, model weights, training parameters and timers unchanged. No live
deployment or migration was performed during this review. Future deployment must
pin the corrected sources explicitly; changing the active checkout in place
would trip its source authentication.

## Findings and corrections

| Finding | Evidence before correction | Corrected behavior |
|---|---|---|
| High: an interrupted RUNNING attempt was ignored when choosing the next attempt | Synthetic unfinished ledger followed by run started a new fit | Under the worker lock, record UNKNOWN / InterruptedAttempt and preserve the seal; fail closed because the cause is unverified |
| High: permanent/unknown/exhausted ledger saved just before RESULT could reopen | All three terminal-ledger cases started another fit | Reconstruct FAILED from retained evidence without changing the ledger or consuming an attempt |
| Medium: completed-ledger finalization created another attempt | Seal and fit were reused, but the ledger bytes changed | Use the same completed attempt; adopt the receipt, retain the ledger byte for byte; missing receipt or changed donor fails without another fit |
| Medium: WAIT_RESOURCES kept a finite ETA | Resource-wait state with seconds=99 remained estimable | Report NOT_ESTIMABLE / null while resource admission blocks progress |

A hard process interruption is deliberately not assumed to be a harmless GPU
transient. Typed, caught transients retain the original initial-plus-two retry
policy and 30/60 second cooldowns. No memory cap, architecture, optimizer,
stopping rule, data mask, donor, feature list or DOIN code was changed.

## Reproduction and traceability

Run from the corrected review checkout, CPU only:

```bash
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
crispdm-run -q -m 2G -t 120s -n preq-ops-review -- \
python -m pytest -q -p no:cacheprovider \
tests/i6dmo/test_prequential_recovery.py \
tests/i6dmo/test_prequential_closure.py \
tests/i6dmo/test_prequential_test.py \
tests/i6dmo/test_retsu_prequential_audit.py \
tests/i6dmo/test_weekly_test.py
```

| Gate | Observed result |
|---|---|
| Candidate's original five-file focused suite | 54 passed, 1 intentional xfailed; 3.12 s |
| Added interrupted/terminal/completed-ledger and resource-ETA probes against original candidate | 6 failed; 0.70 s |
| First corrected five-file suite | 60 passed, 1 intentional xfailed; 3.19 s |
| Additional missing-receipt/changed-donor finalization probes before their guard | 2 failed; 0.61 s |
| Final corrected five-file suite | 62 passed, 1 intentional xfailed; 3.22 s |
| Eleven scientific files against exact base | No diff; the existing byte-identity test also passes |
| git diff --check | Clean |
| Source-pin verification in the active protocol context | Passed; runtime was not changed |
| Coverage function on authenticated, retained partial terminals, read-only | 53 declared per family; each 13 scored + 1 no-pair + 0 failed + 39 pending; invariant holds |

The eight added test cases cover the findings directly:
`test_abrupt_interruption_is_unknown_and_cannot_silently_restart_a_fit`,
`test_crash_after_terminal_ledger_does_not_reopen_an_attempt` (three cases),
`test_crash_after_completed_ledger_reuses_update_without_an_extra_attempt`,
`test_resource_wait_has_no_finite_eta`, and
`test_completed_attempt_finalization_never_refits_or_rewrites_ledger` (two cases).

At the last generated active STATUS read: 26 complete, two no-pair terminals,
28/106 terminal, zero failures, 28/28 warehouse readbacks, zero pending and zero
discrepancies. A short-family update was running. Mean-fit ETA was 12,938.6 s,
excluding resource waits and publication. These are a partial operational
snapshot, not new annual scientific conclusions.

## Limits and next action

All failure injection remains synthetic, in-process CPU testing. No actual
SIGKILL/power-loss, GPU retry, or annual production closure was exercised. The
existing RUNNING marker is still not independent proof of process liveness.
Historical FAILED terminals remain terminal. An interrupted attempt with an
unverified cause now requires audited recovery rather than automatic refitting.

Allow the active campaign to continue. Retain this corrected candidate for an
explicit source-pinned operational release; do not silently hot-swap it or
claim that the current runtime already has these corrections.
