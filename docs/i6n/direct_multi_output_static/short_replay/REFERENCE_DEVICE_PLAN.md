# Reference-Device Reproduction

Owner directive: explain the failed short-only reproduction and correct defects,
not search seeds or checkpoints until a desired validation number appears.

The completed secondary-device replay changed GPU class relative to the source.
It establishes neither training equality nor a regression in unchanged source.
The historical saved-weight inference already reproduces all six scores within
4.62e-14 MAE. No historical artifacts or scientific results are overwritten.

## Fixed Check

One additional necessary fit, the third execution including the historical fit
and secondary-device replay. Use the original preferred GPU device and Python
environment. Freeze the same seed, 19 features, MG2 13-slot input, six short
heads, source bundle, original design and original trainer/model file bytes.
Original stopping: mean FIT/INNER, patience 10, min_delta 1e-8, max_epochs 200.
No new stopping criterion, extra pre-fit model construction or TEST access.

The new launcher refuses a GPU name or visible UUID mismatch. It uses a distinct
campaign identity and existing governed publication/readback/mirror machinery.
The main queue pauses only between fits, never by killing a running fit. Its
timer resumes automatically after the diagnostic worker exits; an independent
five-minute recovery timer protects against an interrupted launch.

## Acceptance And Traceability

| Requirement | Test/evidence |
|---|---|
| Reject wrong device before importing trainer | test_replay_device.py |
| Keep original trainer and model byte-identical | legacy_code_proof and historical receipt pins |
| Keep diagnostics separate from closed replay | campaign-name and terminal identity tests |
| Detect any changed update trajectory | test_replay_comparison.py; every epoch weight digest and monitor |
| No tuning to historical validation scores | fixed one-fit budget; compare only after original train_unit finishes |
| Report what actually reproduces | identity checks, selected epoch, learned weights, paired MAE per horizon |
| Preserve data/results and ongoing work | verified mirrors, OLAP readback, automatic main timer recovery |

If training reproduces exactly on the original device, the cross-device change
is isolated as the reproduction failure under these fixed conditions. This is
not proof that every possible backend difference is explained. If not, retain
the failure and investigate the earliest differing operation; do not rerun fits
without an identified correction or a newly justified bounded control.
