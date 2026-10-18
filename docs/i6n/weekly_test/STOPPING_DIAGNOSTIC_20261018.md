# Stopping diagnostic, 2026-10-18

Owner requests investigation of 11-epoch stops and comparison with historical
2000 epochs / patience 50. Preserve the current 200/10 campaign and its evidence.
This probe runs after the parent family completes, on the same GPU class.

Select deterministically three already completed non-BN updates per family:
the earliest 11-epoch legacy4 update, earliest 11-epoch relative1 update, and
longest legacy4 update (earliest on ties). Copy each authenticated inference
donor, original FIT/calibration arrays and next-week inputs. Start each probe
from the same donor weights/scalers, seed, data, optimizer and monitor as its
source. Change only max_epochs=2000 and patience=50; min_delta remains 1e-8.
No random reinitialization. No BatchNorm insertion. No TEST access.

Check that every retained source epoch has identical fit, calibration, monitor
and update count in the probe. A mismatch is explicit and prevents claiming an
isolated stopping-budget contrast. Retain the mismatch and new trajectory.
After selecting the new checkpoint using only the known calibration week,
seal its next-week prediction before loading next-week labels. Compare with
the original checkpoint's saved next-week predictions on identical rows/naives.
This is a six-case diagnostic, not an annual validation or hyperparameter winner.

Tests designed before code: chronology-preserving case selection; strict epoch
prefix comparison with corrupted-history rejection; unsupported-row coverage
and calendar month counts; immutable donor/data/result authentication via the
existing prequential helpers. Resource admission waits for the parent's family
closure and no other GPU job; each unit is governed at 4352 MiB and 7200 s, with
6 GiB parent limit, zero swap and durable source/result records. Failure is
terminal for that case; independent cases continue. Timers publish, read back,
mirror and stop on terminal closure; no agent polling.

Data finding retained for the probe: a representative early-2024 FIT contains
16285-16614 rows over the declared four-calendar-year span, but no training
origins in March-July 2023. The source still contains 1851 timestamps in those
months, including 395 nonfinite GBPUSD values. Spacing-2 input admission marks
666 origins INPUT_VALUE_NOT_FINITE, 1175 INPUT_SLOT_NOT_OBSERVED and only 10
INPUT_OK before combining label support and other window resolutions. Thus a
calendar span is not evidence of complete coverage. Do not fill missing bars,
alter features, or silently replace the sealed dataset during this probe.

The known-week monitor has a median of 24 valid pairs for h72 and a minimum of
6; the 50/50 arithmetic alone cannot establish that this monitor generalizes.
TRAIN_REPLAY remains separately pending and is not replaced by FIT losses.

## Execution receipt

Deployed source `46aad9a38f2dab846096f465b8a5306eda1d81f1`; packet
`f75822e98ddff1201a6103ec0a42a6fc38818a104df1defa818d6b7970f4f66f`.
Twelve focused tests passed, including refusal to run before parent completion
and detection of changed epoch weights. Both workers reported WAIT_PARENT on
the start check (0/3 cases each). Worker and coordinator timers are enabled.
No probe scientific fit has run at this receipt; the future result includes
the exact-prefix verdict and next-week comparison. Source campaign unaffected.
