# Saved-weight VALIDATION reproduction

## Completed inference replay

All 102 weekly checkpoints were authenticated and loaded on the preferred GPU,
with their own original configurations, scales and prepared VALIDATION inputs.
No fits and no TEST reads occurred in either replay arm. Each family's 51
checkpoints represent the original weekly four-calendar-year fits, not a single
end-of-year model retrospectively applied to earlier weeks.

The direct .keras archive path reproduced the improvements but not bitwise
predictions (maximum price difference: short 3.6626872534917254e-6; long
1.4587656206765942e-5). This was not accepted as exact or hidden behind a tolerance.
The second arm loaded the same tensors into the original model builder and used
the original monitoring inference path and batch partition:

| Family | Saved tensor identities | Predictions and metrics |
|---|---|---|
| Long, 51 checkpoints | Exact | Exact, maximum prediction difference 0 |
| Short, 51 checkpoints | Exact | Not bitwise; maximum prediction difference 7.622376063665115e-10 |

No cause for the remaining short cross-device numerical difference is certified.
Both families preserve every original improvement at the precision below.
Relative error is 100*(MAE_h-naive_h)/naive_h on each head's own business-masked
rows and original price units; annual aggregation pools absolute-error sums/counts.

| Horizon | Short original | Short reloaded | Long original | Long reloaded |
|---|---:|---:|---:|---:|
| 2 h | -0.213308% | -0.213308% | - | - |
| 3 h | -0.712522% | -0.712522% | - | - |
| 4 h | -0.867257% | -0.867257% | - | - |
| 5 h | -0.797215% | -0.797215% | - | - |
| 6 h | -1.555002% | -1.555002% | - | - |
| 12 h | -3.267697% | -3.267697% | - | - |
| 18 h | -4.571488% | -4.571488% | -1.822553% | -1.822553% |
| 24 h | - | - | -3.220871% | -3.220871% |
| 36 h | - | - | -8.381802% | -8.381802% |
| 48 h | - | - | -15.674639% | -15.674639% |
| 60 h | - | - | -23.735451% | -23.735451% |
| 72 h | - | - | -24.564042% | -24.564042% |

## Paired same-year diagnostic

On each of the 50 later available weeks of the same VALIDATION year, compare
its own retained checkpoint against the previous available week's checkpoint.
Both use exactly the same target rows, masks and per-horizon naive. Donor
availability is checked before inference. The diagnostic excludes the first
available week in each family, so these numbers must not be compared as if
they were the full 51-week annual population above.

| Family / head | Its own week's selected checkpoint | Previous week's checkpoint |
|---|---:|---:|
| Short / 2 h | -0.185459% | +0.691441% |
| Short / 12 h | -3.332729% | +3.438974% |
| Long / 18 h | -1.498634% | +4.207928% |
| Long / 72 h | -25.472027% | +13.989831% |

Every head loses to its naive under this shifted diagnostic, even within 2024.
This supports a checkpoint-selection/generalization problem; changing to 2025
is not required to observe deterioration. It does not isolate all causes or
reproduce the chained warm-start TEST procedure.

The original code builds a new model for each weekly fit. Each family's 51
initial tensor hashes are identical, consistent with that seeded fresh-model
procedure. TEST instead continued from the previous model's trained weights.

## Small warm-start pilot and automation

A separate four-fit diagnostic runs the first two following VALIDATION weeks
per family, from the first retained weekly donor. It reuses the existing
prequential runner, unchanged trainer, four-year arrays, patience10, min_delta1e-8,
max200 and mean TRAIN/already-observed-week monitor. The optional split label
only prevents this VALIDATION pilot from being misreported as TEST; defaults
and the deployed TEST runtime are unchanged. Pilot status/result files, not
this launch-time return, determine completion and measured results.

`python -m tools.i6dmo.validation_replay` provides prepare, deploy, run,
status, analyse, publish, prepare-pilot, pilot and collect actions. No watcher
agent is needed. The coordinator collector runs finitely on a timer, copies
results, publishes annual and per-week metrics through the durable outbox,
verifies warehouse readback, and stops only its own timer after all three
arms publish. SQLite/locks remain on the native filesystem; retained artifacts
use the configured experiment storage. Results remain on both executing and
coordinating machines.

Tests: original missing-module PRE failed collection; five replay contracts
passed, then 27 replay/prequential/independent-audit tests passed including the
new VALIDATION split-order test. The final combined count is recorded in the
session return. Synthetic tests are not a GPU-fit certificate.
