# Weekly monitor contrast

Owner authorization: 2026-10-18. Pause the remaining window/core sweep at a
completed-fit boundary. Preserve results and publication. The existing stopping
implementation is not declared defective: the question is its generalization.

## Frozen experiment

Stage A runs three monitors, for short7 and long6 separately: `legacy4`,
`relative4`, `relative1`. Stage B repeats those three with one BatchNormalization
layer between the temporal core and the private heads. No other architecture,
gradient objective, learning rate, seed, window, mask or stopping parameters
change. BN uses momentum .99 and epsilon .001; inference never updates its
statistics. A BN insertion imports all original layers by name and initializes
the new transform to identity. BN comparisons are paired against their Stage A
counterpart, not a retrospectively selected different monitor.

All fits use four calendar years for gradients. `relative1` restricts only the
FIT monitor to the last calendar year. Each normalized monitor uses its own
fixed naive for each set/head: mean_h(MAE/naive - 1), averaged 50/50 between
FIT monitor and already observed calibration week. Zero naive fails explicitly;
unsupported heads are recorded and excluded, never assigned zero skill.
Legacy4 keeps the old TRAIN-naive denominator. Patience 10, min_delta 1e-8,
max_epochs 200, restore lowest monitor. No recursive weekly replay per epoch.

Initialization: every arm starts from an independent copy of the first retained
weekly VALIDATION checkpoint for its family (week 2024-01-07). Its calibration
week is excluded from forward scores. Predict all remaining supported 2024
weeks in order, seal before labels, score, update for next week. No 2025 reads.
This is reused development data and retrospectively selected features, not an
independent out-of-sample claim. Freeze donor/scaler/config hashes before fits.
All arms have identical per-head evaluation supports and naives. Dates absent
from the source packet, including boundary weeks, are reported as uncovered.

The calibration week is disjoint from the four-year gradient period for that
update. Later updates may include previously observed weeks. Full future 2024
is never supplied to the epoch monitor. The initial January checkpoint is a
common experimental starting point, not a fresh fit under each new monitor.

## Reporting and order

Per epoch retain both monitor components, sums/counts, own naives, source row
digests, best epoch and stop reason. Per week retain forecast, checkpoint,
scalers, FIT and calibration metrics separately from forward VALIDATION.
Pool annual sums/counts; do not average weekly ratios. Report all heads.
Internal FIT is NOT TRAIN_REPLAY. Full retrospective adaptive TRAIN replay is
still required for the owner's TRAIN reporting and is a separate subsequent
measurement; mark it pending rather than substituting static FIT.

This campaign establishes comparative forward VALIDATION performance and cost.
It cannot yet produce the requested TRAIN_REPLAY/VALIDATION gap. It does not
automatically promote a model or reopen TEST. One-year/six-month/three-month
gradient histories follow this monitor contrast as separate designs. They are
not executed by this campaign. BatchNorm is not assumed to cure overfitting.

## Automation and resources

Preferred GPU runs all short arms; secondary GPU all long arms. A family remains
on the same GPU class across comparisons. One heavy process per host. Each timer
runs one weekly unit under the existing resource governor. A result and all its
files are authenticated before reuse. No fit is repeated after UPDATE receipt.
Transient process deaths have a bounded retry ledger; integrity/unknown failures
are terminal and leave independent arms schedulable. No model shrinkage or cap
reduction on refusal. Status distinguishes waiting, failed, pending and complete;
ETA is measured from completed fits and unavailable on resource blockage.
Coordinator collects durable results, publishes via the existing outbox with
readback, keeps two copies, and stops only this campaign's timers at closure.

## Traceability and tests (defined before implementation)

| Requirement | Behavioral evidence required |
| --- | --- |
| MC01: no gradient-population change | monitor subset leaves all FIT arrays intact; real trainer update counts |
| MC02: own naive, paired rows | hand-computed two-head errors, zero-naive and unsupported-head cases |
| MC03: strict temporal order | existing prequential perturbation/seal tests; future donor refusal |
| MC04: one-year means calendar year | boundary/leap-day selection and out-of-period rejection |
| MC05: default compatibility | original early-stopper tests and legacy evaluator parity |
| MC06: BN only at core output | named layer/tensor equality, archive reload, no moving-stat change at inference |
| MC07: resumable identities | changed packet/code/result rejected; existing update receipt reused |
| MC08: no misleading reporting | no TRAIN_REPLAY value until computed; own per-head naive and counts |
| MC09: durable execution | bounded attempts, lock, resource refusal, independent-arm advance |

Method: repository method document absent in this worktree; applied the portable
test-led-data-centric-development checklist. S0-S8 defined here; implementation
and verification evidence is appended in RETURN_MONITOR_CONTRAST_20261018.md.
