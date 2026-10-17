# Best Forecast Registry

## Discovery And Requirements

The owner supplied the matched MG2m table as the intended reference. Its
identity is I6-DMO-SHORT-MG2-MATCHED-R0, not the earlier unmatched MG2.
Artifacts exist. The failure was reference selection and absence of an
always-current best-results index, not loss of the stored model.

BR01: retain every accepted result's model/config, seed, data/scaler/code and
weight hashes, monitor formula and split roles, patience/delta/max epochs,
actual/best epochs and exact per-horizon MAE/naive/counts.
BR02: maintain best per horizon and best complete six-head recipe separately.
BR03: never rank TRAIN against validation or partial periods against annual
results. Never claim different original populations are a controlled ranking.
BR04: recompute metrics from authenticated saved predictions; no fits or TEST.
BR05: create a common-support annual comparison from actual origin/price/target
pairs, and identify it by its own digest. A changed cohort is a new comparison,
not an improvement over the old cohort.
BR06: every new strict best in a comparison creates a durable immutable event
with old/new profile identities. Retain historical comparison cohorts.
BR07: automatic bounded CPU timer, status at any time, governed OLAP event
publication/readback, local storage and second-host copy, without agent polling.
BR08: pin the recovered MG2m reference explicitly; do not change its identity
or retrain it merely to rebuild an index. Keep later early-stop variants separate.
BR09: preserve predeclared fixed-epoch diagnostic checkpoints, their actual
weights/predictions and parent history, in a separately published class. Never
promote them as monitor-selected champions or silently discard an observed best.

## Design And Verification

Read only sealed legacy trainer receipts and current batch results. Cache only
verified immutable profiles, not mutable status. Preserve original reported
metrics plus independently recomputed metrics. The common annual board uses
intersection origins per horizon and rejects conflicting actual/origin prices.
All numbers remain original-price errors with their own horizon naive.

Incomplete cadence periods remain individual PERIOD records. Aggregate a
cadence only when the existing continuation.aggregate contract confirms every
expected period and annual support; retain every member checkpoint. An annual
aggregate is not a single trained model.

Persist profiles, cohort champions and events under an exclusive lock; atomic
JSON and the existing outbox provide restart/idempotence. SQLite/WAL stay on
ext4. Public summaries contain source labels and hashes, not host paths or
credentials. Archived event cohorts never get overwritten when new data
changes the common intersection. Native-population numbers are descriptive,
not a controlled cross-population ranking.

Acceptance: CPU fixtures cover identity/data corruption, per-horizon naive,
complete-recipe versus mosaic, period exclusion, source overlap, common-cohort
drift, typed quarantine, idempotent events and publication retry without fit.
Real acceptance is read-only recovery of the cited MG2m vector, comparison of
later results, automatic timer/publication/mirror delivery and unchanged main
GPU campaign. No new training and no Satoshi delegation.
