# Retsu: PS0/PS1 source and transform coverage

This is one bounded CPU assignment. It runs **in parallel** with the existing
Gamma PS3-R E/F jobs. Do not stop, relaunch, duplicate or modify those jobs.
The canonical plan is `docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md` and its current state is
`docs/tres_temas_entrevista/program_v3/CURRENT_EXECUTION.md`. The retained
366-feature reconciliation is already done; do not redo it.

## Start

Use a new predictor worktree and branch from
`origin/satoshi/canonical-exec-20261003`, not the live checkout. Write only
under `docs/audits/evidence/canonical_20261003/source_transform_coverage/`.
Read `AGENTS.md` before touching code. No GPU, service restart, data deletion,
credential discovery or live trading. Limit tests to this new tool.

## Inputs

- `laneA/batch_001/inventory_sources.csv` (columns include provider, state,
  frequency, event_time, availability_time, files and batch).
- `laneA/batch_001/transform_variants.csv` (prefix probes and declared status).
- `laneA/batch_{001,002,003}/inventory_columns.csv`, `coverage_matrix.csv`,
  `batch_report.json`, and `digests.json`.
- `coverage_reconciliation/coverage_reconciliation.json` for the existing
  366 = 279 + 87 partition. This is a feature population, not proof that all
  source subscriptions or transform variants were admitted.
- Read-only resource contracts/discovery in `data-gov` and `data-lake` when
  needed. Record exact paths and hashes; never treat an API subscription or a
  resource listing as ingested historical bytes.

## Deliverable

Implement a deterministic, standard-library-only generator plus focused tests
which produce `source_coverage.csv`, `transform_coverage.csv`, and `REPORT.json`.
One source/resource row must say: provider, resource identity, frequency,
event/availability clock and timezone, historical TRAIN intersection if
established, contract/byte status, current disposition, reason, and next action.
One transform row must say: variant identity, prefix-causality result,
admissibility, PS1/PS4 profile state, denominator and pending action.
Every input file used gets SHA-256 in the report. Keep states distinct:
`MEASURED`, `DECLARED_ONLY`, `NOT_AVAILABLE_FOR_TRAIN`, `NOT_INGESTED`,
`NOT_APPLICABLE`, `UNKNOWN` and `PENDING_PROFILE`; do not convert an unknown
to zero coverage or a pending row to a rejection. Preserve duplicate paths as
separate inventory records unless the retained identity proves they are the
same object.

Reconcile, at minimum, Yahoo Finance, FXMacroData and Alpaca, plus all other
providers actually present in the inventory; do not hard-code the provider
list as the denominator. Check the exact train interval in the retained
contract before claiming a source usable. For variants, distinguish trailing
MODWT-Haar/multitaper/Hilbert/STL/Kalman filter from global DWT/Hilbert/STL
and RTS smoother. A prefix probe establishes causal computability, not
predictive utility, extractibility, or selection.

Tests must fail on an omitted source, a duplicated identity with conflicting
states, a missing availability clock presented as measured, a transform with a
prefix violation presented as admissible, and a fake TRAIN overlap. Compare
output denominators against input rows; do not assume 388 inventory rows are
388 selected features. Run the generator and tests twice to check deterministic
bytes. No learned model, causal effect or financial metric is measured here.

## Return

Commit and publish the branch. Return the commit, tests, report counts by
status and provider, unresolved blockers, and exact paths. `NO_NEW_MODEL_MEASUREMENT`
is expected. Do not edit current state, queue, chart, or GPU driver files;
the coordinator will integrate after checking the evidence.
