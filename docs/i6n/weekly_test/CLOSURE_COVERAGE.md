# Closure coverage (part B of ORDEN_SATOSHI_OPERACION.md)

Candidate, not deployed. `weekly_test.tick` now writes `COVERAGE.json` on every tick, embeds a
per-family summary in `STATUS.json`, adds `coverage` to `ANNUAL_TEST.json` at closure and appends
a coverage section to `RESULTS.md`. No metric changes: the annual pool is still sums of absolute
errors and pair counts per horizon with each head's own paired naive (`weekly18.pool_metrics`).

Per family: `declared_weeks`, `scored_weeks`, `no_pair_weeks`, `failed_weeks`, `pending_weeks`,
with the exact invariant `scored + no_pair + failed + pending == declared` (a broken invariant
raises `COVERAGE_INVARIANT`). The no-pair weeks are listed with their cell id, market open, market
close and a `year_boundary` flag (true for `WEEKLY-2024-12-29` and for the end-of-year week that
closes in January). `prediction_coverage` is `scored / declared`; it is never 100% while a
no-pair, failed or pending week exists.

Per horizon: `weeks_with_pairs`, `weeks_without_support` (no-pair weeks plus scored weeks whose
head reports `NO_VALID_TARGET_IN_PERIOD`), their ids, and `pairs_scored` (the N actually pooled).

Glossary (receipts are never renamed): `SIN_PARES` is the descriptive name of the persisted week
state `SKIPPED_NO_EVAL_ROWS`; `NO_VALID_TARGET_IN_PERIOD` is the per-head disposition inside a
scored week. A no-pair week is not a zero MAE and not a successful fit.

Tests: `tests/i6dmo/test_prequential_closure.py` (both annual boundaries, a head without support
in a useful week, a failure, an incomplete closure, the tick seam with the real family heads).
