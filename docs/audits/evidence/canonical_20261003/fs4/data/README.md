# FS4 warehouse and closure (data agent return, 2026-10-07)

Order: `docs/handoffs/SATOSHI_FS4_EXECUTION_2026_10_07.md` step 3; plan §5-§6 FS4-06/07/08/09/13.
Discipline: implemented, tested on throwaway stores, reported once. Nothing deployed; nothing
written to the live cube or the controller DB; the live controller was read once (read-only) for
its expected counts. Roles only; the service token stays in the environment.

## Code

| Path | What |
|---|---|
| `olap/store/src/predictor_olap_store/fs4_store.py` | the single implementation: DDL, `prepare_terminal`, `submit_terminals` (idempotent, FS4-09 enforced), `read_terminals`, `reconcile`, `verify_receipt`, host documents |
| `olap/store/src/predictor_olap_store/query.py` | fs4 DDL appended to the start-up DDL; `write_fs4_terminals` / `read_fs4_terminals` / `reconcile_fs4` (same lock and transaction path as fs-phase23); MODULE_SHA256 re-pinned, PENDING_REVIEW extended |
| `olap/store/.../provider.py`, `olap/duckdb_store/.../provider.py` | the three capabilities declared; packages 0.1.6 / 0.1.5 |
| `olap/migrations/fs4/0001_fs4_extractibility_additive.sql` | rendered from the code; see `MIGRATION_PLAN.md` |
| `tools/olap_duckdb_migrate.py` | snapshot boundary names the five fs4 relations |
| `tools/fs4_warehouse.py` | `open_warehouse(path_or_url)`, DuckDB file or service (`/api/v2/fs4/*`, token from `WAREHOUSE_TOKEN`, phase-2/3 transport with bounded retries); CLI render-migration / migrate [--dry-run] / readback / reconcile / compare-readback |
| `tools/fs4_closure.py` | the follower: `tick` submits COMPLETE terminals from the controller DB (read-only), verifies readback per task, writes receipts/, quarantine/, STATUS.json (from the task store) and EXTRACTIBILITY_COMPLETE.json from evidence; `status`, `close`, `expected` |
| `tools/fs4_deploy/closure_tick.sh`, `fs4-closure.service`, `fs4-closure.timer`, `install_closure.sh` | systemd --user oneshot every 2 min under `crispdm-run -m 1G`, coordinator only |
| data-warehouse `data_warehouse_service/{backends,web}.py`, `tests/test_fs4_routes.py` | capabilities and routes, version 0.1.2 |

## API (host routes; backend owns the terminals, host owns auth / gating / status codes)

- `POST /api/v2/fs4/terminals` body `{"plan_sha256", "terminals": [terminal document...], "host_role"?}`
  -> receipt (`fs4.warehouse_receipt.v1`: plan_sha256, terminal_count, distinct_tasks, inserted,
  duplicates_ignored, terminals_sha256, task_ids, receipt_sha256); 201 on insert, 200 on pure
  replay, 400 typed refusal (nothing written), 422 without the capability, 401 without the token.
- `GET /api/v2/fs4/terminals?plan_sha256=&task_id=&population_id=&feature_id=&fold_id=&arm=&after=&limit=`
  -> `{"terminals": [document as submitted + terminal_sha256], "count", "next_after"}` paged by task_id (limit 1..5000).
- `POST /api/v2/fs4/reconcile` body `{"plan_sha256", "expected"?: {"total", "by_population"}, "receipts"?}`
  -> `fs4.warehouse_reconcile.v1`: stored totals / by_population / by_arm / by_population_arm /
  terminals_sha256, count_matches_expected, triples {complete, partial, inconsistent, examples},
  receipts {issued, tasks_without_receipt, receipts_cover_store}, receipts_verified, complete.

Examples generated from a throwaway file: `EXAMPLE_TERMINAL_DOCUMENT.json`, `EXAMPLE_RECEIPT.json`,
`EXAMPLE_READ_PAGE.json`, `EXAMPLE_RECONCILIATION.json`, `EXAMPLE_READBACK_REPORT.json`.

## Closure rule (tools/fs4_closure.py)

`EXTRACTIBILITY_COMPLETE.json` is written only when, from the task store and the warehouse:
every admitted task (6,495) is COMPLETE or typed-refused (a FAILED reason carrying a declared
refusal code; lease expiry, runner crash or a non-JSON result are technical and block);
no COMPLETE result fails the store's validation (finite, typed, planned task_id); every arm
triple shares rows/mask/input/population_n/naive_mae (FS4-09, checked in the task store and
reported by the warehouse); every COMPLETE task has a readback-verified receipt; and the
warehouse reconciliation is complete with the same terminals digest as the controller's stored
results. The file carries the plan, the counts, the typed refusals, the receipts digest, the
reconciliation and its own closure_sha256; a second pass leaves it byte-identical.

`STATUS.json` (`fs4.closure_status.v1`) is generated every pass from the task store: state,
expected (total, by_population), complete, pending, active, failed {technical, typed_refused},
workers, last_heartbeat, rate_tasks_per_hour, eta_seconds or null with eta_reason, warehouse
{verified_receipts, pending_submit, quarantined, last_pass}, validation findings, closure reasons.

Deploy: `LIVE_DEPLOY_STEPS_FS4.md`. Tests: `TEST_RESULTS.txt`.
