# Live deployment steps, FS4 warehouse (deploy agent; nothing here was executed by the data agent)

Written exactly as the phase-2/3 data agent's `LIVE_DEPLOY_STEPS.md`. The data agent did NOT
install into the service environment, restart the service, write to the live cube, or call any
live route. Host references are roles; the token lives in the environment only.

Preconditions: predictor branch `satoshi/fs4-warehouse-20261007` and data-warehouse branch
`satoshi/fs4-write-route-20261007` (built on the deployed master 5ca5937) merged or checked out
at the tips named in `README.md`; all suites green on the coordinator (`TEST_RESULTS.txt`).

1. Install, into the warehouse host's virtualenv on the coordinator (the one the
   `crispdm-data-warehouse-olap.service` unit runs), in this order:
   - `pip install <predictor checkout>/olap/store`          -> predictor-olap-store 0.1.6
   - `pip install <predictor checkout>/olap/duckdb_store`   -> predictor-duckdb-store 0.1.5
   - `pip install <data-warehouse checkout>`                -> data-warehouse-service 0.1.2
   Verify: `python -c "import predictor_olap_store, predictor_duckdb_store; print(predictor_olap_store.__version__, predictor_duckdb_store.__version__)"`
   prints `0.1.6 0.1.5`, and `pip show data-warehouse-service` prints 0.1.2.
2. Take the pre-migration safety copy with the existing procedure
   (`tools/olap_duckdb_migrate.py snapshot --owner-stopped ...` during the restart window) or
   accept the service's own checkpoint; record its SNAPSHOT.json beside this file. The snapshot
   boundary now names the five fs4 relations, so a post-migration snapshot can be VERIFIED.
3. Restart the service: `sudo systemctl restart crispdm-data-warehouse-olap.service`.
   `_ensure_schema` runs the additive DDL at start-up (five CREATE ... IF NOT EXISTS; the
   rendered statements are `olap/migrations/fs4/0001_fs4_extractibility_additive.sql`,
   SHA-256 `bad9467d34f8311307e956aaae0274ce7cd3db39075da0640fe57267a09cf602`). Nothing existing is altered.
4. Verify through the running service, token from the environment, never from a file in
   the repository:
   - `GET /api/v1/host` lists `write_fs4_terminals`, `read_fs4_terminals`, `reconcile_fs4`
     under capabilities (the three phase-2/3 names stay listed);
   - `GET /api/v1/query?sql=SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'fs4%' OR table_name = 'feature_extractibility_v1' ORDER BY 1 LIMIT 20`
     lists `feature_extractibility_v1`, `fs4_extractibility_coverage`,
     `fs4_extractibility_triples`, `fs4_load_receipt`, `fs4_load_receipt_task`;
   - the phase-1 and phase-2/3 counts are unchanged (compare with the pre-restart query of
     `df_fact_*` and `feature_*` counts).
5. Smoke the write route with ONE synthetic terminal under a throwaway plan_sha256
   (`sha256("fs4-smoke:<date>")`; the shape is `EXAMPLE_TERMINAL_DOCUMENT.json`), read it back
   (`GET /api/v2/fs4/terminals?plan_sha256=...&task_id=...`), reconcile it
   (`POST /api/v2/fs4/reconcile` with `{"plan_sha256": ..., "expected": {"total": 1}}`) and
   record the three responses in `LIVE_SMOKE.json`. The smoke plan stays in the store as
   evidence; it is not the campaign plan and the closure never reads it (the closure reconciles
   by the controller's plan_sha256 only).
6. Hand the follower its environment on the coordinator, `~/.config/fs4/closure.env` (mode
   0600), with the variables named in `tools/fs4_deploy/closure_tick.sh`:
   `FS4_WAREHOUSE=http://127.0.0.1:<port>` (the service), `WAREHOUSE_TOKEN=<from the service
   environment>`, `FS4_DB=<coordinator state>/fs4/queue_v2.sqlite`, `FS4_STATE=<coordinator
   state>/fs4/closure`, `FS4_CODE=<predictor checkout>`, `FS4_PYTHON=<interpreter with duckdb
   and sqlalchemy>`, `FS4_CLOSURE_CAP=1G`. Then
   `bash tools/fs4_deploy/install_closure.sh` (it refuses before enabling the timer when a path,
   the interpreter or the controller DB is not real). The timer fires `fs4-closure.service`
   every two minutes; each pass runs under `crispdm-run -m 1G`.
7. Read progress only from `<FS4_STATE>/STATUS.json` (generated from the task store) and, at
   the end, `<FS4_STATE>/EXTRACTIBILITY_COMPLETE.json`. No log is to be watched.

Rollback: the migration adds relations only; the previous packages can be reinstalled and the
service restarted, and the five relations remain inert. No existing row is touched. The
follower is a timer unit: `systemctl --user disable --now fs4-closure.timer` stops it; its
receipts directory is evidence and is kept.
