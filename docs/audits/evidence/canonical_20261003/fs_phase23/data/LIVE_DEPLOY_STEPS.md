# Live deployment steps (integration agent; nothing here was executed by the data agent)

Preconditions: both branches merged or checked out at the tips named in RETURN/README;
all suites green on the coordinator (`TEST_RESULTS.txt`).

1. Install, into the warehouse host's virtualenv on the coordinator (the one the
   `crispdm-data-warehouse-olap.service` unit runs), in this order:
   - `pip install <predictor checkout>/olap/store`          -> predictor-olap-store 0.1.5
   - `pip install <predictor checkout>/olap/duckdb_store`   -> predictor-duckdb-store 0.1.4
   - `pip install <data-warehouse checkout>`                -> data-warehouse-service 0.1.1
     (branch `satoshi/fs-phase23-write-route-20261006`, built on the deployed 2d4550d)
   Verify: `python -c "import predictor_olap_store, predictor_duckdb_store; print(predictor_olap_store.__version__, predictor_duckdb_store.__version__)"`
   prints `0.1.5 0.1.4`, and `pip show data-warehouse-service` prints 0.1.1.
2. Take the pre-migration safety copy with the existing procedure
   (`tools/olap_duckdb_migrate.py snapshot --owner-stopped ...` during the restart window) or
   accept the service's own checkpoint; record its SNAPSHOT.json beside this file.
3. Restart the service: `sudo systemctl restart crispdm-data-warehouse-olap.service`.
   `_ensure_schema` runs the additive DDL at start-up (checkpointed before any write).
4. Verify through the running service, token from the environment, never from a file in
   the repository:
   - `GET /api/v1/host` lists `write_fs_phase23_rows`, `read_fs_phase23_rows`,
     `reconcile_fs_phase23` under capabilities;
   - `GET /api/v1/query?sql=SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'feature_%' OR table_name LIKE 'fs_phase23%' ORDER BY 1 LIMIT 20`
     lists the ten relations;
   - the phase-1 counts are unchanged (compare with the pre-restart query of
     `df_fact_*` counts).
5. Smoke the write route with ONE synthetic row under a throwaway run_id
   (`phase2-smoke:<date>`), read it back (`GET .../rows`), reconcile it, and record the three
   responses in `LIVE_SMOKE.json`. The run stays in the store as evidence; it is not a
   population identity and the closure never counts it.
6. Hand `<WH>` = `http://127.0.0.1:5057` (plus `WAREHOUSE_TOKEN` in the follower's
   environment) to the follower: `tools/fs_phase23_warehouse.open_warehouse(<WH>)`.
7. After phase-2 closure: snapshot per `SNAPSHOT_PROCEDURE.md`; the boundary now includes
   the ten relations so the result can be `VERIFIED_SNAPSHOT`.

Rollback: the migration adds relations only; the previous packages can be reinstalled and
the service restarted, and the ten relations remain inert. No existing row is touched.
