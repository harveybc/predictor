# Feature-selection DuckDB traceability

| Requirement | Implementation | Evidence |
|---|---|---|
| Exact `50bddf3` validation | `feature_selection.py` | `test_validation_contract_is_the_exact_file_from_data_warehouse_50bddf3` |
| Six normalized families plus run and receipt | `feature_selection_store.ddl` | `test_real_duckdb_provider_atomically_loads_every_family_and_receipt` |
| One owner transaction | `Plugin.write_feature_selection_envelope` | contradiction rollback test |
| Immutable natural row identity | `feature_selection_store._insert_row` | contradiction rollback test |
| Idempotent replay and serialization | receipt lookup plus DuckDB owner lock | replay and concurrent-replay tests |
| Invalid input writes nothing | defensive provider validation | invalid-row-digest test |
| Read-only analytical surface | six `df_feature_*` views | reopen and dashboard test |
| Snapshot counts and content digests | `SNAPSHOT_RELATIONS`, `_governance_digests` | verified-snapshot test |
| No live warehouse mutation | temporary-path fixtures only | test fixture and commands above |
