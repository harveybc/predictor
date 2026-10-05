# Phase-1 feature-selection storage on DuckDB

## Contract

The executable transport validator is an exact copy of
`data_warehouse_service/feature_selection.py` at data-warehouse commit `50bddf3`.
Its SHA-256 is
`91fcb4fde495239a4e0a21d3a39f0b66d50bd0a5df4865db7bd720b454f5f75a` and a test
pins those bytes.

`PredictorDuckdbStore` inherits `write_feature_selection_envelope(document)` from
`predictor_olap_store.query.Plugin`. The method validates defensively, takes the DuckDB
provider's owner lock and writes one transaction. Workers do not receive the DuckDB path.

## Relations

One run dimension and seven facts are append-only:

- `df_dim_feature_selection_run`
- `df_fact_sampling_quality`
- `df_fact_variable_profile`
- `df_fact_information_metric`
- `df_fact_pair_relation`
- `df_fact_feature_causal_evidence`
- `df_fact_feature_selection_decision`
- `df_fact_feature_selection_load_receipt`

Every scientific row is keyed by the contract's content-derived immutable identity. An
identical replay is a no-op. Reusing a run or row identity with different content raises and
rolls back the entire envelope.

Six read-only views expose profiles, the causal ladder, decisions, coverage, failures and the
run dashboard. They are never accepted as substitutes for the underlying immutable facts.

## Snapshot

`tools/olap_duckdb_migrate.py snapshot` now includes all eight phase-1 tables and six views in
the boundary held across checkpoint and copy. It records a row count and deterministic content
digest for each relation, including an explicit digest for an empty relation. Missing phase-1
relations make the artifact `UNVERIFIED_COPY`.

## Verification

All tests use temporary DuckDB files. They never open the configured warehouse:

```bash
PYTHONPATH="$PWD/olap/store/src:$PWD/olap/duckdb_store/src" \
  python -m pytest -q \
  olap/store/tests \
  tests/test_migration_cli_contract.py \
  tests/test_olap_duckdb_migrate.py
```

Acceptance evidence covers exact-contract parity, six-family atomic ingestion, replay,
contradiction rollback, concurrent calls, rejection before writes, reopen persistence,
queryable views and verified snapshot parity.

No service restart or live-cube migration is part of this change.
