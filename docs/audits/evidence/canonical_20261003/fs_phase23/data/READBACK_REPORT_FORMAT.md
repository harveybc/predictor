# Readback and reconciliation documents (consumed by close-phase2 / close-phase3)

Producer: `predictor_olap_store.fs_phase23_store` through `tools/fs_phase23_warehouse.py`
(`open_warehouse(<file or URL>)`, `readback`, `reconcile`, `compare-readback`). Examples are
synthetic (12 metric + 6 gate rows on a throwaway file): `EXAMPLE_READBACK_REPORT.json`,
`EXAMPLE_RECONCILIATION_{EURUSD,ETH}.json`, `EXAMPLE_RECEIPT_{EURUSD,ETH}.json`.

## Digest rule (shared with the driver)

`row_sha256 = sha256(canonical_json(row))` of the row **as submitted** (sorted keys, compact
separators, ascii, no NaN); `rows_sha256 = sha256(''.join(sorted(row_sha256)))`; an empty
set digests to `sha256('')`. SQL twin:
`coalesce(sha256(string_agg(row_sha256, '' ORDER BY row_sha256)), '<sha256 of empty>')`.
Readback returns the stored `payload`, i.e. the submitted row verbatim, so
`rows_sha256(read_run(...)) == receipt.rows_sha256` is the readback gate.

## `fs_phase23.warehouse_receipt.v2` (submit_rows)

```
schema, backend ("duckdb" | "service" | kind), run_id, run_sha256, population_id, table,
unit_id (when every row shares one), row_count (submitted), distinct_identities,
inserted, duplicates_ignored, rows_sha256, host_role, shard_id, submitted_at,
receipt_sha256 = sha256(canonical_json(receipt minus receipt_sha256))
```

## `fs_phase23.warehouse_reconcile.v2` (reconcile)

```
schema, backend, run_id, run_sha256, population_id, phase, registration, receipts_verified,
complete, reconciled_at, reconciliation_sha256
tables.<table>:
  count, rows_sha256                 the two fields the driver compares
  receipts, receipted_inserted, receipts_cover_store (receipted_inserted == count)
  expected (declared via register_run expected_json or the request), count_matches_expected
```

`complete` is true only when every declared count matches and every stored row is covered
by a receipt. A presented receipt with a foreign run/population, a tampered digest, or one
the store never issued is refused and no document is produced.

## `fs_phase23_readback_report.v1` (readback; same SQL locally and through the service)

```
schema, source, run_id (null = all runs), generated_at, report_sha256
tables.<table>: total, table_rows_sha256, by_asset, by_method, by_host_role,
  groups[] per (population_id, method, host_role, fold): n, rows_sha256,
             measured / insufficient_support / failed (null where the table has no state)
```

`compare-readback LOCAL REMOTE` exits 0 when every group agrees (n and digest), 3 otherwise.
Closure gate: agreement between the follower's local store and the warehouse for every
(asset, method, host_role, fold) group, and per-asset totals equal to the contract
denominators times folds, lags and metrics.
