# Readback reconciliation report format (consumed by close-phase2 / close-phase3)

Producer: `tools/fs_phase23_warehouse.py readback` (`--duckdb <copy>` or
`--service-url <url>` with the token in `WAREHOUSE_TOKEN`), and `reconcile`.
Examples (synthetic, 12 rows, generated on a throwaway file):
`EXAMPLE_READBACK_REPORT.json`, `EXAMPLE_RECONCILIATION_EURUSD.json`,
`EXAMPLE_RECONCILIATION_ETH.json`, `EXAMPLE_RECEIPT_*.json`.

## `fs_phase23_readback_report.v1`

```
schema, source ("local" | "service"), run_id (null = all runs), generated_at, report_sha256
tables.<table>:
  total                 rows
  table_rows_sha256     digest of the group digests (order independent)
  by_asset              {population_id: n}
  by_method             {method: n}
  by_host_role          {host_role: n}        roles only; UNDECLARED when absent
  groups[]              one per (population_id, method, host_role, fold):
    n, rows_sha256 = sha256(string_agg(row_sha256, '' ORDER BY row_sha256)),
    measured, insufficient_support, failed   (null for tables without a state column)
```

The same SQL is executed locally and through the service, so two reports are compared
group by group with `compare-readback LOCAL REMOTE` (exit 0 agree, 3 differ). The closure
gate requires `agree == true` between the follower's local terminal store and the
warehouse for every (asset, method, host_role, fold) group, and the per-asset totals equal
the contract denominators times the declared folds, lags and methods.

## `fs_phase23_reconciliation.v1` (per run)

```
run_id, run_sha256, population_id, phase, receipts_verified, complete, reconciliation_sha256
tables.<table>:
  expected                  from the run's declared expected_json (null if undeclared)
  stored, stored_rows_sha256
  receipts, receipted_new_rows
  count_matches_expected    present when expected is declared
  receipts_cover_store      receipted_new_rows == stored  (rows without a receipt fail)
```

`complete` is true only when every declared count matches and every stored row is covered
by a receipt. A receipt presented with a foreign run or population identity, a tampered
digest, or one the store never issued is refused and the report is not produced.

## `fs_phase23_load_receipt.v1`

```
run_id, run_sha256, population_id, table, submitted, distinct_identities, stored_new,
already_stored, rows_sha256, host_role, shard_id, submitted_at, receipt_sha256
```

`rows_sha256` is the digest of the submitted identities' content digests; a replay of the
same terminal returns `stored_new = 0` with the same `rows_sha256`.
