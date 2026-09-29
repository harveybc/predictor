# Literature-aligned TSL metrics in the existing warehouse

## Operational readiness

The deployed DuckDB provider already accepts the existing governed metric
schema. No schema migration, service restart or production test insertion was
required. The receipt convention for upcoming runs is
[`contracts/tsl_literature_metrics.v1.json`](contracts/tsl_literature_metrics.v1.json).
It covers Electricity, Weather and Traffic, with their registered resource IDs
and verified distributor hashes.

Primary results are **MSE and MAE in the reference protocol's normalized
target space**, each averaged over every scored window, forecast step and
target channel. The lookback, split, scaler, author implementation and numeric
precision must be pinned by the experiment; the warehouse does not decide
them. A matching metric name alone is not evidence of comparability.

Use the existing campaign -> delivery -> terminal -> warehouse route, placing
the contract context in terminal `tags`, not adding arbitrary columns to each
metric object. Include paired persistence MAE/MSE on exactly the same targets.
Horizons 96/192/336/720 are STEPS: Weather steps are ten minutes; Electricity
and Traffic steps are one hour. Store both step and elapsed-time horizons.

Store each seed separately. Published values and our measured values must not
be conflated. Any paper comparison names its source, table and exact protocol;
changes to the task require a matched rerun, not a direct published-score
comparison. Four-horizon averages and seed summaries are separate aggregates,
not extra single-horizon observations.

The general warehouse remains generic. This contract is a producer convention,
not a global validator silently retrofitted to historical runs. Existing
training runners are not automatically rewritten by installing this contract.
Upcoming Weather/Traffic runners must explicitly populate its context fields.

## Verification performed

`tools/test_tsl_warehouse_receipt.py` ran using the production warehouse's
Python environment and installed provider, on a temporary DuckDB file only:

- 3 datasets x 4 horizons x 2 seeds = 24 distinct terminals, 96 metric rows.
- Four metrics per terminal: model MAE/MSE and paired persistence MAE/MSE.
- Exact metric, unit, horizon, split, value, tags and resource roundtrip.
- Duplicate resubmission does not duplicate rows.
- Disposing and reopening the connection retains the counts.
- NaN and positive/negative infinity are rejected with zero terminal rows.
- 2 tests passed. Fixture values are fabricated transport tests, not model scores.

Read-only query against the live warehouse returned HTTP 200 and 24 historic
rows each for `sota.test.mse_normalized` and `sota.test.mae_normalized`.
Historic records use `unit=z` for both. They remain unchanged. The new
convention uses `z^2` for MSE and `z` for MAE. No historical numeric score was
changed or reinterpreted as a new measurement.

## Query template (DuckDB, read-only)

```sql
SELECT t.campaign_key, t.unit_id, t.generation,
       json_extract_string(t.tags_json, '$.dataset') AS dataset,
       json_extract_string(t.tags_json, '$.model') AS model,
       json_extract_string(t.tags_json, '$.seed') AS seed,
       json_extract_string(t.tags_json, '$.protocol_sha256') AS protocol_sha256,
       json_extract_string(t.tags_json, '$.evaluation_population_sha256') AS population_sha256,
       m.metric, m.value, m.unit, m.split, m.horizon
FROM gov_terminal t
JOIN gov_terminal_metric m USING (terminal_sha256)
WHERE json_extract_string(t.tags_json, '$.metric_contract') = 'tsl_literature_metrics.v1'
  AND t.status = 'COMPLETED'
ORDER BY t.campaign_key, t.unit_id, t.generation, m.metric
LIMIT 1000
```

This query is an inventory, not a scientific closure: select accepted runs and
adjudicated generations before aggregating. It currently returns no new
scientific results because no benchmark run was launched for this setup.

## Formula sources checked

- [iTransformer author metrics](https://raw.githubusercontent.com/thuml/iTransformer/main/utils/metrics.py)
- [PatchTST author metrics](https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/utils/metrics.py)

Both implement mean absolute error and mean squared error directly on the
arrays handed to the scorer. These URLs document the formula check; production
experiments must pin the actual code commit rather than rely on moving main.
