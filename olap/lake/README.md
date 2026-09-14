# OLAP warehouse service

The warehouse HTTP adapter and AdminLTE operator console for
[data-gov](https://github.com/harveybc/data-gov). This service belongs to
**predictor**, under `olap/lake`; its historical directory and distribution
name (`olap-lake`) are retained for compatibility. The storage model is a
**warehouse**, not a file lake.

It connects to PostgreSQL, inventories tables and views, exposes bounded
SELECT queries, and stores governed experiment outcomes in additive `gov_*`
tables. It does not train predictors or reset the existing cube.

## Quickstart

Use a dedicated environment in this directory, not the predictor training
environment. Python 3.12 is exercised; no minimum is enforced in setup.py.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m app.main --help
python -m pytest tests -q
```

For PostgreSQL, configure `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` and
`PGPASSWORD` in the service environment. The database must already exist.
No database password belongs in a committed config or in the web page.

```bash
export DATA_GOV_LAKE_TOKEN_FILE=/absolute/path/to/data-gov/var/lake_token
sh scripts/serve.sh
```

UI: `http://127.0.0.1:5057`. `/healthz` returns plain `ok`; it checks the
process, not successful database connectivity. Use inventory and a small
query to check the configured database.

For a disposable local database, set `sqlite_path` in your JSON config and
run `python -m app.main --load_config /absolute/path/to/dev.json`. The service
creates its reporting tables; it does not create historical `fact_performance`
or dimension tables. SQLite is a development backend, not evidence that a
PostgreSQL deployment was tested.

## Dashboard, inventory and configuration

- Table and view inventory in the configured schema, without COUNT scans of
  every table. Unknown row counts are shown as **Not scanned**.
- Per-resource schema: column names, SQL types, nullability, defaults, primary
  keys, foreign keys and indexes.
- Bounded SELECT query form with actual results and their serialization hash.
- Pending settings: title, schema, holdout start and governance dashboard URL.
- Locally bundled AdminLTE/Bootstrap CSS, no CDN required for these pages.

Saving settings writes `examples/config/local.json` (or the configured
`operator_config_path`) and leaves the active engine unchanged. Activate
explicitly during a controlled restart:

```bash
sh scripts/serve.sh --load_config examples/config/local.json
```

`data_gov_url` is a navigation link, not registration or a callback. Configure
this warehouse's `base_url` and policies in data-gov separately. The operator
console is local administration, not a remotely hosted multi-tenant SQL editor.
Host disk statistics refer to the **adapter host**, not a remote database host.

## Configuration

| Setting | Meaning |
|---|---|
| `query_plugin` | `sql_query` backend |
| `schema` | PostgreSQL schema, default `public`; SQLite uses its default schema |
| `lake_id` | Stable governance store ID, default `olap_cube` |
| `title`, `description` | Catalog display metadata |
| `web_host`, `web_port` | Listener, default loopback / 5057 |
| `holdout_start`, `time_column` | Existing query-result temporal checks |
| `sqlite_path` | Disposable/local backend instead of PostgreSQL |
| `data_gov_url` | Link to the governance dashboard |
| `PG*` environment | PostgreSQL connection |
| `PGUSER_WRITE`, `PGPASSWORD_WRITE` | Optional separate reporting role |

The default read role also initializes the additive reporting schema. A
separate write role needs the INSERT and SELECT privileges used by reporting
and duplicate detection, not just INSERT. Apply grants for the actual deployed
schema; do not reuse an incomplete legacy three-table grant recipe.

The lake service token comes from `DATA_GOV_LAKE_TOKEN` or
`DATA_GOV_LAKE_TOKEN_FILE`. This token identifies the adapter connection; it
is not the experiment's data-gov API key.

## Connect to data-gov

Add to its `lakes[]` list:

```json
{
  "plugin": "http_lake",
  "lake_id": "olap_cube",
  "title": "Experiment warehouse",
  "kind": "warehouse",
  "engine": "sql_olap",
  "base_url": "http://127.0.0.1:5057"
}
```

Allow the intended principal the `discover`, `query` and `write_terminal`
verbs on that ID. `write_metrics` is legacy compatibility. No extra permission
is conferred by the warehouse label. Follow the
[complete integration examples](https://github.com/harveybc/data-gov/blob/master/docs/INTEGRATION_EXAMPLES.md).

## Tables and result grains

All rows below use the configured PostgreSQL schema (normally `public`). The
adapter creates these objects additively. Existing predictor tables are not
truncated or recreated.

| Object | Grain and purpose |
|---|---|
| `gov_terminal` | One unit/generation outcome, full canonical body, campaign/code/config identity and status |
| `gov_terminal_metric` | Metric/split/horizon attached to a terminal |
| `gov_terminal_dataset` | Verified input delivery and its source/delivered hash, role and temporal evidence |
| `gov_terminal_artifact` | Output artifact role, hash and byte count per terminal |
| `gov_report` | Legacy experiment metric report and report identity |
| `gov_metric` | Metric row attached to a legacy report |
| `gov_dataset` | Dataset lineage attached to a legacy report |
| `gov_metric_current` | Legacy view selecting the current report's metrics; not a Flow v3 terminal view |

The historical OLAP fact/dimension implementation lives in
[`../`](../), with scripts in that directory. Its presence depends on
database initialization and prior ETL runs. Discover the database rather than
assuming a committed schema file has been applied. Business semantics of
arbitrary external tables are not inferred from their column names.

## HTTP interface

API calls require the configured lake service token.

| Method and path | Result |
|---|---|
| `GET /api/v1/describe` | Store metadata |
| `GET /api/v1/storage` | Adapter-host storage observation |
| `GET /api/v1/discover` | Tables and views |
| `GET /api/v1/schema?resource=...` | SQL schema metadata for an inventoried resource |
| `GET /api/v1/query?sql=...` | SELECT results, byte count and canonical JSON hash |
| `POST /api/v1/metrics` | Legacy report write |
| `POST /api/v2/terminals` | Current governed outcome write |
| `GET /api/v2/terminals?campaign_sha256=...` | Terminal identities for reconciliation |

Queries require a trailing `LIMIT`, at most 5,000. Supported query entry
forms are SELECT/WITH; writes use structured reporting, not arbitrary SQL.
Reporting returns 201 for a new record and 200 for an identical retry;
invalid content returns 400, database failure 503. Experiments should call
data-gov instead of invoking these adapter endpoints directly.

## Use with a coding agent

> Work only in predictor/olap/lake with a separate environment. Read this
> README and tests. Create a temporary SQLite database or explicitly disposable
> PostgreSQL database. Verify inventory, schema, bounded query, terminal writes,
> idempotent retry and the data-gov three-service check. Report precise input,
> output and reconciliation evidence. Do not initialize, reset, truncate or
> migrate the real analytical database; do not launch model training.

The package exposes `olaplake.pipeline`, `olaplake.web` and `olaplake.query`
entry points. Add a backend only when behavior cannot be expressed by its
configuration. The proposed external-provider migration is documented in
[data-gov](https://github.com/harveybc/data-gov/blob/master/docs/STORE_PACKAGES_DESIGN.md);
it has not replaced this implementation.

## Tests and limits

Tests cover SQLite queries, reflection, UI, API reporting, duplicate writes and
Flow v3 terminals. `tests/test_write_metrics_pg.py` additionally needs
`DATA_GOV_TEST_PG_URL` pointing to a **throwaway** database and skips otherwise.
Use the data-gov E2E command to verify lake -> governance -> warehouse together.

A result hash is not a versioned database snapshot. Result-level date checks
do not make arbitrary SQL a safe source of causal training windows. No full
database administration, general SQL-write API, external-beta acceptance or
universal consumer adoption is claimed. Preserve third-party asset notices;
dataset rights and repository licensing are separate concerns.
