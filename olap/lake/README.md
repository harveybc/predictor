# OLAP cube lake service

Operator UI + HTTP adapter for [data-gov](https://github.com/harveybc/data-gov)
(`docs/04_FLOW_V2.md` §3 "Lake side").

SELECT for `query`; append-only `gov_*` through `write_metrics`; never
`reset_olap`.

Connects with `PGHOST` `PGPORT` `PGDATABASE` `PGUSER` `PGPASSWORD` (same as
`olap/init_db.py`), or with `sqlite_path` for tests and the lab.

- UI: http://127.0.0.1:5057
- API (lake token): `GET /api/v1/discover`, `GET /api/v1/query`,
  `POST /api/v1/metrics`

## Token

The lake token is read only from `DATA_GOV_LAKE_TOKEN` or the file named by
`DATA_GOV_LAKE_TOKEN_FILE`. `scripts/serve.sh` exports it from the data-gov
`var/lake_token` file for the operator; the code itself never looks there.

## Metrics (`POST /api/v1/metrics`)

Body: the report data-gov forwards (`experiment_key`, `actor`, `lake`,
`metrics`, `datasets` with their lineage, `report_sha256`, at most 16 MiB).
The plugin recomputes `report_sha256` from the canonical body and refuses a
mismatch. Answers `201` stored, `200` already stored (with the lineage stored
the first time), `400` invalid or hash mismatch, `503` database error.

The `gov_report`, `gov_metric`, `gov_dataset` tables, their indexes and the
`gov_metric_current` view are created once at engine creation, in autocommit,
additively (`CREATE ... IF NOT EXISTS`); nothing existing is touched. One
transaction per report: `INSERT ... ON CONFLICT (report_sha256) DO NOTHING`,
then the metric and dataset rows only when the report was new.

### Write role (recommended hardening)

Writes may use a dedicated role through `PGUSER_WRITE` / `PGPASSWORD_WRITE`
(defaulting to `PGUSER` / `PGPASSWORD`). The DDL and the read of a stored
lineage go through `PGUSER`, so the write role needs INSERT only:

```sql
CREATE ROLE gov_writer LOGIN PASSWORD '<password>';
GRANT INSERT ON public.gov_report, public.gov_metric, public.gov_dataset TO gov_writer;
```

Start the service once with `PGUSER` (which owns the tables) so the DDL has
run before the write role is used.

```bash
cd olap/lake
pip install -r requirements.txt
pip install -e .
python -m pytest tests -q
sh scripts/serve.sh
```

The PostgreSQL test in `tests/test_write_metrics_pg.py` runs only when
`DATA_GOV_TEST_PG_URL` names a throwaway database (sqlalchemy URL); it is
skipped otherwise.
