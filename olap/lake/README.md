# OLAP cube lake service

Read-only operator UI + HTTP adapter for [data-gov](https://github.com/harveybc/data-gov).

Connects with `PGHOST` `PGPORT` `PGDATABASE` `PGUSER` `PGPASSWORD` (same as
`olap/init_db.py`). **Does not** run `reset_olap.py`. SELECT only.

- UI: http://127.0.0.1:5057
- API: `/api/v1/discover`, `/api/v1/query`

```bash
cd olap/lake
pip install -r requirements.txt
pip install -e .
python3 -m pytest tests -q
sh scripts/serve.sh
```
