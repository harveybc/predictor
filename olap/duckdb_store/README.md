# predictor-duckdb-store

The predictor OLAP cube on **DuckDB**, as a backend of a `data-warehouse` host.

DuckDB is the OLAP engine for the current campaign. PostgreSQL references elsewhere in this
repository are either the **legacy migration source** or unrelated services (Metabase's own
application database, other projects) — neither is affected by this package.

## What it is

An external provider registered through the host's existing entry-point group, with the same
capability names and the same governed result contracts as the PostgreSQL provider:

    datawarehouse.backends: predictor_duckdb = predictor_duckdb_store.provider:backend

It **subclasses** `predictor_olap_store.query.Plugin` instead of reimplementing it. The
governed schema, the availability-contract dimension and the temporal validation the producer's
contract requires are shared code: a second copy of those rules would agree with the first
until the day it did not, and the entire point of the dimension is that two readers cannot
disagree about what a contract says.

What it overrides is only what the engine decides:

| concern | why it differs |
|---|---|
| connection | a file on a local persistent volume, no server and no credentials |
| schema qualification | DuckDB's default schema is `main`, not `public` |
| schema creation | `duckdb_engine` presents as a PostgreSQL dialect, so SQLAlchemy tries to set `AUTOCOMMIT` through psycopg2 and the DuckDB connection raises; DDL runs in ordinary transactions |
| writes | DuckDB admits exactly one writer, so this process serialises its own writes |
| `storage` / `describe` | reported from the file, including `root`, which the host's `describe` reads |
| `running_engine` | what the RUNNING process uses — library version, open file, inode, pid — not what the configuration says |

## Configuration

```json
{
  "backend": {
    "distribution": "predictor-duckdb-store",
    "entry_point": "predictor_duckdb",
    "settings": {
      "duckdb_path": "/home/<user>/.local/state/crispdm-duckdb/prod/cube.duckdb",
      "schema": "main",
      "memory_limit": "2GB",
      "threads": 2,
      "min_free_bytes": 2147483648,
      "holdout_start": "2025-01-01",
      "lake_id": "olap_cube"
    }
  }
}
```

No credential appears in the configuration, because there is nothing to authenticate to.
`min_free_bytes` makes the provider refuse to open the cube on a volume that is nearly full,
rather than run out of disk during a write and leave a file nobody can explain.

## One owner

**A worker never opens the cube file.** DuckDB takes an exclusive lock, so a second process —
even a reader — is refused while the service holds it. That is the rule working, not a fault:
all access goes through the warehouse host, and analytics that need the file directly must wait
for a snapshot (see below) rather than reach past the owner.

## Operating it

```bash
# what the running process actually uses
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:5057/api/v1/host

# a bounded read-only query; LIMIT is required
curl -s -G -H "Authorization: Bearer $TOKEN" \
     --data-urlencode "sql=SELECT status, count(*) n FROM gov_terminal GROUP BY 1 LIMIT 20" \
     http://127.0.0.1:5057/api/v1/query

# the operator console: inventory, relation schema, bounded query
xdg-open http://127.0.0.1:5057/
```

### Taking a snapshot

`cp cube.duckdb` is **not** a backup. Recent transactions live in `cube.duckdb.wal`, and a copy
of the main file alone silently omits them — measured: a copy taken while two freshly accepted
terminals were outstanding contained neither. Use:

```bash
python tools/olap_duckdb_migrate.py snapshot \
    --source .../cube.duckdb --target .../snapshot.duckdb \
    --expect-terminals <what the service reports> --out SNAPSHOT.json
```

It copies both files, `CHECKPOINT`s the copy, reopens it and verifies the count. A snapshot
nobody read is a belief, not a backup.

## Analytics

The deployed Metabase (v0.56.3) ships 17 drivers and **none of them is DuckDB** — verified
against its own `/api/session/properties`, not assumed. Analytical access is therefore the
host's console and `/api/v1/query` today. That parity gap is declared in the work plan with an
owner; PostgreSQL is **not** kept quietly serving the cube to paper over it.
