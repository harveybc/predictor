"""Read-only cube access. Never truncates. SELECT only."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

_WRITE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|copy|grant|revoke|create|vacuum|call|do)\b",
    re.I,
)
_BANNED = re.compile(
    r"\b(pg_sleep|pg_read_file|pg_ls_dir|lo_import|dblink|file_fdw)\b",
    re.I,
)
_LIMIT = re.compile(r"\blimit\s+(\d+)\s*;?\s*$", re.I)
MAX_LIMIT = 5000


class Plugin:
    plugin_params = {
        "sqlite_path": None,
        "schema": "public",
        "holdout_start": "2025-01-01",
        "time_column": None,
        "lake_id": "olap_cube",
        "title": "predictor OLAP",
        "description": "",
        "kind": "sql_olap",
    }

    def __init__(self):
        self.params = dict(self.plugin_params)
        self._engine = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        self._engine = None

    def engine(self):
        if self._engine is not None:
            return self._engine
        sqlite_path = self.params.get("sqlite_path")
        if sqlite_path:
            self._engine = create_engine(f"sqlite:///{sqlite_path}")
            return self._engine
        host = os.getenv("PGHOST", "127.0.0.1")
        port = os.getenv("PGPORT", "5432")
        db = os.getenv("PGDATABASE", "predictor_olap")
        user = os.getenv("PGUSER") or "metabase"
        password = os.getenv("PGPASSWORD")
        if not password:
            raise RuntimeError("PGPASSWORD is not set")
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        self._engine = create_engine(url, pool_pre_ping=True)
        return self._engine

    def discover(self):
        insp = inspect(self.engine())
        dialect = self.engine().dialect.name
        schema = None if dialect == "sqlite" else (self.params.get("schema") or "public")
        names = insp.get_table_names(schema=schema)
        items = []
        with self.engine().connect() as conn:
            for name in names:
                if schema:
                    q = text(f'SELECT COUNT(*) FROM "{schema}"."{name}"')
                else:
                    q = text(f'SELECT COUNT(*) FROM "{name}"')
                n = conn.execute(q).scalar()
                items.append({"resource_id": name, "kind": "table", "rows": int(n or 0)})
        return items

    def list_resources(self):
        return self.discover()

    def _guard(self, sql: str) -> str:
        text_sql = (sql or "").strip().rstrip(";").strip()
        if ";" in text_sql:
            raise ValueError("multiple statements are not allowed")
        head = text_sql.split(None, 1)[0].lower() if text_sql else ""
        if head not in {"select", "with"}:
            raise ValueError("only SELECT/WITH is allowed")
        if _WRITE.search(text_sql):
            raise ValueError("write SQL is not allowed")
        if _BANNED.search(text_sql):
            raise ValueError("function is not allowed")
        match = _LIMIT.search(text_sql)
        if not match:
            raise ValueError("LIMIT n is required")
        if int(match.group(1)) > MAX_LIMIT:
            raise ValueError("LIMIT too large")
        return text_sql

    def query(self, sql: str):
        guarded = self._guard(sql)
        with self.engine().connect() as conn:
            result = conn.execute(text(guarded))
            rows = [dict(row._mapping) for row in result]
        holdout = self.params.get("holdout_start")
        if holdout:
            limit = date.fromisoformat(str(holdout)[:10])
            time_cols = [self.params.get("time_column")] if self.params.get("time_column") else []
            if rows:
                for key in rows[0]:
                    lk = str(key).lower()
                    if lk in {"ts", "time", "date", "datetime", "timestamp"} or "time" in lk or "date" in lk:
                        if key not in time_cols:
                            time_cols.append(key)
            for row in rows:
                for col in time_cols:
                    if not col or col not in row or not row[col]:
                        continue
                    try:
                        day = date.fromisoformat(str(row[col])[:10])
                    except ValueError:
                        continue
                    if day >= limit:
                        raise PermissionError("holdout")
        canonical = json.dumps(rows, default=str, sort_keys=True, separators=(",", ":"))
        return {
            "rows": rows,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "bytes": len(canonical.encode()),
        }

    def storage(self):
        import shutil

        sqlite_path = self.params.get("sqlite_path")
        if sqlite_path:
            path = Path(sqlite_path)
            usage = shutil.disk_usage(path.parent if path.parent.exists() else Path("."))
            size = path.stat().st_size if path.exists() else 0
            root = str(path)
        else:
            usage = shutil.disk_usage(".")
            size = 0
            root = "postgresql (PGHOST)"
        return {
            "root": root,
            "host_total": usage.total,
            "host_used": usage.used,
            "host_free": usage.free,
            "lake_bytes": size,
        }

    def describe(self):
        try:
            n_resources = len(self.discover())
        except Exception:
            n_resources = None
        return {
            "lake_id": self.params.get("lake_id"),
            "title": self.params.get("title"),
            "description": self.params.get("description"),
            "kind": self.params.get("kind"),
            "root_path": self.storage()["root"],
            "holdout_start": self.params.get("holdout_start"),
            "n_resources": n_resources,
        }
