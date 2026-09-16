"""The DuckDB engine under the predictor cube's own governed logic.

What is different from PostgreSQL, and is handled here rather than assumed away:

* **one writer.** A DuckDB file admits a single writing process. That is not a limitation to
  work around — the order requires exactly one warehouse-owned process — but it does mean
  writes inside this process must be serialised, and that analytics must not be able to hold
  the write path open while outcomes queue behind it;
* **no server, so no credentials.** The database is a file on a local persistent volume. The
  configuration names the path; nothing embeds a secret;
* **bounded resources.** Memory limit, thread count and a free-space check are declared and
  applied on every connection, because an analytical engine that is asked for a large scan will
  otherwise take whatever the host has.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from predictor_olap_store.query import Plugin as _Cube
from sqlalchemy import create_engine, text

#: The capability surface, identical to the PostgreSQL provider's. The host resolves backends
#: by these names; a DuckDB cube that offered fewer would silently change what governance can do.
CAPABILITIES = ("describe", "storage", "discover", "schema", "query",
                "write_metrics", "write_terminal", "terminal_digests",
                "write_availability_contracts", "resolve_delivery_availability")

#: Refuse to open a database on a volume with less free space than this. An OLAP engine that
#: runs out of disk mid-write leaves a file nobody can explain.
DEFAULT_MIN_FREE_BYTES = 2 * 1024 ** 3


class DuckdbUnavailable(RuntimeError):
    """The database cannot be opened safely. Reported, never worked around."""


class PredictorDuckdbStore(_Cube):
    plugin_params = {
        **_Cube.plugin_params,
        "duckdb_path": None,
        # DuckDB's own default schema. `public` is created if a deployment asks for it, so a
        # query written against the PostgreSQL cube keeps working unchanged.
        "schema": "main",
        "memory_limit": "2GB",
        "threads": 2,
        "min_free_bytes": DEFAULT_MIN_FREE_BYTES,
        "read_only": False,
        "kind": "duckdb_olap",
        "title": "predictor OLAP (DuckDB)",
    }

    def __init__(self):
        super().__init__()
        # Serialises this process's writes. DuckDB permits one writer; without this, two
        # concurrent governed terminals inside the host would contend at the file level and
        # surface as an opaque IO error instead of simply queueing.
        self._write_lock = threading.RLock()

    # -- engine ---------------------------------------------------------------
    def _database_path(self) -> Path:
        path = self.params.get("duckdb_path")
        if not path:
            raise DuckdbUnavailable("duckdb_path is required: this provider owns a file")
        return Path(str(path)).expanduser()

    def _check_space(self, path: Path) -> None:
        minimum = int(self.params.get("min_free_bytes") or 0)
        if minimum <= 0:
            return
        target = path.parent if path.parent.exists() else Path(path.anchor or ".")
        free = shutil.disk_usage(target).free
        if free < minimum:
            raise DuckdbUnavailable(
                f"{target} has {free} bytes free, below the declared minimum {minimum}; "
                "refusing to open the cube rather than run out of disk during a write")

    def engine(self):
        if self._engine is not None:
            return self._engine
        path = self._database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._check_space(path)
        config = {
            "memory_limit": str(self.params.get("memory_limit") or "2GB"),
            "threads": int(self.params.get("threads") or 2),
        }
        if self.params.get("read_only"):
            config["access_mode"] = "READ_ONLY"
        engine = create_engine(f"duckdb:///{path}", connect_args={"config": config})
        self._engine = engine
        self._ensure_schema(engine)
        return engine

    def write_engine(self):
        """One engine: DuckDB has a single writer and no separate write credentials."""
        return self.engine()

    def _qualified(self, name: str) -> str:
        schema = self.params.get("schema") or "main"
        return f'"{schema}"."{name}"'

    def _ddl(self, dialect: str):
        statements = list(super()._ddl(dialect))
        schema = self.params.get("schema") or "main"
        if schema != "main":
            statements.insert(0, f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        return statements

    def _ensure_schema(self, engine):
        """The same DDL, applied without an isolation level DuckDB does not have.

        `duckdb_engine` presents itself as a PostgreSQL dialect, so SQLAlchemy tries to set
        `AUTOCOMMIT` through psycopg2's own call and the DuckDB connection raises
        `AttributeError: no attribute 'set_isolation_level'`. DDL here runs in ordinary
        transactions instead, which DuckDB supports; the statements themselves are unchanged,
        so the schema this cube gets is the schema the PostgreSQL cube gets.
        """
        from sqlalchemy import inspect

        with engine.begin() as conn:
            for statement in self._ddl(engine.dialect.name):
                conn.execute(text(statement))
        schema = self.params.get("schema") or "main"
        columns = {column["name"] for column in
                   inspect(engine).get_columns("gov_terminal_dataset", schema=schema)}
        if "availability_contract_sha256" not in columns:
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {self._qualified('gov_terminal_dataset')} "
                    "ADD COLUMN availability_contract_sha256 TEXT"))

    # -- write serialisation ---------------------------------------------------
    def write_metrics(self, report):
        with self._write_lock:
            return super().write_metrics(report)

    def write_terminal(self, terminal):
        with self._write_lock:
            return super().write_terminal(terminal)

    def write_availability_contracts(self, contracts):
        with self._write_lock:
            return super().write_availability_contracts(contracts)

    def schema(self, resource_id):
        """The host's console asks for `schema`; the cube implements `resource_schema`.

        Declared here because this provider offers it: the PostgreSQL provider does not list
        the capability, so its console page says the store exposes no schema metadata. Same
        underlying call, one name the host actually looks for.
        """
        return self.resource_schema(resource_id)

    # -- identity --------------------------------------------------------------
    def capabilities(self):
        return CAPABILITIES

    def describe(self):
        body = super().describe() if hasattr(_Cube, "describe") else {}
        path = None
        try:
            path = str(self._database_path())
        except DuckdbUnavailable:
            pass
        body.update(kind="duckdb_olap", engine="duckdb", database_path=path,
                    memory_limit=self.params.get("memory_limit"),
                    threads=self.params.get("threads"))
        return body

    def storage(self):
        """Bytes on disk, from the file itself rather than from an engine estimate."""
        try:
            path = self._database_path()
        except DuckdbUnavailable:
            return {"bytes": None}
        total = 0
        for candidate in (path, Path(str(path) + ".wal")):
            if candidate.is_file():
                total += candidate.stat().st_size
        return {"bytes": total, "path": str(path),
                "free_bytes": shutil.disk_usage(path.parent).free if path.parent.exists()
                else None}

    def source_identity(self):
        import duckdb

        from . import __version__

        return {"kind": "python_distribution", "distribution": "predictor-duckdb-store",
                "version": __version__, "module": __name__,
                "engine": "duckdb", "engine_version": duckdb.__version__,
                "database_path": str(self._database_path()) if self.params.get("duckdb_path")
                else None}

    def running_engine(self):
        """What the RUNNING process is actually using — D5 asks for this, not configuration.

        Reported from the live connection and the open file, so a host that was started with
        one configuration and is serving another cannot claim the configuration's answer.
        """
        import duckdb

        with self.engine().connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar()
            databases = [dict(row._mapping) for row in conn.execute(text("PRAGMA database_list"))]
        path = self._database_path()
        return {"engine": "duckdb", "library_version": duckdb.__version__,
                "server_version": version, "databases": databases,
                "database_path": str(path), "exists": path.is_file(),
                "inode": path.stat().st_ino if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
                "pid": os.getpid()}


def backend():
    return PredictorDuckdbStore()
