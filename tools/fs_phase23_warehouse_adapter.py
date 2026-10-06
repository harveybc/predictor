#!/usr/bin/env python3
"""Thin warehouse adapter with the DATA agent's documented interface.

MARKED FOR REPLACEMENT.  The authoritative module is ``tools/fs_phase23_warehouse.py``
(DATA agent: additive migration on the deployed DuckDB warehouse).  The campaign resolves
that module first and falls back to this adapter only when it is absent.  This adapter
implements the same three calls over a throwaway file database so the driver can be
tested end to end without the live service:

    submit_rows(run_id, table, rows) -> receipt
    read_run(run_id, table, unit_id=None) -> list[dict]
    reconcile(run_id) -> {"run_id", "tables": {table: {"count", "rows_sha256"}}}

Rows are stored with a unique key ``(run_id, table, row_key)`` so a resubmission of the
same rows inserts nothing (the receipt reports ``duplicates_ignored``).  The receipt
carries the run identity and the digest of the submitted rows: the campaign rejects a
receipt whose identity is not the population identity or whose digest does not match.

Backend: DuckDB when importable, otherwise sqlite3 (standard library).  The backend name
is reported in every receipt so evidence never hides which engine held the rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - depends on the environment
    import duckdb  # type: ignore
except Exception:  # noqa: BLE001
    duckdb = None

TABLES = (
    "feature_pair_metrics", "feature_pair_stability", "feature_pair_gate",
    "feature_alias_groups", "feature_redundancy_clusters",
    "feature_filter_rankings", "feature_filter_subsets",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def rows_digest(rows: list[dict]) -> str:
    """Order-independent digest of a row set: sha256 of sorted per-row digests."""
    parts = sorted(hashlib.sha256(_canonical(r)).hexdigest() for r in rows)
    return hashlib.sha256("".join(parts).encode()).hexdigest()


class WarehouseAdapter:
    schema_version = "fs_phase23_warehouse_adapter.v1"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = "duckdb" if duckdb is not None else "sqlite3"
        if self.backend == "duckdb":
            self._con = duckdb.connect(str(self.path))
        else:
            self._con = sqlite3.connect(str(self.path), timeout=60)
        self._migrate()

    # ----------------------------------------------------------------- schema (additive)
    def _migrate(self) -> None:
        for table in TABLES:
            self._execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "run_id VARCHAR NOT NULL, unit_id VARCHAR, row_key VARCHAR NOT NULL, "
                "row_sha256 VARCHAR NOT NULL, submitted_at DOUBLE, payload VARCHAR NOT NULL, "
                "PRIMARY KEY (run_id, row_key))")
        self._execute(
            "CREATE TABLE IF NOT EXISTS fs_phase23_receipts ("
            "receipt_sha256 VARCHAR PRIMARY KEY, run_id VARCHAR, table_name VARCHAR, unit_id VARCHAR, "
            "row_count INTEGER, inserted INTEGER, duplicates_ignored INTEGER, rows_sha256 VARCHAR, submitted_at DOUBLE)")
        self._commit()

    def _execute(self, sql: str, params: tuple = ()):
        if self.backend == "duckdb":
            return self._con.execute(sql.replace("?", "?"), params)
        return self._con.execute(sql, params)

    def _commit(self) -> None:
        if self.backend == "sqlite3":
            self._con.commit()

    # ----------------------------------------------------------------- interface
    def submit_rows(self, run_id: str, table: str, rows: list[dict]) -> dict:
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        if not run_id:
            raise ValueError("run_id is required")
        now = time.time()
        inserted = 0
        duplicates = 0
        unit_ids = set()
        for row in rows:
            if row.get("run_id") != run_id:
                raise ValueError(f"row run_id {row.get('run_id')!r} != submitted run_id {run_id!r}")
            key = row.get("row_key")
            if not key:
                raise ValueError("every row needs a row_key")
            unit_ids.add(row.get("unit_id"))
            payload = _canonical(row).decode()
            sha = hashlib.sha256(payload.encode()).hexdigest()
            if self.backend == "duckdb":
                exists = self._con.execute(f"SELECT 1 FROM {table} WHERE run_id=? AND row_key=?", (run_id, key)).fetchone()
                if exists:
                    duplicates += 1
                    continue
                self._con.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", (run_id, row.get("unit_id"), key, sha, now, payload))
                inserted += 1
            else:
                cur = self._con.execute(
                    f"INSERT OR IGNORE INTO {table} VALUES (?,?,?,?,?,?)", (run_id, row.get("unit_id"), key, sha, now, payload))
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    duplicates += 1
        digest = rows_digest(rows)
        unit_id = next(iter(unit_ids)) if len(unit_ids) == 1 else None
        receipt = {
            "schema": "fs_phase23.warehouse_receipt.v1", "backend": self.backend, "run_id": run_id, "table": table,
            "unit_id": unit_id, "row_count": len(rows), "inserted": inserted, "duplicates_ignored": duplicates,
            "rows_sha256": digest, "submitted_at": now,
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        self._execute("INSERT INTO fs_phase23_receipts VALUES (?,?,?,?,?,?,?,?,?)",
                      (receipt["receipt_sha256"], run_id, table, unit_id, len(rows), inserted, duplicates, digest, now))
        self._commit()
        return receipt

    def read_run(self, run_id: str, table: str, unit_id: str | None = None) -> list[dict]:
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        if unit_id is None:
            cur = self._execute(f"SELECT payload FROM {table} WHERE run_id=? ORDER BY row_key", (run_id,))
        else:
            cur = self._execute(f"SELECT payload FROM {table} WHERE run_id=? AND unit_id=? ORDER BY row_key", (run_id, unit_id))
        return [json.loads(p[0]) for p in cur.fetchall()]

    def reconcile(self, run_id: str) -> dict:
        tables = {}
        for table in TABLES:
            cur = self._execute(f"SELECT row_sha256 FROM {table} WHERE run_id=?", (run_id,))
            shas = sorted(r[0] for r in cur.fetchall())
            tables[table] = {"count": len(shas), "rows_sha256": hashlib.sha256("".join(shas).encode()).hexdigest()}
        return {"schema": "fs_phase23.warehouse_reconcile.v1", "backend": self.backend, "run_id": run_id, "tables": tables,
                "reconciled_at": time.time()}

    def snapshot(self, out_path: Path) -> dict:
        """Physical copy of the throwaway database (the deployed warehouse snapshots through its owner)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "duckdb":
            self._con.execute("CHECKPOINT")
        else:
            self._con.commit()
        data = self.path.read_bytes()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, out_path)
        return {"path": str(out_path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "backend": self.backend}

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass


def open_warehouse(path: Path) -> WarehouseAdapter:
    return WarehouseAdapter(Path(path))
