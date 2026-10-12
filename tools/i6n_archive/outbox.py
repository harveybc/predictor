"""Durable local outbox/inbox for the lane D archive (one SQLite file per campaign root).

This file is the publisher's own state, not an analytic store and not a warehouse
substitute. Four states are kept in separate columns and each is written only by the step
that verified it:

* ``LOCAL_COMPLETE``          canonical body + manifest bytes stored here with their digests;
* ``OBJECT_DURABLE``          the lake answered the digest on PUT and an exact GET re-verified it;
* ``CHAIN_FINALIZED``         the reference tx sits on a verified chain at >= confirmation depth;
* ``OLAP_READBACK_VERIFIED``  every projected report was read back from the warehouse exactly.

A missing state is ``None`` and is never shown as green. A reorg clears CHAIN_FINALIZED and
OLAP_READBACK_VERIFIED for affected items; the append-only history keeps what happened.
The inbox keys projections by (row key, block hash, finality state) = projection_key.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATES = ("LOCAL_COMPLETE", "OBJECT_DURABLE", "CHAIN_FINALIZED", "OLAP_READBACK_VERIFIED")
_COLUMN = {s: s.lower() for s in STATES}

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    search_candidate_sha256 TEXT NOT NULL,
    archive_mode TEXT NOT NULL,
    body BLOB NOT NULL,
    manifest BLOB NOT NULL,
    body_digest TEXT NOT NULL,
    manifest_digest TEXT NOT NULL,
    tx_timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reference_json TEXT,
    reference_tx_id TEXT,
    local_complete TEXT,
    object_durable TEXT,
    chain_finalized TEXT,
    olap_readback_verified TEXT
);
CREATE TABLE IF NOT EXISTS item_history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL, event TEXT NOT NULL,
    detail TEXT NOT NULL, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    projection_key TEXT PRIMARY KEY,
    item_id TEXT,
    row_key_sha256 TEXT NOT NULL,
    report_sha256 TEXT NOT NULL,
    block_hash TEXT NOT NULL,
    tx_id TEXT NOT NULL,
    finality_state TEXT NOT NULL,
    state TEXT NOT NULL,
    receipt TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS row_index (
    row_key_sha256 TEXT PRIMARY KEY, report_sha256 TEXT NOT NULL, item_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS etl_done (
    tx_id TEXT NOT NULL, block_hash TEXT NOT NULL, item_id TEXT, reports INTEGER NOT NULL, done_at TEXT NOT NULL,
    PRIMARY KEY (tx_id, block_hash)
);
CREATE INDEX IF NOT EXISTS items_body_idx ON items (body_digest, archive_mode);
CREATE TABLE IF NOT EXISTS inbox_history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, projection_key TEXT NOT NULL, state TEXT NOT NULL,
    detail TEXT NOT NULL, at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Outbox:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(_DDL)

    def close(self) -> None:
        self._conn.close()

    # -- meta ------------------------------------------------------------------------------
    def meta(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self._conn.execute("SELECT key, value FROM meta")}

    def bind_meta(self, values: dict) -> None:
        """First binding wins; a different later binding is a refusal, not an overwrite."""
        current = self.meta()
        for key, value in values.items():
            if key in current and current[key] != value:
                raise ValueError(f"OUTBOX_BOUND_TO_OTHER_{key.upper()}")
        with self._tx():
            for key, value in values.items():
                self._conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, _dump(value)))

    # -- items -----------------------------------------------------------------------------
    def _tx(self):
        conn = self._conn

        class _T:
            def __enter__(self):
                conn.execute("BEGIN IMMEDIATE")

            def __exit__(self, kind, exc, tb):
                conn.execute("ROLLBACK" if kind else "COMMIT")
                return False

        return _T()

    def insert_item(self, *, item_id: str, experiment: str, search_candidate_sha256: str, archive_mode: str,
                    body: bytes, manifest: bytes, body_digest: str, manifest_digest: str,
                    tx_timestamp: str, row_reports: dict | None = None) -> bool:
        """``row_reports`` maps each projected row key to its report digest. A row key already
        bound to another report is refused HERE (ROW_KEY_CONFLICT), never later at the ETL."""
        with self._tx():
            for row_key, report in sorted((row_reports or {}).items()):
                bound = self._conn.execute("SELECT report_sha256, item_id FROM row_index WHERE row_key_sha256 = ?",
                                           (row_key,)).fetchone()
                if bound is not None and bound["report_sha256"] != report:
                    from tools.i6n_archive.records import ArchiveRecordError

                    raise ArchiveRecordError(f"ROW_KEY_CONFLICT: row {row_key[:12]} is already archived with "
                                             f"different content (item {bound['item_id'][:12]})")
            row = self._conn.execute("SELECT body, manifest FROM items WHERE item_id = ?", (item_id,)).fetchone()
            if row is not None:
                if bytes(row["body"]) != body or bytes(row["manifest"]) != manifest:
                    raise ValueError("OUTBOX_CONFLICT: same item id, different bytes")
                return False
            evidence = {"body_digest": body_digest, "manifest_digest": manifest_digest,
                        "body_size": len(body), "verified_at": _now()}
            self._conn.execute(
                "INSERT INTO items (item_id, experiment, search_candidate_sha256, archive_mode, body, manifest,"
                " body_digest, manifest_digest, tx_timestamp, created_at, local_complete)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (item_id, experiment, search_candidate_sha256, archive_mode, body, manifest, body_digest,
                 manifest_digest, tx_timestamp, _now(), _dump(evidence)))
            self._history(item_id, "LOCAL_COMPLETE", evidence)
            for row_key, report in sorted((row_reports or {}).items()):
                self._conn.execute("INSERT OR IGNORE INTO row_index (row_key_sha256, report_sha256, item_id)"
                                   " VALUES (?,?,?)", (row_key, report, item_id))
        return True

    def _history(self, item_id: str, event: str, detail: Any) -> None:
        self._conn.execute("INSERT INTO item_history (item_id, event, detail, at) VALUES (?,?,?,?)",
                           (item_id, event, _dump(detail), _now()))

    def _row(self, row) -> dict:
        out = {k: row[k] for k in ("item_id", "experiment", "search_candidate_sha256", "archive_mode",
                                   "body_digest", "manifest_digest", "tx_timestamp", "created_at",
                                   "reference_tx_id")}
        out["reference"] = json.loads(row["reference_json"]) if row["reference_json"] else None
        out["states"] = {s: (json.loads(row[_COLUMN[s]]) if row[_COLUMN[s]] else None) for s in STATES}
        out["green"] = all(out["states"][s] for s in STATES)
        return out

    _META = ("item_id, experiment, search_candidate_sha256, archive_mode, body_digest, manifest_digest,"
             " tx_timestamp, created_at, reference_json, reference_tx_id, local_complete, object_durable,"
             " chain_finalized, olap_readback_verified")  # never the BLOBs

    def item(self, item_id: str) -> dict | None:
        row = self._conn.execute(f"SELECT {self._META} FROM items WHERE item_id = ?", (item_id,)).fetchone()
        return self._row(row) if row else None

    def item_by_body(self, body_digest: str, archive_mode: str) -> dict | None:
        row = self._conn.execute(f"SELECT {self._META} FROM items WHERE body_digest = ? AND archive_mode = ?",
                                 (body_digest, archive_mode)).fetchone()
        return self._row(row) if row else None

    def items(self) -> list[dict]:
        return [self._row(r) for r in self._conn.execute(
            f"SELECT {self._META} FROM items ORDER BY created_at, item_id")]

    # -- ETL watermark: a (tx, block) is done once its projection was read back exactly -------
    def done_keys(self) -> set[tuple[str, str]]:
        return {(r[0], r[1]) for r in self._conn.execute("SELECT tx_id, block_hash FROM etl_done")}

    def mark_done(self, tx_id: str, block_hash: str, item_id: str | None, reports: int) -> None:
        with self._tx():
            self._conn.execute("INSERT OR IGNORE INTO etl_done (tx_id, block_hash, item_id, reports, done_at)"
                               " VALUES (?,?,?,?,?)", (tx_id, block_hash, item_id, reports, _now()))

    def forget_block(self, block_hash: str) -> int:
        with self._tx():
            return self._conn.execute("DELETE FROM etl_done WHERE block_hash = ?", (block_hash,)).rowcount

    def bytes_of(self, item_id: str) -> tuple[bytes, bytes]:
        row = self._conn.execute("SELECT body, manifest FROM items WHERE item_id = ?", (item_id,)).fetchone()
        return bytes(row["body"]), bytes(row["manifest"])

    def set_state(self, item_id: str, state: str, evidence: dict) -> None:
        if state not in STATES or state == "LOCAL_COMPLETE":
            raise ValueError("UNKNOWN_OR_IMMUTABLE_STATE")
        with self._tx():
            self._conn.execute(f"UPDATE items SET {_COLUMN[state]} = ? WHERE item_id = ?", (_dump(evidence), item_id))
            self._history(item_id, state, evidence)

    def clear_state(self, item_id: str, state: str, event: str, detail: dict) -> None:
        with self._tx():
            self._conn.execute(f"UPDATE items SET {_COLUMN[state]} = NULL WHERE item_id = ?", (item_id,))
            self._history(item_id, event, dict(detail, cleared=state))

    def set_reference(self, item_id: str, tx_id: str, reference: dict) -> None:
        with self._tx():
            self._conn.execute("UPDATE items SET reference_tx_id = ?, reference_json = ? WHERE item_id = ?",
                               (tx_id, _dump(reference), item_id))
            self._history(item_id, "REFERENCE_SUBMITTED", {"tx_id": tx_id})

    def clear_reference(self, item_id: str, reason: str) -> None:
        with self._tx():
            self._conn.execute("UPDATE items SET reference_tx_id = NULL WHERE item_id = ?", (item_id,))
            self._history(item_id, "REFERENCE_ORPHANED", {"reason": reason})

    def history(self, item_id: str) -> list[dict]:
        return [{"seq": r["seq"], "event": r["event"], "detail": json.loads(r["detail"]), "at": r["at"]}
                for r in self._conn.execute("SELECT * FROM item_history WHERE item_id = ? ORDER BY seq", (item_id,))]

    # -- inbox -----------------------------------------------------------------------------
    def upsert_projection(self, *, projection_key: str, item_id: str | None, row_key_sha256: str,
                          report_sha256: str, block_hash: str, tx_id: str, finality_state: str, state: str,
                          receipt: dict) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT INTO inbox (projection_key, item_id, row_key_sha256, report_sha256, block_hash, tx_id,"
                " finality_state, state, receipt, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(projection_key) DO UPDATE SET state = excluded.state, receipt = excluded.receipt,"
                " updated_at = excluded.updated_at",
                (projection_key, item_id, row_key_sha256, report_sha256, block_hash, tx_id, finality_state, state,
                 _dump(receipt), _now()))
            self._conn.execute("INSERT INTO inbox_history (projection_key, state, detail, at) VALUES (?,?,?,?)",
                               (projection_key, state, _dump(receipt), _now()))

    def mark_block_revoked(self, block_hash: str, detail: dict) -> list[str]:
        with self._tx():
            keys = [r[0] for r in self._conn.execute(
                "SELECT projection_key FROM inbox WHERE block_hash = ? AND state != 'REVOKED'", (block_hash,))]
            for key in keys:
                self._conn.execute("UPDATE inbox SET state = 'REVOKED', updated_at = ? WHERE projection_key = ?",
                                   (_now(), key))
                self._conn.execute("INSERT INTO inbox_history (projection_key, state, detail, at) VALUES (?,?,?,?)",
                                   (key, "REVOKED", _dump(detail), _now()))
        return keys

    def projections(self, state: str | None = None) -> list[dict]:
        sql = "SELECT * FROM inbox" + (" WHERE state = ?" if state else "") + " ORDER BY updated_at"
        rows = self._conn.execute(sql, (state,) if state else ())
        return [dict(r) | {"receipt": json.loads(r["receipt"])} for r in rows]

    def active_blocks(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT DISTINCT block_hash FROM inbox WHERE state != 'REVOKED'")}
