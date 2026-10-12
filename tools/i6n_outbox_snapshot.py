#!/usr/bin/env python3
"""Seal an immutable SQLite snapshot against one optimizer terminal ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

from tools.i6n_archive import records


SNAPSHOT_SCHEMA = "i6n.outbox.snapshot.v1"
TERMINAL_SCHEMA = "i6n.island.terminal.v1"
STABLE_FIELDS = (
    "island_id", "domain_id", "producer_peer_id", "horizon", "seed",
    "plan_sha256", "candidate_ledger_sha256", "evaluated_candidate_count",
    "item_count", "items_sha256",
)


class SnapshotPending(RuntimeError):
    """The source outbox or terminal is not available yet."""


class SnapshotRefusal(RuntimeError):
    """The source, terminal, or retained snapshot contradicts its identity."""


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value) -> str:
    raw = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _read_terminal(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise SnapshotPending("TERMINAL_NOT_AVAILABLE")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotRefusal("TERMINAL_INVALID_JSON") from exc
    if (not isinstance(document, dict) or document.get("schema") != TERMINAL_SCHEMA
            or not isinstance(document.get("payload"), dict)):
        raise SnapshotRefusal("TERMINAL_ENVELOPE_INVALID")
    payload = document["payload"]
    digest = _sha(payload)
    if document.get("payload_sha256") != digest:
        raise SnapshotRefusal("TERMINAL_DIGEST_MISMATCH")
    required = {
        "island_id", "horizon", "seed", "plan_sha256", "contract_sha256",
        "checkpoint_identity_sha256", "candidate_ledger_sha256",
        "evaluated_candidate_count",
    }
    if not required <= set(payload) or payload.get("status") != "COMPLETE":
        raise SnapshotRefusal("TERMINAL_IDENTITY_INCOMPLETE")
    if (payload["horizon"] not in (4, 72)
            or type(payload["evaluated_candidate_count"]) is not int
            or payload["evaluated_candidate_count"] < 1):
        raise SnapshotRefusal("TERMINAL_VALUES_INVALID")
    return payload, digest


def _candidate_summary(payload: dict) -> dict:
    """Recreate the exact cache value hashed by Plugin._record_island_terminal."""
    rows = payload["per_fold"]
    origins_binding = {row["origins_binding"] for row in rows}
    if len(origins_binding) != 1:
        raise SnapshotRefusal("OUTBOX_ORIGINS_BINDING_MIXED")
    binding = next(iter(origins_binding))
    origins = ({row["fold"]: row["origins_sha256"] for row in rows}
               if binding == "VERIFIED" else None)
    metrics = {
        "per_fold": {row["fold"]: row["ratio_to_naive"] for row in rows},
        "receipts": {row["fold"]: row["receipt"] for row in rows},
        "origins_binding": binding,
        "origins_sha256": origins,
        "aggregation": payload["aggregation"],
        "relative_error_vs_naive": payload["fitness"],
        "genotype_canonical_sha256": payload["genotype_canonical_sha256"],
        "effective_config_sha256": payload["effective_config_sha256"],
        "plan_sha256": payload["plan_sha256"],
        "contract_sha256": payload["contract_sha256"],
        "checkpoint_identity_sha256": payload["checkpoint_identity_sha256"],
    }
    aggregate = payload["aggregate_metrics"]
    return {
        "fitness": payload["fitness"],
        "hyper_dict": payload["effective_config"],
        "genotype_canonical_sha256": payload["genotype_canonical_sha256"],
        "train_mae": aggregate["train_mae"],
        "train_naive_mae": aggregate["train_naive_mae"],
        "val_mae": aggregate["inner_mae"],
        "val_naive_mae": aggregate["inner_naive_mae"],
        "metrics": metrics,
    }


def _inspect(path: Path, terminal: dict) -> dict:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise SnapshotRefusal(f"OUTBOX_OPEN_FAILED: {type(exc).__name__}") from exc
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise SnapshotRefusal(f"OUTBOX_QUICK_CHECK_FAILED: {integrity}")
        meta = {row["key"]: json.loads(row["value"])
                for row in connection.execute("SELECT key, value FROM meta")}
        rows = list(connection.execute(
            "SELECT item_id, search_candidate_sha256, body, manifest, "
            "body_digest, manifest_digest FROM items ORDER BY item_id"
        ))
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise SnapshotRefusal(f"OUTBOX_SCHEMA_INVALID: {type(exc).__name__}") from exc
    finally:
        connection.close()

    required_meta = {"store_id", "domain_id", "producer_peer_id", "mode"}
    if set(meta) != required_meta or meta["mode"] != "external_payload":
        raise SnapshotRefusal("OUTBOX_META_INVALID")
    if len(rows) != terminal["evaluated_candidate_count"]:
        raise SnapshotRefusal(
            "OUTBOX_CANDIDATE_COUNT_MISMATCH: expected "
            f"{terminal['evaluated_candidate_count']}, observed {len(rows)}")

    ledger = {}
    item_identities = []
    for row in rows:
        body = bytes(row["body"])
        manifest = bytes(row["manifest"])
        if (_sha(body) != row["body_digest"] or _sha(manifest) != row["manifest_digest"]
                or row["item_id"] != row["body_digest"]):
            raise SnapshotRefusal("OUTBOX_ITEM_DIGEST_MISMATCH")
        try:
            external = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SnapshotRefusal("OUTBOX_BODY_INVALID_JSON") from exc
        if (external.get("domain_id") != meta["domain_id"]
                or external.get("producer_peer_id") != meta["producer_peer_id"]
                or external.get("payload_schema") != records.ISLAND_CANDIDATE_SCHEMA):
            raise SnapshotRefusal("OUTBOX_EXTERNAL_IDENTITY_MISMATCH")
        try:
            payload = records.validate_island_candidate_payload(external.get("payload"))
        except Exception as exc:
            raise SnapshotRefusal(f"OUTBOX_CANDIDATE_INVALID: {exc}") from exc
        for name, expected in (
            ("island_id", terminal["island_id"]),
            ("horizon_hours", terminal["horizon"]),
            ("seed", terminal["seed"]),
            ("plan_sha256", terminal["plan_sha256"]),
            ("contract_sha256", terminal["contract_sha256"]),
            ("checkpoint_identity_sha256", terminal["checkpoint_identity_sha256"]),
        ):
            if payload.get(name) != expected:
                raise SnapshotRefusal(f"OUTBOX_TERMINAL_{name.upper()}_MISMATCH")
        identity = payload["genotype_canonical_sha256"]
        if identity in ledger:
            raise SnapshotRefusal("OUTBOX_DUPLICATE_GENOTYPE")
        ledger[identity] = _candidate_summary(payload)
        item_identities.append({
            "item_id": row["item_id"],
            "search_candidate_sha256": row["search_candidate_sha256"],
            "body_digest": row["body_digest"],
            "manifest_digest": row["manifest_digest"],
        })

    ledger_sha256 = _sha({key: ledger[key] for key in sorted(ledger)})
    if ledger_sha256 != terminal["candidate_ledger_sha256"]:
        raise SnapshotRefusal("OUTBOX_CANDIDATE_LEDGER_MISMATCH")
    return {
        "island_id": terminal["island_id"],
        "domain_id": meta["domain_id"],
        "producer_peer_id": meta["producer_peer_id"],
        "horizon": terminal["horizon"],
        "seed": terminal["seed"],
        "plan_sha256": terminal["plan_sha256"],
        "candidate_ledger_sha256": ledger_sha256,
        "evaluated_candidate_count": len(rows),
        "item_count": len(rows),
        "items_sha256": _sha(item_identities),
        "sqlite_sha256": _sha(path.read_bytes()),
        "size_bytes": path.stat().st_size,
    }


def _write_once(path: Path, document: dict) -> None:
    raw = _canonical(document) + b"\n"
    if path.is_file():
        if path.read_bytes() != raw:
            raise SnapshotRefusal("SNAPSHOT_RECEIPT_CONFLICT")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(raw_tmp, 0o600)
        os.replace(raw_tmp, path)
    finally:
        if os.path.exists(raw_tmp):
            os.unlink(raw_tmp)


def _read_seal(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotRefusal("SNAPSHOT_RECEIPT_INVALID_JSON") from exc
    if (not isinstance(document, dict) or document.get("schema") != SNAPSHOT_SCHEMA
            or not isinstance(document.get("payload"), dict)
            or _sha(document["payload"]) != document.get("payload_sha256")):
        raise SnapshotRefusal("SNAPSHOT_RECEIPT_DIGEST_MISMATCH")
    return document["payload"]


def snapshot(source: str | Path, destination: str | Path, *, terminal_receipt: str | Path,
             receipt: str | Path) -> dict:
    """Back up a live WAL database and retain the first terminal-bound snapshot."""
    source, destination, receipt = Path(source), Path(destination), Path(receipt)
    terminal, terminal_receipt_sha256 = _read_terminal(Path(terminal_receipt))
    if destination.is_file():
        evidence = _inspect(destination, terminal)
        state = "EXISTING_VERIFIED"
        if receipt.is_file():
            sealed = _read_seal(receipt)
            if (any(sealed.get(field) != evidence[field] for field in STABLE_FIELDS)
                    or sealed.get("terminal_receipt_sha256") != terminal_receipt_sha256):
                raise SnapshotRefusal("SNAPSHOT_RECEIPT_CONTENT_MISMATCH")
            return {"state": state, "receipt": str(receipt), **sealed}
    else:
        if not source.is_file():
            raise SnapshotPending("SOURCE_OUTBOX_NOT_AVAILABLE")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        temporary = Path(raw_tmp)
        temporary.unlink()
        try:
            source_connection = sqlite3.connect(
                f"file:{source}?mode=ro", uri=True, timeout=30, isolation_level=None)
            destination_connection = sqlite3.connect(str(temporary), timeout=30)
            try:
                source_connection.backup(destination_connection)
            finally:
                destination_connection.close()
                source_connection.close()
            evidence = _inspect(temporary, terminal)
            # Delivery/finality/readback columns advance after this seal. Candidate
            # bytes and identities are independently re-verified at closure.
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            state = "CREATED"
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    payload = dict(evidence, terminal_receipt_sha256=terminal_receipt_sha256)
    envelope = {"schema": SNAPSHOT_SCHEMA, "payload": payload, "payload_sha256": _sha(payload)}
    _write_once(receipt, envelope)
    return {"state": state, "receipt": str(receipt), **payload}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = snapshot(args.source, args.destination,
                          terminal_receipt=args.terminal_receipt, receipt=args.receipt)
    except SnapshotPending as exc:
        print(json.dumps({"state": "WAIT", "reason": str(exc)}, sort_keys=True))
        return 4
    except SnapshotRefusal as exc:
        print(json.dumps({"state": "REFUSED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
