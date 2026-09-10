"""A durable local outbox, so science never waits on a database.

The audit's finding was blunt: the cube is up and queryable, but
nothing feeds it. The only path into it was a human running the
backfill CLI, so every run, candidate and cell that finished in
between simply never arrived.

Coupling the producer directly to PostgreSQL would fix the gap and
create a worse one — a scientific run would then fail because a
database was restarting. So the producer writes here instead:

  * `emit()` appends one envelope or event to a local directory,
    content-addressed, O_EXCL, fsynced. It touches no network and
    raises only if the LOCAL write genuinely fails;
  * a separate CPU loader drains the outbox into the cube,
    retries what it could not load, and records what it did;
  * failures and inconclusive results are emitted exactly like
    successes, because a cube that only receives good news is a
    marketing tool.

The outbox is append-only. Nothing here deletes an entry: a
loaded entry is recorded as loaded, and the evidence stays.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

OUTBOX_ENV = "CRISPDM_OLAP_OUTBOX"
DEFAULT_OUTBOX = (Path.home() / ".local/share/predictor"
                  / "olap_outbox")

PENDING = "pending"
LOADED = "loaded"
FAILED = "failed"

STATE_DIRS = (PENDING, LOADED, FAILED)


class OutboxRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def outbox_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(OUTBOX_ENV)
    return Path(env) if env else DEFAULT_OUTBOX


def ensure_outbox(root: str | Path | None = None) -> Path:
    r = outbox_root(root)
    for state in STATE_DIRS:
        (r / state).mkdir(parents=True, exist_ok=True)
    (r / "receipts").mkdir(parents=True, exist_ok=True)
    return r


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def emit(document: dict, *, kind: str,
         root: str | Path | None = None) -> dict:
    """Append one document to the outbox. Never contacts a
    database; never fails because one is unavailable."""
    if kind not in ("envelope", "event"):
        raise OutboxRefusal(f"unknown outbox kind {kind!r}")
    r = ensure_outbox(root)
    body = {"outbox_kind": kind, "document": document}
    payload = json.dumps(body, sort_keys=True).encode()
    digest = _sha(payload)
    name = f"{kind}-{digest}.json"
    for state in (LOADED, PENDING, FAILED):
        if (r / state / name).is_file():
            # already emitted: the outbox is append-only and
            # content-addressed, so this is a no-op, not a
            # duplicate
            return {"outbox_entry": name, "state": state,
                    "digest": digest, "written": False}
    target = r / PENDING / name
    fd = os.open(str(target),
                 os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"outbox_entry": name, "state": PENDING,
            "digest": digest, "written": True}


def pending_entries(root: str | Path | None = None) -> list[Path]:
    r = ensure_outbox(root)
    return sorted((r / PENDING).glob("*.json"))


def read_entry(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def mark(path: Path, state: str, *, reason: str = "",
         root: str | Path | None = None) -> Path:
    """Move an entry between states. The BYTES are preserved; a
    loaded entry is recorded, never deleted."""
    if state not in STATE_DIRS:
        raise OutboxRefusal(f"unknown outbox state {state!r}")
    r = ensure_outbox(root)
    path = Path(path)
    target = r / state / path.name
    if target.exists():
        path.unlink(missing_ok=True)
        return target
    os.replace(path, target)
    if reason:
        (r / state / (path.stem + ".reason")).write_text(reason)
    return target


def counts(root: str | Path | None = None) -> dict:
    r = ensure_outbox(root)
    return {s: len(list((r / s).glob("*.json")))
            for s in STATE_DIRS}


def heartbeat(root: str | Path | None = None,
              *, now: float | None = None) -> dict:
    """What an operator needs to see without opening a database:
    last ingestion, pending, failed and lag."""
    r = ensure_outbox(root)
    now = now if now is not None else time.time()
    c = counts(root)
    loaded = sorted((r / LOADED).glob("*.json"),
                    key=lambda p: p.stat().st_mtime)
    last_loaded_at = (loaded[-1].stat().st_mtime
                      if loaded else None)
    pend = sorted((r / PENDING).glob("*.json"),
                  key=lambda p: p.stat().st_mtime)
    oldest_pending_at = (pend[0].stat().st_mtime
                         if pend else None)
    # No nulls: the same rule the envelopes obey. An unknown is
    # written UNAVAILABLE so a reader never mistakes it for zero.
    doc = {
        "schema": "crispdm.olap_outbox_heartbeat.v1",
        "outbox_root": str(r.name),
        "published_at_epoch": round(now, 3),
        "pending": c[PENDING],
        "loaded": c[LOADED],
        "failed": c[FAILED],
        "last_ingestion_epoch": (round(last_loaded_at, 3)
                                 if last_loaded_at
                                 else "UNAVAILABLE"),
        "oldest_pending_epoch": (round(oldest_pending_at, 3)
                                 if oldest_pending_at
                                 else "UNAVAILABLE"),
        "lag_seconds": (round(now - oldest_pending_at, 3)
                        if oldest_pending_at else 0.0),
        "healthy": c[FAILED] == 0,
    }
    (r / "HEARTBEAT.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc
