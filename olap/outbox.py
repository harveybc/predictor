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

# C37 (order 2026-09-11): a dead-letter is not a health state.
#
# `healthy` used to be `failed == 0`. Because the outbox never deletes
# anything, ONE permanent refusal from 2026-09-10 left the alarm red
# for the rest of the system's life, and — worse — it was the same red
# a dead loader would show. An operator who cannot tell "a known,
# already-understood refusal is on file" from "the loader stopped" has
# no alarm at all.
#
# So four things are separated and never folded back together:
#
#   * BACKLOG      — pending work. Retryable by construction; only
#                    OVERDUE backlog is a problem;
#   * DEAD-LETTER, UNADJUDICATED — a refusal nobody has ruled on. It
#                    needs attention, it is not an outage;
#   * DEAD-LETTER, ADJUDICATED — a refusal that was examined and
#                    dispositioned. The envelope and its reason stay on
#                    disk forever; what changes is that a human has
#                    answered it;
#   * PROCESS      — is the loader alive and is its heartbeat fresh.
#
# `healthy` answers only the last two questions that are about the
# SERVICE: alive, fresh, and not sitting on overdue backlog.
UNADJUDICATED = "DEAD_LETTER_UNADJUDICATED"
ADJUDICATED = "DEAD_LETTER_ADJUDICATED"
SUPERSEDED = "DEAD_LETTER_SUPERSEDED"

ADJUDICATIONS = (ADJUDICATED, SUPERSEDED)

#: how stale a heartbeat may be before the process is not fresh. The
#: loader beats every 30 s, so two missed beats plus slack.
DEFAULT_MAX_HEARTBEAT_AGE_S = 120.0
#: how long a pending entry may wait before the backlog is overdue.
DEFAULT_MAX_BACKLOG_LAG_S = 900.0


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
    return _entries(r / PENDING)


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


#: sidecars that live beside an entry and are not entries themselves.
SIDECAR_SUFFIX = ".adjudication.json"


def _entries(d: Path) -> list[Path]:
    """The entries in a state directory, excluding sidecars. An
    adjudication is a note ABOUT an envelope, never another envelope,
    and counting it as one would inflate the dead-letter count the
    moment someone ruled on it."""
    return sorted(p for p in d.glob("*.json")
                  if not p.name.endswith(SIDECAR_SUFFIX))


def counts(root: str | Path | None = None) -> dict:
    r = ensure_outbox(root)
    return {s: len(_entries(r / s)) for s in STATE_DIRS}


def adjudicate(entry: str | Path, *, disposition: str, reason: str,
               superseded_by: str | None = None,
               root: str | Path | None = None) -> dict:
    """Rule on a dead-letter WITHOUT deleting it.

    The envelope and the refusal reason stay exactly where they are.
    What is added is a sidecar recording that a human examined this
    refusal, what they decided and — when the work was later carried by
    another envelope — which one superseded it. Evidence is never
    destroyed to turn a flag green.
    """
    if disposition not in ADJUDICATIONS:
        raise OutboxRefusal(
            f"unknown dead-letter disposition {disposition!r}; "
            f"expected one of {list(ADJUDICATIONS)}")
    if not reason.strip():
        raise OutboxRefusal(
            "a dead-letter adjudication without a reason is a deletion "
            "with extra steps")
    if disposition == SUPERSEDED and not superseded_by:
        raise OutboxRefusal(
            "SUPERSEDED must name the entry that carried the work")
    r = ensure_outbox(root)
    name = Path(entry).name
    target = r / FAILED / name
    if not target.is_file():
        raise OutboxRefusal(
            f"{name} is not a dead-letter in this outbox")
    doc = {
        "schema": "crispdm.olap_outbox_adjudication.v1",
        "entry": name,
        "entry_sha256": _sha(target.read_bytes()),
        "disposition": disposition,
        "reason": reason,
        "superseded_by": superseded_by or "UNAVAILABLE",
        "adjudicated_at_epoch": round(time.time(), 3),
    }
    out = r / FAILED / (Path(name).stem + ".adjudication.json")
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def requeue(entry: str | Path,
            root: str | Path | None = None) -> dict:
    """Give an ADJUDICATED dead-letter another chance, without erasing
    the fact that it failed.

    The failed entry, its reason and its adjudication all stay on disk.
    A COPY of the bytes goes back to pending, so the loader can carry
    it under whatever contract now applies. The history then reads
    truthfully: this envelope was refused on one date, ruled on, and
    loaded on another.

    Requeueing something nobody has ruled on is forbidden — that is how
    a permanent refusal becomes an infinite retry.
    """
    r = ensure_outbox(root)
    name = Path(entry).name
    src = r / FAILED / name
    if not src.is_file():
        raise OutboxRefusal(f"{name} is not a dead-letter in this outbox")
    adj_p = r / FAILED / (Path(name).stem + ".adjudication.json")
    if not adj_p.is_file():
        raise OutboxRefusal(
            f"{name} has not been adjudicated; a refusal nobody has "
            "ruled on must not be retried")
    adj = json.loads(adj_p.read_text())
    if adj.get("disposition") != ADJUDICATED:
        raise OutboxRefusal(
            f"{name} is {adj.get('disposition')}, which is not a "
            "disposition that permits another attempt")
    target = r / PENDING / name
    if target.is_file():
        return {"outbox_entry": name, "state": PENDING,
                "requeued": False, "why": "already pending"}
    if (r / LOADED / name).is_file():
        return {"outbox_entry": name, "state": LOADED,
                "requeued": False, "why": "already loaded"}
    payload = src.read_bytes()
    fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    adj["requeued_at_epoch"] = round(time.time(), 3)
    adj_p.write_text(json.dumps(adj, indent=1, sort_keys=True) + "\n")
    return {"outbox_entry": name, "state": PENDING, "requeued": True,
            "dead_letter_preserved": True}


def dead_letters(root: str | Path | None = None) -> list[dict]:
    """Every refusal on file, with whether anyone has ruled on it."""
    r = ensure_outbox(root)
    out = []
    for entry in _entries(r / FAILED):
        adj_p = r / FAILED / (entry.stem + ".adjudication.json")
        reason_p = r / FAILED / (entry.stem + ".reason")
        adj = json.loads(adj_p.read_text()) if adj_p.is_file() else None
        out.append({
            "entry": entry.name,
            "state": adj["disposition"] if adj else UNADJUDICATED,
            "reason": (reason_p.read_text().strip()
                       if reason_p.is_file() else "UNAVAILABLE"),
            "adjudication": adj or "UNAVAILABLE",
        })
    return out


def health(root: str | Path | None = None, *, now: float | None = None,
           published_at: float | None = None,
           max_heartbeat_age_s: float = DEFAULT_MAX_HEARTBEAT_AGE_S,
           max_backlog_lag_s: float = DEFAULT_MAX_BACKLOG_LAG_S) -> dict:
    """Four separated questions, and one `healthy` that answers only
    the ones about the SERVICE.

    A reader calls this to find out whether the loader is alive; the
    dead-letter counts travel beside it so an unresolved refusal is
    visible without being mistaken for an outage.
    """
    r = ensure_outbox(root)
    now = now if now is not None else time.time()
    c = counts(root)

    # A READER takes the beat from the file on disk. The loader itself
    # passes the beat it is writing: asking it to read the previous
    # file would make its very first heartbeat report an unhealthy
    # process, which is the opposite of what just happened.
    published = published_at
    if published is None:
        hb_p = r / "HEARTBEAT.json"
        if hb_p.is_file():
            try:
                published = json.loads(hb_p.read_text()).get(
                    "published_at_epoch")
            except ValueError:
                published = None
    age = (round(now - published, 3)
           if isinstance(published, (int, float)) else None)
    process_fresh = age is not None and age <= max_heartbeat_age_s

    pend = sorted(_entries(r / PENDING),
                  key=lambda p: p.stat().st_mtime)
    oldest = pend[0].stat().st_mtime if pend else None
    lag = round(now - oldest, 3) if oldest else 0.0
    backlog_overdue = lag > max_backlog_lag_s

    dl = dead_letters(root)
    unadjudicated = [d for d in dl if d["state"] == UNADJUDICATED]

    return {
        "process_fresh": process_fresh,
        "heartbeat_age_seconds": age if age is not None else "UNAVAILABLE",
        "max_heartbeat_age_seconds": max_heartbeat_age_s,
        "backlog_pending": c[PENDING],
        "backlog_lag_seconds": lag,
        "backlog_overdue": backlog_overdue,
        "max_backlog_lag_seconds": max_backlog_lag_s,
        "dead_letters_total": len(dl),
        "dead_letters_unadjudicated": len(unadjudicated),
        "dead_letters_adjudicated": len(dl) - len(unadjudicated),
        # The service question, and ONLY the service question.
        "healthy": bool(process_fresh and not backlog_overdue),
        # The separate, human-facing question.
        "attention_required": bool(unadjudicated),
    }


def heartbeat(root: str | Path | None = None,
              *, now: float | None = None) -> dict:
    """What an operator needs to see without opening a database:
    last ingestion, pending, failed and lag."""
    r = ensure_outbox(root)
    now = now if now is not None else time.time()
    c = counts(root)
    loaded = sorted(_entries(r / LOADED),
                    key=lambda p: p.stat().st_mtime)
    last_loaded_at = (loaded[-1].stat().st_mtime
                      if loaded else None)
    pend = sorted(_entries(r / PENDING),
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
    }
    # The heartbeat is written BY the loader, so at write time the
    # process is alive by construction; freshness is what a READER
    # evaluates, and `health()` is where a reader evaluates it.
    doc.update({k: v for k, v in
                health(root, now=now,
                       published_at=doc["published_at_epoch"]).items()
                if k not in doc})
    (r / "HEARTBEAT.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc
