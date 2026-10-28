"""Durable, exclusive multihorizon claims and immutable terminal receipts.

Only the coordinator owns this SQLite database. Workers receive a sealed job
and return its pre-issued lease. Restarting does not create another training.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid

TERMINAL = ("COMPLETED", "FAILED", "NOT_APPLICABLE")


def digest(value):
    """Hash structured identity, preserving scientifically significant order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(value):
    value = {k: v for k, v in value.items() if k != "sha256"}
    return dict(value, sha256=digest(value))


def verify(value):
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("record digest mismatch")
    return value


def atomic_json(path, value):
    """Atomic replacement with unique staging names (multiple outbox writers safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)


class Queue:
    """WAL queue, ordered stage barrier, one active claim per execution slot."""

    def __init__(self, path, design):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.design = verify(design)
        self.identity = design["sha256"]
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS meta(identity TEXT PRIMARY KEY)")
            old = conn.execute("SELECT identity FROM meta").fetchone()
            if old and old[0] != self.identity:
                raise ValueError("queue design identity differs")
            conn.execute("INSERT OR IGNORE INTO meta VALUES (?)", (self.identity,))
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, stage TEXT, model TEXT, ordinal INTEGER,
                spec TEXT, state TEXT DEFAULT 'QUEUED', slot TEXT, lease TEXT,
                claimed REAL, terminal TEXT, published INTEGER DEFAULT 0)""")
            for rank, arm in enumerate(design["arms"]):
                for week in design["weeks"]:
                    for model in design.get("models", ["NEURAL"]):
                        job_id = f"{model}_{arm['arm_id']}_w{week['ordinal']:03d}"
                        spec = dict(id=job_id, arm=arm, week=week, model=model,
                                    seed=design["seed"], design_sha256=self.identity,
                                    input_sha256=design["input_sha256"])
                        conn.execute("INSERT OR IGNORE INTO jobs(id,stage,model,ordinal,spec) VALUES (?,?,?,?,?)",
                                     (job_id, arm["stage"], model, rank*100+week["ordinal"], json.dumps(spec)))

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def claim(self, slot, model, max_samples=None, measured_samples=None):
        """Claim atomically. Statistical controls have their own stage barrier."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM jobs WHERE slot=? AND state='RUNNING'", (slot,)).fetchone():
                return None
            pending_d = conn.execute("SELECT 1 FROM jobs WHERE model=? AND stage='DOWNSAMPLING' AND state IN ('QUEUED','RUNNING')", (model,)).fetchone()
            stage = "DOWNSAMPLING" if pending_d else "HOURLY"
            rows = conn.execute("SELECT * FROM jobs WHERE state='QUEUED' AND stage=? AND model=? ORDER BY ordinal", (stage, model))
            for row in rows:
                spec = json.loads(row["spec"])
                if max_samples is not None and spec["arm"]["samples"] > max_samples:
                    continue
                if measured_samples is not None and spec["arm"]["samples"] not in measured_samples and spec["week"]["ordinal"] != 0:
                    continue
                lease = uuid.uuid4().hex
                conn.execute("UPDATE jobs SET state='RUNNING',slot=?,lease=?,claimed=? WHERE id=?", (slot, lease, time.time(), row["id"]))
                return dict(spec, lease=lease, slot=slot)
            return None

    def running(self):
        with self.connect() as conn:
            return [dict(json.loads(r["spec"]), lease=r["lease"], slot=r["slot"], claimed=r["claimed"])
                    for r in conn.execute("SELECT * FROM jobs WHERE state='RUNNING'")]

    def complete(self, job, terminal):
        verify(terminal)
        if (terminal.get("job_id") != job["id"] or terminal.get("lease") != job["lease"]
                or terminal.get("design_sha256") != self.identity or terminal.get("status") not in TERMINAL):
            raise ValueError("terminal lease, job, design or status mismatch")
        body = json.dumps(terminal, sort_keys=True, allow_nan=False)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT state,lease,terminal FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if not row or row["lease"] != job["lease"]:
                raise ValueError("lease is not retained by supervisor")
            if row["terminal"] is not None and row["terminal"] != body:
                raise ValueError("terminal conflict: completed fitting may not be replaced")
            conn.execute("UPDATE jobs SET state=?,terminal=? WHERE id=?", (terminal["status"], body, job["id"]))

    def status(self):
        with self.connect() as conn:
            counts = {r["state"]: r["n"] for r in conn.execute("SELECT state,count(*) n FROM jobs GROUP BY state")}
            stages = [dict(r) for r in conn.execute("SELECT stage,model,state,count(*) n FROM jobs GROUP BY stage,model,state")]
            cells = [json.loads(r[0]) for r in conn.execute("SELECT terminal FROM jobs WHERE state='COMPLETED'")]
            durations = [c["elapsed_seconds"] for c in cells if c.get("elapsed_seconds")]
            active_slots = len(self.running())
            # Projection explicitly excludes larger unmeasured model classes.
            eta = None if not durations or not active_slots else sum(durations)/len(durations)*counts.get("QUEUED", 0)/active_slots
            return dict(schema="predictor.i6m.status.v1", design_sha256=self.identity,
                        timestamp=time.time(), counts=counts, stages=stages,
                        total=sum(counts.values()), running=self.running(),
                        eta_seconds_same_cost_projection=eta,
                        eta_scope="same-cost projection, not measured hourly-large ETA",
                        test_read=False)
