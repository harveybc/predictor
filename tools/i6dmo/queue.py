"""Exclusive durable work queue for one I6-DMO campaign: units, leased attempts, terminals, admissions.

One SQLite file (WAL, synchronous=FULL) under ``<root>/state``. Every state change is one
IMMEDIATE transaction, so power loss leaves either the old or the new state.

Policy (ACCEPTANCE DMO07/11/13/17, PLAN section 6):
- a unit's identity is fixed once set; a different identity under the same unit id is refused;
- a verified terminal is adopted, never refit; a late receipt after a terminal is discarded;
- an attempt whose lease expired, or whose supervisor/host disappeared, becomes INTERRUPTED and stays
  in the history; its successor is a RECOVERY attempt (no resume support is claimed);
- FAILED, FAILED_OOM and FAILED_TIMEOUT are terminal: no retry, no smaller cap, no device change;
- at most three physical attempts per unit; an admission refusal starts nothing and is not an attempt;
- one heavy lease per physical host: while any heavy attempt holds an unexpired lease on a host,
  no other heavy attempt starts on that host, whichever GPU it would use (the 5070 Ti beside the
  5090 is not a parallel slot);
- a refused memory cap is remembered per unit and later caps are never below it.
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

from tools.i6n_swarm.common import canonical

SCHEMA = "i6dmo.queue.v1"
STATES = ("WAIT_ARTIFACT", "WAIT_OWNER_RULING", "WAIT_DESIGN_RULING", "WAIT_DEPENDENCY", "QUEUED", "WAIT_RESOURCE", "RUNNING", "INTERRUPTED",
          "COMPLETE", "FAILED", "BLOCKED")
TERMINAL_FAILURES = ("FAILED", "FAILED_OOM", "FAILED_TIMEOUT")
MAX_PHYSICAL_ATTEMPTS = 3

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS units(
  unit_id TEXT PRIMARY KEY, spec TEXT NOT NULL, identity_sha256 TEXT, kind TEXT NOT NULL, device TEXT NOT NULL,
  state TEXT NOT NULL, cause TEXT, disposition TEXT, min_cap_bytes INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS attempts(
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, unit_id TEXT NOT NULL, idx INTEGER NOT NULL, kind TEXT NOT NULL,
  owner TEXT NOT NULL, role TEXT NOT NULL, host_key TEXT NOT NULL, boot_id TEXT NOT NULL,
  device_class TEXT NOT NULL, heavy INTEGER NOT NULL, token INTEGER NOT NULL UNIQUE, started_at REAL NOT NULL,
  renewed_at REAL NOT NULL, lease_expires_at REAL NOT NULL, ended_at REAL, outcome TEXT NOT NULL, cause TEXT,
  progress TEXT, attempt_dir TEXT, launch TEXT, cap_bytes INTEGER, peak_bytes INTEGER, costs TEXT);
CREATE INDEX IF NOT EXISTS attempts_unit ON attempts(unit_id);
CREATE INDEX IF NOT EXISTS attempts_running ON attempts(outcome, host_key);
CREATE TABLE IF NOT EXISTS terminals(
  unit_id TEXT PRIMARY KEY, identity_sha256 TEXT, terminal TEXT NOT NULL, terminal_sha256 TEXT NOT NULL,
  source TEXT NOT NULL, source_detail TEXT, attempt_id INTEGER, accepted_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS admissions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, unit_id TEXT NOT NULL, role TEXT NOT NULL, at REAL NOT NULL,
  admit INTEGER NOT NULL, cause TEXT, cap_bytes INTEGER, evidence TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, unit_id TEXT, event TEXT NOT NULL, detail TEXT);
"""


class QueueError(RuntimeError):
    pass


class IdentityConflict(QueueError):
    pass


def _loads(text):
    return None if text is None else json.loads(text)


class Queue:
    def __init__(self, path, clock=time.time, readonly=False):
        self.path = Path(path)
        self.clock = clock
        self._lock = threading.RLock()
        if readonly:
            self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, isolation_level=None, timeout=30,
                                       check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), isolation_level=None, timeout=30, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript(DDL)
            self._db.execute("INSERT OR IGNORE INTO meta VALUES('schema', ?)", (SCHEMA,))
            self._db.execute("INSERT OR IGNORE INTO meta VALUES('fence', '0')")
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=30000")

    @classmethod
    def open_readonly(cls, path, clock=time.time):
        return cls(path, clock=clock, readonly=True)

    def close(self):
        self._db.close()

    def _tx(self):
        queue = self

        class _Tx:
            def __enter__(self_inner):
                queue._lock.acquire()
                queue._db.execute("BEGIN IMMEDIATE")
                return queue._db

            def __exit__(self_inner, kind, exc, tb):
                try:
                    queue._db.execute("ROLLBACK" if kind else "COMMIT")
                finally:
                    queue._lock.release()
                return False
        return _Tx()

    def _event(self, db, event, unit_id=None, detail=None):
        db.execute("INSERT INTO events(at, unit_id, event, detail) VALUES(?,?,?,?)",
                   (self.clock(), unit_id, event, None if detail is None else canonical(detail)))

    def event(self, event, unit_id=None, detail=None):
        with self._tx() as db:
            self._event(db, event, unit_id, detail)

    # ------------------------------------------------------------------ units
    @staticmethod
    def _unit(row):
        if row is None:
            return None
        out = dict(row)
        out["spec"] = json.loads(out["spec"])
        return out

    def ensure_unit(self, unit_id, spec, state="WAIT_DEPENDENCY"):
        if state not in STATES:
            raise QueueError(f"unknown state {state}")
        identity = spec.get("identity_sha256")
        with self._tx() as db:
            now = self.clock()
            row = db.execute("SELECT * FROM units WHERE unit_id=?", (unit_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO units(unit_id, spec, identity_sha256, kind, device, state, created_at, "
                           "updated_at) VALUES(?,?,?,?,?,?,?,?)",
                           (unit_id, canonical(spec), identity, spec["kind"], spec.get("device", "cpu"), state,
                            now, now))
                self._event(db, "UNIT_REGISTERED", unit_id, dict(identity_sha256=identity, state=state))
            elif row["identity_sha256"] is None and identity is not None:
                db.execute("UPDATE units SET spec=?, identity_sha256=?, state=?, cause=NULL, updated_at=? "
                           "WHERE unit_id=?", (canonical(spec), identity, state, now, unit_id))
                self._event(db, "UNIT_IDENTITY_SEALED", unit_id, dict(identity_sha256=identity))
            elif row["identity_sha256"] != identity:
                raise IdentityConflict(f"{unit_id}: identity is fixed once set")
            return self._unit(db.execute("SELECT * FROM units WHERE unit_id=?", (unit_id,)).fetchone())

    def unit(self, unit_id):
        with self._lock:
            return self._unit(self._db.execute("SELECT * FROM units WHERE unit_id=?", (unit_id,)).fetchone())

    def units(self):
        with self._lock:
            return [self._unit(r) for r in self._db.execute("SELECT * FROM units ORDER BY created_at, unit_id")]

    def set_state(self, unit_id, state, cause=None, disposition=None):
        if state not in STATES:
            raise QueueError(f"unknown state {state}")
        with self._tx() as db:
            db.execute("UPDATE units SET state=?, cause=?, disposition=COALESCE(?, disposition), updated_at=? "
                       "WHERE unit_id=?", (state, cause, disposition, self.clock(), unit_id))
            self._event(db, "STATE", unit_id, dict(state=state, cause=cause))

    # --------------------------------------------------------------- attempts
    @staticmethod
    def _attempt(row):
        out = dict(row)
        for key in ("progress", "launch", "costs"):
            out[key] = _loads(out[key])
        out["heavy"] = bool(out["heavy"])
        return out

    def attempts(self, unit_id=None):
        with self._lock:
            if unit_id is None:
                rows = self._db.execute("SELECT * FROM attempts ORDER BY attempt_id").fetchall()
            else:
                rows = self._db.execute("SELECT * FROM attempts WHERE unit_id=? ORDER BY attempt_id",
                                        (unit_id,)).fetchall()
            return [self._attempt(r) for r in rows]

    def attempt_by_token(self, token):
        with self._lock:
            row = self._db.execute("SELECT * FROM attempts WHERE token=?", (token,)).fetchone()
            return None if row is None else self._attempt(row)

    def running(self, role=None):
        with self._lock:
            if role is None:
                rows = self._db.execute("SELECT * FROM attempts WHERE outcome='RUNNING'").fetchall()
            else:
                rows = self._db.execute("SELECT * FROM attempts WHERE outcome='RUNNING' AND role=?",
                                        (role,)).fetchall()
            return [self._attempt(r) for r in rows]

    def _next_token(self, db):
        token = int(db.execute("SELECT value FROM meta WHERE key='fence'").fetchone()[0]) + 1
        db.execute("UPDATE meta SET value=? WHERE key='fence'", (str(token),))
        return token

    def claim(self, unit_id, *, owner, role, host_key, boot_id, device_class, heavy, ttl, cap_bytes=None):
        if ttl <= 0:
            raise QueueError("ttl must be positive")
        with self._tx() as db:
            now = self.clock()
            unit = db.execute("SELECT * FROM units WHERE unit_id=?", (unit_id,)).fetchone()
            if unit is None:
                raise QueueError(f"unknown unit {unit_id}")
            terminal = db.execute("SELECT * FROM terminals WHERE unit_id=?", (unit_id,)).fetchone()
            if terminal is not None:
                return dict(result="ADOPT", terminal_sha256=terminal["terminal_sha256"])
            if unit["state"] in ("FAILED", "BLOCKED"):
                return dict(result="REFUSED", cause=unit["cause"] or unit["state"])
            if unit["identity_sha256"] is None:
                return dict(result="REFUSED", cause="IDENTITY_NOT_SEALED")
            for attempt in db.execute("SELECT * FROM attempts WHERE unit_id=? AND outcome='RUNNING'",
                                      (unit_id,)).fetchall():
                if attempt["lease_expires_at"] > now:
                    return dict(result="HELD", holder=dict(role=attempt["role"], token=attempt["token"],
                                                           lease_expires_at=attempt["lease_expires_at"]))
                self._interrupt(db, attempt, "LEASE_EXPIRED", now)
            if heavy:
                busy = db.execute("SELECT unit_id FROM attempts WHERE outcome='RUNNING' AND heavy=1 AND host_key=? "
                                  "AND lease_expires_at>? AND unit_id!=? LIMIT 1", (host_key, now, unit_id)).fetchone()
                if busy is not None:
                    self._event(db, "HOST_HEAVY_LEASE_HELD", unit_id, dict(holder=busy["unit_id"], role=role))
                    return dict(result="WAIT_RESOURCE", cause="HOST_HEAVY_LEASE_HELD")
            physical = db.execute("SELECT * FROM attempts WHERE unit_id=? ORDER BY attempt_id", (unit_id,)).fetchall()
            kind = "INITIAL"
            if physical:
                last = physical[-1]
                if last["outcome"] != "INTERRUPTED":
                    cause = f"LAST_ATTEMPT_{last['outcome']}"
                    db.execute("UPDATE units SET state='FAILED', cause=?, updated_at=? WHERE unit_id=?",
                               (cause, now, unit_id))
                    return dict(result="REFUSED", cause=cause)
                if len(physical) >= MAX_PHYSICAL_ATTEMPTS:
                    db.execute("UPDATE units SET state='FAILED', cause='ATTEMPTS_EXHAUSTED', updated_at=? "
                               "WHERE unit_id=?", (now, unit_id))
                    self._event(db, "ATTEMPTS_EXHAUSTED", unit_id)
                    return dict(result="REFUSED", cause="ATTEMPTS_EXHAUSTED")
                kind = "RECOVERY"
            token = self._next_token(db)
            index = len(physical) + 1
            cur = db.execute(
                "INSERT INTO attempts(unit_id, idx, kind, owner, role, host_key, boot_id, device_class, heavy, token, "
                "started_at, renewed_at, lease_expires_at, outcome, cap_bytes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "'RUNNING',?)", (unit_id, index, kind, owner, role, host_key, boot_id, device_class, int(heavy),
                                 token, now, now, now + ttl, cap_bytes))
            db.execute("UPDATE units SET state='RUNNING', cause=NULL, updated_at=? WHERE unit_id=?", (now, unit_id))
            self._event(db, "CLAIMED", unit_id, dict(token=token, kind=kind, index=index, role=role,
                                                     device_class=device_class))
            return dict(result="CLAIMED", token=token,
                        attempt=dict(attempt_id=cur.lastrowid, index=index, kind=kind))

    def _interrupt(self, db, attempt, cause, now):
        db.execute("UPDATE attempts SET outcome='INTERRUPTED', cause=?, ended_at=? WHERE attempt_id=?",
                   (cause, now, attempt["attempt_id"]))
        has_terminal = db.execute("SELECT 1 FROM terminals WHERE unit_id=?", (attempt["unit_id"],)).fetchone()
        if not has_terminal:
            db.execute("UPDATE units SET state='INTERRUPTED', cause=?, updated_at=? WHERE unit_id=?",
                       (cause, now, attempt["unit_id"]))
        self._event(db, "INTERRUPTED", attempt["unit_id"], dict(token=attempt["token"], cause=cause))

    def interrupt(self, token, cause):
        with self._tx() as db:
            row = db.execute("SELECT * FROM attempts WHERE token=? AND outcome='RUNNING'", (token,)).fetchone()
            if row is None:
                return False
            self._interrupt(db, row, cause, self.clock())
            return True

    def heartbeat(self, unit_id, token, ttl, progress=None):
        with self._tx() as db:
            now = self.clock()
            row = db.execute("SELECT * FROM attempts WHERE unit_id=? AND token=?", (unit_id, token)).fetchone()
            if row is None or row["outcome"] != "RUNNING":
                return False
            newer = db.execute("SELECT 1 FROM attempts WHERE unit_id=? AND token>?", (unit_id, token)).fetchone()
            if newer:
                return False
            db.execute("UPDATE attempts SET renewed_at=?, lease_expires_at=?, progress=COALESCE(?, progress) "
                       "WHERE token=?", (now, now + ttl, None if progress is None else canonical(progress), token))
            return True

    def set_launch(self, token, attempt_dir, launch):
        with self._tx() as db:
            db.execute("UPDATE attempts SET attempt_dir=?, launch=? WHERE token=?",
                       (str(attempt_dir), canonical(launch), token))

    def complete(self, unit_id, token, terminal, terminal_sha256, *, identity_sha256=None, peak_bytes=None,
                 costs=None):
        with self._tx() as db:
            now = self.clock()
            row = db.execute("SELECT * FROM attempts WHERE unit_id=? AND token=?", (unit_id, token)).fetchone()
            if row is None:
                raise QueueError("unknown attempt token")
            if db.execute("SELECT 1 FROM terminals WHERE unit_id=?", (unit_id,)).fetchone():
                self._event(db, "LATE_RECEIPT_DISCARDED", unit_id, dict(token=token))
                return dict(result="DISCARDED")
            if row["outcome"] != "RUNNING":
                self._event(db, "LATE_RECEIPT_ACCEPTED", unit_id, dict(token=token, outcome=row["outcome"]))
            db.execute("UPDATE attempts SET outcome='COMPLETE', ended_at=?, peak_bytes=?, costs=? WHERE token=?",
                       (now, peak_bytes, None if costs is None else canonical(costs), token))
            db.execute("INSERT INTO terminals VALUES(?,?,?,?,?,?,?,?)",
                       (unit_id, identity_sha256, canonical(terminal), terminal_sha256, "LOCAL", None,
                        row["attempt_id"], now))
            db.execute("UPDATE units SET state='COMPLETE', cause=NULL, disposition=?, updated_at=? WHERE unit_id=?",
                       (terminal.get("status"), now, unit_id))
            self._event(db, "COMPLETE", unit_id, dict(token=token, terminal_sha256=terminal_sha256))
            return dict(result="COMPLETE")

    def fail(self, unit_id, token, outcome, cause, *, peak_bytes=None, costs=None):
        if outcome not in TERMINAL_FAILURES + ("INTERRUPTED",):
            raise QueueError(f"bad outcome {outcome}")
        with self._tx() as db:
            now = self.clock()
            row = db.execute("SELECT * FROM attempts WHERE unit_id=? AND token=?", (unit_id, token)).fetchone()
            if row is None or row["outcome"] != "RUNNING":
                return dict(result="IGNORED")
            db.execute("UPDATE attempts SET outcome=?, cause=?, ended_at=?, peak_bytes=?, costs=? WHERE token=?",
                       (outcome, cause, now, peak_bytes, None if costs is None else canonical(costs), token))
            state = "INTERRUPTED" if outcome == "INTERRUPTED" else "FAILED"
            db.execute("UPDATE units SET state=?, cause=?, updated_at=? WHERE unit_id=?",
                       (state, f"{outcome}:{cause}", now, unit_id))
            self._event(db, outcome, unit_id, dict(token=token, cause=cause))
            return dict(result=state)

    def adopt(self, unit_id, terminal, terminal_sha256, *, identity_sha256, source_detail):
        with self._tx() as db:
            if db.execute("SELECT 1 FROM terminals WHERE unit_id=?", (unit_id,)).fetchone():
                return dict(result="ALREADY_TERMINAL")
            now = self.clock()
            db.execute("INSERT INTO terminals VALUES(?,?,?,?,?,?,?,?)",
                       (unit_id, identity_sha256, canonical(terminal), terminal_sha256, "ADOPTED",
                        canonical(source_detail), None, now))
            db.execute("UPDATE units SET state='COMPLETE', cause='ADOPTED', disposition=?, updated_at=? "
                       "WHERE unit_id=?", (terminal.get("status"), now, unit_id))
            self._event(db, "ADOPTED", unit_id, dict(terminal_sha256=terminal_sha256))
            return dict(result="ADOPTED")

    # -------------------------------------------------------------- terminals
    def terminal(self, unit_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM terminals WHERE unit_id=?", (unit_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["terminal"] = json.loads(out["terminal"])
        out["source_detail"] = _loads(out["source_detail"])
        return out

    def terminals(self):
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT unit_id FROM terminals ORDER BY accepted_at")]
        return [self.terminal(i) for i in ids]

    # ------------------------------------------------------------- admission
    def record_admission(self, unit_id, role, admit, cause, cap_bytes, evidence=None):
        with self._tx() as db:
            db.execute("INSERT INTO admissions(unit_id, role, at, admit, cause, cap_bytes, evidence) "
                       "VALUES(?,?,?,?,?,?,?)", (unit_id, role, self.clock(), int(admit), cause, cap_bytes,
                                                 None if evidence is None else canonical(evidence)))
            if not admit:
                if cap_bytes:
                    db.execute("UPDATE units SET min_cap_bytes=MAX(min_cap_bytes, ?) WHERE unit_id=?",
                               (int(cap_bytes), unit_id))
                db.execute("UPDATE units SET state=CASE WHEN state IN ('QUEUED','INTERRUPTED') THEN 'WAIT_RESOURCE' "
                           "ELSE state END, cause=?, updated_at=? WHERE unit_id=?", (f"{role}:{cause}", self.clock(),
                                                                                     unit_id))
                self._event(db, "ADMISSION_REFUSED", unit_id, dict(role=role, cause=cause, cap_bytes=cap_bytes))

    def min_cap(self, unit_id):
        with self._lock:
            row = self._db.execute("SELECT min_cap_bytes FROM units WHERE unit_id=?", (unit_id,)).fetchone()
        return 0 if row is None else int(row[0])

    def refused_roles(self, unit_id):
        with self._lock:
            return sorted({r[0] for r in self._db.execute(
                "SELECT role FROM admissions WHERE unit_id=? AND admit=0", (unit_id,))})

    def admissions(self, role=None, limit=50):
        with self._lock:
            if role is None:
                rows = self._db.execute("SELECT * FROM admissions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM admissions WHERE role=? ORDER BY id DESC LIMIT ?",
                                        (role, limit)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["evidence"] = _loads(item["evidence"])
            item["admit"] = bool(item["admit"])
            out.append(item)
        return out

    def measured_peak(self, device_class=None, kind=None, family=None, resource_class=None):
        """Largest measured full-tree peak of completed attempts (per device class / unit kind / family).

        ``resource_class`` narrows further for units that declare one (matched arms share a family but
        not a model: a 13-slot peak is never the 25-slot requirement)."""
        sql = ("SELECT MAX(a.peak_bytes) FROM attempts a JOIN units u ON u.unit_id=a.unit_id "
               "WHERE a.outcome='COMPLETE' AND a.peak_bytes IS NOT NULL")
        args = []
        if device_class:
            sql += " AND a.device_class=?"
            args.append(device_class)
        if kind:
            sql += " AND u.kind=?"
            args.append(kind)
        if family:
            sql += " AND json_extract(u.spec, '$.family')=?"
            args.append(family)
        if resource_class:
            sql += " AND json_extract(u.spec, '$.resource_class')=?"
            args.append(resource_class)
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def events(self, limit=500):
        with self._lock:
            rows = self._db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["detail"] = _loads(item["detail"])
            out.append(item)
        return out
