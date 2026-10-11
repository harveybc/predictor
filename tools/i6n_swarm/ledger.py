"""Durable campaign ledger: global fit identity, leases with fencing tokens, attempts, outbox.

One SQLite file (WAL) on the coordinator is the single authority. Every state
change is one IMMEDIATE transaction, so a crash leaves either the old or the
new state. Workers on other hosts reach it through ``lease_service``.

Policy (plan section 5, orders 2.B):
- a verified terminal receipt is adopted, never refit;
- a RUNNING attempt whose lease expired, or whose host booted again, is
  INTERRUPTED; its successor is RESUME when a complete checkpoint (optimizer,
  rng, epoch, stopping) exists, otherwise the single linked RECOVERY attempt;
- at most three physical attempts per fit_id; an admission refusal that
  started nothing (NOT_STARTED) is not a physical attempt;
- FAILED, FAILED_OOM and FAILED_TIMEOUT are terminal: no retry, no smaller cap;
- fencing tokens are globally monotonic; a stale holder cannot renew, and its
  late receipt is accepted only if the fit has no terminal yet (first verified
  terminal wins, later receipts are discarded, never published).
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

from tools.i6n_swarm import common

SCHEMA = "i6n.swarm.ledger.v1"
UNIT_STATES = ("QUEUED", "RUNNING", "COMPLETE", "WAIT_RESOURCE", "INTERRUPTED", "FAILED", "WAIT_DEPENDENCY")
REPORT_OUTCOMES = ("INTERRUPTED", "FAILED", "FAILED_OOM", "FAILED_TIMEOUT", "NOT_STARTED")
MAX_PHYSICAL_ATTEMPTS = 3
CHECKPOINT_KEYS = ("optimizer", "rng", "stopping")
SPEC_KEYS = frozenset(("stage", "dag_node", "horizon", "fold", "experiment", "genotype_canonical_sha256",
                       "effective_config_sha256", "search_candidate_sha256", "device_class", "fixture"))
RECEIPT_KEYS = frozenset(("fit_id", "fold", "experiment", "effective_config_sha256", "genotype_canonical_sha256",
                          "shared_receipt", "candidate_receipt"))

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS units(
  fit_id TEXT PRIMARY KEY, stage TEXT NOT NULL, dag_node TEXT NOT NULL, horizon INTEGER NOT NULL,
  fold TEXT NOT NULL, experiment TEXT NOT NULL, genotype TEXT NOT NULL, search_candidate TEXT NOT NULL,
  effective_config_sha256 TEXT NOT NULL, device_class TEXT NOT NULL, fixture INTEGER NOT NULL,
  spec TEXT NOT NULL, spec_sha256 TEXT NOT NULL, state TEXT NOT NULL, cause TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS units_node ON units(dag_node, state);
CREATE TABLE IF NOT EXISTS attempts(
  attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, fit_id TEXT NOT NULL, idx INTEGER NOT NULL, kind TEXT NOT NULL,
  parent_attempt INTEGER, parent_status_sha256 TEXT, status_sha256 TEXT, owner TEXT NOT NULL, host TEXT NOT NULL,
  boot_id TEXT NOT NULL, device_class TEXT NOT NULL, token INTEGER NOT NULL UNIQUE, started_at REAL NOT NULL,
  renewed_at REAL NOT NULL, lease_expires_at REAL NOT NULL, ended_at REAL, outcome TEXT NOT NULL, cause TEXT,
  checkpoint TEXT, progress TEXT, physical INTEGER NOT NULL DEFAULT 1, peak_bytes INTEGER);
CREATE INDEX IF NOT EXISTS attempts_fit ON attempts(fit_id);
CREATE INDEX IF NOT EXISTS attempts_running ON attempts(outcome, host);
CREATE TABLE IF NOT EXISTS terminals(
  fit_id TEXT PRIMARY KEY, receipt TEXT NOT NULL, receipt_sha256 TEXT NOT NULL, attempt_id INTEGER,
  source TEXT NOT NULL, source_detail TEXT, accepted_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(
  id INTEGER PRIMARY KEY AUTOINCREMENT, fit_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL UNIQUE, created_at REAL NOT NULL, state TEXT NOT NULL,
  delivered_at REAL, delivery TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, fit_id TEXT, node_id TEXT, event TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS dag(
  node_id TEXT PRIMARY KEY, stage TEXT NOT NULL, ord INTEGER NOT NULL, depends_on TEXT NOT NULL, spec TEXT NOT NULL,
  state TEXT NOT NULL, cause TEXT, summary TEXT, activated_at REAL, completed_at REAL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS search_state(
  node_id TEXT NOT NULL, generation INTEGER NOT NULL, state TEXT NOT NULL, state_sha256 TEXT NOT NULL,
  at REAL NOT NULL, PRIMARY KEY(node_id, generation));
CREATE TABLE IF NOT EXISTS dag_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, node_id TEXT NOT NULL, event TEXT NOT NULL, detail TEXT, at REAL NOT NULL,
  UNIQUE(node_id, event));
"""


class LedgerError(ValueError):
    pass


class ReceiptMismatch(LedgerError):
    """A receipt does not belong to this fit (another candidate, fold or config)."""


class BudgetExhausted(LedgerError):
    pass


def _loads(text):
    return None if text is None else json.loads(text)


class Ledger:
    def __init__(self, path, clock=time.time, readonly=False):
        self.path = Path(path)
        self.clock = clock
        self.readonly = readonly
        self._lock = threading.RLock()
        if readonly:
            self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, isolation_level=None,
                                       check_same_thread=False, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False, timeout=30)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript(DDL)
            self._db.execute("INSERT OR IGNORE INTO meta VALUES('schema', ?)", (SCHEMA,))
            self._db.execute("INSERT OR IGNORE INTO meta VALUES('fence', '0')")
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=30000")

    @classmethod
    def open_readonly(cls, path):
        return cls(path, readonly=True)

    def close(self):
        self._db.close()

    # ------------------------------------------------------------ plumbing
    def _tx(self):
        ledger = self

        class _Tx:
            def __enter__(self_inner):
                ledger._lock.acquire()
                ledger._db.execute("BEGIN IMMEDIATE")
                return ledger._db

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    ledger._db.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    ledger._lock.release()
                return False
        return _Tx()

    def _q(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _event(self, db, event, fit_id=None, node_id=None, detail=None):
        db.execute("INSERT INTO events(at, fit_id, node_id, event, detail) VALUES(?,?,?,?,?)",
                   (self.clock(), fit_id, node_id, event, None if detail is None else common.canonical(detail)))

    def _next_token(self, db):
        token = int(db.execute("SELECT value FROM meta WHERE key='fence'").fetchone()[0]) + 1
        db.execute("UPDATE meta SET value=? WHERE key='fence'", (str(token),))
        return token

    @staticmethod
    def _unit_row(row):
        if row is None:
            return None
        out = dict(row)
        out["spec"] = json.loads(out["spec"])
        out["fixture"] = bool(out["fixture"])
        out["genotype_canonical_sha256"] = out["genotype"]
        return out

    @staticmethod
    def _attempt_row(row):
        out = dict(row)
        out["index"] = out.pop("idx")
        out["checkpoint"] = _loads(out["checkpoint"])
        out["progress"] = _loads(out["progress"])
        out["physical"] = bool(out["physical"])
        return out

    def _set_unit(self, db, fit_id, state, cause=None):
        db.execute("UPDATE units SET state=?, cause=?, updated_at=? WHERE fit_id=?", (state, cause, self.clock(), fit_id))

    # ------------------------------------------------------------ units
    def ensure_unit(self, fit_id, spec, budget=None):
        """Register a fit unit (idempotent). ``budget`` caps distinct new genotypes per dag node."""
        if not isinstance(fit_id, str) or len(fit_id) != 64:
            raise LedgerError("fit_id must be a sha256 hex digest")
        missing = SPEC_KEYS - set(spec)
        if missing:
            raise LedgerError(f"unit spec lacks {sorted(missing)}")
        text = common.canonical(spec)
        digest = common.sha256_json(spec)
        with self._tx() as db:
            row = db.execute("SELECT spec_sha256, state FROM units WHERE fit_id=?", (fit_id,)).fetchone()
            if row is not None:
                if row["spec_sha256"] != digest:
                    raise ValueError("fit unit spec is immutable; a changed spec is a different fit_id")
                return row["state"]
            if budget is not None:
                known = db.execute("SELECT 1 FROM units WHERE dag_node=? AND genotype=? LIMIT 1",
                                   (spec["dag_node"], spec["genotype_canonical_sha256"])).fetchone()
                if known is None:
                    used = db.execute(
                        "SELECT COUNT(DISTINCT u.genotype) FROM units u LEFT JOIN terminals t ON t.fit_id=u.fit_id "
                        "WHERE u.dag_node=? AND (t.source IS NULL OR t.source != 'PRIOR')",
                        (spec["dag_node"],)).fetchone()[0]
                    if used >= budget:
                        raise BudgetExhausted(f"BUDGET_EXHAUSTED: {used} new candidates already in {spec['dag_node']}")
            now = self.clock()
            db.execute("INSERT INTO units VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                fit_id, spec["stage"], spec["dag_node"], int(spec["horizon"]), spec["fold"], spec["experiment"],
                spec["genotype_canonical_sha256"], spec["search_candidate_sha256"], spec["effective_config_sha256"],
                spec["device_class"], int(bool(spec["fixture"])), text, digest, "QUEUED", None, now, now))
            self._event(db, "UNIT_REGISTERED", fit_id, spec["dag_node"])
            return "QUEUED"

    def unit(self, fit_id):
        rows = self._q("SELECT * FROM units WHERE fit_id=?", (fit_id,))
        return self._unit_row(rows[0]) if rows else None

    def units(self, dag_node=None, stage=None):
        sql, args = "SELECT * FROM units", []
        where = []
        if dag_node is not None:
            where.append("dag_node=?")
            args.append(dag_node)
        if stage is not None:
            where.append("stage=?")
            args.append(stage)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [self._unit_row(r) for r in self._q(sql + " ORDER BY created_at, fit_id", args)]

    def attempts(self, fit_id=None):
        if fit_id is None:
            rows = self._q("SELECT * FROM attempts ORDER BY attempt_id")
        else:
            rows = self._q("SELECT * FROM attempts WHERE fit_id=? ORDER BY attempt_id", (fit_id,))
        return [self._attempt_row(r) for r in rows]

    def terminal(self, fit_id):
        rows = self._q("SELECT * FROM terminals WHERE fit_id=?", (fit_id,))
        if not rows:
            return None
        out = dict(rows[0])
        out["receipt"] = json.loads(out["receipt"])
        out["source_detail"] = _loads(out["source_detail"])
        return out

    def terminals(self):
        out = {}
        for row in self._q("SELECT * FROM terminals"):
            item = dict(row)
            item["receipt"] = json.loads(item["receipt"])
            item["source_detail"] = _loads(item["source_detail"])
            out[item["fit_id"]] = item
        return out

    # ------------------------------------------------------------ claims
    def claim(self, fit_id, owner, host, boot_id, device_class, ttl, exclusive_host=True):
        if ttl <= 0:
            raise LedgerError("ttl must be positive")
        with self._tx() as db:
            now = self.clock()
            unit = db.execute("SELECT * FROM units WHERE fit_id=?", (fit_id,)).fetchone()
            if unit is None:
                raise LedgerError("unknown fit_id; register the unit first")
            terminal = db.execute("SELECT * FROM terminals WHERE fit_id=?", (fit_id,)).fetchone()
            if terminal is not None:
                return dict(result="ADOPT", terminal=dict(receipt=json.loads(terminal["receipt"]),
                            receipt_sha256=terminal["receipt_sha256"], source=terminal["source"]))
            if unit["state"] == "FAILED":
                return dict(result="REFUSED", cause=unit["cause"])
            running = db.execute("SELECT * FROM attempts WHERE fit_id=? AND outcome='RUNNING' ORDER BY token DESC",
                                 (fit_id,)).fetchall()
            for attempt in running:
                if attempt["lease_expires_at"] > now:
                    return dict(result="HELD", holder=dict(owner=attempt["owner"], host=attempt["host"],
                                token=attempt["token"], lease_expires_at=attempt["lease_expires_at"]))
                db.execute("UPDATE attempts SET outcome='INTERRUPTED', cause='LEASE_EXPIRED', ended_at=? "
                           "WHERE attempt_id=?", (now, attempt["attempt_id"]))
                self._event(db, "LEASE_EXPIRED", fit_id, detail=dict(token=attempt["token"]))
            if exclusive_host:
                busy = db.execute("SELECT fit_id, token FROM attempts WHERE outcome='RUNNING' AND host=? "
                                  "AND lease_expires_at>? AND fit_id!=? AND device_class NOT IN ('cpu','cpu_fixture') "
                                  "LIMIT 1", (host, now, fit_id)).fetchone()
                if busy is not None:
                    self._set_unit(db, fit_id, "WAIT_RESOURCE", "HOST_GPU_BUSY")
                    return dict(result="WAIT_RESOURCE", cause="HOST_GPU_BUSY")
            physical = db.execute("SELECT * FROM attempts WHERE fit_id=? AND physical=1 ORDER BY attempt_id",
                                  (fit_id,)).fetchall()
            kind, parent, parent_status, checkpoint = "INITIAL", None, None, None
            if physical:
                last = physical[-1]
                if last["outcome"] != "INTERRUPTED":
                    cause = f"LAST_ATTEMPT_{last['outcome']}"
                    self._set_unit(db, fit_id, "FAILED", cause)
                    return dict(result="REFUSED", cause=cause)
                if len(physical) >= MAX_PHYSICAL_ATTEMPTS:
                    self._set_unit(db, fit_id, "FAILED", "ATTEMPTS_EXHAUSTED")
                    self._event(db, "ATTEMPTS_EXHAUSTED", fit_id)
                    return dict(result="REFUSED", cause="ATTEMPTS_EXHAUSTED")
                parent, parent_status = last["attempt_id"], last["status_sha256"]
                ck = _loads(last["checkpoint"])
                if isinstance(ck, dict) and all(ck.get(k) for k in CHECKPOINT_KEYS) and ck.get("epoch") is not None:
                    kind, checkpoint = "RESUME", ck
                elif any(a["kind"] == "RECOVERY" for a in physical):
                    self._set_unit(db, fit_id, "FAILED", "RECOVERY_EXHAUSTED")
                    self._event(db, "RECOVERY_EXHAUSTED", fit_id)
                    return dict(result="REFUSED", cause="RECOVERY_EXHAUSTED")
                else:
                    kind = "RECOVERY"
            token = self._next_token(db)
            index = len(physical) + 1
            cur = db.execute(
                "INSERT INTO attempts(fit_id, idx, kind, parent_attempt, parent_status_sha256, owner, host, boot_id, "
                "device_class, token, started_at, renewed_at, lease_expires_at, outcome, checkpoint, physical) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'RUNNING',?,1)",
                (fit_id, index, kind, parent, parent_status, owner, host, boot_id, device_class, token, now, now,
                 now + ttl, None if checkpoint is None else common.canonical(checkpoint)))
            self._set_unit(db, fit_id, "RUNNING", None)
            self._event(db, "CLAIMED", fit_id, detail=dict(token=token, kind=kind, index=index, host=host))
            return dict(result="CLAIMED", token=token, attempt=dict(
                attempt_id=cur.lastrowid, index=index, kind=kind, parent_attempt=parent,
                parent_status_sha256=parent_status, checkpoint=checkpoint))

    def renew(self, fit_id, token, ttl, progress=None):
        with self._tx() as db:
            now = self.clock()
            row = db.execute("SELECT * FROM attempts WHERE fit_id=? AND token=?", (fit_id, token)).fetchone()
            if row is None or row["outcome"] != "RUNNING" or row["lease_expires_at"] < now:
                return False
            if db.execute("SELECT 1 FROM terminals WHERE fit_id=?", (fit_id,)).fetchone() is not None:
                return False
            newer = db.execute("SELECT 1 FROM attempts WHERE fit_id=? AND token>?", (fit_id, token)).fetchone()
            if newer is not None:
                return False
            db.execute("UPDATE attempts SET renewed_at=?, lease_expires_at=?, progress=COALESCE(?, progress) "
                       "WHERE attempt_id=?", (now, now + ttl, None if progress is None else common.canonical(progress),
                                              row["attempt_id"]))
            return True

    def _verify_receipt(self, unit, fit_id, doc):
        if not isinstance(doc, dict) or set(doc) != RECEIPT_KEYS:
            raise ReceiptMismatch(f"receipt document requires exactly {sorted(RECEIPT_KEYS)}")
        expected = dict(fit_id=fit_id, fold=unit["fold"], experiment=unit["experiment"],
                        effective_config_sha256=unit["effective_config_sha256"],
                        genotype_canonical_sha256=unit["genotype"])
        for key, value in expected.items():
            if doc[key] != value:
                raise ReceiptMismatch(f"receipt {key} does not belong to this fit")

    def _publication(self, unit, fit_id, doc, attempt, disposition, cause=None):
        return dict(kind="FOLD_TERMINAL", disposition=disposition, cause=cause, fit_id=fit_id, fold=unit["fold"],
                    experiment=unit["experiment"], stage=unit["stage"], dag_node=unit["dag_node"],
                    horizon=unit["horizon"], genotype_canonical_sha256=unit["genotype"],
                    search_candidate_sha256=unit["search_candidate"],
                    effective_config_sha256=unit["effective_config_sha256"], seed=common.SEED,
                    direction=common.DIRECTION,
                    receipt_sha256=None if doc is None else common.sha256_json(doc),
                    attempt=None if attempt is None else dict(index=attempt["idx"], token=attempt["token"],
                                                              kind=attempt["kind"]))

    def _outbox(self, db, fit_id, payload):
        db.execute("INSERT OR IGNORE INTO outbox(fit_id, kind, payload, payload_sha256, created_at, state) "
                   "VALUES(?,?,?,?,?,'PENDING')", (fit_id, payload["kind"], common.canonical(payload),
                                                    common.sha256_json(payload), self.clock()))

    def complete(self, fit_id, token, doc):
        with self._tx() as db:
            now = self.clock()
            unit = db.execute("SELECT * FROM units WHERE fit_id=?", (fit_id,)).fetchone()
            if unit is None:
                raise LedgerError("unknown fit_id")
            self._verify_receipt(unit, fit_id, doc)
            attempt = db.execute("SELECT * FROM attempts WHERE fit_id=? AND token=?", (fit_id, token)).fetchone()
            if attempt is None:
                raise LedgerError("unknown fencing token for this fit")
            digest = common.sha256_json(doc)
            terminal = db.execute("SELECT * FROM terminals WHERE fit_id=?", (fit_id,)).fetchone()
            if terminal is not None:
                if terminal["receipt_sha256"] == digest and terminal["attempt_id"] == attempt["attempt_id"]:
                    return "ALREADY"
                if attempt["outcome"] == "RUNNING":
                    db.execute("UPDATE attempts SET outcome='DUPLICATE_DISCARDED', ended_at=? WHERE attempt_id=?",
                               (now, attempt["attempt_id"]))
                self._event(db, "DUPLICATE_DISCARDED", fit_id, detail=dict(token=token))
                return "DUPLICATE_DISCARDED"
            if attempt["outcome"] not in ("RUNNING", "INTERRUPTED", "SUPERSEDED"):
                raise LedgerError(f"attempt already terminal: {attempt['outcome']}")
            newer = db.execute("SELECT 1 FROM attempts WHERE fit_id=? AND token>?", (fit_id, token)).fetchone()
            late = attempt["outcome"] != "RUNNING" or attempt["lease_expires_at"] < now or newer is not None
            db.execute("INSERT INTO terminals VALUES(?,?,?,?,?,?,?)",
                       (fit_id, common.canonical(doc), digest, attempt["attempt_id"], "FIT", None, now))
            db.execute("UPDATE attempts SET outcome='COMPLETE', cause=?, ended_at=? WHERE attempt_id=?",
                       ("LATE_RECEIPT" if late else None, now, attempt["attempt_id"]))
            db.execute("UPDATE attempts SET outcome='SUPERSEDED', cause='TERMINAL_FROM_OTHER_ATTEMPT', ended_at=? "
                       "WHERE fit_id=? AND outcome='RUNNING'", (now, fit_id))
            self._set_unit(db, fit_id, "COMPLETE", None)
            self._outbox(db, fit_id, self._publication(unit, fit_id, doc, attempt, "COMPLETE"))
            self._event(db, "COMPLETE_LATE" if late else "COMPLETE", fit_id, detail=dict(token=token))
            return "ACCEPTED_LATE" if late else "ACCEPTED"

    def report(self, fit_id, token, outcome, cause, parent_status_sha256=None, checkpoint=None, peak_bytes=None):
        """Record a non-success outcome of the attempt holding ``token``.

        ``parent_status_sha256`` is the digest of this attempt's evaluator
        STATUS file, handed to the next linked attempt.
        """
        if outcome not in REPORT_OUTCOMES:
            raise LedgerError(f"unsupported outcome {outcome}")
        with self._tx() as db:
            now = self.clock()
            unit = db.execute("SELECT * FROM units WHERE fit_id=?", (fit_id,)).fetchone()
            attempt = db.execute("SELECT * FROM attempts WHERE fit_id=? AND token=?", (fit_id, token)).fetchone()
            if unit is None or attempt is None:
                raise LedgerError("unknown fit or fencing token")
            if db.execute("SELECT 1 FROM terminals WHERE fit_id=?", (fit_id,)).fetchone() is not None:
                return "IGNORED_TERMINAL_EXISTS"
            newer = db.execute("SELECT 1 FROM attempts WHERE fit_id=? AND token>?", (fit_id, token)).fetchone()
            if attempt["outcome"] not in ("RUNNING", "INTERRUPTED") or (attempt["outcome"] != "RUNNING" and newer):
                return "IGNORED_STALE"
            db.execute("UPDATE attempts SET outcome=?, cause=?, ended_at=?, status_sha256=COALESCE(?, status_sha256), "
                       "checkpoint=COALESCE(?, checkpoint), physical=?, peak_bytes=COALESCE(?, peak_bytes) "
                       "WHERE attempt_id=?",
                       (outcome, cause, now, parent_status_sha256,
                        None if checkpoint is None else common.canonical(checkpoint),
                        0 if outcome == "NOT_STARTED" else 1, peak_bytes, attempt["attempt_id"]))
            if newer:
                return "RECORDED_HISTORY"
            if outcome == "INTERRUPTED":
                self._set_unit(db, fit_id, "INTERRUPTED", cause)
            elif outcome == "NOT_STARTED":
                self._set_unit(db, fit_id, "WAIT_RESOURCE", cause)
            else:
                self._set_unit(db, fit_id, "FAILED", f"{outcome}: {cause}")
                self._outbox(db, fit_id, self._publication(unit, fit_id, None, attempt, "FAILED", f"{outcome}: {cause}"))
            self._event(db, outcome, fit_id, detail=dict(token=token, cause=cause))
            return "RECORDED"

    def adopt_prior(self, fit_id, doc, source):
        """Adopt an already verified receipt of the SAME fit_id (no attempt, no fit)."""
        with self._tx() as db:
            unit = db.execute("SELECT * FROM units WHERE fit_id=?", (fit_id,)).fetchone()
            if unit is None:
                raise LedgerError("unknown fit_id")
            self._verify_receipt(unit, fit_id, doc)
            if db.execute("SELECT 1 FROM terminals WHERE fit_id=?", (fit_id,)).fetchone() is not None:
                return "ALREADY"
            if db.execute("SELECT 1 FROM attempts WHERE fit_id=? AND outcome='RUNNING'", (fit_id,)).fetchone():
                raise LedgerError("cannot adopt while an attempt is running")
            db.execute("INSERT INTO terminals VALUES(?,?,?,?,?,?,?)", (fit_id, common.canonical(doc),
                       common.sha256_json(doc), None, "PRIOR", common.canonical(source), self.clock()))
            self._set_unit(db, fit_id, "COMPLETE", None)
            self._outbox(db, fit_id, self._publication(unit, fit_id, doc, None, "ADOPTED_CACHE"))
            self._event(db, "ADOPTED_PRIOR", fit_id, detail=source)
            return "ADOPTED_PRIOR"

    def recover(self, host=None, boot_id=None):
        """Boot/tick recovery: previous-boot attempts of ``host`` and expired leases become INTERRUPTED."""
        changed = []
        with self._tx() as db:
            now = self.clock()
            for row in db.execute("SELECT * FROM attempts WHERE outcome='RUNNING'").fetchall():
                cause = None
                if host is not None and row["host"] == host and row["boot_id"] != boot_id:
                    cause = "HOST_REBOOT"
                elif row["lease_expires_at"] < now:
                    cause = "LEASE_EXPIRED"
                if cause is None:
                    continue
                db.execute("UPDATE attempts SET outcome='INTERRUPTED', cause=?, ended_at=? WHERE attempt_id=?",
                           (cause, now, row["attempt_id"]))
                if db.execute("SELECT 1 FROM terminals WHERE fit_id=?", (row["fit_id"],)).fetchone() is None:
                    self._set_unit(db, row["fit_id"], "INTERRUPTED", cause)
                self._event(db, cause, row["fit_id"], detail=dict(token=row["token"]))
                changed.append(dict(fit_id=row["fit_id"], attempt_id=row["attempt_id"], cause=cause))
        return changed

    def running_on_host(self, host):
        return [self._attempt_row(r) for r in self._q(
            "SELECT * FROM attempts WHERE outcome='RUNNING' AND host=? AND lease_expires_at>=?", (host, self.clock()))]

    def next_claimable(self, stages, device_class=None, limit=50):
        """Candidate fit units for a fixed-unit worker (no claim is made here)."""
        marks = ",".join("?" for _ in stages)
        rows = self._q(f"SELECT * FROM units WHERE stage IN ({marks}) AND state IN "
                       "('QUEUED','INTERRUPTED','WAIT_RESOURCE') ORDER BY created_at, fit_id LIMIT ?",
                       (*stages, limit))
        return [self._unit_row(r) for r in rows]

    # ------------------------------------------------------------ outbox
    def outbox_pending(self):
        return [dict(id=r["id"], fit_id=r["fit_id"], kind=r["kind"], payload=json.loads(r["payload"]),
                     payload_sha256=r["payload_sha256"])
                for r in self._q("SELECT * FROM outbox WHERE state='PENDING' ORDER BY id")]

    def outbox_all(self):
        return [dict(id=r["id"], fit_id=r["fit_id"], state=r["state"], payload=json.loads(r["payload"]),
                     delivery=_loads(r["delivery"])) for r in self._q("SELECT * FROM outbox ORDER BY id")]

    def outbox_mark(self, item_id, delivery):
        with self._tx() as db:
            db.execute("UPDATE outbox SET state='DELIVERED', delivered_at=?, delivery=? WHERE id=? AND state='PENDING'",
                       (self.clock(), common.canonical(delivery), item_id))

    # ------------------------------------------------------------ DAG
    def dag_upsert(self, node_id, stage, ord_, depends_on, spec):
        with self._tx() as db:
            row = db.execute("SELECT spec, depends_on FROM dag WHERE node_id=?", (node_id,)).fetchone()
            if row is not None:
                if json.loads(row["spec"]) != spec or json.loads(row["depends_on"]) != list(depends_on):
                    raise ValueError(f"DAG node {node_id} differs from the sealed plan")
                return False
            db.execute("INSERT INTO dag VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (node_id, stage, ord_, common.canonical(list(depends_on)), common.canonical(spec),
                        "WAIT_DEPENDENCY" if depends_on else "QUEUED", None, None, None, None, self.clock()))
            return True

    def dag_nodes(self):
        out = []
        for r in self._q("SELECT * FROM dag ORDER BY ord"):
            item = dict(r)
            item["depends_on"] = json.loads(item["depends_on"])
            item["spec"] = json.loads(item["spec"])
            item["summary"] = _loads(item["summary"])
            out.append(item)
        return out

    def dag_set(self, node_id, state, cause=None, summary=None):
        if state not in UNIT_STATES:
            raise LedgerError(f"invalid DAG state {state}")
        with self._tx() as db:
            now = self.clock()
            row = db.execute("SELECT state FROM dag WHERE node_id=?", (node_id,)).fetchone()
            db.execute("UPDATE dag SET state=?, cause=?, summary=COALESCE(?, summary), updated_at=?, "
                       "activated_at=CASE WHEN activated_at IS NULL AND ? IN ('QUEUED','RUNNING') THEN ? "
                       "ELSE activated_at END, completed_at=CASE WHEN ? IN ('COMPLETE','FAILED') THEN ? "
                       "ELSE completed_at END WHERE node_id=?",
                       (state, cause, None if summary is None else common.canonical(summary), now, state, now,
                        state, now, node_id))
            if row is not None and row["state"] != state:
                self._event(db, "DAG_" + state, node_id=node_id, detail=dict(cause=cause))

    def dag_event(self, node_id, event, detail=None):
        """Idempotent stage event, e.g. SEARCH_TERMINAL written by the swarm wrapper."""
        with self._tx() as db:
            db.execute("INSERT OR IGNORE INTO dag_events(node_id, event, detail, at) VALUES(?,?,?,?)",
                       (node_id, event, None if detail is None else common.canonical(detail), self.clock()))

    def dag_events(self, node_id=None):
        rows = self._q("SELECT * FROM dag_events" + (" WHERE node_id=?" if node_id else "") + " ORDER BY id",
                       (node_id,) if node_id else ())
        return [dict(node_id=r["node_id"], event=r["event"], detail=_loads(r["detail"]), at=r["at"]) for r in rows]

    def events(self, limit=200):
        return [dict(r) for r in self._q("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------ search state (E's plain JSON)
    def put_search_state(self, node_id, generation, state):
        """Persist one immutable generation of the swarm search state (budgets, seen, seeds, generation)."""
        digest = common.sha256_json(state)
        with self._tx() as db:
            row = db.execute("SELECT state_sha256 FROM search_state WHERE node_id=? AND generation=?",
                             (node_id, int(generation))).fetchone()
            if row is not None:
                if row["state_sha256"] != digest:
                    raise LedgerError("search state of a generation is immutable")
                return "ALREADY"
            db.execute("INSERT INTO search_state VALUES(?,?,?,?,?)",
                       (node_id, int(generation), common.canonical(state), digest, self.clock()))
            return "STORED"

    def search_state(self, node_id):
        rows = self._q("SELECT * FROM search_state WHERE node_id=? ORDER BY generation DESC LIMIT 1", (node_id,))
        if not rows:
            return None
        return dict(node_id=node_id, generation=rows[0]["generation"], state=json.loads(rows[0]["state"]),
                    state_sha256=rows[0]["state_sha256"])
