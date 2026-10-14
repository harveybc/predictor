"""Durable warehouse outbox for I6-DMO metric reports with exact readback (DMO12).

Reports use the existing governed metrics schema (``predictor_olap_store.query.normalise_report``)
and identity (``report_sha256``). The warehouse ``POST /api/v1/metrics`` route is idempotent by that
digest, so a retry after an outage or restart cannot duplicate facts. An item is DELIVERED only when
a readback of ``gov_metric`` for its digest returns exactly the expected (metric, value, unit,
horizon, split) multiset; a differing readback is MISMATCH and never closes delivery.

Delivery never trains: this module imports nothing from training or the model. The HTTP client
takes its bearer token from the process environment at call time (the existing fleet's
EnvironmentFile mechanism); it never reads a token file, stores or prints the token.
"""
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path

STORE_SRC = Path(__file__).resolve().parents[2] / "olap" / "store" / "src"
if str(STORE_SRC) not in sys.path:
    sys.path.insert(0, str(STORE_SRC))
from predictor_olap_store.query import normalise_report, report_sha256  # noqa: E402

from tools.i6n_archive.sink import _urllib  # noqa: E402  (shared stdlib HTTP transport)
from tools.i6n_swarm.common import canonical  # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS items(
  report_sha256 TEXT PRIMARY KEY, body TEXT NOT NULL, source_unit TEXT NOT NULL, state TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, created_at REAL NOT NULL, first_try_at REAL,
  posted_at REAL, delivered_at REAL, readback_sha256 TEXT);
CREATE TABLE IF NOT EXISTS history(
  id INTEGER PRIMARY KEY AUTOINCREMENT, report_sha256 TEXT NOT NULL, at REAL NOT NULL, event TEXT NOT NULL,
  detail TEXT);
"""
STATES = ("PENDING", "POSTED", "DELIVERED", "MISMATCH")


def seal_report(report):
    """Return a copy carrying its canonical ``report_sha256``; refuse a different claimed digest."""
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    digest = report_sha256(body)
    if "report_sha256" in report and report["report_sha256"] != digest:
        raise ValueError("REPORT_DIGEST_MISMATCH")
    return dict(body, report_sha256=digest)


def expected_rows(report):
    normalised = normalise_report({k: v for k, v in report.items() if k != "report_sha256"})
    return sorted(_row_key(m) for m in normalised["metrics"])


def _row_key(row):
    value = row.get("value")
    horizon = row.get("horizon")
    return (row["metric"], None if value is None else float(value), row.get("unit"),
            None if horizon is None else int(horizon), row.get("split"))


def _rows_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


class Outbox:
    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._db = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(DDL)
        self._db.row_factory = sqlite3.Row

    @classmethod
    def open_readonly(cls, path):
        box = cls.__new__(cls)
        box.path, box.clock = Path(path), time.time
        box._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None, timeout=30)
        box._db.row_factory = sqlite3.Row
        return box

    def close(self):
        self._db.close()

    def _history(self, digest, event, detail=None):
        self._db.execute("INSERT INTO history(report_sha256, at, event, detail) VALUES(?,?,?,?)",
                         (digest, self.clock(), event, None if detail is None else canonical(detail)))

    def enqueue(self, report, *, source_unit):
        sealed = seal_report(report)
        digest = sealed["report_sha256"]
        body = canonical(sealed)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            row = self._db.execute("SELECT body FROM items WHERE report_sha256=?", (digest,)).fetchone()
            if row is None:
                self._db.execute("INSERT INTO items(report_sha256, body, source_unit, state, created_at) "
                                 "VALUES(?,?,?,?,?)", (digest, body, source_unit, "PENDING", self.clock()))
                self._history(digest, "ENQUEUED", dict(source_unit=source_unit))
            elif row["body"] != body:
                raise ValueError("OUTBOX_BODY_CONFLICT")
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return digest

    def items(self):
        return [dict(r) for r in self._db.execute("SELECT * FROM items ORDER BY created_at, report_sha256")]

    def history(self, digest=None):
        sql, args = "SELECT * FROM history", ()
        if digest:
            sql, args = sql + " WHERE report_sha256=?", (digest,)
        return [dict(r) for r in self._db.execute(sql + " ORDER BY id", args)]

    def _set(self, digest, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        self._db.execute(f"UPDATE items SET {cols} WHERE report_sha256=?", (*fields.values(), digest))

    def _readback(self, client, report):
        rows = client.query_metrics(report["report_sha256"])
        actual = sorted(_row_key(r) for r in rows)
        return actual, actual == expected_rows(report)

    def pump(self, client):
        """One bounded delivery pass. Never retrains; failures stay PENDING/POSTED for the next pass."""
        for item in self.items():
            if item["state"] in ("DELIVERED", "MISMATCH"):
                continue
            report = json.loads(item["body"])
            digest = item["report_sha256"]
            now = self.clock()
            self._set(digest, attempts=item["attempts"] + 1, first_try_at=item["first_try_at"] or now)
            try:
                if item["state"] == "POSTED":
                    actual, exact = self._readback(client, report)
                    if exact:
                        self._delivered(digest, actual)
                        continue
                    if actual:
                        self._mismatch(digest, actual)
                        continue
                receipt = client.post_report(report)
                if not isinstance(receipt, dict) or receipt.get("report_sha256") != digest:
                    raise RuntimeError("WAREHOUSE_RECEIPT_DIGEST_MISMATCH")
                self._set(digest, state="POSTED", posted_at=self.clock(), last_error=None)
                self._history(digest, "POSTED")
                actual, exact = self._readback(client, report)
                if exact:
                    self._delivered(digest, actual)
                else:
                    self._mismatch(digest, actual)
            except Exception as exc:  # delivery failure: retry later, never retrain
                self._set(digest, last_error=str(exc)[:300])
                self._history(digest, "DELIVERY_ERROR", dict(error=str(exc)[:300]))
        return self.summary()

    def _delivered(self, digest, actual):
        self._set(digest, state="DELIVERED", delivered_at=self.clock(), readback_sha256=_rows_sha(actual),
                  last_error=None)
        self._history(digest, "DELIVERED_READBACK_EXACT", dict(rows=len(actual)))

    def _mismatch(self, digest, actual):
        self._set(digest, state="MISMATCH", last_error="READBACK_MISMATCH", readback_sha256=_rows_sha(actual))
        self._history(digest, "READBACK_MISMATCH", dict(rows=len(actual)))

    def summary(self):
        items = self.items()
        counts = {s: sum(1 for i in items if i["state"] == s) for s in STATES}
        durations = [i["delivered_at"] - i["first_try_at"] for i in items
                     if i["state"] == "DELIVERED" and i["first_try_at"] is not None]
        return dict(expected=len(items), delivered=counts["DELIVERED"],
                    pending=counts["PENDING"] + counts["POSTED"], posted=counts["POSTED"],
                    mismatch=counts["MISMATCH"], delivery_seconds=durations,
                    last_errors=sorted({i["last_error"] for i in items if i["last_error"]}))


class HttpWarehouse:
    """The governed metrics API. The token comes from ``os.environ[token_env]`` at call time only."""

    def __init__(self, url, token_env, transport=None):
        self._base = url.rstrip("/")
        self._token_env = token_env
        self._transport = transport or _urllib

    def __repr__(self):
        return "HttpWarehouse(<configured url>)"

    def _token(self):
        value = os.environ.get(self._token_env)
        if not value:
            raise RuntimeError(f"TOKEN_ENV_MISSING: {self._token_env} (run under the unit's EnvironmentFile)")
        return value

    def _call(self, method, path, body=None):
        headers = {"Authorization": "Bearer " + self._token()}
        data = None
        if body is not None:
            data = json.dumps(body, sort_keys=True, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        status, raw = self._transport(method, self._base + path, data=data, headers=headers)
        if status not in (200, 201):
            raise RuntimeError(f"WAREHOUSE_HTTP_{status}")
        return json.loads(raw)

    def post_report(self, report):
        return self._call("POST", "/api/v1/metrics", report)

    def query_metrics(self, digest):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("bad digest")
        sql = ("SELECT report_sha256, metric, value, unit, horizon, split FROM gov_metric "
               f"WHERE report_sha256='{digest}' LIMIT 10000")
        return self._call("GET", "/api/v1/query?" + urllib.parse.urlencode({"sql": sql}))["rows"]
