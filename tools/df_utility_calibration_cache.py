#!/usr/bin/env python3
"""Verified reuse of calibration records of the SAME scientific computation (Q2).

A record is stored under the digest of its canonical computation key (harness code, numeric
dependencies, generator and parameters, seed tape, length, operator declaration, branch pair
and widths, probe model, window, blocks, rows policy, target, horizon, margin, effective alpha,
failure policy, simulations and confidence — never a family or unit label). A lookup is a hit
only when the stored bytes match their recorded digest, the record passes the CURRENT verifier
(every simulation recounted, bound re-derived), and its key equals the consumer's field by field.
A consumer keeps its own campaign binding and declares the source; a hit is never passed off as
new simulations. Accounting keeps unique computations, producers, consumers, reads and costs
apart. Writes are atomic (temporary directory + rename); a race keeps the first record and
records the second as a consumer; a different record for the same key is quarantined, not taken.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")
META = "META.json"
RECORD = "record.json"
CONFLICT = "CONFLICT.json"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class CalibrationCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, key_sha: str) -> Path:
        return self.root / key_sha

    def _locked(self, key_sha: str):
        lock = self.root / f".{key_sha}.lock"
        fh = open(lock, "a+")
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fh

    # --- lookup -------------------------------------------------------------------------------------
    def lookup(self, key: dict) -> tuple:
        """(record bytes, provenance) on a verified hit; (None, why) otherwise — an incomplete or
        conflicting entry is a typed miss, never an exception."""
        key_sha = H.computation_sha256(key)
        d = self._dir(key_sha)
        if not d.is_dir():
            return None, "absent"
        if (d / CONFLICT).is_file():
            return None, "conflict recorded for this computation; disposition pending, nothing served"
        if not (d / RECORD).is_file() or not (d / META).is_file():
            return None, f"incomplete entry ({'record' if not (d / RECORD).is_file() else 'META'} absent)"
        try:
            meta = json.loads((d / META).read_text())
            body = (d / RECORD).read_bytes()
        except (OSError, ValueError):
            return None, "corrupt (unreadable)"
        if hashlib.sha256(body).hexdigest() != meta.get("record_sha256"):
            return None, "corrupt (bytes differ from the recorded digest)"
        try:
            rec = json.loads(body)
        except ValueError:
            return None, "corrupt (not JSON)"
        problems = H.calibration_record_problems(rec)
        if problems:
            return None, "record does not pass the current verifier: " + "; ".join(problems)
        if rec.get("computation_sha256") != key_sha or rec.get("computation") != key:
            differing = sorted(k for k in set(key) | set(rec.get("computation") or {})
                               if key.get(k) != (rec.get("computation") or {}).get(k))
            return None, f"another computation ({', '.join(differing)})"
        derived = H.derive_calibration(rec)
        if derived["problems"]:
            return None, "record does not recount: " + "; ".join(derived["problems"])
        return body, {"key_sha256": key_sha, "producer": meta.get("producer"), "record_sha256": meta.get("record_sha256"),
                      "stored_at": meta.get("stored_at")}

    def _entry_complete(self, d: Path) -> bool:
        if not (d / RECORD).is_file() or not (d / META).is_file():
            return False
        try:
            meta = json.loads((d / META).read_text())
            body = (d / RECORD).read_bytes()
            json.loads(body)
        except (OSError, ValueError):
            return False
        return hashlib.sha256(body).hexdigest() == meta.get("record_sha256")

    def _quarantine_incomplete(self, d: Path, key_sha: str, reason: str) -> str:
        q = self.root / "quarantine"
        q.mkdir(exist_ok=True)
        target = q / f"{key_sha}.{time.time_ns()}.{os.getpid()}"
        os.rename(d, target)
        (target / "WHY.json").write_text(json.dumps({"reason": reason, "at": _now()}))
        return str(target)

    # --- store ---------------------------------------------------------------------------------------
    def store(self, key_sha: str, body: bytes, producer: dict) -> dict:
        """Atomic, typed. The first complete record for a key wins. A later record that is
        scientifically equal (same simulations, counts and bounds; cost/provenance may differ)
        is a duplicate producer; a scientifically different one is a CONFLICT with a recorded
        pending disposition (nothing served until resolved). An incomplete entry (record or META
        absent, malformed, interrupted publication) is quarantined apart and the new record
        becomes usable: recovery through the normal miss path, never an unhandled exception."""
        rec = json.loads(body)
        if rec.get("computation_sha256") != key_sha:
            raise ValueError("record's computation key is not the store key")
        digest = hashlib.sha256(body).hexdigest()
        d = self._dir(key_sha)
        with self._locked(key_sha):
            recovered = None
            if d.exists() and not self._entry_complete(d) and not (d / CONFLICT).is_file():
                recovered = self._quarantine_incomplete(d, key_sha, "incomplete or malformed entry found at store")
            if d.is_dir():
                meta = json.loads((d / META).read_text())
                if (d / CONFLICT).is_file():
                    self._add_conflict(d, body, digest, producer)
                    return {"status": "CONFLICT_RECORDED", "record_sha256": digest}
                if meta.get("record_sha256") == digest:
                    meta.setdefault("duplicate_producers", []).append({**producer, "at": _now(), "scientifically_equal": True})
                    self._write_meta(d, meta)
                    return {"status": "DUPLICATE_PRODUCER", "record_sha256": digest, "scientifically_equal": True}
                existing = json.loads((d / RECORD).read_bytes())
                if scientific_fingerprint(existing) == scientific_fingerprint(rec):
                    meta.setdefault("duplicate_producers", []).append({**producer, "at": _now(), "scientifically_equal": True,
                                                                        "record_sha256": digest,
                                                                        "differs_in": "cost/provenance only"})
                    self._write_meta(d, meta)
                    return {"status": "DUPLICATE_PRODUCER", "record_sha256": digest, "scientifically_equal": True}
                self._add_conflict(d, body, digest, producer, existing_digest=meta.get("record_sha256"))
                return {"status": "CONFLICT_RECORDED", "record_sha256": digest, "served": None}
            tmp = Path(tempfile.mkdtemp(prefix=f".{key_sha[:12]}-", dir=self.root))
            (tmp / RECORD).write_bytes(body)
            meta = {"schema": "df_utility_calibration_cache_meta.v2", "key_sha256": key_sha, "record_sha256": digest,
                    "stored_at": _now(), "producer": producer, "simulations": rec.get("n_sims"),
                    "cost_cpu_seconds": (rec.get("cost") or {}).get("cpu_seconds"),
                    "consumers": {}, "reads": 0, "verification_seconds_measured": 0.0,
                    "avoided_cpu_seconds_projected": 0.0, "quarantined_incomplete": [recovered] if recovered else []}
            (tmp / META).write_text(json.dumps(meta, indent=1))
            os.rename(tmp, d)
            return {"status": "RECOVERED_INCOMPLETE_ENTRY" if recovered else "STORED", "record_sha256": digest,
                    **({"quarantined": recovered} if recovered else {})}

    def _add_conflict(self, d: Path, body: bytes, digest: str, producer: dict, existing_digest: str | None = None):
        q = d / "conflicts"
        q.mkdir(exist_ok=True)
        (q / f"{digest}.json").write_bytes(body)
        path = d / CONFLICT
        doc = json.loads(path.read_text()) if path.is_file() else {
            "schema": "df_utility_calibration_cache_conflict.v1", "disposition": "PENDING",
            "records": [{"record_sha256": existing_digest, "role": "served_before_conflict"}] if existing_digest else [],
            "note": "scientifically different records for one computation key; nothing is served until a "
                    "recorded disposition names the valid one"}
        doc["records"].append({"record_sha256": digest, "producer": producer, "at": _now()})
        path.write_text(json.dumps(doc, indent=1))

    def _write_meta(self, d: Path, meta: dict):
        tmp = d / f"{META}.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, d / META)

    # --- accounting ------------------------------------------------------------------------------
    def note_consumer(self, key_sha: str, *, consumer: dict, kind: str, verification_seconds: float) -> None:
        """Idempotent by attempt identity: a hit reported again for the same attempt neither
        multiplies the projected avoided work nor the reads."""
        d = self._dir(key_sha)
        ident = str(consumer.get("attempt_dir") or consumer.get("id") or json.dumps(consumer, sort_keys=True))
        with self._locked(key_sha):
            meta = json.loads((d / META).read_text())
            if isinstance(meta.get("consumers"), list):                 # meta v1
                meta["consumers"] = {str(c.get("attempt_dir", i)): c for i, c in enumerate(meta["consumers"])}
                meta.setdefault("verification_seconds_measured", meta.pop("verification_seconds_total", 0.0))
                meta.setdefault("avoided_cpu_seconds_projected", meta.pop("saved_cpu_seconds", 0.0))
            if ident in meta["consumers"]:
                return
            meta["consumers"][ident] = {**consumer, "kind": kind, "at": _now(), "verification_seconds": verification_seconds}
            meta["reads"] = len(meta["consumers"])
            meta["verification_seconds_measured"] = round(meta.get("verification_seconds_measured", 0.0) + verification_seconds, 3)
            if kind == "HIT":
                meta["avoided_cpu_seconds_projected"] = round(meta.get("avoided_cpu_seconds_projected", 0.0)
                                                             + float(meta.get("cost_cpu_seconds") or 0.0), 3)
            self._write_meta(d, meta)

    def status(self) -> dict:
        entries = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir() and (p / META).is_file()):
            meta = json.loads((d / META).read_text())
            consumers = meta.get("consumers") or {}
            consumers = list(consumers.values()) if isinstance(consumers, dict) else consumers
            entries.append({"key_sha256": meta["key_sha256"], "simulations": meta.get("simulations"),
                            "cost_cpu_seconds": meta.get("cost_cpu_seconds"), "reads": len(consumers),
                            "hits": sum(1 for c in consumers if c.get("kind") == "HIT"),
                            "avoided_cpu_seconds_projected": meta.get("avoided_cpu_seconds_projected", meta.get("saved_cpu_seconds", 0.0)),
                            "verification_seconds_measured": meta.get("verification_seconds_measured", meta.get("verification_seconds_total", 0.0)),
                            "duplicate_producers": len(meta.get("duplicate_producers", [])),
                            "quarantined_incomplete": len(meta.get("quarantined_incomplete", [])),
                            "conflict": (d / CONFLICT).is_file()})
        return {"schema": "df_utility_calibration_cache_status.v2", "root": str(self.root),
                "unique_computations": len(entries), "entries": entries,
                "simulations_unique": sum(e["simulations"] or 0 for e in entries),
                "hits": sum(e["hits"] for e in entries),
                "avoided_cpu_seconds_projected": round(sum(e["avoided_cpu_seconds_projected"] or 0 for e in entries), 3),
                "verification_seconds_measured": round(sum(e["verification_seconds_measured"] or 0 for e in entries), 3),
                "conflicts": sum(1 for e in entries if e["conflict"]),
                "note": "avoided CPU is a projection from the producers' recorded cost; verification is measured"}


def scientific_fingerprint(rec: dict) -> dict:
    """What makes two records the same scientific result: the computation, every simulation, the
    counts and the bounds — not cost, not provenance."""
    return {"computation_sha256": rec.get("computation_sha256"), "per_sim_sha256": rec.get("per_sim_sha256"),
            "scored": rec.get("scored"), "failed": rec.get("failed"), "advances": rec.get("advances"),
            "false_advance_rate": rec.get("false_advance_rate"), "upper_bound": rec.get("upper_bound")}
