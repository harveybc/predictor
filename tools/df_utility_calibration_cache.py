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
        """(record bytes, provenance) on a verified hit; (None, why) otherwise."""
        key_sha = H.computation_sha256(key)
        d = self._dir(key_sha)
        if not (d / RECORD).is_file() or not (d / META).is_file():
            return None, "absent"
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

    # --- store ---------------------------------------------------------------------------------------
    def store(self, key_sha: str, body: bytes, producer: dict) -> dict:
        """Atomic: the first record for a key wins; an identical later one is a duplicate
        producer; a different one is quarantined beside it and never served."""
        rec = json.loads(body)
        if rec.get("computation_sha256") != key_sha:
            raise ValueError("record's computation key is not the store key")
        digest = hashlib.sha256(body).hexdigest()
        d = self._dir(key_sha)
        with self._locked(key_sha):
            if (d / RECORD).is_file():
                meta = json.loads((d / META).read_text())
                if meta.get("record_sha256") == digest:
                    meta.setdefault("duplicate_producers", []).append({**producer, "at": _now()})
                    self._write_meta(d, meta)
                    return {"status": "DUPLICATE_PRODUCER", "record_sha256": digest}
                q = d / "quarantine"
                q.mkdir(exist_ok=True)
                (q / f"{digest}.json").write_bytes(body)
                meta.setdefault("quarantined", []).append({**producer, "record_sha256": digest, "at": _now()})
                self._write_meta(d, meta)
                return {"status": "QUARANTINED_DIFFERENT_RECORD", "record_sha256": digest, "served": meta["record_sha256"]}
            tmp = Path(tempfile.mkdtemp(prefix=f".{key_sha[:12]}-", dir=self.root))
            (tmp / RECORD).write_bytes(body)
            meta = {"schema": "df_utility_calibration_cache_meta.v1", "key_sha256": key_sha, "record_sha256": digest,
                    "stored_at": _now(), "producer": producer, "simulations": rec.get("n_sims"),
                    "cost_cpu_seconds": (rec.get("cost") or {}).get("cpu_seconds"),
                    "consumers": [], "reads": 0, "verification_seconds_total": 0.0, "saved_cpu_seconds": 0.0}
            (tmp / META).write_text(json.dumps(meta, indent=1))
            os.rename(tmp, d)
            return {"status": "STORED", "record_sha256": digest}

    def _write_meta(self, d: Path, meta: dict):
        tmp = d / f"{META}.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, d / META)

    # --- accounting ------------------------------------------------------------------------------
    def note_consumer(self, key_sha: str, *, consumer: dict, kind: str, verification_seconds: float) -> None:
        d = self._dir(key_sha)
        with self._locked(key_sha):
            meta = json.loads((d / META).read_text())
            meta["consumers"].append({**consumer, "kind": kind, "at": _now(), "verification_seconds": verification_seconds})
            meta["reads"] += 1
            meta["verification_seconds_total"] = round(meta["verification_seconds_total"] + verification_seconds, 3)
            if kind == "HIT":
                meta["saved_cpu_seconds"] = round(meta["saved_cpu_seconds"] + float(meta.get("cost_cpu_seconds") or 0.0), 3)
            self._write_meta(d, meta)

    def status(self) -> dict:
        entries = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir() and (p / META).is_file()):
            meta = json.loads((d / META).read_text())
            entries.append({"key_sha256": meta["key_sha256"], "simulations": meta.get("simulations"),
                            "cost_cpu_seconds": meta.get("cost_cpu_seconds"), "reads": meta.get("reads"),
                            "hits": sum(1 for c in meta.get("consumers", []) if c.get("kind") == "HIT"),
                            "saved_cpu_seconds": meta.get("saved_cpu_seconds"),
                            "verification_seconds_total": meta.get("verification_seconds_total"),
                            "quarantined": len(meta.get("quarantined", []))})
        return {"schema": "df_utility_calibration_cache_status.v1", "root": str(self.root),
                "unique_computations": len(entries), "entries": entries,
                "simulations_unique": sum(e["simulations"] or 0 for e in entries),
                "hits": sum(e["hits"] for e in entries),
                "saved_cpu_seconds": round(sum(e["saved_cpu_seconds"] or 0 for e in entries), 3),
                "verification_seconds_total": round(sum(e["verification_seconds_total"] or 0 for e in entries), 3)}
