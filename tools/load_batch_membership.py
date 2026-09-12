#!/usr/bin/env python3
"""C62 (order 2026-09-12): load EXACT batch membership into the cube.

The cube knew which RUN produced a row and nothing about which BATCH a
variable belonged to. "Which variables were in batch_00007, exactly?"
had no answer, so no batch could be re-run or audited on its own.

Membership is read from the v2 terminals under descriptor-first
custody, never from the producer's in-memory state, and is written
idempotently: loading the same batch twice changes nothing, loading a
batch with one variable added or removed changes its membership
digest and is refused rather than silently merged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptor_custody import Custody  # noqa: E402


class LoadRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def read_batches(state_dir: Path, v2_dirname: str) -> dict:
    custody = Custody(state_dir, require_owner=True)
    try:
        snap = custody.walk_to(v2_dirname)
        batches: dict[str, list[dict]] = {}
        attempt = None
        for name in sorted(snap.files):
            if not name.endswith(".json"):
                continue
            d = snap.read(name).json()
            key = d.get("batch") or "UNDECLARED"
            batches.setdefault(key, []).append({
                "variable_id": d["variable_id"],
                "outcome": d.get("outcome") or "UNDECLARED",
                "terminal_sha256": d["terminal_sha256"],
                "source_sha256": (d.get("source") or {}).get("sha256"),
                "measured_at": d.get("measured_at"),
            })
            attempt = attempt or d.get("superseded_at")
        return {"batches": batches, "reads": len(custody.reads())}
    finally:
        custody.close()


def membership_digest(rows) -> str:
    return hashlib.sha256(
        json.dumps(sorted(r["variable_id"] for r in rows),
                   separators=(",", ":")).encode()).hexdigest()


def load(dsn: str, attempt: str, batches: dict, observed_at: str,
         *, dry_run: bool) -> dict:
    from sqlalchemy import create_engine, text
    engine = create_engine(dsn)
    written, unchanged, refused = 0, 0, []
    try:
        with engine.begin() as c:
            for key, rows in sorted(batches.items()):
                digest = membership_digest(rows)
                prior = c.execute(text(
                    "SELECT membership_sha256, variables_declared "
                    "FROM public.dim_characterization_batch "
                    "WHERE terminal_attempt=:a AND batch_key=:b"),
                    {"a": attempt, "b": key}).first()
                if prior and prior[0] != digest:
                    refused.append({
                        "batch": key,
                        "why": "this batch is already loaded with a "
                               "DIFFERENT membership; a membership is "
                               "not silently merged",
                        "recorded_sha256": prior[0],
                        "offered_sha256": digest})
                    continue
                if prior:
                    unchanged += 1
                    continue
                if dry_run:
                    written += 1
                    continue
                outcomes = {}
                for r in rows:
                    outcomes[r["outcome"]] = outcomes.get(
                        r["outcome"], 0) + 1
                c.execute(text(
                    "INSERT INTO public.dim_characterization_batch "
                    "(terminal_attempt, batch_key, variables_declared, "
                    " membership_sha256, outcomes_json, observed_at) "
                    "VALUES (:a,:b,:n,:d, CAST(:o AS jsonb), :t)"),
                    {"a": attempt, "b": key, "n": len(rows),
                     "d": digest, "o": json.dumps(outcomes),
                     "t": observed_at})
                for r in rows:
                    c.execute(text(
                        "INSERT INTO public.bridge_batch_variable "
                        "(terminal_attempt, batch_key, variable_id, "
                        " outcome, terminal_sha256, source_sha256) "
                        "VALUES (:a,:b,:v,:o,:t,:s) "
                        "ON CONFLICT DO NOTHING"),
                        {"a": attempt, "b": key,
                         "v": r["variable_id"], "o": r["outcome"],
                         "t": r["terminal_sha256"],
                         "s": r["source_sha256"]})
                written += 1
        with engine.connect() as c:
            integrity = [dict(r) for r in c.execute(text(
                "SELECT * FROM public.v_batch_membership_integrity "
                "WHERE terminal_attempt=:a ORDER BY batch_key"),
                {"a": attempt}).mappings()]
    finally:
        engine.dispose()
    return {"batches_written": written, "batches_unchanged": unchanged,
            "refused": refused, "integrity": integrity}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--v2-dirname", default="terminals_v2")
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--observed-at", required=True)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    found = read_batches(a.state_dir.expanduser(), a.v2_dirname)
    dsn = a.dsn or (
        "postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:"
        "{PGPORT}/{PGDATABASE}".format(
            PGHOST=os.environ.get("PGHOST", "localhost"),
            PGPORT=os.environ.get("PGPORT", "5432"),
            **{k: os.environ[k] for k in
               ("PGUSER", "PGPASSWORD", "PGDATABASE")}))
    out = load(dsn, a.attempt, found["batches"], a.observed_at,
               dry_run=a.dry_run)
    summary = {
        "attempt": a.attempt,
        "batches_found": len(found["batches"]),
        "variables_found": sum(len(v) for v in found["batches"].values()),
        "custody_reads": found["reads"],
        "batches_written": out["batches_written"],
        "batches_unchanged": out["batches_unchanged"],
        "refused": out["refused"],
        "integrity_states": sorted({r["state"]
                                    for r in out["integrity"]}),
        "batches_not_exact": [r["batch_key"] for r in out["integrity"]
                              if r["state"] != "EXACT"],
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
