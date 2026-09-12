#!/usr/bin/env python3
"""C92: load terminals v4 and verification v3 into the cube, additively.

Reads TERMINAL_SUPERSESSION.v3.json and every v4 terminal under custody,
refuses a terminal naming another verification, writes one run, one row
per variable and one row per compared descriptor, writes nothing on a
second load, refuses a conflicting stored row, never touches
fact_variable_characterization or C73's tables, and reports outbox
health — backlog and dead letters — before and after.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from descriptor_custody import Custody  # noqa: E402


class LoadRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def read_v4(state_dir: Path):
    c = Custody(state_dir, require_owner=True)
    try:
        idx = c.root_snapshot().read("TERMINAL_SUPERSESSION.v3.json").json()
        snap = c.walk_to("terminals_v4")
        bodies = [snap.read(n).json() for n in sorted(snap.files) if n.endswith(".json")]
    finally:
        c.close()
    if any(b.get("verification_sha256") != idx["verification_sha256"] for b in bodies):
        raise LoadRefusal("a v4 terminal names another verification")
    if len(bodies) != idx["v4_written"]:
        raise LoadRefusal("v4 terminal count differs from the index")
    return idx, bodies


def load(dsn, idx, bodies, report, observed_at):
    from sqlalchemy import create_engine, text
    e = create_engine(dsn)
    v = idx["verification_sha256"]
    w = {"runs": 0, "variables": 0, "descriptors": 0}
    try:
        with e.begin() as c:
            prior = c.execute(text("SELECT population_verdict FROM public.dim_terminal_verification_v2 "
                                   "WHERE verification_sha256=:v"), {"v": v}).first()
            if prior and prior[0] != report["population"]["verdict"]:
                raise LoadRefusal("run already loaded with a different verdict")
            if not prior:
                c.execute(text(
                    "INSERT INTO public.dim_terminal_verification_v2 VALUES "
                    "(:v,:s,:cr,:p,:d,CAST(:l AS jsonb),CAST(:sb AS jsonb),:t,now())"),
                    {"v": v, "s": report["schema"],
                     "cr": report["census_identity"]["recomputed_canonical"],
                     "p": report["population"]["verdict"],
                     "d": report["population"]["divergence_count"],
                     "l": json.dumps(report["layers"]),
                     "sb": json.dumps(report["semantic_sweep"]["by_state"]), "t": observed_at})
                w["runs"] = 1
            for b in bodies:
                app = b["appearances"]
                first = next(iter(app.values()), {}) if app else {}
                sem = first.get("semantic", {})
                row = {"v": v, "id": b["variable_id"], "o": b["producer_declared_outcome"],
                       "l": b["layer"], "ss": sem.get("state"), "at": sem.get("arrow_type"),
                       "wd": json.dumps(b.get("published_numeric_descriptors_withdrawn", [])),
                       "t": b["terminal_sha256"]}
                have = c.execute(text("SELECT layer, terminal_sha256 FROM public.fact_terminal_verification_variable_v2 "
                                      "WHERE verification_sha256=:v AND variable_id=:id"), row).first()
                if have and tuple(have) != (row["l"], row["t"]):
                    raise LoadRefusal(f"{b['variable_id']}: stored row differs")
                if not have:
                    c.execute(text("INSERT INTO public.fact_terminal_verification_variable_v2 VALUES "
                                   "(:v,:id,:o,:l,:ss,:at,CAST(:wd AS jsonb),:t)"), row)
                    w["variables"] += 1
                for entry in app.values():
                    for dname, st in (entry.get("recomputation") or {}).get("descriptors", {}).items():
                        drow = {"v": v, "id": b["variable_id"], "d": dname,
                                "s": st["state"], "sp": st["specificity"]}
                        h = c.execute(text("SELECT state FROM public.fact_terminal_verification_descriptor_v2 "
                                           "WHERE verification_sha256=:v AND variable_id=:id AND descriptor=:d"), drow).first()
                        if h and h[0] != drow["s"]:
                            raise LoadRefusal(f"{b['variable_id']}/{dname}: stored row differs")
                        if not h:
                            c.execute(text("INSERT INTO public.fact_terminal_verification_descriptor_v2 "
                                           "VALUES (:v,:id,:d,:s,:sp)"), drow)
                            w["descriptors"] += 1
    finally:
        e.dispose()
    return w


def outbox_health():
    try:
        from olap import outbox as ob
        return ob.health(ob.ensure_outbox())
    except Exception as exc:  # a broken outbox is reported, never hidden
        return {"healthy": False, "error": type(exc).__name__}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--observed-at", required=True)
    ap.add_argument("--dsn")
    a = ap.parse_args(argv)
    idx, bodies = read_v4(a.state_dir.expanduser())
    report = json.loads(a.report.read_text())
    if report["verification_sha256"] != idx["verification_sha256"]:
        raise LoadRefusal("report is not the verification the v4 terminals came from")
    dsn = a.dsn or "postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{h}:{p}/{PGDATABASE}".format(
        h=os.environ.get("PGHOST", "localhost"), p=os.environ.get("PGPORT", "5432"), **os.environ)
    before = outbox_health()
    w = load(dsn, idx, bodies, report, a.observed_at)
    print(json.dumps({"written": w, "outbox_before": before, "outbox_after": outbox_health()},
                     indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
