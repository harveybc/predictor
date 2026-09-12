#!/usr/bin/env python3
"""C73: load a v3 terminal supersession into the cube, additively.

Reads the supersession index and every v3 terminal under
descriptor-first custody, refuses if any terminal names another
verification, and writes one run, one row per variable and one row per
compared descriptor. A second load of the same run writes nothing; a
row whose content differs from what is already stored for the same key
refuses. fact_variable_characterization is never written. Outbox
health is reported before and after, so backlog and dead letters are
never hidden by a direct load.
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
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(
        v, bool) else None


def read_supersession(state_dir: Path, v3_dirname: str = "terminals_v3"):
    custody = Custody(state_dir, require_owner=True)
    try:
        index = custody.root_snapshot().read(
            "TERMINAL_SUPERSESSION.v2.json").json()
        snap = custody.walk_to(v3_dirname)
        bodies = [snap.read(n).json() for n in sorted(snap.files)
                  if n.endswith(".json")]
    finally:
        custody.close()
    vsha = index["verification_sha256"]
    foreign = [b["variable_id"] for b in bodies
               if b.get("verification_sha256") != vsha]
    if foreign:
        raise LoadRefusal(f"{len(foreign)} v3 terminals name another "
                          "verification than the supersession index")
    if len(bodies) != index["v3_terminals_written"]:
        raise LoadRefusal("the v3 directory does not hold the number of "
                          "terminals the index declares")
    return index, bodies


def load(dsn: str, index: dict, bodies: list[dict], report: dict,
         observed_at: str) -> dict:
    from sqlalchemy import create_engine, text
    engine = create_engine(dsn)
    vsha = index["verification_sha256"]
    written = {"runs": 0, "variables": 0, "descriptors": 0}
    try:
        with engine.begin() as c:
            prior = c.execute(text(
                "SELECT population_verdict, recomputation_verdict "
                "FROM public.dim_terminal_verification "
                "WHERE verification_sha256=:v"), {"v": vsha}).first()
            run = (index["population_verdict"],
                   index["recomputation_verdict"])
            if prior and tuple(prior) != run:
                raise LoadRefusal("this verification is already loaded "
                                  "with different verdicts")
            if not prior:
                c.execute(text(
                    "INSERT INTO public.dim_terminal_verification "
                    "(verification_sha256, report_schema, "
                    " population_verdict, recomputation_verdict, "
                    " divergence_count, variables_by_layer, "
                    " published_attempt, observed_at) VALUES "
                    "(:v,:s,:p,:r,:d,CAST(:l AS jsonb),:a,:t)"),
                    {"v": vsha, "s": report["schema"], "p": run[0],
                     "r": run[1],
                     "d": report["population"]["divergence_count"],
                     "l": json.dumps(
                         report["recomputation"]["variables_by_layer"]),
                     "a": report["recomputation"]["published_attempt"],
                     "t": observed_at})
                written["runs"] = 1
            for b in bodies:
                src = b["source"]["sha256"] if isinstance(
                    b["source"], dict) else None
                row = {"v": vsha, "id": b["variable_id"],
                       "o": b["producer_declared_outcome"],
                       "l": b["layer"], "r": b["recomputation"],
                       "s": src, "t": b["terminal_sha256"]}
                have = c.execute(text(
                    "SELECT layer, recomputation, terminal_sha256 FROM "
                    "public.fact_terminal_verification_variable WHERE "
                    "verification_sha256=:v AND variable_id=:id"),
                    row).first()
                if have and tuple(have) != (row["l"], row["r"], row["t"]):
                    raise LoadRefusal(f"{b['variable_id']}: a stored row "
                                      "differs from the terminal")
                if not have:
                    c.execute(text(
                        "INSERT INTO public.fact_terminal_verification_"
                        "variable VALUES (:v,:id,:o,:l,:r,:s,:t)"), row)
                    written["variables"] += 1
                for d, st in sorted(b["descriptors"].items()):
                    rec = st.get("recomputed")
                    drow = {"v": vsha, "id": b["variable_id"], "d": d,
                            "l": st["layer"], "st": st["state"],
                            "sp": st["specificity"],
                            "rs": st.get("reason"),
                            "pv": _num(st.get("published")),
                            "rv": _num(rec),
                            "rn": rec if isinstance(rec, str) else None}
                    have = c.execute(text(
                        "SELECT layer, state FROM public.fact_terminal_"
                        "verification_descriptor WHERE verification_sha256"
                        "=:v AND variable_id=:id AND descriptor=:d"),
                        drow).first()
                    if have and tuple(have) != (drow["l"], drow["st"]):
                        raise LoadRefusal(f"{b['variable_id']}/{d}: a "
                                          "stored row differs")
                    if not have:
                        c.execute(text(
                            "INSERT INTO public.fact_terminal_verification"
                            "_descriptor VALUES (:v,:id,:d,:l,:st,:sp,:rs,"
                            ":pv,:rv,:rn)"), drow)
                        written["descriptors"] += 1
    finally:
        engine.dispose()
    return written


def outbox_health():
    try:
        from olap import outbox as ob
        return ob.health(ob.ensure_outbox())
    except Exception as exc:  # the loader must not hide a broken outbox
        return {"healthy": False, "error": type(exc).__name__}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--observed-at", required=True)
    ap.add_argument("--dsn", default=None)
    a = ap.parse_args(argv)
    index, bodies = read_supersession(a.state_dir.expanduser())
    report = json.loads(a.report.read_text())
    if report["verification_sha256"] != index["verification_sha256"]:
        raise LoadRefusal("the report is not the verification the v3 "
                          "terminals were written from")
    dsn = a.dsn or ("postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{h}:{p}/"
                    "{PGDATABASE}".format(h=os.environ.get("PGHOST",
                                                           "localhost"),
                                          p=os.environ.get("PGPORT", "5432"),
                                          **os.environ))
    before = outbox_health()
    written = load(dsn, index, bodies, report, a.observed_at)
    after = outbox_health()
    print(json.dumps({"written": written, "outbox_before": before,
                      "outbox_after": after}, indent=1, sort_keys=True,
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
