#!/usr/bin/env python3
"""What D3 actually needs before it may be measured, checked rather than asserted.

Block 3 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`:
"identify its actual prerequisites". §4 of the sealed D3 design names three:

    R2   a repaired adjudicator
    R6   a CURRENT coverage view
    N3   a reconciled productive micro-run

    "...para toda medicion que fundamente decisiones."

The word that matters is *actual*. A work-plan table saying `IMPLEMENTED` is a claim about a
past moment; what governs whether D3 can be measured today is whether the thing is there now.
So each prerequisite is probed against the running service and the checkout, and a
prerequisite that cannot be probed is `UNVERIFIABLE`, never assumed satisfied.

Read-only throughout: the same query route the consistency watch uses, and no write method.

usage:
  df_d3_prerequisites.py --service-url URL --out REPORT.json
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The selection views R6 proposes. Without them, a measurement that cites coverage cannot say
#: WHICH run and WHICH code digest it counted, and `df_fact_coverage` holds more than one.
R6_VIEWS = ("df_coverage_current", "df_coverage_history")
#: The adjudicator R2 repaired.
R2_TOOL = REPO / "tools" / "df_d2_adjudicate.py"
#: What N3 leaves behind when it has actually run in production.
N3_ACT = REPO / "docs" / "handoffs" / "MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Reader:
    """Read-only access to the running warehouse. There is no write method on this class."""

    def __init__(self, url: str, token_env: str, schema: str):
        self.url, self.schema = url.rstrip("/"), schema
        self.token = os.environ.get(token_env)
        if not self.token:
            raise SystemExit(f"{token_env} is not set: the service token comes from the "
                             "environment, never from an argument")

    def query(self, sql: str) -> list:
        request = urllib.request.Request(
            f"{self.url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
            headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(request, timeout=120) as handle:
            return json.load(handle)["rows"]

    def relations(self) -> set:
        return {row["table_name"] for row in self.query(
            "SELECT table_name FROM information_schema.tables ORDER BY 1 LIMIT 5000")}

    def count(self, relation: str):
        try:
            return self.query(f'SELECT count(*) AS n FROM "{self.schema}"."{relation}"'
                              " LIMIT 1")[0]["n"]
        except Exception:
            return None


def check_r2() -> dict:
    """The adjudicator exists in this checkout and offers the repaired entry point."""
    if not R2_TOOL.is_file():
        return {"state": "ABSENT", "detail": f"{R2_TOOL.name} is not in this checkout"}
    body = R2_TOOL.read_text(encoding="utf-8")
    return {"state": "PRESENT", "tool": str(R2_TOOL.relative_to(REPO)),
            "bytes": len(body.encode()),
            "detail": "present in the checkout; whether its verdicts are accepted is a "
                      "review question, not a probe"}


def check_n3() -> dict:
    """The production restart act. Its absence would mean the productive route is unproven."""
    if not N3_ACT.is_file():
        return {"state": "UNVERIFIABLE",
                "detail": "the production restart act is not in this checkout"}
    return {"state": "PRESENT", "act": str(N3_ACT.relative_to(REPO)),
            "detail": "the act exists; it records a restart executed by the reviewer"}


def check_r6(reader: Reader) -> dict:
    """The coverage selection views, in the cube that a D3 measurement would actually read."""
    relations = reader.relations()
    present = sorted(view for view in R6_VIEWS if view in relations)
    missing = sorted(view for view in R6_VIEWS if view not in relations)
    facts = {name: reader.count(name) for name in
             ("df_fact_coverage", "df_fact_coverage_v2", "df_fact_coverage_v1_v2_map")
             if name in relations}
    ambiguity = None
    if "df_fact_coverage" in relations:
        try:
            rows = reader.query(
                'SELECT count(DISTINCT run_id) AS runs, count(DISTINCT code_sha256) AS codes'
                f' FROM "{reader.schema}"."df_fact_coverage" LIMIT 1')[0]
            ambiguity = {"runs": rows["runs"], "code_digests": rows["codes"]}
        except Exception as exc:
            ambiguity = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    state = "PRESENT" if not missing else "NOT_APPLIED"
    return {"state": state, "views_present": present, "views_missing": missing,
            "coverage_rows": facts, "unselected_population": ambiguity,
            "detail": None if not missing else
            (f"{missing} do not exist in the cube, so a coverage figure read today cannot say "
             "which run and which code digest it counted; the fact tables hold more than one")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--schema", default="main")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    reader = Reader(args.service_url, args.token_env, args.schema)
    prerequisites = {"R2_adjudicator": check_r2(),
                     "R6_coverage_selection": check_r6(reader),
                     "N3_productive_micro_run": check_n3()}
    blocking = sorted(name for name, outcome in prerequisites.items()
                      if outcome["state"] not in ("PRESENT",))
    report = {
        "schema": "df_d3_prerequisites.v1", "generated_utc": now(),
        "service": args.service_url,
        "step": "D3 — quantization/compression, time-frequency, detectors",
        "design": "docs/integracion_workplan_2026_09_10/"
                  "07_DISENO_D3_CUANTIZACION_TIEMPO_FRECUENCIA_DETECTORES_2026_09_14.md",
        "prerequisites": prerequisites,
        "blocking": blocking,
        "ready_to_measure": not blocking,
        "note": ("Probed, not read from a status table. A prerequisite recorded as satisfied "
                 "in a past packet is a claim about that moment; what governs a measurement "
                 "today is what is there today. Nothing here executes D3, selects an operator "
                 "or writes anything."),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ready_to_measure": report["ready_to_measure"],
                      "blocking": blocking,
                      "prerequisites": {name: outcome["state"]
                                        for name, outcome in prerequisites.items()}}, indent=1))
    return 0 if report["ready_to_measure"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
