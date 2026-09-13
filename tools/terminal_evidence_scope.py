#!/usr/bin/env python3
"""C112 (order 2026-09-12): say exactly what terminal evidence establishes.

The auditor accepted terminals v4 WITH PHYSICAL AND STATISTICAL SCOPE:
1,501 variables have numeric descriptors recomputed from their bound
bytes; that does not mean 1,501 variables have verified semantics, unit,
role or license. Four scopes are kept apart here, in executable form:

  NUMERIC_DESCRIPTORS_RECOMPUTED  numeric descriptors recomputed from bytes
  PHYSICAL_TYPE_KNOWN             physical storage type known
  SEMANTIC_DECLARATIONS_KNOWN     semantics, role, unit, license and missing
                                  policy declared
  PRODUCER_AUTHORITY_ONLY         not identifiable; the producer's word only

`describe()` is the only place a count becomes a sentence, and
`language_problems()` refuses any "verified variables" phrasing. No scope
by itself licenses a variable: population membership is the member-by-
member join in per_variable_design_v5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DECISION = "TERMINALS_V4_ACCEPTED_WITH_PHYSICAL_AND_STATISTICAL_SCOPE"
DECIDED_BY = "EXTERNAL: MUSASHI_AUDIT_ROUND7_C87_C105_2026_09_12 (recorded, not issued here)"
SCHEMA = "crispdm.terminals_v4_disposition.v1"

SCOPES = {
    "NUMERIC_DESCRIPTORS_RECOMPUTED": "variables with numeric descriptors recomputed from their bound bytes",
    "PHYSICAL_TYPE_KNOWN": "variables whose physical storage type is known",
    "SEMANTIC_DECLARATIONS_KNOWN": "variables with semantics, role, unit, license and missing policy declared",
    "PRODUCER_AUTHORITY_ONLY": "variables not identifiable, resting on the producer's declaration only",
    "SEMANTICALLY_UNRESOLVED": "variables whose semantic type is unresolved; no numeric value of theirs is eligible",
}
NOT_IMPLIED = {
    "NUMERIC_DESCRIPTORS_RECOMPUTED": ["semantics", "role", "unit", "license", "missing policy"],
    "PHYSICAL_TYPE_KNOWN": ["semantics", "unit", "role", "license"],
    "SEMANTIC_DECLARATIONS_KNOWN": ["independent recomputation"],
    "PRODUCER_AUTHORITY_ONLY": ["independent recomputation", "physical type"],
    "SEMANTICALLY_UNRESOLVED": ["eligibility of any numeric value"],
}
FORBIDDEN = re.compile(
    r"\b(?:verified\s+(?:variables?|terminals?|columns?)"
    r"|(?:variables?|terminals?|columns?)\s+(?:verified|verificad[oa]s?)"
    r"|variables?\s+verificadas?)\b", re.IGNORECASE)
UNDECLARED = {None, "", "UNKNOWN"}
SEMANTIC_FIELDS = ("semantics", "role", "unit", "license", "missing_policy")


class ScopeRefusal(ValueError):
    pass


def describe(scope: str, n: int) -> str:
    if scope not in SCOPES:
        raise ScopeRefusal(f"UNKNOWN_SCOPE: {scope!r}")
    if type(n) is not int or n < 0:
        raise ScopeRefusal("COUNT_MUST_BE_A_NON_NEGATIVE_INTEGER")
    text = f"{n:,} {SCOPES[scope]}"
    if language_problems(text):
        raise ScopeRefusal(f"FORBIDDEN_LANGUAGE: {text}")
    return text


def language_problems(text: str) -> list[str]:
    return [m.group(0) for m in FORBIDDEN.finditer(text)]


def semantic_declarations_known(census: dict) -> int:
    """Variables declaring every semantic field with something other than
    UNKNOWN. Absence counts as undeclared."""
    return sum(1 for v in census.get("variables", [])
               if all(v.get(f) not in UNDECLARED for f in SEMANTIC_FIELDS))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_disposition(verification: Path, supersession: Path, census: Path | None) -> dict:
    ver = json.loads(Path(verification).read_text())
    sup = json.loads(Path(supersession).read_text())
    if sup["verification_sha256"] != ver["verification_sha256"]:
        raise ScopeRefusal("SUPERSESSION_NOT_BOUND_TO_THIS_VERIFICATION")
    layers = ver["layers"]
    if layers != sup["layers"]:
        raise ScopeRefusal("LAYER_COUNTS_DIFFER_BETWEEN_VERIFICATION_AND_SUPERSESSION")
    if layers.get("DIVERGES", 0):
        raise ScopeRefusal("A_DIVERGING_TERMINAL_CANNOT_BE_ACCEPTED_WITH_SCOPE")
    total = ver["population"]["terminals_read"]
    counts = {
        "NUMERIC_DESCRIPTORS_RECOMPUTED": layers["INDEPENDENTLY_RECOMPUTED"],
        "PHYSICAL_TYPE_KNOWN": layers["INDEPENDENTLY_RECOMPUTED"] + layers["PHYSICALLY_TYPED"]
        + layers["SEMANTICALLY_UNRESOLVED"],
        "SEMANTIC_DECLARATIONS_KNOWN": semantic_declarations_known(json.loads(Path(census).read_text()))
        if census else None,
        "PRODUCER_AUTHORITY_ONLY": layers["PRODUCER_DECLARED"],
        "SEMANTICALLY_UNRESOLVED": layers["SEMANTICALLY_UNRESOLVED"],
    }
    if counts["PHYSICAL_TYPE_KNOWN"] + counts["PRODUCER_AUTHORITY_ONLY"] + layers["SOURCE_BOUND"] != total:
        raise ScopeRefusal("SCOPES_DO_NOT_PARTITION_THE_POPULATION")
    doc = {
        "schema": SCHEMA, "decision": DECISION, "decided_by": DECIDED_BY,
        "binds": {"verification_sha256": ver["verification_sha256"],
                  "supersession_sha256": sup["supersession_sha256"],
                  "verification_file_sha256": _sha(verification),
                  "supersession_file_sha256": _sha(supersession),
                  "census_file_sha256": _sha(census) if census else "NOT_SUPPLIED"},
        "terminals": total,
        "columns_physically_typed": ver["semantic_sweep"]["columns_evaluated"],
        "scopes": {s: {"count": n, "means": SCOPES[s], "does_not_imply": NOT_IMPLIED[s],
                       "sentence": describe(s, n) if n is not None else "NOT_DERIVED: census not supplied"}
                   for s, n in counts.items()},
        "consumption": "no scope by itself licenses a variable for an experiment; membership "
                       "requires the member-by-member join of per_variable_design_v5",
        "cube": "the six evidence layers may stay in the cube; none of them licenses a variable",
        "grants_nothing": True,
    }
    text = json.dumps(doc, sort_keys=True)
    if language_problems(text):
        raise ScopeRefusal(f"FORBIDDEN_LANGUAGE_IN_DISPOSITION: {language_problems(text)}")
    if str(Path.home()) in text:
        raise ScopeRefusal("ABSOLUTE_HOME_PATH")
    doc["disposition_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verification", type=Path, required=True)
    ap.add_argument("--supersession", type=Path, required=True)
    ap.add_argument("--census", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    doc = build_disposition(a.verification, a.supersession, a.census)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if a.out:
        if a.out.exists():
            raise SystemExit(f"REFUSED: {a.out.name} exists; write-once")
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
