#!/usr/bin/env python3
"""C69-C72 (order 2026-09-12): verify the lake terminals for what they
actually claim, and name the result by what was actually checked.

The previous verifier printed TERMINALS_VERIFIED_EXACT after checking
population and filenames. The audit reproduced six ways that verdict
was false through its public API: an absent source, a fabricated
descriptor block, a terminal pointing at another variable's appearance
(silently rebound to the census's first appearance), absolute,
traversal and symlink sources whose outside bytes were read, a terminal
with an unknown outcome, an extra key and a wrong type, and a
self-consistent ledger and census over a substituted population.

What this verifier now does, refusing or diverging rather than
repairing:

  C69/C71  exact schemas and types for the pre-result ledger, the
           census objects it consumes and both terminal shapes; an
           EXPECTED census content digest supplied from outside, so a
           substituted population refuses before any terminal is read;
  C71      each MEASURED terminal binds its OWN declared appearance,
           that appearance must be listed by the variable, and it names
           exactly one file, read by contained components under
           descriptor-first custody (no absolute path, no traversal, no
           link followed) with digest and size checked against the
           census from the same bytes;
  C70      the population verdict is
           TERMINAL_POPULATION_AND_SOURCE_BINDING_VERIFIED or _DIVERGES
           and nothing more; the producer's MEASURED is reported as
           PRODUCER_DECLARED_MEASURED;
  C72      the published descriptor rows are compared, row by row, with
           values recomputed from those source bytes by
           `lake_descriptor_recompute`, which imports no producer code.
           Descriptors whose contract leaves a choice open are
           NOT_INDEPENDENTLY_VERIFIABLE, never counted as verified.

Nothing here grants eligibility. v1 and v2 terminals are read only; the
v3 supersession is written beside them and refuses if either changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptor_custody import (Custody, CustodyRefusal,  # noqa: E402
                                LeafIdentityRefusal)
import lake_descriptor_recompute as R  # noqa: E402

REPORT_SCHEMA = "crispdm.lake_terminal_verification.v2"
V3_SCHEMA = "crispdm.lake_characterization_terminal.v3"
LEDGER_SCHEMA = "crispdm.lake_characterization_pre_ledger.v1"

POPULATION_VERIFIED = "TERMINAL_POPULATION_AND_SOURCE_BINDING_VERIFIED"
POPULATION_DIVERGES = "TERMINAL_POPULATION_AND_SOURCE_BINDING_DIVERGES"
RECOMPUTED_AGREES = "INDEPENDENT_RECOMPUTATION_NO_DIVERGENCE"
RECOMPUTED_DIVERGES = "INDEPENDENT_RECOMPUTATION_DIVERGES"
RECOMPUTATION_NOT_RUN = "INDEPENDENT_RECOMPUTATION_NOT_RUN"

PRODUCER_DECLARED = "PRODUCER_DECLARED"
SOURCE_BOUND = "SOURCE_BOUND"
INDEPENDENTLY_RECOMPUTED = "INDEPENDENTLY_RECOMPUTED"

AGREES = "AGREES"
DIVERGES = "DIVERGES"
NOT_INDEPENDENTLY_VERIFIABLE = "NOT_INDEPENDENTLY_VERIFIABLE"

MEASURED = "MEASURED"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
TEMPORAL_AXIS_NAMES = frozenset({"timestamp", "close_time"})

LEDGER_TYPES = {"census_sha256": str, "censused_at": str,
                "conceptual_variables": int, "identities": list,
                "physical_appearances": int, "pre_ledger_sha256": str,
                "rule": str, "schema": str, "written_at": str}
TERMINAL_TYPES = {
    MEASURED: {"appearance": str, "batch": str, "concept_name": str,
               "descriptors": int, "entity": str, "measured_at": str,
               "not_identifiable": int, "outcome": str, "rows_used": int,
               "variable_id": str},
    NOT_IDENTIFIABLE: {"batch": str, "concept_name": str, "entity": str,
                       "measured_at": str, "outcome": str, "reason": str,
                       "variable_id": str},
}
#: census fields this verifier consumes, with exact types. Fields it
#: does not consume are not claimed to be validated.
CENSUS_APPEARANCE_TYPES = {"appearance_id": str, "relative_path": str,
                           "physical_sha256": str, "size_bytes": int}
CENSUS_VARIABLE_TYPES = {"variable_id": str, "appearances": list,
                         "concept_name": str, "entity": str}

BASE_DESCRIPTORS = frozenset({"characterization_disposition",
                              "n_observations", "missing_count",
                              "missingness_fraction", "non_finite_count",
                              "duplicate_count"})
FULL_DESCRIPTORS = BASE_DESCRIPTORS | frozenset(R.SPECIFICITY) - {
    "insufficient_finite_observations"}
INSUFFICIENT_DESCRIPTORS = BASE_DESCRIPTORS | {
    "insufficient_finite_observations"}
HEX64 = re.compile(r"[0-9a-f]{64}")


class VerificationRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")
        self.reason = msg


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_obj(o) -> str:
    return sha_bytes(json.dumps(o, sort_keys=True, separators=(",", ":"),
                                default=str).encode())


def exact_type(value, typ) -> bool:
    if typ is int:
        return type(value) is int
    return isinstance(value, typ)


def schema_violations(doc, types: dict, *, exact_keys: bool) -> list[str]:
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    out = []
    if exact_keys:
        extra = sorted(set(doc) - set(types))
        if extra:
            out.append(f"undeclared keys {extra}")
    for k, t in types.items():
        if k not in doc:
            out.append(f"missing key {k!r}")
        elif not exact_type(doc[k], t):
            out.append(f"{k!r} is {type(doc[k]).__name__}, not {t.__name__}")
    return out


def contained_parts(rel) -> list[str] | None:
    """A lake-relative POSIX path split into components, or None if it
    could name anything outside the lake root."""
    if type(rel) is not str or not rel or rel.startswith("/") \
            or "\\" in rel or "\x00" in rel:
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    return parts


# ------------------------------------------------------------ census
def load_census(census_path: Path, expected_sha256: str) -> dict:
    if type(expected_sha256) is not str or not HEX64.fullmatch(
            expected_sha256):
        raise VerificationRefusal(
            "an expected census content digest (64 lowercase hex) is "
            "required; a census that only agrees with itself proves no "
            "population")
    custody = Custody(census_path.parent, require_owner=True)
    try:
        art = custody.root_snapshot().read(census_path.name)
        doc = art.json()
    finally:
        custody.close()
    declared = doc.get("census_sha256")
    if declared != expected_sha256:
        raise VerificationRefusal(
            f"the census declares {str(declared)[:16]} but the expected "
            f"authority is {expected_sha256[:16]}: a substituted "
            "population is refused before any terminal is read")
    if census_path.name != f"census-{expected_sha256}.json":
        raise VerificationRefusal(
            "the census is not filed under its expected content address")
    problems = []
    for key, typ in (("appearances", list), ("variables", list)):
        if not isinstance(doc.get(key), typ):
            problems.append(f"census {key} is not a list")
    if problems:
        raise VerificationRefusal("; ".join(problems))
    apps, variables = {}, {}
    for a in doc["appearances"]:
        bad = schema_violations(a, CENSUS_APPEARANCE_TYPES,
                                exact_keys=False)
        if not bad and not HEX64.fullmatch(a["physical_sha256"]):
            bad.append("physical_sha256 is not 64 lowercase hex")
        if bad or a["appearance_id"] in apps:
            problems.append(f"appearance {a.get('appearance_id')!r}: "
                            f"{bad or ['duplicate id']}")
            continue
        apps[a["appearance_id"]] = a
    for v in doc["variables"]:
        bad = schema_violations(v, CENSUS_VARIABLE_TYPES, exact_keys=False)
        if not bad and (not v["appearances"] or len(set(v["appearances"]))
                        != len(v["appearances"])
                        or not all(type(x) is str for x in v["appearances"])):
            bad.append("appearances is not a non-empty list of distinct ids")
        if not bad and any(x not in apps for x in v["appearances"]):
            bad.append("lists an appearance the census does not define")
        if bad or v["variable_id"] in variables:
            problems.append(f"variable {v.get('variable_id')!r}: "
                            f"{bad or ['duplicate id']}")
            continue
        variables[v["variable_id"]] = v
    by_path: dict[str, set] = {}
    for a in apps.values():
        by_path.setdefault(a["relative_path"], set()).add(
            (a["physical_sha256"], a["size_bytes"]))
    for rel, facts in by_path.items():
        if len(facts) > 1:
            problems.append(f"{rel}: one path, {len(facts)} physical "
                            "identities in the census")
    if problems:
        raise VerificationRefusal(
            f"census schema: {problems[:5]} ({len(problems)} problems)")
    return {"census_sha256": declared, "file_sha256": art.sha256,
            "appearances": apps, "variables": variables}


# ------------------------------------------------------------ sources
class LakeReader:
    """Every source is read through ONE custody over the lake root,
    descending component by component; directory snapshots are
    retained for the whole verification."""

    def __init__(self, lake_root: Path):
        self.custody = Custody(lake_root, require_owner=True)
        self._dirs: dict[str, object] = {}

    def _dir(self, parts: list[str]):
        key = "/".join(parts)
        if key not in self._dirs:
            snap = self.custody.root_snapshot()
            for i, part in enumerate(parts):
                sub = "/".join(parts[:i + 1])
                if sub in self._dirs:
                    snap = self._dirs[sub]
                    continue
                if part not in snap.dirs:
                    kind = ("SOURCE_OUTSIDE_ROOT_OR_LINK"
                            if part in snap.others else "SOURCE_ABSENT")
                    raise _SourceProblem(kind, f"directory {sub} is not a "
                                               "directory of the lake")
                snap = snap.subdir(part)
                self._dirs[sub] = snap
        return self._dirs[key]

    def read(self, rel: str):
        parts = contained_parts(rel)
        if parts is None:
            raise _SourceProblem("SOURCE_OUTSIDE_ROOT_OR_LINK",
                                 "the path is absolute, empty or climbs "
                                 "out of the lake root")
        snap = self._dir(parts[:-1]) if len(parts) > 1 \
            else self.custody.root_snapshot()
        name = parts[-1]
        if name not in snap.files:
            kind = ("SOURCE_OUTSIDE_ROOT_OR_LINK" if name in snap.others
                    else "SOURCE_NOT_REGULAR" if name in snap.dirs
                    else "SOURCE_ABSENT")
            raise _SourceProblem(kind, f"{name} is not a regular file "
                                       "inventoried in its directory")
        try:
            return snap.read(name)
        except LeafIdentityRefusal as exc:
            raise _SourceProblem("SOURCE_IDENTITY_CHANGED", str(exc))
        except CustodyRefusal as exc:
            raise _SourceProblem("SOURCE_UNREADABLE", str(exc))

    def close(self):
        self.custody.close()


class _SourceProblem(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def column_window(payload: bytes, column: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(pa.BufferReader(payload))
    names = table.column_names
    if names.count(column) != 1:
        return None, len(table), f"column {column!r} appears " \
                                  f"{names.count(column)} times"
    col = table.column(column)
    if not (pa.types.is_floating(col.type) or pa.types.is_integer(col.type)
            or pa.types.is_decimal(col.type)):
        return None, len(table), f"column {column!r} is {col.type}, not " \
                                 "numeric"
    win = R.window_rows(len(table))
    import numpy as np
    arr = col.slice(0, win["rows_used"]).to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=np.float64), len(table), None


# ------------------------------------------------------------ published
def load_published_from_olap(dsn: str, attempt: str) -> dict:
    from sqlalchemy import create_engine, text
    engine = create_engine(dsn)
    out: dict[str, dict] = {}
    duplicates = []
    try:
        with engine.connect() as c:
            for r in c.execute(text(
                    "SELECT variable_id, descriptor, value, value_text, "
                    "identifiable, source_id, source_sha256, "
                    "window_contract FROM public.fact_variable_"
                    "characterization WHERE terminal_attempt = :a"),
                    {"a": attempt}).mappings():
                row = out.setdefault(r["variable_id"], {})
                if r["descriptor"] in row:
                    duplicates.append((r["variable_id"], r["descriptor"]))
                row[r["descriptor"]] = {
                    "value": r["value"], "value_text": r["value_text"],
                    "identifiable": bool(r["identifiable"]),
                    "source_id": r["source_id"],
                    "source_sha256": r["source_sha256"],
                    "window_contract": (json.loads(r["window_contract"])
                                        if r["window_contract"] else None)}
    finally:
        engine.dispose()
    return {"rows": out, "duplicates": duplicates, "attempt": attempt}


def compare_variable(vid: str, outcome: str, published: dict,
                     window, rows_in_file: int, source: dict,
                     census_sha256: str) -> dict:
    """Row-by-row comparison of one MEASURED variable."""
    rec = R.recompute(window)
    non_finite = int(rec["non_finite_count"][0])
    expected = (INSUFFICIENT_DESCRIPTORS
                if "insufficient_finite_observations" in rec
                else FULL_DESCRIPTORS)
    present = set(published)
    rows = {}
    problems = []
    if present != expected:
        problems.append({"kind": "ROW_CORRESPONDENCE_DIVERGES",
                         "missing": sorted(expected - present),
                         "unexpected": sorted(present - expected)})
    win = R.window_rows(rows_in_file)
    for d in sorted(present & expected):
        p = published[d]
        wc = p.get("window_contract") or {}
        for k in ("rows_total", "rows_used", "capped", "row_cap"):
            if k in wc and wc[k] != win[k]:
                problems.append({"kind": "WINDOW_DIVERGES", "descriptor": d,
                                 "field": k, "published": wc[k],
                                 "recomputed": win[k]})
        # The disposition row adjudicates a census variable, so it is
        # published against the census content digest; every
        # measurement row is published against the source file digest.
        # My first real run applied the file rule to the disposition row
        # and reported all 1,505 variables as diverging for it.
        expected_src = (census_sha256 if d == "characterization_disposition"
                        else source["sha256"])
        if p.get("source_sha256") != expected_src:
            problems.append({"kind": "PUBLISHED_SOURCE_DIGEST_DIVERGES",
                             "descriptor": d})
        if d == "characterization_disposition":
            ok = p.get("value_text") == outcome == MEASURED
            rows[d] = {"state": AGREES if ok else DIVERGES,
                       "specificity": R.FULLY, "layer": (
                           INDEPENDENTLY_RECOMPUTED if ok else SOURCE_BOUND)}
            continue
        level, why = R.specificity(d, non_finite)
        value, ident = rec.get(d, (None, None))
        computed = d in rec
        if computed:
            agrees = (bool(p["identifiable"]) == bool(ident)
                      and R.agree(d, p["value"], value if ident else None))
        else:
            agrees = None
        if level == R.FULLY:
            state = AGREES if agrees else DIVERGES
            layer = INDEPENDENTLY_RECOMPUTED if agrees else SOURCE_BOUND
        else:
            state, layer = NOT_INDEPENDENTLY_VERIFIABLE, SOURCE_BOUND
        rows[d] = {"state": state, "specificity": level, "reason": why,
                   "layer": layer,
                   "agrees_under_declared_reading": (
                       agrees if level != R.FULLY else None),
                   "published": p["value"], "published_identifiable":
                       p["identifiable"],
                   "recomputed": value if computed else "NOT_COMPUTED",
                   "recomputed_identifiable": ident if computed else None}
    diverged = problems or any(r["state"] == DIVERGES for r in rows.values())
    return {"variable_id": vid, "descriptors": rows, "problems": problems,
            "non_finite_in_window": non_finite,
            "layer": SOURCE_BOUND if diverged else INDEPENDENTLY_RECOMPUTED,
            "recomputation": DIVERGES if diverged else AGREES}


# ------------------------------------------------------------ verify
def verify(state_dir: Path, lake_root: Path, census_path: Path, *,
           expected_census_sha256: str, published: dict | None = None,
           strict_mode: bool = False) -> dict:
    census = load_census(census_path, expected_census_sha256)
    custody = Custody(state_dir, require_owner=True, strict_mode=strict_mode)
    lake = None
    divergences: list[dict] = []
    try:
        root = custody.root_snapshot()
        ledger = root.read("PRE_LEDGER.json").json()
        bad = schema_violations(ledger, LEDGER_TYPES, exact_keys=True)
        if not bad and ledger["schema"] != LEDGER_SCHEMA:
            bad.append(f"schema is {ledger['schema']!r}")
        if not bad and (not all(type(i) is str for i in ledger["identities"])
                        or len(set(ledger["identities"]))
                        != len(ledger["identities"])
                        or len(ledger["identities"])
                        != ledger["conceptual_variables"]):
            bad.append("identities are not a set of strings of the "
                       "declared size")
        if bad:
            raise VerificationRefusal(f"pre-result ledger schema: {bad}")
        if ledger["census_sha256"] != census["census_sha256"]:
            raise VerificationRefusal(
                "the pre-result ledger was built from a different census")
        ids = set(ledger["identities"])
        if ids != set(census["variables"]):
            raise VerificationRefusal(
                "the ledger identities are not the census variables")

        terminals = root.subdir("terminals")
        docs: dict[str, dict] = {}
        for name in sorted(terminals.files):
            rel_problem = {"file": name}
            try:
                art = terminals.read(name)
                doc = art.json()
            except CustodyRefusal as exc:
                divergences.append(rel_problem | {
                    "kind": "TERMINAL_UNREADABLE", "detail": str(exc)[:160]})
                continue
            outcome = doc.get("outcome")
            if outcome not in TERMINAL_TYPES:
                divergences.append(rel_problem | {
                    "kind": "TERMINAL_SCHEMA", "detail":
                        [f"unknown outcome {outcome!r}"]})
                continue
            bad = schema_violations(doc, TERMINAL_TYPES[outcome],
                                    exact_keys=True)
            if bad:
                divergences.append(rel_problem | {"kind": "TERMINAL_SCHEMA",
                                                  "detail": bad})
                continue
            vid = doc["variable_id"]
            if name != hashlib.sha256(vid.encode()).hexdigest()[:32] + \
                    ".json":
                divergences.append(rel_problem | {
                    "kind": "TERMINAL_MISFILED", "variable_id": vid})
                continue
            if vid in docs:
                divergences.append(rel_problem | {
                    "kind": "TERMINAL_DUPLICATE", "variable_id": vid})
                continue
            var = census["variables"].get(vid)
            if var is None:
                divergences.append(rel_problem | {
                    "kind": "TERMINAL_UNKNOWN_VARIABLE", "variable_id": vid})
                continue
            if (doc["entity"], doc["concept_name"]) != (var["entity"],
                                                        var["concept_name"]):
                divergences.append(rel_problem | {
                    "kind": "TERMINAL_CENSUS_IDENTITY_MISMATCH",
                    "variable_id": vid})
                continue
            docs[vid] = doc | {"_file": name, "_sha256": art.sha256,
                               "_leaf_binding": art.facts()["leaf_binding"]}
        missing = sorted(ids - set(docs))
        for vid in missing:
            divergences.append({"kind": "TERMINAL_MISSING",
                                "variable_id": vid})

        # C71: MEASURED binds its own appearance and exactly one file.
        groups: dict[str, list[str]] = {}
        for vid, doc in docs.items():
            if doc["outcome"] != MEASURED:
                continue
            app_id = doc["appearance"]
            if app_id not in census["variables"][vid]["appearances"]:
                divergences.append({
                    "kind": "APPEARANCE_NOT_OWNED_BY_VARIABLE",
                    "variable_id": vid, "appearance": app_id})
                doc["_unbound"] = True
                continue
            groups.setdefault(census["appearances"][app_id]["relative_path"],
                              []).append(vid)

        lake = LakeReader(lake_root)
        bound: dict[str, dict] = {}
        comparisons: dict[str, dict] = {}
        pub_rows = (published or {}).get("rows")
        for rel in sorted(groups):
            app = next(census["appearances"][docs[v]["appearance"]]
                       for v in groups[rel])
            try:
                art = lake.read(rel)
            except _SourceProblem as exc:
                for vid in groups[rel]:
                    docs[vid]["_unbound"] = True
                divergences.append({"kind": exc.kind, "source": rel,
                                    "variables_affected": len(groups[rel]),
                                    "detail": exc.detail[:160]})
                continue
            if art.sha256 != app["physical_sha256"] or \
                    art.size != app["size_bytes"]:
                for vid in groups[rel]:
                    docs[vid]["_unbound"] = True
                divergences.append({
                    "kind": "SOURCE_DIGEST_OR_SIZE_DIVERGES", "source": rel,
                    "variables_affected": len(groups[rel])})
                continue
            facts = {"logical_id": rel, "sha256": art.sha256,
                     "bytes": art.size,
                     "leaf_binding": art.facts()["leaf_binding"]}
            for vid in groups[rel]:
                bound[vid] = facts
            if pub_rows is None:
                continue
            for vid in groups[rel]:
                window, rows_in_file, why = column_window(
                    art.raw(), docs[vid]["concept_name"])
                if window is None:
                    comparisons[vid] = {"variable_id": vid,
                                        "recomputation": DIVERGES,
                                        "layer": SOURCE_BOUND,
                                        "problems": [{"kind":
                                                      "OUTCOME_DIVERGES",
                                                      "detail": why}],
                                        "descriptors": {}}
                    continue
                comparisons[vid] = compare_variable(
                    vid, MEASURED, pub_rows.get(vid, {}), window,
                    rows_in_file, facts, census["census_sha256"])
            del art

        if pub_rows is not None:
            extra_subjects = sorted(set(pub_rows) - ids)
            if extra_subjects:
                divergences.append({"kind": "PUBLISHED_UNKNOWN_SUBJECTS",
                                    "count": len(extra_subjects)})
            for vid, doc in docs.items():
                if doc["outcome"] != NOT_IDENTIFIABLE:
                    continue
                row = pub_rows.get(vid, {})
                ok = (set(row) == {"characterization_disposition"}
                      and row["characterization_disposition"]["value_text"]
                      == NOT_IDENTIFIABLE)
                comparisons[vid] = {
                    "variable_id": vid, "layer": PRODUCER_DECLARED,
                    "recomputation": AGREES if ok else DIVERGES,
                    "problems": [] if ok else [{
                        "kind": "ROW_CORRESPONDENCE_DIVERGES"}],
                    "axis_claim": ("TEMPORAL_AXIS_NAME"
                                   if doc["concept_name"]
                                   in TEMPORAL_AXIS_NAMES
                                   else "NOT_A_KNOWN_AXIS_NAME"),
                    "descriptors": {}}
            if published.get("duplicates"):
                divergences.append({"kind": "PUBLISHED_DUPLICATE_ROWS",
                                    "count": len(published["duplicates"])})

        population = POPULATION_DIVERGES if divergences else \
            POPULATION_VERIFIED
        states: dict[str, dict[str, int]] = {}
        for comp in comparisons.values():
            for d, r in comp["descriptors"].items():
                states.setdefault(d, {}).setdefault(r["state"], 0)
                states[d][r["state"]] += 1
        recomputation_diverged = [v for v, c in comparisons.items()
                                  if c["recomputation"] == DIVERGES]
        report = {
            "schema": REPORT_SCHEMA,
            "retired_verdict": "TERMINALS_VERIFIED_EXACT is no longer "
                               "emitted: it named a recomputation that "
                               "had not happened",
            "independence": "no producer module is imported; descriptor "
                            "formulas come from lake_descriptor_recompute",
            "authority": {"expected_census_sha256": expected_census_sha256,
                          "census_file_sha256": census["file_sha256"]},
            "custody": {"physical_paths": "WITHHELD",
                        "state_reads": len(custody.reads()),
                        "lake_reads": len(lake.custody.reads()),
                        "strict_mode": strict_mode},
            "population": {
                "verdict": population,
                "declared": len(ids),
                "terminals_read": len(docs),
                "producer_declared_outcomes": {
                    f"PRODUCER_DECLARED_{k}": sum(
                        1 for d in docs.values() if d["outcome"] == k)
                    for k in TERMINAL_TYPES},
                "sources_bound": len({f["logical_id"] for f in
                                      bound.values()}),
                "variables_source_bound": len(bound),
                "divergences": divergences,
                "divergence_count": len(divergences)},
            "recomputation": {
                "verdict": (RECOMPUTATION_NOT_RUN if pub_rows is None else
                            RECOMPUTED_DIVERGES if recomputation_diverged
                            else RECOMPUTED_AGREES),
                "published_attempt": (published or {}).get("attempt"),
                "variables_compared": len(comparisons),
                "variables_diverging": len(recomputation_diverged),
                "variables_by_layer": {
                    layer: sum(1 for c in comparisons.values()
                               if c["layer"] == layer)
                    for layer in (PRODUCER_DECLARED, SOURCE_BOUND,
                                  INDEPENDENTLY_RECOMPUTED)},
                "descriptor_states": states,
                "tolerance": {"counts": "exact", "reals_abs": R.ABS_TOL,
                              "reals_rel": R.REL_TOL},
                "specificity": {d: {"level": lv, "reason": why,
                                    "declared_reading":
                                        R.DECLARED_READINGS.get(d)}
                                for d, (lv, why) in R.SPECIFICITY.items()},
                "first_divergences": [comparisons[v] for v in
                                      recomputation_diverged[:5]]},
            "grants_nothing": "binding and recomputation are evidence a "
                              "later gate may consume; neither makes a "
                              "variable eligible",
        }
        report["verification_sha256"] = sha_obj(report)
        return report | {"_docs": docs, "_bound": bound,
                         "_comparisons": comparisons}
    finally:
        custody.close()
        if lake is not None:
            lake.close()


# ------------------------------------------------------------ C72 v3
def tree_content_digest(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(directory).iterdir()):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def supersede_v3(state_dir: Path, report: dict, *, superseded_at: str,
                 code_identity: dict, v3_dirname: str = "terminals_v3"
                 ) -> dict:
    if report["recomputation"]["verdict"] == RECOMPUTATION_NOT_RUN:
        raise VerificationRefusal("v3 records a recomputation; none ran")
    v3 = state_dir / v3_dirname
    if v3.exists():
        raise VerificationRefusal(f"{v3_dirname} exists; written once")
    guarded = {d: tree_content_digest(state_dir / d)
               for d in ("terminals", "terminals_v2")
               if (state_dir / d).is_dir()}
    v3.mkdir()
    written = 0
    for vid, doc in sorted(report["_docs"].items()):
        comp = report["_comparisons"].get(vid)
        src = report["_bound"].get(vid)
        body = {
            "schema": V3_SCHEMA,
            "supersedes": {"v1_file": doc["_file"],
                           "v1_sha256": doc["_sha256"],
                           "v1_and_v2": "BYTE_INTACT"},
            "variable_id": vid,
            "producer_declared_outcome": f"PRODUCER_DECLARED_{doc['outcome']}",
            "layer": (comp or {}).get("layer", PRODUCER_DECLARED),
            "recomputation": (comp or {}).get("recomputation",
                                              "NOT_COMPARED"),
            "source": src or "NOT_BOUND",
            "descriptors": (comp or {}).get("descriptors", {}),
            "problems": (comp or {}).get("problems", []),
            "population_verdict": report["population"]["verdict"],
            "verification_sha256": report["verification_sha256"],
            "code_identity": code_identity,
            "superseded_at": superseded_at,
            "grants_nothing": "evidence only; no eligibility",
        }
        body["terminal_sha256"] = sha_obj(body)
        (v3 / doc["_file"]).write_text(json.dumps(body, indent=1,
                                                  sort_keys=True) + "\n")
        written += 1
    after = {d: tree_content_digest(state_dir / d) for d in guarded}
    if after != guarded:
        raise VerificationRefusal("v1 or v2 changed during the v3 write")
    index = {"schema": "crispdm.lake_terminal_supersession.v2",
             "superseded_at": superseded_at, "v3_directory": v3_dirname,
             "v3_terminals_written": written,
             "guarded_content_digests": guarded,
             "verification_sha256": report["verification_sha256"],
             "population_verdict": report["population"]["verdict"],
             "recomputation_verdict": report["recomputation"]["verdict"]}
    index["supersession_sha256"] = sha_obj(index)
    (state_dir / "TERMINAL_SUPERSESSION.v2.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--lake-root", required=True, type=Path)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--expected-census-sha256", required=True)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--attempt", default="crispdm-c49-2026-09-12")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--supersede-v3", action="store_true")
    ap.add_argument("--superseded-at", default=None)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)
    published = (load_published_from_olap(a.dsn, a.attempt)
                 if a.dsn else None)
    report = verify(a.state_dir.expanduser(), a.lake_root.expanduser(),
                    a.census.expanduser(),
                    expected_census_sha256=a.expected_census_sha256,
                    published=published, strict_mode=a.strict)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(public, indent=1, sort_keys=True,
                                       default=str) + "\n")
    out = {"population": public["population"]["verdict"],
           "divergences": public["population"]["divergence_count"],
           "recomputation": public["recomputation"]["verdict"],
           "layers": public["recomputation"]["variables_by_layer"],
           "variables_diverging":
               public["recomputation"]["variables_diverging"]}
    if a.supersede_v3:
        me = Path(__file__)
        idx = supersede_v3(
            a.state_dir.expanduser(), report,
            superseded_at=a.superseded_at or "UNDECLARED",
            code_identity={
                "verifier_sha256": sha_bytes(me.read_bytes()),
                "recompute_sha256": sha_bytes(
                    (me.parent / "lake_descriptor_recompute.py").read_bytes()),
                "custody_sha256": sha_bytes(
                    (me.parent / "descriptor_custody.py").read_bytes())})
        out["v3_written"] = idx["v3_terminals_written"]
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
