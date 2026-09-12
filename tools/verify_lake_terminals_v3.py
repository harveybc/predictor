#!/usr/bin/env python3
"""C90-C92 (order 2026-09-12): terminal verification v3.

The v2 verifier (kept unchanged as history) had three gaps the audit
reproduced: it compared the expected census digest with the census's own
declared text and filename and never recomputed it, so an appearance
re-pointed to another parquet verified; it parsed JSON leniently, so a
duplicated key verified with the last value winning; and it measured an
int64 datetime sentinel as a finite number.

v3:
  C90  recomputes the census content digest exactly as the producer
       defines it — sha256 of json.dumps of the document without
       census_sha256, keys sorted — and requires it to equal the
       reviewer-supplied expectation, the declared digest and the
       filename; records the raw byte digest separately; parses every
       JSON strictly through custody; and applies exact key-and-type
       schemas to the census top level, appearances, variables, lineage,
       the pre-ledger and both terminal shapes. Nested content of
       unconsumed top-level census objects is checked by type at the top
       level only, and the report says so;
  C91  derives a semantic contract per column before any statistic, and
       recomputes only NUMERIC_MEASURABLE columns; covers all 1,965
       variables, reading every appearance of the axis variables whose
       terminals declare none;
  C92  layers: PRODUCER_DECLARED, SOURCE_BOUND, PHYSICALLY_TYPED,
       INDEPENDENTLY_RECOMPUTED, SEMANTICALLY_UNRESOLVED, DIVERGES; writes
       terminals_v4 beside v1-v3, whose content digests are taken through
       the corrected custody before and after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptor_custody import Custody, CustodyRefusal  # noqa: E402
import lake_descriptor_recompute as R  # noqa: E402
import lake_semantic_contract as S  # noqa: E402
from verify_lake_terminals import (LakeReader, _SourceProblem,  # noqa: E402
                                   schema_violations, sha_obj, sha_bytes,
                                   load_published_from_olap)

REPORT_SCHEMA = "crispdm.lake_terminal_verification.v3"
V4_SCHEMA = "crispdm.lake_characterization_terminal.v4"
CENSUS_SCHEMA = "financial_data.incremental_census.v1"
LEDGER_SCHEMA = "crispdm.lake_characterization_pre_ledger.v1"

POP_VERIFIED = "TERMINAL_POPULATION_AND_CANONICAL_CENSUS_VERIFIED"
POP_DIVERGES = "TERMINAL_POPULATION_OR_CANONICAL_CENSUS_DIVERGES"

PRODUCER_DECLARED = "PRODUCER_DECLARED"
SOURCE_BOUND = "SOURCE_BOUND"
PHYSICALLY_TYPED = "PHYSICALLY_TYPED"
INDEPENDENTLY_RECOMPUTED = "INDEPENDENTLY_RECOMPUTED"
SEMANTICALLY_UNRESOLVED = "SEMANTICALLY_UNRESOLVED"
DIVERGES = "DIVERGES"
LAYERS = (PRODUCER_DECLARED, SOURCE_BOUND, PHYSICALLY_TYPED,
          INDEPENDENTLY_RECOMPUTED, SEMANTICALLY_UNRESOLVED, DIVERGES)

HEX64 = re.compile(r"[0-9a-f]{64}")
CENSUS_TOP = {"appearances": list, "availability_contract": dict,
              "census_sha256": str, "censused_at": str, "conflicts": list,
              "coverage": dict, "delta": dict, "dictionary_coverage": dict,
              "equivalence_classes": list, "external_full_profiles": list,
              "gaps": dict, "manifest_generated_at": str,
              "manifest_stage": str, "policy": dict, "schema": str,
              "value_profiles_sampled": list, "variables": list}
APPEARANCE = {"appearance_id": str, "bytes_read_for_digest": int,
              "ctime_ns": int, "declared_columns": list,
              "declared_rows": int, "digest_state": str, "entity": str,
              "frequency": str, "manifest_status": str, "mtime_ns": int,
              "period_end": str, "period_start": str,
              "physical_sha256": str, "presence": str,
              "profile_depth": str, "provenance": (dict, str),
              "relative_path": str, "size_bytes": int,
              "source_class": str}
VARIABLE = {"appearance_count": int, "appearances": list,
            "availability_contract": str, "available_time": str,
            "concept_name": str, "entity": str, "event_time": str,
            "frequencies": list, "license": str, "lineage": dict,
            "physical_type": str, "profile_depth": str, "role": str,
            "semantics": str, "semantics_source": str,
            "source_class": str, "unit": str, "variable_id": str}
LINEAGE_REQUIRED = {"provenance_files", "source_declarations",
                    "upstream_join_rule", "upstream_source_dirs"}
LINEAGE_OPTIONAL = {"ambiguous_provenance_candidates"}
LEDGER = {"census_sha256": str, "censused_at": str,
          "conceptual_variables": int, "identities": list,
          "physical_appearances": int, "pre_ledger_sha256": str,
          "rule": str, "schema": str, "written_at": str}
TERMINAL = {
    "MEASURED": {"appearance": str, "batch": str, "concept_name": str,
                 "descriptors": int, "entity": str, "measured_at": str,
                 "not_identifiable": int, "outcome": str,
                 "rows_used": int, "variable_id": str},
    "NOT_IDENTIFIABLE": {"batch": str, "concept_name": str, "entity": str,
                         "measured_at": str, "outcome": str,
                         "reason": str, "variable_id": str},
}
BASE = frozenset({"characterization_disposition", "n_observations",
                  "missing_count", "missingness_fraction",
                  "non_finite_count", "duplicate_count"})
FULL = BASE | frozenset(R.SPECIFICITY) - {"insufficient_finite_observations"}
INSUFFICIENT = BASE | {"insufficient_finite_observations"}


class VerificationRefusal(SystemExit):
    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.reason = msg


def canonical_census_sha256(doc: dict) -> str:
    """The producer's _self_sha, reimplemented from its definition."""
    body = {k: doc[k] for k in sorted(doc) if k != "census_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()
                          ).hexdigest()


def load_census(census_path: Path, expected: str) -> dict:
    if type(expected) is not str or not HEX64.fullmatch(expected):
        raise VerificationRefusal("EXPECTATION_REQUIRED",
                                  "a reviewer-supplied census digest "
                                  "(64 lowercase hex) is required")
    c = Custody(census_path.parent, require_owner=True)
    try:
        art = c.root_snapshot().read(census_path.name)
        doc = art.json()
        read_facts = art.facts()
    except CustodyRefusal as exc:
        raise VerificationRefusal("CENSUS_UNREADABLE", str(exc))
    finally:
        c.close()
    bad = schema_violations(doc, CENSUS_TOP, exact_keys=True)
    if not bad and doc["schema"] != CENSUS_SCHEMA:
        bad.append(f"schema is {doc['schema']!r}")
    if bad:
        raise VerificationRefusal("CENSUS_SCHEMA", f"top level: {bad}")
    recomputed = canonical_census_sha256(doc)
    checks = {"reviewer_supplied_expectation": expected,
              "declared_in_document": doc["census_sha256"],
              "recomputed_canonical": recomputed,
              "filename_address": census_path.name[len("census-"):-len(".json")]
              if census_path.name.startswith("census-") else None}
    if len(set(checks.values())) != 1:
        raise VerificationRefusal(
            "CENSUS_DIGEST_MISMATCH",
            f"the census content does not hash to its expected identity: "
            f"{ {k: str(v)[:12] for k, v in checks.items()} }")
    problems, apps, variables = [], {}, {}
    for a in doc["appearances"]:
        b = schema_violations(a, APPEARANCE, exact_keys=True)
        if not b and not HEX64.fullmatch(a["physical_sha256"]):
            b.append("physical_sha256 not hex64")
        if not b and not all(type(x) is str for x in a["declared_columns"]):
            b.append("declared_columns not strings")
        if b or a.get("appearance_id") in apps:
            problems.append((a.get("appearance_id"), b or ["duplicate"]))
            continue
        apps[a["appearance_id"]] = a
    for v in doc["variables"]:
        b = schema_violations(v, VARIABLE, exact_keys=True)
        if not b:
            lk = set(v["lineage"])
            if not LINEAGE_REQUIRED <= lk or lk - LINEAGE_REQUIRED - LINEAGE_OPTIONAL:
                b.append(f"lineage keys {sorted(lk)}")
            if v["appearance_count"] != len(v["appearances"]) or \
                    len(set(v["appearances"])) != len(v["appearances"]) or \
                    any(x not in apps for x in v["appearances"]):
                b.append("appearances inconsistent")
        if b or v.get("variable_id") in variables:
            problems.append((v.get("variable_id"), b or ["duplicate"]))
            continue
        variables[v["variable_id"]] = v
    if problems:
        raise VerificationRefusal("CENSUS_SCHEMA",
                                  f"{len(problems)} objects: {problems[:3]}")
    return {"doc": doc, "appearances": apps, "variables": variables,
            "identity": dict(checks, raw_file_sha256=read_facts["sha256"],
                             expectation_role="REVIEWER_SUPPLIED_EXPECTATION")}


def custody_content_digest(state_dir: Path, sub: str) -> str | None:
    c = Custody(state_dir, require_owner=True)
    try:
        root = c.root_snapshot()
        if sub not in root.dirs:
            return None
        snap = root.subdir(sub)
        h = hashlib.sha256()
        for name in sorted(snap.files):
            h.update(name.encode())
            h.update(bytes.fromhex(snap.read(name).sha256))
        return h.hexdigest()
    finally:
        c.close()


def _window(table, column):
    win = R.window_rows(len(table))
    return win


def verify(state_dir: Path, lake_root: Path, census_path: Path, *,
           expected_census_sha256: str, published: dict | None = None
           ) -> dict:
    census = load_census(census_path, expected_census_sha256)
    div: list[dict] = []
    c = Custody(state_dir, require_owner=True)
    lake = None
    try:
        root = c.root_snapshot()
        ledger = root.read("PRE_LEDGER.json").json()
        bad = schema_violations(ledger, LEDGER, exact_keys=True)
        if not bad and (ledger["schema"] != LEDGER_SCHEMA
                        or ledger["census_sha256"] != census["identity"]["recomputed_canonical"]
                        or sorted(ledger["identities"]) != sorted(census["variables"])
                        or len(ledger["identities"]) != ledger["conceptual_variables"]):
            bad.append("ledger does not describe the canonical census population")
        if bad:
            raise VerificationRefusal("LEDGER_SCHEMA", str(bad))
        ids = set(ledger["identities"])
        term = root.subdir("terminals")
        docs = {}
        for name in sorted(term.files):
            try:
                d = term.read(name).json()
            except CustodyRefusal as exc:
                div.append({"kind": "TERMINAL_UNREADABLE_OR_LENIENT_JSON",
                            "file": name, "detail": str(exc)[:160]})
                continue
            o = d.get("outcome")
            b = (["unknown outcome"] if o not in TERMINAL else
                 schema_violations(d, TERMINAL[o], exact_keys=True))
            vid = d.get("variable_id")
            if not b and name != hashlib.sha256(vid.encode()).hexdigest()[:32] + ".json":
                b = ["misfiled"]
            if not b and (vid not in census["variables"] or vid in docs):
                b = ["unknown or duplicate variable"]
            if not b and (d["entity"], d["concept_name"]) != (
                    census["variables"][vid]["entity"],
                    census["variables"][vid]["concept_name"]):
                b = ["identity differs from census"]
            if b:
                div.append({"kind": "TERMINAL_SCHEMA", "file": name,
                            "detail": b})
                continue
            docs[vid] = d
        for vid in sorted(ids - set(docs)):
            div.append({"kind": "TERMINAL_MISSING", "variable_id": vid})

        # which files to read: MEASURED -> their declared appearance;
        # NOT_IDENTIFIABLE -> every appearance of the variable (sweep)
        wanted: dict[str, list[tuple[str, str]]] = {}
        for vid, d in docs.items():
            var = census["variables"][vid]
            if d["outcome"] == "MEASURED":
                if d["appearance"] not in var["appearances"]:
                    div.append({"kind": "APPEARANCE_NOT_OWNED_BY_VARIABLE",
                                "variable_id": vid})
                    continue
                apps_for = [d["appearance"]]
            else:
                apps_for = list(var["appearances"])
            for app_id in apps_for:
                rel = census["appearances"][app_id]["relative_path"]
                wanted.setdefault(rel, []).append((vid, app_id))

        lake = LakeReader(lake_root)
        pub = (published or {}).get("rows")
        per_var: dict[str, dict] = {vid: {"variable_id": vid,
                                          "producer_declared_outcome":
                                              f"PRODUCER_DECLARED_{d['outcome']}",
                                          "appearances": {}}
                                    for vid, d in docs.items()}
        sweep = {"columns_evaluated": 0, "by_state": {}, "cases": []}
        import pyarrow as pa
        import pyarrow.parquet as pq
        for rel in sorted(wanted):
            app0 = census["appearances"][wanted[rel][0][1]]
            try:
                art = lake.read(rel)
            except _SourceProblem as exc:
                div.append({"kind": exc.kind, "source": rel,
                            "variables_affected": len(wanted[rel])})
                continue
            if art.sha256 != app0["physical_sha256"] or art.size != app0["size_bytes"]:
                div.append({"kind": "SOURCE_DIGEST_OR_SIZE_DIVERGES",
                            "source": rel})
                continue
            table = pq.read_table(pa.BufferReader(art.raw()))
            win = R.window_rows(len(table))
            src = {"logical_id": rel, "sha256": art.sha256, "bytes": art.size,
                   "leaf_binding": art.facts()["leaf_binding"]}
            for vid, app_id in wanted[rel]:
                col = docs[vid]["concept_name"]
                entry = {"source": src, "window": win}
                if table.column_names.count(col) != 1:
                    entry["semantic"] = {"state": S.SEMANTIC_TYPE_UNRESOLVED,
                                         "reasons": ["column absent or ambiguous"]}
                else:
                    entry["semantic"] = S.column_contract(
                        table.column(col), col, census["variables"][vid],
                        win["rows_used"])
                st = entry["semantic"]["state"]
                sweep["columns_evaluated"] += 1
                sweep["by_state"][st] = sweep["by_state"].get(st, 0) + 1
                sem = entry["semantic"]
                if st != S.NUMERIC_MEASURABLE or sem.get("sentinel_candidate_counts"):
                    sweep["cases"].append({"variable_id": vid, "column": col,
                                           "source": rel, "state": st,
                                           "arrow_type": sem.get("arrow_type"),
                                           "sentinels": sem.get("sentinel_candidate_counts"),
                                           "reasons": sem.get("reasons")})
                if docs[vid]["outcome"] == "MEASURED" and st == S.NUMERIC_MEASURABLE \
                        and pub is not None:
                    import numpy as np
                    arr = table.column(col).slice(0, win["rows_used"])
                    window = np.asarray(arr.to_numpy(zero_copy_only=False),
                                        dtype=np.float64)
                    entry["recomputation"] = _compare(
                        pub.get(vid, {}), window, win, art.sha256,
                        census["identity"]["recomputed_canonical"])
                per_var[vid]["appearances"][app_id] = entry
            del table, art

        for vid, pv in per_var.items():
            d = docs[vid]
            if d["outcome"] != "MEASURED":
                pv["layer"] = PRODUCER_DECLARED
                states = {e["semantic"]["state"] for e in pv["appearances"].values()}
                pv["axis_semantics"] = sorted(states)
                continue
            e = pv["appearances"].get(d["appearance"])
            if e is None:
                pv["layer"] = PRODUCER_DECLARED
                continue
            st = e["semantic"]["state"]
            if st != S.NUMERIC_MEASURABLE:
                pv["layer"] = SEMANTICALLY_UNRESOLVED
                if pub is not None and pub.get(vid):
                    numeric_pub = sorted(k for k, r in pub[vid].items()
                                         if r.get("value") is not None)
                    pv["published_numeric_descriptors_withdrawn"] = numeric_pub
            elif "recomputation" not in e:
                pv["layer"] = PHYSICALLY_TYPED
            elif e["recomputation"]["diverged"]:
                pv["layer"] = DIVERGES
            else:
                pv["layer"] = INDEPENDENTLY_RECOMPUTED

        layers = {k: 0 for k in LAYERS}
        for pv in per_var.values():
            layers[pv["layer"]] += 1
        report = {
            "schema": REPORT_SCHEMA,
            "census_identity": census["identity"],
            "schemas": {
                "exact": ["census top level", "appearance", "variable",
                          "lineage keys", "pre-ledger", "terminal MEASURED",
                          "terminal NOT_IDENTIFIABLE"],
                "declared_limit": "nested content of unconsumed census "
                                  "top-level objects is checked by type at "
                                  "the top level only"},
            "json": "strict: duplicate keys and non-finite constants refused",
            "population": {"verdict": POP_DIVERGES if div else POP_VERIFIED,
                           "declared": len(ids), "terminals_read": len(docs),
                           "divergences": div, "divergence_count": len(div)},
            "semantic_sweep": sweep,
            "layers": layers,
            "recomputation_run": pub is not None,
            "grants_nothing": "evidence only; no eligibility",
        }
        report["verification_sha256"] = sha_obj(report)
        return report | {"_per_var": per_var}
    finally:
        c.close()
        if lake is not None:
            lake.close()


def _compare(published: dict, window, win, source_sha, census_sha) -> dict:
    rec = R.recompute(window)
    non_finite = int(rec["non_finite_count"][0])
    expected = INSUFFICIENT if "insufficient_finite_observations" in rec else FULL
    rows, problems = {}, []
    if set(published) != expected:
        problems.append({"kind": "ROW_CORRESPONDENCE",
                         "missing": sorted(expected - set(published)),
                         "unexpected": sorted(set(published) - expected)})
    for d in sorted(set(published) & expected):
        p = published[d]
        wc = p.get("window_contract") or {}
        for k in ("rows_total", "rows_used", "capped", "row_cap"):
            if k in wc and wc[k] != win[k]:
                problems.append({"kind": "WINDOW", "descriptor": d, "field": k})
        want_src = census_sha if d == "characterization_disposition" else source_sha
        if p.get("source_sha256") != want_src:
            problems.append({"kind": "PUBLISHED_SOURCE_DIGEST", "descriptor": d})
        if d == "characterization_disposition":
            ok = p.get("value_text") == "MEASURED"
            rows[d] = {"state": "AGREES" if ok else "DIVERGES",
                       "specificity": R.FULLY}
            continue
        level, why = R.specificity(d, non_finite)
        value, ident = rec.get(d, (None, None))
        agrees = (d in rec and bool(p["identifiable"]) == bool(ident)
                  and R.agree(d, p["value"], value if ident else None))
        if level == R.FULLY:
            state = "AGREES" if agrees else "DIVERGES"
        else:
            state = "NOT_INDEPENDENTLY_VERIFIABLE"
        rows[d] = {"state": state, "specificity": level, "reason": why}
    diverged = bool(problems) or any(r["state"] == "DIVERGES" for r in rows.values())
    return {"descriptors": rows, "problems": problems, "diverged": diverged}


def supersede_v4(state_dir: Path, report: dict, *, superseded_at: str,
                 code_identity: dict) -> dict:
    guarded = {s: custody_content_digest(state_dir, s)
               for s in ("terminals", "terminals_v2", "terminals_v3")}
    v4 = state_dir / "terminals_v4"
    if v4.exists():
        raise VerificationRefusal("WRITE_ONCE", "terminals_v4 exists")
    v4.mkdir()
    for vid, pv in sorted(report["_per_var"].items()):
        body = {"schema": V4_SCHEMA, "variable_id": vid,
                "producer_declared_outcome": pv["producer_declared_outcome"],
                "layer": pv["layer"], "appearances": pv["appearances"],
                "published_numeric_descriptors_withdrawn":
                    pv.get("published_numeric_descriptors_withdrawn", []),
                "axis_semantics": pv.get("axis_semantics"),
                "population_verdict": report["population"]["verdict"],
                "verification_sha256": report["verification_sha256"],
                "code_identity": code_identity,
                "superseded_at": superseded_at,
                "supersedes": ["terminals", "terminals_v2", "terminals_v3"],
                "grants_nothing": "evidence only"}
        body["terminal_sha256"] = sha_obj(body)
        name = hashlib.sha256(vid.encode()).hexdigest()[:32] + ".json"
        (v4 / name).write_text(json.dumps(body, indent=1, sort_keys=True,
                                          default=str) + "\n")
    after = {s: custody_content_digest(state_dir, s) for s in guarded}
    if after != guarded:
        raise VerificationRefusal("HISTORY_MOVED", "v1-v3 changed during write")
    idx = {"schema": "crispdm.lake_terminal_supersession.v3",
           "superseded_at": superseded_at, "v4_written": len(report["_per_var"]),
           "guarded_content_digests_via_custody": guarded,
           "verification_sha256": report["verification_sha256"],
           "layers": report["layers"]}
    idx["supersession_sha256"] = sha_obj(idx)
    (state_dir / "TERMINAL_SUPERSESSION.v3.json").write_text(
        json.dumps(idx, indent=1, sort_keys=True) + "\n")
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--lake-root", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--expected-census-sha256", required=True)
    ap.add_argument("--dsn")
    ap.add_argument("--attempt", default="crispdm-c49-2026-09-12")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--supersede-v4", action="store_true")
    ap.add_argument("--superseded-at", default="UNDECLARED")
    a = ap.parse_args(argv)
    pub = load_published_from_olap(a.dsn, a.attempt) if a.dsn else None
    r = verify(a.state_dir.expanduser(), a.lake_root.expanduser(),
               a.census.expanduser(), expected_census_sha256=a.expected_census_sha256,
               published=pub)
    public = {k: v for k, v in r.items() if not k.startswith("_")}
    if a.report:
        a.report.write_text(json.dumps(public, indent=1, sort_keys=True, default=str) + "\n")
    out = {"population": public["population"]["verdict"],
           "divergences": public["population"]["divergence_count"],
           "layers": public["layers"], "sweep": public["semantic_sweep"]["by_state"]}
    if a.supersede_v4:
        me = Path(__file__).parent
        idx = supersede_v4(a.state_dir.expanduser(), r, superseded_at=a.superseded_at,
                           code_identity={f: sha_bytes((me / f).read_bytes()) for f in (
                               "verify_lake_terminals_v3.py", "lake_semantic_contract.py",
                               "lake_descriptor_recompute.py", "descriptor_custody.py",
                               "verify_lake_terminals.py")})
        out["v4_written"] = idx["v4_written"]
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
