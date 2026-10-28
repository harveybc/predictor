#!/usr/bin/env python3
"""Phase-4 candidate consolidation (plan section 1 and 4; acceptance FS4-01, FS4-02).

Consumes ``CANDIDATES_FOR_VALIDATION.json`` with its ``PHASE_3_FILTER_COMPLETE.json`` (and the
phase-2 ``ALIAS_GROUPS.json`` beside them when present) and produces one consolidated candidate
list per population:

* phase-2 exact aliases are replaced by the declared representative before anything else;
* candidates with identical member sets for the same target are merged into ONE set that keeps
  every contributing method (so the ALL_ADMISSIBLE, UNIVARIATE_MI, CAUSAL_SUPPORTED and RANDOM_K
  controls survive as method tags, never as duplicate trainings);
* every consolidated set is bound to population, phase-1 identity, target, horizon, K, the
  ORDERED (sorted) member list, the phase-3 closure digest and the phase-3 unit;
* member order in the source never changes an identity (FS4-02);
* the source denominator is preserved: counts per target before and after merging.

Nothing here reads VALIDATION or TEST. This is pure python (json + hashlib).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = "fs4.consolidated_candidates.v1"
CONTROL_METHODS = ("ALL_ADMISSIBLE", "UNIVARIATE_MI", "CAUSAL_SUPPORTED", "RANDOM_K")
FILTER_METHODS = ("SPEARMAN_CLUSTER", "MRMR", "JMI", "MRMR_CAUSAL", "JMI_CAUSAL")
SOURCE_SCHEMA = "fs_phase23.candidates_for_validation.v1"


class Refusal(ValueError):
    """A mismatch that must not become an accepted candidate."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def set_identity(population_id: str, identity: str, target_id: str, members) -> str:
    """Identity of a consolidated set: population, phase-1 identity, target and SORTED members."""
    return digest({"population_id": population_id, "identity": identity, "target_id": target_id,
                   "members": sorted(members)})


def load_alias_map(path: Path | None) -> dict[str, str]:
    """member -> representative from phase-2 ALIAS_GROUPS.json (only ALIAS_GROUP dispositions)."""
    if path is None or not Path(path).is_file():
        return {}
    groups = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    for g in groups:
        if g.get("disposition") != "ALIAS_GROUP" or not g.get("representative"):
            continue
        rep = g["representative"]
        for m in g.get("members", []):
            if m != rep:
                if m in out and out[m] != rep:
                    raise Refusal(f"ALIAS_CONFLICT: {m} -> {out[m]} and {rep}")
                out[m] = rep
    return out


def _check_closure(source: dict, closure_path: Path) -> dict | None:
    if closure_path.is_file():
        closure = json.loads(closure_path.read_text())
        if closure.get("state") != "PHASE_3_FILTER_COMPLETE" or closure.get("closure_sha256") != source.get("phase3_closure_sha256"):
            raise Refusal("PHASE3_CLOSURE_MISMATCH")
        if closure.get("population_id") != source.get("population_id") or closure.get("identity") != source.get("identity"):
            raise Refusal("PHASE3_CLOSURE_POPULATION_OR_IDENTITY_MISMATCH")
        return closure
    if source.get("phase3_closure_sha256"):
        raise Refusal("PHASE3_CLOSURE_MISSING")
    return None


def consolidate(candidates_path, closure_path=None, alias_groups_path=None) -> dict:
    candidates_path = Path(candidates_path)
    source = json.loads(candidates_path.read_text())
    closure_path = Path(closure_path) if closure_path else candidates_path.with_name("PHASE_3_FILTER_COMPLETE.json")
    alias_groups_path = Path(alias_groups_path) if alias_groups_path else candidates_path.with_name("ALIAS_GROUPS.json")
    closure = _check_closure(source, closure_path)
    population = source.get("population_id")
    identity = source.get("identity")
    if not population or not identity or source.get("final_selection") is not False or source.get("uses_test_split") is True:
        raise Refusal("CANDIDATE_IDENTITY_OR_STATE_INVALID")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise Refusal("EMPTY_CANDIDATES")
    aliases = load_alias_map(alias_groups_path)
    units = {u["target_id"]: u["unit_id"] for u in (closure or {}).get("units", [])}

    merged: dict[tuple, dict] = {}
    alias_replacements = 0
    per_target_source: dict[str, int] = {}
    for c in candidates:
        if c.get("population_id", population) != population or c.get("identity", identity) != identity:
            raise Refusal("MIXED_POPULATION_OR_IDENTITY")
        members = c.get("members")
        if not isinstance(members, list) or not members or len(members) != len(set(members)):
            raise Refusal("INVALID_SUBSET")
        target = c.get("target_id")
        if not target or not c.get("subset_sha256") or not c.get("method"):
            raise Refusal("CANDIDATE_MISSING_TARGET_OR_DIGEST")
        if c.get("is_final_selection") is True:
            raise Refusal("CANDIDATE_IDENTITY_OR_STATE_INVALID")
        horizon = c.get("horizon_hours")
        if type(horizon) is not int or horizon < 1:
            raise Refusal(f"CANDIDATE_HORIZON_INVALID: {target}")
        declared_k = c.get("k")
        if type(declared_k) is not int or declared_k != len(members):
            raise Refusal(f"CANDIDATE_K_MISMATCH: {target} {c.get('method')}")
        per_target_source[target] = per_target_source.get(target, 0) + 1
        resolved = []
        for m in members:
            rep = aliases.get(m, m)
            if rep != m:
                alias_replacements += 1
            resolved.append(rep)
        resolved = sorted(set(resolved))
        key = (target, tuple(resolved))
        entry = merged.get(key)
        if entry is None:
            entry = merged[key] = {
                "set_id": set_identity(population, identity, target, resolved),
                "population_id": population, "identity": identity, "target_id": target, "horizon_hours": horizon,
                "k": len(resolved), "declared_k": declared_k, "members": resolved, "n_features": len(resolved),
                "methods": [], "control_methods": [], "source_count": 0, "source_subset_sha256": [],
                "phase3_closure_sha256": source.get("phase3_closure_sha256"), "phase3_unit_id": units.get(target),
            }
        elif entry["horizon_hours"] != horizon:
            raise Refusal(f"HORIZON_CONFLICT_FOR_TARGET: {target}")
        entry["methods"] = sorted(set(entry["methods"]) | {c["method"]})
        entry["control_methods"] = sorted(set(entry["methods"]) & set(CONTROL_METHODS))
        entry["source_count"] += 1
        entry["source_subset_sha256"] = sorted(set(entry["source_subset_sha256"]) | {c["subset_sha256"]})
        entry["declared_k"] = max(entry["declared_k"], declared_k)
    sets = sorted(merged.values(), key=lambda s: (s["target_id"], s["set_id"]))
    per_target = {t: {"source": n, "consolidated": sum(1 for s in sets if s["target_id"] == t)}
                  for t, n in sorted(per_target_source.items())}
    body = {
        "schema": SCHEMA, "population_id": population, "identity": identity,
        "source_schema": source.get("schema"), "source_path": str(candidates_path),
        "source_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "phase3_closure_sha256": source.get("phase3_closure_sha256"),
        "alias_groups_sha256": hashlib.sha256(alias_groups_path.read_bytes()).hexdigest() if alias_groups_path.is_file() else None,
        "source_candidate_count": len(candidates), "consolidated_count": len(sets),
        "merged_duplicates": len(candidates) - len(sets), "alias_replacements": alias_replacements,
        "unique_features": len({m for s in sets for m in s["members"]}),
        "per_target": per_target, "control_methods": list(CONTROL_METHODS), "filter_methods": list(FILTER_METHODS),
        "final_selection": False, "uses_validation_split": False, "uses_test_split": False,
        "sets": sets,
    }
    body["consolidated_sha256"] = digest(sets)
    return body


def write_atomic(path: Path, body: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
    os.replace(tmp, path)


def consolidate_to_file(candidates_path, out_path, closure_path=None, alias_groups_path=None) -> dict:
    body = consolidate(candidates_path, closure_path, alias_groups_path)
    out_path = Path(out_path)
    if out_path.is_file():
        prior = json.loads(out_path.read_text())
        if prior.get("consolidated_sha256") == body["consolidated_sha256"]:
            return prior
    write_atomic(out_path, body)
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", required=True, help="CANDIDATES_FOR_VALIDATION.json of one population")
    ap.add_argument("--out", required=True, help="CONSOLIDATED_CANDIDATES.json to write (atomic, idempotent)")
    ap.add_argument("--closure", help="PHASE_3_FILTER_COMPLETE.json (default: beside the candidates)")
    ap.add_argument("--alias-groups", help="ALIAS_GROUPS.json (default: beside the candidates)")
    a = ap.parse_args(argv)
    try:
        body = consolidate_to_file(a.candidates, a.out, a.closure, a.alias_groups)
    except (Refusal, OSError, json.JSONDecodeError) as exc:
        print(canonical({"error": str(exc)}), file=sys.stderr)
        return 2
    print(canonical({k: body[k] for k in ("population_id", "source_candidate_count", "consolidated_count", "merged_duplicates",
                                           "alias_replacements", "unique_features", "per_target", "consolidated_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
