#!/usr/bin/env python3
"""Fail-closed selected-feature manifest gate (lane D, order 2026-10-03).

Every modular / DOIN campaign builder entry point must call
``require_selected_manifest`` (or ``require_campaign_inputs`` for a campaign
or input-binding dict) before it materializes, trains or dispatches anything.
The gate admits a consumed feature set only when ALL of the following hold:

1. one manifest of schema ``feature_selection.manifest.v1`` exists and loads;
2. it declares ``dataset_id``, at least one target with positive horizons,
   a ``producer`` role, an ``independent_decision_record`` reference and a
   non-empty ``source_digests`` map of sha256 values;
3. every feature carries a state in {selected, rejected, pending} and at least
   one reason code; ``admissible``, ``profiled`` or any other state refuses
   the WHOLE manifest (a mechanical inventory is not a selection);
4. at least one feature is ``selected`` (an all-pending manifest refuses);
5. ``manifest_sha256`` re-derives (canonical JSON without that field);
6. a separate decision record of schema ``feature_selection.decision.v1``
   is supplied, re-derives its own ``record_sha256``, binds this manifest's
   ``manifest_sha256``, matches the manifest's decision reference, carries the
   verdict ``ACCEPT_SELECTED_SET``, and was written by a role different from
   the producer role (a producer-self-certified manifest refuses);
7. when source files are supplied, each re-hashes to its declared digest;
8. every consumed feature is present and ``selected``; the consumed set is
   non-empty.

Anything else refuses, and a refused manifest refuses EVERY consumed feature
(``refused_features`` lists them). There is no override flag.

The phrase ``FINAL_ADMISSIBLE_DECLARATION_BOUND`` means "mechanically
admissible", never ``FEATURE_SELECTION_COMPLETE``; a binding carrying it
without a passing selection manifest refuses.

Stdlib only (no TensorFlow, no numpy) so it can guard any entry point cheaply.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MANIFEST_SCHEMA = "feature_selection.manifest.v1"
DECISION_SCHEMA = "feature_selection.decision.v1"
ACCEPT_VERDICT = "ACCEPT_SELECTED_SET"
STATES = ("selected", "rejected", "pending")
ADMISSIBLE_ONLY_STATUS = "FINAL_ADMISSIBLE_DECLARATION_BOUND"
# Schemas known to be inventories / admissibility declarations / producer-frozen
# development lists. Named so the refusal says WHY, not only "wrong schema".
NON_SELECTION_SCHEMAS = {
    "m03_admissible_inputs.v1": "ADMISSIBLE_DECLARATION_IS_NOT_SELECTION",
    "selected_feature_manifest.v1": "LEGACY_PRODUCER_FROZEN_LIST_IS_NOT_SELECTION",
    "m04.input_binding.v1": "ADMISSIBLE_BINDING_IS_NOT_SELECTION",
    "feature_train_manifest.v2": "TRAIN_MANIFEST_IS_NOT_SELECTION",
}
# Roles that are the producer by definition; a decision by them is not independent.
PRODUCER_ROLES = {"producer", "successor technical lead", "technical lead", "self"}
REQUIRED_MANIFEST_KEYS = ("schema", "manifest_version", "dataset_id", "targets", "features", "producer",
                          "independent_decision_record", "source_digests", "manifest_sha256")
REQUIRED_DECISION_KEYS = ("schema", "decision_id", "decider_role", "decided_on", "manifest_sha256",
                          "verdict", "record_sha256")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SelectionManifestRefused(RuntimeError):
    """Raised when the gate refuses; ``report`` holds the full refusal."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("SELECTED_MANIFEST_GATE_REFUSED: " + "; ".join(report["reasons"]))


def canonical_sha256(obj: dict, drop: str) -> str:
    body = {k: v for k, v in obj.items() if k != drop}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(obj_or_path, label, reasons):
    if obj_or_path is None:
        reasons.append(f"MISSING_{label}")
        return None
    if isinstance(obj_or_path, dict):
        return obj_or_path
    p = Path(obj_or_path)
    if not p.is_file():
        reasons.append(f"MISSING_{label}: {p} does not exist")
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        reasons.append(f"UNREADABLE_{label}: {e}")
        return None
    if not isinstance(d, dict):
        reasons.append(f"UNREADABLE_{label}: top level is not an object")
        return None
    return d


def _role(x) -> str:
    if isinstance(x, dict):
        x = x.get("role")
    return x.strip().lower() if isinstance(x, str) else ""


def _check_manifest(m: dict, reasons: list, expected_dataset_id) -> dict:
    """Validate the manifest body; returns {name: state} for well-formed features."""
    schema = m.get("schema")
    if schema != MANIFEST_SCHEMA:
        tag = NON_SELECTION_SCHEMAS.get(schema, "WRONG_SCHEMA")
        reasons.append(f"{tag}: schema {schema!r} is not {MANIFEST_SCHEMA}")
        if m.get("input_set_status") == ADMISSIBLE_ONLY_STATUS:
            reasons.append(f"ADMISSIBLE_BINDING_IS_NOT_SELECTION: input_set_status {ADMISSIBLE_ONLY_STATUS}")
        return {}
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in m]
    if missing:
        reasons.append(f"MISSING_FIELDS: {missing}")
    if not isinstance(m.get("manifest_version"), int) or m.get("manifest_version", 0) < 1:
        reasons.append("BAD_VERSION: manifest_version must be an integer >= 1")
    if not isinstance(m.get("dataset_id"), str) or not m.get("dataset_id"):
        reasons.append("MISSING_DATASET_ID")
    elif expected_dataset_id is not None and m["dataset_id"] != expected_dataset_id:
        reasons.append(f"DATASET_MISMATCH: manifest {m['dataset_id']!r} != consumer {expected_dataset_id!r}")
    targets = m.get("targets")
    if not isinstance(targets, list) or not targets:
        reasons.append("MISSING_TARGETS")
    else:
        for t in targets:
            hz = t.get("horizons") if isinstance(t, dict) else None
            if not (isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"]
                    and isinstance(hz, list) and hz
                    and all(isinstance(h, int) and not isinstance(h, bool) and h > 0 for h in hz)):
                reasons.append(f"BAD_TARGET: {t!r} needs a name and positive integer horizons")
    if not _role(m.get("producer")):
        reasons.append("MISSING_PRODUCER_ROLE")
    ref = m.get("independent_decision_record")
    if not (isinstance(ref, dict) and ref.get("decision_id") and _role(ref.get("decider_role"))):
        reasons.append("MISSING_INDEPENDENT_DECISION_REFERENCE")
    if m.get("self_certified") is True:
        reasons.append("PRODUCER_SELF_CERTIFIED: manifest declares self_certified")
    src = m.get("source_digests")
    if not isinstance(src, dict) or not src:
        reasons.append("MISSING_SOURCE_DIGESTS")
    else:
        bad = sorted(k for k, v in src.items() if not (isinstance(v, str) and _HEX64.match(v)))
        if bad:
            reasons.append(f"BAD_SOURCE_DIGESTS: {bad}")
    feats = m.get("features")
    states = {}
    if not isinstance(feats, list) or not feats:
        reasons.append("MISSING_FEATURES")
        return states
    bad_state, no_reason, dup = {}, [], []
    for f in feats:
        name = f.get("name") if isinstance(f, dict) else None
        if not isinstance(name, str) or not name:
            reasons.append(f"BAD_FEATURE_ENTRY: {f!r}")
            continue
        if name in states:
            dup.append(name)
        st = f.get("state")
        if st not in STATES:
            bad_state.setdefault(str(st), []).append(name)
        rc = f.get("reason_codes")
        if not (isinstance(rc, list) and rc and all(isinstance(r, str) and r for r in rc)):
            no_reason.append(name)
        states[name] = st
    for st, names in sorted(bad_state.items()):
        reasons.append(f"NON_SELECTION_STATE: {len(names)} feature(s) in state {st!r} "
                       f"(only {'/'.join(STATES)} are decisions)")
    if no_reason:
        reasons.append(f"MISSING_REASON_CODES: {len(no_reason)} feature(s), first {no_reason[:5]}")
    if dup:
        reasons.append(f"DUPLICATE_FEATURES: {sorted(set(dup))[:5]}")
    if states and not any(s == "selected" for s in states.values()):
        reasons.append("NO_SELECTED_FEATURE: manifest is admissible/profiled/pending-only")
    if isinstance(m.get("manifest_sha256"), str) and m["manifest_sha256"] != canonical_sha256(m, "manifest_sha256"):
        reasons.append("DIGEST_MISMATCH: manifest_sha256 does not re-derive")
    return states


def _check_decision(m: dict, d: dict, reasons: list) -> None:
    if d.get("schema") != DECISION_SCHEMA:
        reasons.append(f"WRONG_DECISION_SCHEMA: {d.get('schema')!r}")
        return
    missing = [k for k in REQUIRED_DECISION_KEYS if k not in d]
    if missing:
        reasons.append(f"DECISION_MISSING_FIELDS: {missing}")
        return
    if d["record_sha256"] != canonical_sha256(d, "record_sha256"):
        reasons.append("DIGEST_MISMATCH: decision record_sha256 does not re-derive")
    if d["manifest_sha256"] != m.get("manifest_sha256"):
        reasons.append("DIGEST_MISMATCH: decision does not bind this manifest_sha256")
    if d["verdict"] != ACCEPT_VERDICT:
        reasons.append(f"DECISION_NOT_ACCEPTED: verdict {d['verdict']!r}")
    ref = m.get("independent_decision_record") or {}
    if ref.get("decision_id") != d["decision_id"] or _role(ref.get("decider_role")) != _role(d["decider_role"]):
        reasons.append("DECISION_REFERENCE_MISMATCH: manifest reference does not name this record")
    decider, producer = _role(d["decider_role"]), _role(m.get("producer"))
    if not decider or decider == producer or decider in PRODUCER_ROLES:
        reasons.append(f"PRODUCER_SELF_CERTIFIED: decider role {decider!r} is not independent of "
                       f"producer role {producer!r}")


def evaluate(manifest, consumed_features, *, decision_record=None, expected_dataset_id=None,
             source_files=None) -> dict:
    """Return the gate report without raising. ``admitted`` is True only when nothing refused."""
    reasons: list[str] = []
    consumed = list(consumed_features or [])
    if not consumed:
        reasons.append("EMPTY_CONSUMED_SET")
    if len(set(consumed)) != len(consumed):
        reasons.append("DUPLICATE_CONSUMED_FEATURES")
    m = _load(manifest, "MANIFEST", reasons)
    states = {}
    if m is not None:
        states = _check_manifest(m, reasons, expected_dataset_id)
        if m.get("schema") == MANIFEST_SCHEMA:
            d = _load(decision_record, "DECISION_RECORD", reasons)
            if d is not None:
                _check_decision(m, d, reasons)
            for name, path in (source_files or {}).items():
                want = (m.get("source_digests") or {}).get(name)
                if want is None:
                    reasons.append(f"UNDECLARED_SOURCE: {name}")
                elif not Path(path).is_file():
                    reasons.append(f"MISSING_SOURCE_FILE: {name}")
                elif file_sha256(path) != want:
                    reasons.append(f"DIGEST_MISMATCH: source {name}")
    per_feature = {}
    for f in consumed:
        st = states.get(f)
        per_feature[f] = st if st is not None else "ABSENT"
    not_present = [f for f, s in per_feature.items() if s == "ABSENT"]
    not_selected = [f for f, s in per_feature.items() if s not in ("ABSENT", "selected")]
    if m is not None and m.get("schema") == MANIFEST_SCHEMA:
        if not_present:
            reasons.append(f"CONSUMED_NOT_IN_MANIFEST: {len(not_present)} feature(s), first {not_present[:5]}")
        if not_selected:
            reasons.append(f"CONSUMED_NOT_SELECTED: {len(not_selected)} feature(s), first {not_selected[:5]}")
    admitted = not reasons
    refused = [] if admitted else consumed
    return {"gate": "selected_manifest_gate.v1", "admitted": admitted,
            "consumed_count": len(consumed), "refused_count": len(refused),
            "refused_features": refused, "reasons": reasons,
            "manifest_schema": (m or {}).get("schema"),
            "manifest_sha256": (m or {}).get("manifest_sha256") if admitted else None}


def require_selected_manifest(manifest, consumed_features, **kw) -> dict:
    """Fail-closed entry: return the admitting report or raise SelectionManifestRefused."""
    report = evaluate(manifest, consumed_features, **kw)
    if not report["admitted"]:
        raise SelectionManifestRefused(report)
    return report


def require_campaign_inputs(campaign: dict, consumed_features, *, base_dir=None, **kw) -> dict:
    """Gate for a campaign / input-binding dict.

    The campaign must name ``selection_manifest`` and ``selection_decision``
    (paths, relative to ``base_dir`` when given, or inline dicts). A binding whose
    only claim is ``FINAL_ADMISSIBLE_DECLARATION_BOUND`` has neither and refuses.
    """
    def resolve(v):
        if isinstance(v, str) and base_dir is not None and not Path(v).is_absolute():
            return str(Path(base_dir) / v)
        return v
    c = campaign or {}
    binding = c.get("input_binding") if isinstance(c.get("input_binding"), dict) else c
    man = c.get("selection_manifest", binding.get("selection_manifest"))
    dec = c.get("selection_decision", binding.get("selection_decision"))
    report = evaluate(resolve(man), consumed_features, decision_record=resolve(dec), **kw)
    if ADMISSIBLE_ONLY_STATUS in (binding.get("input_set_status"), c.get("input_set_status")) and man is None:
        report["reasons"].insert(0, "ADMISSIBLE_BINDING_IS_NOT_SELECTION: input_set_status "
                                    f"{ADMISSIBLE_ONLY_STATUS} without a selection manifest")
        report["admitted"] = False
        report["refused_features"] = list(consumed_features or [])
        report["refused_count"] = len(report["refused_features"])
    if not report["admitted"]:
        raise SelectionManifestRefused(report)
    return report


def features_declared_by(obj: dict) -> list:
    """The column list a non-selection artifact would hand a builder (used by tests and the CLI)."""
    for key in ("all_admissible_control", "features"):
        v = obj.get(key)
        if isinstance(v, list) and v:
            return [x["name"] if isinstance(x, dict) else x for x in v]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=False)
    ap.add_argument("--decision", required=False)
    ap.add_argument("--dataset-id", required=False)
    ap.add_argument("--features", nargs="*", help="consumed features; default: every feature the file declares")
    a = ap.parse_args(argv)
    feats = a.features
    if not feats and a.manifest and Path(a.manifest).is_file():
        feats = features_declared_by(json.loads(Path(a.manifest).read_text()))
    rep = evaluate(a.manifest, feats, decision_record=a.decision, expected_dataset_id=a.dataset_id)
    out = dict(rep)
    out["refused_features"] = f"{len(rep['refused_features'])} listed" if rep["refused_features"] else []
    print(json.dumps(out, indent=1))
    return 0 if rep["admitted"] else 2


if __name__ == "__main__":
    sys.exit(main())
