#!/usr/bin/env python3
"""C118, C119, C121 (order 2026-09-12): per-variable preprocessing design v5.

v5 corrects the IMPLEMENTATION of v4 and nothing scientific. v4's prose
required every member to satisfy terminal, semantics, lineage, time,
license and missing policy at once, but `derive_population()` counted each
artifact separately and formed panels from CAUSAL_ACTIVE columns plus the
presence of a temporal contract: six datasets with no terminal and no
declared variable came out BANK_SUFFICIENT_FOR_REVIEW. Its validator took
booleans, NaN and duplicates, and its Holm took p outside [0, 1] and an
incomplete family.

v5:

* joins member by member on (dataset_id, dataset_sha256, column); a
  member exists only if ONE key satisfies all eight conditions, and every
  aggregate is derived from the membership ledger rows;
* refuses duplicates instead of counting them once;
* parses strictly (no duplicate keys, no NaN/Infinity) and types every
  field it checks (a boolean is not a number);
* requires the complete Holm family with p in [0, 1], binds every
  contrast to one frozen sample digest and derives LOPO from panel rows;
* copies v4's scientific blocks and refuses any difference from them.

It imports no numeric library and computes no score.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V4_DOCUMENT = REPO / "docs/audits/evidence/PER_VARIABLE_PREPROCESSING_DESIGN.v4.json"


def _load_v4():
    spec = importlib.util.spec_from_file_location("per_variable_design_v4_frozen",
                                                  HERE / "per_variable_design_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4 = _load_v4()
sha_obj = V4.sha_obj

SCHEMA = "crispdm.per_variable_preprocessing_design.v5"
STATUS = V4.STATUS
MIN_PANELS = V4.MIN_PANELS
MIN_VARIABLES_PER_PANEL = V4.MIN_VARIABLES_PER_PANEL
MAX_MISSING_FRACTION = 0.20
MIN_OBSERVATIONS = 2000
FAMILY = ("H1", "H2_vs_A0", "H2_vs_A3")
UNKNOWN = "UNKNOWN"
# A declaration whose value is one of these declares nothing.
UNDECLARED_VALUES = frozenset({UNKNOWN, "NONE"})
LICENSE_REQUIRED = "EXTERNAL_V5_DESIGN_REVIEW_AND_LICENSE_REQUIRED"
TEMPORAL_SEMANTIC_TYPES = frozenset({"date", "datetime", "time", "timestamp",
                                     "timestamp_ms", "epoch_ms", "epoch_seconds"})
# Unchanged from v4, byte for byte; validate() refuses any difference.
SCIENTIFIC_BLOCKS = ("hypotheses", "arms", "panels", "temporal_quality",
                     "common_sample", "operators", "model", "evaluation",
                     "budget", "inference", "rules")
REQUIRED = SCIENTIFIC_BLOCKS + ("schema", "status", "supersedes", "population",
                                "implementation", "license", "grants_nothing",
                                "design_sha256")
DECLARED_RESULT_KEYS = frozenset({"lopo_result", "lopo_passed", "passed", "p_value",
                                  "p_values", "scores", "results"})
CONDITIONS = (
    "terminal_independently_recomputed",
    "semantic_state_numeric_measurable",
    "dag_causal_active_with_complete_binding",
    "role_input_feature",
    "semantic_type_unit_license_declared",
    "missing_and_sentinel_policy_declared",
    "temporal_contract_and_mask_same_dataset",
    "missingness_and_observation_limits",
)
# The census schemas the join reads. An unschema'd document lists its rows
# under `variables`; the C115 successor census declares its schema, lists
# rows under `rows` and declares exactly these row keys.
SUCCESSOR_CENSUS_SCHEMA = "financial_data.eth_h4_successor_semantic_census.v1"
SUCCESSOR_CENSUS_ROW_KEYS = ("variable_id", "dataset_id", "dataset_sha256", "column",
                             "physical_type", "semantic_type", "semantics", "role", "unit",
                             "license", "license_source", "missing_policy", "sentinel_policy",
                             "producer", "symbol", "lookback_bars", "evidence")
HEX64 = re.compile(r"[0-9a-f]{64}")


class StrictJsonRefusal(ValueError):
    pass


class DesignRefusal(ValueError):
    pass


class InferenceRefusal(ValueError):
    pass


class PopulationRefusal(ValueError):
    pass


class ScoringRefusal(RuntimeError):
    pass


# ------------------------------------------------------------ strict parsing
def strict_json_loads(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise StrictJsonRefusal(f"DUPLICATE_KEY: {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise StrictJsonRefusal(f"NON_FINITE_CONSTANT: {name}")

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def strict_json_file(path):
    return strict_json_loads(Path(path).read_bytes())


def _read_json(path) -> tuple[object, str]:
    """One read: the digest and the document come from the same bytes."""
    raw = Path(path).read_bytes()
    return strict_json_loads(raw), hashlib.sha256(raw).hexdigest()


def _is_int(x) -> bool:
    return type(x) is int


def _is_number(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def _non_finite_paths(node, prefix=""):
    if type(node) is float and not math.isfinite(node):
        yield prefix or "<root>"
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _non_finite_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _non_finite_paths(v, f"{prefix}[{i}]")


def _keys(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from _keys(v)


def _unique(xs) -> bool:
    return isinstance(xs, list) and len(xs) == len({json.dumps(x, sort_keys=True) for x in xs})


# ------------------------------------------------------------------- design
def v4_reference(path=V4_DOCUMENT) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    doc = strict_json_loads(raw)
    problems = V4.validate(doc)
    if problems:
        raise DesignRefusal(f"V4_REFERENCE_INVALID: {problems}")
    return doc, hashlib.sha256(raw).hexdigest()


def build_design(v4_path=V4_DOCUMENT) -> dict:
    v4, v4_file_sha = v4_reference(v4_path)
    d = {k: copy.deepcopy(v4[k]) for k in SCIENTIFIC_BLOCKS}
    d.update({
        "schema": SCHEMA, "status": STATUS,
        "supersedes": {
            "document": "PER_VARIABLE_PREPROCESSING_DESIGN.v4.json",
            "file_sha256": v4_file_sha, "design_sha256": v4["design_sha256"],
            "rewritten": False, "scientific_change": "NONE",
            "unchanged_blocks": list(SCIENTIFIC_BLOCKS),
            "why": "v4's population derivation counted artifacts separately "
                   "and never joined them, so a bank with no terminal and no "
                   "declared variable was sufficient; its validator and Holm "
                   "accepted booleans, NaN, duplicates, p outside [0, 1] and "
                   "an incomplete family. v5 changes the implementation and "
                   "its bindings only."},
        "population": {
            "artifacts": {
                "terminals": "successor characterization terminals (C116), one per column",
                "semantic_census": "ETH_H4_SUCCESSOR_SEMANTIC_CENSUS.v1 (C115)",
                "lineage": "financial_data.feature_dag.v4 with PRODUCER_BINDING_MANIFEST.v2",
                "temporal": "ETH_H4_SUCCESSOR_TEMPORAL_CONTRACT.v3 and its mask (C114), "
                            "one per panel dataset"},
            "requires_all": copy.deepcopy(v4["population"]["requires_all"]),
            "excluded_by_rule": copy.deepcopy(v4["population"]["excluded_by_rule"]),
            "empty": v4["population"]["empty"],
            "join": {
                "key": ["dataset_id", "dataset_sha256", "column"],
                "member": "one key satisfying every condition at once",
                "conditions": list(CONDITIONS),
                "limits": {"max_missing_fraction": MAX_MISSING_FRACTION,
                           "min_observations": MIN_OBSERVATIONS},
                "duplicates": "REFUSED",
                "aggregates": "derived only from membership ledger rows; "
                              "aggregates supplied by inputs are ignored"}},
        "implementation": {
            "population": "per_variable_design_v5.derive_population",
            "strict_parser": "per_variable_design_v5.strict_json_loads",
            "validator": "per_variable_design_v5.validate",
            "holm": "per_variable_design_v5.holm_adjust_family",
            "panel_contrast": "per_variable_design_v5.panel_contrast_rows",
            "common_sample": "per_variable_design_v5.family_on_one_sample",
            "lopo": "per_variable_design_v5.lopo_from_panel_rows",
            "scoring": "per_variable_design_v5.score refuses"},
        "license": {"scoring": "NOT_GRANTED", "required": LICENSE_REQUIRED,
                    "allowed": copy.deepcopy(v4["license"]["allowed"])},
        "grants_nothing": v4["grants_nothing"],
    })
    d["design_sha256"] = sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    return d


def typed_problems(d: dict) -> list[str]:
    """Type and domain checks that stand on their own, without v4."""
    p = [f"NON_FINITE: {x}" for x in _non_finite_paths(d)]
    try:
        h = d["hypotheses"]
        m1 = h["H1"]["margin"]
        if not (_is_number(m1) and 0.0 <= m1 < 1.0):
            p.append("H1 margin must be a finite number in [0, 1), not a boolean")
        ms, cs = h["H2"]["margins"], h["H2"]["contrasts"]
        if not (isinstance(ms, list) and isinstance(cs, list) and len(ms) == len(cs)
                and all(_is_number(m) and 0.0 <= m < 1.0 for m in ms)):
            p.append("H2 margins must be finite numbers in [0, 1), one per contrast")
        if not _unique(cs):
            p.append("duplicate H2 contrast")
        if type(h["H2"]["requires_both"]) is not bool:
            p.append("H2 requires_both must be a boolean")
        pn = d["panels"]
        if not (_is_int(pn["minimum_independent_panels"]) and pn["minimum_independent_panels"] >= MIN_PANELS):
            p.append("minimum_independent_panels must be an integer >= 6")
        if not (_is_int(pn["minimum_eligible_variables_per_panel"])
                and pn["minimum_eligible_variables_per_panel"] >= MIN_VARIABLES_PER_PANEL):
            p.append("minimum_eligible_variables_per_panel must be an integer >= 5")
        inf = d["inference"]
        if not (_is_int(inf["minimum_panels_for_inference"]) and inf["minimum_panels_for_inference"] >= MIN_PANELS):
            p.append("minimum_panels_for_inference must be an integer >= 6")
        if not (type(inf["confidence"]) is float and 0.0 < inf["confidence"] < 1.0):
            p.append("confidence must be a float in (0, 1)")
        if inf["family"] != list(FAMILY):
            p.append(f"family must be exactly {list(FAMILY)} with no duplicate")
        ev = d["evaluation"]
        for k in ("origins", "test_block_bars"):
            if not (_is_int(ev[k]) and ev[k] > 0):
                p.append(f"evaluation.{k} must be a positive integer")
        if not (_is_int(ev["embargo_bars"]) and ev["embargo_bars"] >= 0):
            p.append("evaluation.embargo_bars must be a non-negative integer")
        if not (_is_number(ev["inner_validation_fraction"]) and 0.0 < ev["inner_validation_fraction"] < 1.0):
            p.append("inner_validation_fraction must be in (0, 1)")
        seeds = ev["seeds"]
        if not (isinstance(seeds, list) and seeds and all(_is_int(s) for s in seeds) and _unique(seeds)):
            p.append("seeds must be distinct integers")
        ops = d["operators"]
        if not (isinstance(ops, list) and all(isinstance(o, str) for o in ops) and _unique(ops)):
            p.append("operators must be distinct names")
        mo = d["model"]
        if not (_is_int(mo["lags"]) and mo["lags"] > 0 and _is_number(mo["alpha"])):
            p.append("model lags/alpha types")
        bu = d["budget"]
        if bu["same_for_every_arm"] is not True or bu["accelerator"] != "NONE" \
                or not (_is_int(bu["cpu_wall_seconds_per_panel_arm_seed"])
                        and bu["cpu_wall_seconds_per_panel_arm_seed"] > 0):
            p.append("budget")
        req = d["population"]["requires_all"]
        if not (isinstance(req, list) and _unique(req)):
            p.append("population requirements must be distinct")
    except (KeyError, TypeError) as exc:
        p.append(f"STRUCTURE: {exc!r}")
    declared = sorted(DECLARED_RESULT_KEYS & set(_keys(d)))
    if declared:
        p.append(f"DECLARED_RESULT_IN_DESIGN: {declared}")
    return p


def validate(d, *, v4_path=V4_DOCUMENT, reviewed_sha256: str | None = None) -> list[str]:
    if not isinstance(d, dict) or set(d) != set(REQUIRED):
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        return [f"keys differ: expected {sorted(REQUIRED)}, got {got}"]
    p = []
    if d["schema"] != SCHEMA or d["status"] != STATUS:
        p.append("schema or status")
    if d["design_sha256"] != sha_obj({k: v for k, v in d.items() if k != "design_sha256"}):
        p.append("design_sha256 does not match the content")
    if reviewed_sha256 is not None and d["design_sha256"] != reviewed_sha256:
        p.append("DELTA_AFTER_REVIEW")
    v4, v4_file_sha = v4_reference(v4_path)
    sup = d["supersedes"]
    if not isinstance(sup, dict) or sup.get("file_sha256") != v4_file_sha \
            or sup.get("design_sha256") != v4["design_sha256"] \
            or sup.get("scientific_change") != "NONE" or sup.get("rewritten") is not False:
        p.append("supersedes must bind v4 by file and design digest with scientific_change NONE")
    for block in SCIENTIFIC_BLOCKS:
        if sha_obj(d[block]) != sha_obj(v4[block]):
            p.append(f"SCIENTIFIC_CHANGE: {block} differs from v4")
    pop = d["population"]
    for k in ("requires_all", "excluded_by_rule"):
        if not isinstance(pop, dict) or sha_obj(pop.get(k)) != sha_obj(v4["population"][k]):
            p.append(f"SCIENTIFIC_CHANGE: population.{k} differs from v4")
    join = pop.get("join") if isinstance(pop, dict) else None
    if not isinstance(join, dict) or join.get("conditions") != list(CONDITIONS) \
            or join.get("key") != ["dataset_id", "dataset_sha256", "column"] \
            or join.get("duplicates") != "REFUSED":
        p.append("population join must declare the exact key, the eight conditions and refusal of duplicates")
    lic = d["license"]
    if not isinstance(lic, dict) or lic.get("scoring") != "NOT_GRANTED" or lic.get("required") != LICENSE_REQUIRED:
        p.append("scoring may not be granted")
    p.extend(typed_problems(d))
    return p


# ---------------------------------------------------------------- inference
def _check_p(name, p):
    if type(p) not in (int, float) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise InferenceRefusal(f"P_OUT_OF_DOMAIN: {name}={p!r}")


def holm_adjust_family(pvalues: dict, family=FAMILY) -> dict:
    fam = list(family)
    if len(set(fam)) != len(fam):
        raise InferenceRefusal("DUPLICATE_CONTRAST_IN_FAMILY")
    if list(fam) != list(FAMILY):
        raise InferenceRefusal(f"FOREIGN_FAMILY: {fam}")
    if not isinstance(pvalues, dict) or set(pvalues) != set(fam):
        got = sorted(pvalues) if isinstance(pvalues, dict) else type(pvalues).__name__
        raise InferenceRefusal(f"INCOMPLETE_OR_FOREIGN_FAMILY: expected {sorted(fam)}, got {got}")
    for k, p in pvalues.items():
        _check_p(k, p)
    return V4.holm_adjust(pvalues)


ROW_KEYS = frozenset({"panel", "contrast", "sample_sha256", "value"})


def _check_rows(rows, *, contrast, sample_sha256):
    if contrast not in FAMILY:
        raise InferenceRefusal(f"FOREIGN_CONTRAST: {contrast!r}")
    if not (isinstance(sample_sha256, str) and HEX64.fullmatch(sample_sha256)):
        raise InferenceRefusal("FROZEN_SAMPLE_DIGEST_REQUIRED")
    if not isinstance(rows, list):
        raise InferenceRefusal("PANEL_ROWS_MUST_BE_A_LIST")
    panels, values = [], []
    for r in rows:
        if not isinstance(r, dict) or set(r) != ROW_KEYS:
            raise InferenceRefusal("PANEL_ROW_SCHEMA")
        if r["contrast"] != contrast:
            raise InferenceRefusal(f"FOREIGN_CONTRAST_ROW: {r['contrast']!r}")
        if r["sample_sha256"] != sample_sha256:
            raise InferenceRefusal(f"SAMPLE_NOT_FROZEN: panel {r['panel']!r} used another sample")
        if not isinstance(r["panel"], str) or r["panel"] in panels:
            raise InferenceRefusal(f"DUPLICATE_OR_UNNAMED_PANEL: {r['panel']!r}")
        if type(r["value"]) not in (int, float) or not math.isfinite(r["value"]):
            raise InferenceRefusal(f"PANEL_VALUE_NOT_A_FINITE_NUMBER: {r['panel']!r}")
        panels.append(r["panel"])
        values.append(float(r["value"]))
    return panels, values


def panel_contrast_rows(rows, *, contrast: str, sample_sha256: str, level: float) -> dict:
    if type(level) is not float or not 0.0 < level < 1.0:
        raise InferenceRefusal("LEVEL_OUT_OF_DOMAIN")
    panels, values = _check_rows(rows, contrast=contrast, sample_sha256=sample_sha256)
    out = V4.panel_contrast(values, level)
    out.update(contrast=contrast, sample_sha256=sample_sha256, panels_used=sorted(panels))
    return out


def family_on_one_sample(contrast_rows: dict) -> str:
    """Every contrast of the family, each row on the same frozen sample."""
    if not isinstance(contrast_rows, dict) or set(contrast_rows) != set(FAMILY):
        raise InferenceRefusal("INCOMPLETE_OR_FOREIGN_FAMILY")
    digests = set()
    for contrast, rows in contrast_rows.items():
        if not isinstance(rows, list):
            raise InferenceRefusal("PANEL_ROWS_MUST_BE_A_LIST")
        for r in rows:
            if not isinstance(r, dict) or set(r) != ROW_KEYS:
                raise InferenceRefusal("PANEL_ROW_SCHEMA")
            digests.add(r["sample_sha256"])
    if len(digests) != 1:
        raise InferenceRefusal(f"CONTRASTS_ON_DIFFERENT_SAMPLES: {len(digests)}")
    sample = digests.pop()
    for contrast, rows in contrast_rows.items():
        _check_rows(rows, contrast=contrast, sample_sha256=sample)
    return sample


def lopo_from_panel_rows(rows, *, contrast: str, sample_sha256: str, level: float) -> dict:
    """Leave-one-panel-out, recomputed from the panel rows; never accepted
    as a declaration."""
    panels, _ = _check_rows(rows, contrast=contrast, sample_sha256=sample_sha256)
    return {"contrast": contrast, "sample_sha256": sample_sha256,
            "derived_from": "panel_rows",
            "leave_out": {p: panel_contrast_rows([r for r in rows if r["panel"] != p],
                                                 contrast=contrast, sample_sha256=sample_sha256,
                                                 level=level)
                          for p in sorted(panels)}}


def score(*_args, **_kwargs):
    raise ScoringRefusal(LICENSE_REQUIRED)


# --------------------------------------------------------------- population
def _declared(v) -> bool:
    return isinstance(v, str) and v.strip() != "" and v not in UNDECLARED_VALUES


def census_rows(cen) -> list:
    """The row list of a census this join knows, by exact schema."""
    if not isinstance(cen, dict):
        raise PopulationRefusal("CENSUS_NOT_AN_OBJECT")
    schema = cen.get("schema")
    if schema is None:
        rows = cen.get("variables")
        if not isinstance(rows, list):
            raise PopulationRefusal("CENSUS_WITHOUT_VARIABLES")
        return rows
    if schema != SUCCESSOR_CENSUS_SCHEMA:
        raise PopulationRefusal(f"UNKNOWN_CENSUS_SCHEMA: {schema!r}")
    if cen.get("row_keys") != list(SUCCESSOR_CENSUS_ROW_KEYS) or not isinstance(cen.get("rows"), list):
        raise PopulationRefusal("SUCCESSOR_CENSUS_ROW_KEYS_OR_ROWS")
    for r in cen["rows"]:
        if not isinstance(r, dict) or set(r) != set(SUCCESSOR_CENSUS_ROW_KEYS):
            got = sorted(r) if isinstance(r, dict) else type(r).__name__
            raise PopulationRefusal(f"SUCCESSOR_CENSUS_ROW_SCHEMA: {got}")
    return cen["rows"]


def _key(row, where):
    if not isinstance(row, dict):
        raise PopulationRefusal(f"ROW_NOT_AN_OBJECT in {where}")
    k = (row.get("dataset_id"), row.get("dataset_sha256"), row.get("column"))
    if not all(isinstance(x, str) and x for x in k):
        raise PopulationRefusal(f"KEY_INCOMPLETE in {where}: {k}")
    return k


def _index(rows, where):
    out = {}
    for r in rows:
        k = _key(r, where)
        if k in out:
            raise PopulationRefusal(f"DUPLICATE_{where.upper()}_ROW: {k}")
        out[k] = r
    return out


def _dag_rows(dag: dict, manifest: dict | None) -> list[dict]:
    """DAG nodes carry dataset_id and column; the dataset digest comes from
    the node or, when absent, from exactly one manifest binding."""
    bound = {}
    for b in (manifest or {}).get("bindings", []):
        k = (b.get("dataset_id"), b.get("output_column"))
        if k in bound:
            raise PopulationRefusal(f"DUPLICATE_BINDING: {k}")
        bound[k] = b
    rows = []
    for n in dag.get("nodes", []):
        r = dict(n)
        if "dataset_sha256" not in r:
            b = bound.get((n.get("dataset_id"), n.get("column")))
            r["dataset_sha256"] = b["dataset_sha256"] if b and isinstance(b.get("dataset_sha256"), str) \
                else "UNBOUND_NO_DATASET_DIGEST"
        rows.append(r)
    return rows


def _temporal_index(contracts: list[dict]) -> tuple[dict, list]:
    out, unusable = {}, []
    for i, doc in enumerate(contracts):
        ds = doc.get("dataset") if isinstance(doc, dict) else None
        k = ((ds or {}).get("dataset_id"), (ds or {}).get("dataset_sha256"))
        if not all(isinstance(x, str) and x for x in k):
            unusable.append({"contract": i, "reason": "NO_DATASET_ID_AND_DIGEST"})
            continue
        if k in out:
            raise PopulationRefusal(f"DUPLICATE_TEMPORAL_CONTRACT: {k}")
        out[k] = doc
    return out, unusable


def _member_row(k, t, c, n, tc) -> dict:
    cond, reasons = {}, []

    if t is None:
        reasons.append("NO_TERMINAL_FOR_KEY")
    elif c is not None and t.get("variable_id") != c.get("variable_id"):
        reasons.append("TERMINAL_VARIABLE_ID_DIFFERS_FROM_CENSUS")
    elif t.get("layer") != "INDEPENDENTLY_RECOMPUTED":
        reasons.append(f"TERMINAL_LAYER_{t.get('layer')}")
    cond[CONDITIONS[0]] = t is not None and c is not None \
        and t.get("variable_id") == c.get("variable_id") \
        and t.get("layer") == "INDEPENDENTLY_RECOMPUTED"

    temporal_semantics = c is not None and str(c.get("semantic_type", "")).lower() in TEMPORAL_SEMANTIC_TYPES
    if t is not None and t.get("semantic_state") != "NUMERIC_MEASURABLE":
        reasons.append(f"SEMANTIC_STATE_{t.get('semantic_state')}")
    if temporal_semantics:
        reasons.append("TIMESTAMP_EXCLUDED_BY_RULE")
    cond[CONDITIONS[1]] = t is not None and t.get("semantic_state") == "NUMERIC_MEASURABLE" \
        and not temporal_semantics

    binding = (n or {}).get("binding") if n else None
    cond[CONDITIONS[2]] = n is not None and n.get("class") == "CAUSAL_ACTIVE" \
        and isinstance(binding, dict) and binding.get("complete") is True
    if n is None:
        reasons.append("NO_DAG_NODE_FOR_KEY")
    elif not cond[CONDITIONS[2]]:
        reasons.append(f"DAG_{n.get('class')}_OR_INCOMPLETE_BINDING")

    if c is None:
        reasons.append("NO_CENSUS_ROW_FOR_KEY")
    cond[CONDITIONS[3]] = c is not None and c.get("role") == "input_feature"
    if c is not None and not cond[CONDITIONS[3]]:
        reasons.append(f"ROLE_{c.get('role', 'ABSENT')}")

    fields5 = ("semantic_type", "unit", "license", "license_source")
    missing5 = [f for f in fields5 if c is None or not _declared(c.get(f))]
    cond[CONDITIONS[4]] = c is not None and not missing5
    if c is not None and missing5:
        reasons.append("UNDECLARED_" + "_".join(f.upper() for f in missing5))

    fields6 = ("missing_policy", "sentinel_policy")
    missing6 = [f for f in fields6 if c is None or not _declared(c.get(f))]
    cond[CONDITIONS[5]] = c is not None and not missing6
    if c is not None and missing6:
        reasons.append("UNDECLARED_" + "_".join(f.upper() for f in missing6))

    mask = (tc or {}).get("mask_artifact") if tc else None
    cond[CONDITIONS[6]] = tc is not None and isinstance(mask, dict) \
        and mask.get("dataset_id") == k[0] and mask.get("dataset_sha256") == k[1] \
        and isinstance(mask.get("sha256"), str) and bool(HEX64.fullmatch(mask["sha256"]))
    if tc is None:
        reasons.append("NO_TEMPORAL_CONTRACT_FOR_DATASET_AND_DIGEST")
    elif not cond[CONDITIONS[6]]:
        reasons.append("MASK_NOT_BOUND_TO_SAME_DATASET")

    obs = (t or {}).get("observations")
    mf = (t or {}).get("missing_fraction")
    cond[CONDITIONS[7]] = t is not None and _is_int(obs) and obs >= MIN_OBSERVATIONS \
        and _is_number(mf) and 0.0 <= mf <= MAX_MISSING_FRACTION
    if t is not None and not cond[CONDITIONS[7]]:
        reasons.append("MISSINGNESS_OR_OBSERVATION_LIMIT")

    vid = (c or {}).get("variable_id") if c else (t or {}).get("variable_id")
    return {"dataset_id": k[0], "dataset_sha256": k[1], "column": k[2],
            "variable_id": vid, "member": all(cond.values()),
            "conditions": cond, "reasons": sorted(set(reasons))}


def derive_population(*, terminals_dir, dag, census, temporal_contracts,
                      binding_manifest=None) -> dict:
    missing = [name for name, x in (("terminals_dir", terminals_dir), ("dag", dag),
                                    ("census", census))
               if x is None or not Path(x).exists()]
    if missing:
        return {"state": "UNDETERMINED", "missing_artifacts": missing,
                "members": 0, "verdict": "BANK_INSUFFICIENT"}
    dag_doc, dag_sha = _read_json(dag)
    man, man_sha = _read_json(binding_manifest) if binding_manifest else (None, None)
    cen, cen_sha = _read_json(census)
    terms, th, names = [], hashlib.sha256(), []
    for p in sorted(Path(terminals_dir).glob("*.json")):
        doc, s = _read_json(p)
        terms.append(doc)
        th.update(p.name.encode() + b"\0" + bytes.fromhex(s))
        names.append(p.name)
    contracts, contract_ids = [], []
    for p in temporal_contracts:
        doc, s = _read_json(p)
        contracts.append(doc)
        contract_ids.append({"name": Path(p).name, "sha256": s})
    inputs = {"terminals": {"count": len(names), "content_sha256": th.hexdigest(),
                            "rule": "sha256 over sorted (file name, NUL, file sha256)"},
              "dag": {"name": Path(dag).name, "sha256": dag_sha},
              "binding_manifest": {"name": Path(binding_manifest).name, "sha256": man_sha}
              if binding_manifest else "NOT_SUPPLIED",
              "census": {"name": Path(census).name, "sha256": cen_sha,
                         "schema": cen.get("schema", "UNSCHEMAD") if isinstance(cen, dict) else None},
              "temporal_contracts": contract_ids}

    variables = census_rows(cen)
    ids = [v.get("variable_id") for v in variables if isinstance(v, dict)]
    if len(ids) != len(set(ids)):
        raise PopulationRefusal("DUPLICATE_CENSUS_VARIABLE_ID")
    dag_idx = _index(_dag_rows(dag_doc, man), "dag")
    term_idx = _index(terms, "terminal")
    cen_idx = _index(variables, "census")
    temporal, unusable = _temporal_index(contracts)

    candidates = sorted(set(dag_idx) | set(term_idx) | set(cen_idx))
    ledger = [_member_row(k, term_idx.get(k), cen_idx.get(k), dag_idx.get(k),
                          temporal.get(k[:2])) for k in candidates]
    members = [r for r in ledger if r["member"]]
    by_dataset = {}
    for r in members:
        by_dataset.setdefault((r["dataset_id"], r["dataset_sha256"]), []).append(r["column"])
    panels = {f"{ds}@{dsha[:12]}": sorted(cols) for (ds, dsha), cols in sorted(by_dataset.items())
              if len(cols) >= MIN_VARIABLES_PER_PANEL}
    below = {f"{ds}@{dsha[:12]}": len(cols) for (ds, dsha), cols in sorted(by_dataset.items())
             if len(cols) < MIN_VARIABLES_PER_PANEL}
    sufficient = len(panels) >= MIN_PANELS
    exclusions = {}
    for r in ledger:
        for reason in r["reasons"]:
            exclusions[reason] = exclusions.get(reason, 0) + 1
    return {
        "state": "DERIVED",
        "rule": "a member is one (dataset_id, dataset_sha256, column) satisfying every "
                "condition at once; every count below is derived from the ledger rows",
        "inputs": inputs,
        "input_aggregates": "IGNORED",
        "ignored_input_keys": sorted(k for k in cen if k not in ("variables", "rows", "schema", "row_keys")),
        "unusable_temporal_contracts": unusable,
        "candidates": len(ledger),
        "eligible_variables": len(members),
        "panels": panels, "panel_count": len(panels),
        "datasets_below_minimum": below,
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "deficit": {"panels_required": MIN_PANELS, "panels_qualifying": len(panels),
                    "panels_missing": max(0, MIN_PANELS - len(panels)),
                    "variables_per_panel_required": MIN_VARIABLES_PER_PANEL},
        "verdict": "BANK_SUFFICIENT_FOR_REVIEW" if sufficient else "BANK_INSUFFICIENT",
        "members": sum(len(c) for c in panels.values()) if sufficient else 0,
        "population_sha256": sha_obj(sorted([r["dataset_id"], r["dataset_sha256"], r["column"],
                                             r["variable_id"]] for r in members)),
        "ledger": ledger,
        "ledger_sha256": sha_obj(ledger),
    }


def _write_once(path: Path, doc: dict) -> None:
    if path.exists():
        raise SystemExit(f"REFUSED: {path.name} exists; outputs are write-once")
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if str(Path.home()) in text:
        raise SystemExit("REFUSED: absolute home path in a published document")
    path.write_text(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", type=Path)
    ap.add_argument("--validate", type=Path)
    ap.add_argument("--terminals-dir", type=Path)
    ap.add_argument("--dag", type=Path)
    ap.add_argument("--binding-manifest", type=Path)
    ap.add_argument("--census", type=Path)
    ap.add_argument("--temporal-contract", type=Path, action="append", default=[])
    ap.add_argument("--ledger-out", type=Path)
    a = ap.parse_args(argv)
    rc = 0
    if a.write:
        d = build_design()
        _write_once(a.write, d)
        print(json.dumps({"design_sha256": d["design_sha256"]}))
    if a.validate:
        try:
            problems = validate(strict_json_file(a.validate))
        except StrictJsonRefusal as exc:
            problems = [f"STRICT_JSON: {exc}"]
        print(json.dumps({"problems": problems}, indent=1, sort_keys=True))
        rc = 1 if problems else rc
    if a.census or a.dag or a.terminals_dir:
        pop = derive_population(terminals_dir=a.terminals_dir, dag=a.dag, census=a.census,
                                temporal_contracts=a.temporal_contract,
                                binding_manifest=a.binding_manifest)
        if a.ledger_out:
            _write_once(a.ledger_out, pop)
        print(json.dumps({k: v for k, v in pop.items() if k != "ledger"}, indent=1, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main())
