#!/usr/bin/env python3
"""C129 (order 2026-09-12): the common dataset and variable contract.

Every bank (public, financial, synthetic) produces this contract through
its own adapter and keeps the source's own metadata verbatim under
`original_fields`, so nothing the source declared is lost in translation.

The contract records identity, source, license, semantics, unit,
frequency, event and availability time, missingness, sentinels,
partitions and digests. What is not known is written UNKNOWN, never
guessed. A contract describes; it never grants: PUBLICLY_ELIGIBLE and
LIVE_ELIGIBLE refuse anywhere inside one.

Partitions are chronological and frozen before any profile is computed.
Sealed periods are listed and excluded, never read.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

DATASET_SCHEMA = "crispdm.data_foundation.dataset_contract.v1"
VARIABLE_SCHEMA = "crispdm.data_foundation.variable_contract.v1"
UNKNOWN = "UNKNOWN"
BANKS = ("PUBLIC", "FINANCIAL", "SYNTHETIC")
LICENSE_STATES = ("OPEN_ATTRIBUTION", "OPEN_PUBLIC_DOMAIN", "RESTRICTED_NO_DERIVATIVES",
                  "RESTRICTED_NON_COMMERCIAL", "TERMS_REQUIRE_REVIEW",
                  "INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE", "NOT_APPLICABLE_GENERATED", UNKNOWN)
LICENSE_STATES_NEEDING_EVIDENCE = ("OPEN_ATTRIBUTION", "OPEN_PUBLIC_DOMAIN", "RESTRICTED_NO_DERIVATIVES",
                                   "RESTRICTED_NON_COMMERCIAL", "TERMS_REQUIRE_REVIEW")
TIMESTAMP_MEANINGS = ("PERIOD_START", "PERIOD_END", "INSTANT", "SAMPLE_INDEX", UNKNOWN)
AVAILABILITY_RULES = ("SAMPLE_INDEX", "PERIOD_END", "BAR_CLOSE", "PUBLICATION_TIME",
                      "INSTANT_PLUS_DECLARED_DELAY", UNKNOWN)
FREQUENCY_WORDS = ("IRREGULAR", "NOT_APPLICABLE", UNKNOWN)
ROLES = ("INPUT_CANDIDATE", "IDENTIFIER", "TIMESTAMP", "EXCLUDED", UNKNOWN)
PRODUCER_KINDS = ("SOURCE_MEASUREMENT", "DERIVED_BY_CODE", "GENERATOR", UNKNOWN)
PARTITIONS = ("train", "calibration", "confirmation")
FORBIDDEN_GRANTS = ("PUBLICLY_ELIGIBLE", "LIVE_ELIGIBLE")
HEX64 = re.compile(r"[0-9a-f]{64}")


class ContractRefusal(ValueError):
    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("REFUSED: " + "; ".join(self.problems[:8]))


# ------------------------------------------------------------------ digests
def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_obj(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def variable_id_for(dataset_id: str, name: str) -> str:
    return hashlib.sha256(f"{dataset_id}\0{name}".encode()).hexdigest()


def files_content_sha256(files: list) -> str:
    return sha_obj(sorted([f["name"], f["sha256"]] for f in files))


def chronological_partitions(n: int, fractions=(0.6, 0.2, 0.2)) -> dict:
    """Contiguous [start, end) index blocks in time order, frozen by n and
    the fractions alone."""
    if type(n) is not int or n < 3:
        raise ContractRefusal([f"partitions need an integer length >= 3, got {n!r}"])
    if len(fractions) != 3 or any(type(f) not in (int, float) or not f > 0 for f in fractions) \
            or abs(sum(fractions) - 1.0) > 1e-9:
        raise ContractRefusal(["partition fractions must be three positive numbers summing to 1"])
    a = int(math.floor(n * fractions[0]))
    b = int(math.floor(n * (fractions[0] + fractions[1])))
    return {"train": [0, a], "calibration": [a, b], "confirmation": [b, n]}


# ------------------------------------------------------------------- schema
NUMBER = "number"      # int or float, never bool, always finite
NUM_OR_WORD = "number_or_word"


def _enum(values):
    return ("enum", tuple(values))


def _list_of(spec):
    return ("list_of", spec)


EVIDENCE = {"source": str, "sha256": str}
FILE = {"name": str, "bytes": int, "sha256": str, "role": str}
DEPENDENCE = {"dataset_id": str, "relation": str, "modelled": bool}
VARIABLE = {
    "schema": str, "variable_id": str, "dataset_id": str, "name": str,
    "semantics": {"type": str, "description": str, "evidence": _list_of(EVIDENCE)},
    "unit": {"value": str, "evidence": _list_of(EVIDENCE)},
    "producer": {"kind": _enum(PRODUCER_KINDS), "reference": str},
    "physical_type": str,
    "frequency_nominal_seconds": NUM_OR_WORD,
    "event_time": str,
    "available_time_rule": _enum(AVAILABILITY_RULES),
    "missingness": {"encoding": str, "policy": str},
    "sentinels": {"values": list, "policy": str},
    "role": _enum(ROLES),
    "license_state": _enum(LICENSE_STATES),
    "original_fields": dict,
}
DATASET = {
    "schema": str, "dataset_id": str, "version": str, "bank": _enum(BANKS),
    "files": _list_of(FILE), "content_sha256": str,
    "source": {"provider": str, "official_url": str, "citation": str, "doi": str, "upstream_owner": str},
    "license": {"state": _enum(LICENSE_STATES), "id": str, "url": str, "text_sha256": str,
                "attribution_required": str, "redistribution": str, "derivatives": str,
                "evidence": _list_of(EVIDENCE)},
    "time": {"frequency_nominal_seconds": NUM_OR_WORD, "timezone": str,
             "timestamp_meaning": _enum(TIMESTAMP_MEANINGS), "range_start": str, "range_end": str,
             "availability_rule": _enum(AVAILABILITY_RULES), "availability_delay_seconds": NUM_OR_WORD},
    "panel": {"aligned_common_grid": bool, "n_series": int, "alignment_rule": str},
    "partitions": {"scheme": str, "fractions": {p: NUMBER for p in PARTITIONS},
                   "boundaries": {p: list for p in PARTITIONS},
                   "sealed_periods_excluded": list, "frozen_before_profile": bool},
    "dependence": _list_of(DEPENDENCE),
    "variables": _list_of(VARIABLE),
    "original_fields": dict,
    "contract_sha256": str,
}


def _is_number(x) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def _check(node, spec, path, problems):
    if isinstance(spec, dict):
        if not isinstance(node, dict):
            problems.append(f"{path}: expected an object")
            return
        if set(node) != set(spec):
            problems.append(f"{path}: keys differ (missing {sorted(set(spec) - set(node))}, "
                            f"extra {sorted(set(node) - set(spec))})")
        for k, sub in spec.items():
            if k in node:
                _check(node[k], sub, f"{path}.{k}", problems)
    elif isinstance(spec, tuple) and spec[0] == "enum":
        if node not in spec[1]:
            problems.append(f"{path}: {node!r} is not one of {list(spec[1])}")
    elif isinstance(spec, tuple) and spec[0] == "list_of":
        if not isinstance(node, list):
            problems.append(f"{path}: expected a list")
            return
        for i, x in enumerate(node):
            _check(x, spec[1], f"{path}[{i}]", problems)
    elif spec == NUMBER:
        if not _is_number(node):
            problems.append(f"{path}: expected a finite number, got {node!r}")
    elif spec == NUM_OR_WORD:
        if not (_is_number(node) or node in FREQUENCY_WORDS):
            problems.append(f"{path}: expected a finite number or one of {list(FREQUENCY_WORDS)}")
    elif spec is bool:
        if type(node) is not bool:
            problems.append(f"{path}: expected a boolean")
    elif spec is int:
        if type(node) is not int:
            problems.append(f"{path}: expected an integer")
    elif not isinstance(node, spec) or type(node) is bool:
        problems.append(f"{path}: expected {spec.__name__}")


def _walk_strings(node, path=""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield f"{path}.{k}", str(k)
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_strings(v, f"{path}[{i}]")


def _walk_floats(node, path=""):
    if type(node) is float:
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_floats(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_floats(v, f"{path}[{i}]")


def validate(doc) -> list[str]:
    p: list[str] = []
    _check(doc, DATASET, "dataset", p)
    if p:
        return p
    for path, s in _walk_strings(doc):
        if s in FORBIDDEN_GRANTS:
            p.append(f"{path}: a contract never grants {s}")
        if str(Path.home()) in s or s.startswith("/home/"):
            p.append(f"{path}: absolute home path")
    p += [f"{path}: non-finite number" for path, x in _walk_floats(doc) if not math.isfinite(x)]
    if doc["schema"] != DATASET_SCHEMA:
        p.append("dataset.schema")
    if not doc["dataset_id"].strip():
        p.append("dataset.dataset_id is empty")

    for f in doc["files"]:
        if not HEX64.fullmatch(f["sha256"]) or f["bytes"] < 0:
            p.append(f"file {f['name']!r}: sha256 must be hex64 and bytes non-negative")
    if len({f["name"] for f in doc["files"]}) != len(doc["files"]):
        p.append("duplicate file names")
    if doc["bank"] != "SYNTHETIC" and not doc["files"]:
        p.append("a non-synthetic dataset must bind its files")
    if doc["content_sha256"] != files_content_sha256(doc["files"]):
        p.append("content_sha256 does not re-derive from the files")

    lic = doc["license"]
    if lic["state"] in LICENSE_STATES_NEEDING_EVIDENCE and not lic["evidence"]:
        p.append(f"license state {lic['state']} needs evidence")
    for e in lic["evidence"]:
        if not (HEX64.fullmatch(e["sha256"]) or e["sha256"] == "UNAVAILABLE"):
            p.append("license evidence sha256 must be hex64 or UNAVAILABLE")
    if doc["bank"] == "SYNTHETIC" and lic["state"] != "NOT_APPLICABLE_GENERATED":
        p.append("a synthetic dataset's license state is NOT_APPLICABLE_GENERATED")
    if doc["bank"] != "SYNTHETIC" and lic["state"] == "NOT_APPLICABLE_GENERATED":
        p.append("only generated data may be NOT_APPLICABLE_GENERATED")

    part = doc["partitions"]
    fr = [part["fractions"][k] for k in PARTITIONS]
    if any(not f > 0 for f in fr) or abs(sum(fr) - 1.0) > 1e-9:
        p.append("partition fractions must be positive and sum to 1")
    bounds = [part["boundaries"][k] for k in PARTITIONS]
    if any(len(b) != 2 or any(type(x) is not int for x in b) or b[0] >= b[1] for b in bounds) \
            or bounds[0][0] != 0 or bounds[0][1] != bounds[1][0] or bounds[1][1] != bounds[2][0]:
        p.append("partition boundaries must be contiguous, ordered, non-empty [start, end) integer blocks")
    if part["frozen_before_profile"] is not True:
        p.append("partitions must be frozen before any profile")

    ids, names = set(), set()
    for v in doc["variables"]:
        if v["schema"] != VARIABLE_SCHEMA:
            p.append(f"variable {v['name']!r}: schema")
        if v["dataset_id"] != doc["dataset_id"]:
            p.append(f"variable {v['name']!r}: belongs to another dataset")
        if v["variable_id"] != variable_id_for(doc["dataset_id"], v["name"]):
            p.append(f"variable {v['name']!r}: variable_id does not re-derive")
        if v["variable_id"] in ids or v["name"] in names:
            p.append(f"variable {v['name']!r}: duplicate")
        ids.add(v["variable_id"]); names.add(v["name"])
        for e in v["semantics"]["evidence"] + v["unit"]["evidence"]:
            if not (HEX64.fullmatch(e["sha256"]) or e["sha256"] == "UNAVAILABLE"):
                p.append(f"variable {v['name']!r}: evidence sha256 must be hex64 or UNAVAILABLE")
        if v["unit"]["value"] != UNKNOWN and not v["unit"]["evidence"]:
            p.append(f"variable {v['name']!r}: a declared unit needs evidence")
        if v["semantics"]["type"] != UNKNOWN and not v["semantics"]["evidence"]:
            p.append(f"variable {v['name']!r}: declared semantics need evidence")
        if doc["bank"] == "SYNTHETIC" and v["license_state"] != "NOT_APPLICABLE_GENERATED":
            p.append(f"variable {v['name']!r}: synthetic license state")
    if doc["panel"]["n_series"] < 0:
        p.append("panel.n_series must be non-negative")

    if doc["contract_sha256"] != sha_obj({k: x for k, x in doc.items() if k != "contract_sha256"}):
        p.append("contract_sha256 does not re-derive")
    return p


def seal(doc: dict) -> dict:
    """Fill the two derived digests, then refuse unless the result validates."""
    doc = dict(doc)
    doc["content_sha256"] = ""
    doc["contract_sha256"] = ""
    # The exact schema is checked BEFORE any digest, so a malformed field
    # (a NaN, a boolean as a number) refuses with its path instead of failing
    # inside strict JSON serialization.
    structural: list[str] = []
    _check(doc, DATASET, "dataset", structural)
    if structural:
        raise ContractRefusal(structural)
    doc["content_sha256"] = files_content_sha256(doc["files"])
    body = {k: v for k, v in doc.items() if k != "contract_sha256"}
    try:
        doc["contract_sha256"] = sha_obj(body)
    except (TypeError, ValueError) as exc:
        raise ContractRefusal([f"not serializable as strict JSON: {exc}"]) from exc
    problems = validate(doc)
    if problems:
        raise ContractRefusal(problems)
    return doc


def variable(dataset_id: str, name: str, **fields) -> dict:
    """A variable contract with UNKNOWN for everything not supplied."""
    v = {
        "schema": VARIABLE_SCHEMA, "variable_id": variable_id_for(dataset_id, name),
        "dataset_id": dataset_id, "name": name,
        "semantics": {"type": UNKNOWN, "description": UNKNOWN, "evidence": []},
        "unit": {"value": UNKNOWN, "evidence": []},
        "producer": {"kind": UNKNOWN, "reference": UNKNOWN},
        "physical_type": UNKNOWN, "frequency_nominal_seconds": UNKNOWN, "event_time": UNKNOWN,
        "available_time_rule": UNKNOWN, "missingness": {"encoding": UNKNOWN, "policy": UNKNOWN},
        "sentinels": {"values": [], "policy": UNKNOWN}, "role": UNKNOWN, "license_state": UNKNOWN,
        "original_fields": {},
    }
    for k, x in fields.items():
        if k not in v:
            raise ContractRefusal([f"variable field {k!r} is not in the contract"])
        v[k] = x
    return v
