"""Contract validation for phase-1 feature-selection envelopes.

The host validates transport-level invariants. Storage engines remain responsible for
performing the resulting inserts in one transaction and enforcing immutable identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime


SCHEMA_VERSION = "feature_selection_envelope.v1"
ROW_FAMILIES = (
    "sampling_quality",
    "variable_profiles",
    "information_metrics",
    "pair_relations",
    "causal_evidence",
    "selection_decisions",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FINAL_DECISIONS = {"SELECTED", "REJECTED", "NEUTRAL", "UNAVAILABLE"}
CAUSAL_STATES = {
    "IDENTIFIED", "NOT_IDENTIFIED", "SUPPORTED", "NOT_SUPPORTED",
    "INCONCLUSIVE", "UNAVAILABLE", "FAILED",
}
METRIC_STATES = {
    "MEASURED", "INCONCLUSIVE", "UNAVAILABLE", "FAILED", "NOT_APPLICABLE",
    "NOT_IDENTIFIED", "SUPPORTED", "NOT_SUPPORTED",
}


IDENTITY_FIELDS = {
    "sampling_quality": ("feature_id", "split", "metric_name", "population_id", "fold"),
    "variable_profiles": ("feature_id", "split", "metric_name", "population_id", "fold"),
    "information_metrics": (
        "feature_id", "target_id", "horizon", "split", "metric_name", "population_id", "fold",
    ),
    "pair_relations": (
        "feature_id", "target_id", "horizon", "split", "lag", "metric_name",
        "population_id", "fold",
    ),
    "causal_evidence": (
        "feature_id", "target_id", "horizon", "split", "rung", "estimand", "estimator",
        "population_id", "fold",
    ),
    "selection_decisions": ("feature_id", "target_id", "horizon", "method", "population_id"),
}
ALLOWED_FIELDS = {
    "sampling_quality": {
        "feature_id", "split", "metric_name", "metric_value", "state", "unit",
        "population_id", "fold", "row_sha256",
    },
    "variable_profiles": {
        "feature_id", "split", "metric_name", "metric_value", "state", "unit",
        "population_id", "fold", "row_sha256",
    },
    "information_metrics": {
        "feature_id", "target_id", "horizon", "split", "metric_name", "metric_value",
        "state", "population_id", "fold", "row_sha256",
    },
    "pair_relations": {
        "feature_id", "target_id", "horizon", "split", "lag", "metric_name",
        "metric_value", "state", "population_id", "fold", "row_sha256",
    },
    "causal_evidence": {
        "feature_id", "target_id", "horizon", "split", "rung", "estimand", "estimator",
        "state", "effect", "lower", "upper", "support_n", "assumptions",
        "adjustment_set", "evidence_sha256", "population_id", "fold", "row_sha256",
    },
    "selection_decisions": {
        "feature_id", "target_id", "horizon", "method", "score", "rank", "decision",
        "rule", "evidence_sha256", "population_id", "row_sha256",
    },
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def without_digest(value: dict, key: str) -> dict:
    return {name: item for name, item in value.items() if name != key}


def _required_text(value: dict, key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return item


def _optional_text(value: dict, key: str, path: str) -> None:
    if key in value and value[key] is not None and not isinstance(value[key], str):
        raise ValueError(f"{path}.{key} must be a string or null")


def _hex(value: dict, key: str, path: str) -> str:
    item = _required_text(value, key, path)
    if not HEX64.fullmatch(item):
        raise ValueError(f"{path}.{key} must be a lowercase SHA-256 digest")
    return item


def _finite(value: dict, key: str, path: str, *, required: bool = False) -> None:
    item = value.get(key)
    if item is None and not required:
        return
    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
        raise ValueError(f"{path}.{key} must be a finite number")


def _integer(value: dict, key: str, path: str, *, minimum: int | None = None) -> None:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{path}.{key} must be an integer")
    if minimum is not None and item < minimum:
        raise ValueError(f"{path}.{key} must be >= {minimum}")


def _base_metric(row: dict, path: str) -> None:
    _required_text(row, "feature_id", path)
    _required_text(row, "split", path)
    _required_text(row, "metric_name", path)
    state = _required_text(row, "state", path)
    if state not in METRIC_STATES:
        raise ValueError(f"{path}.state is not recognized")
    _finite(row, "metric_value", path, required=state == "MEASURED")
    for key in ("unit", "population_id", "fold"):
        _optional_text(row, key, path)


def _target_metric(row: dict, path: str) -> None:
    _base_metric(row, path)
    _required_text(row, "target_id", path)
    _integer(row, "horizon", path, minimum=1)


def _validate_row(family: str, row: dict, path: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{path} must be an object")
    unknown = sorted(set(row) - ALLOWED_FIELDS[family])
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {unknown}")
    if family in {"sampling_quality", "variable_profiles"}:
        _base_metric(row, path)
    elif family == "information_metrics":
        _target_metric(row, path)
    elif family == "pair_relations":
        _target_metric(row, path)
        _integer(row, "lag", path)
    elif family == "causal_evidence":
        _required_text(row, "feature_id", path)
        _required_text(row, "target_id", path)
        _required_text(row, "split", path)
        _integer(row, "horizon", path, minimum=1)
        _integer(row, "rung", path, minimum=1)
        if row["rung"] > 3:
            raise ValueError(f"{path}.rung must be one of 1, 2 or 3")
        _required_text(row, "estimand", path)
        _required_text(row, "estimator", path)
        state = _required_text(row, "state", path)
        if state not in CAUSAL_STATES:
            raise ValueError(f"{path}.state is not a recognized causal state")
        for key in ("effect", "lower", "upper"):
            _finite(row, key, path)
        _integer(row, "support_n", path, minimum=0)
        for key in ("assumptions", "adjustment_set"):
            if not isinstance(row.get(key), list) or not all(isinstance(x, str) for x in row[key]):
                raise ValueError(f"{path}.{key} must be a list of strings")
        _hex(row, "evidence_sha256", path)
        for key in ("population_id", "fold"):
            _optional_text(row, key, path)
    else:
        _required_text(row, "feature_id", path)
        _required_text(row, "target_id", path)
        _integer(row, "horizon", path, minimum=1)
        _required_text(row, "method", path)
        decision = _required_text(row, "decision", path)
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"{path}.decision must be a final selection decision")
        _finite(row, "score", path)
        if row.get("rank") is not None:
            _integer(row, "rank", path, minimum=1)
        _required_text(row, "rule", path)
        _hex(row, "evidence_sha256", path)
        _optional_text(row, "population_id", path)

    claimed = _hex(row, "row_sha256", path)
    actual = digest(without_digest(row, "row_sha256"))
    if claimed != actual:
        raise ValueError(f"{path}.row_sha256 does not cover the stored row")


def row_identity(run_id: str, family: str, row: dict) -> str:
    identity = {"run_id": run_id, "family": family}
    identity.update({key: row.get(key) for key in IDENTITY_FIELDS[family]})
    return digest(identity)


def validate_envelope(document: dict) -> dict:
    """Return a defensive copy after validating every byte-addressed identity."""
    if not isinstance(document, dict):
        raise ValueError("document must be a JSON object")
    unknown_document = sorted(set(document) - {"schema_version", "run", "rows",
                                                "envelope_sha256"})
    if unknown_document:
        raise ValueError(f"document contains unknown fields: {unknown_document}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    run = document.get("run")
    if not isinstance(run, dict):
        raise ValueError("run must be an object")
    unknown_run = sorted(set(run) - {"run_id", "campaign_sha256", "code_sha256",
                                     "input_sha256", "inventory_sha256", "created_at"})
    if unknown_run:
        raise ValueError(f"run contains unknown fields: {unknown_run}")
    _required_text(run, "run_id", "run")
    for key in ("campaign_sha256", "code_sha256", "input_sha256", "inventory_sha256"):
        _hex(run, key, "run")
    created_at = _required_text(run, "created_at", "run")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("run.created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("run.created_at must include a timezone")

    rows = document.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("rows must be an object")
    unknown = sorted(set(rows) - set(ROW_FAMILIES))
    if unknown:
        raise ValueError(f"rows contains unknown families: {unknown}")
    for family in ROW_FAMILIES:
        values = rows.get(family)
        if not isinstance(values, list):
            raise ValueError(f"rows.{family} must be a list")
        seen = set()
        for index, row in enumerate(values):
            path = f"rows.{family}[{index}]"
            _validate_row(family, row, path)
            identity = row_identity(run["run_id"], family, row)
            if identity in seen:
                raise ValueError(f"{path} duplicates an immutable row identity")
            seen.add(identity)

    claimed = _hex(document, "envelope_sha256", "document")
    actual = digest(without_digest(document, "envelope_sha256"))
    if claimed != actual:
        raise ValueError("document.envelope_sha256 does not cover the envelope")
    return json.loads(canonical_json(document))
