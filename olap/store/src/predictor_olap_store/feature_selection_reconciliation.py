"""Validation and canonical responses for phase-1 warehouse reconciliation."""

from __future__ import annotations

import json

from .feature_selection import HEX64, digest, without_digest


REQUEST_SCHEMA = "phase1.warehouse_reconciliation_request.v1"
RESPONSE_SCHEMA = "phase1.warehouse_reconciliation.v1"
IDENTITY_FIELDS = {
    "feature_id",
    "feature_key",
    "terminal_state",
    "terminal_sha256",
    "envelope_sha256",
    "warehouse_receipt_sha256",
}
TERMINAL_STATES = {"COMPLETED", "UNAVAILABLE"}


def _text(value: dict, key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return item


def _sha(value: dict, key: str, path: str, *, nullable: bool = False) -> str | None:
    item = value.get(key)
    if nullable and item is None:
        return None
    if not isinstance(item, str) or not HEX64.fullmatch(item):
        suffix = " or null" if nullable else ""
        raise ValueError(f"{path}.{key} must be a lowercase SHA-256 digest{suffix}")
    return item


def validate_reconciliation_request(document: dict) -> dict:
    """Return a canonical defensive copy of an authenticated readback request."""
    if not isinstance(document, dict):
        raise ValueError("reconciliation request must be a JSON object")
    allowed = {
        "schema", "population_id", "plan_sha256", "expected_count",
        "identities_sha256", "identities", "authentication_profile", "request_sha256",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"reconciliation request contains unknown fields: {unknown}")
    if document.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"schema must be {REQUEST_SCHEMA}")
    _text(document, "population_id", "request")
    _sha(document, "plan_sha256", "request")
    _text(document, "authentication_profile", "request")

    identities = document.get("identities")
    if not isinstance(identities, list):
        raise ValueError("request.identities must be a list")
    expected_count = document.get("expected_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("request.expected_count must be an integer")
    if expected_count <= 0:
        raise ValueError("request.expected_count must be positive")
    if expected_count != len(identities):
        raise ValueError("request.expected_count differs from the identity population")

    seen_features: set[str] = set()
    seen_keys: set[str] = set()
    seen_terminals: set[str] = set()
    seen_envelopes: set[str] = set()
    for index, identity in enumerate(identities):
        path = f"request.identities[{index}]"
        if not isinstance(identity, dict):
            raise ValueError(f"{path} must be an object")
        unknown_identity = sorted(set(identity) - IDENTITY_FIELDS)
        missing_identity = sorted(IDENTITY_FIELDS - set(identity))
        if unknown_identity or missing_identity:
            raise ValueError(
                f"{path} has unknown fields {unknown_identity} and missing fields {missing_identity}"
            )
        feature_id = _text(identity, "feature_id", path)
        feature_key = _text(identity, "feature_key", path)
        state = _text(identity, "terminal_state", path)
        if state not in TERMINAL_STATES:
            raise ValueError(f"{path}.terminal_state is not recognized")
        terminal = _sha(identity, "terminal_sha256", path)
        envelope = _sha(identity, "envelope_sha256", path, nullable=True)
        receipt = _sha(identity, "warehouse_receipt_sha256", path, nullable=True)
        if state == "COMPLETED" and (envelope is None or receipt is None):
            raise ValueError(f"{path} COMPLETED identity requires envelope and receipt digests")
        if state == "UNAVAILABLE" and (envelope is not None or receipt is not None):
            raise ValueError(f"{path} UNAVAILABLE identity cannot claim warehouse payloads")
        if (
            feature_id in seen_features
            or feature_key in seen_keys
            or terminal in seen_terminals
            or (envelope is not None and envelope in seen_envelopes)
        ):
            raise ValueError("request.identities contains duplicates")
        seen_features.add(feature_id)
        seen_keys.add(feature_key)
        seen_terminals.add(terminal)
        if envelope is not None:
            seen_envelopes.add(envelope)

    claimed_identities = _sha(document, "identities_sha256", "request")
    if claimed_identities != digest(identities):
        raise ValueError("request.identities_sha256 does not cover identities")
    claimed_request = _sha(document, "request_sha256", "request")
    if claimed_request != digest(without_digest(document, "request_sha256")):
        raise ValueError("request.request_sha256 does not cover the request")
    return json.loads(json.dumps(document, sort_keys=True, separators=(",", ":")))


def reconciliation_response(request: dict, observed_identities: list[dict]) -> dict:
    """Build the canonical successful response after backend verification."""
    response = {
        "schema": RESPONSE_SCHEMA,
        "state": "RECONCILED",
        "population_id": request["population_id"],
        "plan_sha256": request["plan_sha256"],
        "expected_count": request["expected_count"],
        "identities_sha256": request["identities_sha256"],
        "authentication_profile": request["authentication_profile"],
        "request_sha256": request["request_sha256"],
        "observed_count": len(observed_identities),
        "observed_identities": observed_identities,
        "observed_identities_sha256": digest(observed_identities),
        "complete": len(observed_identities) == request["expected_count"],
        "contradictions": [],
    }
    response["reconciliation_sha256"] = digest(response)
    return response
