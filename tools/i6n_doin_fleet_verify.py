#!/usr/bin/env python3
"""Verify that exactly four DOIN nodes expose one identical run and chain.

The input is a private deployment spec.  This module never writes that spec,
starts a service, or includes node URLs in its report.  It performs two
read-only HTTP GETs per node: ``/chain/status`` and ``/status``.

Spec schema::

    {
      "schema": "i6n.doin_fleet_verify.spec.v1",
      "expected": {
        "seed": 6102026,
        "config_digest": "<64 lowercase hexadecimal characters>",
        "component_versions": {
          "doin-node": "<exact revision>",
          "doin-core": "<exact revision>",
          "doin-plugins": "<exact revision>",
          "predictor": "<exact revision>"
        }
      },
      "nodes": [
        {"node_id": "<non-secret label>", "base_url": "<private URL>"}
      ]
    }

The four URLs belong only in the private spec.  Runtime responses must expose
``seed`` and ``config_digest`` from ``/status``, either at the root or below
``optimization``.  ``/chain/status`` must expose ``network_id``,
``genesis_hash``, component versions, height, tip, and finalized height.  A
finalized hash is compared when the endpoint exposes it.  If ``/status`` also
publishes network or run identity, it must agree with the canonical fields.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SPEC_SCHEMA = "i6n.doin_fleet_verify.spec.v1"
STATUS_SCHEMA = "i6n.doin_fleet_verify.status.v1"
COMPONENTS = ("doin-node", "doin-core", "doin-plugins", "predictor")
MAX_RESPONSE_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FleetSpecError(ValueError):
    """Raised when the private fleet specification is incomplete or unsafe."""


FetchJSON = Callable[[str, float], Mapping[str, Any]]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FleetSpecError(f"{label} must be a JSON object")
    return value


def _validated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_object(spec, "spec")
    if root.get("schema") != SPEC_SCHEMA:
        raise FleetSpecError(f"schema must equal {SPEC_SCHEMA!r}")

    expected = _require_object(root.get("expected"), "expected")
    seed = expected.get("seed")
    if not _is_int(seed) or not 0 <= seed <= 0xFFFFFFFF:
        raise FleetSpecError("expected.seed must be a uint32 integer")

    config_digest = expected.get("config_digest")
    if not isinstance(config_digest, str) or not _SHA256_RE.fullmatch(config_digest):
        raise FleetSpecError(
            "expected.config_digest must be 64 lowercase hexadecimal characters"
        )

    versions = _require_object(
        expected.get("component_versions"), "expected.component_versions"
    )
    if set(versions) != set(COMPONENTS):
        raise FleetSpecError(
            "expected.component_versions must contain exactly "
            + ", ".join(COMPONENTS)
        )
    for component in COMPONENTS:
        version = versions[component]
        if not isinstance(version, str) or not version.strip():
            raise FleetSpecError(
                f"expected.component_versions.{component} must be a non-empty string"
            )

    nodes = root.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise FleetSpecError("spec must declare exactly four nodes")

    clean_nodes: list[dict[str, str]] = []
    node_ids: set[str] = set()
    base_urls: set[str] = set()
    for index, raw_node in enumerate(nodes):
        node = _require_object(raw_node, f"nodes[{index}]")
        node_id = node.get("node_id")
        base_url = node.get("base_url")
        if not isinstance(node_id, str) or not node_id.strip():
            raise FleetSpecError(f"nodes[{index}].node_id must be a non-empty string")
        if node_id in node_ids:
            raise FleetSpecError(f"duplicate node_id: {node_id}")
        if not isinstance(base_url, str) or not base_url.strip():
            raise FleetSpecError(f"nodes[{index}].base_url must be a non-empty URL")

        normalized_url = base_url.rstrip("/")
        parsed = urlsplit(normalized_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise FleetSpecError(
                f"nodes[{index}].base_url must be an HTTP(S) origin without "
                "credentials, query, or fragment"
            )
        if normalized_url in base_urls:
            raise FleetSpecError("duplicate node base_url")
        node_ids.add(node_id)
        base_urls.add(normalized_url)
        clean_nodes.append({"node_id": node_id, "base_url": normalized_url})

    return {
        "schema": SPEC_SCHEMA,
        "expected": {
            "seed": seed,
            "config_digest": config_digest,
            "component_versions": {key: versions[key] for key in COMPONENTS},
        },
        "nodes": clean_nodes,
    }


def load_spec(path: Path) -> dict[str, Any]:
    """Read and validate a private JSON fleet specification."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FleetSpecError(f"cannot read private spec: {type(exc).__name__}") from exc
    return _validated_spec(payload)


def _fetch_json(url: str, timeout: float) -> Mapping[str, Any]:
    """Read one bounded JSON object using urllib and no ambient credentials."""

    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if not 200 <= status < 300:
            raise OSError(f"HTTP status {status}")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds size limit")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("response root is not an object")
    return payload


def _error(
    code: str,
    node_id: str,
    field: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "node_id": node_id, "field": field}
    if expected is not None:
        item["expected"] = expected
    if observed is not None:
        item["observed"] = observed
    return item


def _required_string(
    payload: Mapping[str, Any],
    field: str,
    node_id: str,
    errors: list[dict[str, Any]],
) -> str | None:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        errors.append(_error("MISSING_FIELD", node_id, field))
        return None
    return value


def _required_nonnegative_int(
    payload: Mapping[str, Any],
    field: str,
    node_id: str,
    errors: list[dict[str, Any]],
) -> int | None:
    value = payload.get(field)
    if not _is_int(value) or value < 0:
        errors.append(_error("MISSING_FIELD", node_id, field))
        return None
    return value


def _optional_string(
    payload: Mapping[str, Any],
    field: str,
    node_id: str,
    errors: list[dict[str, Any]],
) -> str | None:
    if field not in payload:
        return None
    value = payload[field]
    if not isinstance(value, str) or not value:
        errors.append(_error("MISSING_FIELD", node_id, field))
        return None
    return value


def _run_identity(
    status: Mapping[str, Any],
    node_id: str,
    errors: list[dict[str, Any]],
) -> tuple[object | None, object | None]:
    """Read root or nested run identity and reject contradictory duplicates."""

    nested_value = status.get("optimization")
    if nested_value is not None and not isinstance(nested_value, Mapping):
        errors.append(_error("MISSING_FIELD", node_id, "status.optimization"))
        nested: Mapping[str, Any] = {}
    elif isinstance(nested_value, Mapping):
        nested = nested_value
    else:
        nested = {}

    seed = nested.get("seed") if "seed" in nested else status.get("seed")
    config_digest = (
        nested.get("config_digest")
        if "config_digest" in nested
        else status.get("config_digest")
    )

    for field, canonical in (("seed", seed), ("config_digest", config_digest)):
        if field in nested and field in status and nested[field] != status[field]:
            errors.append(
                _error(
                    "ENDPOINT_IDENTITY_MISMATCH",
                    node_id,
                    field,
                    expected=canonical,
                    observed=status[field],
                )
            )
    return seed, config_digest


def _read_node(
    node: Mapping[str, str],
    expected: Mapping[str, Any],
    fetch_json: FetchJSON,
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_id = node["node_id"]
    base_url = node["base_url"]
    endpoint_payloads: dict[str, Mapping[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    for endpoint in ("chain/status", "status"):
        try:
            payload = fetch_json(f"{base_url}/{endpoint}", timeout)
            if not isinstance(payload, Mapping):
                raise ValueError("response root is not an object")
            endpoint_payloads[endpoint] = payload
        except Exception as exc:  # one bad endpoint refuses this node, but others are queried
            errors.append(
                _error(
                    "NODE_UNREACHABLE",
                    node_id,
                    f"/{endpoint}",
                    observed=type(exc).__name__,
                )
            )

    record: dict[str, Any] = {
        "node_id": node_id,
        "reachable": len(endpoint_payloads) == 2,
        "status": "REFUSED",
        "errors": errors,
    }
    if len(endpoint_payloads) != 2:
        return record, errors

    chain = endpoint_payloads["chain/status"]
    status = endpoint_payloads["status"]
    seed, config_digest = _run_identity(status, node_id, errors)
    if not _is_int(seed):
        errors.append(_error("MISSING_FIELD", node_id, "status.seed"))
    elif seed != expected["seed"]:
        errors.append(
            _error(
                "SEED_MISMATCH",
                node_id,
                "status.seed",
                expected=expected["seed"],
                observed=seed,
            )
        )

    if not isinstance(config_digest, str) or not config_digest:
        errors.append(_error("MISSING_FIELD", node_id, "status.config_digest"))
    elif config_digest != expected["config_digest"]:
        errors.append(
            _error(
                "CONFIG_DIGEST_MISMATCH",
                node_id,
                "status.config_digest",
                expected=expected["config_digest"],
                observed=config_digest,
            )
        )

    versions = chain.get("component_versions")
    if not isinstance(versions, Mapping):
        errors.append(
            _error("MISSING_FIELD", node_id, "chain.component_versions")
        )
        versions = {}
    for component in COMPONENTS:
        observed = versions.get(component)
        wanted = expected["component_versions"][component]
        if observed is None:
            errors.append(
                _error(
                    "MISSING_FIELD",
                    node_id,
                    f"chain.component_versions.{component}",
                )
            )
        elif observed != wanted:
            errors.append(
                _error(
                    "COMPONENT_VERSION_MISMATCH",
                    node_id,
                    f"chain.component_versions.{component}",
                    expected=wanted,
                    observed=observed,
                )
            )

    network_id = _required_string(chain, "network_id", node_id, errors)
    genesis_hash = _required_string(chain, "genesis_hash", node_id, errors)
    status_network = _optional_string(status, "network_id", node_id, errors)
    status_genesis = _optional_string(status, "genesis_hash", node_id, errors)
    if network_id is not None and status_network is not None and network_id != status_network:
        errors.append(
            _error(
                "ENDPOINT_IDENTITY_MISMATCH",
                node_id,
                "network_id",
                expected=network_id,
                observed=status_network,
            )
        )
    if genesis_hash is not None and status_genesis is not None and genesis_hash != status_genesis:
        errors.append(
            _error(
                "ENDPOINT_IDENTITY_MISMATCH",
                node_id,
                "genesis_hash",
                expected=genesis_hash,
                observed=status_genesis,
            )
        )

    chain_height = _required_nonnegative_int(chain, "chain_height", node_id, errors)
    tip_hash = _required_string(chain, "tip_hash", node_id, errors)
    finalized_height = _required_nonnegative_int(
        chain, "finalized_height", node_id, errors
    )
    finalized_hash = _optional_string(chain, "finalized_hash", node_id, errors)
    if (
        chain_height is not None
        and finalized_height is not None
        and finalized_height > chain_height
    ):
        errors.append(
            _error(
                "INVALID_CHAIN_STATE",
                node_id,
                "finalized_height",
                expected=f"<= {chain_height}",
                observed=finalized_height,
            )
        )

    finalized_checkpoint: dict[str, Any] = {"height": finalized_height}
    if finalized_hash is not None:
        finalized_checkpoint["hash"] = finalized_hash

    record.update(
        {
            "seed": seed,
            "config_digest": config_digest,
            "component_versions": {
                component: versions.get(component) for component in COMPONENTS
            },
            "network_id": network_id,
            "genesis_hash": genesis_hash,
            "chain_height": chain_height,
            "tip_hash": tip_hash,
            "finalized_checkpoint": finalized_checkpoint,
        }
    )
    if not errors:
        record["status"] = "MATCH"
    return record, errors


def _compare_fleet(
    records: Sequence[dict[str, Any]], errors: list[dict[str, Any]]
) -> dict[str, Any]:
    complete = [record for record in records if record["reachable"]]
    fleet: dict[str, Any] = {
        "expected_nodes": 4,
        "responding_nodes": len(complete),
    }
    if not complete:
        return fleet

    baseline = complete[0]
    comparisons = (
        ("network_id", "NETWORK_MISMATCH"),
        ("genesis_hash", "GENESIS_MISMATCH"),
        ("chain_height", "CHAIN_HEIGHT_MISMATCH"),
        ("tip_hash", "TIP_HASH_MISMATCH"),
        ("finalized_checkpoint", "FINALIZED_CHECKPOINT_MISMATCH"),
    )
    for record in complete[1:]:
        for field, code in comparisons:
            if record.get(field) != baseline.get(field):
                error = _error(
                    code,
                    record["node_id"],
                    field,
                    expected=baseline.get(field),
                    observed=record.get(field),
                )
                errors.append(error)
                record["errors"].append(error)
                record["status"] = "REFUSED"

    fleet.update(
        {
            "network_id": baseline.get("network_id"),
            "genesis_hash": baseline.get("genesis_hash"),
            "chain_height": baseline.get("chain_height"),
            "tip_hash": baseline.get("tip_hash"),
            "finalized_checkpoint": baseline.get("finalized_checkpoint"),
        }
    )
    return fleet


def verify_fleet(
    spec: Mapping[str, Any],
    *,
    fetch_json: FetchJSON = _fetch_json,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Return a fail-closed, machine-readable fleet verification report."""

    validated = _validated_spec(spec)
    if timeout <= 0:
        raise FleetSpecError("timeout must be greater than zero")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for node in validated["nodes"]:
        record, node_errors = _read_node(
            node,
            validated["expected"],
            fetch_json,
            timeout,
        )
        records.append(record)
        errors.extend(node_errors)

    fleet = _compare_fleet(records, errors)
    ready = not errors and fleet["responding_nodes"] == 4
    return {
        "schema": STATUS_SCHEMA,
        "status": "READY" if ready else "REFUSED",
        "ready": ready,
        "expected": validated["expected"],
        "fleet": fleet,
        "nodes": records,
        "errors": errors,
    }


def _invalid_spec_report(detail: str) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "status": "REFUSED",
        "ready": False,
        "errors": [{"code": "INVALID_SPEC", "detail": detail}],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify seed, config, versions, and chain consensus for four DOIN nodes."
    )
    parser.add_argument("--spec", required=True, type=Path, help="private JSON spec")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds per request")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fetch_json: FetchJSON = _fetch_json,
) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        report = verify_fleet(spec, fetch_json=fetch_json, timeout=args.timeout)
    except FleetSpecError as exc:
        report = _invalid_spec_report(str(exc))
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2

    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
