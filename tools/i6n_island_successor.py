#!/usr/bin/env python3
"""Durable, unattended successor for the governed four-island H4 -> H72 run.

The tool is intentionally a coordinator, not an optimizer.  It consumes four
sealed H4 terminal receipts and one externally produced archive/OLAP closure,
then invokes an idempotent H72 launcher with a stable transition identity.  A
crash after launch is recovered by adopting the launch receipt on the next
timer tick.

Private paths, commands, environment values and host identities live only in
the JSON spec passed with ``--spec``.  ``--status`` is strictly read-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from tools.i6n_doin_fleet_verify import (
    FleetSpecError,
    _fetch_json as _fleet_fetch_json,
    verify_fleet,
)


SPEC_SCHEMA = "i6n.island_successor.spec.v1"
TERMINAL_SCHEMA = "i6n.island.terminal.v1"
CLOSURE_SCHEMA = "i6n.island.archive_closure.v2"
START_SCHEMA = "i6n.island.start_receipt.v1"
INTENT_SCHEMA = "i6n.island.transition_intent.v1"
RECEIPT_SCHEMA = "i6n.island.transition_receipt.v1"
STATUS_SCHEMA = "i6n.island_successor.status.v1"
COMPONENTS = ("doin-node", "doin-core", "doin-plugins", "predictor")
REPO_ROOT = Path(__file__).resolve().parents[1]


class SpecError(ValueError):
    """The private successor specification is malformed or unsafe."""


class Refusal(RuntimeError):
    """Existing evidence contradicts the governed transition contract."""


class FleetPending(RuntimeError):
    """The launched H72 fleet is not fully observable yet; a later tick may retry."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _nonempty_string(value: Any, field: str, error=SpecError) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{field} must be a nonempty string")
    return value


def _sha256_string(value: Any, field: str) -> str:
    value = _nonempty_string(value, field, Refusal)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Refusal(f"INVALID_TERMINAL_FIELD: {field}")
    return value


def _absolute_path(value: Any, field: str) -> Path:
    text = _nonempty_string(value, field)
    path = Path(text)
    if not path.is_absolute():
        raise SpecError(f"{field} must be an absolute path")
    return path


def _exact_components(value: Any, field: str, *, expected: Mapping[str, str] | None = None) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(COMPONENTS):
        raise Refusal(f"COMPONENT_VERSION_MISMATCH: {field} must name exactly {list(COMPONENTS)}")
    result = {name: _nonempty_string(value[name], f"{field}.{name}", Refusal) for name in COMPONENTS}
    if expected is not None and result != dict(expected):
        raise Refusal(f"COMPONENT_VERSION_MISMATCH: {field}")
    return result


def _atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    """Replace one JSON file durably, including its containing directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refusal(f"INVALID_JSON: {field}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise Refusal(f"INVALID_JSON: {field} root must be an object")
    return value


def _read_envelope(path: Path, schema: str, field: str) -> tuple[dict[str, Any], str]:
    document = _read_json(path, field)
    if document.get("schema") != schema or set(document) != {"schema", "payload", "payload_sha256"}:
        raise Refusal(f"INVALID_ENVELOPE: {field}")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise Refusal(f"INVALID_ENVELOPE: {field}.payload")
    observed = document.get("payload_sha256")
    actual = _digest(payload)
    if observed != actual:
        raise Refusal(f"DIGEST_MISMATCH: {field}")
    return payload, actual


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate the private spec without touching runtime state."""
    path = Path(path)
    if _inside(path, REPO_ROOT):
        raise SpecError("the private successor spec must be outside the repository")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read private spec: {type(exc).__name__}") from exc
    if not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"schema must equal {SPEC_SCHEMA}")

    state_dir = _absolute_path(spec.get("state_dir"), "state_dir")
    terminal_values = spec.get("terminal_receipts")
    if not isinstance(terminal_values, list) or len(terminal_values) != 4:
        raise SpecError("terminal_receipts must contain exactly four paths")
    terminal_paths = [_absolute_path(value, f"terminal_receipts[{index}]")
                      for index, value in enumerate(terminal_values)]
    if len({path.resolve() for path in terminal_paths}) != 4:
        raise SpecError("terminal_receipts must contain four distinct paths")

    expected = spec.get("expected")
    expected_keys = {
        "seed", "h4_scientific_config_sha256", "h72_scientific_config_sha256",
        "network_id", "genesis_hash", "component_versions",
    }
    if not isinstance(expected, dict) or set(expected) != expected_keys:
        raise SpecError(f"expected must contain exactly {sorted(expected_keys)}")
    if type(expected["seed"]) is not int:
        raise SpecError("expected.seed must be an integer")
    for field in expected_keys - {"seed", "component_versions"}:
        _nonempty_string(expected[field], f"expected.{field}")
    try:
        expected_components = _exact_components(expected["component_versions"], "expected.component_versions")
    except Refusal as exc:
        raise SpecError(str(exc)) from exc

    closure = _absolute_path(spec.get("archive_closure_receipt"), "archive_closure_receipt")
    launch = spec.get("h72_start")
    if not isinstance(launch, dict):
        raise SpecError("h72_start must be an object")
    command = launch.get("command")
    if (not isinstance(command, list) or not command
            or any(not isinstance(part, str) or not part for part in command)):
        raise SpecError("h72_start.command must be a nonempty argv list")
    start_receipt = _absolute_path(launch.get("receipt"), "h72_start.receipt")
    cwd = _absolute_path(launch.get("working_directory"), "h72_start.working_directory")
    nodes = launch.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise SpecError("h72_start.nodes must contain exactly four live node endpoints")
    clean_nodes = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != {"node_id", "base_url"}:
            raise SpecError(f"h72_start.nodes[{index}] must contain exactly node_id and base_url")
        clean_nodes.append({
            "node_id": _nonempty_string(node["node_id"], f"h72_start.nodes[{index}].node_id"),
            "base_url": _nonempty_string(node["base_url"], f"h72_start.nodes[{index}].base_url"),
        })
    timeout = launch.get("verification_timeout_seconds", 5.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise SpecError("h72_start.verification_timeout_seconds must be greater than zero")
    environment = launch.get("environment", {})
    if (not isinstance(environment, dict)
            or any(not isinstance(key, str) or not key or not isinstance(value, str)
                   for key, value in environment.items())):
        raise SpecError("h72_start.environment must map nonempty names to strings")
    poll = spec.get("poll_interval_seconds", 60)
    retry = spec.get("launch_retry_seconds", 300)
    if type(poll) is not int or poll <= 0:
        raise SpecError("poll_interval_seconds must be a positive integer")
    if type(retry) is not int or retry < poll:
        raise SpecError("launch_retry_seconds must be an integer at least as large as poll_interval_seconds")

    return {
        "schema": SPEC_SCHEMA,
        "spec_path": path.resolve(),
        "state_dir": state_dir,
        "terminal_receipts": terminal_paths,
        "archive_closure_receipt": closure,
        "poll_interval_seconds": poll,
        "launch_retry_seconds": retry,
        "expected": dict(expected, component_versions=expected_components),
        "h72_start": {
            "command": list(command), "receipt": start_receipt, "working_directory": cwd,
            "environment": dict(environment), "nodes": clean_nodes,
            "verification_timeout_seconds": float(timeout),
        },
    }


def _validate_terminal(payload: Mapping[str, Any], expected: Mapping[str, Any], field: str) -> None:
    if payload.get("status") != "COMPLETE":
        raise Refusal(f"TERMINAL_NOT_COMPLETE: {field}: {payload.get('status')}")
    _nonempty_string(payload.get("island_id"), f"{field}.island_id", Refusal)
    if payload.get("horizon") != 4:
        raise Refusal(f"HORIZON_MISMATCH: {field}")
    if payload.get("seed") != expected["seed"]:
        raise Refusal(f"SEED_MISMATCH: {field}")
    if payload.get("scientific_config_sha256") != expected["h4_scientific_config_sha256"]:
        raise Refusal(f"SCIENTIFIC_CONFIG_MISMATCH: {field}")
    if payload.get("network_id") != expected["network_id"]:
        raise Refusal(f"NETWORK_MISMATCH: {field}")
    if payload.get("genesis_hash") != expected["genesis_hash"]:
        raise Refusal(f"GENESIS_MISMATCH: {field}")
    _exact_components(payload.get("component_versions"), f"{field}.component_versions",
                      expected=expected["component_versions"])
    for name in ("chain_height", "finalized_height", "terminal_generation"):
        if type(payload.get(name)) is not int or payload[name] < 0:
            raise Refusal(f"INVALID_TERMINAL_FIELD: {field}.{name}")
    for name in ("tip_hash", "finalized_hash", "stop_reason"):
        _nonempty_string(payload.get(name), f"{field}.{name}", Refusal)
    for name in (
        "champion_sha256", "plan_sha256", "contract_sha256",
        "checkpoint_identity_sha256", "final_population_sha256",
        "candidate_ledger_sha256",
    ):
        _sha256_string(payload.get(name), f"{field}.{name}")
    fitness = payload.get("champion_fitness")
    if type(fitness) not in (int, float) or not math.isfinite(fitness):
        raise Refusal(f"INVALID_TERMINAL_FIELD: {field}.champion_fitness")
    evaluated = payload.get("evaluated_candidate_count")
    if type(evaluated) is not int or evaluated < 1:
        raise Refusal(f"INVALID_TERMINAL_FIELD: {field}.evaluated_candidate_count")


def _terminal_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, path in enumerate(spec["terminal_receipts"]):
        if not path.is_file():
            continue
        payload, receipt_sha = _read_envelope(path, TERMINAL_SCHEMA, f"terminal_receipts[{index}]")
        _validate_terminal(payload, spec["expected"], f"terminal_receipts[{index}]")
        records.append({"payload": payload, "receipt_sha256": receipt_sha})

    identities = [record["payload"]["island_id"] for record in records]
    digests = [record["receipt_sha256"] for record in records]
    if len(identities) != len(set(identities)) or len(digests) != len(set(digests)):
        raise Refusal("DUPLICATE_TERMINAL: copied receipt or repeated island identity")

    if records:
        baseline = records[0]["payload"]
        comparisons = (
            ("network_id", "NETWORK_MISMATCH"),
            ("genesis_hash", "GENESIS_MISMATCH"),
            ("chain_height", "CHAIN_HEIGHT_MISMATCH"),
            ("tip_hash", "TIP_HASH_MISMATCH"),
            ("finalized_height", "FINALIZED_HEIGHT_MISMATCH"),
            ("finalized_hash", "FINALIZED_HASH_MISMATCH"),
        )
        for record in records[1:]:
            for name, code in comparisons:
                if record["payload"].get(name) != baseline.get(name):
                    raise Refusal(f"{code}: {record['payload']['island_id']}")
    return {
        "found": len(records), "valid": len(records), "required": 4,
        "records": records,
    }


def _validate_closure(spec: Mapping[str, Any], terminals: Mapping[str, Any]) -> dict[str, Any] | None:
    path = spec["archive_closure_receipt"]
    if not path.is_file():
        return None
    payload, receipt_sha = _read_envelope(path, CLOSURE_SCHEMA, "archive_closure_receipt")
    expected = spec["expected"]
    if payload.get("status") != "GREEN":
        raise Refusal("ARCHIVE_NOT_GREEN: closure status")
    if payload.get("horizon") != 4 or payload.get("scientific_config_sha256") != expected["h4_scientific_config_sha256"]:
        raise Refusal("CLOSURE_IDENTITY_MISMATCH")
    wanted_terminals = sorted(record["receipt_sha256"] for record in terminals["records"])
    if payload.get("terminal_receipt_sha256s") != wanted_terminals:
        raise Refusal("CLOSURE_TERMINALS_MISMATCH")
    baseline = terminals["records"][0]["payload"]
    for name, code in (
        ("network_id", "NETWORK_MISMATCH"), ("genesis_hash", "GENESIS_MISMATCH"),
    ):
        if payload.get(name) != baseline.get(name):
            raise Refusal(f"{code}: archive closure")
    terminal_fields = ("chain_height", "tip_hash", "finalized_height", "finalized_hash")
    terminal_chain = payload.get("terminal_chain")
    expected_terminal = {name: baseline.get(name) for name in terminal_fields}
    if terminal_chain != expected_terminal:
        raise Refusal("TERMINAL_CHAIN_MISMATCH: archive closure")
    relation = payload.get("chain_relation")
    if relation not in {"EXACT", "DESCENDANT_VERIFIED"}:
        raise Refusal("CHAIN_RELATION_INVALID: archive closure")
    current_height = payload.get("chain_height")
    current_finalized = payload.get("finalized_height")
    if (type(current_height) is not int or current_height < baseline["chain_height"]
            or type(current_finalized) is not int
            or current_finalized < baseline["finalized_height"]
            or not isinstance(payload.get("tip_hash"), str) or not payload["tip_hash"]
            or not isinstance(payload.get("finalized_hash"), str) or not payload["finalized_hash"]):
        raise Refusal("CLOSURE_CHAIN_NOT_DESCENDANT")
    if relation == "EXACT" and any(
            payload.get(name) != baseline.get(name) for name in terminal_fields):
        raise Refusal("EXACT_CHAIN_RELATION_MISMATCH")
    archive = payload.get("archive")
    if (not isinstance(archive, dict) or type(archive.get("items")) is not int
            or archive.get("items") < 1 or archive.get("green") != archive.get("items")
            or archive.get("missing") != [] or archive.get("submitted_unfinalized") != 0):
        raise Refusal("ARCHIVE_NOT_GREEN")
    olap = payload.get("olap")
    if (not isinstance(olap, dict) or olap.get("status") != "GREEN"
            or olap.get("chain_verified") is not True or olap.get("readback_verified") is not True):
        raise Refusal("OLAP_NOT_GREEN")
    return {"payload": payload, "receipt_sha256": receipt_sha}


def _transition_id(spec: Mapping[str, Any], terminals: Mapping[str, Any], closure: Mapping[str, Any]) -> str:
    chain = terminals["records"][0]["payload"]
    return _digest({
        "schema": "i6n.island.transition_identity.v1",
        "h4_scientific_config_sha256": spec["expected"]["h4_scientific_config_sha256"],
        "h72_scientific_config_sha256": spec["expected"]["h72_scientific_config_sha256"],
        "terminal_receipt_sha256s": sorted(record["receipt_sha256"] for record in terminals["records"]),
        "archive_closure_sha256": closure["receipt_sha256"],
        "network_id": chain["network_id"], "genesis_hash": chain["genesis_hash"],
        "chain_height": chain["chain_height"], "tip_hash": chain["tip_hash"],
        "component_versions": spec["expected"]["component_versions"],
    })


def _read_optional(path: Path, schema: str, field: str) -> tuple[dict[str, Any], str] | None:
    return _read_envelope(path, schema, field) if path.is_file() else None


def _expected_start_payload(spec: Mapping[str, Any], transition_id: str,
                            fleet_report: Mapping[str, Any]) -> dict[str, Any]:
    fleet = fleet_report.get("fleet")
    nodes = fleet_report.get("nodes")
    if (fleet_report.get("ready") is not True or fleet_report.get("status") != "READY"
            or not isinstance(fleet, Mapping) or not isinstance(nodes, list) or len(nodes) != 4):
        raise Refusal("H72_FLEET_NOT_READY")
    expected = spec["expected"]
    if fleet.get("network_id") != expected["network_id"]:
        raise Refusal("NETWORK_MISMATCH: live H72 fleet")
    if fleet.get("genesis_hash") != expected["genesis_hash"]:
        raise Refusal("GENESIS_MISMATCH: live H72 fleet")
    finalized = fleet.get("finalized_checkpoint")
    if (not isinstance(finalized, Mapping) or type(finalized.get("height")) is not int
            or not isinstance(finalized.get("hash"), str) or not finalized["hash"]):
        raise Refusal("FINALIZED_CHECKPOINT_MISSING: live H72 fleet")
    node_evidence = []
    for node in sorted(nodes, key=lambda item: str(item.get("node_id"))):
        if node.get("status") != "MATCH" or node.get("reachable") is not True:
            raise Refusal("H72_FLEET_NOT_READY")
        node_evidence.append({
            "node_id": node.get("node_id"), "seed": node.get("seed"),
            "scientific_config_sha256": node.get("config_digest"),
            "component_versions": node.get("component_versions"),
            "network_id": node.get("network_id"), "genesis_hash": node.get("genesis_hash"),
            "chain_height": node.get("chain_height"), "tip_hash": node.get("tip_hash"),
            "finalized_checkpoint": node.get("finalized_checkpoint"),
        })
    evidence = {
        "expected": fleet_report.get("expected"), "fleet": dict(fleet), "nodes": node_evidence,
    }
    return {
        "status": "LIVE_FLEET_VERIFIED", "horizon": 72,
        "seed": expected["seed"],
        "scientific_config_sha256": expected["h72_scientific_config_sha256"],
        "transition_id": transition_id,
        "network_id": fleet["network_id"], "genesis_hash": fleet["genesis_hash"],
        "chain_height": fleet["chain_height"], "tip_hash": fleet["tip_hash"],
        "finalized_height": finalized["height"], "finalized_hash": finalized["hash"],
        "component_versions": dict(expected["component_versions"]),
        "node_ids": [node["node_id"] for node in node_evidence],
        "fleet_evidence_sha256": _digest(evidence),
        "fleet_evidence": evidence,
    }


def _validate_start(spec: Mapping[str, Any], transition_id: str) -> dict[str, Any] | None:
    result = _read_optional(spec["h72_start"]["receipt"], START_SCHEMA, "h72_start.receipt")
    if result is None:
        return None
    payload, receipt_sha = result
    if (payload.get("status") != "LIVE_FLEET_VERIFIED" or payload.get("horizon") != 72
            or payload.get("seed") != spec["expected"]["seed"]
            or payload.get("scientific_config_sha256") != spec["expected"]["h72_scientific_config_sha256"]
            or payload.get("transition_id") != transition_id
            or payload.get("network_id") != spec["expected"]["network_id"]
            or payload.get("genesis_hash") != spec["expected"]["genesis_hash"]):
        raise Refusal("START_RECEIPT_MISMATCH")
    _exact_components(payload.get("component_versions"), "h72_start.component_versions",
                      expected=spec["expected"]["component_versions"])
    evidence = payload.get("fleet_evidence")
    if (not isinstance(evidence, Mapping)
            or payload.get("fleet_evidence_sha256") != _digest(evidence)
            or not isinstance(payload.get("node_ids"), list)
            or len(payload["node_ids"]) != 4 or len(set(payload["node_ids"])) != 4):
        raise Refusal("START_RECEIPT_MISMATCH")
    fleet = evidence.get("fleet")
    nodes = evidence.get("nodes")
    if (not isinstance(fleet, Mapping) or not isinstance(nodes, list) or len(nodes) != 4
            or fleet.get("chain_height") != payload.get("chain_height")
            or fleet.get("tip_hash") != payload.get("tip_hash")
            or fleet.get("finalized_checkpoint") != {
                "height": payload.get("finalized_height"), "hash": payload.get("finalized_hash")}
            or any(node.get("node_id") not in payload["node_ids"] for node in nodes)):
        raise Refusal("START_RECEIPT_MISMATCH")
    return {"payload": payload, "receipt_sha256": receipt_sha}


def _verify_live_h72(spec: Mapping[str, Any], transition_id: str, *, fetch_json: Callable) -> dict[str, Any]:
    fleet_spec = {
        "schema": "i6n.doin_fleet_verify.spec.v1",
        "expected": {
            "seed": spec["expected"]["seed"],
            "config_digest": spec["expected"]["h72_scientific_config_sha256"],
            "component_versions": spec["expected"]["component_versions"],
        },
        "nodes": spec["h72_start"]["nodes"],
    }
    try:
        report = verify_fleet(
            fleet_spec, fetch_json=fetch_json,
            timeout=spec["h72_start"]["verification_timeout_seconds"],
        )
    except FleetSpecError as exc:
        raise Refusal(f"H72_FLEET_SPEC_INVALID: {exc}") from exc
    if report.get("ready") is not True:
        codes = sorted({error.get("code", "UNKNOWN") for error in report.get("errors", [])})
        if codes and set(codes) <= {"NODE_UNREACHABLE", "MISSING_FIELD"}:
            raise FleetPending("H72_FLEET_PENDING: " + ",".join(codes))
        raise Refusal("H72_FLEET_NOT_READY: " + ",".join(codes))
    payload = _expected_start_payload(spec, transition_id, report)
    envelope = {"schema": START_SCHEMA, "payload": payload, "payload_sha256": _digest(payload)}
    path = spec["h72_start"]["receipt"]
    if path.is_file():
        existing = _read_json(path, "h72_start.receipt")
        if existing != envelope:
            raise Refusal("START_RECEIPT_CONFLICT")
    else:
        _atomic_json(path, envelope)
    return {"payload": payload, "receipt_sha256": envelope["payload_sha256"]}


def _read_intent(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    path = spec["state_dir"] / "INTENT.json"
    if not path.is_file():
        return None
    value = _read_json(path, "INTENT.json")
    if value.get("schema") != INTENT_SCHEMA:
        raise Refusal("INVALID_TRANSITION_INTENT")
    return value


def _read_transition_receipt(spec: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    return _read_optional(spec["state_dir"] / "TRANSITION_RECEIPT.json", RECEIPT_SCHEMA,
                          "TRANSITION_RECEIPT.json")


def _eta(state: str, spec: Mapping[str, Any], now: float, intent: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if state == "COMPLETE":
        return {"kind": "exact", "seconds": 0, "reason": "transition receipt is durable"}
    if state in {"WAIT_H72_RECEIPT", "WAIT_H72_FLEET"} and intent and intent.get("last_launch_at") is not None:
        remaining = max(0, int(spec["launch_retry_seconds"] - (now - float(intent["last_launch_at"]))))
        return {"kind": "bounded", "seconds": remaining, "reason": "next idempotent launch retry"}
    if state in {"READY_TO_START_H72", "H72_STARTED_UNACKNOWLEDGED"}:
        return {"kind": "bounded", "seconds": spec["poll_interval_seconds"], "reason": "next successor tick"}
    return {"kind": "unknown", "seconds": None, "reason": "upstream terminal or archive duration is unknown"}


def _report(state: str, spec: Mapping[str, Any], terminals: Mapping[str, Any], now: float, *,
            closure: Mapping[str, Any] | None = None, transition_id: str | None = None,
            intent: Mapping[str, Any] | None = None, detail: str | None = None) -> dict[str, Any]:
    gates = {
        "terminals": terminals["valid"] == 4,
        "fleet_identity": terminals["valid"] == 4,
        "archive_olap": closure is not None,
        "transition": state == "COMPLETE",
    }
    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "detail": detail,
        "transition_id": transition_id,
        "generated_at": now,
        "progress": {
            "gates_complete": sum(gates.values()), "gates_total": len(gates), "gates": gates,
            "terminals": {key: terminals[key] for key in ("found", "required", "valid")},
        },
        "eta": _eta(state, spec, now, intent),
    }


def _inspect_loaded(spec: Mapping[str, Any], now: float) -> dict[str, Any]:
    terminals = _terminal_evidence(spec)
    if terminals["valid"] < 4:
        return _report("WAIT_TERMINALS", spec, terminals, now)
    closure = _validate_closure(spec, terminals)
    if closure is None:
        return _report("WAIT_ARCHIVE", spec, terminals, now)
    transition_id = _transition_id(spec, terminals, closure)
    intent = _read_intent(spec)
    if intent is not None and intent.get("transition_id") != transition_id:
        raise Refusal("TRANSITION_EVIDENCE_CHANGED")
    completed = _read_transition_receipt(spec)
    if completed is not None:
        payload, _receipt_sha = completed
        if payload.get("status") != "COMPLETE" or payload.get("transition_id") != transition_id:
            raise Refusal("TRANSITION_RECEIPT_MISMATCH")
        _validate_start(spec, transition_id)
        return _report("COMPLETE", spec, terminals, now, closure=closure,
                       transition_id=transition_id, intent=intent)
    started = _validate_start(spec, transition_id)
    if started is not None:
        return _report("H72_STARTED_UNACKNOWLEDGED", spec, terminals, now, closure=closure,
                       transition_id=transition_id, intent=intent)
    state = "WAIT_H72_RECEIPT" if intent and intent.get("launch_attempts", 0) else "READY_TO_START_H72"
    return _report(state, spec, terminals, now, closure=closure,
                   transition_id=transition_id, intent=intent)


def status(spec_path: str | Path, *, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Return current evidence and ETA without creating or changing any file."""
    spec = load_spec(spec_path)
    return _inspect_loaded(spec, float(now()))


@contextmanager
def _exclusive_state(spec: Mapping[str, Any]):
    spec["state_dir"].mkdir(parents=True, exist_ok=True)
    lock_path = spec["state_dir"] / "LOCK"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _write_status(spec: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    _atomic_json(spec["state_dir"] / "STATUS.json", report)


def _finish(spec: Mapping[str, Any], report: Mapping[str, Any], start: Mapping[str, Any],
            terminals: Mapping[str, Any], closure: Mapping[str, Any], now: float) -> dict[str, Any]:
    payload = {
        "status": "COMPLETE", "transition_id": report["transition_id"],
        "h4_terminal_receipt_sha256s": sorted(record["receipt_sha256"] for record in terminals["records"]),
        "archive_closure_sha256": closure["receipt_sha256"],
        "h72_start_receipt_sha256": start["receipt_sha256"],
        "h4_scientific_config_sha256": spec["expected"]["h4_scientific_config_sha256"],
        "h72_scientific_config_sha256": spec["expected"]["h72_scientific_config_sha256"],
        "completed_at": now,
    }
    envelope = {"schema": RECEIPT_SCHEMA, "payload": payload, "payload_sha256": _digest(payload)}
    path = spec["state_dir"] / "TRANSITION_RECEIPT.json"
    if path.is_file():
        existing = _read_json(path, "TRANSITION_RECEIPT.json")
        if existing != envelope:
            raise Refusal("TRANSITION_RECEIPT_CONFLICT")
    else:
        _atomic_json(path, envelope)
    completed = _report("COMPLETE", spec, terminals, now, closure=closure,
                        transition_id=report["transition_id"], intent=_read_intent(spec))
    _write_status(spec, completed)
    return completed


def tick(spec_path: str | Path, *, run_command: Callable[..., Any] = subprocess.run,
         fetch_json: Callable = _fleet_fetch_json, now: Callable[[], float] = time.time,
         after_launch: Callable[[], None] | None = None) -> dict[str, Any]:
    """Advance at most one durable transition step and return machine-readable status."""
    spec = load_spec(spec_path)
    with _exclusive_state(spec):
        timestamp = float(now())
        report = _inspect_loaded(spec, timestamp)
        if report["state"] in {"WAIT_TERMINALS", "WAIT_ARCHIVE", "COMPLETE"}:
            _write_status(spec, report)
            return report

        terminals = _terminal_evidence(spec)
        closure = _validate_closure(spec, terminals)
        if closure is None:  # guarded by inspect; retained as a fail-closed invariant
            raise Refusal("ARCHIVE_NOT_GREEN")
        transition_id = report["transition_id"]
        intent = _read_intent(spec)
        start = _validate_start(spec, transition_id)
        if start is not None or (intent is not None and intent.get("launch_attempts", 0)):
            try:
                verified_start = _verify_live_h72(spec, transition_id, fetch_json=fetch_json)
            except FleetPending as exc:
                last_launch = intent.get("last_launch_at") if intent else None
                if (last_launch is None
                        or timestamp - float(last_launch) < spec["launch_retry_seconds"]):
                    waiting = _report("WAIT_H72_FLEET", spec, terminals, timestamp, closure=closure,
                                      transition_id=transition_id, intent=intent, detail=str(exc))
                    _write_status(spec, waiting)
                    return waiting
            else:
                return _finish(spec, report, verified_start, terminals, closure, timestamp)

        if intent is None:
            intent = {
                "schema": INTENT_SCHEMA, "transition_id": transition_id,
                "created_at": timestamp, "launch_attempts": 0, "last_launch_at": None,
                "last_returncode": None,
            }
            _atomic_json(spec["state_dir"] / "INTENT.json", intent)
        elif intent.get("transition_id") != transition_id:
            raise Refusal("TRANSITION_EVIDENCE_CHANGED")

        last_launch = intent.get("last_launch_at")
        if last_launch is not None and timestamp - float(last_launch) < spec["launch_retry_seconds"]:
            waiting = _report("WAIT_H72_RECEIPT", spec, terminals, timestamp, closure=closure,
                              transition_id=transition_id, intent=intent)
            _write_status(spec, waiting)
            return waiting

        intent["launch_attempts"] = int(intent.get("launch_attempts", 0)) + 1
        intent["last_launch_at"] = timestamp
        intent["last_returncode"] = None
        _atomic_json(spec["state_dir"] / "INTENT.json", intent)

        environment = dict(os.environ)
        environment.update(spec["h72_start"]["environment"])
        environment["I6N_TRANSITION_ID"] = transition_id
        result = run_command(spec["h72_start"]["command"],
                             cwd=str(spec["h72_start"]["working_directory"]),
                             env=environment, check=False)
        intent["last_returncode"] = int(result.returncode)
        _atomic_json(spec["state_dir"] / "INTENT.json", intent)
        if after_launch is not None:
            after_launch()

        try:
            start = _verify_live_h72(spec, transition_id, fetch_json=fetch_json)
        except FleetPending as exc:
            waiting = _report("WAIT_H72_FLEET", spec, terminals, timestamp, closure=closure,
                              transition_id=transition_id, intent=intent, detail=str(exc))
            _write_status(spec, waiting)
            return waiting
        if start is not None:
            return _finish(spec, report, start, terminals, closure, timestamp)
        detail = ("launcher returned without a start receipt" if result.returncode == 0
                  else f"launcher return code {result.returncode}; retry remains idempotent")
        waiting = _report("WAIT_H72_RECEIPT", spec, terminals, timestamp, closure=closure,
                          transition_id=transition_id, intent=intent, detail=detail)
        _write_status(spec, waiting)
        return waiting


def _refusal_report(exc: Exception, now: float) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA, "state": "REFUSED", "detail": str(exc),
        "generated_at": now,
        "progress": {"gates_complete": 0, "gates_total": 4},
        "eta": {"kind": "blocked", "seconds": None, "reason": str(exc)},
    }


def _persist_refusal(spec_path: Path, report: Mapping[str, Any]) -> None:
    """Best-effort operator evidence for a failed timer tick; never used by --status."""
    try:
        spec = load_spec(spec_path)
        _atomic_json(spec["state_dir"] / "STATUS.json", report)
    except (OSError, SpecError, Refusal):
        return


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path, help="private JSON spec outside the repository")
    parser.add_argument("--status", action="store_true", help="read-only status; never launch H72")
    args = parser.parse_args(argv)
    try:
        report = status(args.spec) if args.status else tick(args.spec)
    except (SpecError, Refusal) as exc:
        report = _refusal_report(exc, time.time())
        if not args.status:
            _persist_refusal(args.spec, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
