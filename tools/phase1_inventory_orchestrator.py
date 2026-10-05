#!/usr/bin/env python3
"""Restart-safe, CPU-only orchestration for feature-selection phase 1.

The orchestrator owns placement and lifecycle, not scientific calculations. A
configured worker receives one authenticated inventory row and must return one
typed terminal result. A configured finalizer performs population-wide work
(for example BH/FDR) only after every inventory member has an explicit terminal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


CONFIG_SCHEMA = "phase1.inventory_orchestrator.v1"
PLAN_SCHEMA = "phase1.inventory_plan.v1"
COLUMN_RESULT_SCHEMA = "phase1.column_result.v1"
TERMINAL_SCHEMA = "phase1.inventory_terminal.v1"
FINALIZER_RESULT_SCHEMA = "phase1.finalizer_result.v1"
GATE_SCHEMA = "phase1.completion_gate.v1"
TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "UNAVAILABLE"})


class Phase1Refusal(RuntimeError):
    """A fail-closed configuration, identity or lifecycle refusal."""


@dataclass(frozen=True)
class InventoryItem:
    feature_id: str
    key: str
    row_count: int
    byte_count: int
    estimated_cost: float
    available: bool
    inventory_row: dict[str, str]
    inventory_file: str


def _canonical_bytes(document: Any) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha(document: Any) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _atomic_json(path: Path, document: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, sort_keys=True, indent=2) + "\n"
    if exclusive:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resolve(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise Phase1Refusal(f"expected config schema {CONFIG_SCHEMA}")
    if config.get("phase") != "PHASE_1":
        raise Phase1Refusal("this executable accepts PHASE_1 only")
    base = config_path.parent
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(base)
    config["state_root"] = _resolve(base, config["state_root"])
    inventory = config.get("inventory", {})
    inventory["files"] = [_resolve(base, value) for value in inventory.get("files", [])]
    if not inventory["files"]:
        raise Phase1Refusal("inventory.files must not be empty")
    if int(inventory.get("expected_total", 0)) <= 0:
        raise Phase1Refusal("inventory.expected_total must be positive")
    hosts = config.get("hosts", [])
    host_ids = [host.get("id") for host in hosts]
    if len(host_ids) != len(set(host_ids)) or not all(host_ids):
        raise Phase1Refusal("host ids must be present and unique")
    if len([host for host in hosts if host.get("size_class") == "small"]) != 1:
        raise Phase1Refusal("exactly one small host must be declared")
    if not any(host.get("size_class") == "large" for host in hosts):
        raise Phase1Refusal("at least one large host must be declared")
    environment = config.get("worker", {}).get("environment", {})
    if environment.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise Phase1Refusal("phase 1 is CPU-only; CUDA_VISIBLE_DEVICES must be empty")
    if environment.get("NVIDIA_VISIBLE_DEVICES") not in (None, "", "void", "none"):
        raise Phase1Refusal("phase 1 cannot expose NVIDIA devices")
    for section in ("worker", "finalizer"):
        command = config.get(section, {}).get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(token, str) and token for token in command
        ):
            raise Phase1Refusal(f"{section}.command must be a non-empty token list")
    return config


def _integer(value: Any, *, field: str, feature_id: str) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError) as error:
        raise Phase1Refusal(f"{feature_id}: {field} is not an integer") from error
    if number < 0:
        raise Phase1Refusal(f"{feature_id}: {field} is negative")
    return number


def load_inventory(config: dict[str, Any]) -> tuple[InventoryItem, ...]:
    specification = config["inventory"]
    id_field = specification.get("id_field", "feature_id")
    row_field = specification.get("row_count_field", "train_rows")
    bytes_field = specification.get("bytes_field")
    include = specification.get("include", {})
    availability_field = specification.get("availability_field")
    available_values = set(specification.get("available_values", ["AVAILABLE"]))
    bytes_per_row = int(specification.get("bytes_per_row", 8))
    bytes_per_cost_unit = float(specification.get("bytes_per_cost_unit", 8))
    if bytes_per_row <= 0 or not math.isfinite(bytes_per_cost_unit) or bytes_per_cost_unit <= 0:
        raise Phase1Refusal("inventory byte-cost parameters must be positive")
    items: list[InventoryItem] = []
    seen: set[str] = set()
    for file_name in specification["files"]:
        inventory_path = Path(file_name)
        with inventory_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if any(row.get(field) != str(value) for field, value in include.items()):
                    continue
                feature_id = (row.get(id_field) or "").strip()
                if not feature_id:
                    raise Phase1Refusal(f"{inventory_path}: inventory row has no {id_field}")
                if feature_id in seen:
                    raise Phase1Refusal(f"duplicate feature id {feature_id!r}")
                seen.add(feature_id)
                row_count = _integer(row.get(row_field), field=row_field, feature_id=feature_id)
                if bytes_field and row.get(bytes_field) not in (None, ""):
                    byte_count = _integer(
                        row.get(bytes_field), field=bytes_field, feature_id=feature_id
                    )
                else:
                    byte_count = row_count * bytes_per_row
                estimated_cost = float(row_count) + float(byte_count) / bytes_per_cost_unit
                available = (
                    True
                    if not availability_field
                    else row.get(availability_field) in available_values
                )
                key = hashlib.sha256(feature_id.encode("utf-8")).hexdigest()
                items.append(
                    InventoryItem(
                        feature_id=feature_id,
                        key=key,
                        row_count=row_count,
                        byte_count=byte_count,
                        estimated_cost=estimated_cost,
                        available=available,
                        inventory_row=dict(row),
                        inventory_file=str(inventory_path),
                    )
                )
    observed = len(items)
    expected = int(specification["expected_total"])
    if observed != expected:
        raise Phase1Refusal(f"inventory expected {expected} rows but observed {observed}")
    return tuple(sorted(items, key=lambda item: item.feature_id))


def _state_root(config: dict[str, Any]) -> Path:
    return Path(config["state_root"])


def _ensure_layout(config: dict[str, Any]) -> None:
    root = _state_root(config)
    for name in (
        "claims",
        "host_claims",
        "stale_claims",
        "inputs",
        "attempts",
        "terminals",
        "warehouse_receipts",
        "warehouse_failures",
        "warehouse_claims",
        "finalizer",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def _plan_path(config: dict[str, Any]) -> Path:
    return _state_root(config) / "PLAN.json"


def _assignment(config: dict[str, Any], items: Iterable[InventoryItem]) -> dict[str, str]:
    ordered = sorted(items, key=lambda item: (item.estimated_cost, item.feature_id))
    fraction = float(config.get("assignment", {}).get("smallest_fraction", 0.2))
    if not 0.0 < fraction < 1.0:
        raise Phase1Refusal("assignment.smallest_fraction must be between zero and one")
    small = next(host["id"] for host in config["hosts"] if host["size_class"] == "small")
    large = sorted(host["id"] for host in config["hosts"] if host["size_class"] == "large")
    small_count = min(len(ordered), max(1, math.ceil(len(ordered) * fraction)))
    assignment = {item.feature_id: small for item in ordered[:small_count]}
    loads = {host: 0.0 for host in large}
    for item in sorted(
        ordered[small_count:], key=lambda candidate: (-candidate.estimated_cost, candidate.feature_id)
    ):
        host = min(large, key=lambda candidate: (loads[candidate], candidate))
        assignment[item.feature_id] = host
        loads[host] += item.estimated_cost
    return assignment


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_layout(config)
    items = load_inventory(config)
    assigned = _assignment(config, items)
    inventory_identity = [
        {
            "feature_id": item.feature_id,
            "key": item.key,
            "row_count": item.row_count,
            "byte_count": item.byte_count,
            "estimated_cost": item.estimated_cost,
            "available": item.available,
            "inventory_row_sha256": _sha(item.inventory_row),
        }
        for item in items
    ]
    document: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "phase": "PHASE_1",
        "inventory_total": len(items),
        "inventory_sha256": _sha(inventory_identity),
        "config_sha256": _sha(
            {key: value for key, value in config.items() if not key.startswith("_")}
        ),
        "items": [
            {
                **identity,
                "host_id": assigned[identity["feature_id"]],
                "inventory_file": next(
                    item.inventory_file
                    for item in items
                    if item.feature_id == identity["feature_id"]
                ),
            }
            for identity in inventory_identity
        ],
    }
    document["plan_sha256"] = _sha(document)
    path = _plan_path(config)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise Phase1Refusal("existing plan differs from the deterministic plan")
        return existing
    _atomic_json(path, document, exclusive=True)
    return document


def load_plan(config: dict[str, Any]) -> dict[str, Any]:
    path = _plan_path(config)
    if not path.is_file():
        raise Phase1Refusal("PLAN.json is absent; run plan first")
    plan = json.loads(path.read_text(encoding="utf-8"))
    digest = plan.pop("plan_sha256", None)
    observed = _sha(plan)
    plan["plan_sha256"] = digest
    if digest != observed or plan.get("schema") != PLAN_SCHEMA:
        raise Phase1Refusal("PLAN.json identity is invalid")
    return plan


def _host(config: dict[str, Any], host_id: str) -> dict[str, Any]:
    for host in config["hosts"]:
        if host["id"] == host_id:
            return host
    raise Phase1Refusal(f"unknown host {host_id!r}")


def _terminal_path(config: dict[str, Any], key: str) -> Path:
    return _state_root(config) / "terminals" / f"{key}.json"


def _read_terminal(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = document.pop("terminal_sha256", None)
    observed = _sha(document)
    document["terminal_sha256"] = digest
    if document.get("schema") != TERMINAL_SCHEMA or digest != observed:
        raise Phase1Refusal(f"invalid terminal {path}")
    if document.get("state") not in TERMINAL_STATES:
        raise Phase1Refusal(f"invalid terminal state in {path}")
    return document


def _archive_stale_claim(config: dict[str, Any], path: Path) -> bool:
    metadata_path = path / "claim.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        started = float(metadata["started_epoch"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        started = path.stat().st_mtime
    heartbeat = path / "heartbeat"
    freshness = heartbeat.stat().st_mtime if heartbeat.is_file() else started
    stale_after = float(config.get("claims", {}).get("stale_after_seconds", 3600))
    if time.time() - freshness <= stale_after:
        return False
    archive = _state_root(config) / "stale_claims" / (
        f"{path.name}.{time.time_ns()}.{uuid.uuid4().hex}"
    )
    try:
        os.replace(path, archive)
    except FileNotFoundError:
        return True
    return True


def _claim(
    config: dict[str, Any], path: Path, metadata: dict[str, Any]
) -> bool:
    try:
        path.mkdir()
    except FileExistsError:
        if not _archive_stale_claim(config, path):
            return False
        try:
            path.mkdir()
        except FileExistsError:
            return False
    metadata = {**metadata, "started_epoch": time.time(), "pid": os.getpid()}
    _atomic_json(path / "claim.json", metadata, exclusive=True)
    (path / "heartbeat").touch()
    return True


def _release_claim(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class _ClaimHeartbeat:
    def __init__(self, paths: Iterable[Path], stale_after_seconds: float) -> None:
        self.paths = tuple(paths)
        self.interval = max(0.01, min(30.0, stale_after_seconds / 3.0))
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            for path in self.paths:
                heartbeat = path / "heartbeat"
                if heartbeat.parent.is_dir():
                    heartbeat.touch()

    def __enter__(self) -> "_ClaimHeartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        self.thread.join(timeout=max(1.0, self.interval * 2.0))


def _format_command(tokens: list[str], values: dict[str, str]) -> list[str]:
    try:
        return [token.format_map(values) for token in tokens]
    except KeyError as error:
        raise Phase1Refusal(f"unknown command placeholder {error.args[0]!r}") from error


def _cpu_environment(config: dict[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in config["worker"].get("environment", {}).items()}
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PHASE1_ONLY": "1",
            "PHASE2_FORBIDDEN": "1",
        }
    )
    return environment


def _http_submit(document: dict[str, Any], warehouse: dict[str, Any]) -> dict[str, Any]:
    payload = _canonical_bytes(document)
    headers = {"Content-Type": "application/json"}
    token_env = warehouse.get("token_env")
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise Phase1Refusal(f"warehouse token environment {token_env!r} is absent")
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        warehouse["url"], data=payload, headers=headers, method="POST"
    )
    with urllib.request.urlopen(
        request, timeout=float(warehouse.get("timeout_seconds", 30))
    ) as response:
        body = response.read()
        return {
            "status": response.status,
            "body": json.loads(body) if body else None,
        }


Submitter = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _submit_terminal(
    config: dict[str, Any], terminal: dict[str, Any], submit: Submitter
) -> dict[str, Any] | None:
    envelope = terminal.get("result", {}).get("envelope")
    if not isinstance(envelope, dict):
        return None
    key = terminal["key"]
    receipt = _state_root(config) / "warehouse_receipts" / f"{key}.json"
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    submission_claim = _state_root(config) / "warehouse_claims" / f"{key}.claim"
    if not _claim(
        config,
        submission_claim,
        {"feature_id": terminal["feature_id"], "key": key, "kind": "warehouse"},
    ):
        return None
    try:
        if receipt.is_file():
            return json.loads(receipt.read_text(encoding="utf-8"))
        response = submit(envelope, config["warehouse"])
        document = {
            "schema": "phase1.warehouse_receipt.v1",
            "feature_id": terminal["feature_id"],
            "key": key,
            "envelope_sha256": _sha(envelope),
            "response": response,
        }
        _atomic_json(receipt, document, exclusive=True)
        failure = _state_root(config) / "warehouse_failures" / f"{key}.json"
        failure.unlink(missing_ok=True)
        return document
    except Exception as error:  # retained for automatic retry; science stays local
        _atomic_json(
            _state_root(config) / "warehouse_failures" / f"{key}.json",
            {
                "schema": "phase1.warehouse_submission_failure.v1",
                "feature_id": terminal["feature_id"],
                "key": key,
                "error_type": type(error).__name__,
                "reason": str(error)[:500],
            },
        )
        return None
    finally:
        _release_claim(submission_claim)


def _retry_submissions(config: dict[str, Any], submit: Submitter) -> None:
    for path in sorted((_state_root(config) / "terminals").glob("*.json")):
        terminal = _read_terminal(path)
        _submit_terminal(config, terminal, submit)


def _write_unavailable_terminal(
    config: dict[str, Any], plan: dict[str, Any], item: dict[str, Any], host_id: str
) -> dict[str, Any]:
    document = {
        "schema": TERMINAL_SCHEMA,
        "feature_id": item["feature_id"],
        "key": item["key"],
        "state": "UNAVAILABLE",
        "host_id": host_id,
        "plan_sha256": plan["plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
        "estimated_cost": item["estimated_cost"],
        "duration_seconds": 0.0,
        "return_code": None,
        "result": {
            "schema": COLUMN_RESULT_SCHEMA,
            "feature_id": item["feature_id"],
            "state": "UNAVAILABLE",
            "reason": "inventory availability state is not admitted",
        },
    }
    document["terminal_sha256"] = _sha(document)
    _atomic_json(_terminal_path(config, item["key"]), document, exclusive=True)
    return document


def _run_item(
    config: dict[str, Any], plan: dict[str, Any], item: dict[str, Any], host_id: str
) -> dict[str, Any]:
    source_items = {entry.feature_id: entry for entry in load_inventory(config)}
    source = source_items[item["feature_id"]]
    if _sha(source.inventory_row) != item["inventory_row_sha256"]:
        raise Phase1Refusal(
            f"{item['feature_id']}: inventory row changed after PLAN.json was frozen"
        )
    if not source.available:
        return _write_unavailable_terminal(config, plan, item, host_id)
    input_path = _state_root(config) / "inputs" / f"{item['key']}.json"
    _atomic_json(input_path, source.inventory_row)
    attempt_id = uuid.uuid4().hex
    output_path = _state_root(config) / "attempts" / f"{item['key']}.{attempt_id}.json"
    values = {
        "feature_id": item["feature_id"],
        "feature_key": item["key"],
        "host_id": host_id,
        "inventory_row_path": str(input_path),
        "worker_output_path": str(output_path),
        "state_root": str(_state_root(config)),
        "plan_path": str(_plan_path(config)),
    }
    host = _host(config, host_id)
    worker_command = _format_command(config["worker"]["command"], values)
    cpu_contract = [
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "NVIDIA_VISIBLE_DEVICES=void",
        "PHASE1_ONLY=1",
        "PHASE2_FORBIDDEN=1",
    ]
    command = list(host.get("command_prefix", [])) + cpu_contract + worker_command
    started = time.monotonic()
    return_code: int | None = None
    reason: str | None = None
    try:
        process = subprocess.run(
            command,
            check=False,
            timeout=float(config["worker"].get("timeout_seconds", 3600)),
            env=_cpu_environment(config),
            capture_output=True,
            text=True,
        )
        return_code = process.returncode
        if output_path.is_file():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if result.get("schema") != COLUMN_RESULT_SCHEMA:
                raise Phase1Refusal("worker returned an unknown result schema")
            if result.get("feature_id") != item["feature_id"]:
                raise Phase1Refusal("worker result belongs to another feature")
            if result.get("state") not in TERMINAL_STATES:
                raise Phase1Refusal("worker returned a non-terminal state")
            if return_code != 0 and result.get("state") == "COMPLETED":
                raise Phase1Refusal(
                    "worker declared COMPLETED with a non-zero return code"
                )
        elif return_code in set(config["worker"].get("unavailable_exit_codes", [])):
            result = {
                "schema": COLUMN_RESULT_SCHEMA,
                "feature_id": item["feature_id"],
                "state": "UNAVAILABLE",
                "reason": f"worker returned configured unavailable code {return_code}",
            }
        else:
            result = {
                "schema": COLUMN_RESULT_SCHEMA,
                "feature_id": item["feature_id"],
                "state": "FAILED",
                "reason": "worker did not produce a terminal result",
                "stderr": process.stderr[-1000:],
            }
    except subprocess.TimeoutExpired:
        result = {
            "schema": COLUMN_RESULT_SCHEMA,
            "feature_id": item["feature_id"],
            "state": "FAILED",
            "reason": "worker timeout",
        }
        reason = "TIMEOUT"
    except Exception as error:
        result = {
            "schema": COLUMN_RESULT_SCHEMA,
            "feature_id": item["feature_id"],
            "state": "FAILED",
            "reason": str(error)[:1000],
            "error_type": type(error).__name__,
        }
        reason = type(error).__name__
    duration = max(0.0, time.monotonic() - started)
    document = {
        "schema": TERMINAL_SCHEMA,
        "feature_id": item["feature_id"],
        "key": item["key"],
        "state": result["state"],
        "host_id": host_id,
        "plan_sha256": plan["plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
        "estimated_cost": item["estimated_cost"],
        "duration_seconds": duration,
        "return_code": return_code,
        "command_sha256": _sha(command),
        "failure_class": reason,
        "result": result,
    }
    document["terminal_sha256"] = _sha(document)
    _atomic_json(_terminal_path(config, item["key"]), document, exclusive=True)
    return document


def run_host_once(
    config: dict[str, Any], host_id: str, *, submit: Submitter = _http_submit
) -> dict[str, Any]:
    """Run at most one assigned series; never overlaps work on the same host."""

    _ensure_layout(config)
    plan = load_plan(config)
    _host(config, host_id)
    _retry_submissions(config, submit)
    host_key = hashlib.sha256(host_id.encode("utf-8")).hexdigest()
    host_claim = _state_root(config) / "host_claims" / f"{host_key}.claim"
    if not _claim(config, host_claim, {"host_id": host_id, "kind": "host"}):
        return {"action": "IDLE", "reason": "HOST_BUSY", "host_id": host_id}
    try:
        for item in sorted(
            (entry for entry in plan["items"] if entry["host_id"] == host_id),
            key=lambda entry: (entry["estimated_cost"], entry["feature_id"]),
        ):
            terminal_path = _terminal_path(config, item["key"])
            if terminal_path.is_file():
                continue
            claim_path = _state_root(config) / "claims" / f"{item['key']}.claim"
            if not _claim(
                config,
                claim_path,
                {
                    "feature_id": item["feature_id"],
                    "key": item["key"],
                    "host_id": host_id,
                    "kind": "series",
                },
            ):
                continue
            try:
                if terminal_path.is_file():
                    continue
                stale_after = float(
                    config.get("claims", {}).get("stale_after_seconds", 3600)
                )
                with _ClaimHeartbeat((host_claim, claim_path), stale_after):
                    terminal = _run_item(config, plan, item, host_id)
                _submit_terminal(config, terminal, submit)
                return {
                    "action": "TERMINAL_WRITTEN",
                    "host_id": host_id,
                    "feature_id": item["feature_id"],
                    "terminal_state": terminal["state"],
                }
            finally:
                _release_claim(claim_path)
        return {"action": "IDLE", "reason": "NO_PENDING_ITEM", "host_id": host_id}
    finally:
        _release_claim(host_claim)


def _terminal_coverage(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    terminals: dict[str, dict[str, Any]] = {}
    for item in plan["items"]:
        path = _terminal_path(config, item["key"])
        if not path.is_file():
            continue
        terminal = _read_terminal(path)
        if terminal["feature_id"] != item["feature_id"]:
            raise Phase1Refusal(f"terminal {path} belongs to another feature")
        if terminal["plan_sha256"] != plan["plan_sha256"]:
            raise Phase1Refusal(f"terminal {path} belongs to another plan")
        terminals[item["feature_id"]] = terminal
    return terminals


def finalize_if_ready(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_layout(config)
    plan = load_plan(config)
    gate_path = _state_root(config) / "PHASE_1_COMPLETE.json"
    if gate_path.is_file():
        require_phase1_complete(config)
        return {"action": "PHASE_1_COMPLETE", "already_complete": True}
    terminals = _terminal_coverage(config, plan)
    if len(terminals) != plan["inventory_total"]:
        return {
            "action": "WAITING_FOR_TERMINALS",
            "expected": plan["inventory_total"],
            "observed": len(terminals),
        }
    finalizer_lock = _state_root(config) / "finalizer" / "claim"
    if not _claim(config, finalizer_lock, {"kind": "finalizer", "host_id": "coordinator"}):
        return {"action": "FINALIZER_BUSY"}
    try:
        output_path = _state_root(config) / "finalizer" / f"result.{uuid.uuid4().hex}.json"
        values = {
            "state_root": str(_state_root(config)),
            "plan_path": str(_plan_path(config)),
            "terminals_dir": str(_state_root(config) / "terminals"),
            "finalizer_output_path": str(output_path),
        }
        command = _format_command(config["finalizer"]["command"], values)
        environment = _cpu_environment(config)
        stale_after = float(config.get("claims", {}).get("stale_after_seconds", 3600))
        with _ClaimHeartbeat((finalizer_lock,), stale_after):
            process = subprocess.run(
                command,
                check=False,
                timeout=float(config["finalizer"].get("timeout_seconds", 3600)),
                env=environment,
                capture_output=True,
                text=True,
            )
        if process.returncode != 0 or not output_path.is_file():
            return {
                "action": "FINALIZER_FAILED",
                "return_code": process.returncode,
                "stderr": process.stderr[-1000:],
            }
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("schema") != FINALIZER_RESULT_SCHEMA:
            raise Phase1Refusal("finalizer returned an unknown schema")
        if result.get("state") != "PHASE_1_COMPLETE":
            return {"action": "FINALIZER_INCOMPLETE", "result": result}
        if int(result.get("terminal_count", -1)) != plan["inventory_total"]:
            raise Phase1Refusal("finalizer terminal count differs from the plan")
        _atomic_json(_state_root(config) / "finalizer" / "result.json", result)
        gate = {
            "schema": GATE_SCHEMA,
            "state": "PHASE_1_COMPLETE",
            "plan_sha256": plan["plan_sha256"],
            "inventory_sha256": plan["inventory_sha256"],
            "terminal_count": len(terminals),
            "terminal_set_sha256": _sha(
                sorted(terminal["terminal_sha256"] for terminal in terminals.values())
            ),
            "finalizer_result_sha256": _sha(result),
        }
        gate["gate_sha256"] = _sha(gate)
        _atomic_json(gate_path, gate, exclusive=True)
        return {"action": "PHASE_1_COMPLETE", "gate": gate}
    finally:
        _release_claim(finalizer_lock)


def require_phase1_complete(config: dict[str, Any]) -> dict[str, Any]:
    plan = load_plan(config)
    gate_path = _state_root(config) / "PHASE_1_COMPLETE.json"
    if not gate_path.is_file():
        raise Phase1Refusal("PHASE_1_COMPLETE gate is absent; phase 2 and GPU work are forbidden")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    digest = gate.pop("gate_sha256", None)
    observed = _sha(gate)
    gate["gate_sha256"] = digest
    if gate.get("schema") != GATE_SCHEMA or digest != observed:
        raise Phase1Refusal("PHASE_1_COMPLETE gate identity is invalid")
    if gate.get("plan_sha256") != plan["plan_sha256"]:
        raise Phase1Refusal("PHASE_1_COMPLETE belongs to another plan")
    terminals = _terminal_coverage(config, plan)
    if len(terminals) != plan["inventory_total"]:
        raise Phase1Refusal("PHASE_1_COMPLETE no longer has complete terminal coverage")
    observed_set = _sha(sorted(item["terminal_sha256"] for item in terminals.values()))
    if gate.get("terminal_set_sha256") != observed_set:
        raise Phase1Refusal("PHASE_1_COMPLETE terminal set changed")
    return gate


def _run_until_idle(config: dict[str, Any], submit: Submitter = _http_submit) -> dict[str, Any]:
    """Cooperatively drain all hosts; each host still owns one process at a time."""

    import concurrent.futures

    hosts = sorted(host["id"] for host in config["hosts"])
    actions: list[dict[str, Any]] = []
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
            cycle = list(
                executor.map(lambda host_id: run_host_once(config, host_id, submit=submit), hosts)
            )
        actions.extend(cycle)
        if all(item["action"] == "IDLE" for item in cycle):
            break
    return {"actions": actions, "finalizer": finalize_if_ready(config)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    subcommands.add_parser("plan")
    host = subcommands.add_parser("run-host")
    host.add_argument("--host", required=True)
    subcommands.add_parser("run")
    subcommands.add_parser("finalize")
    subcommands.add_parser("gate-phase2")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.operation == "plan":
            result = build_plan(config)
        elif arguments.operation == "run-host":
            result = run_host_once(config, arguments.host)
        elif arguments.operation == "run":
            result = _run_until_idle(config)
        elif arguments.operation == "finalize":
            result = finalize_if_ready(config)
        else:
            result = require_phase1_complete(config)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Phase1Refusal as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
