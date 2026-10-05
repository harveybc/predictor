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


CONFIG_SCHEMA = "phase1.inventory_orchestrator.v2"
PLAN_SCHEMA = "phase1.inventory_plan.v2"
COLUMN_REQUEST_SCHEMA = "phase1.column_request.v1"
COLUMN_RESULT_SCHEMA = "phase1.column_result.v1"
TERMINAL_SCHEMA = "phase1.inventory_terminal.v2"
FINALIZER_RESULT_SCHEMA = "phase1.finalizer_result.v1"
GATE_SCHEMA = "phase1.completion_gate.v2"
RECONCILIATION_SCHEMA = "phase1.warehouse_reconciliation.v1"
CLOSURE_TERMINAL_STATES = frozenset({"COMPLETED", "UNAVAILABLE"})


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
    population = config.get("population")
    if not isinstance(population, dict) or not population.get("id"):
        raise Phase1Refusal("population.id must identify one independent population")
    if not population.get("target_pack"):
        raise Phase1Refusal("population.target_pack must be declared")
    mode = population.get("mode", "CAUSAL")
    if mode not in {"CAUSAL", "PROFILE_ONLY"}:
        raise Phase1Refusal("population.mode must be CAUSAL or PROFILE_ONLY")
    if mode == "PROFILE_ONLY":
        campaign = population.get("campaign_sha256")
        if not isinstance(campaign, str) or len(campaign) != 64:
            raise Phase1Refusal("PROFILE_ONLY population requires campaign_sha256")
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
    if config.get("worker", {}).get("transport") != "stdio-json-v1":
        raise Phase1Refusal("worker.transport must be stdio-json-v1")
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
    retries = config.get("retries", {})
    if int(retries.get("max_attempts", 0)) <= 0:
        raise Phase1Refusal("retries.max_attempts must be positive")
    retryable_codes = set(retries.get("retryable_exit_codes", []))
    retryable_classes = set(retries.get("retryable_failure_classes", []))
    if (
        not {75, 137, -9} <= retryable_codes
        or not {"ADMISSION", "OOM"} <= retryable_classes
    ):
        raise Phase1Refusal(
            "admission/OOM/rc75 must remain retryable (codes 75, 137, -9; "
            "classes ADMISSION and OOM)"
        )
    warehouse = config.get("warehouse", {})
    if not warehouse.get("submit_url") or not warehouse.get("reconcile_url"):
        raise Phase1Refusal("warehouse submit_url and reconcile_url are required")
    if not warehouse.get("token_env") or not warehouse.get("auth_profile"):
        raise Phase1Refusal(
            "warehouse token_env and auth_profile are required for authenticated readback"
        )
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
        "failures",
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
        "population_id": config["population"]["id"],
        "target_pack": config["population"]["target_pack"],
        "mode": config["population"].get("mode", "CAUSAL"),
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
    if document["mode"] == "PROFILE_ONLY":
        document["campaign_sha256"] = config["population"]["campaign_sha256"]
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
    if document.get("state") not in CLOSURE_TERMINAL_STATES:
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


def _http_post(
    document: dict[str, Any], warehouse: dict[str, Any], *, url_field: str
) -> dict[str, Any]:
    payload = _canonical_bytes(document)
    headers = {"Content-Type": "application/json"}
    token_env = warehouse.get("token_env")
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise Phase1Refusal(f"warehouse token environment {token_env!r} is absent")
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        warehouse[url_field], data=payload, headers=headers, method="POST"
    )
    with urllib.request.urlopen(
        request, timeout=float(warehouse.get("timeout_seconds", 30))
    ) as response:
        body = response.read()
        return {
            "status": response.status,
            "body": json.loads(body) if body else None,
        }


def _http_submit(document: dict[str, Any], warehouse: dict[str, Any]) -> dict[str, Any]:
    return _http_post(document, warehouse, url_field="submit_url")


def _http_reconcile(document: dict[str, Any], warehouse: dict[str, Any]) -> dict[str, Any]:
    response = _http_post(document, warehouse, url_field="reconcile_url")
    body = response.get("body")
    if not isinstance(body, dict):
        raise Phase1Refusal("warehouse reconciliation returned no JSON object")
    return body


Submitter = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
Reconciler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _require_accepted_submission(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise Phase1Refusal("warehouse submission returned no receipt object")
    payload = response.get("body") if "status" in response else response
    if "status" in response and not 200 <= int(response["status"]) < 300:
        raise Phase1Refusal("warehouse submission was not accepted")
    accepted = isinstance(payload, dict) and (
        payload.get("accepted") is True
        or payload.get("stored") is True
        or payload.get("already_stored") is True
    )
    if not accepted:
        raise Phase1Refusal("warehouse submission returned no accepted receipt")


def _envelope_identity(envelope: dict[str, Any]) -> str:
    """Use the warehouse contract identity, retaining legacy fixture support."""
    claimed = envelope.get("envelope_sha256")
    if isinstance(claimed, str) and len(claimed) == 64:
        try:
            int(claimed, 16)
        except ValueError:
            pass
        else:
            return claimed
    return _sha(envelope)


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
        _require_accepted_submission(response)
        document = {
            "schema": "phase1.warehouse_receipt.v1",
            "feature_id": terminal["feature_id"],
            "key": key,
            "envelope_sha256": _envelope_identity(envelope),
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


def _attempt_directory(config: dict[str, Any], key: str) -> Path:
    return _state_root(config) / "attempts" / key


def _failure_path(config: dict[str, Any], key: str) -> Path:
    return _state_root(config) / "failures" / f"{key}.json"


def _attempt_paths(config: dict[str, Any], key: str) -> list[Path]:
    directory = _attempt_directory(config, key)
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _worker_request(
    config: dict[str, Any], plan: dict[str, Any], item: dict[str, Any], source: InventoryItem,
    attempt_number: int,
) -> dict[str, Any]:
    request = {
        "schema": COLUMN_REQUEST_SCHEMA,
        "feature_id": item["feature_id"],
        "feature_key": item["key"],
        "population_id": plan["population_id"],
        "target_pack": plan["target_pack"],
        "plan_sha256": plan["plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
        "inventory_row_sha256": item["inventory_row_sha256"],
        "inventory_row": source.inventory_row,
        "attempt_number": attempt_number,
    }
    request["request_sha256"] = _sha(request)
    return request


def _parse_worker_result(
    stdout: str, *, feature_id: str, return_code: int
) -> dict[str, Any]:
    if not stdout.strip():
        raise Phase1Refusal("worker returned no canonical JSON result")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise Phase1Refusal("worker stdout is not one canonical JSON result") from error
    if not isinstance(result, dict) or result.get("schema") != COLUMN_RESULT_SCHEMA:
        raise Phase1Refusal("worker returned an unknown result schema")
    if result.get("feature_id") != feature_id:
        raise Phase1Refusal("worker result belongs to another feature")
    if result.get("state") not in {"COMPLETED", "UNAVAILABLE", "FAILED"}:
        raise Phase1Refusal("worker returned an unknown disposition")
    if return_code != 0 and result.get("state") in CLOSURE_TERMINAL_STATES:
        raise Phase1Refusal("worker declared closure evidence with a non-zero return code")
    if result.get("state") == "COMPLETED" and not isinstance(result.get("envelope"), dict):
        raise Phase1Refusal("COMPLETED worker result has no warehouse envelope")
    return result


def _failure_record(
    config: dict[str, Any], plan: dict[str, Any], item: dict[str, Any],
    host_id: str, attempt_count: int, last_attempt: dict[str, Any],
) -> dict[str, Any]:
    document = {
        "schema": "phase1.retry_exhausted.v1",
        "feature_id": item["feature_id"],
        "key": item["key"],
        "host_id": host_id,
        "plan_sha256": plan["plan_sha256"],
        "attempt_count": attempt_count,
        "state": "RETRY_EXHAUSTED",
        "last_attempt_sha256": last_attempt["attempt_sha256"],
    }
    document["failure_sha256"] = _sha(document)
    path = _failure_path(config, item["key"])
    if not path.exists():
        _atomic_json(path, document, exclusive=True)
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
        terminal = _write_unavailable_terminal(config, plan, item, host_id)
        return {"kind": "TERMINAL", "terminal": terminal}
    attempts = _attempt_paths(config, item["key"])
    max_attempts = int(config["retries"]["max_attempts"])
    if len(attempts) >= max_attempts:
        last = json.loads(attempts[-1].read_text(encoding="utf-8"))
        failure = _failure_record(config, plan, item, host_id, len(attempts), last)
        return {"kind": "RETRY_EXHAUSTED", "failure": failure}
    attempt_number = len(attempts) + 1
    request = _worker_request(config, plan, item, source, attempt_number)
    values = {
        "feature_id": item["feature_id"],
        "feature_key": item["key"],
        "host_id": host_id,
        "population_id": plan["population_id"],
        "target_pack": plan["target_pack"],
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
    attempt_path = _attempt_directory(config, item["key"]) / f"{attempt_number:03d}.json"
    started_record = {
        "schema": "phase1.worker_attempt.v1",
        "feature_id": item["feature_id"],
        "key": item["key"],
        "host_id": host_id,
        "attempt_number": attempt_number,
        "state": "RUNNING",
        "request_sha256": request["request_sha256"],
        "command_sha256": _sha(command),
    }
    _atomic_json(attempt_path, started_record, exclusive=True)
    started = time.monotonic()
    return_code: int | None = None
    failure_class: str | None = None
    stdout = ""
    stderr = ""
    try:
        process = subprocess.run(
            command,
            input=json.dumps(request, sort_keys=True),
            check=False,
            timeout=float(config["worker"].get("timeout_seconds", 3600)),
            env=_cpu_environment(config),
            capture_output=True,
            text=True,
        )
        return_code = process.returncode
        stdout, stderr = process.stdout, process.stderr
        result = _parse_worker_result(
            stdout, feature_id=item["feature_id"], return_code=return_code
        )
        if result["state"] == "FAILED":
            declared_class = result.get("failure_class")
            failure_class = (
                declared_class
                if isinstance(declared_class, str) and declared_class
                else "WORKER_FAILED"
            )
    except subprocess.TimeoutExpired:
        result = {
            "schema": COLUMN_RESULT_SCHEMA,
            "feature_id": item["feature_id"],
            "state": "FAILED",
            "reason": "worker timeout",
        }
        failure_class = "TIMEOUT"
    except Exception as error:
        result = {
            "schema": COLUMN_RESULT_SCHEMA,
            "feature_id": item["feature_id"],
            "state": "FAILED",
            "reason": str(error)[:1000],
            "error_type": type(error).__name__,
        }
        failure_class = type(error).__name__
    duration = max(0.0, time.monotonic() - started)
    attempt = {
        "schema": "phase1.worker_attempt.v1",
        "feature_id": item["feature_id"],
        "key": item["key"],
        "host_id": host_id,
        "attempt_number": attempt_number,
        "state": result["state"],
        "request_sha256": request["request_sha256"],
        "command_sha256": _sha(command),
        "duration_seconds": duration,
        "return_code": return_code,
        "failure_class": failure_class,
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
    }
    attempt["attempt_sha256"] = _sha(attempt)
    _atomic_json(attempt_path, attempt)
    if result["state"] in CLOSURE_TERMINAL_STATES:
        terminal = {
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
            "attempt_sha256": attempt["attempt_sha256"],
            "result": result,
        }
        terminal["terminal_sha256"] = _sha(terminal)
        _atomic_json(_terminal_path(config, item["key"]), terminal, exclusive=True)
        return {"kind": "TERMINAL", "terminal": terminal}
    retryable = (
        return_code in set(config["retries"].get("retryable_exit_codes", []))
        or result["state"]
        in set(config["retries"].get("retryable_result_states", []))
        or failure_class
        in set(config["retries"].get("retryable_failure_classes", []))
    )
    if not retryable or attempt_number >= max_attempts:
        failure = _failure_record(
            config, plan, item, host_id, attempt_number, attempt
        )
        return {"kind": "RETRY_EXHAUSTED", "failure": failure}
    return {"kind": "RETRY_SCHEDULED", "attempt": attempt}


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
            if _failure_path(config, item["key"]).is_file():
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
                    outcome = _run_item(config, plan, item, host_id)
                if outcome["kind"] == "RETRY_SCHEDULED":
                    return {
                        "action": "RETRY_SCHEDULED",
                        "host_id": host_id,
                        "feature_id": item["feature_id"],
                    }
                if outcome["kind"] == "RETRY_EXHAUSTED":
                    return {
                        "action": "RETRY_EXHAUSTED",
                        "host_id": host_id,
                        "feature_id": item["feature_id"],
                    }
                terminal = outcome["terminal"]
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


def _failure_coverage(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    failures: dict[str, dict[str, Any]] = {}
    for item in plan["items"]:
        path = _failure_path(config, item["key"])
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        digest = document.pop("failure_sha256", None)
        observed = _sha(document)
        document["failure_sha256"] = digest
        if (
            document.get("schema") != "phase1.retry_exhausted.v1"
            or digest != observed
            or document.get("feature_id") != item["feature_id"]
            or document.get("plan_sha256") != plan["plan_sha256"]
        ):
            raise Phase1Refusal(f"invalid retry-exhausted record {path}")
        failures[item["feature_id"]] = document
    return failures


def _warehouse_identities(
    config: dict[str, Any], plan: dict[str, Any], terminals: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    identities: list[dict[str, Any]] = []
    missing_receipts: list[str] = []
    for item in sorted(plan["items"], key=lambda row: row["feature_id"]):
        terminal = terminals[item["feature_id"]]
        envelope = terminal.get("result", {}).get("envelope")
        envelope_sha256 = _envelope_identity(envelope) if isinstance(envelope, dict) else None
        receipt_sha256 = None
        if terminal["state"] == "COMPLETED":
            if envelope_sha256 is None:
                raise Phase1Refusal(
                    f"{item['feature_id']}: COMPLETED terminal has no envelope"
                )
            receipt_path = (
                _state_root(config) / "warehouse_receipts" / f"{item['key']}.json"
            )
            if not receipt_path.is_file():
                missing_receipts.append(item["feature_id"])
            else:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("feature_id") != item["feature_id"]
                    or receipt.get("key") != item["key"]
                    or receipt.get("envelope_sha256") != envelope_sha256
                ):
                    raise Phase1Refusal(
                        f"{item['feature_id']}: warehouse receipt identity mismatch"
                    )
                receipt_sha256 = _sha(receipt)
        identities.append(
            {
                "feature_id": item["feature_id"],
                "feature_key": item["key"],
                "terminal_state": terminal["state"],
                "terminal_sha256": terminal["terminal_sha256"],
                "envelope_sha256": envelope_sha256,
                "warehouse_receipt_sha256": receipt_sha256,
            }
        )
    return identities, missing_receipts


def _validate_reconciliation(
    response: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise Phase1Refusal("warehouse reconciliation is not an object")
    digest = response.get("reconciliation_sha256")
    unsigned = {key: value for key, value in response.items() if key != "reconciliation_sha256"}
    if digest != _sha(unsigned):
        raise Phase1Refusal("warehouse reconciliation digest is invalid")
    expected = {
        "schema": RECONCILIATION_SCHEMA,
        "state": "RECONCILED",
        "plan_sha256": request["plan_sha256"],
        "expected_count": request["expected_count"],
        "identities_sha256": request["identities_sha256"],
        "authentication_profile": request["authentication_profile"],
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise Phase1Refusal(
                f"warehouse reconciliation {field} differs from the request"
            )
    return response


def _warehouse_reconciliation(
    config: dict[str, Any], plan: dict[str, Any], identities: list[dict[str, Any]],
    reconcile: Reconciler,
) -> dict[str, Any]:
    path = _state_root(config) / "warehouse_reconciliation.json"
    request = {
        "schema": "phase1.warehouse_reconciliation_request.v1",
        "population_id": plan["population_id"],
        "plan_sha256": plan["plan_sha256"],
        "expected_count": plan["inventory_total"],
        "identities_sha256": _sha(identities),
        "identities": identities,
        "authentication_profile": config["warehouse"]["auth_profile"],
    }
    request["request_sha256"] = _sha(request)
    if path.is_file():
        return _validate_reconciliation(
            json.loads(path.read_text(encoding="utf-8")), request
        )
    response = _validate_reconciliation(reconcile(request, config["warehouse"]), request)
    _atomic_json(path, response, exclusive=True)
    return response


def finalize_if_ready(
    config: dict[str, Any], *, submit: Submitter = _http_submit,
    reconcile: Reconciler = _http_reconcile,
) -> dict[str, Any]:
    _ensure_layout(config)
    plan = load_plan(config)
    gate_path = _state_root(config) / "PHASE_1_COMPLETE.json"
    if gate_path.is_file():
        require_phase1_complete(config)
        return {"action": "PHASE_1_COMPLETE", "already_complete": True}
    failures = _failure_coverage(config, plan)
    if failures:
        return {
            "action": "BLOCKED_BY_FAILURES",
            "failed": sorted(failures),
            "observed": len(failures),
        }
    terminals = _terminal_coverage(config, plan)
    if len(terminals) != plan["inventory_total"]:
        return {
            "action": "WAITING_FOR_TERMINALS",
            "expected": plan["inventory_total"],
            "observed": len(terminals),
        }
    _retry_submissions(config, submit)
    identities, missing_receipts = _warehouse_identities(config, plan, terminals)
    if missing_receipts:
        return {
            "action": "WAITING_FOR_WAREHOUSE_RECEIPTS",
            "missing": missing_receipts,
            "observed": plan["inventory_total"] - len(missing_receipts),
        }
    reconciliation = _warehouse_reconciliation(
        config, plan, identities, reconcile
    )
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
        final_envelope = result.get("envelope")
        if not isinstance(final_envelope, dict):
            raise Phase1Refusal("finalizer result has no warehouse envelope")
        final_envelope_sha256 = _sha(final_envelope)
        final_receipt_path = _state_root(config) / "finalizer" / "warehouse_receipt.json"
        if final_receipt_path.is_file():
            final_receipt = json.loads(final_receipt_path.read_text(encoding="utf-8"))
            if final_receipt.get("envelope_sha256") != final_envelope_sha256:
                raise Phase1Refusal("final warehouse receipt belongs to another envelope")
        else:
            try:
                response = submit(final_envelope, config["warehouse"])
                _require_accepted_submission(response)
            except Exception as error:
                return {
                    "action": "WAITING_FOR_FINAL_WAREHOUSE_RECEIPT",
                    "error_type": type(error).__name__,
                    "reason": str(error)[:500],
                }
            final_receipt = {
                "schema": "phase1.final_warehouse_receipt.v1",
                "plan_sha256": plan["plan_sha256"],
                "envelope_sha256": final_envelope_sha256,
                "response": response,
            }
            final_receipt["receipt_sha256"] = _sha(final_receipt)
            _atomic_json(final_receipt_path, final_receipt, exclusive=True)
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
            "final_envelope_sha256": final_envelope_sha256,
            "final_warehouse_receipt_sha256": _sha(final_receipt),
            "warehouse_identity_set_sha256": _sha(identities),
            "warehouse_receipt_set_sha256": _sha(
                sorted(
                    identity["warehouse_receipt_sha256"]
                    for identity in identities
                    if identity["warehouse_receipt_sha256"] is not None
                )
            ),
            "warehouse_reconciliation_sha256": _sha(reconciliation),
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
    failures = _failure_coverage(config, plan)
    if failures:
        raise Phase1Refusal("PHASE_1_COMPLETE coexists with retry-exhausted failures")
    identities, missing_receipts = _warehouse_identities(config, plan, terminals)
    if missing_receipts:
        raise Phase1Refusal("PHASE_1_COMPLETE has missing warehouse receipts")
    if gate.get("warehouse_identity_set_sha256") != _sha(identities):
        raise Phase1Refusal("PHASE_1_COMPLETE warehouse identity set changed")
    receipt_set_sha256 = _sha(
        sorted(
            identity["warehouse_receipt_sha256"]
            for identity in identities
            if identity["warehouse_receipt_sha256"] is not None
        )
    )
    if gate.get("warehouse_receipt_set_sha256") != receipt_set_sha256:
        raise Phase1Refusal("PHASE_1_COMPLETE warehouse receipt set changed")
    reconciliation_path = _state_root(config) / "warehouse_reconciliation.json"
    if not reconciliation_path.is_file():
        raise Phase1Refusal("PHASE_1_COMPLETE has no warehouse reconciliation")
    request = {
        "plan_sha256": plan["plan_sha256"],
        "expected_count": plan["inventory_total"],
        "identities_sha256": _sha(identities),
        "authentication_profile": config["warehouse"]["auth_profile"],
    }
    reconciliation = _validate_reconciliation(
        json.loads(reconciliation_path.read_text(encoding="utf-8")), request
    )
    if gate.get("warehouse_reconciliation_sha256") != _sha(reconciliation):
        raise Phase1Refusal("PHASE_1_COMPLETE warehouse reconciliation changed")
    finalizer_result_path = _state_root(config) / "finalizer" / "result.json"
    final_receipt_path = _state_root(config) / "finalizer" / "warehouse_receipt.json"
    if not finalizer_result_path.is_file() or not final_receipt_path.is_file():
        raise Phase1Refusal("PHASE_1_COMPLETE has no final warehouse receipt")
    finalizer_result = json.loads(finalizer_result_path.read_text(encoding="utf-8"))
    final_envelope = finalizer_result.get("envelope")
    final_receipt = json.loads(final_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(final_envelope, dict):
        raise Phase1Refusal("PHASE_1_COMPLETE finalizer has no warehouse envelope")
    if gate.get("finalizer_result_sha256") != _sha(finalizer_result):
        raise Phase1Refusal("PHASE_1_COMPLETE finalizer result changed")
    if gate.get("final_envelope_sha256") != _sha(final_envelope):
        raise Phase1Refusal("PHASE_1_COMPLETE final envelope changed")
    if final_receipt.get("envelope_sha256") != _sha(final_envelope):
        raise Phase1Refusal("PHASE_1_COMPLETE final receipt identity changed")
    if gate.get("final_warehouse_receipt_sha256") != _sha(final_receipt):
        raise Phase1Refusal("PHASE_1_COMPLETE final warehouse receipt changed")
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
