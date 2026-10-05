#!/usr/bin/env python3
"""Read-only JSON status for the phase-1 inventory orchestrator."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tools.phase1_inventory_orchestrator import (
    Phase1Refusal,
    _read_terminal,
    _state_root,
    _terminal_path,
    load_config,
    load_plan,
    require_phase1_complete,
)


STATUS_SCHEMA = "phase1.inventory_status.v1"


def _active_claims(config: dict[str, Any]) -> dict[str, str | None]:
    current = {host["id"]: None for host in config["hosts"]}
    root = _state_root(config) / "claims"
    if not root.is_dir():
        return current
    for claim in sorted(root.glob("*.claim")):
        try:
            document = json.loads((claim / "claim.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        host_id = document.get("host_id")
        if host_id in current:
            if current[host_id] is not None:
                raise Phase1Refusal(f"host {host_id!r} has multiple active series claims")
            current[host_id] = document.get("feature_id")
    return dict(sorted(current.items()))


def _eta(
    plan: dict[str, Any],
    terminals: dict[str, dict[str, Any]],
) -> float | None:
    pending = [item for item in plan["items"] if item["feature_id"] not in terminals]
    if not pending:
        return 0.0
    rates_by_host: dict[str, list[float]] = {}
    all_rates: list[float] = []
    for terminal in terminals.values():
        cost = float(terminal.get("estimated_cost", 0))
        duration = float(terminal.get("duration_seconds", 0))
        if cost > 0 and duration >= 0 and math.isfinite(duration):
            rate = duration / cost
            rates_by_host.setdefault(terminal["host_id"], []).append(rate)
            all_rates.append(rate)
    if not all_rates:
        return None
    fallback = statistics.median(all_rates)
    seconds_by_host: dict[str, float] = {}
    for item in pending:
        host_id = item["host_id"]
        samples = rates_by_host.get(host_id)
        rate = statistics.median(samples) if samples else fallback
        seconds_by_host[host_id] = seconds_by_host.get(host_id, 0.0) + (
            float(item["estimated_cost"]) * rate
        )
    return round(max(seconds_by_host.values(), default=0.0), 3)


def build_status(config: dict[str, Any]) -> dict[str, Any]:
    plan = load_plan(config)
    current = _active_claims(config)
    terminals: dict[str, dict[str, Any]] = {}
    integrity_errors: list[dict[str, str]] = []
    occupied_terminal_slots = 0
    for item in plan["items"]:
        path = _terminal_path(config, item["key"])
        if not path.is_file():
            continue
        occupied_terminal_slots += 1
        try:
            terminal = _read_terminal(path)
            if terminal.get("feature_id") != item["feature_id"]:
                raise Phase1Refusal("terminal feature identity differs from the plan")
            terminals[item["feature_id"]] = terminal
        except Exception as error:
            integrity_errors.append(
                {"feature_id": item["feature_id"], "reason": str(error)}
            )
    counts = {state: 0 for state in ("COMPLETED", "FAILED", "UNAVAILABLE")}
    for terminal in terminals.values():
        counts[terminal["state"]] += 1
    running = {feature_id for feature_id in current.values() if feature_id is not None}
    pending = plan["inventory_total"] - occupied_terminal_slots - len(running)
    if pending < 0:
        raise Phase1Refusal("terminal and active populations exceed the plan denominator")
    try:
        require_phase1_complete(config)
        phase_state = "PHASE_1_COMPLETE"
    except Phase1Refusal:
        phase_state = "PHASE_1_RUNNING"
    receipts = _state_root(config) / "warehouse_receipts"
    report = {
        "schema": STATUS_SCHEMA,
        "phase_state": phase_state,
        "plan_sha256": plan["plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
        "total": plan["inventory_total"],
        "completed": counts["COMPLETED"],
        "failed": counts["FAILED"] + len(integrity_errors),
        "unavailable": counts["UNAVAILABLE"],
        "pending": pending,
        "running": len(running),
        "current": current,
        "eta_seconds": _eta(plan, terminals),
        "warehouse_submitted": len(list(receipts.glob("*.json"))) if receipts.is_dir() else 0,
        "integrity_errors": integrity_errors,
    }
    report["denominator_reconciles"] = (
        report["completed"]
        + report["failed"]
        + report["unavailable"]
        + report["pending"]
        + report["running"]
        == report["total"]
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = build_status(load_config(arguments.config))
    except Phase1Refusal as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
