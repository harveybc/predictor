#!/usr/bin/env python3
"""Bridge phase-1 orchestration closure to retained population evidence.

EURUSD has a complete, digest-reconciled causal population that can be adopted.
ETH deliberately remains provisional: the pinned stdio worker does not transport
the raw per-column causal cells needed for a population-wide BH/FDR closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verified_terminals(plan: dict[str, Any], terminal_dir: Path) -> list[dict[str, Any]]:
    expected = {item["feature_id"] for item in plan.get("items", [])}
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted(terminal_dir.glob("*.json")):
        terminal = _load(path)
        claimed = terminal.pop("terminal_sha256", None)
        if claimed != _digest(terminal):
            raise ValueError(f"terminal digest mismatch: {path}")
        terminal["terminal_sha256"] = claimed
        if terminal.get("plan_sha256") != plan.get("plan_sha256"):
            raise ValueError(f"terminal belongs to another plan: {path}")
        feature = terminal.get("feature_id")
        if feature in observed:
            raise ValueError(f"duplicate terminal for {feature}")
        observed[feature] = terminal
    if set(observed) != expected:
        raise ValueError("terminal population differs from the orchestrator plan")
    return [observed[name] for name in sorted(observed)]


def finalize(population: str, plan_path: Path, terminal_dir: Path, output: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    terminals = _verified_terminals(plan, terminal_dir)
    if population == "EURUSD":
        bundle_root = Path(__file__).resolve().parent.parent
        adoption_path = bundle_root / "adoption" / "ADOPTION.json"
        adoption = _load(adoption_path)
        claimed = adoption.pop("adoption_sha256", None)
        if claimed != _digest(adoption):
            raise ValueError("EURUSD adoption report digest mismatch")
        adoption["adoption_sha256"] = claimed
        if adoption.get("state") != "ADOPTED_VERIFIED_EVIDENCE":
            raise ValueError("EURUSD retained evidence was not adopted")
        if int(adoption.get("feature_count", -1)) != len(terminals):
            raise ValueError("EURUSD adopted and worker populations differ")
        result = {
            "schema": "phase1.finalizer_result.v1",
            "state": "PHASE_1_COMPLETE",
            "terminal_count": len(terminals),
            "closure": "ADOPTED_VERIFIED_EVIDENCE",
            "adoption_sha256": claimed,
        }
    elif population == "ETH":
        result = {
            "schema": "phase1.finalizer_result.v1",
            "state": "WAITING_FOR_GLOBAL_CAUSAL_FINALIZATION",
            "terminal_count": len(terminals),
            "reason": (
                "ETH is provisional and pinned stdio results do not transport "
                "the raw causal cells required for population-wide BH/FDR"
            ),
        }
    else:
        raise ValueError(f"unsupported population: {population}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("EURUSD", "ETH"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--terminals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        finalize(args.population, args.plan, args.terminals, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(f"REFUSED: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
