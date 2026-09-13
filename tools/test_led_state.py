#!/usr/bin/env python3
"""Validate and summarize a persistent DGPD project state."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = "dgpd_project_state.v1"
METHOD_VERSION = "1.0"
STAGES = [
    "S0_DISCOVERY",
    "S1_REQUIREMENTS",
    "S2_USE_CASES",
    "S3_ACCEPTANCE_TEST_DESIGN",
    "S4_ARCHITECTURE",
    "S5_SYSTEM_TEST_DESIGN",
    "S6_COMPONENT_DESIGN",
    "S7_INTEGRATION_TEST_DESIGN",
    "S8_UNIT_TEST_DESIGN",
    "S9_UNIT_IMPLEMENTATION",
    "S10_INTEGRATION_VERIFICATION",
    "S11_SYSTEM_VERIFICATION",
    "S12_ALPHA_ACCEPTANCE",
    "S13_BETA_ACCEPTANCE",
    "S14_RELEASE_OPERATION",
]
STATUSES = {"NOT_STARTED", "IN_PROGRESS", "PASSED", "BLOCKED", "DEFERRED"}
TOP_KEYS = {
    "schema",
    "method_version",
    "project_id",
    "objective",
    "updated_at",
    "current_stage",
    "stages",
    "requirements",
    "traceability_file",
    "next_allowed_actions",
    "blocked_actions",
    "open_decisions",
    "history",
}
STAGE_KEYS = {"id", "status", "optional", "artifacts", "evidence", "review"}
REQUIREMENT_KEYS = {"id", "text", "status", "source"}
HISTORY_KEYS = {"at", "stage", "event"}


class DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("state root must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], where: str, errors: list[str]) -> None:
    observed = set(value)
    if observed != expected:
        errors.append(
            f"{where}: keys differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _valid_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _local_reference_exists(repo_root: Path, reference: str) -> bool:
    if reference.startswith(("http://", "https://")):
        return True
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (repo_root / path).exists()


def validate_state(state: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    _exact_keys(state, TOP_KEYS, "state", errors)
    if errors:
        return errors

    if state["schema"] != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if state["method_version"] != METHOD_VERSION:
        errors.append(f"method_version must be {METHOD_VERSION}")
    if not isinstance(state["project_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]+", state["project_id"]
    ):
        errors.append("project_id has an invalid format")
    if not isinstance(state["objective"], str) or len(state["objective"].strip()) < 10:
        errors.append("objective must be a descriptive string")
    if not _valid_datetime(state["updated_at"]):
        errors.append("updated_at must be an ISO-8601 datetime with timezone")

    stages = state["stages"]
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        errors.append(f"stages must contain exactly {len(STAGES)} entries")
        return errors

    stage_ids: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"stages[{index}] must be an object")
            continue
        _exact_keys(stage, STAGE_KEYS, f"stages[{index}]", errors)
        if set(stage) != STAGE_KEYS:
            continue
        stage_ids.append(stage["id"])
        if stage["status"] not in STATUSES:
            errors.append(f"stages[{index}].status is invalid")
        if not isinstance(stage["optional"], bool):
            errors.append(f"stages[{index}].optional must be boolean")
        for field in ("artifacts", "evidence"):
            if not isinstance(stage[field], list) or not all(
                isinstance(item, str) and item for item in stage[field]
            ):
                errors.append(f"stages[{index}].{field} must be a string list")
        if stage["review"] is not None and not isinstance(stage["review"], str):
            errors.append(f"stages[{index}].review must be string or null")
        if stage["status"] == "DEFERRED" and not stage["optional"]:
            errors.append(f"{stage['id']}: only optional stages may be deferred")
        if stage["status"] == "PASSED":
            if not stage["evidence"] or not stage["review"]:
                errors.append(f"{stage['id']}: PASSED requires evidence and review")
            for reference in stage["artifacts"] + stage["evidence"]:
                if not _local_reference_exists(repo_root, reference):
                    errors.append(f"{stage['id']}: missing or unsafe reference {reference}")

    if stage_ids != STAGES:
        errors.append("stage IDs or order differ from the DGPD sequence")
        return errors

    current = state["current_stage"]
    if current not in STAGES:
        errors.append("current_stage is not a DGPD stage")
        return errors
    current_index = STAGES.index(current)
    current_status = stages[current_index]["status"]
    if current_status not in {"IN_PROGRESS", "BLOCKED"}:
        errors.append("current_stage must be IN_PROGRESS or BLOCKED")
    for index, stage in enumerate(stages):
        if index < current_index and stage["status"] not in {"PASSED", "DEFERRED"}:
            errors.append(f"{stage['id']}: prior stage is not closed")
        if index > current_index and stage["status"] not in {"NOT_STARTED", "DEFERRED"}:
            errors.append(f"{stage['id']}: future stage started before its gate")

    requirements = state["requirements"]
    if not isinstance(requirements, list):
        errors.append("requirements must be a list")
    else:
        ids: list[str] = []
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, dict):
                errors.append(f"requirements[{index}] must be an object")
                continue
            _exact_keys(requirement, REQUIREMENT_KEYS, f"requirements[{index}]", errors)
            if set(requirement) != REQUIREMENT_KEYS:
                continue
            ids.append(requirement["id"])
            if not re.fullmatch(r"[A-Z][A-Z0-9_-]+-[0-9]{3}", requirement["id"]):
                errors.append(f"requirements[{index}].id is invalid")
            if requirement["status"] not in {"DRAFT", "ACCEPTED", "SUPERSEDED"}:
                errors.append(f"requirements[{index}].status is invalid")
            if not all(
                isinstance(requirement[field], str) and requirement[field].strip()
                for field in ("text", "source")
            ):
                errors.append(f"requirements[{index}] text/source must be nonempty")
        if len(ids) != len(set(ids)):
            errors.append("requirement IDs must be unique")

    traceability = state["traceability_file"]
    if not isinstance(traceability, str) or not _local_reference_exists(repo_root, traceability):
        errors.append("traceability_file is missing or unsafe")
    for field in ("next_allowed_actions", "blocked_actions", "open_decisions"):
        value = state[field]
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            errors.append(f"{field} must be a string list")
    if isinstance(state["next_allowed_actions"], list) and not state["next_allowed_actions"]:
        errors.append("next_allowed_actions must not be empty")

    history = state["history"]
    if not isinstance(history, list) or not history:
        errors.append("history must be a nonempty list")
    else:
        for index, event in enumerate(history):
            if not isinstance(event, dict):
                errors.append(f"history[{index}] must be an object")
                continue
            _exact_keys(event, HISTORY_KEYS, f"history[{index}]", errors)
            if set(event) != HISTORY_KEYS:
                continue
            if not _valid_datetime(event["at"]):
                errors.append(f"history[{index}].at is invalid")
            if event["stage"] not in STAGES:
                errors.append(f"history[{index}].stage is invalid")
            if not isinstance(event["event"], str) or not event["event"].strip():
                errors.append(f"history[{index}].event is empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "status"))
    parser.add_argument("state", type=Path)
    args = parser.parse_args()

    state_path = args.state.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        state = load_state(state_path)
    except ValueError as exc:
        print(f"DGPD_STATE_INVALID: {exc}")
        return 2
    errors = validate_state(state, repo_root)
    if errors:
        print("DGPD_STATE_INVALID")
        for error in errors:
            print(f"- {error}")
        return 2
    if args.command == "status":
        print(f"project: {state['project_id']}")
        print(f"stage: {state['current_stage']}")
        print("next:")
        for action in state["next_allowed_actions"]:
            print(f"- {action}")
    else:
        print(f"DGPD_STATE_OK: {state['project_id']} at {state['current_stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
