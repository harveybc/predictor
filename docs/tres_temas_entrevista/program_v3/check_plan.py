"""Check the program's documentary coverage, never scientific validity."""

import json
from pathlib import Path


REQUIRED = {
    "P-MOD": {"MOD-E0-DEV", "MOD-E1", "MOD-CONF", "MOD-E3"},
    "P-L2": {"L2-PUBLIC-RL"},
    "P-CAP": {"CAP-CALIBRATION"},
    "P-PRE": {"PRE-NOISE"},
    "P-TRN": {"TRN-TRANSFER"},
    "P-INC": {"INC-SIMULATION"},
}
STAGES = {
    "discovery", "requirements", "acceptance_test_design", "architecture",
    "system_test_design", "component_design", "integration_test_design",
    "unit_test_design", "implementation", "unit_verification",
    "integration_verification", "system_verification", "alpha_acceptance",
    "external_review", "release",
}


def validate(state, root):
    issues = []
    if state.get("schema") != "research_program_method_state.v1":
        issues.append("unknown schema")
    if state.get("current_stage") not in STAGES:
        issues.append("unknown stage")
    if not state.get("next_allowed_actions"):
        issues.append("next actions required")

    def exact_list(field, expected, label):
        values = state.get(field, [])
        if not isinstance(values, list) or sorted(values) != sorted(expected):
            issues.append(label)

    exact_list("signal_steps", [f"STEP-{i:02}" for i in range(1, 14)], "signal steps")
    exact_list("compression_lanes", [f"C{i}" for i in range(1, 8)], "compression lanes")
    exact_list("shared_requirements", ["FE", "EVENT", "META", "I-INFO", "H-ES",
                                      "GOVERNANCE", "WEEKLY-RL"], "shared requirements")
    coverage = state.get("proposal_coverage", {})
    if set(coverage) != set(REQUIRED):
        issues.append("proposal coverage")
    for proposal, tasks in REQUIRED.items():
        if not tasks.issubset(coverage.get(proposal, [])):
            issues.append(f"proposal coverage {proposal}")

    documents = state.get("documents", {})
    for name in ("master", "metrics", "orders"):
        path = documents.get(name)
        if not path or not (root / path).is_file():
            issues.append(f"missing document {name}")

    tasks = state.get("tasks", [])
    by_id = {}
    for task in tasks:
        task_id = task.get("id", "")
        if not task_id or task_id in by_id:
            issues.append(f"duplicate task or missing id: {task_id}")
        by_id[task_id] = task
        for field in ("owner", "next_action", "deliverable"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                issues.append(f"{task_id}: {field} required")
        status = task.get("status")
        if status not in {"NOT_STARTED", "DESIGNED", "IMPLEMENTED", "EXECUTED",
                          "VERIFIED", "REVIEWED", "BLOCKED"}:
            issues.append(f"{task_id}: unknown status")
        if status in {"EXECUTED", "VERIFIED", "REVIEWED"} and not task.get("evidence"):
            issues.append(f"{task_id}: evidence required")
    required_tasks = set().union(*REQUIRED.values(), {"BUSINESS-CONTRACT"})
    required_tasks.add(state.get("first_experiment"))
    for task_id in sorted(required_tasks, key=str):
        if task_id not in by_id:
            issues.append(f"unknown task {task_id}")

    # Check the whole dependency graph, including branches not first in the queue.
    visiting, visited = set(), set()

    def visit(task_id):
        if task_id not in by_id:
            issues.append(f"unknown dependency {task_id}")
            return
        if task_id in visiting:
            issues.append(f"dependency cycle at {task_id}")
            return
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].get("depends_on", []):
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)
    return issues


def main():
    root = Path(__file__).resolve().parent
    state = json.loads((root / "PROJECT_METHOD_STATE.json").read_text())
    issues = validate(state, root)
    print(json.dumps({"documentary_coverage": "FAIL" if issues else "PASS",
                      "scientific_approval": False, "issues": issues}, indent=2))
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main())
