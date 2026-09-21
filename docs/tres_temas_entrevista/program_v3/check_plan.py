"""Check the program's documentary coverage, never scientific validity."""

import json
from pathlib import Path


REQUIRED = {
    "P-MOD": {"MOD-E0-DEV", "MOD-ARCH-COMPARE", "MOD-E1",
              "MOD-FROZEN-PREFIX", "MOD-CORE-PRETRAIN", "MOD-CONF", "MOD-E3",
              "FIN-LOSS-OPT"},
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
LEARNING_REGIMES = {
    "R0": {"detector_initialization": "RANDOM", "detector_update": "TRAINABLE", "rest_update": "TRAINABLE"},
    "R1": {"detector_initialization": "SHARED_PRETRAINED", "detector_update": "FROZEN", "rest_update": "TRAINABLE"},
    "R2": {"detector_initialization": "SHARED_PRETRAINED", "detector_update": "TRAINABLE", "rest_update": "TRAINABLE"},
}
EXECUTION_GOVERNANCE = {
    "new_runs": "REGISTER_AND_DELIVER_BEFORE_DATA_PREPARATION_OR_FIT",
    "historical_import": "SEPARATE_RETROSPECTIVE_EVIDENCE",
}
LITERATURE_COMPARABILITY = {
    "scope": "ALL_DOMAINS_INCLUDING_FINANCE",
    "required_before": "NEW_SCIENTIFIC_TRAINING",
    "modes": ["REPRODUCTION", "MATCHED_DOMAIN_COMPARISON"],
    "unmatched_published_scores_are_comparators": False,
}
CLOSURE_REPORTING = {
    "scope": "CURRENT_AND_ALL_FUTURE_ORDER_SETS",
    "required_columns": ["task_horizon_split", "metric_and_scale", "model_error",
                         "naive_error", "skill_vs_naive", "literature_value_and_source",
                         "comparability_status"],
    "missing_comparator": "EXPLICIT_NOT_COMPARABLE_WITH_REASON_NO_INVENTED_VALUE",
    "no_new_measurement": "LABEL_PRIOR_VERIFIED_RESULT_OR_NO_NEW_MEASUREMENT",
}


def validate(state, root):
    issues = []
    if state.get("schema") != "research_program_method_state.v1":
        issues.append("unknown schema")
    if state.get("current_stage") not in STAGES:
        issues.append("unknown stage")
    if not state.get("next_allowed_actions"):
        issues.append("next actions required")
    if state.get("learning_regimes") != LEARNING_REGIMES:
        issues.append("learning regimes contract")
    if state.get("execution_governance") != EXECUTION_GOVERNANCE:
        issues.append("execution governance contract")
    if state.get("literature_comparability") != LITERATURE_COMPARABILITY:
        issues.append("literature comparability contract")
    if state.get("closure_reporting") != CLOSURE_REPORTING:
        issues.append("closure reporting contract")

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
    for name in ("master", "metrics", "orders", "core_pretraining", "financial_loss_policy"):
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
    required_tasks = set().union(*REQUIRED.values(), {"BUSINESS-CONTRACT", "BENCHMARK-CONTRACTS"})
    required_tasks.add(state.get("first_experiment"))
    for task_id in sorted(required_tasks, key=str):
        if task_id not in by_id:
            issues.append(f"unknown task {task_id}")

    required_dependencies = {
        "FIN-LOSS-OPT": ({"BUSINESS-CONTRACT", "BENCHMARK-CONTRACTS"}, "financial loss prerequisites"),
        "MOD-E1": ({"MOD-E0-DEV", "MOD-ARCH-COMPARE"}, "E1 prerequisites"),
        "MOD-FROZEN-PREFIX": ({"MOD-E1"}, "prefix prerequisites"),
        "MOD-CORE-PRETRAIN": ({"MOD-E1", "MOD-FROZEN-PREFIX"}, "core prerequisites"),
    }
    for task_id, (dependencies, label) in required_dependencies.items():
        if not dependencies.issubset(by_id.get(task_id, {}).get("depends_on", [])):
            issues.append(label)

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
