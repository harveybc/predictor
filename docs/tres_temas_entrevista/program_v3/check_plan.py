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
ARCHITECTURE_ORDER = [
    "dataset_target_split_contract",
    "dataset_specific_selection",
    "grouping_and_receptive_fields",
    "matched_ARCH_controls",
    "branch_pretraining",
    "E1_R0_R1_R2",
    "select_and_fix_branch_prefix",
    "H_CORE",
    "core_transfer",
    "freeze_final_representation",
    "Dense_vs_NEAT",
    "SAC_and_DQN_raw_vs_modular",
    "deferred_causal_calendar_model_input",
]
BUSINESS_EVALUATION = {
    "primary_mode": "BUSINESS_WEEKLY_WALK_FORWARD",
    "sensitivity_mode": "BUSINESS_MONTHLY_WALK_FORWARD",
    "literature_mode": "LITERATURE_STATIC",
    "rolling_train_calendar_years": 4,
    "validation_and_test": "COMPLETE_CONSECUTIVE_ELIGIBLE_WEEKS_WITH_DISPOSITIONS",
    "frozen_artifact": "UPDATE_AND_DECISION_PROCEDURE_NOT_ONE_STATIC_CHECKPOINT",
    "update_modes": ["FULL_RETRAIN_ROLLING_4Y", "WARM_UPDATE_ROLLING_4Y"],
    "strict_memory_claim_allowed_for": ["FULL_RETRAIN_ROLLING_4Y"],
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
    if state.get("business_evaluation") != BUSINESS_EVALUATION:
        issues.append("business weekly evaluation contract")

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
    for name in ("master", "metrics", "orders", "queue", "checklist", "feature_selection",
                 "modular_stack", "core_pretraining", "financial_loss_policy",
                 "business_weekly", "business_weekly_traceability"):
        path = documents.get(name)
        if not path or not (root / path).is_file():
            issues.append(f"missing document {name}")

    news = state.get("news_live_track", {})
    if news.get("allowed_modes") != ["SHADOW", "MT5_DEMO", "ALPACA_PAPER"] or news.get("real_capital_authorized") is not False:
        issues.append("news live scope")
    if news.get("blocks_scientific_experiments") is not False:
        issues.append("news must not block independent science")
    if not documents.get("news_live") or not (root / documents["news_live"]).is_file():
        issues.append("missing news live document")

    architecture = state.get("architecture_comparison", {})
    if architecture.get("order") != ARCHITECTURE_ORDER:
        issues.append("architecture sequence")
    if architecture.get("R3_defined") is not False:
        issues.append("undefined R3 must remain absent")
    if architecture.get("NEAT_role") != "LATE_EVOLVED_HEAD_NOT_HYPERPARAMETER_OPTIMIZER":
        issues.append("NEAT role")
    if architecture.get("DEAP_role") != "CONFIGURATION_AND_HYPERPARAMETER_SEARCH":
        issues.append("DEAP role")
    if architecture.get("DOIN_role") != "DISTRIBUTED_CANDIDATE_EVALUATION":
        issues.append("DOIN role")

    selection = state.get("feature_selection", {})
    if selection.get("business_manifest") != "EURUSD_FIRST":
        issues.append("business feature manifest")
    if selection.get("not_identified_means_rejected") is not False:
        issues.append("causal abstention semantics")
    if selection.get("future_target_in_operational_encoder") is not False:
        issues.append("operational target leakage")
    if selection.get("extractibility_controls") != ["raw", "random_encoder", "trained_encoder"]:
        issues.append("extractibility controls")
    if selection.get("calendar_episode_use_now") != "TREATMENT_CONTROL_DISCOVERY_FOR_FEATURE_SELECTION_ONLY":
        issues.append("calendar episode selection scope")
    if selection.get("calendar_as_model_input") != "DEFERRED_FINAL_OPTIONAL_AFTER_NEAT_RL_AND_PAPER_BASELINE":
        issues.append("calendar model input deferral")

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
        if status not in {"NOT_STARTED", "IN_PROGRESS", "DESIGNED", "IMPLEMENTED", "EXECUTED",
                          "VERIFIED", "REVIEWED", "BLOCKED"}:
            issues.append(f"{task_id}: unknown status")
        if status in {"EXECUTED", "VERIFIED", "REVIEWED"} and not task.get("evidence"):
            issues.append(f"{task_id}: evidence required")
    required_tasks = set().union(
        *REQUIRED.values(),
        {"BUSINESS-CONTRACT", "BUSINESS-WEEKLY-WALK-FORWARD", "BENCHMARK-CONTRACTS"},
    )
    required_tasks.add(state.get("first_experiment"))
    required_tasks.update({"NEWS-ADAPTER", "NEWS-SHADOW", "NEWS-PAPER", "CAL-CAUSAL-INPUT"})
    for task_id in sorted(required_tasks, key=str):
        if task_id not in by_id:
            issues.append(f"unknown task {task_id}")

    required_dependencies = {
        "BUSINESS-WEEKLY-WALK-FORWARD": ({"BUSINESS-CONTRACT"}, "business weekly prerequisites"),
        "FIN-LOSS-OPT": ({"BUSINESS-CONTRACT", "BENCHMARK-CONTRACTS"}, "financial loss prerequisites"),
        "MOD-E1": ({"MOD-E0-DEV", "MOD-ARCH-COMPARE"}, "E1 prerequisites"),
        "MOD-FROZEN-PREFIX": ({"MOD-E1"}, "prefix prerequisites"),
        "MOD-CORE-PRETRAIN": ({"MOD-E1", "MOD-FROZEN-PREFIX"}, "core prerequisites"),
        "NEWS-SHADOW": ({"NEWS-ADAPTER"}, "news shadow prerequisites"),
        "NEWS-PAPER": ({"NEWS-SHADOW", "BUSINESS-CONTRACT"}, "news paper prerequisites"),
        "CAL-CAUSAL-INPUT": ({"MOD-E3", "NEWS-PAPER"}, "causal calendar input prerequisites"),
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

    queue_path = documents.get("queue")
    if queue_path and (root / queue_path).is_file():
        queue = json.loads((root / queue_path).read_text())
        queued = {lane.get("id"): lane for lane in queue.get("lanes", [])}
        neat = queued.get("HEAD-DENSE-NEAT", {})
        if "MOD-HCORE" not in neat.get("depends_on", []):
            issues.append("NEAT queue sequence")
        hcore = queued.get("MOD-HCORE", {})
        if "MOD-ARCH-E1" not in hcore.get("depends_on", []):
            issues.append("H-CORE queue sequence")
        if queue.get("superseded", {}).get("all_321_or_all_83_as_selected") is not False:
            issues.append("mechanical inventory cannot equal selection")
        calendar_input = queued.get("CALENDAR-CAUSAL-MODEL-INPUT", {})
        if calendar_input.get("state") != "DEFERRED_FINAL_OPTIONAL":
            issues.append("calendar model input queue state")
        if not {"HEAD-DENSE-NEAT", "RL-RAW-MODULAR", "TRADING-PAPER"}.issubset(
                calendar_input.get("depends_on", [])):
            issues.append("calendar model input queue sequence")

    checklist_path = documents.get("checklist")
    if checklist_path and (root / checklist_path).is_file():
        checklist = json.loads((root / checklist_path).read_text())
        items = {item.get("id"): item for item in checklist.get("items", [])}
        required_checklist = {"I0", "I0-W", "I1", "I2", "I3", "I4-C", "I4-R", "I5",
                              "I6-A", "I6-B", "I7", "I7-H", "I8", "I9-N",
                              "I9-R", "I10", "I11"}
        if set(items) != required_checklist:
            issues.append("master checklist coverage")
        if "I7" not in items.get("I7-H", {}).get("depends_on", []):
            issues.append("checklist H-CORE sequence")
        if "I0" not in items.get("I0-W", {}).get("depends_on", []):
            issues.append("checklist business weekly sequence")
        if "I0-W" not in items.get("I5", {}).get("depends_on", []):
            issues.append("checklist selection business sequence")
        if "I7-H" not in items.get("I9-N", {}).get("depends_on", []):
            issues.append("checklist NEAT sequence")
        if not {"I9-N", "I9-R", "I10"}.issubset(items.get("I11", {}).get("depends_on", [])):
            issues.append("checklist calendar input sequence")

    traceability_path = documents.get("business_weekly_traceability")
    if traceability_path and (root / traceability_path).is_file():
        traceability = json.loads((root / traceability_path).read_text())
        requirements = traceability.get("requirements", [])
        ids = [item.get("id") for item in requirements]
        if ids != [f"BW{i:02}" for i in range(1, 19)]:
            issues.append("business weekly traceability coverage")
        allowed = {"IMPLEMENTED", "PARTIAL", "PLANNED"}
        if any(item.get("state") not in allowed for item in requirements):
            issues.append("business weekly traceability state")
        for item in requirements:
            if item.get("state") == "IMPLEMENTED" and not item.get("evidence"):
                issues.append(f"business weekly traceability evidence {item.get('id')}")
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
