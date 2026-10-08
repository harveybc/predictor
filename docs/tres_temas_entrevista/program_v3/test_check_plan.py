"""Documentary regression checks; these never authorize an experiment."""

import copy
import json
from pathlib import Path
import unittest

from check_plan import validate


ROOT = Path(__file__).resolve().parent


class PlanChecks(unittest.TestCase):
    def setUp(self):
        self.state = json.loads((ROOT / "PROJECT_METHOD_STATE.json").read_text())

    def test_actual_state(self):
        self.assertEqual(validate(self.state, ROOT), [])

    def test_feature_selector_chat_stays_after_program(self):
        checklist_path = ROOT / "MASTER_CHECKLIST.json"
        checklist = json.loads(checklist_path.read_text())
        item = next(item for item in checklist["items"] if item["id"] == "I12")
        self.assertEqual(item["state"], "NOT_STARTED")
        self.assertIn("I11", item["depends_on"])
        self.assertNotIn("I12", next(i for i in checklist["items"] if i["id"] == "I5").get("depends_on", []))

    def test_denoising_inputs_precedes_architecture_and_not_targets(self):
        checklist = json.loads((ROOT / "MASTER_CHECKLIST.json").read_text())
        items = {item["id"]: item for item in checklist["items"]}
        self.assertEqual(items["I5-P"]["state"], "EVIDENCE_PARTIAL")
        self.assertIn("I5", items["I5-P"]["depends_on"])
        self.assertIn("I5-P", items["I6-A"]["depends_on"])
        self.assertIn("denoising the predictive target or naive during I5-P", checklist["forbidden_shortcuts"])

    def test_news_cannot_authorize_real_capital(self):
        self.state["news_live_track"]["real_capital_authorized"] = True
        self.assertIn("news live scope", validate(self.state, ROOT))

    def test_news_cannot_block_independent_science(self):
        self.state["news_live_track"]["blocks_scientific_experiments"] = True
        self.assertIn("news must not block independent science", validate(self.state, ROOT))

    def test_news_paper_requires_shadow_and_business_contract(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "NEWS-PAPER")
        task["depends_on"] = []
        self.assertIn("news paper prerequisites", validate(self.state, ROOT))

    def test_in_progress_task_is_not_completed(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "SOTA-REPRO")
        task["status"] = "IN_PROGRESS"
        task["evidence"] = []
        self.assertEqual(validate(self.state, ROOT), [])

    def test_unrecognized_task_status_still_refuses(self):
        self.state["tasks"][0]["status"] = "TRUST_ME_DONE"
        self.assertIn("unknown status", " ".join(validate(self.state, ROOT)))

    def test_closure_report_applies_to_current_and_future_orders(self):
        self.state["closure_reporting"]["scope"] = "FUTURE_ONLY"
        self.assertIn("closure reporting contract", " ".join(validate(self.state, ROOT)))

    def test_closure_report_requires_naive_and_literature(self):
        for field in ("naive_error", "literature_value_and_source"):
            state = copy.deepcopy(self.state)
            state["closure_reporting"]["required_columns"].remove(field)
            self.assertIn("closure reporting contract", " ".join(validate(state, ROOT)))

    def test_business_mode_cannot_be_replaced_by_static(self):
        self.state["business_evaluation"]["primary_mode"] = "LITERATURE_STATIC"
        self.assertIn("business weekly evaluation contract", " ".join(validate(self.state, ROOT)))

    def test_business_window_is_four_calendar_years(self):
        self.state["business_evaluation"]["rolling_train_calendar_years"] = 1
        self.assertIn("business weekly evaluation contract", " ".join(validate(self.state, ROOT)))

    def test_frozen_artifact_is_procedure_not_checkpoint(self):
        self.state["business_evaluation"]["frozen_artifact"] = "ONE_STATIC_CHECKPOINT"
        self.assertIn("business weekly evaluation contract", " ".join(validate(self.state, ROOT)))

    def test_warm_update_cannot_claim_strict_four_year_memory(self):
        self.state["business_evaluation"]["strict_memory_claim_allowed_for"].append(
            "WARM_UPDATE_ROLLING_4Y"
        )
        self.assertIn("business weekly evaluation contract", " ".join(validate(self.state, ROOT)))

    def test_literature_policy_cannot_omit_finance(self):
        self.state["literature_comparability"]["scope"] = "PUBLIC_DATA_ONLY"
        self.assertIn("literature comparability contract", " ".join(validate(self.state, ROOT)))

    def test_unmatched_published_scores_are_not_comparators(self):
        self.state["literature_comparability"]["unmatched_published_scores_are_comparators"] = True
        self.assertIn("literature comparability contract", " ".join(validate(self.state, ROOT)))

    def test_benchmark_contract_task_cannot_disappear(self):
        self.state["tasks"] = [t for t in self.state["tasks"] if t["id"] != "BENCHMARK-CONTRACTS"]
        for task in self.state["tasks"]:
            task["depends_on"] = [d for d in task["depends_on"] if d != "BENCHMARK-CONTRACTS"]
        self.assertIn("unknown task BENCHMARK-CONTRACTS", " ".join(validate(self.state, ROOT)))

    def test_financial_loss_comparison_cannot_disappear(self):
        self.state["proposal_coverage"]["P-MOD"].remove("FIN-LOSS-OPT")
        self.state["tasks"] = [t for t in self.state["tasks"] if t["id"] != "FIN-LOSS-OPT"]
        self.assertIn("unknown task FIN-LOSS-OPT", " ".join(validate(self.state, ROOT)))

    def test_financial_loss_policy_must_exist(self):
        self.state["documents"]["financial_loss_policy"] = "missing.md"
        self.assertIn("missing document financial_loss_policy", " ".join(validate(self.state, ROOT)))

    def test_business_weekly_contract_must_exist(self):
        self.state["documents"]["business_weekly"] = "missing.md"
        self.assertIn("missing document business_weekly", " ".join(validate(self.state, ROOT)))

    def test_business_weekly_traceability_must_exist(self):
        self.state["documents"]["business_weekly_traceability"] = "missing.json"
        self.assertIn(
            "missing document business_weekly_traceability",
            " ".join(validate(self.state, ROOT)),
        )

    def test_business_weekly_task_cannot_disappear(self):
        self.state["tasks"] = [
            t for t in self.state["tasks"] if t["id"] != "BUSINESS-WEEKLY-WALK-FORWARD"
        ]
        self.assertIn("unknown task BUSINESS-WEEKLY-WALK-FORWARD", " ".join(validate(self.state, ROOT)))

    def test_business_weekly_requires_business_contract(self):
        task = next(
            t for t in self.state["tasks"] if t["id"] == "BUSINESS-WEEKLY-WALK-FORWARD"
        )
        task["depends_on"] = []
        self.assertIn("business weekly prerequisites", " ".join(validate(self.state, ROOT)))

    def test_financial_loss_requires_business_contract(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "FIN-LOSS-OPT")
        task["depends_on"] = []
        self.assertIn("financial loss prerequisites", " ".join(validate(self.state, ROOT)))

    def test_missing_proposal(self):
        del self.state["proposal_coverage"]["P-L2"]
        self.assertIn("proposal coverage", " ".join(validate(self.state, ROOT)))

    def test_duplicate_step_is_not_coverage(self):
        self.state["signal_steps"][-1] = "STEP-12"
        self.assertIn("signal steps", " ".join(validate(self.state, ROOT)))

    def test_missing_compression(self):
        self.state["compression_lanes"].pop()
        self.assertIn("compression lanes", " ".join(validate(self.state, ROOT)))

    def test_missing_rl_deliverable(self):
        self.state["tasks"] = [x for x in self.state["tasks"] if x["id"] != "MOD-E3"]
        self.assertIn("unknown task MOD-E3", " ".join(validate(self.state, ROOT)))

    def test_cycle(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "MOD-E0-DEV")
        task["depends_on"] = ["MOD-E1"]
        self.assertIn("dependency cycle", " ".join(validate(self.state, ROOT)))

    def test_done_without_evidence(self):
        self.state["tasks"][0]["status"] = "VERIFIED"
        self.state["tasks"][0]["evidence"] = []
        self.assertIn("evidence required", " ".join(validate(self.state, ROOT)))

    def test_unknown_stage(self):
        self.state["current_stage"] = "all_done_trust_me"
        self.assertIn("unknown stage", " ".join(validate(self.state, ROOT)))

    def test_no_owner_or_next_action(self):
        self.state["tasks"][0]["owner"] = ""
        self.state["tasks"][0]["next_action"] = ""
        issues = " ".join(validate(self.state, ROOT))
        self.assertIn("owner", issues)
        self.assertIn("next_action", issues)

    def test_dead_document_link(self):
        self.state["documents"]["master"] = "missing.md"
        self.assertIn("missing document", " ".join(validate(self.state, ROOT)))

    def test_duplicate_task(self):
        self.state["tasks"].append(copy.deepcopy(self.state["tasks"][0]))
        self.assertIn("duplicate task", " ".join(validate(self.state, ROOT)))

    def test_core_hypothesis_cannot_disappear(self):
        self.state["proposal_coverage"]["P-MOD"].remove("MOD-CORE-PRETRAIN")
        self.state["tasks"] = [t for t in self.state["tasks"] if t["id"] != "MOD-CORE-PRETRAIN"]
        self.assertIn("unknown task MOD-CORE-PRETRAIN", " ".join(validate(self.state, ROOT)))

    def test_core_requires_prefix_stage(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "MOD-CORE-PRETRAIN")
        task["depends_on"].remove("MOD-FROZEN-PREFIX")
        self.assertIn("core prerequisites", " ".join(validate(self.state, ROOT)))

    def test_prefix_requires_extractor_development_stage(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "MOD-FROZEN-PREFIX")
        task["depends_on"] = []
        self.assertIn("prefix prerequisites", " ".join(validate(self.state, ROOT)))

    def test_core_hypothesis_document_exists(self):
        self.state["documents"]["core_pretraining"] = "missing.md"
        self.assertIn("missing document core_pretraining", " ".join(validate(self.state, ROOT)))

    def test_architecture_comparison_cannot_disappear(self):
        self.state["proposal_coverage"]["P-MOD"].remove("MOD-ARCH-COMPARE")
        self.state["tasks"] = [t for t in self.state["tasks"] if t["id"] != "MOD-ARCH-COMPARE"]
        for task in self.state["tasks"]:
            task["depends_on"] = [d for d in task["depends_on"] if d != "MOD-ARCH-COMPARE"]
        self.assertIn("unknown task MOD-ARCH-COMPARE", " ".join(validate(self.state, ROOT)))

    def test_e1_requires_architecture_comparison(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "MOD-E1")
        task["depends_on"].remove("MOD-ARCH-COMPARE")
        self.assertIn("E1 prerequisites", " ".join(validate(self.state, ROOT)))

    def test_learning_regimes_cannot_disappear(self):
        self.state.pop("learning_regimes", None)
        self.assertIn("learning regimes contract", " ".join(validate(self.state, ROOT)))

    def test_learning_regimes_cannot_be_redefined_as_representation_arms(self):
        self.state["learning_regimes"] = {"R0": "raw", "R1": "grouping", "R2": "grouping_and_fusion"}
        self.assertIn("learning regimes contract", " ".join(validate(self.state, ROOT)))

    def test_local_terminals_cannot_replace_governed_execution(self):
        self.state["execution_governance"] = {"new_runs": "LOCAL_TERMINALS_THEN_REGISTER", "historical_import": "SEPARATE_RETROSPECTIVE_EVIDENCE"}
        self.assertIn("execution governance contract", " ".join(validate(self.state, ROOT)))

    def test_governance_prerequisite_cannot_disappear(self):
        self.state.pop("execution_governance", None)
        self.assertIn("execution governance contract", " ".join(validate(self.state, ROOT)))

    def test_neat_cannot_become_hyperparameter_optimizer(self):
        self.state["architecture_comparison"]["NEAT_role"] = "HYPERPARAMETER_OPTIMIZER"
        self.assertIn("NEAT role", " ".join(validate(self.state, ROOT)))

    def test_hcore_cannot_precede_e1(self):
        order = self.state["architecture_comparison"]["order"]
        order.remove("H_CORE")
        order.insert(order.index("E1_R0_R1_R2"), "H_CORE")
        self.assertIn("architecture sequence", " ".join(validate(self.state, ROOT)))

    def test_undefined_r3_cannot_be_smuggled_in(self):
        self.state["architecture_comparison"]["R3_defined"] = True
        self.assertIn("undefined R3", " ".join(validate(self.state, ROOT)))

    def test_not_identified_is_not_automatic_rejection(self):
        self.state["feature_selection"]["not_identified_means_rejected"] = True
        self.assertIn("causal abstention semantics", " ".join(validate(self.state, ROOT)))

    def test_future_target_cannot_enter_operational_encoder(self):
        self.state["feature_selection"]["future_target_in_operational_encoder"] = True
        self.assertIn("operational target leakage", " ".join(validate(self.state, ROOT)))

    def test_extractibility_needs_three_controls(self):
        self.state["feature_selection"]["extractibility_controls"].remove("random_encoder")
        self.assertIn("extractibility controls", " ".join(validate(self.state, ROOT)))

    def test_calendar_now_is_only_evidence_for_selection(self):
        self.state["feature_selection"]["calendar_episode_use_now"] = "PREDICTOR_INPUT"
        self.assertIn("calendar episode selection scope", " ".join(validate(self.state, ROOT)))

    def test_calendar_model_input_remains_final_and_optional(self):
        self.state["feature_selection"]["calendar_as_model_input"] = "RUN_NOW"
        self.assertIn("calendar model input deferral", " ".join(validate(self.state, ROOT)))

    def test_calendar_task_requires_primary_model_and_paper_baseline(self):
        task = next(t for t in self.state["tasks"] if t["id"] == "CAL-CAUSAL-INPUT")
        task["depends_on"] = []
        self.assertIn("causal calendar input prerequisites", " ".join(validate(self.state, ROOT)))

    def test_checklist_is_required(self):
        self.state["documents"]["checklist"] = "missing.json"
        self.assertIn("missing document checklist", " ".join(validate(self.state, ROOT)))

    def test_phase23_selection_plan_is_required(self):
        self.state["documents"]["feature_selection_phase23"] = "missing.md"
        self.assertIn(
            "missing document feature_selection_phase23",
            " ".join(validate(self.state, ROOT)),
        )


if __name__ == "__main__":
    unittest.main()
