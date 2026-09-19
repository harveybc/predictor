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
        self.state["tasks"][0]["depends_on"] = ["MOD-E1"]
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


if __name__ == "__main__":
    unittest.main()
