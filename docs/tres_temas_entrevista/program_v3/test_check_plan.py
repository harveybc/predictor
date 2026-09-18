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


if __name__ == "__main__":
    unittest.main()
