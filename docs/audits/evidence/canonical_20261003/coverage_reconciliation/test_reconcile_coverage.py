import copy
import unittest
from pathlib import Path

from reconcile_coverage import build_report, validate_report


REPO_ROOT = Path(__file__).resolve().parents[5]


class CoverageReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report(REPO_ROOT)

    def test_campaign_denominators_and_partition(self):
        summary = self.report["summary"]
        self.assertEqual(summary["admissible_features"], 366)
        self.assertEqual(summary["ps3c_join_features"], 279)
        self.assertEqual(summary["low_priority_outside_join"], 87)
        self.assertEqual(summary["extractibility_queue_features"], 137)
        self.assertEqual(summary["join_features_outside_extractibility_queue"], 142)
        self.assertEqual(summary["outside_queue_tier3_only"], 132)
        self.assertEqual(summary["outside_queue_calendar_conditioning"], 10)
        self.assertEqual(summary["unaccounted_features"], 0)

    def test_every_admissible_feature_has_one_auditable_row(self):
        rows = self.report["features"]
        keys = {(row["batch"], row["feature_id"]) for row in rows}
        self.assertEqual(len(rows), 366)
        self.assertEqual(len(keys), 366)
        self.assertTrue(all(row["admissible"] for row in rows))
        self.assertTrue(all("selected" not in " ".join(row.keys()).lower() for row in rows))

    def test_excluded_87_have_explicit_low_priority_reason_and_all_cells(self):
        omitted = [row for row in self.report["features"] if not row["in_ps3c_join"]]
        self.assertEqual(len(omitted), 87)
        for row in omitted:
            self.assertEqual(row["coverage_status"], "LOW_PRIORITY_OUTSIDE_PS3C_JOIN")
            self.assertEqual(row["outside_join_reason"], "PROVISIONAL_LOW_PRIORITY_IN_ALL_14_CELLS")
            self.assertEqual(row["ps2_cell_count"], 14)
            self.assertEqual(row["ps2_status_counts"], {"PROVISIONAL_LOW_PRIORITY": 14})
            self.assertEqual(row["rung2_state_counts"].get("IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS", 0), 0)
            self.assertEqual(row["rung3_state_counts"].get("COUNTERFACTUAL_UNDER_DECLARED_SCM", 0), 0)

    def test_joined_but_not_queued_reasons_partition_142(self):
        rows = [row for row in self.report["features"] if row["in_ps3c_join"] and not row["in_extractibility_queue"]]
        self.assertEqual(len(rows), 142)
        reasons = {reason: sum(row["extractibility_omission_reason"] == reason for row in rows)
                   for reason in ("TIER_3_ONLY_DEFERRED_BY_QUEUE_RULE", "CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES")}
        self.assertEqual(reasons, {
            "TIER_3_ONLY_DEFERRED_BY_QUEUE_RULE": 132,
            "CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES": 10,
        })

    def test_full_feature_ladder_state_counts(self):
        ladder = self.report["summary"]["feature_ladder_states"]
        self.assertEqual(ladder["rung1"], {"ASSOCIATION_REPORTED": 5124})
        self.assertEqual(ladder["rung2"], {
            "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS": 1076,
            "NOT_EVALUATED": 210,
            "NOT_IDENTIFIED": 3838,
        })
        self.assertEqual(ladder["rung3"], {
            "COUNTERFACTUAL_UNDER_DECLARED_SCM": 1048,
            "NOT_EVALUATED": 210,
            "NOT_IDENTIFIED": 3866,
        })

    def test_batch_report_normalization_is_reproduced_without_erasing_raw_states(self):
        batch = self.report["summary"]["batches"]["batch_001"]
        self.assertEqual(batch["raw_feature_rung_states"]["rung2"], {
            "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS": 450,
            "NOT_EVALUATED": 154,
            "NOT_IDENTIFIED": 40,
        })
        self.assertEqual(batch["batch_report_normalized_rung_states"]["rung2"], {
            "ESTIMATED": 450,
            "NOT_APPLICABLE": 140,
            "NOT_IDENTIFIED": 279,
        })
        hours_since = next(
            row for row in self.report["features"]
            if row["feature_id"] == "px.hours_since_prev_bar"
        )
        self.assertEqual(hours_since["rung2_state_counts"], {"NOT_EVALUATED": 14})
        self.assertEqual(
            self.report["summary"]["separate_event_ladder_states"]["rung2"],
            {"NOT_IDENTIFIED": 225},
        )

    def test_validator_rejects_missing_rows_and_false_omission_reason(self):
        missing = copy.deepcopy(self.report)
        missing["features"].pop()
        with self.assertRaises(ValueError):
            validate_report(missing)

        mislabeled = copy.deepcopy(self.report)
        row = next(x for x in mislabeled["features"] if not x["in_ps3c_join"])
        row["outside_join_reason"] = "UNEXPLAINED"
        with self.assertRaises(ValueError):
            validate_report(mislabeled)

    def test_every_input_digest_is_sha256_shaped(self):
        hashes = self.report["input_sha256"]
        self.assertEqual(len(hashes), 38)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        self.assertTrue(all(all(ch in "0123456789abcdef" for ch in value) for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
