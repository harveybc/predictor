"""Checks for the metadata-only progress report; no ML or fleet work is launched."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "progress_report", Path(__file__).parents[1] / "tools/render_master_progress.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ProgressReceiptTests(unittest.TestCase):
    def setUp(self):
        candidates = [dict(cid="a" * 16, seed=2021, config_id="paired",
                           label="paired", objective=0.3, status="verified"),
                      dict(cid="b" * 16, seed=2022, config_id="paired",
                           label="paired", objective=0.3, status="verified"),
                      dict(status="SUPERSEDED_OLD_ARCH")]
        receipt = dict(exact_match=True,
                       objective=dict(rescored_value=0.3, split="validation"),
                       metrics=dict(baseline_MAE=0.5),
                       digests=dict(validation_sha256="population"),
                       per_horizon={"1": dict(skill_MAE=0.4)})
        self.files = {
            "STATUS.json": dict(observed_at="2026-10-01T03:00:00Z", devices=[],
                                jobs=[dict(id="m02-ecl-v2-donors-r7g2",
                                           progress=dict(completed=1, total=321), eta={})]),
            "corrected_QUEUE_r0_v1.export.json": dict(candidates=candidates),
            "coverage_laneB.json": dict(old_denominator=dict(covered=1, distinct_rows=4),
                                        new_denominator=dict(files=4), sources=dict(rows=1),
                                        transform_ledger=dict(rows=2, evaluated_true=0)),
            "traffic_h96_closure_table.json": dict(rows=[{}] * 3,
                                                  mean={m: dict(mean=0.3, naive=0.5,
                                                                published=0.3)
                                                        for m in ("mse", "mae")}),
            "a" * 16 + ".json": copy.deepcopy(receipt),
            "b" * 16 + ".json": copy.deepcopy(receipt),
        }

    def collect(self):
        with patch.object(REPORT, "read_json", side_effect=lambda path, _: self.files[path.name]):
            return REPORT.collect(Path("unused"))

    def test_paired_result_excludes_superseded_rows(self):
        result = self.collect()
        self.assertEqual(result["best"]["mae"], 0.3)
        self.assertEqual(result["best"]["naive"], 0.5)
        self.assertEqual(result["rows"][2]["total"], 2)

    def test_bad_receipts_are_not_plotted(self):
        for field, value in (("exact_match", False), ("exact_match", "true"),
                             ("objective", dict(rescored_value=0.4, split="validation")),
                             ("objective", dict(rescored_value=0.3, split="test")),
                             ("metrics", dict(baseline_MAE=0.6)),
                             ("digests", dict(validation_sha256="other"))):
            with self.subTest(field=field, value=value):
                self.setUp()
                self.files["a" * 16 + ".json"][field] = value
                with self.assertRaises(ValueError):
                    self.collect()

    def test_incomplete_pair_rejected(self):
        self.files["corrected_QUEUE_r0_v1.export.json"]["candidates"][1]["status"] = "pending"
        with self.assertRaises(ValueError):
            self.collect()

    def test_owner_timezone(self):
        self.assertEqual(REPORT.local_time("2026-10-01T03:00:00Z"), "22:00")


if __name__ == "__main__":
    unittest.main()
