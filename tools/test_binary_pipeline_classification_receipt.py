"""The receipt as the existing binary evaluation path actually produces it.

Two things are checked that a schema test cannot:

* the contract's macro-F1 and accuracy are compared against **scikit-learn's**
  implementations on the same rows. That is an independent oracle, not the same
  arithmetic run twice;
* the pipeline is checked for the call. A receipt builder nothing calls is a
  parallel system, so `test_the_binary_pipeline_actually_calls_the_builder` reads
  the pipeline's own source and fails if the call is removed.

Run:  python tools/test_binary_pipeline_classification_receipt.py
"""
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402
from sklearn.metrics import accuracy_score, f1_score          # noqa: E402

from app import classification_receipt as cr                  # noqa: E402
from pipeline_plugins import binary_metrics as bm             # noqa: E402

CONFIG = {
    "predictor_plugin": "ann_binary",
    "signal_type": "direction",
    "olap_experiment_key": "cb04_receipt_wiring_test",
    "x_train_file": "examples/data/fixture_x_train.csv",
    "x_test_file": "examples/data/fixture_x_test.csv",
    "y_train_file": "examples/data/fixture_y_train.csv",
    "y_test_file": "examples/data/fixture_y_test.csv",
    "window_size": 48,
    "epochs": 2,
    "learning_rate": 0.001,
    "predicted_horizons": [1],
    "seed": 1,
}


def arrays(n=200, seed=7):
    generator = np.random.default_rng(seed)
    truth = generator.integers(0, 2, size=n)
    # a probability that is informative but far from perfect
    probability = np.clip(0.5 + (truth - 0.5) * 0.5
                          + generator.normal(0, 0.22, size=n), 0.001, 0.999)
    train_truth = np.concatenate([np.zeros(130, dtype=int), np.ones(70, dtype=int)])
    return truth.astype(np.float32), probability.astype(np.float32), train_truth


class BuilderInTheEvaluationPath(unittest.TestCase):
    def setUp(self):
        self.truth, self.probability, self.train_truth = arrays()

    def test_the_receipt_is_a_valid_contract_receipt(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        self.assertEqual(receipt["schema"], cr.SCHEMA)
        for field in cr.CONTRACT["required_receipt_fields"]:
            self.assertIn(field, receipt)
        self.assertEqual(receipt["corpus_class"], "PROGRAMME_INTERNAL_DATASET")
        self.assertEqual(receipt["supervision_regime"], "FULL_FINETUNE")
        self.assertIs(receipt["labelled_rows_fit_head"], True)

    def test_macro_f1_agrees_with_scikit_learn_when_no_row_abstains(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        predicted = (self.probability >= 0.5).astype(int)
        self.assertAlmostEqual(cr.read_metric(receipt, "MACRO_F1"),
                              f1_score(self.truth.astype(int), predicted, average="macro"),
                              places=10)
        self.assertAlmostEqual(cr.read_metric(receipt, "ACCURACY"),
                              accuracy_score(self.truth.astype(int), predicted),
                              places=10)

    def test_macro_f1_agrees_with_scikit_learn_on_the_answered_rows_when_abstaining(self):
        config = dict(CONFIG, abstain_band=0.08)
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=config)
        answered = np.abs(self.probability - 0.5) >= 0.08
        self.assertGreater(int((~answered).sum()), 0, "the fixture must produce abstentions")
        predicted = (self.probability[answered] >= 0.5).astype(int)
        reference = self.truth[answered].astype(int)
        self.assertAlmostEqual(cr.read_metric(receipt, "MACRO_F1"),
                              f1_score(reference, predicted, average="macro"), places=10)
        self.assertAlmostEqual(cr.read_metric(receipt, "ACCURACY"),
                              accuracy_score(reference, predicted), places=10)
        self.assertEqual(receipt["abstention"]["denominator_policy"], "ANSWERED_ONLY")
        self.assertEqual(receipt["population"]["abstained"], int((~answered).sum()))
        self.assertAlmostEqual(receipt["abstention"]["coverage"],
                              int((~answered).sum()) / len(self.truth))

    def test_without_an_abstention_rule_no_row_leaves_the_denominator(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        self.assertEqual(receipt["population"]["abstained"], 0)
        self.assertEqual(receipt["abstention"]["coverage"], 0.0)
        self.assertEqual(receipt["abstention"]["denominator_policy"],
                         "FULL_POPULATION_ABSTENTION_WRONG")
        self.assertIn("no abstention rule", receipt["abstention"]["abstention_rule"])

    def test_the_naive_is_the_train_majority_scored_on_the_same_rows(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        naive = receipt["paired_naive"]
        self.assertEqual(naive["policy"], "MAJORITY_CLASS_FROM_TRAIN")
        self.assertEqual(naive["evaluation_population_sha256"],
                         receipt["evaluation_population_sha256"])
        self.assertNotEqual(naive["train_label_population_sha256"],
                            receipt["evaluation_population_sha256"])
        # the train majority is class 0 (130 against 70), so the naive predicts 0
        expected = f1_score(self.truth.astype(int), np.zeros(len(self.truth), dtype=int),
                            average="macro", zero_division=0)
        self.assertAlmostEqual(naive["value"], expected, places=10)
        self.assertEqual(naive["family"], receipt["author_primary_metric"]["family"])

    def test_the_confusion_closes_over_the_population(self):
        config = dict(CONFIG, abstain_band=0.08)
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=config)
        confusion = receipt["per_class_confusion"]
        self.assertEqual(sum(sum(row) for row in confusion["matrix"]),
                         receipt["population"]["total"])
        self.assertEqual(sum(confusion["abstained"]), receipt["population"]["abstained"])
        self.assertEqual(sum(confusion["support"]), receipt["population"]["total"])

    def test_the_receipt_projects_onto_the_governed_terminal(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        tags = cr.terminal_tags(receipt)
        for name in cr.CONTRACT["required_context_tags"]:
            self.assertIn(name, tags)
        rows = {row["metric"]: row for row in cr.terminal_metrics(receipt)}
        self.assertEqual(rows["classification.macro_f1"]["unit"], "macro_f1_fraction")
        self.assertEqual(rows["classification.accuracy"]["unit"], "accuracy_fraction")
        self.assertIn("classification.naive_macro_f1", rows)
        self.assertNotIn("classification.map", rows,
                         "this pipeline does not produce MAP and must not appear to")

    def test_a_receipt_from_this_path_reads_as_macro_f1_and_nothing_else(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        with self.assertRaises(cr.MetricNotCarried) as caught:
            cr.read_metric(receipt, "MAP")
        self.assertIn("MAP", str(caught.exception))
        self.assertIn("MACRO_F1", str(caught.exception))

    def test_the_receipt_is_written_beside_the_results(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        with tempfile.TemporaryDirectory(prefix="cb04-receipt-") as temporary:
            path = Path(temporary) / "results_classification_receipt.json"
            bm.save_binary_classification_receipt(receipt, str(path))
            written = json.loads(path.read_text())
        self.assertEqual(written["receipt_sha256"], receipt["receipt_sha256"])
        self.assertEqual(written["execution_authority"], "NONE")

    def test_an_internal_run_can_never_become_a_provider_quality_badge(self):
        receipt = bm.build_binary_classification_receipt(
            self.truth, self.probability, self.train_truth, config=CONFIG)
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("ann_binary", [receipt])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_NO_BUSINESS_CORPUS_MEASUREMENT")


class TheWiringItself(unittest.TestCase):
    def test_the_binary_pipeline_actually_calls_the_builder(self):
        from pipeline_plugins import binary_pipeline
        source = inspect.getsource(binary_pipeline)
        self.assertIn("build_binary_classification_receipt", source)
        self.assertIn("save_binary_classification_receipt", source)
        run = inspect.getsource(binary_pipeline.BinaryPipelinePlugin.run_prediction_pipeline)
        self.assertIn("build_binary_classification_receipt(", run,
                      "the receipt must be built inside the pipeline's own run, not beside it")
        self.assertIn("save_binary_classification_receipt(", run)

    def test_the_builder_uses_the_shared_contract_module(self):
        source = inspect.getsource(bm.build_binary_classification_receipt)
        self.assertIn("from app import classification_receipt", source)
        self.assertIn("receipts.build_receipt(document)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
