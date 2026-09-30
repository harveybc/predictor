"""Schema tests for classification_metrics.v1 — written before the implementation.

These tests are about *reading*: whether a stored classification number can be
mistaken for a different metric, a different population, or a quality claim it
does not carry. They use the real contract document and the real receipt module;
nothing here is stubbed, and no function under test is replaced by a double.

Run:  python tools/test_classification_receipt_schema.py
"""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import classification_receipt as cr  # noqa: E402

CONTRACT_PATH = ROOT / "docs/contracts/classification_metrics.v1.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text())

H = {name: hashlib.sha256(name.encode()).hexdigest() for name in
     ("eval", "train", "calib", "corpus", "ckpt", "protocol", "scorer", "other_eval")}


def confusion_3x3(correct=(8, 8, 8), wrong=(1, 1, 1), abstained=(1, 1, 1)):
    """rows = reference class, columns = predicted class + a final ABSTAINED column."""
    rows = []
    for i in range(3):
        row = [0, 0, 0, abstained[i]]
        row[i] = correct[i]
        row[(i + 1) % 3] = wrong[i]
        rows.append(row)
    return rows


#: The default fixture's confusion is symmetric, so its accuracy and its macro-F1
#: are the SAME number, 8/9. That is the sharpest fixture available for the rule
#: under test: one value, three families, and nothing about the value itself that
#: tells a reader which family it came from.
SYMMETRIC_VALUE = 8 / 9


def document(**overrides):
    """A complete, valid receipt document. Values are fabricated schema fixtures."""
    confusion = overrides.pop("per_class_confusion", confusion_3x3())
    total = sum(sum(row) for row in confusion)
    abstained = sum(row[-1] for row in confusion)
    doc = {
        "task_id": "news_relevance_eurusd.v1",
        "corpus_class": "BUSINESS_HELD_OUT",
        "corpus_id": "cb04_business_news.v1",
        "corpus_sha256": H["corpus"],
        "evidence_class": "MEASUREMENT",
        "supervision_regime": "ZERO_SHOT_PROMPTED",
        "labelled_rows_fit_head": False,
        "provider": "probe_provider",
        "checkpoint": "probe_provider_h1.keras",
        "checkpoint_sha256": H["ckpt"],
        # the base document declares a path that loaded weights and observed
        # itself doing so, because its evidence_class is MEASUREMENT. The
        # promotion rule lives in tools/test_classification_provenance.py.
        "answering_path": {"path_id": "probe_provider.serve",
                           "kind": "MODEL_CHECKPOINT_LOADED",
                           "weights_present": True,
                           "served_checkpoint": "probe_provider_h1.keras",
                           "served_checkpoint_sha256": H["ckpt"],
                           "attestation": "OBSERVED_FROM_ANSWERING_PATH"},
        "author_primary_metric": {
            "family": "MACRO_F1",
            "name": "macro-F1",
            "value": SYMMETRIC_VALUE,
            "denominator_policy": "ANSWERED_ONLY",
        },
        "paired_naive": {
            "family": "MACRO_F1",
            "policy": "MAJORITY_CLASS_FROM_TRAIN",
            "value": 0.2,
            "seed": None,
            "train_label_population_sha256": H["train"],
            "evaluation_population_sha256": H["eval"],
        },
        "class_vocabulary": ["related", "unrelated", "unclear"],
        "per_class_confusion": confusion,
        "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED",
        "calibrated": False,
        "calibration_split": {"split_id": "NONE", "population_sha256": None,
                              "rows": 0, "fitted_parameters": None},
        "abstention": {
            "abstained": abstained,
            "answered": total - abstained,
            "coverage": abstained / total,
            "denominator_policy": "ANSWERED_ONLY",
            "abstention_rule": "the provider answered `unclear` as a refusal token",
        },
        "population": {"total": total, "answered": total - abstained, "abstained": abstained,
                       "independent_units": total, "repeats": 1, "clustered_by": "NONE"},
        "evaluation_split": "test",
        "evaluation_population_sha256": H["eval"],
        "declared_fields": [],
        "limitations": "fabricated schema fixture; not a model score",
        "protocol_sha256": H["protocol"],
        "scorer_sha256": H["scorer"],
        "seed": "1",
    }
    doc.update(overrides)
    return doc


class ContractDocument(unittest.TestCase):
    """The contract document itself, before any behaviour."""

    def test_schema_name_and_no_execution_authority(self):
        self.assertEqual(CONTRACT["schema"], "classification_metrics.v1")
        self.assertIs(CONTRACT["authorises_broker_deployment"], False)
        self.assertEqual(CONTRACT["execution_authority"], "NONE")

    def test_the_three_confusable_metrics_are_separate_families(self):
        for family in ("MAP", "ACCURACY", "MACRO_F1"):
            self.assertIn(family, CONTRACT["metric_families"])

    def test_every_family_has_its_own_metric_key_and_its_own_unit(self):
        keys = [spec["metric"] for spec in CONTRACT["metric_families"].values()]
        units = [spec["unit"] for spec in CONTRACT["metric_families"].values()]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(units), len(set(units)))

    def test_each_family_refuses_comparison_with_every_other_family(self):
        names = set(CONTRACT["metric_families"])
        for family, spec in CONTRACT["metric_families"].items():
            self.assertEqual(spec["comparable_with"], [family])
            self.assertEqual(set(spec["refuse_comparison_with"]), names - {family})

    def test_router_corpus_is_declared_as_router_not_classifier(self):
        router = CONTRACT["router_corpus"]
        self.assertEqual(router["prompts"] * router["repeats"], router["stored_verdicts"])
        self.assertEqual(router["independent_units"], router["prompts"])
        self.assertIn("NOT_CLASSIFIER", router["measures"])

    def test_contract_loaded_by_the_module_is_the_document_on_disk(self):
        self.assertEqual(cr.CONTRACT, CONTRACT)


class ReceiptFields(unittest.TestCase):
    """Each required fact is its own field, and a missing one is refused by name."""

    def test_a_complete_document_is_sealed(self):
        receipt = cr.build_receipt(document())
        self.assertEqual(receipt["schema"], "classification_metrics.v1")
        for field in CONTRACT["required_receipt_fields"]:
            self.assertIn(field, receipt)
        self.assertEqual(len(receipt["receipt_sha256"]), 64)

    def test_every_required_field_is_refused_by_name_when_absent(self):
        derived = {"schema", "receipt_id", "receipt_sha256"}
        for field in CONTRACT["required_receipt_fields"]:
            if field in derived:
                continue
            doc = document()
            doc.pop(field, None)
            with self.assertRaises(cr.ReceiptRefused) as caught:
                cr.build_receipt(doc)
            self.assertIn(field, str(caught.exception))

    def test_the_seven_named_facts_are_all_present_and_none_is_omitted(self):
        receipt = cr.build_receipt(document())
        self.assertEqual(receipt["author_primary_metric"]["name"], "macro-F1")
        self.assertEqual(receipt["paired_naive"]["policy"], "MAJORITY_CLASS_FROM_TRAIN")
        self.assertEqual(receipt["class_vocabulary"]["labels"],
                         ["related", "unrelated", "unclear"])
        self.assertEqual(len(receipt["per_class_confusion"]["matrix"]), 3)
        self.assertEqual(receipt["probability_semantics"], "SOFTMAX_POSTERIOR_UNCALIBRATED")
        self.assertIn("coverage", receipt["abstention"])
        self.assertEqual(receipt["calibration_split"]["split_id"], "NONE")


class MetricsAreNotInterchangeable(unittest.TestCase):
    """The rule with teeth: MAP, accuracy and macro-F1 are different metrics."""

    def _receipt(self, family, name, value=SYMMETRIC_VALUE):
        return cr.build_receipt(document(
            author_primary_metric={"family": family, "name": name, "value": value,
                                   "denominator_policy": "ANSWERED_ONLY"},
            paired_naive={"family": family, "policy": "MAJORITY_CLASS_FROM_TRAIN",
                          "value": 0.2, "seed": None,
                          "train_label_population_sha256": H["train"],
                          "evaluation_population_sha256": H["eval"]}))

    def test_identical_values_in_different_families_get_different_identities(self):
        identities = {family: self._receipt(family, name)["metric_identity_sha256"]
                      for family, name in (("MAP", "MAP"), ("ACCURACY", "accuracy"),
                                           ("MACRO_F1", "macro-F1"))}
        self.assertEqual(len(set(identities.values())), 3)

    def test_identical_values_in_different_families_get_different_units(self):
        units = {cr.terminal_metrics(self._receipt(f, n))[0]["unit"]
                 for f, n in (("MAP", "MAP"), ("ACCURACY", "accuracy"), ("MACRO_F1", "macro-F1"))}
        self.assertEqual(len(units), 3)

    def test_identical_values_in_different_families_get_different_metric_keys(self):
        keys = {cr.terminal_metrics(self._receipt(f, n))[0]["metric"]
                for f, n in (("MAP", "MAP"), ("ACCURACY", "accuracy"), ("MACRO_F1", "macro-F1"))}
        self.assertEqual(len(keys), 3)

    def test_reading_a_metric_the_receipt_does_not_carry_is_refused_by_name(self):
        receipt = self._receipt("MAP", "MAP")
        self.assertEqual(cr.read_metric(receipt, "MAP"), SYMMETRIC_VALUE)
        with self.assertRaises(cr.MetricNotCarried) as caught:
            cr.read_metric(receipt, "ACCURACY")
        message = str(caught.exception)
        self.assertIn("ACCURACY", message)
        self.assertIn("MAP", message)

    def test_comparing_two_different_families_refuses_and_names_both(self):
        pairs = (("MAP", "ACCURACY"), ("MAP", "MACRO_F1"), ("ACCURACY", "MACRO_F1"))
        for left, right in pairs:
            with self.assertRaises(cr.IncomparableMetrics) as caught:
                cr.compare(self._receipt(left, left), self._receipt(right, right))
            message = str(caught.exception)
            self.assertIn(left, message)
            self.assertIn(right, message)

    def test_comparing_the_same_family_on_the_same_protocol_is_allowed(self):
        # MAP is the family the confusion cannot derive, so two MAP values may
        # differ on one confusion without contradicting it.
        verdict = cr.compare(self._receipt("MAP", "MAP", 0.6),
                             self._receipt("MAP", "MAP", 0.4))
        self.assertEqual(verdict["family"], "MAP")
        self.assertAlmostEqual(verdict["difference"], 0.2)

    def test_comparing_the_same_family_on_a_different_population_refuses(self):
        other = cr.build_receipt(document(evaluation_population_sha256=H["other_eval"],
                                          paired_naive={"family": "MACRO_F1",
                                                        "policy": "MAJORITY_CLASS_FROM_TRAIN",
                                                        "value": 0.2, "seed": None,
                                                        "train_label_population_sha256": H["train"],
                                                        "evaluation_population_sha256": H["other_eval"]}))
        with self.assertRaises(cr.IncomparableProtocol):
            cr.compare(self._receipt("MACRO_F1", "macro-F1"), other)

    def test_a_value_outside_its_family_range_is_refused(self):
        for family, value in (("ACCURACY", 1.4), ("MACRO_F1", -0.1), ("MCC", 1.2)):
            with self.assertRaises(cr.ReceiptRefused):
                self._receipt(family, family, value)

    def test_a_non_finite_value_is_refused(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(cr.ReceiptRefused):
                self._receipt("ACCURACY", "accuracy", value)


class PairedNaive(unittest.TestCase):
    def test_a_naive_of_a_different_family_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(paired_naive={
                "family": "ACCURACY", "policy": "MAJORITY_CLASS_FROM_TRAIN", "value": 0.2,
                "seed": None, "train_label_population_sha256": H["train"],
                "evaluation_population_sha256": H["eval"]}))
        self.assertIn("MACRO_F1", str(caught.exception))
        self.assertIn("ACCURACY", str(caught.exception))

    def test_a_naive_scored_on_other_rows_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(paired_naive={
                "family": "MACRO_F1", "policy": "MAJORITY_CLASS_FROM_TRAIN", "value": 0.2,
                "seed": None, "train_label_population_sha256": H["train"],
                "evaluation_population_sha256": H["other_eval"]}))
        self.assertIn("same rows", str(caught.exception).lower())

    def test_a_train_derived_naive_fitted_on_the_evaluation_rows_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(paired_naive={
                "family": "MACRO_F1", "policy": "MAJORITY_CLASS_FROM_TRAIN", "value": 0.2,
                "seed": None, "train_label_population_sha256": H["eval"],
                "evaluation_population_sha256": H["eval"]}))

    def test_an_unknown_naive_policy_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(paired_naive={
                "family": "MACRO_F1", "policy": "WHATEVER_LOOKED_GOOD", "value": 0.2,
                "seed": None, "train_label_population_sha256": H["train"],
                "evaluation_population_sha256": H["eval"]}))


class AbstentionAndConfusion(unittest.TestCase):
    def test_coverage_that_disagrees_with_the_counts_is_refused(self):
        doc = document()
        doc["abstention"]["coverage"] = 0.0
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(doc)
        self.assertIn("coverage", str(caught.exception))

    def test_a_confusion_that_does_not_close_over_the_population_is_refused(self):
        doc = document()
        for field in ("total", "answered", "independent_units"):
            doc["population"][field] += 7
        doc["abstention"]["answered"] += 7
        doc["abstention"]["coverage"] = (doc["population"]["abstained"]
                                        / doc["population"]["total"])
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(doc)
        self.assertIn("close over the population", str(caught.exception))

    def test_abstained_rows_are_not_silently_dropped_from_the_denominator(self):
        doc = document()
        doc["abstention"].pop("denominator_policy")
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(doc)
        self.assertIn("denominator_policy", str(caught.exception))

    def test_coverage_is_carried_as_its_own_metric_row(self):
        rows = {row["metric"] for row in cr.terminal_metrics(cr.build_receipt(document()))}
        self.assertIn("classification.abstention_coverage", rows)
        self.assertIn("classification.answered_fraction", rows)

    def test_per_class_confusion_reaches_the_warehouse_as_per_class_rows(self):
        rows = {row["metric"] for row in cr.terminal_metrics(cr.build_receipt(document()))}
        for i in range(3):
            for kind in ("support", "predicted", "correct", "abstained"):
                self.assertIn(f"classification.{kind}.class_{i}", rows)

    def test_an_accuracy_that_contradicts_its_own_confusion_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(secondary_metrics={"ACCURACY": 0.99}))
        self.assertIn("ACCURACY", str(caught.exception))

    def test_a_vocabulary_that_does_not_match_the_confusion_shape_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(class_vocabulary=["related", "unrelated"]))


class ProbabilityAndCalibration(unittest.TestCase):
    def test_calibrated_without_a_calibration_split_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(calibrated=True))
        self.assertIn("calibration_split", str(caught.exception))

    def test_calibrated_contradicting_the_semantics_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(
                calibrated=True,
                probability_semantics="SOFTMAX_POSTERIOR_UNCALIBRATED",
                calibration_split={"split_id": "dev", "population_sha256": H["calib"],
                                   "rows": 50, "fitted_parameters": {"temperature": 1.3}}))

    def test_a_calibration_split_equal_to_the_evaluation_split_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(
                calibrated=True,
                probability_semantics="SOFTMAX_POSTERIOR_TEMPERATURE_CALIBRATED",
                calibration_split={"split_id": "test", "population_sha256": H["calib"],
                                   "rows": 50, "fitted_parameters": {"temperature": 1.3}}))
        self.assertIn("evaluation", str(caught.exception).lower())

    def test_a_calibration_population_equal_to_the_evaluation_population_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(
                calibrated=True,
                probability_semantics="ISOTONIC_CALIBRATED",
                calibration_split={"split_id": "dev", "population_sha256": H["eval"],
                                   "rows": 50, "fitted_parameters": {"bins": 10}}))

    def test_a_calibrated_receipt_with_a_distinct_split_is_accepted(self):
        receipt = cr.build_receipt(document(
            calibrated=True,
            probability_semantics="SOFTMAX_POSTERIOR_TEMPERATURE_CALIBRATED",
            calibration_split={"split_id": "dev", "population_sha256": H["calib"],
                               "rows": 50, "fitted_parameters": {"temperature": 1.3}}))
        self.assertIs(receipt["calibrated"], True)
        self.assertEqual(receipt["calibration_split"]["split_id"], "dev")

    def test_entropy_confidence_may_not_carry_probability_metrics(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(
                probability_semantics="ENTROPY_DERIVED_CONFIDENCE_NOT_P_CORRECT",
                probability_metrics={"ece": 0.1, "ece_bins": 15}))
        self.assertIn("P(correct)", str(caught.exception))

    def test_ece_without_declared_bins_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(probability_metrics={"ece": 0.1}))


class RouterCorpusIsNotAClassifier(unittest.TestCase):
    def test_a_router_corpus_receipt_may_not_carry_a_classification_metric(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(corpus_class="ROUTER_PROMPT_CORPUS",
                                      corpus_id=CONTRACT["router_corpus"]["corpus_id"]))
        self.assertIn("router", str(caught.exception).lower())

    def test_repeated_prompts_may_not_be_counted_as_independent_examples(self):
        doc = document()
        doc["population"].update(total=95, answered=95, abstained=0,
                                 independent_units=95, repeats=5, clustered_by="NONE")
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(doc)
        self.assertIn("independent", str(caught.exception).lower())

    def test_repeats_require_a_declared_clustering(self):
        confusion = [[19, 0, 0, 0], [0, 19, 0, 0], [0, 0, 57, 0]]
        doc = document(per_class_confusion=confusion)
        doc["population"].update(total=95, answered=95, abstained=0,
                                 independent_units=19, repeats=5, clustered_by="NONE")
        doc["abstention"].update(abstained=0, answered=95, coverage=0.0)
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(doc)
        self.assertIn("clustered_by", str(caught.exception))


class DeclarationsAreMarked(unittest.TestCase):
    def test_a_declared_field_is_recorded_as_a_declaration(self):
        receipt = cr.build_receipt(document(evidence_class="DECLARATION",
                                            declared_fields=["author_primary_metric"]))
        self.assertEqual(receipt["evidence_class"], "DECLARATION")
        self.assertIn("author_primary_metric", receipt["declared_fields"])

    def test_a_declared_field_that_is_not_a_receipt_field_is_refused(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(document(declared_fields=["vibes"]))

    def test_a_measurement_claiming_declared_content_must_say_so(self):
        receipt = cr.build_receipt(document(declared_fields=["checkpoint"]))
        self.assertEqual(receipt["evidence_class"], "MEASUREMENT")
        self.assertIn("checkpoint", receipt["declared_fields"])

    def test_zero_shot_is_refused_when_labelled_rows_fit_a_head(self):
        with self.assertRaises(cr.ReceiptRefused) as caught:
            cr.build_receipt(document(supervision_regime="ZERO_SHOT_PROMPTED",
                                      labelled_rows_fit_head=True))
        self.assertIn("zero", str(caught.exception).lower())


class ProviderQualityBadge(unittest.TestCase):
    def _business_measurement(self):
        return cr.build_receipt(document())

    def test_a_badge_needs_a_business_measurement(self):
        badge = cr.provider_quality_badge("probe_provider", [self._business_measurement()])
        self.assertEqual(badge["provider"], "probe_provider")
        self.assertEqual(badge["execution_authority"], "NONE")
        self.assertIs(badge["authorises_broker_deployment"], False)

    def test_no_badge_from_the_router_score(self):
        router = cr.build_router_reliability_record(
            corpus_id=CONTRACT["router_corpus"]["corpus_id"],
            correct=79, stored_verdicts=95, prompts=19, repeats=5,
            evidence_class="RECOUNT_OF_STORED_VERDICTS")
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("probe_provider", [router])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_ROUTER_SCORE_IS_NOT_CLASSIFIER_QUALITY")

    def test_a_router_record_is_not_even_a_classification_receipt(self):
        router = cr.build_router_reliability_record(
            corpus_id=CONTRACT["router_corpus"]["corpus_id"],
            correct=79, stored_verdicts=95, prompts=19, repeats=5,
            evidence_class="RECOUNT_OF_STORED_VERDICTS")
        self.assertNotEqual(router["schema"], cr.SCHEMA)
        self.assertEqual(router["independent_units"], 19)
        for family in CONTRACT["metric_families"]:
            self.assertNotIn(family, json.dumps(router))

    def test_no_badge_from_a_declaration_alone(self):
        declared = cr.build_receipt(document(evidence_class="DECLARATION",
                                             declared_fields=["author_primary_metric"]))
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("probe_provider", [declared])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT")

    def test_no_badge_from_a_recount_or_a_published_reference(self):
        for evidence, refusal in (
                ("RECOUNT_OF_STORED_VERDICTS", "BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT"),
                ("PUBLISHED_REFERENCE", "BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT"),
                ("TRANSPORT_TEST_NOT_SCIENCE", "BADGE_REFUSED_TRANSPORT_TEST_IS_NOT_SCIENCE")):
            receipt = cr.build_receipt(document(evidence_class=evidence))
            with self.assertRaises(cr.BadgeRefused) as caught:
                cr.provider_quality_badge("probe_provider", [receipt])
            self.assertEqual(caught.exception.refusal, refusal)

    def test_no_badge_from_public_benchmarks_alone(self):
        public = cr.build_receipt(document(corpus_class="PUBLIC_BENCHMARK",
                                           corpus_id="ag_news.test.v1"))
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("probe_provider", [public])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_NO_BUSINESS_CORPUS_MEASUREMENT")

    def test_no_badge_survives_a_router_record_mixed_into_good_evidence(self):
        router = cr.build_router_reliability_record(
            corpus_id=CONTRACT["router_corpus"]["corpus_id"],
            correct=85, stored_verdicts=95, prompts=19, repeats=5,
            evidence_class="RECOUNT_OF_STORED_VERDICTS")
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("probe_provider",
                                      [self._business_measurement(), router])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_ROUTER_SCORE_IS_NOT_CLASSIFIER_QUALITY")

    def test_no_badge_ever_carries_execution_authority(self):
        badge = cr.provider_quality_badge("probe_provider", [self._business_measurement()])
        serialised = json.dumps(badge)
        self.assertNotIn("true", serialised.split('"authorises_broker_deployment": ')[1][:8])
        self.assertEqual(badge["execution_authority"], "NONE")
        self.assertEqual(set(badge) & {"deploy", "broker", "order", "account"}, set())


class TerminalProjection(unittest.TestCase):
    def test_every_required_context_tag_is_present_and_is_a_string(self):
        tags = cr.terminal_tags(cr.build_receipt(document()))
        for name in CONTRACT["required_context_tags"]:
            self.assertIn(name, tags)
            self.assertIsInstance(tags[name], str)

    def test_metric_rows_use_the_nine_field_governed_schema(self):
        fields = {"metric", "value", "split", "horizon", "std_dev",
                  "min_value", "max_value", "unit"}
        for row in cr.terminal_metrics(cr.build_receipt(document())):
            self.assertEqual(set(row), fields)

    def test_the_primary_and_its_naive_are_separate_rows_with_the_same_unit(self):
        rows = {row["metric"]: row for row in cr.terminal_metrics(cr.build_receipt(document()))}
        self.assertIn("classification.macro_f1", rows)
        self.assertIn("classification.naive_macro_f1", rows)
        self.assertEqual(rows["classification.macro_f1"]["unit"],
                         rows["classification.naive_macro_f1"]["unit"])
        self.assertNotEqual(rows["classification.macro_f1"]["value"],
                            rows["classification.naive_macro_f1"]["value"])

    def test_the_full_confusion_is_bound_by_a_digest(self):
        receipt = cr.build_receipt(document())
        self.assertEqual(len(receipt["per_class_confusion"]["confusion_sha256"]), 64)
        self.assertEqual(cr.terminal_tags(receipt)["confusion_sha256"],
                         receipt["per_class_confusion"]["confusion_sha256"])

    def test_the_receipt_digest_is_stable_and_covers_the_content(self):
        first = cr.build_receipt(document())
        second = cr.build_receipt(document())
        self.assertEqual(first["receipt_sha256"], second["receipt_sha256"])
        changed = cr.build_receipt(document(limitations="a different stated limitation"))
        self.assertNotEqual(first["receipt_sha256"], changed["receipt_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
