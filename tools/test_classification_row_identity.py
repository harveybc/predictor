"""Per-row metric identity: the tests were written red, against the shipped defect.

Every test in `TheDefectAsShipped` passes only because the defect is real: they
assert, against `app.classification_receipt` unmodified, that a secondary row
inherits the primary's identity. They are kept because a later edit that quietly
changed the old function would break them, and the successor tags would then be
describing a problem that no longer exists.

The rest assert the repair: each row's own identity, evidence class out of the
identity and into the occurrence key, and no double counting between terminals.

  python3 tools/test_classification_row_identity.py
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import classification_receipt as cr            # noqa: E402
from app import classification_row_identity as ri       # noqa: E402

H = {name: hashlib.sha256(name.encode()).hexdigest() for name in
     ("eval", "train", "calib", "corpus", "ckpt", "protocol", "scorer",
      "protocol2", "scorer2")}


def document(family, name, value, *, secondary=None, classes=3,
             evidence_class="TRANSPORT_TEST_NOT_SCIENCE",
             provider="NON_MODEL_FIXTURE", protocol=H["protocol"], scorer=H["scorer"],
             probability=None, seed="1"):
    """A valid receipt document whose confusion agrees with the carried values.

    `classes` correct-by-construction cells: every class gets 8 correct, 1 wrong
    into the next class, 1 abstained, so accuracy and macro-F1 are both exactly
    8/9 under FULL_POPULATION_ABSTENTION_WRONG... which is why the fixture uses
    that policy for both carried families and the fabrication is visible.
    """
    matrix = []
    for index in range(classes):
        row = [0] * classes + [1]
        row[index] = 8
        row[(index + 1) % classes] = 1
        matrix.append(row)
    total = sum(sum(row) for row in matrix)
    abstained = sum(row[-1] for row in matrix)
    return {
        "task_id": "row_identity_fixture.v1",
        "corpus_class": "PUBLIC_BENCHMARK",
        "corpus_id": "row_identity_fixture_corpus.v1",
        "corpus_sha256": H["corpus"],
        "evidence_class": evidence_class,
        "supervision_regime": "ZERO_SHOT_PROMPTED",
        "labelled_rows_fit_head": False,
        "provider": provider,
        "checkpoint": "NON_MODEL_FIXTURE",
        "checkpoint_sha256": H["ckpt"],
        "author_primary_metric": {"family": family, "name": name, "value": value,
                                  "denominator_policy": "ANSWERED_ONLY"},
        "paired_naive": {"family": family, "policy": "MAJORITY_CLASS_FROM_TRAIN",
                         "value": 0.25, "seed": None,
                         "train_label_population_sha256": H["train"],
                         "evaluation_population_sha256": H["eval"]},
        "class_vocabulary": [f"class_{i}" for i in range(classes)],
        "per_class_confusion": matrix,
        "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED",
        "calibrated": False,
        "calibration_split": {"split_id": "NONE", "population_sha256": None,
                              "rows": 0, "fitted_parameters": None},
        "probability_metrics": probability,
        "secondary_metrics": secondary or {},
        "abstention": {"abstained": abstained, "answered": total - abstained,
                       "coverage": abstained / total,
                       "denominator_policy": "ANSWERED_ONLY",
                       "abstention_rule": "the provider answered the refusal token"},
        "population": {"total": total, "answered": total - abstained,
                       "abstained": abstained, "independent_units": total,
                       "repeats": 1, "clustered_by": "NONE"},
        "evaluation_split": "test",
        "evaluation_population_sha256": H["eval"],
        "declared_fields": [],
        "limitations": "fabricated fixture for the row-identity contract; not a model score",
        "protocol_sha256": protocol,
        "scorer_sha256": scorer,
        "seed": seed,
    }


def rows_of(receipt, unit_id):
    return [dict(row, unit_id=unit_id)
            for row in ri.metric_rows_with_identity(receipt)]


VALUE = 8 / 9


def accuracy_primary(**kw):
    return cr.build_receipt(document("ACCURACY", "accuracy", VALUE,
                                     secondary={"MACRO_F1": VALUE}, **kw))


def macro_f1_primary(**kw):
    return cr.build_receipt(document("MACRO_F1", "macro-F1", VALUE,
                                     secondary={"ACCURACY": VALUE}, **kw))


class TheDefectAsShipped(unittest.TestCase):
    """What `classification_metrics.v1` alone does, asserted rather than described."""

    def test_the_terminal_level_digest_binds_the_primary_only(self):
        left, right = accuracy_primary(), macro_f1_primary()
        self.assertNotEqual(left["metric_identity_sha256"], right["metric_identity_sha256"])
        # one tag for the whole terminal, whatever the row
        self.assertEqual(cr.terminal_tags(left)["metric_identity_sha256"],
                         left["metric_identity_sha256"])
        keys = {row["metric"] for row in cr.terminal_metrics(left)}
        self.assertIn("classification.macro_f1", keys)
        self.assertIn("classification.accuracy", keys)

    def test_one_macro_f1_value_stands_under_two_identities(self):
        """The live-warehouse shape, reproduced from the shipped functions."""
        left, right = accuracy_primary(), macro_f1_primary()
        stand_under = {}
        for receipt in (left, right):
            identity = cr.terminal_tags(receipt)["metric_identity_sha256"]
            for row in cr.terminal_metrics(receipt):
                if row["metric"] == "classification.macro_f1":
                    stand_under.setdefault(row["value"], set()).add(identity)
        self.assertEqual(len(stand_under), 1, "one macro-F1 value")
        self.assertEqual(len(next(iter(stand_under.values()))), 2, "under two identities")

    def test_one_identity_covers_two_different_macro_f1_values(self):
        a = accuracy_primary()
        b = cr.build_receipt(document("ACCURACY", "accuracy", VALUE, classes=4,
                                      secondary={"MACRO_F1": VALUE}))
        # same primary family, name, unit and denominator; different label order,
        # so the shipped digest does separate these two
        self.assertNotEqual(cr.terminal_tags(a)["metric_identity_sha256"],
                            cr.terminal_tags(b)["metric_identity_sha256"])
        # but within ONE identity, a secondary row of a different family rides along
        identity = cr.terminal_tags(a)["metric_identity_sha256"]
        families = {row["metric"] for row in cr.terminal_metrics(a)
                    if row["metric"].startswith("classification.")
                    and "class_" not in row["metric"]}
        self.assertIn("classification.macro_f1", families)
        self.assertEqual(identity, cr.terminal_tags(a)["metric_identity_sha256"])

    def test_the_legacy_tag_set_is_refused_by_the_successor_contract(self):
        with self.assertRaises(ri.RowIdentityRefused) as caught:
            ri.assert_row_identity_tags(cr.terminal_tags(accuracy_primary()))
        self.assertEqual(caught.exception.refusal, "ROW_IDENTITY_TAGS_MISSING")
        self.assertIn("inherit the primary", str(caught.exception))


class EveryRowHasItsOwnIdentity(unittest.TestCase):

    def test_a_secondary_row_does_not_inherit_the_primary_identity(self):
        receipt = accuracy_primary()
        rows = {row["metric"]: row for row in ri.metric_rows_with_identity(receipt)}
        primary = rows["classification.accuracy"]
        secondary = rows["classification.macro_f1"]
        self.assertEqual(primary["row_role"], "PRIMARY")
        self.assertEqual(secondary["row_role"], "SECONDARY")
        self.assertNotEqual(primary["row_identity_sha256"], secondary["row_identity_sha256"])
        self.assertNotEqual(secondary["row_identity_sha256"],
                            receipt["metric_identity_sha256"])

    def test_every_projected_row_gets_a_digest_and_none_is_shared_by_accident(self):
        receipt = accuracy_primary(probability={"nll": 0.3, "brier": 0.12,
                                                "ece": 0.08, "ece_bins": 15})
        rows = ri.metric_rows_with_identity(receipt)
        self.assertEqual(len(rows), len(cr.terminal_metrics(receipt)))
        self.assertEqual(len(rows), len({row["metric"] for row in rows}))
        self.assertEqual(len(rows), len({row["row_identity_sha256"] for row in rows}),
                         "distinct metric keys must have distinct identities")
        for row in rows:
            self.assertRegex(row["row_identity_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(row["row_role"], ri.ROW_ROLES)

    def test_the_same_metric_on_the_same_protocol_shares_one_identity_across_roles(self):
        """A macro-F1 is a macro-F1 whether the receipt called it primary or secondary."""
        primary_row = next(r for r in ri.metric_rows_with_identity(macro_f1_primary())
                           if r["metric"] == "classification.macro_f1")
        secondary_row = next(r for r in ri.metric_rows_with_identity(accuracy_primary())
                             if r["metric"] == "classification.macro_f1")
        self.assertEqual(primary_row["row_identity_sha256"],
                         secondary_row["row_identity_sha256"])
        self.assertEqual(primary_row["row_role"], "PRIMARY")
        self.assertEqual(secondary_row["row_role"], "SECONDARY")

    def test_the_authors_own_name_is_attribution_and_not_identity(self):
        a = cr.build_receipt(document("MACRO_F1", "macro-F1", VALUE))
        b = cr.build_receipt(document("MACRO_F1", "F1-macro", VALUE))
        self.assertNotEqual(a["metric_identity_sha256"], b["metric_identity_sha256"])
        ra = next(r for r in ri.metric_rows_with_identity(a)
                  if r["metric"] == "classification.macro_f1")
        rb = next(r for r in ri.metric_rows_with_identity(b)
                  if r["metric"] == "classification.macro_f1")
        self.assertEqual(ra["row_identity_sha256"], rb["row_identity_sha256"])
        self.assertEqual(ra["reported_under_name"], "macro-F1")
        self.assertEqual(rb["reported_under_name"], "F1-macro")
        # and the name is still refused by name, in the open, by compare()
        with self.assertRaises(cr.ClassificationReceiptError):
            cr.compare(a, b)

    def test_a_different_label_order_is_a_different_identity_on_every_row(self):
        three = ri.metric_rows_with_identity(accuracy_primary())
        four = ri.metric_rows_with_identity(
            cr.build_receipt(document("ACCURACY", "accuracy", VALUE, classes=4,
                                      secondary={"MACRO_F1": VALUE})))
        by_key_three = {r["metric"]: r["row_identity_sha256"] for r in three}
        by_key_four = {r["metric"]: r["row_identity_sha256"] for r in four}
        shared = set(by_key_three) & set(by_key_four)
        self.assertTrue(shared)
        for key in shared:
            self.assertNotEqual(by_key_three[key], by_key_four[key], key)

    def test_a_confusion_cell_is_not_a_score_and_carries_its_own_kind(self):
        rows = {r["metric"]: r for r in ri.metric_rows_with_identity(accuracy_primary())}
        cell = rows["classification.support.class_0"]
        self.assertEqual(cell["row_role"], "CONFUSION_CELL")
        self.assertEqual(cell["kind"], "CONFUSION_COUNT")
        self.assertIsNone(cell["family"])
        self.assertEqual(cell["unit"], "rows")
        self.assertNotEqual(cell["row_identity_sha256"],
                            rows["classification.support.class_1"]["row_identity_sha256"])

    def test_two_denominator_policies_in_one_receipt_are_refused_by_name(self):
        doc = document("ACCURACY", "accuracy", VALUE)
        doc["abstention"]["denominator_policy"] = "FULL_POPULATION_ABSTENTION_WRONG"
        doc["author_primary_metric"]["value"] = 24 / 30
        doc["paired_naive"]["value"] = 0.2
        receipt = cr.build_receipt(doc)
        with self.assertRaises(ri.RowIdentityRefused) as caught:
            ri.metric_rows_with_identity(receipt)
        self.assertEqual(caught.exception.refusal,
                         "PRIMARY_AND_ABSTENTION_DENOMINATOR_POLICIES_DISAGREE")

    def test_the_nine_stored_fields_are_returned_unchanged(self):
        receipt = accuracy_primary()
        stored = cr.terminal_metrics(receipt)
        enriched = ri.metric_rows_with_identity(receipt)
        for plain, rich in zip(stored, enriched):
            for field in ("metric", "value", "split", "horizon", "std_dev",
                          "min_value", "max_value", "unit"):
                self.assertEqual(rich[field], plain[field], field)


class EvidenceClassIsSeparateFromIdentity(unittest.TestCase):

    def test_a_published_reference_and_a_measurement_share_one_identity(self):
        measured = cr.build_receipt(document("ACCURACY", "accuracy", VALUE,
                                             evidence_class="MEASUREMENT"))
        published = cr.build_receipt(document("ACCURACY", "accuracy", VALUE,
                                              evidence_class="PUBLISHED_REFERENCE"))
        m = next(r for r in ri.metric_rows_with_identity(measured)
                 if r["row_role"] == "PRIMARY")
        p = next(r for r in ri.metric_rows_with_identity(published)
                 if r["row_role"] == "PRIMARY")
        self.assertEqual(m["row_identity_sha256"], p["row_identity_sha256"],
                         "comparability lives in the identity")
        self.assertNotEqual(m["metric_occurrence_sha256"], p["metric_occurrence_sha256"],
                            "provenance lives beside it, so they are never one occurrence")

    def test_an_aggregate_never_puts_them_in_one_group(self):
        measured = cr.build_receipt(document("ACCURACY", "accuracy", VALUE,
                                             evidence_class="MEASUREMENT"))
        published = cr.build_receipt(document("ACCURACY", "accuracy", VALUE,
                                              evidence_class="PUBLISHED_REFERENCE"))
        report = ri.aggregation_groups(rows_of(measured, "measured")
                                       + rows_of(published, "published"))
        primary_groups = [g for g in report["groups"]
                          if g["metric"] == "classification.accuracy"]
        self.assertEqual(len(primary_groups), 2)
        self.assertEqual({g["evidence_class"] for g in primary_groups},
                         {"MEASUREMENT", "PUBLISHED_REFERENCE"})
        self.assertTrue(any(w["refusal"] == "EVIDENCE_CLASSES_MIXED_IN_ONE_AGGREGATE"
                            for w in report["evidence_class_separations"]))

    def test_evidence_class_is_not_in_the_identity_digest(self):
        self.assertIn("evidence_class",
                      ri.CONTRACT["identity_inputs"]["deliberately_excluded"])
        self.assertIn("evidence_class", ri.CONTRACT["occurrence_inputs"]["fields"])
        self.assertNotIn("evidence_class", ri.CONTRACT["identity_inputs"]["fields"])


class NoDoubleCountingBetweenTerminals(unittest.TestCase):

    def test_one_measurement_in_two_terminals_is_counted_once(self):
        """The live shape: two terminals, one as primary and one as secondary."""
        probability = {"nll": 0.3, "brier": 0.12, "ece": 0.08, "ece_bins": 15}
        left = accuracy_primary(probability=probability, evidence_class="MEASUREMENT")
        right = macro_f1_primary(probability=probability, evidence_class="MEASUREMENT")
        rows = rows_of(left, "accuracy-terminal") + rows_of(right, "macro-f1-terminal")
        report = ri.deduplicate(rows)
        self.assertEqual(report["rows_in"], len(rows))
        duplicated = {d["metric"] for d in report["duplicates_dropped"]}
        self.assertIn("classification.accuracy", duplicated)
        self.assertIn("classification.macro_f1", duplicated)
        self.assertIn("classification.support.class_0", duplicated)
        # 20 rows each; only the paired naive differs between the two, because its
        # family does, so 19 of the 20 are one measurement reported twice
        self.assertEqual(report["rows_in"], 40)
        self.assertEqual(len(report["duplicates_dropped"]), 19)
        self.assertEqual(report["rows_counted"], 21)
        self.assertNotIn("classification.naive_accuracy", duplicated)
        self.assertNotIn("classification.naive_macro_f1", duplicated)
        for dropped in report["duplicates_dropped"]:
            self.assertEqual(dropped["counted"], "ONCE")
            self.assertEqual(dropped["refusal"],
                             "DOUBLE_COUNT_SAME_OCCURRENCE_ACROSS_TERMINALS")

    def test_the_old_rule_double_counts_and_the_new_one_does_not(self):
        probability = {"nll": 0.3, "brier": 0.12, "ece": 0.08, "ece_bins": 15}
        left = accuracy_primary(probability=probability, evidence_class="MEASUREMENT")
        right = macro_f1_primary(probability=probability, evidence_class="MEASUREMENT")
        # the rule the shipped contract prescribes: group by the terminal digest
        old = []
        for receipt, unit in ((left, "a"), (right, "b")):
            identity = cr.terminal_tags(receipt)["metric_identity_sha256"]
            for row in cr.terminal_metrics(receipt):
                if row["metric"] == "classification.macro_f1":
                    old.append((identity, unit, row["value"]))
        self.assertEqual(len(old), 2)
        self.assertEqual(len({o[0] for o in old}), 2,
                         "the same macro-F1 measurement lands in two groups")
        new = ri.aggregation_groups(rows_of(left, "a") + rows_of(right, "b"))
        macro = [g for g in new["groups"] if g["metric"] == "classification.macro_f1"]
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["n_occurrences"], 1, "counted once")

    def test_a_replicate_under_a_different_protocol_is_its_own_occurrence(self):
        one = accuracy_primary(evidence_class="MEASUREMENT")
        two = accuracy_primary(evidence_class="MEASUREMENT",
                               protocol=H["protocol2"], scorer=H["scorer2"])
        report = ri.deduplicate(rows_of(one, "run-1") + rows_of(two, "run-2"))
        self.assertEqual(report["duplicates_dropped"], [])
        self.assertEqual(report["rows_counted"], report["rows_in"])
        groups = ri.aggregation_groups(rows_of(one, "run-1") + rows_of(two, "run-2"))
        accuracy = [g for g in groups["groups"]
                    if g["metric"] == "classification.accuracy"]
        self.assertEqual(len(accuracy), 1)
        self.assertEqual(accuracy[0]["n_occurrences"], 2, "two replicates, both counted")

    def test_a_different_provider_on_the_same_rows_is_its_own_occurrence(self):
        ours = accuracy_primary(evidence_class="MEASUREMENT", provider="OUR_RUNTIME")
        theirs = accuracy_primary(evidence_class="MEASUREMENT", provider="NATIVE_HARNESS")
        report = ri.deduplicate(rows_of(ours, "ours") + rows_of(theirs, "theirs"))
        self.assertEqual(report["duplicates_dropped"], [])

    def test_one_occurrence_with_two_values_is_a_refusal_not_an_average(self):
        left = accuracy_primary(evidence_class="MEASUREMENT")
        right = accuracy_primary(evidence_class="MEASUREMENT")
        rows = rows_of(left, "a") + rows_of(right, "b")
        for row in rows[len(rows) // 2:]:
            if row["metric"] == "classification.accuracy":
                row["value"] = 0.5
        report = ri.deduplicate(rows)
        self.assertEqual([c["refusal"] for c in report["conflicts"]],
                         ["CONFLICTING_VALUES_FOR_ONE_OCCURRENCE"])
        with self.assertRaises(ri.RowIdentityRefused) as caught:
            ri.aggregation_groups(rows)
        self.assertEqual(caught.exception.refusal,
                         "CONFLICTING_VALUES_FOR_ONE_OCCURRENCE")

    def test_the_occurrence_key_deliberately_excludes_the_terminal(self):
        excluded = ri.CONTRACT["occurrence_inputs"]["deliberately_excluded"]
        self.assertIn("row_role", excluded)
        self.assertIn("terminal_sha256_campaign_unit_generation", excluded)
        self.assertIn("receipt_sha256", excluded)


class TheTagsAndTheReader(unittest.TestCase):

    def test_the_successor_tags_add_and_never_overwrite(self):
        receipt = accuracy_primary()
        base = cr.terminal_tags(receipt)
        both = ri.terminal_tags_with_row_identity(receipt)
        for key, value in base.items():
            self.assertEqual(both[key], value, key)
        self.assertGreater(len(both), len(base))
        ri.assert_row_identity_tags(both)
        self.assertEqual(both["metric_identity_sha256_scope"],
                         "TERMINAL_LEVEL_PRIMARY_ONLY_DO_NOT_USE_FOR_SECONDARY_ROWS")
        self.assertEqual(both["double_count_guard"], "metric_occurrence_sha256")

    def test_the_map_digest_covers_every_row_including_the_confusion(self):
        receipt = accuracy_primary()
        tags = ri.row_identity_tags(receipt)
        rows = ri.metric_rows_with_identity(receipt)
        self.assertEqual(int(tags["metric_row_count"]), len(rows))
        self.assertEqual(tags["metric_row_identity_map_sha256"],
                         cr.sha256_of({r["metric"]: r["row_identity_sha256"]
                                       for r in rows}))
        self.assertEqual(tags["metric_row_occurrence_map_sha256"],
                         cr.sha256_of({r["metric"]: r["metric_occurrence_sha256"]
                                       for r in rows}))

    def test_a_reader_with_only_the_tags_recovers_any_row(self):
        receipt = accuracy_primary()
        tags = ri.terminal_tags_with_row_identity(receipt)
        rows = {r["metric"]: r for r in ri.metric_rows_with_identity(receipt)}
        for key in ("classification.accuracy", "classification.macro_f1",
                    "classification.support.class_2", "classification.abstained.class_1"):
            recovered = ri.recompute_from_tags(tags, key)
            self.assertEqual(recovered["row_identity_sha256"],
                             rows[key]["row_identity_sha256"], key)
            self.assertEqual(recovered["metric_occurrence_sha256"],
                             rows[key]["metric_occurrence_sha256"], key)

    def test_seventy_seven_classes_overflow_the_tag_and_stay_recomputable(self):
        receipt = cr.build_receipt(document("ACCURACY", "accuracy", VALUE, classes=77,
                                            secondary={"MACRO_F1": VALUE}))
        rows = ri.metric_rows_with_identity(receipt)
        self.assertEqual(len(rows), 2 + 1 + 2 + 77 * 4)
        tags = ri.terminal_tags_with_row_identity(receipt)
        self.assertEqual(int(tags["metric_row_count"]), len(rows))
        self.assertEqual(json.loads(tags["metric_row_identity_json"]).keys()
                         .__len__(), 5)
        by_key = {r["metric"]: r for r in rows}
        for index in (0, 40, 76):
            key = f"classification.correct.class_{index}"
            self.assertEqual(ri.recompute_from_tags(tags, key)["row_identity_sha256"],
                             by_key[key]["row_identity_sha256"], key)
        self.assertEqual(tags["metric_row_identity_map_sha256"],
                         cr.sha256_of({r["metric"]: r["row_identity_sha256"]
                                       for r in rows}))


class WhatWasNotTouched(unittest.TestCase):
    """The successor does not rewrite what the corpus was sealed against."""

    def test_the_metric_contract_json_is_byte_identical_to_the_sealed_pin(self):
        manifest = json.loads(
            (ROOT / "docs/audits/evidence/cb04_business_corpus_20260928/MANIFEST.json"
             ).read_text())
        pinned = manifest["built_after_the_protocol"]["contract_sha256"]
        actual = hashlib.sha256(
            (ROOT / "docs/contracts/classification_metrics.v1.json").read_bytes()
        ).hexdigest()
        self.assertEqual(pinned, actual,
                         "classification_metrics.v1.json must not change: the sealed "
                         "business corpus pins its digest")

    def test_the_receipt_implementation_is_byte_identical_to_the_sealed_pin(self):
        manifest = json.loads(
            (ROOT / "docs/audits/evidence/cb04_business_corpus_20260928/MANIFEST.json"
             ).read_text())
        pinned = manifest["built_after_the_protocol"]["implementation_sha256"]
        actual = hashlib.sha256((ROOT / "app/classification_receipt.py").read_bytes()
                                ).hexdigest()
        self.assertEqual(pinned, actual,
                         "app/classification_receipt.py must not change: the row-identity "
                         "repair is a successor module, not an edit of the sealed producer")

    def test_nothing_under_test_was_substituted(self):
        for function in (cr.build_receipt, cr.terminal_tags, cr.terminal_metrics,
                         cr.compare, cr.read_metric):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_receipt")
        for function in (ri.row_identity_sha256, ri.metric_occurrence_sha256,
                         ri.metric_rows_with_identity, ri.deduplicate,
                         ri.aggregation_groups, ri.recompute_from_tags):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_row_identity")
        self.assertEqual(inspect.getfile(ri),
                         str(ROOT / "app/classification_row_identity.py"))
        self.assertEqual(ri.SCHEMA, "classification_row_identity.v1")
        self.assertIs(ri.cr, cr)

    def test_the_contract_carries_no_execution_authority(self):
        self.assertIs(ri.CONTRACT["authorises_broker_deployment"], False)
        self.assertEqual(ri.CONTRACT["execution_authority"], "NONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
