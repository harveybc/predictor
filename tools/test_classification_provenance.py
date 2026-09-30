"""The producer side of the provenance contract: a declared test cannot become a model result.

Why these tests exist
---------------------
Two live facts. In the product, a declared non-model path published a real
checkpoint's macro-F1 and its `n` beside its own answers. In the warehouse, a
receipt written by a non-model path carried a 64-hex `checkpoint_sha256` that was
a digest of nothing a model ever loaded, and was shaped exactly like a real one.

The obvious repair - refuse a particular word - is not a contract. A declared
test may be named anything, and a real measurement may legitimately contain an
awkward word in its own prose. So what is checked here is **where the evidence
came from**: which path answered, whether weights were present, which checkpoint
served, and whether the record being quoted belongs to that same path.

`test_the_gate_never_matches_a_word` is load-bearing: it fails if anybody
reintroduces a string match on the withdrawn word into the gate.

Run:  python tools/test_classification_provenance.py
"""
import ast
import copy
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
from app import classification_provenance as cp         # noqa: E402

#: the word the product's first repair matched on. It appears in this test file,
#: which is the point: it must appear nowhere in the gate.
WITHDRAWN_WORD = "fix" "ture"

H = {name: hashlib.sha256(name.encode()).hexdigest() for name in
     ("eval", "train", "calib", "corpus", "weights", "other_weights",
      "protocol", "scorer")}


def model_path(**over):
    """A path that loaded a checkpoint and observed itself doing so."""
    path = {"path_id": "predictor_plugins.binary.ann",
            "kind": "MODEL_CHECKPOINT_LOADED",
            "weights_present": True,
            "served_checkpoint": "ann_binary_h1.keras",
            "served_checkpoint_sha256": H["weights"],
            "attestation": "OBSERVED_FROM_ANSWERING_PATH"}
    path.update(over)
    return path


def non_model_path(**over):
    """A declared non-model path: a constant answer table, and it says so."""
    path = {"path_id": "canned_answer_table.v1",
            "kind": "NON_MODEL_CONSTANT",
            "weights_present": False,
            "served_checkpoint": cp.NO_CHECKPOINT,
            "served_checkpoint_sha256": cp.NO_CHECKPOINT,
            "attestation": "OBSERVED_FROM_ANSWERING_PATH"}
    path.update(over)
    return path


def document(*, path, evidence_class, corpus_class="BUSINESS_HELD_OUT",
             family="MACRO_F1", name="macro-F1", value=8 / 9,
             task_id="news_relevance_eurusd.v1", limitations=None, **over):
    """A receipt document that is valid apart from whatever a test changes."""
    matrix = []
    for index in range(3):
        row = [0, 0, 0, 1]
        row[index] = 8
        row[(index + 1) % 3] = 1
        matrix.append(row)
    total = sum(sum(row) for row in matrix)
    abstained = sum(row[-1] for row in matrix)
    body = {
        "task_id": task_id,
        "corpus_class": corpus_class,
        "corpus_id": "cb04_business_news.v1",
        "corpus_sha256": H["corpus"],
        "evidence_class": evidence_class,
        "supervision_regime": "ZERO_SHOT_PROMPTED",
        "labelled_rows_fit_head": False,
        "provider": "laya_news",
        "checkpoint": path["served_checkpoint"],
        "checkpoint_sha256": path["served_checkpoint_sha256"],
        "answering_path": copy.deepcopy(path),
        "author_primary_metric": {"family": family, "name": name, "value": value,
                                  "denominator_policy": "ANSWERED_ONLY"},
        "paired_naive": {"family": family, "policy": "MAJORITY_CLASS_FROM_TRAIN",
                         "value": 0.25, "seed": None,
                         "train_label_population_sha256": H["train"],
                         "evaluation_population_sha256": H["eval"]},
        "class_vocabulary": ["related", "unrelated", "unclear"],
        "per_class_confusion": matrix,
        "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED",
        "calibrated": False,
        "calibration_split": {"split_id": "NONE", "population_sha256": None,
                              "rows": 0, "fitted_parameters": None},
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
        "limitations": limitations or "one evaluation over the declared rows; nothing else",
        "protocol_sha256": H["protocol"],
        "scorer_sha256": H["scorer"],
        "seed": "1",
    }
    body.update(over)
    return body


class TheGateIsNotASpellCheck(unittest.TestCase):
    """The withdrawn word must be incidental to every refusal."""

    def test_the_gate_never_matches_a_word(self):
        source = Path(inspect.getfile(cp)).read_text()
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                body = getattr(node, "body", None) or []
                if body and isinstance(body[0], ast.Expr) and \
                        isinstance(body[0].value, ast.Constant) and \
                        isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        offending = [node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)
                     and id(node) not in docstrings
                     and WITHDRAWN_WORD in node.value.lower()]
        self.assertEqual(offending, [],
                         "the gate must validate provenance, not match a word")

    def test_the_gate_reads_its_vocabulary_from_the_contract(self):
        """No second copy of the vocabulary that could drift from the contract."""
        self.assertEqual(set(cp.PATH_KINDS), set(cr.CONTRACT["answering_path"]["kinds"]))
        self.assertEqual(cp.MODEL_KINDS,
                         {kind for kind, spec in cr.CONTRACT["answering_path"]["kinds"].items()
                          if spec["weights_present"]})
        self.assertTrue(cp.MODEL_KINDS)
        self.assertTrue(set(cp.PATH_KINDS) - cp.MODEL_KINDS)


class AProvenanceBlockThatMisstatesItself(unittest.TestCase):

    def test_an_undeclared_kind_is_refused_by_name(self):
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(document(path=model_path(kind="SOMETHING_ELSE"),
                                      evidence_class="MEASUREMENT"))
        self.assertEqual(caught.exception.refusal, cp.KIND_NOT_DECLARED)

    def test_a_kind_that_contradicts_its_own_weights_flag_is_refused(self):
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(document(path=non_model_path(weights_present=True),
                                      evidence_class="DECLARATION"))
        self.assertEqual(caught.exception.refusal, cp.KIND_CONTRADICTS_WEIGHTS)

    def test_a_path_that_served_no_checkpoint_may_not_carry_a_digest(self):
        """The defect in its original shape: a fabricated 64-hex checkpoint digest."""
        path = non_model_path(served_checkpoint_sha256=H["weights"])
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(document(path=path, evidence_class="DECLARATION",
                                      checkpoint_sha256=H["weights"]))
        self.assertEqual(caught.exception.refusal, cp.DIGEST_ON_A_PATH_THAT_SERVED_NONE)

    def test_a_serving_path_must_name_the_checkpoint_it_served(self):
        path = model_path(served_checkpoint_sha256=cp.NO_CHECKPOINT)
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(document(path=path, evidence_class="MEASUREMENT",
                                      checkpoint_sha256=cp.NO_CHECKPOINT))
        self.assertEqual(caught.exception.refusal, cp.SERVING_PATH_MUST_NAME_ITS_CHECKPOINT)

    def test_an_undeclared_attestation_is_refused(self):
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(document(path=model_path(attestation="SOMEBODY_SAID_SO"),
                                      evidence_class="MEASUREMENT"))
        self.assertEqual(caught.exception.refusal, cp.ATTESTATION_NOT_DECLARED)

    def test_a_missing_provenance_block_is_refused_rather_than_assumed(self):
        """Absence is not coincidence: an undeclared path is not a model path."""
        body = document(path=model_path(), evidence_class="MEASUREMENT")
        del body["answering_path"]
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(body)


class TheRecordMustBelongToTheAnsweringPath(unittest.TestCase):

    def test_a_receipt_may_not_quote_a_checkpoint_the_path_did_not_serve(self):
        body = document(path=model_path(), evidence_class="MEASUREMENT")
        body["checkpoint_sha256"] = H["other_weights"]
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(body)
        self.assertEqual(caught.exception.refusal, cp.RECORD_IS_NOT_OF_THE_ANSWERING_PATH)
        self.assertIn(H["other_weights"][:12], str(caught.exception))

    def test_a_receipt_may_not_quote_a_checkpoint_name_the_path_did_not_serve(self):
        body = document(path=model_path(), evidence_class="MEASUREMENT")
        body["checkpoint"] = "some_other_model.keras"
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(body)
        self.assertEqual(caught.exception.refusal, cp.RECORD_IS_NOT_OF_THE_ANSWERING_PATH)


class APromotionIntoAModelResult(unittest.TestCase):

    def test_a_path_with_no_weights_cannot_be_a_measurement(self):
        with self.assertRaises(cp.PromotionRefused) as caught:
            cr.build_receipt(document(path=non_model_path(),
                                      evidence_class="MEASUREMENT"))
        self.assertEqual(caught.exception.refusal, cp.NO_WEIGHTS_IS_NOT_A_MODEL_RESULT)
        self.assertIn("NON_MODEL_CONSTANT", str(caught.exception))

    def test_the_promotion_is_refused_when_no_field_carries_the_withdrawn_word(self):
        """The counterexample that decides the design.

        Every string in this document is scrubbed of the word the product's first
        repair matched on. A gate built on that word admits this. This one does
        not, because it reads the declared path instead.
        """
        body = document(path=non_model_path(path_id="canned_answer_table.v1"),
                        evidence_class="MEASUREMENT",
                        task_id="news_relevance_eurusd.v1",
                        limitations="answers came from a constant table")
        self.assertNotIn(WITHDRAWN_WORD, json.dumps(body).lower(),
                         "this counterexample is only meaningful without the word")
        with self.assertRaises(cp.PromotionRefused) as caught:
            cr.build_receipt(body)
        self.assertEqual(caught.exception.refusal, cp.NO_WEIGHTS_IS_NOT_A_MODEL_RESULT)
        self.assertNotIn(WITHDRAWN_WORD, str(caught.exception).lower())

    def test_a_measurement_needs_a_path_that_was_actually_established(self):
        with self.assertRaises(cp.PromotionRefused) as caught:
            cr.build_receipt(document(path=model_path(attestation="NOT_ESTABLISHED"),
                                      evidence_class="MEASUREMENT"))
        self.assertEqual(caught.exception.refusal, cp.MEASUREMENT_WITHOUT_AN_ESTABLISHED_PATH)

    def test_a_declared_smoke_corpus_cannot_carry_a_measurement(self):
        with self.assertRaises(cp.PromotionRefused) as caught:
            cr.build_receipt(document(path=model_path(), evidence_class="MEASUREMENT",
                                      corpus_class="SMOKE_FIXTURE"))
        self.assertEqual(caught.exception.refusal, cp.DECLARED_CORPUS_IS_NOT_A_MEASUREMENT)


class ADeclaredTestStaysADeclaredTest(unittest.TestCase):
    """Kept, and legible as what it is."""

    def test_a_declared_non_model_receipt_is_built_and_says_so_on_its_face(self):
        receipt = cr.build_receipt(document(path=non_model_path(),
                                            evidence_class="TRANSPORT_TEST_NOT_SCIENCE"))
        self.assertEqual(receipt["evidence_role"], cp.DECLARED_NON_MODEL_TEST)
        self.assertEqual(receipt["answering_path"]["kind"], "NON_MODEL_CONSTANT")
        self.assertIs(receipt["answering_path"]["weights_present"], False)
        self.assertEqual(receipt["checkpoint_sha256"], cp.NO_CHECKPOINT)
        tags = cr.terminal_tags(receipt)
        self.assertEqual(tags["evidence_role"], cp.DECLARED_NON_MODEL_TEST)
        self.assertEqual(tags["weights_present"], "false")
        self.assertEqual(tags["answering_path_kind"], "NON_MODEL_CONSTANT")
        self.assertEqual(tags["served_checkpoint_sha256"], cp.NO_CHECKPOINT)
        self.assertEqual(tags["provenance_sha256"],
                         cp.provenance_sha256(receipt["answering_path"]))

    def test_a_declaration_from_a_non_model_path_is_accepted(self):
        receipt = cr.build_receipt(document(path=non_model_path(kind="NON_MODEL_RULE"),
                                            evidence_class="DECLARATION"))
        self.assertEqual(receipt["evidence_role"], cp.DECLARED_NON_MODEL_TEST)

    def test_a_published_reference_is_a_path_not_run_here(self):
        path = non_model_path(kind="THIRD_PARTY_PATH_NOT_RUN_HERE",
                              path_id="author_published_table",
                              served_checkpoint="laya@010bacef",
                              attestation="NOT_ESTABLISHED")
        receipt = cr.build_receipt(document(path=path,
                                            evidence_class="PUBLISHED_REFERENCE",
                                            checkpoint="laya@010bacef"))
        self.assertEqual(receipt["evidence_role"], cp.DECLARED_NON_MODEL_TEST)
        self.assertEqual(receipt["checkpoint_sha256"], cp.NO_CHECKPOINT,
                         "a value read from a table is not bytes we hold")


class ARealMeasurementIsNotBlocked(unittest.TestCase):
    """The second counterexample: an awkward word in honest prose changes nothing."""

    def test_a_model_measurement_whose_own_text_carries_the_word_is_accepted(self):
        body = document(path=model_path(), evidence_class="MEASUREMENT",
                        task_id=f"news_relevance_{WITHDRAWN_WORD}_window.v1",
                        limitations=f"the evaluation harness replaced a {WITHDRAWN_WORD} "
                                    f"with the real corpus midway; see the protocol")
        self.assertIn(WITHDRAWN_WORD, json.dumps(body).lower())
        receipt = cr.build_receipt(body)
        self.assertEqual(receipt["evidence_role"], cp.MODEL_RESULT)
        self.assertEqual(cr.terminal_tags(receipt)["evidence_role"], cp.MODEL_RESULT)
        self.assertEqual(receipt["checkpoint_sha256"], H["weights"])

    def test_a_model_that_was_never_checkpointed_can_still_be_measured(self):
        """An in-process fit has no file to digest. Refusing it would be over-reach."""
        path = model_path(kind="MODEL_IN_PROCESS_NOT_CHECKPOINTED",
                          served_checkpoint="ann_binary_h1",
                          served_checkpoint_sha256=cp.NOT_DIGESTED)
        receipt = cr.build_receipt(document(path=path, evidence_class="MEASUREMENT",
                                            checkpoint="ann_binary_h1",
                                            checkpoint_sha256=cp.NOT_DIGESTED))
        self.assertEqual(receipt["evidence_role"], cp.MODEL_RESULT)

    def test_a_path_with_no_weights_may_not_claim_it_was_not_digested(self):
        path = non_model_path(served_checkpoint_sha256=cp.NOT_DIGESTED)
        with self.assertRaises(cp.ProvenanceRefused) as caught:
            cr.build_receipt(document(path=path, evidence_class="DECLARATION",
                                      checkpoint_sha256=cp.NOT_DIGESTED))
        self.assertEqual(caught.exception.refusal, cp.DIGEST_ON_A_PATH_THAT_SERVED_NONE)


class ABadgeThatWouldRestOnADeclaredTest(unittest.TestCase):

    def business_measurement(self):
        return cr.build_receipt(document(path=model_path(), evidence_class="MEASUREMENT",
                                         corpus_class="BUSINESS_HELD_OUT"))

    def test_a_real_business_measurement_still_earns_the_badge(self):
        badge = cr.provider_quality_badge("laya_news", [self.business_measurement()])
        self.assertEqual(badge["execution_authority"], "NONE")
        self.assertEqual(badge["business_evidence"][0]["answering_path_kind"],
                         "MODEL_CHECKPOINT_LOADED")

    def test_a_badge_refuses_a_record_with_no_declared_provenance(self):
        record = self.business_measurement()
        del record["answering_path"]
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("laya_news", [record])
        self.assertEqual(caught.exception.refusal, cp.BADGE_PROVENANCE_NOT_DECLARED)

    def test_a_badge_refuses_a_declared_non_model_path_even_when_relabelled(self):
        """The builder refuses this document; the badge refuses the forged record too."""
        record = self.business_measurement()
        record["answering_path"] = non_model_path()
        record["evidence_role"] = cp.MODEL_RESULT          # forged
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("laya_news", [record])
        self.assertEqual(caught.exception.refusal, cp.BADGE_NON_MODEL_ANSWERING_PATH)
        self.assertNotIn(WITHDRAWN_WORD, str(caught.exception).lower())

    def test_a_badge_refuses_a_path_that_was_only_declared_by_a_configuration(self):
        record = self.business_measurement()
        record["answering_path"]["attestation"] = "DECLARED_BY_CONFIGURATION"
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("laya_news", [record])
        self.assertEqual(caught.exception.refusal, cp.BADGE_ANSWERING_PATH_NOT_OBSERVED)

    def test_a_badge_refuses_a_record_whose_checkpoint_is_not_the_one_that_served(self):
        record = self.business_measurement()
        record["checkpoint_sha256"] = H["other_weights"]
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("laya_news", [record])
        self.assertEqual(caught.exception.refusal, cp.BADGE_RECORD_NOT_OF_THE_PATH)


class TheContractItself(unittest.TestCase):

    def test_the_contract_declares_the_provenance_field_and_its_tags(self):
        contract = cr.CONTRACT
        self.assertIn("answering_path", contract["required_receipt_fields"])
        for tag in ("answering_path_id", "answering_path_kind", "weights_present",
                    "served_checkpoint", "served_checkpoint_sha256",
                    "provenance_attestation", "provenance_sha256", "evidence_role"):
            self.assertIn(tag, contract["required_context_tags"], tag)

    def test_every_projected_tag_is_present_for_both_roles(self):
        for path, evidence in ((model_path(), "MEASUREMENT"),
                               (non_model_path(), "TRANSPORT_TEST_NOT_SCIENCE")):
            receipt = cr.build_receipt(document(path=path, evidence_class=evidence))
            tags = cr.terminal_tags(receipt)
            for name in cr.CONTRACT["required_context_tags"]:
                self.assertIn(name, tags)
                self.assertIsInstance(tags[name], str)

    def test_the_badge_refusals_are_declared_in_the_contract(self):
        declared = cr.CONTRACT["provider_quality_badge"]["refusals"]
        for refusal in (cp.BADGE_PROVENANCE_NOT_DECLARED, cp.BADGE_NON_MODEL_ANSWERING_PATH,
                        cp.BADGE_ANSWERING_PATH_NOT_OBSERVED, cp.BADGE_RECORD_NOT_OF_THE_PATH):
            self.assertIn(refusal, declared)

    def test_the_promotion_refusals_are_declared_in_the_contract(self):
        declared = cr.CONTRACT["promotion_rule"]["refusals"]
        for refusal in (cp.NO_WEIGHTS_IS_NOT_A_MODEL_RESULT,
                        cp.MEASUREMENT_WITHOUT_AN_ESTABLISHED_PATH,
                        cp.DECLARED_CORPUS_IS_NOT_A_MEASUREMENT):
            self.assertIn(refusal, declared)

    def test_the_provenance_digest_changes_with_every_field(self):
        base = cp.provenance_sha256(model_path())
        for field, value in (("path_id", "another_path"), ("kind", "MODEL_IN_PROCESS_NOT_CHECKPOINTED"),
                             ("weights_present", False), ("served_checkpoint", "other.keras"),
                             ("served_checkpoint_sha256", H["other_weights"]),
                             ("attestation", "DECLARED_BY_CONFIGURATION")):
            self.assertNotEqual(base, cp.provenance_sha256(model_path(**{field: value})), field)


class NothingUnderTestWasSubstituted(unittest.TestCase):

    def test_the_functions_are_the_modules_own(self):
        for function in (cr.build_receipt, cr.terminal_tags, cr.provider_quality_badge):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_receipt")
        for function in (cp.build_answering_path, cp.promotion_check,
                         cp.admit_classification_terminal, cp.provenance_sha256):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_provenance")
        self.assertEqual(inspect.getfile(cp), str(ROOT / "app/classification_provenance.py"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
