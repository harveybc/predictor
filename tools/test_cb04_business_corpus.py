"""The frozen business corpus: sealed, held apart, and opened only on the record.

Runs against the real corpus files and the real loader. The ledger is the one
mutable artifact, so each test that opens labels restores the ledger it found,
and one test asserts that opening labels is what appends to it.

Run:  python tools/test_cb04_business_corpus.py
"""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import business_corpus as bc  # noqa: E402


class LedgerRestored(unittest.TestCase):
    """Anything that opens labels writes to the ledger; put it back afterwards."""

    def setUp(self):
        self.saved = bc.LEDGER_PATH.read_bytes() if bc.LEDGER_PATH.exists() else b""

    def tearDown(self):
        bc.LEDGER_PATH.write_bytes(self.saved)


class Seals(unittest.TestCase):
    def test_both_files_match_the_manifest(self):
        seals = bc.verify_seals()
        recorded = bc.manifest()
        self.assertEqual(seals["items_sha256"], recorded["items_sha256"])
        self.assertEqual(seals["labels_sha256"], recorded["labels_sha256"])
        self.assertNotEqual(seals["items_sha256"], seals["labels_sha256"])

    def test_a_moved_corpus_is_refused_rather_than_used(self):
        saved = bc.ITEMS_PATH.read_bytes()
        try:
            bc.ITEMS_PATH.write_bytes(saved + b'{"item_id":"injected"}\n')
            with self.assertRaises(bc.CorpusRefused) as caught:
                bc.load_items()
            self.assertIn("frozen corpus", str(caught.exception))
        finally:
            bc.ITEMS_PATH.write_bytes(saved)
        bc.verify_seals()

    def test_the_manifest_records_how_the_corpus_was_built(self):
        recorded = bc.manifest()
        self.assertEqual(recorded["corpus_class"], "BUSINESS_HELD_OUT")
        self.assertGreaterEqual(len(recorded["construction_procedure"]), 5)
        self.assertIn("contract_sha256", recorded["built_after_the_protocol"])
        self.assertIn("implementation_sha256", recorded["built_after_the_protocol"])

    def test_the_protocol_pin_follows_the_contract_through_its_amendments(self):
        """The pin is appended to, never rewritten.

        The original pin records the contract as it stood when the corpus was
        sealed and stays exactly as written. When the contract changes, an
        amendment is appended saying why and carrying the new digests, and the
        head of that chain must be the contract as it stands now.
        """
        import hashlib
        pin = bc.manifest()["built_after_the_protocol"]
        head = (pin.get("amendments") or [pin])[-1]
        for name, path in (("contract_sha256", "docs/contracts/classification_metrics.v1.json"),
                           ("implementation_sha256", "app/classification_receipt.py")):
            actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            self.assertEqual(head[name], actual,
                             f"{path} changed after the corpus was sealed; append an amendment "
                             f"to built_after_the_protocol explaining the change")

    def test_every_amendment_states_that_the_corpus_bytes_did_not_move(self):
        recorded = bc.manifest()
        for amendment in recorded["built_after_the_protocol"].get("amendments") or []:
            self.assertIs(amendment["corpus_bytes_unchanged"], True)
            self.assertEqual(amendment["items_sha256"], recorded["items_sha256"])
            self.assertEqual(amendment["labels_sha256"], recorded["labels_sha256"])
            self.assertTrue(amendment["why"])
            self.assertTrue(amendment["what_was_not_done"])
        bc.verify_seals()

    def test_the_original_pin_is_still_the_original_pin(self):
        pin = bc.manifest()["built_after_the_protocol"]
        self.assertEqual(pin["contract_sha256"],
                         "006d81865b635cc87770942709ce3d3f3dbfd8735346f170a679903f5bd7cb77",
                         "the pin the corpus was sealed against may not be rewritten")

    def test_the_named_refusal_is_recorded_and_not_dressed_up(self):
        recorded = bc.manifest()
        self.assertIn("INDEPENDENT_THIRD_PARTY_LABELS_UNAVAILABLE_NO_FEED_ENTITLEMENT",
                      recorded["named_refusals"])
        self.assertIn("not independently labelled: the labels are the author's",
                      recorded["what_it_is_not"])
        self.assertIs(recorded["authorises_broker_deployment"], False)
        self.assertEqual(recorded["execution_authority"], "NONE")


class HeldApart(unittest.TestCase):
    def test_the_items_file_carries_no_label(self):
        text = bc.ITEMS_PATH.read_text()
        for vocabulary in ("related", "unrelated", "unclear", "restatement", "correction",
                           "immediate", "session"):
            self.assertNotIn(f'"{vocabulary}"', text,
                             f"the items file leaks the label token {vocabulary!r}")

    def test_items_and_labels_are_separate_files_covering_the_same_rows(self):
        items = bc.load_items()
        labels = json.loads(json.dumps([
            json.loads(line) for line in bc.LABELS_PATH.read_text().splitlines() if line.strip()]))
        self.assertEqual({item["item_id"] for item in items},
                         {row["item_id"] for row in labels})
        self.assertEqual(len(items), bc.manifest()["rows"])

    def test_loading_items_records_nothing(self):
        before = len(bc.ledger_entries())
        bc.load_items()
        bc.load_items()
        self.assertEqual(len(bc.ledger_entries()), before)


class NewsUseNotSentiment(unittest.TestCase):
    def test_all_three_questions_are_news_use_questions(self):
        questions = bc.manifest()["questions"]
        self.assertEqual(set(questions), set(bc.QUESTIONS))
        self.assertIn("relevance", questions)
        self.assertIn("novelty", questions)
        self.assertIn("window", questions)
        self.assertNotIn("sentiment", questions)

    def test_every_label_value_is_populated(self):
        for question in bc.QUESTIONS:
            counts = bc.label_counts(question)
            for value, count in counts.items():
                self.assertGreater(count, 0, f"{question}={value} has no rows")

    def test_the_corpus_contains_restatement_chains_and_a_correction(self):
        counts = bc.label_counts("novelty")
        self.assertGreaterEqual(counts["restatement"], 5)
        self.assertGreaterEqual(counts["correction"], 3)

    def test_the_corpus_contains_positive_tone_irrelevance(self):
        items = {item["item_id"]: item for item in bc.load_items()}
        tone = items["b011"]["headline"] + items["b011"]["body"]
        self.assertIn("best", tone.lower())

    def test_the_corpus_contains_both_interface_languages(self):
        languages = {item["language"] for item in bc.load_items()}
        self.assertEqual(languages, {"en", "es"})

    def test_every_item_is_marked_synthetic(self):
        self.assertTrue(all(item["synthetic"] is True for item in bc.load_items()))


class OpeningTheLabels(LedgerRestored):
    def test_labels_without_a_stated_reason_are_refused(self):
        with self.assertRaises(bc.CorpusRefused):
            bc.open_labels(actor="satoshi", reason="", provider="p")
        with self.assertRaises(bc.CorpusRefused) as caught:
            bc.open_labels(actor="satoshi", reason="because", provider="p")
        self.assertIn("untouched", str(caught.exception))
        with self.assertRaises(bc.CorpusRefused):
            bc.open_labels(actor="", reason="scoring a candidate provider once", provider="p")
        self.assertEqual(len(bc.ledger_entries()), 0)

    def test_opening_the_labels_appends_a_numbered_use(self):
        first = bc.open_labels(actor="satoshi", provider="NON_MODEL_FIXTURE",
                               reason="contract test of the use ledger, no model run")
        self.assertEqual(len(first), bc.manifest()["rows"])
        entries = bc.ledger_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["use_index"], 1)
        self.assertEqual(entries[0]["labels_sha256"], bc.manifest()["labels_sha256"])
        bc.open_labels(actor="satoshi", provider="NON_MODEL_FIXTURE",
                       reason="second contract test of the use ledger, no model run")
        entries = bc.ledger_entries()
        self.assertEqual([entry["use_index"] for entry in entries], [1, 2])

    def test_a_second_use_is_visible_rather_than_prevented(self):
        for index in range(3):
            bc.open_labels(actor="satoshi", provider="NON_MODEL_FIXTURE",
                           reason=f"repeat number {index} of a contract test, no model run")
        self.assertEqual(len(bc.ledger_entries()), 3,
                         "the ledger must make repeated scoring visible")


class NaiveMayNotSeeTheAnswers(unittest.TestCase):
    def test_a_prior_equal_to_the_held_out_distribution_is_refused(self):
        with self.assertRaises(bc.CorpusRefused) as caught:
            bc.refuse_prior_fitted_on_held_out("relevance", bc.label_counts("relevance"))
        self.assertIn("not a baseline", str(caught.exception))

    def test_a_development_prior_is_accepted(self):
        accepted = bc.refuse_prior_fitted_on_held_out(
            "relevance", {"related": 40, "unrelated": 40, "unclear": 20})
        self.assertEqual(accepted["related"], 40)

    def test_an_unknown_question_is_refused(self):
        with self.assertRaises(bc.CorpusRefused):
            bc.label_counts("sentiment")


class PopulationBinding(unittest.TestCase):
    def test_the_population_digest_covers_the_rows_and_the_question(self):
        identifiers = [item["item_id"] for item in bc.load_items()]
        whole = bc.evaluation_population_sha256("relevance", identifiers)
        self.assertEqual(len(whole), 64)
        self.assertNotEqual(whole, bc.evaluation_population_sha256("novelty", identifiers))
        self.assertNotEqual(whole, bc.evaluation_population_sha256("relevance",
                                                                  identifiers[:-1]))
        self.assertNotEqual(whole, bc.evaluation_population_sha256("relevance",
                                                                  list(reversed(identifiers))))

    def test_rows_outside_the_corpus_are_refused(self):
        with self.assertRaises(bc.CorpusRefused) as caught:
            bc.evaluation_population_sha256("relevance", ["b001", "not-a-row"])
        self.assertIn("not-a-row", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
