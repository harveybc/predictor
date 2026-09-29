"""End-to-end: classification receipts through the DEPLOYED provider, on a disposable cube.

These are the tests that matter. A recent defect in this repository stayed
invisible because its test substituted the real function away, so here:

* the store is the installed `PredictorDuckdbStore`, the same distribution the
  warehouse host runs, resolved from site-packages and asserted to be so;
* the receipt is built by `app.classification_receipt.build_receipt`, the same
  function the evaluation path calls, asserted to be the module's own function;
* nothing is patched, mocked, stubbed or wrapped. `TestCase` has no `setUp` that
  replaces a collaborator, and `test_nothing_under_test_was_substituted` fails if
  a later edit introduces one.

It writes only to a temporary DuckDB file and never touches production.

Run with the store host's environment, which has the provider installed:
  /path/to/store-venv/bin/python tools/test_classification_warehouse_receipt.py
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text                                   # noqa: E402
from predictor_duckdb_store.provider import PredictorDuckdbStore  # noqa: E402

from app import classification_receipt as cr                  # noqa: E402

CONTRACT = cr.CONTRACT
H = {name: hashlib.sha256(name.encode()).hexdigest() for name in
     ("eval", "train", "calib", "corpus", "ckpt", "protocol", "scorer")}


def receipt_document(family, name, value, *, corpus_class="BUSINESS_HELD_OUT",
                     evidence_class="TRANSPORT_TEST_NOT_SCIENCE"):
    """A valid receipt document. Values are fabricated: this is a transport test."""
    # symmetric confusion: 8 correct, 1 wrong, 1 abstained per class, so accuracy
    # and macro-F1 are both exactly 8/9 and cannot be told apart by value.
    matrix = []
    for index in range(3):
        row = [0, 0, 0, 1]
        row[index] = 8
        row[(index + 1) % 3] = 1
        matrix.append(row)
    total = sum(sum(row) for row in matrix)
    abstained = sum(row[-1] for row in matrix)
    return {
        "task_id": "news_relevance_eurusd.v1",
        "corpus_class": corpus_class,
        "corpus_id": "cb04_business_news.v1",
        "corpus_sha256": H["corpus"],
        "evidence_class": evidence_class,
        "supervision_regime": "ZERO_SHOT_PROMPTED",
        "labelled_rows_fit_head": False,
        "provider": "NON_MODEL_FIXTURE",
        "checkpoint": "NON_MODEL_FIXTURE",
        "checkpoint_sha256": H["ckpt"],
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
        "limitations": "fabricated transport fixture; it is not a model score",
        "protocol_sha256": H["protocol"],
        "scorer_sha256": H["scorer"],
        "seed": "1",
    }


def terminal_for(receipt, unit_id):
    """One governed terminal carrying this receipt, in the existing schema."""
    body = dict(
        schema="governed_terminal.v1",
        campaign_sha256="c" * 64,
        campaign_key="cb04-classification-disposable-fixture",
        unit_id=unit_id, generation=1, actor="fixture", project="predictor",
        classification="NON_GOVERNING", status="COMPLETED", reason=None,
        started_at="2026-09-28T00:00:00Z", finished_at="2026-09-28T00:00:01Z",
        terminal_lake="olap_cube", config_sha256=cr.sha256_of(CONTRACT),
        code_identity={"kind": "git_commit", "value": "d" * 40},
        costs={"wall_seconds": 1.0},
        tags=cr.terminal_tags(receipt),
        synthetic_spec_sha256=None, deliveries=["0" * 32],
        metrics=cr.terminal_metrics(receipt),
        verified_datasets=[], artifacts=[])
    body["terminal_sha256"] = cr.sha256_of(body)
    return body


class WarehouseReceipts(unittest.TestCase):
    """Real provider, real builder, temporary database."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cb04-classification-disposable-")
        self.store = PredictorDuckdbStore()
        self.store.set_params(duckdb_path=str(Path(self.temp.name) / "cube.duckdb"),
                              schema="main", memory_limit="256MB", threads=1,
                              min_free_bytes=1)
        self.store.engine()

    def tearDown(self):
        self.store.engine().dispose()
        self.temp.cleanup()

    # -- the substitution guard ------------------------------------------------
    def test_nothing_under_test_was_substituted(self):
        self.assertIn("site-packages", inspect.getfile(PredictorDuckdbStore))
        self.assertEqual(PredictorDuckdbStore.__module__,
                         "predictor_duckdb_store.provider")
        self.assertIs(type(self.store), PredictorDuckdbStore)
        self.assertEqual(self.store.write_terminal.__func__,
                         PredictorDuckdbStore.write_terminal)
        for function in (cr.build_receipt, cr.terminal_tags, cr.terminal_metrics,
                         cr.provider_quality_badge, cr.read_metric, cr.compare):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_receipt")
        self.assertEqual(inspect.getfile(cr),
                         str(ROOT / "app/classification_receipt.py"))

    # -- the roundtrip ---------------------------------------------------------
    def test_a_classification_receipt_roundtrips_field_for_field(self):
        receipt = cr.build_receipt(receipt_document("MACRO_F1", "macro-F1", 8 / 9))
        body = terminal_for(receipt, "macro-f1-1")
        self.assertTrue(self.store.write_terminal(body)["stored"])
        with self.store.engine().connect() as con:
            stored_tags = json.loads(con.execute(
                text("SELECT tags_json FROM gov_terminal WHERE terminal_sha256=:d"),
                {"d": body["terminal_sha256"]}).scalar())
            self.assertEqual(stored_tags, body["tags"])
            rows = con.execute(text(
                "SELECT metric, value, unit, split FROM gov_terminal_metric"
                " WHERE terminal_sha256=:d ORDER BY metric"),
                {"d": body["terminal_sha256"]}).fetchall()
        expected = sorted((m["metric"], m["value"], m["unit"], m["split"])
                          for m in body["metrics"])
        self.assertEqual([tuple(r) for r in rows], expected)
        # the seven named facts survive as their own stored fields
        self.assertEqual(stored_tags["author_primary_metric_name"], "macro-F1")
        self.assertEqual(stored_tags["naive_policy"], "MAJORITY_CLASS_FROM_TRAIN")
        self.assertEqual(json.loads(stored_tags["class_vocabulary_json"]),
                         ["related", "unrelated", "unclear"])
        self.assertEqual(stored_tags["confusion_sha256"],
                         receipt["per_class_confusion"]["confusion_sha256"])
        self.assertEqual(stored_tags["probability_semantics"],
                         "SOFTMAX_POSTERIOR_UNCALIBRATED")
        self.assertEqual(stored_tags["calibrated"], "false")
        self.assertEqual(stored_tags["calibration_split"], "NONE")
        self.assertEqual(stored_tags["population_abstained"], "3")
        self.assertEqual(stored_tags["execution_authority"], "NONE")

    def test_resubmitting_the_same_receipt_does_not_duplicate_rows(self):
        receipt = cr.build_receipt(receipt_document("MAP", "MAP", 0.71))
        body = terminal_for(receipt, "map-1")
        self.store.write_terminal(body)
        self.store.write_terminal(body)
        self.store.write_terminal(body)
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), 1)
            self.assertEqual(con.execute(text(
                "SELECT count(*) FROM gov_terminal_metric")).scalar(), len(body["metrics"]))

    def test_the_rows_survive_disposing_and_reopening_the_connection(self):
        receipt = cr.build_receipt(receipt_document("MAP", "MAP", 0.71))
        body = terminal_for(receipt, "map-1")
        self.store.write_terminal(body)
        self.store.engine().dispose()
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), 1)
            self.assertEqual(con.execute(text(
                "SELECT count(*) FROM gov_terminal_metric")).scalar(), len(body["metrics"]))

    # -- the rule, at the far end of the store --------------------------------
    def test_three_families_with_one_value_stay_three_rows_in_the_warehouse(self):
        """8/9 as MAP, as accuracy and as macro-F1: one number, three meanings."""
        value = 8 / 9
        for family, name, unit_id in (("MAP", "MAP", "map"),
                                      ("ACCURACY", "accuracy", "acc"),
                                      ("MACRO_F1", "macro-F1", "macrof1")):
            receipt = cr.build_receipt(receipt_document(family, name, value))
            self.store.write_terminal(terminal_for(receipt, unit_id))
        with self.store.engine().connect() as con:
            rows = con.execute(text(
                "SELECT m.metric, m.unit, m.value,"
                " json_extract_string(t.tags_json, '$.author_primary_metric_family') AS family,"
                " json_extract_string(t.tags_json, '$.author_primary_metric_name') AS author_name,"
                " json_extract_string(t.tags_json, '$.metric_identity_sha256') AS identity"
                " FROM gov_terminal t JOIN gov_terminal_metric m USING (terminal_sha256)"
                " WHERE m.value = :v AND m.metric NOT LIKE 'classification.naive_%'"
                " ORDER BY m.metric"), {"v": value}).fetchall()
        self.assertEqual(len(rows), 3, "one stored value must remain three distinct rows")
        self.assertEqual({r[0] for r in rows},
                         {"classification.map", "classification.accuracy",
                          "classification.macro_f1"})
        self.assertEqual(len({r[1] for r in rows}), 3, "each family keeps its own unit")
        self.assertEqual(len({r[5] for r in rows}), 3, "each family keeps its own identity")
        by_metric = {r[0]: r for r in rows}
        self.assertEqual(by_metric["classification.map"][3], "MAP")
        self.assertEqual(by_metric["classification.accuracy"][4], "accuracy")
        self.assertEqual(by_metric["classification.macro_f1"][4], "macro-F1")

    def test_a_query_for_accuracy_cannot_return_the_map_row(self):
        value = 8 / 9
        for family, name, unit_id in (("MAP", "MAP", "map"),
                                      ("ACCURACY", "accuracy", "acc")):
            receipt = cr.build_receipt(receipt_document(family, name, value))
            self.store.write_terminal(terminal_for(receipt, unit_id))
        with self.store.engine().connect() as con:
            accuracy = con.execute(text(
                "SELECT unit, value FROM gov_terminal_metric"
                " WHERE metric = 'classification.accuracy'")).fetchall()
            mapped = con.execute(text(
                "SELECT unit, value FROM gov_terminal_metric"
                " WHERE metric = 'classification.map'")).fetchall()
        self.assertEqual(len(accuracy), 1)
        self.assertEqual(len(mapped), 1)
        self.assertEqual(accuracy[0][0], "accuracy_fraction")
        self.assertEqual(mapped[0][0], "map_ranking_fraction")
        self.assertEqual(accuracy[0][1], mapped[0][1])   # same number
        self.assertNotEqual(accuracy[0][0], mapped[0][0])  # never the same metric

    def test_reading_two_stored_receipts_of_different_families_still_refuses(self):
        """Comparison refuses on the way out, not only on the way in."""
        stored = {}
        for family, name, unit_id in (("MAP", "MAP", "map"),
                                      ("MACRO_F1", "macro-F1", "macrof1")):
            receipt = cr.build_receipt(receipt_document(family, name, 8 / 9))
            self.store.write_terminal(terminal_for(receipt, unit_id))
            stored[family] = receipt
        with self.store.engine().connect() as con:
            families = [r[0] for r in con.execute(text(
                "SELECT json_extract_string(tags_json, '$.author_primary_metric_family')"
                " FROM gov_terminal ORDER BY unit_id")).fetchall()]
        self.assertEqual(sorted(families), ["MACRO_F1", "MAP"])
        with self.assertRaises(cr.IncomparableMetrics) as caught:
            cr.compare(stored["MAP"], stored["MACRO_F1"])
        self.assertIn("MAP", str(caught.exception))
        self.assertIn("MACRO_F1", str(caught.exception))

    # -- abstention survives the store ---------------------------------------
    def test_abstention_coverage_and_the_confusion_close_from_stored_rows_alone(self):
        receipt = cr.build_receipt(receipt_document("MACRO_F1", "macro-F1", 8 / 9))
        body = terminal_for(receipt, "macro-f1-1")
        self.store.write_terminal(body)
        with self.store.engine().connect() as con:
            rows = dict(con.execute(text(
                "SELECT metric, value FROM gov_terminal_metric"
                " WHERE terminal_sha256=:d"), {"d": body["terminal_sha256"]}).fetchall())
            tags = json.loads(con.execute(text(
                "SELECT tags_json FROM gov_terminal WHERE terminal_sha256=:d"),
                {"d": body["terminal_sha256"]}).scalar())
        total = int(tags["population_total"])
        support = sum(rows[f"classification.support.class_{i}"] for i in range(3))
        abstained = sum(rows[f"classification.abstained.class_{i}"] for i in range(3))
        self.assertEqual(support, total)
        self.assertEqual(abstained, int(tags["population_abstained"]))
        self.assertAlmostEqual(rows["classification.abstention_coverage"], abstained / total)
        self.assertAlmostEqual(rows["classification.abstention_coverage"]
                               + rows["classification.answered_fraction"], 1.0)

    # -- refusals, with nothing left behind -----------------------------------
    def test_a_non_finite_metric_is_refused_and_stores_nothing(self):
        receipt = cr.build_receipt(receipt_document("MAP", "MAP", 0.71))
        for bad in (float("nan"), float("inf"), float("-inf")):
            body = terminal_for(receipt, "map-1")
            body["metrics"][0]["value"] = bad
            del body["terminal_sha256"]
            try:
                body["terminal_sha256"] = cr.sha256_of(body)
            except ValueError:
                continue  # the canonical form itself refuses a non-finite number
            with self.assertRaises(Exception):
                self.store.write_terminal(body)
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), 0)

    def test_a_router_record_has_no_path_into_the_warehouse(self):
        router = cr.build_router_reliability_record(
            corpus_id=CONTRACT["router_corpus"]["corpus_id"],
            correct=79, stored_verdicts=95, prompts=19, repeats=5,
            evidence_class="RECOUNT_OF_STORED_VERDICTS")
        for projection in (cr.terminal_tags, cr.terminal_metrics):
            with self.assertRaises(cr.ReceiptRefused) as caught:
                projection(router)
            self.assertIn("router", str(caught.exception).lower())
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(dict(receipt_document("ACCURACY", "accuracy", 8 / 9),
                                  corpus_class="ROUTER_PROMPT_CORPUS",
                                  corpus_id=CONTRACT["router_corpus"]["corpus_id"]))
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), 0)
            self.assertEqual(con.execute(text(
                "SELECT count(*) FROM gov_terminal_metric")).scalar(), 0)

    def test_a_receipt_whose_metric_contradicts_its_confusion_never_reaches_the_store(self):
        with self.assertRaises(cr.ReceiptRefused):
            cr.build_receipt(dict(receipt_document("ACCURACY", "accuracy", 0.99)))
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), 0)

    # -- the badge, from what the warehouse actually holds --------------------
    def test_a_transport_fixture_in_the_warehouse_yields_no_quality_badge(self):
        receipt = cr.build_receipt(receipt_document("MACRO_F1", "macro-F1", 8 / 9))
        self.store.write_terminal(terminal_for(receipt, "macro-f1-1"))
        with self.store.engine().connect() as con:
            evidence_class = con.execute(text(
                "SELECT json_extract_string(tags_json, '$.evidence_class')"
                " FROM gov_terminal")).scalar()
        self.assertEqual(evidence_class, "TRANSPORT_TEST_NOT_SCIENCE")
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("NON_MODEL_FIXTURE", [receipt])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_TRANSPORT_TEST_IS_NOT_SCIENCE")

    def test_a_public_benchmark_row_in_the_warehouse_yields_no_quality_badge(self):
        receipt = cr.build_receipt(receipt_document(
            "ACCURACY", "accuracy", 8 / 9,
            corpus_class="PUBLIC_BENCHMARK", evidence_class="MEASUREMENT"))
        self.store.write_terminal(terminal_for(receipt, "acc-public"))
        with self.assertRaises(cr.BadgeRefused) as caught:
            cr.provider_quality_badge("NON_MODEL_FIXTURE", [receipt])
        self.assertEqual(caught.exception.refusal,
                         "BADGE_REFUSED_NO_BUSINESS_CORPUS_MEASUREMENT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
