"""The producer-to-store boundary, against the DEPLOYED provider on a disposable cube.

The finding this file exists for is measured here rather than asserted: the
installed general-purpose store accepts a terminal whose tags declare a
non-model answering path and claim a model result, from any producer and under
any actor name. `test_the_store_alone_is_not_a_defence` writes exactly that
terminal through the real provider and shows it stored. The admission gate then
refuses the same bytes by reason.

Nothing is mocked. The store is `PredictorDuckdbStore` from site-packages, the
same distribution the warehouse host runs, and it writes only to a temporary
DuckDB file.

Run with the store host's environment:
  ~/.venvs/store-hosts-duckdb-prod/bin/python tools/test_classification_provenance_store.py
"""
import copy
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

from sqlalchemy import text                                       # noqa: E402
from predictor_duckdb_store.provider import PredictorDuckdbStore  # noqa: E402

from app import classification_receipt as cr                      # noqa: E402
from app import classification_provenance as cp                   # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from test_classification_provenance import (                      # noqa: E402
    WITHDRAWN_WORD, H, document, model_path, non_model_path)


def terminal_for(receipt, unit_id, *, actor="satoshi", classification="NON_GOVERNING"):
    body = dict(
        schema="governed_terminal.v1", campaign_sha256="c" * 64,
        campaign_key="provenance-boundary-disposable", unit_id=unit_id, generation=1,
        actor=actor, project="predictor", classification=classification,
        status="COMPLETED", reason=None,
        started_at="2026-09-29T00:00:00Z", finished_at="2026-09-29T00:00:01Z",
        terminal_lake="olap_cube", config_sha256=cr.sha256_of(cr.CONTRACT),
        code_identity={"kind": "git_commit", "value": "d" * 40},
        costs={"wall_seconds": 1.0}, tags=cr.terminal_tags(receipt),
        synthetic_spec_sha256=None, deliveries=["0" * 32],
        metrics=cr.terminal_metrics(receipt), verified_datasets=[], artifacts=[])
    body["terminal_sha256"] = cr.sha256_of(body)
    return body


def forged(body, **tag_over):
    """The same terminal with its tags edited after sealing, as a stranger would."""
    body = copy.deepcopy(body)
    body["tags"].update(tag_over)
    body["terminal_sha256"] = cr.sha256_of(
        {k: v for k, v in body.items() if k != "terminal_sha256"})
    return body


class ProvenanceAtTheStoreBoundary(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="provenance-boundary-disposable-")
        self.store = PredictorDuckdbStore()
        self.store.set_params(duckdb_path=str(Path(self.temp.name) / "cube.duckdb"),
                              schema="main", memory_limit="256MB", threads=1,
                              min_free_bytes=1)
        self.store.engine()

    def tearDown(self):
        self.store.engine().dispose()
        self.temp.cleanup()

    def declared_test(self):
        return cr.build_receipt(document(path=non_model_path(),
                                         evidence_class="TRANSPORT_TEST_NOT_SCIENCE"))

    def model_measurement(self):
        return cr.build_receipt(document(path=model_path(), evidence_class="MEASUREMENT"))

    # -- the substitution guard ------------------------------------------------
    def test_nothing_under_test_was_substituted(self):
        self.assertIn("site-packages", inspect.getfile(PredictorDuckdbStore))
        self.assertEqual(PredictorDuckdbStore.__module__, "predictor_duckdb_store.provider")
        self.assertEqual(self.store.write_terminal.__func__,
                         PredictorDuckdbStore.write_terminal)
        self.assertEqual(inspect.getmodule(cp.admit_classification_terminal).__name__,
                         "app.classification_provenance")

    # -- the finding -----------------------------------------------------------
    def test_the_store_alone_is_not_a_defence(self):
        """Measured, not asserted: the real store takes the promotion from anyone."""
        body = forged(terminal_for(self.declared_test(), "promoted"),
                      evidence_class="MEASUREMENT", evidence_role=cp.MODEL_RESULT,
                      weights_present="true", answering_path_kind="MODEL_CHECKPOINT_LOADED",
                      served_checkpoint_sha256=H["weights"], checkpoint_sha256=H["weights"])
        self.assertTrue(self.store.write_terminal(body)["stored"],
                        "if this ever fails, the store grew a gate of its own")
        with self.store.engine().connect() as con:
            stored = json.loads(con.execute(text(
                "SELECT tags_json FROM gov_terminal WHERE terminal_sha256=:d"),
                {"d": body["terminal_sha256"]}).scalar())
        self.assertEqual(stored["evidence_role"], cp.MODEL_RESULT)
        # and the gate refuses the same bytes, by reason
        with self.assertRaises(cp.AdmissionRefused) as caught:
            cp.admit_classification_terminal(body)
        self.assertEqual(caught.exception.refusal, cp.ADMISSION_PROVENANCE_DIGEST_MISMATCH)

    def test_a_promotion_with_a_consistently_rewritten_digest_is_still_refused(self):
        """Rewriting the digest too does not help: the rule is the path, not the seal."""
        receipt = self.declared_test()
        body = terminal_for(receipt, "promoted-2")
        path = dict(receipt["answering_path"])
        body = forged(body, evidence_class="MEASUREMENT", evidence_role=cp.MODEL_RESULT,
                      provenance_sha256=cp.provenance_sha256(path))
        with self.assertRaises(cp.AdmissionRefused) as caught:
            cp.admit_classification_terminal(body)
        self.assertEqual(caught.exception.refusal, cp.NO_WEIGHTS_IS_NOT_A_MODEL_RESULT)
        self.assertNotIn(WITHDRAWN_WORD, str(caught.exception).lower())

    def test_an_evidence_role_that_contradicts_the_declared_path_is_refused(self):
        receipt = self.declared_test()
        body = forged(terminal_for(receipt, "role"), evidence_role=cp.MODEL_RESULT,
                      provenance_sha256=cp.provenance_sha256(dict(receipt["answering_path"])))
        with self.assertRaises(cp.AdmissionRefused) as caught:
            cp.admit_classification_terminal(body)
        self.assertEqual(caught.exception.refusal, cp.ADMISSION_ROLE_CONTRADICTS_THE_PATH)

    def test_a_non_model_path_may_not_be_a_governing_terminal(self):
        body = terminal_for(self.declared_test(), "governing", classification="GOVERNING")
        with self.assertRaises(cp.AdmissionRefused) as caught:
            cp.admit_classification_terminal(body)
        self.assertEqual(caught.exception.refusal, cp.ADMISSION_NON_MODEL_MAY_NOT_GOVERN)

    def test_a_classification_terminal_without_provenance_tags_is_refused(self):
        body = terminal_for(self.declared_test(), "bare")
        for tag in ("answering_path_kind", "weights_present", "evidence_role"):
            body["tags"].pop(tag)
        with self.assertRaises(cp.AdmissionRefused) as caught:
            cp.admit_classification_terminal(body)
        self.assertEqual(caught.exception.refusal, cp.ADMISSION_PROVENANCE_TAGS_MISSING)

    # -- the two cases that must keep working ---------------------------------
    def test_a_declared_test_is_admitted_stored_and_still_reads_as_a_test(self):
        body = terminal_for(self.declared_test(), "declared", actor=WITHDRAWN_WORD)
        admission = cp.admit_classification_terminal(body)
        self.assertTrue(admission["admitted"])
        self.assertEqual(admission["evidence_role"], cp.DECLARED_NON_MODEL_TEST)
        self.assertTrue(self.store.write_terminal(body)["stored"])
        with self.store.engine().connect() as con:
            row = con.execute(text(
                "SELECT actor,"
                " json_extract_string(tags_json, '$.evidence_role'),"
                " json_extract_string(tags_json, '$.answering_path_kind'),"
                " json_extract_string(tags_json, '$.weights_present'),"
                " json_extract_string(tags_json, '$.served_checkpoint_sha256')"
                " FROM gov_terminal WHERE unit_id='declared'")).fetchone()
        self.assertEqual(row[0], WITHDRAWN_WORD, "the actor's name is incidental")
        self.assertEqual(row[1], cp.DECLARED_NON_MODEL_TEST)
        self.assertEqual(row[2], "NON_MODEL_CONSTANT")
        self.assertEqual(row[3], "false")
        self.assertEqual(row[4], cp.NO_CHECKPOINT)

    def test_a_measurement_whose_prose_carries_the_word_is_admitted_and_stored(self):
        receipt = cr.build_receipt(document(
            path=model_path(), evidence_class="MEASUREMENT",
            task_id=f"news_relevance_{WITHDRAWN_WORD}_window.v1",
            limitations=f"a {WITHDRAWN_WORD} corpus was replaced by the real one midway"))
        body = terminal_for(receipt, "measured", actor="nightly-eval")
        admission = cp.admit_classification_terminal(body)
        self.assertEqual(admission["evidence_role"], cp.MODEL_RESULT)
        self.assertTrue(self.store.write_terminal(body)["stored"])
        with self.store.engine().connect() as con:
            role = con.execute(text(
                "SELECT json_extract_string(tags_json, '$.evidence_role')"
                " FROM gov_terminal WHERE unit_id='measured'")).scalar()
        self.assertEqual(role, cp.MODEL_RESULT)

    def test_a_query_separates_model_results_from_declared_tests(self):
        for receipt, unit in ((self.declared_test(), "d1"), (self.model_measurement(), "m1")):
            self.store.write_terminal(terminal_for(receipt, unit))
        with self.store.engine().connect() as con:
            rows = con.execute(text(
                "SELECT unit_id FROM gov_terminal"
                " WHERE json_extract_string(tags_json, '$.metric_contract') = :s"
                "   AND json_extract_string(tags_json, '$.evidence_role') = :r"
                " ORDER BY unit_id"), {"s": cr.SCHEMA, "r": cp.MODEL_RESULT}).fetchall()
        self.assertEqual([r[0] for r in rows], ["m1"],
                         "a declared test must never answer a query for model results")

    def test_another_contract_passes_through_untouched(self):
        """The general-purpose warehouse stays generic."""
        body = terminal_for(self.declared_test(), "other")
        body["tags"] = {"metric_contract": "tsl_literature_metrics.v1"}
        body["terminal_sha256"] = cr.sha256_of(
            {k: v for k, v in body.items() if k != "terminal_sha256"})
        admission = cp.admit_classification_terminal(body)
        self.assertTrue(admission["admitted"])
        self.assertEqual(admission["evidence_role"], cp.NOT_THIS_CONTRACT)
        self.assertTrue(self.store.write_terminal(body)["stored"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
