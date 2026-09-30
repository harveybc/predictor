#!/usr/bin/env python3
"""Tests for the BANKING77 native reference path lane.

Three kinds of assertion, deliberately mixed:

* ``TheCorrectedScope`` fails if the blanket impossibility sentence comes back, or if
  the retained measurement it sits beside is edited rather than cited. It reads the
  RETAINED evidence file, so a change to that file breaks these tests: that is the
  point - the correction travels with its measurement or not at all.
* ``TheOfflineRecipe`` proves the pieces the native path no longer needs anything for,
  and each has a NEGATIVE CONTROL: a wrong seed, a wrong shuffle regime and an
  unbalanced population each have to produce a different answer, or the check is not
  measuring what it claims.
* ``NoScoreAndNoBadge`` fails if any number in the report is presented as our own
  measurement, or if a cross-family comparison stops refusing.

    python3 -m pytest tools/test_banking77_native_path.py -q
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import df_banking77_native_path_20260929 as NP  # noqa: E402

EVIDENCE = ROOT / "docs/audits/evidence/banking77_native_path_20260929"
REPORT = EVIDENCE / "NATIVE_PATH.json"
RETAINED_FIT = ROOT / "docs/audits/evidence/cb04_row_identity_20260929/BANKING77_LABEL_FIT.json"
DOC = ROOT / "docs/audits/work_plan/SATOSHI_BANKING77_NATIVE_PATH_2026_09_29.md"


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip(f"{REPORT.name} not produced yet")
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def train_labels():
    _, categories, digest = NP.read_governed_csv(NP.TRAIN_SHA)
    assert digest == NP.TRAIN_SHA
    return categories


class TestTheCorrectedScope:
    def test_the_narrow_sentence_is_the_one_published(self):
        scope = NP.corrected_scope()
        assert "unsupported by the SHIPPED one-shot wrapper" in scope["corrected_sentence"]
        assert "not impossibility for any budget" in scope["corrected_sentence"]

    def test_the_blanket_impossibility_is_withdrawn_and_nothing_else_is(self):
        flags = NP.corrected_scope()["scope_flags"]
        assert flags["impossible_at_any_budget_provider_or_model"] is False
        assert flags["unsupported_by_the_shipped_wrapper_and_its_serializations"] is True

    def test_a_larger_budget_is_not_the_same_reference(self):
        flags = NP.corrected_scope()["scope_flags"]
        assert flags["a_larger_budget_is_a_different_configuration_not_the_same_reference"]

    def test_a_shortlist_must_carry_its_routing_errors_over_the_full_population(self):
        flags = NP.corrected_scope()["scope_flags"]
        assert flags["a_hierarchy_or_retrieval_shortlist_is_a_distinct_method"]
        assert flags["such_a_method_scores_its_routing_errors_over_the_full_population"]

    def test_the_retained_measurement_is_cited_not_revised(self):
        """Every number in the correction must still be the retained file's own."""
        retained = json.loads(RETAINED_FIT.read_text())
        cited = NP.corrected_scope()["retained_numbers_cited_not_revised"]
        blob = json.dumps(retained)
        assert str(cited["max_options_in_the_shipped_question_builder"]) in blob
        assert str(cited["option_tokens_empty_description_floor"]) in blob
        assert str(cited["option_tokens_label_as_its_own_description"]) in blob
        for budget in cited["tested_head_budgets"]:
            assert str(budget) in blob

    def test_the_retained_evidence_digest_is_recorded(self):
        scope = NP.corrected_scope()
        assert scope["retained_fit_evidence_sha256"]
        assert scope["the_measurement_is_retained_unchanged"] is True

    def test_the_two_questions_are_named_separately(self):
        both = NP.corrected_scope()["two_questions_never_blended"]
        assert set(both) == {"native_author_reproduction", "alternative_framework_adaptation"}
        assert "not evidence for the first" in both["alternative_framework_adaptation"]


class TestTheOfflineRecipe:
    def test_the_governed_population_rehashes_to_its_pins(self):
        population = NP.population_identity()
        assert population["train"]["matches"] is True
        assert population["test"]["matches"] is True
        assert population["train"]["rows"] == population["train"]["rows_pinned"] == 10003
        assert population["test"]["rows"] == population["test"]["rows_pinned"] == 3080

    def test_the_test_split_is_exactly_balanced(self):
        population = NP.population_identity()
        assert population["test"]["exactly_balanced"] is True
        assert population["test"]["support_per_class"] == 40
        assert population["test"]["distinct_labels"] == 77

    def test_balance_forces_the_three_identities(self):
        """Independent of any file: equal support makes weighted equal unweighted."""
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        rng = np.random.default_rng(7)
        y_true = np.repeat(np.arange(77), 40)
        y_pred = np.where(rng.random(y_true.size) < 0.9, y_true,
                          rng.integers(0, 77, y_true.size))
        assert f1_score(y_true, y_pred, average="macro") == pytest.approx(
            f1_score(y_true, y_pred, average="weighted"))
        assert recall_score(y_true, y_pred, average="macro") == pytest.approx(
            accuracy_score(y_true, y_pred))

    def test_negative_control_an_unbalanced_population_breaks_them(self):
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        rng = np.random.default_rng(7)
        y_true = np.concatenate([np.zeros(500, int), np.arange(1, 77).repeat(10)])
        y_pred = np.where(rng.random(y_true.size) < 0.9, y_true,
                          rng.integers(0, 77, y_true.size))
        assert f1_score(y_true, y_pred, average="macro") != pytest.approx(
            f1_score(y_true, y_pred, average="weighted"))
        assert recall_score(y_true, y_pred, average="macro") != pytest.approx(
            accuracy_score(y_true, y_pred))

    def test_the_ten_draws_are_distinct_and_exactly_eight_per_label(self, train_labels):
        draws = NP.deterministic_draws(train_labels)
        assert draws["n_experiments"] == 10
        assert draws["all_draws_distinct"] is True
        assert draws["every_draw_is_exactly_k_per_label"] is True
        for record in draws["draws"]:
            assert record["rows"] == 616 == draws["expected_rows_per_draw"]
            assert record["distinct_labels"] == 77

    def test_the_draws_are_reproducible_bit_for_bit(self, train_labels):
        first = NP.deterministic_draws(train_labels)["draw_set_sha256"]
        second = NP.deterministic_draws(train_labels)["draw_set_sha256"]
        assert first == second

    def test_negative_control_a_wrong_seed_changes_every_draw(self, train_labels):
        """If the seed did not matter, the recorded digests would pin nothing."""
        pinned = NP.deterministic_draws(train_labels)
        original = NP.SEED
        try:
            NP.SEED = 43
            other = NP.deterministic_draws(train_labels)
        finally:
            NP.SEED = original
        assert other["draw_set_sha256"] != pinned["draw_set_sha256"]
        for a, b in zip(pinned["draws"], other["draws"]):
            assert a["index_set_sha256"] != b["index_set_sha256"]

    def test_negative_control_resetting_the_index_order_collapses_the_draws(
            self, train_labels):
        """The source carries the shuffled indices forward; a fresh list per experiment
        would make all ten experiments identical. That wrong design must not reproduce
        the recorded digests."""
        wrong = []
        for _ in range(NP.N_EXPERIMENTS):
            idxs = list(range(len(train_labels)))          # the defect: fresh each time
            np.random.RandomState(NP.SEED).shuffle(idxs)
            counter: dict[str, int] = defaultdict(int)
            sampled = []
            for i in idxs:
                label = train_labels[i]
                if counter[label] < NP.SAMPLES_PER_LABEL:
                    sampled.append(i)
                    counter[label] += 1
            wrong.append(NP.digest_of(list(sampled)))
        assert len(set(wrong)) == 1, "the wrong design must collapse to one draw"
        pinned = NP.deterministic_draws(train_labels)
        assert NP.digest_of(wrong) != pinned["draw_set_sha256"]

    def test_the_label_id_mapping_does_not_move_the_draw(self, train_labels):
        """The greedy counter buckets by equality, so renaming the labels must not
        change which rows are drawn - including under the byte-sorted map that would
        mislabel most classes."""
        ids = {name: n for n, name in enumerate(sorted(set(train_labels)))}
        renamed = [str(ids[name]) for name in train_labels]
        assert (NP.deterministic_draws(renamed)["draw_set_sha256"]
                == NP.deterministic_draws(train_labels)["draw_set_sha256"])

    def test_the_train_majority_naive_is_the_pinned_one(self):
        naive = NP.population_identity()["same_row_naive_carried_forward"]
        assert (naive["majority_accuracy_recomputed_on_the_governed_test_rows"]
                == pytest.approx(naive["majority_accuracy_pinned"]))
        assert naive["majority_accuracy_pinned"] == pytest.approx(1 / 77)


class TestThePublishedTarget:
    def test_the_artifact_reproduces_its_own_aggregate(self, report):
        target = report["published_target"]
        assert target["artifact_read"] is True
        assert target["aggregate_reproduces_the_published_value"] is True
        assert target["accuracy"] == NP.PUBLISHED_ACCURACY
        assert target["n_experiments_in_the_artifact"] == 10

    def test_the_dataset_revision_matches_the_contract_pin(self, report):
        assert report["published_target"]["dataset_revision_matches_the_contract_pin"]

    def test_the_artifact_shows_the_identities_the_balance_predicts(self, report):
        identities = report["published_target"]["identities_observed_in_the_artifact"]
        assert all(identities.values())

    def test_the_published_row_is_not_our_measurement(self, report):
        target = report["published_target"]
        assert target["evidence_class"] == "PUBLISHED_REFERENCE"
        assert target["this_is_not_our_measurement"] is True

    def test_the_device_is_not_claimed(self, report):
        assert report["published_target"]["device_recorded_in_the_artifact"] is None


class TestTheDependencyLedger:
    def test_the_blocked_components_are_named_and_priced(self, report):
        ledger = report["dependency_ledger"]
        model = ledger["components"]["model_artifacts_at_the_pinned_revision"]
        assert model["state"] == "MISSING"
        assert model["bytes_whole_repository"] == sum(NP.MODEL_FILES.values())
        assert "download allocation" in model["requires"]

    def test_the_existing_environments_are_not_offered_as_substitutes(self, report):
        env = report["dependency_ledger"]["components"]["third_pinned_environment"]
        assert env["state"] == "MISSING"
        assert "mteb" in env["missing_in_every_existing_environment"]
        assert "sentence-transformers" in env["missing_in_every_existing_environment"]

    def test_everything_else_on_the_path_is_pinned(self, report):
        ledger = report["dependency_ledger"]
        assert ledger["everything_else_on_the_native_path_is_pinned"] is True
        assert ledger["no_authority_is_invented_here"] is True

    def test_the_commands_are_prepared_and_not_run(self, report):
        prepared = report["prepared_commands"]
        assert "NOT run" in prepared["what_this_is"]
        assert report["dependency_ledger"][
            "no_download_install_training_or_gpu_run_happened_in_this_lane"] is True


class TestNoScoreAndNoBadge:
    def test_the_verdict_produces_no_score(self, report):
        assert report["no_score_is_produced_or_fabricated_here"] is True
        assert report["verdict"]["no_badge"] is True
        assert report["verdict"]["no_warehouse_row_written"] is True
        assert report["verdict"]["native_path_state"] == "PINNED_AND_BLOCKED_ON_ACQUISITION"

    def test_no_row_in_the_report_claims_to_be_a_measurement(self, report):
        blob = json.dumps(report)
        assert '"MEASUREMENT"' not in blob
        assert "BUSINESS_HELD_OUT" not in blob

    def test_the_router_corpus_appears_nowhere(self, report):
        blob = json.dumps(report).lower()
        assert "19-prompt" not in blob and "router" not in blob

    def test_a_cross_family_comparison_still_refuses_by_name(self):
        import app.classification_receipt as cr
        left = {"author_primary_metric": {"family": "ACCURACY", "name": "accuracy",
                                          "unit": "u", "value": 0.914578}}
        right = {"author_primary_metric": {"family": "MACRO_F1", "name": "f1",
                                           "unit": "u", "value": 0.913809}}
        with pytest.raises(cr.IncomparableMetrics):
            cr.compare(left, right)

    def test_macro_and_weighted_f1_refuse_even_where_they_coincide(self):
        """The balance makes the two numbers equal here. They are still two families."""
        import app.classification_receipt as cr
        left = {"author_primary_metric": {"family": "MACRO_F1", "name": "f1",
                                          "unit": "u", "value": 0.913809}}
        right = {"author_primary_metric": {"family": "WEIGHTED_F1", "name": "f1_weighted",
                                          "unit": "u", "value": 0.913809}}
        with pytest.raises(cr.IncomparableMetrics):
            cr.compare(left, right)


class TestTheReturn:
    def test_the_deliverable_exists_and_is_signed_by_the_successor(self):
        if not DOC.exists():
            pytest.skip("the return is not written yet")
        text = DOC.read_text()
        assert "Satoshi, successor technical lead" in text
        assert "0.914578" in text

    def test_the_return_carries_the_corrected_sentence_beside_the_measurement(self):
        if not DOC.exists():
            pytest.skip("the return is not written yet")
        text = DOC.read_text()
        assert "706" in text and "894" in text and "12" in text
        assert "not impossibility for any budget" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
