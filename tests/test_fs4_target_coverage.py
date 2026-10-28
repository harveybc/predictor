"""A successor wave covers targets without reopening the whole inventory."""

from tools import fs4_target_coverage as coverage


def test_picks_two_distinct_sets_and_only_missing_features():
    manifest = {"schema": "fs3.preliminary_gpu_triage_bundle.v1", "final_selection": False,
                "populations": [{"population_id": "EURUSD", "gpu_feature_ids": ["a"],
                                 "deferred_feature_ids": ["b", "c", "d"]}]}
    partial = {"schema": "fs4.extractibility_partial_complete.v1",
               "state": "EXTRACTIBILITY_PARTIAL_COMPLETE", "closure_sha256": "parent",
               "populations": {"EURUSD": {"features": {"a": {"status": "MEASURED", "folds": {}},
                                                     "d": {"status": "NOT_AVAILABLE_FOR_TRAIN"}}}}}
    consolidated = {"population_id": "EURUSD", "sets": [
        {"set_id": "s1", "target_id": "Y1", "k": 2, "members": ["a", "b"], "methods": ["MRMR"]},
        {"set_id": "s2", "target_id": "Y1", "k": 2, "members": ["a", "c"], "methods": ["JMI"]},
        {"set_id": "bad", "target_id": "Y1", "k": 2, "members": ["a", "d"], "methods": ["JMI"]},
        {"set_id": "huge", "target_id": "Y1", "k": 99, "members": ["a", "b"], "methods": ["ALL_ADMISSIBLE"]}]}
    ranks = {"Y1": {"a": 1, "b": 2, "c": 3, "d": 4}}
    result = coverage.plan_wave(manifest, partial, {"EURUSD": consolidated}, {"EURUSD": ranks})
    assert result["target_sets"]["EURUSD"]["Y1"] == ["s1", "s2"]
    assert result["added_features"]["EURUSD"] == ["b", "c"]
    assert result["coverage_gaps"] == []


def test_names_uncovered_target():
    manifest = {"schema": "fs3.preliminary_gpu_triage_bundle.v1", "final_selection": False,
                "populations": [{"population_id": "ETH", "gpu_feature_ids": ["a"],
                                 "deferred_feature_ids": ["d"]}]}
    partial = {"schema": "fs4.extractibility_partial_complete.v1",
               "state": "EXTRACTIBILITY_PARTIAL_COMPLETE", "closure_sha256": "parent",
               "populations": {"ETH": {"features": {"d": {"status": "NOT_AVAILABLE_FOR_TRAIN"}}}}}
    sets = {"population_id": "ETH", "sets": [{"set_id": "bad", "target_id": "Y1", "k": 1,
                                               "members": ["d"], "methods": ["JMI"]}]}
    result = coverage.plan_wave(manifest, partial, {"ETH": sets}, {"ETH": {"Y1": {"d": 1}}})
    assert result["coverage_gaps"] == [{"population_id": "ETH", "target_id": "Y1", "available_sets": 0}]
