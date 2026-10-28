"""Only paired, extractible target sets can enter the weekly wave."""

import pytest

from tools import fs4_wave_frontier as wave


def example():
    folds = {f"fold-{n}": {"RAW": 0.2, "RANDOM_ENCODER": 0.3,
                           "TRAINED_ENCODER": 0.1, "naive_mae": 0.4, "population_n": 10}
             for n in range(5)}
    partial = {"schema": "fs4.extractibility_partial_complete.v1",
               "state": "EXTRACTIBILITY_PARTIAL_COMPLETE", "closure_sha256": "partial",
               "denominator": {"admitted": 30, "complete": 30, "not_available_for_train": 0,
                               "sum_equals_admitted": True},
               "populations": {"EURUSD": {"features": {f: {"status": "MEASURED", "folds": folds}
                                                    for f in ("a", "b")}}}}
    cons = {"population_id": "EURUSD", "identity": "data", "consolidated_sha256": "cons",
            "sets": [{"set_id": "s1", "target_id": "Y1", "members": ["a"], "k": 1},
                     {"set_id": "s2", "target_id": "Y1", "members": ["b"], "k": 1},
                     {"set_id": "other", "target_id": "Y1", "members": ["a", "b"], "k": 2}]}
    return cons, partial, {"Y1": ["s1", "s2"]}


def test_seals_only_target_coverage():
    cons, partial, target_sets = example()
    seal = wave.build_seal(cons, partial, target_sets)
    assert seal["frontier_set_ids"] == ["s1", "s2"]
    assert seal["denominator"] == {"sets": 3, "in_frontier": 2, "deferred": 1}
    wave.validate_seal(seal)


def test_refuses_missing_fold_or_arm():
    cons, partial, target_sets = example()
    del partial["populations"]["EURUSD"]["features"]["b"]["folds"]["fold-4"]["TRAINED_ENCODER"]
    with pytest.raises(wave.Refusal, match="INCOMPLETE_EXTRACTIBILITY"):
        wave.build_seal(cons, partial, target_sets)


def test_refuses_unknown_set():
    cons, partial, target_sets = example()
    target_sets["Y1"][1] = "missing"
    with pytest.raises(wave.Refusal, match="TARGET_SET_MISSING"):
        wave.build_seal(cons, partial, target_sets)


def test_rejects_rewritten_seal():
    cons, partial, target_sets = example()
    seal = wave.build_seal(cons, partial, target_sets)
    seal["frontier_set_ids"].pop()
    with pytest.raises(wave.Refusal, match="CORRUPT"):
        wave.validate_seal(seal)
