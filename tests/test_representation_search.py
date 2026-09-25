"""WP06 stage 5: what the search may propose, what it refuses, and what a transform does to a prediction.

Two files are exercised here and no model is fitted: `tools/search_representation.py` (the space, the decoding, the
spec it builds and the refusals it raises before anything is dispatched) and the representation arithmetic of
`tools/fit_pipeline_spec.py` (differencing the input channels, the origin-anchored target transforms and the
inversion that puts a prediction back in kW).

The property that matters most is the last one: whatever a stage models, the truth of a sealed row is a level in kW,
so a transform must be invertible from what the origin itself carries. A transform that is not is a stage that cannot
be scored on the sealed population, and the harness must refuse it rather than score it on other rows.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ENG_SRC = Path("/home/harveybc/Documents/GitHub/.worktrees/feature-eng-m5phet-hierarchical")


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


search = _module("search_representation")
fitter = _module("fit_pipeline_spec")

COLUMNS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
           "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

HAND = {
    "schema": "m5phet.pipeline.v1", "stage": "baseline_hand", "provenance": "DEVELOPMENT",
    "execution_authorized": False, "decisions": [], "features": COLUMNS, "preprocessing": {},
    "extractors": {"all": {"plugin": "HAND", "decision": None}},
    "representation": {"schema": "m5phet.representation.v1", "candidate_id": "hand_household_w60",
                       "provenance": "DEVELOPMENT", "target": {"column": "Global_active_power", "transform": "level"},
                       "windows": [60], "lags": [1], "differencing": {"order": 0}, "features": [],
                       "exogenous": COLUMNS[1:], "calendar": {"clock": "receipt", "columns": []},
                       "sampling": {"step_seconds": 60, "timezone": "UTC"}, "holdout": {"fraction": 0.2}},
    "grouping": {"decision": None, "chosen_by": "HAND", "k": 1, "why": "one block",
                 "groups": [{"group_id": "all", "members": COLUMNS}]},
    "core": {"key": "fused_branches", "chosen_by": "HAND", "decision": None, "why": "the closest declared plugin",
             "encoder_mapping": {"core": "fused_branches", "status": "MAPPED",
                                 "branches": {"all": {"encoder": "tcn", "extractor": "HAND", "status": "MAPPED",
                                                      "mapped_by": "hand", "why": "the tcn family"}}}},
}


def genome(*, window=60, lags=(1,), transform="level", order=0, features=tuple(COLUMNS)):
    bits = [1 if lag in lags else 0 for lag in search.SEARCHABLE_LAGS]
    columns = [1 if name in features else 0 for name in COLUMNS]
    return [window, *bits, search.TRANSFORMS.index(transform), order, *columns]


def point_of(**kwargs):
    return search.decode(genome(**kwargs), COLUMNS)


def check(point, *, seal_window=197):
    spec = search.pipeline_spec_of(point, hand=HAND, stage="s", candidate_id="c", budget_sentence="B.")
    return search.validate(spec["representation"], point=point, seal_window=seal_window,
                           feature_eng_src=FEATURE_ENG_SRC)


# --------------------------------------------------------------------------------------------------- the space

def test_the_declared_space_is_the_genome_and_nothing_else():
    declared = search.space(seal_window=197, columns=COLUMNS)
    genes = [declared["window"]["gene"], *declared["lags"]["genes"], declared["transform"]["gene"],
             declared["differencing_order"]["gene"], *declared["features"]["genes"]]
    assert sorted(genes) == list(range(search.GENOME_LENGTH))
    assert declared["window"]["high"] == 197 and declared["window"]["low"] == search.MIN_WINDOW
    assert declared["lags"]["values"] == [1, 74, 197]
    assert declared["lags"]["motivated_set"] == [1, 74, 197, 1443, 2892]
    assert set(declared["lags"]["excluded"]) == {"1443", "2892"}
    assert all("LAG_EXCEEDS_SEALING_WINDOW" in why for why in declared["lags"]["excluded"].values())
    assert declared["transform"]["values"] == ["level", "diff", "log_return"]
    low, high = search.bounds(seal_window=197)
    assert len(low) == len(high) == search.GENOME_LENGTH and high[0] == 197


def test_a_genome_decodes_to_the_five_declared_quantities():
    point = point_of(window=74, lags=(1, 74), transform="diff", order=0, features=COLUMNS[:2])
    assert point == {"window": 74, "lags": [1, 74], "transform": "diff", "differencing_order": 0,
                     "features": COLUMNS[:2]}


# ------------------------------------------------------------------------------------------------ the refusals

def test_a_lag_beyond_the_window_is_refused_by_the_name_that_kept_the_long_candidates_out():
    code, why = check(point_of(window=110, lags=(1, 197)))
    assert code == search.LAG_EXCEEDS_WINDOW
    assert "seasonal_lag_1443" in why and "reused" in why
    # and the two peaks beyond the sealing window are not in the space at all, by that same refusal
    assert 1443 not in search.SEARCHABLE_LAGS and 2892 not in search.SEARCHABLE_LAGS
    assert point_of(window=197, lags=(1, 1443))["lags"] == [1]


def test_the_empty_subset_and_the_empty_lag_set_are_refused_and_not_repaired():
    assert check(point_of(features=()))[0] == search.NO_FEATURE_SELECTED
    assert check(point_of(lags=()))[0] == search.NO_LAG_DECLARED


def test_a_differencing_order_under_a_differencing_transform_is_refused_by_the_spec_itself():
    code, why = check(point_of(transform="diff", order=1))
    assert code == "AMBIGUOUS_DIFFERENCING" and "not both" in why


def test_a_window_whose_history_plus_differencing_exceeds_the_sealing_window_is_refused():
    assert check(point_of(window=197, lags=(1,), order=1))[0] == "WINDOW_PLUS_DIFFERENCING_EXCEEDS_SEALING_WINDOW"
    assert check(point_of(window=196, lags=(1,), order=1)) is None


def test_a_legal_point_validates_and_two_genomes_with_one_representation_are_one_point():
    point = point_of(window=101, lags=(1, 74), transform="log_return", order=0)
    assert check(point) is None
    spec = search.pipeline_spec_of(point, hand=HAND, stage="s", candidate_id="c", budget_sentence="B.")
    other = search.pipeline_spec_of(point, hand=HAND, stage="other", candidate_id="c", budget_sentence="B.")
    assert (search.identity(spec["representation"], feature_eng_src=FEATURE_ENG_SRC)
            == search.identity(other["representation"], feature_eng_src=FEATURE_ENG_SRC))


def test_the_searched_spec_changes_the_representation_and_nothing_else():
    point = point_of(window=101, lags=(1, 74), features=COLUMNS[:3])
    spec = search.pipeline_spec_of(point, hand=HAND, stage="searched_x", candidate_id="searched_x",
                                   budget_sentence="The budget is B.")
    assert spec["core"] == HAND["core"] and spec["extractors"] == HAND["extractors"]
    assert spec["preprocessing"] == HAND["preprocessing"] and spec["features"] == HAND["features"]
    assert spec["chosen_by"] == "SEARCH" and "The budget is B." in spec["why"]
    assert spec["representation"]["windows"] == [101] and spec["representation"]["lags"] == [1, 74]
    assert spec["grouping"]["groups"] == [{"group_id": "all", "members": COLUMNS[:3]}]
    # the target's own past is declared by the lags, never as an exogenous column
    assert "Global_active_power" not in spec["representation"]["exogenous"]


# ------------------------------------------------------------- the arithmetic a transform does to a prediction

def test_differencing_keeps_every_row_at_its_own_instant():
    values = np.array([[1.0], [3.0], [6.0], [10.0]])
    once = fitter.difference(values, 1)
    assert np.isnan(once[0, 0]) and once[1:, 0].tolist() == [2.0, 3.0, 4.0]
    twice = fitter.difference(values, 2)
    assert np.isnan(twice[:2, 0]).all() and twice[2:, 0].tolist() == [1.0, 1.0]
    assert fitter.difference(values, 0) is values


def test_every_transform_is_invertible_from_the_value_the_origin_carries():
    series = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    origins = np.array([0, 1, 2])
    for transform in fitter.TRANSFORMS:
        modelled = fitter.target_values(series, origins, horizon=2, transform=transform)
        back = fitter.invert_to_level(modelled, series[origins], transform=transform)
        assert np.allclose(back, series[origins + 2]), transform


def test_the_log_return_of_a_column_that_is_not_strictly_positive_is_refused_by_name():
    series = np.array([1.0, 0.0, 4.0, 8.0])
    with pytest.raises(fitter.SpecError, match=fitter.TRANSFORM_NOT_DEFINED):
        fitter.target_values(series, np.array([1]), horizon=2, transform="log_return")


def test_the_harness_says_which_reading_of_differencing_it_applied():
    assert "INPUT channels" in fitter.DIFFERENCING_READING
    assert "cannot be turned back into a level" in fitter.DIFFERENCING_READING


def test_the_ledger_counts_refusals_by_name(tmp_path):
    entries = {"a": {"representation_id": "a", "status": "OK", "mae": 0.5},
               "b": {"representation_id": "b", "status": "REFUSED", "refusal": search.LAG_EXCEEDS_WINDOW},
               "c": {"representation_id": "c", "status": "REFUSED", "refusal": search.LAG_EXCEEDS_WINDOW}}
    path = tmp_path / "ledger.json"
    search.write_ledger(path, entries=entries, header={"schema": search.LEDGER_SCHEMA})
    written = json.loads(path.read_text())
    assert written["evaluated"] == 1 and written["refused"] == 2
    assert written["refusals_by_name"] == {search.LAG_EXCEEDS_WINDOW: 2}
    assert set(search.load_ledger(path)) == {"a", "b", "c"}
