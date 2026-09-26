"""Rules for the DR06 lag recomputation: the clock, the mask, and the two estimators named apart.

Every rule here is a mutant that must be REJECTED. Nothing in this file reads a run root, fits anything
or needs the household panel: the estimators are exercised on short constructed series whose answers are
known by hand, and the artifact rules are exercised on a minimal record.

    CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_dr06_lag_clock.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("_t_dr06", REPO / "tools" / "df_dr06_lag_clock.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _tool()


# --- the order that the two estimators must not share a word ------------------------------------------

def test_the_two_estimator_names_share_no_word():
    assert T._assert_names_share_no_word()["disjoint"] is True
    assert not (T._words(T.ACF) & T._words(T.PEARSON))


def test_a_name_that_shares_a_word_is_rejected(monkeypatch):
    monkeypatch.setattr(T, "PEARSON", "autocorrelation by Pearson over lag-truncated pairs")
    with pytest.raises(AssertionError):
        T._assert_names_share_no_word()


def test_neither_estimator_is_called_bias_corrected():
    for key in ("acf", "pearson"):
        assert "bias" not in T.ESTIMATORS[key]["name"].lower()


# --- the estimators reproduce the retained expressions on a gapless series -----------------------------

def test_on_a_series_with_no_gap_both_estimators_equal_the_retained_expressions():
    rng = np.random.default_rng(11)
    v = np.cumsum(rng.normal(size=400))
    finite = np.ones(v.size, dtype=bool)
    c = v - v.mean()
    denom = float((c * c).sum())
    for lag in (1, 7, 90):
        assert T.acf_at(v, finite, lag)["value"] == pytest.approx(
            float((c[:-lag] * c[lag:]).sum() / denom), abs=1e-14)
        assert T.pearson_at(v, finite, lag)["value"] == pytest.approx(
            float(np.corrcoef(v[:-lag], v[lag:])[0, 1]), abs=1e-14)


def test_the_finite_pair_mask_admits_exactly_the_pairs_with_two_finite_legs():
    v = np.arange(20.0)
    v[7] = np.nan
    finite = np.isfinite(v)
    # lag 1: pairs (6,7) and (7,8) are refused, so 19 - 2 = 17 are admitted
    assert T.pearson_at(v, finite, 1)["pairs"] == 17
    # lag 12: pairs start at 0..7, so exactly the pair whose LEFT leg is 7 is refused
    assert T.pearson_at(v, finite, 12)["pairs"] == (20 - 12) - 1
    # lag 15: no admitted pair has a leg at 7 at all, so nothing is refused
    assert T.pearson_at(v, finite, 15)["pairs"] == 20 - 15


# --- deleting a position is not the same thing as masking a pair ---------------------------------------

def test_deleting_a_position_moves_pairs_to_the_wrong_offset_and_the_mask_does_not():
    """A sawtooth with period 4 and one deleted position: the deleting route reads the wrong phase.

    On the intact clock, offset 4 is a full period and the correlation at it is +1. Delete one position
    and the compressed series is out of phase for every pair that straddles the deletion, so the
    deleting route cannot return +1 while the masked route must.
    """
    n, lag = 400, 200
    v = np.sin(2 * np.pi * np.arange(n) / 4.0)
    v[150] = np.nan
    finite = np.isfinite(v)
    masked = T.pearson_at(v, finite, lag)["value"]
    u = v[finite]
    deleted = T.pearson_at(u, np.ones(u.size, dtype=bool), lag)["value"]
    assert masked == pytest.approx(1.0, abs=1e-9), "a whole number of periods on the intact clock"
    assert abs(deleted - 1.0) > 0.5, "the deleting route reads 150 of its 199 pairs out of phase"
    assert T.straddling_pairs(int(u.size), [150], lag) == 150


def test_the_straddling_count_is_the_number_of_pairs_that_span_the_deletion():
    # 10 kept positions from an 11-position clock with index 5 removed; at offset 3 the pairs
    # starting at compressed 3, 4 and 5 span the removal
    assert T.straddling_pairs(10, [5], 3) == 3
    assert T.straddling_pairs(10, [5], 1) == 1
    # an offset that reaches past the end of the compressed series cannot straddle anything
    assert T.straddling_pairs(10, [5], 10) == 0


def test_with_no_deletion_at_all_the_two_routes_agree_exactly():
    rng = np.random.default_rng(3)
    v = np.cumsum(rng.normal(size=300))
    finite = np.ones(v.size, dtype=bool)
    for lag in (1, 5, 40):
        assert T.pearson_at(v, finite, lag)["value"] == pytest.approx(
            T.pearson_at(v[finite], finite, lag)["value"], abs=1e-15)


# --- the Pearson is not the rescaled autocorrelation function -----------------------------------------

def test_the_rescaled_autocorrelation_function_is_not_the_pearson():
    rng = np.random.default_rng(5)
    v = np.cumsum(rng.normal(size=2000))
    finite = np.ones(v.size, dtype=bool)
    lag = 700
    a = T.acf_at(v, finite, lag)["value"] * v.size / (v.size - lag)
    p = T.pearson_at(v, finite, lag)["value"]
    assert abs(a - p) > 1e-6, "if these ever coincided the 'bias-corrected' label would be defensible"


# --- the artifact rules -------------------------------------------------------------------------------

def _record() -> dict:
    return {
        "schema": "df_dr06_lag_clock.v1",
        "estimators": copy.deepcopy(T.ESTIMATORS),
        "estimator_names_share_no_word": {"disjoint": True},
        "populations": dict(T.POPULATIONS),
        "supersedes": copy.deepcopy(T.SUPERSEDES),
        "not_a_window_selector": dict(T.NOT_A_SELECTOR),
        "rows": [{"lag_minutes": k, **{f: 0.5 for f in T.REQUIRED_ROW_FIELDS}}
                 for k in (1, 60, 1440, 10080)],
        "day_versus_week_inversion": {"persists_on_the_corrected_clock": True},
        "checks": {"a": True},
        "state": "VERIFIED",
    }


def test_the_reference_record_is_admissible():
    assert T.validate(_record()) == []


@pytest.mark.parametrize("drop", [0, 1])
def test_a_record_that_drops_either_superseded_version_is_rejected(drop):
    r = _record()
    del r["supersedes"][drop]
    assert any("superseded" in m for m in T.validate(r))


def test_a_superseded_version_without_its_defect_named_is_rejected():
    r = _record()
    r["supersedes"][1]["defect"] = ""
    assert any("defect" in m for m in T.validate(r))


def test_a_record_that_labels_a_column_bias_corrected_is_rejected():
    r = _record()
    r["rows"][0]["bias_corrected"] = 0.5
    assert any("bias-corrected" in m for m in T.validate(r))


def test_the_withdrawn_label_may_still_be_cited_in_quotes_or_backticks():
    r = _record()
    blob = json.dumps(r)
    assert "'bias-corrected'" in blob and "`bias_corrected`" in blob
    assert T.validate(r) == []


def test_an_unquoted_citation_of_the_withdrawn_label_is_rejected():
    r = _record()
    r["supersedes"][1]["what"] = r["supersedes"][1]["what"].replace("`bias_corrected`",
                                                                   "bias_corrected")
    assert any("bias-corrected" in m for m in T.validate(r))


def test_a_record_that_loses_the_not_a_window_selector_note_is_rejected():
    for field in ("this_table_is", "this_table_is_not", "the_inversion_specifically"):
        r = _record()
        r["not_a_window_selector"].pop(field)
        assert any(field in m for m in T.validate(r))
    r = _record()
    r["not_a_window_selector"] = {}
    assert len(T.validate(r)) >= 3


def test_a_record_that_reads_the_table_as_a_window_choice_is_rejected():
    r = _record()
    r["day_versus_week_inversion"]["reading"] = "therefore use a weekly context window"
    assert any("window choice" in m for m in T.validate(r))
    r = _record()
    r["rows"][3]["note"] = "the optimal window follows from this row"
    assert any("window choice" in m for m in T.validate(r))


@pytest.mark.parametrize("field", list(T.REQUIRED_ROW_FIELDS))
def test_a_row_that_drops_any_required_column_is_rejected(field):
    r = _record()
    r["rows"][2].pop(field)
    assert any(field in m for m in T.validate(r))


def test_a_verified_record_may_not_carry_a_check_that_could_not_run():
    r = _record()
    r["checks"]["clock_is_a_gapless_minute_grid"] = None
    assert any("could not run" in m for m in T.validate(r))


def test_an_absent_run_root_refuses_instead_of_reprinting_the_published_values(tmp_path):
    rec = T.lag_table(tmp_path)
    assert rec["state"] == T.UNCHECKABLE
    assert "rows" not in rec
    assert "as_published_values" not in rec


# --- the clock check is a refusal when its bytes are absent -------------------------------------------

def test_the_clock_check_refuses_when_the_panel_is_absent(tmp_path):
    out = T.clock_check(tmp_path / "nothing.parquet", 0, 10)
    assert out["state"] == T.UNCHECKABLE
    assert "an_index_offset_of_k_is_k_minutes" not in out
