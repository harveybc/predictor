"""RP58: the transformation state comes from the CONTRACT, never from the shape of a prediction.

The defect, in `pipeline_plugins/stl_norm.py` as it stood. Whether a model's output was mapped back
to price space was decided by inspecting that output's own distribution:

  * mean/std branch — `_looks_normalized_like_standard_score` denormalised only when the array looked
    closer to N(0,1) than to N(mean, std);
  * min/max branch — if more than 95% of the values already fell inside [min, max], the array was
    assumed to be real and returned untouched;
  * `denormalize_returns` — if the spread was under 5% of the range, the deltas were assumed real.

So two models trained on the same normalised data, differing only in how confident (how spread out)
their predictions are, could have their errors reported in DIFFERENT UNITS — and a cautious model
that predicts near the mean is exactly the one most likely to be left normalised. An error table
built that way is not comparable across models, which is the whole point of a comparison.

The repair: when the configuration DECLARES the space its arrays are in, that declaration is obeyed
and nothing is inspected. When it does not, the historical heuristic still runs — old configs keep
working — but the guess is RECORDED, and `strict_normalization_contract` turns it into a refusal.
"""
import json

import numpy as np
import pytest

from pipeline_plugins import stl_norm


MINMAX = {"CLOSE": {"min": 1.0, "max": 1.2}}
MEANSTD = {"CLOSE": {"mean": 1.1, "std": 0.05}}


def _cfg(norm, **extra):
    return {"use_normalization_json": norm, "target_column": "CLOSE", **extra}


# --- the defect, kept visible -------------------------------------------------------------------

def test_RP58_without_a_contract_the_decision_still_depends_on_the_shape(monkeypatch):
    """PRE, preserved: two arrays in the SAME normalised space are treated differently.

    Both of these are normalised model outputs. The first is spread like a standard score. The
    second belongs to a model that predicts a nearly constant normalised value which happens to sit
    where real prices sit — nothing forbids that — and for that reason alone it is left in
    normalised units while the first is mapped to price space. Their error tables are then in
    different units, and the comparison between the two models is meaningless.
    """
    rng = np.random.default_rng(0)
    spread = rng.normal(0.0, 1.0, 512)                       # closer to N(0, 1)
    near_prices = rng.normal(1.1, 0.05, 512)                 # closer to N(mean, std) by coincidence
    out_a = stl_norm.denormalize(spread, _cfg(MEANSTD))
    first = stl_norm.last_decision()
    assert first["source"] == "HEURISTIC" and first["space"] == "NORMALIZED"
    assert first["inspected_values"] and "N(0, 1)" in first["why"]
    out_b = stl_norm.denormalize(near_prices, _cfg(MEANSTD))
    assert not np.allclose(out_a, spread)                    # one was mapped back to price space...
    assert np.allclose(out_b, near_prices)                   # ...and the other, equally normalised, was not
    assert stl_norm.last_decision()["space"] == "REAL" and stl_norm.last_decision()["inspected_values"]


def test_RP58_the_contract_settles_the_case_the_shape_got_wrong():
    """The same array of the rule above, with the space declared: it is mapped, as it should be."""
    rng = np.random.default_rng(0)
    rng.normal(0.0, 1.0, 512)
    near_prices = rng.normal(1.1, 0.05, 512)
    out = stl_norm.denormalize(near_prices, _cfg(MEANSTD, prediction_space="NORMALIZED"))
    assert np.allclose(out, near_prices*0.05+1.1)
    assert stl_norm.last_decision()["source"] == "CONTRACT"


def test_RP58_a_declared_space_is_obeyed_and_nothing_is_inspected():
    """The cautious array of the previous rule is denormalised when the contract says NORMALIZED."""
    rng = np.random.default_rng(0)
    cautious = rng.normal(0.0, 0.02, 512)
    out = stl_norm.denormalize(cautious, _cfg(MEANSTD, prediction_space="NORMALIZED"))
    assert np.allclose(out, cautious*0.05+1.1)
    decision = stl_norm.last_decision()
    assert decision["source"] == "CONTRACT" and decision["space"] == "NORMALIZED"
    assert decision["inspected_values"] is False


def test_RP58_a_declared_real_space_is_left_alone_however_it_looks():
    rng = np.random.default_rng(1)
    looks_normalised = rng.normal(0.0, 1.0, 512)          # would have been denormalised by the guess
    out = stl_norm.denormalize(looks_normalised, _cfg(MEANSTD, prediction_space="REAL"))
    assert np.array_equal(out, looks_normalised)
    assert stl_norm.last_decision()["source"] == "CONTRACT"


@pytest.mark.parametrize("norm", [MINMAX, MEANSTD])
def test_RP58_strict_mode_refuses_to_guess(norm):
    rng = np.random.default_rng(2)
    values = rng.normal(0.0, 1.0, 512)
    with pytest.raises(ValueError, match="does not declare"):
        stl_norm.denormalize(values, _cfg(norm, strict_normalization_contract=True))


def test_RP58_the_minmax_branch_obeys_the_contract_too():
    inside = np.linspace(1.05, 1.15, 512)          # inside [min, max]: the guess called this REAL
    out = stl_norm.denormalize(inside, _cfg(MINMAX, prediction_space="NORMALIZED"))
    assert np.allclose(out, inside*0.2+1.0)
    assert stl_norm.last_decision()["source"] == "CONTRACT"


def test_RP58_deltas_follow_the_same_contract():
    rng = np.random.default_rng(3)
    small = rng.normal(0.0, 0.001, 512)            # under 5% of the range: the guess called it REAL
    out = stl_norm.denormalize_returns(small, _cfg(MINMAX, prediction_space="NORMALIZED"))
    assert np.allclose(out, small*0.2)
    assert np.array_equal(stl_norm.denormalize_returns(small, _cfg(MINMAX, prediction_space="REAL")), small)


def test_RP58_the_existing_flag_keeps_working_and_is_recorded():
    values = np.linspace(-1, 1, 512)
    out = stl_norm.denormalize(values, _cfg(MEANSTD, targets_are_denormalized=True))
    assert np.array_equal(out, values)
    assert stl_norm.last_decision()["source"] == "CONTRACT"
    assert stl_norm.last_decision()["space"] == "REAL"


def test_RP58_an_undeclared_space_records_what_it_guessed_and_why():
    rng = np.random.default_rng(4)
    values = rng.normal(0.0, 1.0, 512)
    stl_norm.denormalize(values, _cfg(MEANSTD))
    d = stl_norm.last_decision()
    assert d["source"] == "HEURISTIC" and d["inspected_values"] is True
    assert d["space"] in ("NORMALIZED", "REAL")
    assert "declare" in d["how_to_remove_this_guess"]


def test_RP58_a_declared_space_must_be_one_of_the_two(monkeypatch):
    with pytest.raises(ValueError, match="prediction_space"):
        stl_norm.denormalize(np.zeros(512), _cfg(MEANSTD, prediction_space="probably normalized"))
