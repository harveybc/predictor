"""The quantile head (WP07): what it refuses, and the two properties the graph must have by construction.

The reason an interval from this plugin can be published at all is that its bounds are fitted quantiles and that they
cannot cross. Both are properties of the graph, not of the training run, so both are testable here without fitting
anything: the head is built from `Dense`, `Add` and `Concatenate` with non-negative increments, so a set of untrained
weights already produces ordered outputs. The pinball loss is checked against values worked out by hand.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

tf = pytest.importorskip("tensorflow")

from predictor_plugins.quantile_ann import (NOT_MULTI_BRANCH, QUANTILE_OUT_OF_RANGE,          # noqa: E402
                                            QUANTILES_NOT_ASCENDING, TOO_FEW_QUANTILES,
                                            Plugin, declared_quantiles, pinball_loss)

QUANTILES = [0.05, 0.5, 0.95]


def built(**config):
    params = {"predicted_horizons": [60], "plotted_horizon": 60, "quiet": True}
    params.update(config)
    plugin = Plugin(params)
    plugin.build_model((12, 4), None, {})
    return plugin


# ------------------------------------------------------------------ the declaration


@pytest.mark.parametrize("quantiles, refusal", [
    ([0.5], TOO_FEW_QUANTILES),
    ([], TOO_FEW_QUANTILES),
    ([0.0, 0.5], QUANTILE_OUT_OF_RANGE),
    ([0.5, 1.0], QUANTILE_OUT_OF_RANGE),
    ([0.95, 0.05], QUANTILES_NOT_ASCENDING),
    ([0.5, 0.5], QUANTILES_NOT_ASCENDING),
])
def test_a_quantile_set_that_is_not_one_is_refused_by_name(quantiles, refusal):
    with pytest.raises(ValueError, match=refusal):
        declared_quantiles(quantiles)


def test_the_declared_set_is_returned_unchanged_and_not_reordered():
    assert declared_quantiles(QUANTILES) == QUANTILES


def test_a_grouped_input_is_refused_instead_of_using_the_first_branch():
    plugin = Plugin({"predicted_horizons": [60], "plotted_horizon": 60, "quiet": True})
    with pytest.raises(ValueError, match=NOT_MULTI_BRANCH):
        plugin.build_model([(12, 2), (12, 2)], None, {})


def test_the_plugin_is_a_declared_entry_point():
    declared = (ROOT / "setup.py").read_text()
    assert "'quantile_ann=predictor_plugins.quantile_ann:Plugin'" in declared


# ------------------------------------------------------------------ the graph


def test_the_head_emits_one_value_per_declared_quantile_per_horizon():
    plugin = Plugin({"predicted_horizons": [30, 60], "plotted_horizon": 60, "quiet": True})
    plugin.build_model((12, 4), None, {})
    assert plugin.output_names == ["output_horizon_30", "output_horizon_60"]
    assert [tuple(shape) for shape in plugin.model.output_shape] == [(None, 3), (None, 3)]


def test_the_quantiles_cannot_cross_even_with_untrained_weights():
    plugin = built(quantiles=[0.1, 0.5, 0.9])
    x = np.random.default_rng(7).normal(size=(64, 12, 4)).astype("float32")
    out = np.asarray(plugin.model.predict(x, verbose=0))
    assert out.shape == (64, 3)
    # every increment is a softplus, so it is non-negative for ANY weights: the ordering is a property of the graph
    assert (np.diff(out, axis=1) >= 0).all()


def test_the_loss_is_the_pinball_loss_of_the_declared_levels():
    loss = pinball_loss([0.1, 0.5, 0.9])
    y_true = tf.constant([[1.0]])
    y_pred = tf.constant([[0.0, 1.0, 2.0]])          # under by 1, exact, over by 1
    # under-forecast at q=0.1 costs 0.1 * 1; exact costs 0; over-forecast at q=0.9 costs (1 - 0.9) * 1
    assert float(loss(y_true, y_pred)[0]) == pytest.approx((0.1 + 0.0 + 0.1) / 3)


def test_the_loss_charges_an_over_forecast_more_at_a_low_quantile():
    loss = pinball_loss([0.1])
    over = float(loss(tf.constant([[0.0]]), tf.constant([[1.0]]))[0])
    under = float(loss(tf.constant([[1.0]]), tf.constant([[0.0]]))[0])
    assert over == pytest.approx(0.9) and under == pytest.approx(0.1) and over > under


def test_the_saved_graph_reloads_without_this_module_s_custom_objects(tmp_path):
    """Serving must not need the training package: the exporter loads the graph with `compile=False` in its own venv."""
    plugin = built()
    path = tmp_path / "model.keras"
    plugin.model.save(path)
    import keras

    reloaded = keras.saving.load_model(str(path), compile=False)
    x = np.zeros((1, 12, 4), dtype="float32")
    assert np.allclose(np.asarray(reloaded.predict(x, verbose=0)),
                       np.asarray(plugin.model.predict(x, verbose=0)))
