"""Acceptance tests for the paired objective/optimizer diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("huber_probe", Path(__file__).parents[1] / "tools/df_e1_huber.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def test_factorial_has_four_recipes_and_common_selection():
    assert len(H.RECIPES) == 4
    assert {(r["loss"], r["optimizer"]) for r in H.RECIPES.values()} == {
        (l, o) for l in ("mae", "huber") for o in ("adam", "adamw")}
    assert {r["monitor"] for r in H.RECIPES.values()} == {"val_mae"}


def test_huber_and_weight_decay_are_executed():
    tf = H.P._module("df_mod_e0")._tf()
    loss, opt = H.components(H.RECIPES["huber_adamw"], .003)
    assert isinstance(loss, tf.keras.losses.Huber)
    assert isinstance(opt, tf.keras.optimizers.AdamW)
    assert float(opt.weight_decay) == .004
    # Errors .5 and 2: Huber(delta=1) = .125 and 1.5, not MAE or MSE.
    np.testing.assert_allclose(loss([[0.], [0.]], [[.5], [2.]]).numpy(), .8125)
    w = tf.Variable([1.])
    opt.apply_gradients([(tf.zeros_like(w), w)])
    assert float(w.numpy()[0]) < 1.


def test_adam_has_no_decay():
    _, opt = H.components(H.RECIPES["mae_adam"], .003)
    assert opt.weight_decay is None


@pytest.mark.parametrize("field,value", [("loss", "mse"), ("optimizer", "sgd"), ("monitor", "val_loss")])
def test_unknown_recipe_refuses(field, value):
    recipe = {**H.RECIPES["huber_adamw"], field: value}
    with pytest.raises(ValueError):
        H.components(recipe, .003)


def test_raw_metrics_and_naive_use_identical_rows():
    result = H.metrics(np.array([1., 3.]), np.array([2., 2.]), np.array([0., 0.]), 2.)
    assert result["mae_kw"] == 1.
    assert result["mae_z"] == .5
    assert result["naive_mae_kw"] == 2.
    assert result["skill_vs_naive"] == .5


@pytest.mark.parametrize("prediction", [np.array([1.]), np.array([np.nan, 1.]), np.array([np.inf, 1.])])
def test_partial_or_nonfinite_score_refuses(prediction):
    with pytest.raises(ValueError):
        H.metrics(prediction, np.array([2., 2.]), np.array([0., 0.]), 1.)


def test_empty_score_refuses():
    with pytest.raises(ValueError):
        H.metrics(np.array([]), np.array([]), np.array([]), 1.)
