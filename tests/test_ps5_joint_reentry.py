"""PS5 (M01 half of FS16): joint re-entry of a pair with REFIT of the temporal model, paired seeds.

For a fold, a base set and a candidate pair (a, b) with its target, four arms are refitted from scratch
with the same seeds: base, base+a, base+b, base+a+b. The pair RE-ENTERS only if base+a+b is the strict
minimum of the four arms' mean validation loss (no tolerance). Only rows of the fold (TRAIN) are read.
Synthetic mechanics; labelled DEVELOPMENT.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

from tools import ps5_joint_reentry as ps5

SETTINGS = {"max_epochs": 25, "patience": 5, "batch_size": 32, "learning_rate": 3e-3, "weight_decay": 0.0,
            "loss": "mae", "min_delta": 0.0, "max_updates": 5000}
SMALL_CORE = {"d_model": 12, "heads": 2, "blocks": 1, "ff_dim": 16, "stage_channels": [8, 6, 4]}


def synthetic(n=900, seed=0):
    """y(t+1) = a(t) + b(t) + noise: each member helps, only both together reach the floor; c, d are noise.

    (A pure product a*b was tried first: 420 training rows do not teach the forecaster the interaction
    in a test-sized budget, so the mechanics test uses an additive pair; the joint arm is still required
    to beat both single-member arms strictly.)"""
    rng = np.random.default_rng(seed)
    cols = {k: rng.normal(size=n) for k in ("base0", "a", "b", "c", "d")}
    target = np.full(n, np.nan)
    target[:-1] = cols["a"][:-1] + cols["b"][:-1] + 0.1 * rng.normal(size=n - 1)
    return cols, target


FOLD = {"name": "inner_1", "train": (0, 600), "val": (630, 900)}


def run(cols, target, pair, seeds=(1, 2)):
    return ps5.evaluate_pair(cols, target, FOLD, base=["base0"], pair=pair, seeds=seeds, settings=SETTINGS,
                             window=24, sample_hours=4, core_params=SMALL_CORE, output_steps=6,
                             output_channels=4)


def test_jointly_useful_pair_re_enters_and_noise_pair_gains_less():
    cols, target = synthetic()
    true = run(cols, target, ("a", "b"))
    noise = run(cols, target, ("c", "d"))
    assert true["decision"] == "RE_ENTERS"
    assert true["arms"]["base+a+b"]["mean_val_mae"] < min(true["arms"][k]["mean_val_mae"]
                                                          for k in ("base", "base+a", "base+b"))
    gain_true = true["arms"]["base"]["mean_val_mae"] - true["arms"]["base+a+b"]["mean_val_mae"]
    gain_noise = noise["arms"]["base"]["mean_val_mae"] - noise["arms"]["base+c+d"]["mean_val_mae"]
    assert gain_true > gain_noise
    assert true["label"] == "DEVELOPMENT" and true["naive"]["kind"] == "zero_return"
    for arm in true["arms"].values():
        assert set(arm["per_seed"]) == {"1", "2"} and all(s["updates"] > 0 for s in arm["per_seed"].values())


def test_rows_after_the_fold_are_never_read():
    cols, target = synthetic()
    a = run(cols, target, ("a", "b"), seeds=(1,))
    cols2 = {k: v.copy() for k, v in cols.items()}
    target2 = target.copy()
    for v in cols2.values():
        v[FOLD["val"][1]:] = 1e6                        # beyond the fold: poison
    target2[FOLD["val"][1]:] = 1e6
    cols2 = {k: np.concatenate([v, np.full(50, 1e6)]) for k, v in cols2.items()}
    target2 = np.concatenate([target2, np.full(50, 1e6)])
    b = run(cols2, target2, ("a", "b"), seeds=(1,))
    for arm in a["arms"]:
        assert a["arms"][arm]["mean_val_mae"] == b["arms"][arm]["mean_val_mae"]


def test_decision_rule_is_strict_minimum():
    arms = {"base": 1.0, "base+a": 0.9, "base+b": 0.95, "base+a+b": 0.9}
    assert ps5.decide({k: {"mean_val_mae": v} for k, v in arms.items()}, ("a", "b")) == "DOES_NOT_RE_ENTER"
    arms["base+a+b"] = 0.8999999
    assert ps5.decide({k: {"mean_val_mae": v} for k, v in arms.items()}, ("a", "b")) == "RE_ENTERS"


def test_refuses_a_target_label_outside_the_fold():
    cols, target = synthetic()
    with pytest.raises(ValueError, match="fold"):
        ps5.evaluate_pair(cols, target, {"name": "x", "train": (0, 600), "val": (590, 900)}, base=["base0"],
                          pair=("a", "b"), seeds=(1,), settings=SETTINGS, window=24, sample_hours=4,
                          core_params=SMALL_CORE, output_steps=6, output_channels=4)
