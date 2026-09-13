"""Regression tests for the governed run review blockers: extra flags never replace governed inputs or outputs,
and the config hash does not depend on where the governed config was written."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GR = _load("governed_run")


@pytest.mark.parametrize("flag", ["--x_train_file", "--y_test_file=other.csv", "--results_file", "--load_config",
                                  "--save_config", "--loss_plot_file", "--output_file"])
def test_an_extra_flag_that_replaces_a_governed_path_is_refused(flag):
    with pytest.raises(GR.GovernedRunError, match="refused"):
        GR.refuse_governed_overrides([flag, "x.csv"])


def test_tuning_flags_are_allowed():
    GR.refuse_governed_overrides(["--epochs", "2", "--max_steps_train", "300", "--mc_samples", "2"])


def test_config_hash_ignores_where_the_governed_config_lives():
    base = {"epochs": 2, "x_train_file": "/cache/a/1.csv", "results_file": "/out/a/results.csv"}
    ids = {"x_train_file": "gov:lab/phase_1/d4.csv@" + "a" * 64}
    one = GR.canonical_config(dict(base, load_config="/out/a/governed_config.json"), ids)
    two = GR.canonical_config(dict(base, load_config="/elsewhere/b/governed_config.json"), ids)
    assert one == two and "load_config" not in json.loads(one)
