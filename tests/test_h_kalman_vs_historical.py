"""Lane H: declared equivalence between the closed-form successor and the historical MLE local_level_kalman.

The historical operator (tools/df_operators.py, private numeric kernel) is used ONLY as a reference here, never for a
governing decision. Declared regime of agreement: a local level series with a moderate noise ratio and a long sample."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")
cmp_ = _load("h_kalman_vs_historical")


def test_agreement_in_the_declared_regime_and_documented_disagreement_outside_it():
    rep = cmp_.compare(T=20000, q=0.05, r=0.5, seed=3)
    assert rep["historical"]["status"] == "FITTED"
    assert abs(rep["successor"]["ratio"] - rep["historical"]["ratio"]) / rep["historical"]["ratio"] < 0.35
    assert rep["max_abs_level_difference_over_obs_std"] < 0.05
    # outside the regime: a nearly pure random walk. The historical MLE abstains or sits at its bound; the successor
    # always returns typed, floor-clipped parameters. The two are NOT interchangeable there.
    rw = cmp_.compare(T=5000, q=1.0, r=1e-4, seed=4)
    assert rw["successor"]["status"] == "FITTED"
    assert rw["historical"]["status"] in ("ABSTAIN", "FITTED")
    assert "declared_regime" in rw and rw["declared_regime"] is False
