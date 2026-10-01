"""Lane H: the result table is generated from evidence and never rounds a spread into a claim."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rep = _load("h_kalman_report")
pipe = rep.pipe
test_pipe = _load("test_h_kalman_pipeline") if False else None


def fixture_results():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    tp = _load("test_h_kalman_pipeline")
    d = tp.synthetic_data(R=3000, n_tr=2000, seed=5)
    kal = pipe.build_kalman(d, tp.GROUPS, pipe.VARIANTS["moments_train"])
    arms = pipe.arm_matrices(d, kal, lags=1, controls=True)
    evs = {a: pipe.evaluate_arm(d, arms[a]) for a in ("A", "B", "C", "C_SMOOTHER_NONCAUSAL")}
    paired = {a: pipe.paired_against(d, evs["A"], evs[a], 20, B=200) for a in ("B", "C", "C_SMOOTHER_NONCAUSAL")}
    return {"variants": {"moments_train": {"arms": {"lags1": {"evaluations": pipe.public(evs), "paired_vs_A": paired}}}}}


def test_every_row_carries_the_model_error_beside_each_same_row_naive():
    rows = rep.table_rows(fixture_results())
    assert rows and {r["arm"] for r in rows} == {"A", "B", "C", "C_SMOOTHER_NONCAUSAL"}
    for r in rows:
        for key in ("variant", "lags", "arm", "horizon", "rows", "model_MAE", "model_MSE", "zero_return_MAE", "persistence_MAE",
                    "seasonal_MAE", "train_mean_MAE", "strict_naive", "skill_vs_zero_return_MAE", "eligible"):
            assert key in r, key
    smoother = [r for r in rows if r["arm"] == "C_SMOOTHER_NONCAUSAL"]
    assert all(r["eligible"] is False and r["reading"] == "NON_CAUSAL_REJECTION_CONTROL_NEVER_ELIGIBLE" for r in smoother)


def test_gap_is_called_a_difference_only_when_it_exceeds_the_ci_and_the_block_spread():
    assert rep.reading(delta=-0.01, ci=(-0.02, 0.0), quarters=[-0.03, 0.02, -0.01, 0.0]) == "WITHIN_SPREAD"
    assert rep.reading(delta=-0.01, ci=(-0.02, -0.005), quarters=[-0.011, -0.009, -0.01, -0.01]) == "EXCEEDS_SPREAD"
    assert rep.reading(delta=0.001, ci=(-0.001, 0.003), quarters=[0.0, 0.0, 0.0, 0.0]) == "WITHIN_SPREAD"


def test_markdown_never_uses_the_word_advantage_and_states_the_label():
    md = rep.to_markdown(rep.table_rows(fixture_results()), title="t")
    assert "advantage" not in md.lower() and "DEVELOPMENT" in md and "zero_return" in md.lower()
