"""Q3: controls of known utility — each bound to a requirement, an estimand, a generator with a
justified advantage (or none), a loss and a predeclared criterion; positive controls and
calibrations are distinct roles; a budget exhausted gives a partial, never a verdict."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("df_utility_controls")
H = _load("df_utility_harness")


def test_Q3_every_control_declares_requirement_estimand_generator_pair_and_criterion():
    for name, spec in C.CONTROLS.items():
        assert spec["requirement"] and spec["estimand"] and spec["generator"] and spec["pair"], name
        assert spec["expected"]["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE, H.REFUSED)
        assert 0 < spec["expected"]["min_fraction"] <= 1
    assert C.CONTROLS["positive_H_A"]["pair"] == ("raw_wide", "augmented")
    assert C.CONTROLS["info_loss"]["expected"]["delta_sign"] == -1
    assert C.CONTROLS["future_leak"]["expected"]["why_contains"] == "not causal"


def test_Q3_the_generators_carry_the_advantage_they_claim():
    rng = np.random.default_rng(1)
    x = C.generate("mad_extremeness_drift", 3000, rng)
    z = np.array([C._z(x, t) for t in range(C.W + 1, 3000)])
    inc = np.diff(x)[C.W:]
    assert np.corrcoef(np.minimum(3.0, z), inc)[0, 1] > 0.3         # the unsigned score drives the increment
    m = C.generate("signed_momentum", 3000, np.random.default_rng(2))
    d = np.diff(m)
    assert np.corrcoef(d[:-1], d[1:])[0, 1] > 0.3                    # signed momentum
    zz = np.array([C._z(m, t) for t in range(C.W + 1, 3000)])
    assert abs(np.corrcoef(zz, np.diff(m)[C.W:])[0, 1]) < 0.15       # the unsigned score does not carry it
    w = C.generate("white_null", 3000, np.random.default_rng(3))
    dw = np.diff(w)
    assert abs(np.corrcoef(dw[:-1], dw[1:])[0, 1]) < 0.1


def test_Q3_a_small_run_has_the_declared_structure_and_the_leak_is_refused_before_scoring():
    out = C.run_controls(n=600, replicates=2, seed0=5, budget_cpu_seconds=600.0)
    assert set(out["controls"]) == set(C.CONTROLS) and out["stopped"] is None
    assert set(out["calibrations"]) == {"raw/transformed", "raw_wide/augmented"}
    leak = out["controls"]["future_leak"]
    assert leak["hits"] == 2 and all(r["outcome"] == H.REFUSED and "not causal" in r["why"] for r in leak["rows"])
    for name in ("positive_H_T", "null_contrast", "info_loss"):
        assert all(r["delta_mean"] is not None for r in out["controls"][name]["rows"]), name
    assert "NOT a positive control" in out["calibration_role"]
    assert out["instrument_verdict"] in ("SEPARATES", "DOES_NOT_SEPARATE_OR_INCOMPLETE")


def test_Q3_an_exhausted_budget_stops_and_reports_a_partial_count_never_a_verdict():
    out = C.run_controls(n=600, replicates=3, seed0=5, budget_cpu_seconds=0.0)
    assert out["stopped"] and out["stopped"].startswith("BUDGET_EXHAUSTED")
    assert out["instrument_verdict"] == "DOES_NOT_SEPARATE_OR_INCOMPLETE"
    first = out["controls"][next(iter(out["controls"]))]
    assert first["completed"] < first["of"] and first["met"] is False
