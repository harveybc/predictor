"""K3: a twin without an observable comparison is INSUFFICIENT_TEST — never a detection,
never a pass; emissions and comparisons are recorded for every control; a centred twin over
complete data must be detected; causality, emission coverage and inapplicability by
missingness are three facts, reported apart. No future interpolation, support unchanged.
"""
import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


battery = _load("df_d3_acceptance")
ops = _load("df_d3_operators")

NAN = float("nan")


def values(n=600, seed=5):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) + math.sin(i / 7.0) for i in range(n)]


def with_missing(vals, kind, *, rate=0.10, seed=9):
    rng = random.Random(seed)
    out = list(vals)
    if kind == "isolated":
        out[300] = NAN
    elif kind == "blocks":
        for i in range(250, 290):
            out[i] = NAN
        for i in range(420, 445):
            out[i] = NAN
    elif kind == "mcar":
        for i in range(len(out)):
            if rng.random() < rate:
                out[i] = NAN
    return out


def report_for(kind, missing="none", n=600):
    op = ops.build(kind)
    x = battery.make_input(with_missing(values(n), missing))
    return battery.run_battery(op, x, train=battery.prefix(x, int(n * 0.6)), twin=ops.twin_of(op))


# --- the twin without observable comparisons -----------------------------------------------------

def test_wavelet_twin_under_mcar_is_insufficient_not_a_wrong_twin():
    """support 50 under 10 % MCAR: the twin emits almost nothing and nothing is compared."""
    rep = report_for("wavelet_trailing", "mcar")
    twin = rep["results"]["non_causal_twin"]
    assert twin["outcome"] == "INSUFFICIENT_TEST" and twin["passed"] is None
    assert twin["twin_comparisons"] == 0
    assert "absence of evidence" in twin["detail"]
    assert "non_causal_twin" in rep["undecided"] and rep["verdict"] != "MECHANICALLY_ACCEPTED"
    assert rep["verdict"] == "INCONCLUSIVE" or "non_causal_twin" not in rep["failed"]


def test_every_twin_control_records_emissions_and_comparisons():
    for kind in ("sax_paa_trailing", "stft_trailing", "wavelet_trailing", "butterworth_causal",
                 "cusum_causal", "mad_extremes_trailing", "variance_regime_trailing"):
        twin = report_for(kind)["results"]["non_causal_twin"]
        assert isinstance(twin["twin_emissions"], int) and twin["twin_emissions"] > 0, kind
        assert isinstance(twin["twin_comparisons"], int) and twin["twin_comparisons"] > 0, kind


def test_a_centred_twin_over_complete_data_is_detected():
    for kind in ("wavelet_trailing", "stft_trailing", "butterworth_causal", "sax_paa_trailing"):
        twin = report_for(kind)["results"]["non_causal_twin"]
        assert twin["passed"] is True and twin["outcome"] == "PASSED", kind
        assert twin["twin_prefix_passed"] is False or twin["twin_future_passed"] is False


class NeverFailingTwinOperator(ops.ButterworthCausal):
    """Declares a twin that is in truth causal (itself): compared many times, never failing."""

    def twin(self):
        return {"kind": "butterworth_filtfilt"}


class CausalPretender(ops.ButterworthCausal):
    KIND = "butterworth_filtfilt"

    def describe(self):
        return dict(super().describe(), non_causal_control_of="butterworth_causal")


def test_a_twin_compared_many_times_that_never_fails_is_a_failed_declaration():
    x = battery.make_input(values())
    out = battery.check_non_causal_twin(NeverFailingTwinOperator(), CausalPretender(),
                                        battery.prefix(x, 360), x)
    assert out["passed"] is False and out["outcome"] == "FAILED"
    assert out["twin_comparisons"] > 0 and "never failed" in out["detail"]


# --- causality, coverage and inapplicability, apart ------------------------------------------------

def test_the_report_carries_emission_coverage_apart_from_causality():
    rep = report_for("wavelet_trailing", "mcar")
    cov = rep["coverage"]
    assert set(cov) == {"n", "emitted", "inputs_available"}
    assert cov["inputs_available"] < cov["n"]
    assert cov["emitted"] < 0.05 * cov["n"]          # support 50 under 10 % MCAR
    # causality was not decided against it: nothing FAILED among the causal tests
    for t in ("prefix_all_available", "future_perturbation"):
        assert rep["results"][t]["passed"] is not False
    assert rep["results"]["applicability"]["passed"] is True   # family declaration, apart


@pytest.mark.parametrize("missing", ["isolated", "blocks"])
def test_isolated_and_block_gaps_reduce_coverage_by_the_support_and_keep_causality(missing):
    rep = report_for("wavelet_trailing", missing)
    full = report_for("wavelet_trailing", "none")
    assert rep["coverage"]["emitted"] < full["coverage"]["emitted"]
    lost = full["coverage"]["emitted"] - rep["coverage"]["emitted"]
    support = ops.build("wavelet_trailing").support()["samples"]
    if missing == "isolated":
        assert lost == support                        # one NaN blanks exactly one support
    for t in ("prefix_all_available", "future_perturbation", "chunk_restart"):
        assert rep["results"][t]["passed"] is True, (t, rep["results"][t])
    assert rep["results"]["non_causal_twin"]["passed"] is True


def test_the_support_boundary_emits_exactly_one_output_and_no_future_interpolation():
    op = ops.build("wavelet_trailing")
    support = op.support()["samples"]
    x = battery.make_input(values(support))
    out = op.transform(x, op.fit(battery.prefix(x, support - 1)))
    assert sum(out["available"]) == 1 and out["available"][-1] is True
    y = battery.make_input(with_missing(values(support), "isolated") if False else
                           values(support)[:-1] + [NAN])
    out2 = op.transform(y, op.fit(battery.prefix(y, support - 1)))
    assert sum(out2["available"]) == 0                # a NaN inside the support is never filled


def test_restart_under_missingness_is_consistent_or_insufficient_never_wrong():
    for missing in ("isolated", "blocks", "mcar"):
        rep = report_for("wavelet_trailing", missing)
        r = rep["results"]["chunk_restart"]
        assert r["passed"] is True or r.get("outcome") == "INSUFFICIENT_TEST", (missing, r)
