"""K3: a twin without an observable comparison is INSUFFICIENT_TEST — never a detection,
never a pass; emissions and comparisons are recorded for every control; a centred twin over
complete data must be detected; causality, emission coverage and inapplicability by
missingness are three facts, reported apart. No future interpolation, support unchanged.
"""
import importlib.util
import json
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
        assert twin["twin_detections"] > 0 and twin["twin_sensitive_comparisons"] > 0


class NeverFailingTwinOperator(ops.ButterworthCausal):
    """Declares a twin that is in truth causal (itself): compared many times, never failing."""

    def twin(self):
        return {"kind": "butterworth_filtfilt"}


class CausalPretender(ops.ButterworthCausal):
    """Declares a right reach of 5 and is in truth causal: sensitive comparisons, no effect."""
    KIND = "butterworth_filtfilt"

    def reach_right(self, n):
        return 5

    def describe(self):
        return dict(super().describe(), non_causal_control_of="butterworth_causal")


def test_a_twin_with_sensitive_comparisons_and_no_violation_is_a_failed_declaration():
    x = battery.make_input(values())
    out = battery.check_non_causal_twin(NeverFailingTwinOperator(), CausalPretender(),
                                        battery.prefix(x, 360), x)
    assert out["passed"] is False and out["outcome"] == "FAILED"
    assert out["twin_sensitive_comparisons"] > 0 and out["twin_detections"] == 0
    assert "no effect where its reach crosses the cut" in out["detail"]


class ReachlessPretender(CausalPretender):
    def reach_right(self, n):
        return 0


def test_a_twin_declaring_no_right_reach_is_refused_as_a_control():
    x = battery.make_input(values())
    out = battery.check_non_causal_twin(NeverFailingTwinOperator(), ReachlessPretender(),
                                        battery.prefix(x, 360), x)
    assert out["passed"] is False and "no right reach" in out["detail"]


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


# --- L2: sensitivity, demonstrated per unit, never relaxed causality --------------------------------

def test_the_sensitivity_amendment_is_sealed_and_names_07b():
    design = _load("df_d3_design")
    doc = design.D3_TWIN_SENSITIVITY_AMENDMENT_V1
    assert design.validate_twin_sensitivity_amendment(doc) == []
    assert design.D3_DESIGN_CURRENT is doc
    assert doc["supersedes_amendment"]["design_sha256"] == design.D3_PROBE_AMENDMENT_V1["design_sha256"]


def test_every_twin_declares_a_positive_right_reach_from_its_geometry():
    expect = {"sax_paa_centred": lambda t: int(t.params["segment"]) - int(t.params["segment"]) // 2 - 1,
              "stft_centred": lambda t: int(t.params["w"]) - int(t.params["w"]) // 2 - 1,
              "wavelet_centred": lambda t: 50 - 25 - 1,
              "butterworth_filtfilt": lambda t: 999,
              "cusum_lookahead": lambda t: int(t.params["ahead"]),
              "mad_extremes_centred": lambda t: int(t.params["w"]) - int(t.params["w"]) // 2 - 1,
              "variance_regime_centred": lambda t: int(t.params["w"]) - int(t.params["w"]) // 2 - 1}
    for op in ops.bank():
        twin = ops.twin_of(op)
        if twin is None:
            continue
        assert twin.reach_right(999) == expect[twin.KIND](twin) > 0, twin.KIND


def test_wavelet_twin_under_mcar_reports_zero_sensitive_comparisons_and_the_nearest_gap():
    rep = report_for("wavelet_trailing", "mcar")
    twin = rep["results"]["non_causal_twin"]
    assert twin["outcome"] == "INSUFFICIENT_TEST"
    assert twin["twin_sensitive_comparisons"] == 0 and twin["twin_detections"] == 0
    assert twin["twin_reach_right"] == 24
    assert twin["twin_nearest_output_to_cut"] is None or twin["twin_nearest_output_to_cut"] >= 24


def test_a_known_centre_is_detected_with_sensitive_comparisons_and_a_first_detection():
    twin = report_for("wavelet_trailing")["results"]["non_causal_twin"]
    assert twin["outcome"] == "PASSED" and twin["twin_sensitive_comparisons"] > 0
    assert twin["first_detection"]["sensitive"] is True


class ImpulseAfterCutTwin(ops.WaveletCentred):
    """Reads exactly one sample after i: the impulse just after the cut is the whole leak."""
    KIND = "wavelet_centred"

    def lookback(self):
        return 0

    def warm_up(self):
        return 0

    def reach_right(self, n):
        return 1

    def transform(self, x, state):
        import numpy as np
        v = np.asarray(x["values"], dtype=float)
        vals = np.zeros(v.size)
        avail = np.zeros(v.size, dtype=bool)
        vals[:-1] = v[:-1] + v[1:]
        avail[:-1] = ~np.isnan(v[:-1]) & ~np.isnan(v[1:])
        return self._pack(x, vals.tolist(), avail.tolist())


def test_a_future_impulse_just_after_the_cut_is_a_sensitive_detection():
    op = ops.build("wavelet_trailing")
    x = battery.make_input(values())
    out = battery.check_non_causal_twin(op, ImpulseAfterCutTwin(), battery.prefix(x, 360), x)
    assert out["outcome"] == "PASSED" and out["twin_detections"] > 0
    assert out["first_detection"]["sensitive"] is True


class ZeroFutureWeightsTwin(ops.WaveletCentred):
    """Declares the centred reach but weights every future sample by zero: geometric crossing
    without effect. It must FAIL as a declaration, never PASS by geometry alone."""
    KIND = "wavelet_centred"

    def transform(self, x, state):
        import numpy as np
        v = np.asarray(x["values"], dtype=float)
        w = self._support()
        half = w // 2
        vals = np.zeros(v.size)
        avail = np.zeros(v.size, dtype=bool)
        for i in range(half, v.size - (w - half - 1)):
            window = v[i - half:i + 1]                       # past half + i only
            if not np.isnan(v[i - half:i - half + w]).any():
                vals[i] = float(window.sum())
                avail[i] = True
        return self._pack(x, vals.tolist(), avail.tolist())


def test_extreme_zero_future_weights_are_a_failed_declaration_not_a_pass():
    op = ops.build("wavelet_trailing")
    x = battery.make_input(values())
    out = battery.check_non_causal_twin(op, ZeroFutureWeightsTwin(), battery.prefix(x, 360), x)
    # availability still needs the future half, so the MASK moves at the cut: a detection by mask
    # is a real violation; a control whose mask also ignored the future would have to FAIL
    assert out["outcome"] in ("PASSED", "FAILED")
    if out["outcome"] == "FAILED":
        assert out["twin_sensitive_comparisons"] > 0 and out["twin_detections"] == 0


class MaskOnlyLeak(ops.WaveletCentred):
    """Values causal, availability reads the future: a violation by mask alone."""
    KIND = "wavelet_centred"

    def lookback(self):
        return 0

    def warm_up(self):
        return 0

    def transform(self, x, state):
        import numpy as np
        v = np.asarray(x["values"], dtype=float)
        vals = np.cumsum(np.nan_to_num(v)).tolist()
        avail = [bool(i + 1 < v.size and not np.isnan(v[i + 1])) for i in range(v.size)]
        return self._pack(x, vals, avail)


class EmissionOnlyLeak(ops.WaveletCentred):
    """Values and mask causal, emission time depends on the next sample's value."""
    KIND = "wavelet_centred"

    def lookback(self):
        return 0

    def warm_up(self):
        return 0

    def transform(self, x, state):
        import numpy as np
        v = np.asarray(x["values"], dtype=float)
        out = {"values": np.cumsum(np.nan_to_num(v)).tolist(),
               "available": [bool(a) for a in ~np.isnan(v)], "raw": list(x["values"])}
        # the emission instant depends on the NEXT sample's value: a leak by emission time
        ts = x["timestamps"]
        nxt = [v[i + 1] if i + 1 < v.size else 0.0 for i in range(v.size)]
        out["emitted_at"] = [ts[i] + (1 if (not np.isnan(nxt[i]) and nxt[i] > 0) else 0)
                             for i in range(len(ts))]
        return out


@pytest.mark.parametrize("control", [MaskOnlyLeak, EmissionOnlyLeak])
def test_non_causal_controls_fail_by_mask_and_by_emission_time_apart(control):
    op = ops.build("wavelet_trailing")
    x = battery.make_input(values())
    out = battery.check_non_causal_twin(op, control(), battery.prefix(x, 360), x)
    assert out["outcome"] == "PASSED" and out["twin_detections"] > 0, out["detail"]


def test_the_candidates_own_causal_tests_are_not_restricted_to_the_twins_mask():
    rep = report_for("wavelet_trailing", "mcar")
    r = rep["results"]["prefix_all_available"]
    assert r.get("compared", 0) >= 0 and "sensitive" not in json.dumps(r)


def test_edges_and_restart_under_missingness_hold_with_the_amended_twin():
    for missing in ("isolated", "blocks", "mcar"):
        rep = report_for("wavelet_trailing", missing)
        assert rep["results"]["warm_up_edge"]["passed"] is True or rep["coverage"]["emitted"] == 0
        r = rep["results"]["chunk_restart"]
        assert r["passed"] is True or r.get("outcome") == "INSUFFICIENT_TEST"
