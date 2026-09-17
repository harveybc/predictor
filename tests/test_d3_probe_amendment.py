"""K2: the response probe is built from the training fit and the operator's declared
resolution; three facts are recorded apart; abstention exists only by declaration and costs
the verdict; a real delay never becomes a pass.

Every rule runs the real battery on the real operators or on a deliberately wrong control.
Nothing here tunes an operator: parameters are the declared defaults throughout.
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


contract = _load("df_d3_contract")
battery = _load("df_d3_acceptance")
design = _load("df_d3_design")
ops = _load("df_d3_operators")


def series(n=400, *, offset=0.0, scale=1.0, seed=3):
    rng = random.Random(seed)
    return battery.make_input([offset + scale * rng.gauss(0.0, 1.0) for _ in range(n)])


def probe_of(operator, x, train_n=240):
    train = battery.prefix(x, train_n)
    return battery.check_response_probe(operator, train, x)


# --- the amendment is sealed and names what it supersedes ------------------------------------

def test_the_successor_amendment_is_sealed_before_any_measurement():
    doc = design.D3_PROBE_AMENDMENT_V1
    assert design.validate_probe_amendment(doc) == []
    assert doc["supersedes_amendment"]["design_sha256"] == design.D3_AMENDMENT_V1["design_sha256"]
    current = design.D3_DESIGN_CURRENT
    assert current is doc or current["supersedes_amendment"]["design_sha256"] == doc["design_sha256"]
    assert doc["design_sha256"] != design.D3_AMENDMENT_V1["design_sha256"]
    assert "amplitudes searched until a pass" in doc["probe_construction"]["never"]


def test_a_declaration_without_the_train_fit_scale_is_refused():
    op = ops.build("uniform_decile_quantizer")
    spec = op.describe()
    assert spec["schema"] == "d3_operator_spec.v3"
    assert spec["response_probe"]["scale"] == contract.PROBE_SCALE
    bad = dict(spec, response_probe={"kind": "step", "expected_onset_samples": 0})
    with pytest.raises(contract.SpecRefusal, match="scale"):
        contract.validate_spec(bad)


# --- known-domain quantizer at several scales ---------------------------------------------------

@pytest.mark.parametrize("offset,scale", [(0.0, 1.0), (5000.0, 300.0), (-2.0e6, 1.0e5),
                                          (0.0, 1.0e-9), (0.0, 1.0e9)])
def test_the_decile_quantizer_shows_onset_zero_on_its_own_fitted_domain(offset, scale):
    """v1's fixed +25 step on N(0,1) noise crossed no fitted edge at the impact sample on a
    train in the thousands and crossed one a sample later by luck: an artefact of the
    fixture's scale. Built from the fit, the excitation crosses an edge at the impact."""
    op = ops.build("uniform_decile_quantizer")
    out = probe_of(op, series(offset=offset, scale=scale))
    assert out["identifiable"] is True
    assert out["first_change_observed"] == 0 and out["matches_declared"] is True
    assert out["passed"] is True
    ex = out["excitation"]
    assert ex["baseline"] == pytest.approx(offset - 1.28 * scale, abs=0.4 * scale)
    assert 0 < ex["amplitude"] < ex["scale"]


def test_sax_shows_onset_zero_on_its_own_fitted_domain_at_two_scales():
    for offset, scale in ((0.0, 1.0), (10000.0, 50.0)):
        out = probe_of(ops.build("sax_paa_trailing"), series(offset=offset, scale=scale))
        assert out["identifiable"] is True and out["first_change_observed"] == 0, out
        assert out["passed"] is True


# --- the three facts, apart ---------------------------------------------------------------------

def test_the_probe_records_identifiability_first_change_and_agreement_apart():
    out = probe_of(ops.build("butterworth_causal"), series())
    assert set(("identifiable", "first_change_observed", "matches_declared", "excitation",
                "declared", "emitted_after_impact")) <= set(out)
    assert out["excitation"]["kind"] == "impulse"
    assert out["excitation"]["reason"]


def test_stft_declares_onset_one_and_the_probe_measures_one():
    out = probe_of(ops.build("stft_trailing"), series())
    assert out["declared"] == 1 and out["first_change_observed"] == 1
    assert out["matches_declared"] is True and out["passed"] is True


# --- abstention only by declaration, and it costs the verdict ---------------------------------

def test_a_constant_training_fit_is_unidentified_never_a_pass():
    x = battery.make_input([3.0] * 200 + [3.0 + 0.001 * i for i in range(200)])
    out = probe_of(ops.build("butterworth_causal"), x, train_n=200)
    assert out["passed"] is None and out["outcome"] == "UNIDENTIFIED"
    assert out["identifiable"] is False and "constant" in out["detail"]


def test_saturation_is_unidentified_from_the_fit_with_its_reason():
    op = ops.build("uniform_decile_quantizer")
    state = {"edges": [-5.0, -4.0, -3.0]}       # every edge below the baseline's noise band
    res = op.probe_resolution(state, baseline=0.0, scale=1.0, sigma=0.01)
    assert res["amplitude"] == contract.UNIDENTIFIED and "saturation" in res["reason"]


class Unidentifying(ops.ButterworthCausal):
    """An operator that declares nothing identifiable: it may abstain, and it pays for it."""

    def probe_resolution(self, state, *, baseline, scale, sigma):
        return {"amplitude": contract.UNIDENTIFIED, "reason": "declared by the control"}


def test_a_declared_unidentified_probe_makes_the_verdict_inconclusive_not_accepted():
    op = Unidentifying()
    x = series()
    report = battery.run_battery(op, x, train=battery.prefix(x, 240),
                                 twin=ops.twin_of(op))
    assert report["results"]["response_probe"]["outcome"] == "UNIDENTIFIED"
    assert "response_probe" in report["undecided"]
    assert report["verdict"] == "INCONCLUSIVE" and report["review_ready"] is False


# --- controls the rule must NOT rescue ------------------------------------------------------------

class DelayedQuantizer(ops.UniformDecileQuantizer):
    """A quantizer whose output for t is the level of x[t-1], declaring onset 0: a real delay
    incompatible with its contract. It must keep failing under the amended probe."""

    KIND = "delayed_quantizer_control"

    def transform(self, x, state):
        out = super().transform(x, state)
        values = [0.0] + out["values"][:-1]
        available = [False] + out["available"][:-1]
        return {**out, "values": values, "available": available}

    def lookback(self):
        return 1


def test_a_deliberately_delayed_operator_keeps_failing_with_the_delay_it_shows():
    out = probe_of(DelayedQuantizer(), series())
    assert out["identifiable"] is True
    assert out["first_change_observed"] == 1 and out["declared"] == 0
    assert out["matches_declared"] is False and out["passed"] is False
    assert "declares 0" in out["detail"]


class Contradicting(ops.ButterworthCausal):
    """Declares an identifiable excitation and never moves: the declaration is contradicted."""

    def transform(self, x, state):
        out = super().transform(x, state)
        return {**out, "values": [0.0 for _ in out["values"]]}


def test_a_declared_identifiable_excitation_that_moves_nothing_fails_not_abstains():
    out = probe_of(Contradicting(), series())
    assert out["identifiable"] is True and out["first_change_observed"] is None
    assert out["passed"] is False and "contradicted" in out["detail"]


def test_the_shifted_control_of_j2_still_shows_its_two_sample_delay():
    """J2's control whose output is the value two samples back, declaring onset 0, measured 2
    under v2; under the amended probe it still measures 2 and still fails."""
    tc = importlib.util.spec_from_file_location(
        "d3_tc", Path(__file__).resolve().parent / "test_d3_temporal_contract.py")
    mod = importlib.util.module_from_spec(tc)
    tc.loader.exec_module(mod)

    class Shifted(mod.TrailingMean):
        def transform(self, x, state):
            out = super().transform(x, state)
            out["values"] = [0.0, 0.0] + out["values"][:-2]
            out["available"] = [False, False] + out["available"][:-2]
            return out

    x = mod.x_of()
    out = battery.check_response_probe(Shifted(), battery.prefix(x, 180), x)
    assert out["passed"] is False and out["first_change_observed"] == 2
    assert out["identifiable"] is True and out["matches_declared"] is False


# --- nothing tuned --------------------------------------------------------------------------------

def test_operator_parameters_are_the_declared_defaults_unchanged():
    for op in ops.bank():
        assert op.params == type(op).DEFAULT_PARAMS
    assert ops.build("uniform_decile_quantizer").params == {"levels": 10}
    assert ops.build("wavelet_trailing").support()["samples"] == 50
