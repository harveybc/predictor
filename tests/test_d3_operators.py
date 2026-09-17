"""The nine D3 operators and their twins, measured by the amended battery (J3).

Every operator is accepted by `df_d3_acceptance.run_battery` with its declared twin, every
twin fails the causality tests on its own, the two pointwise codecs are scoped with the
amendment's design reason, and the declarations that were MEASURED rather than assumed are
frozen here: the STFT onset of one sample, the wavelet support from the library, the
recursive state of the Butterworth and CUSUM operators. Missing values yield unavailable
outputs, never numbers. Nothing here promotes an operator.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("df_d3_contract")
battery = _load("df_d3_acceptance")
design = _load("df_d3_design")
ops = _load("df_d3_operators")

pytest.importorskip("scipy")
pytest.importorskip("pywt")

CONTRACT_0S = {"frequency": "1s", "availability": {"label": "WINDOW_START",
                                                  "completion_lag_max": "0s",
                                                  "timezone_evidence": "PRODUCER_STATEMENT",
                                                  "use_class": "LIVE_EQUIVALENT"}}


def series(n: int = 400, seed: int = 7) -> list:
    rng = random.Random(seed)
    return [0.5 * ((i * 37) % 101) / 10.0 - 2.5 + rng.gauss(0, 0.3) for i in range(n)]


@pytest.fixture(scope="module")
def x():
    return battery.make_input(series())


def test_the_bank_is_exactly_the_nine_operators_of_the_amendment():
    kinds = [op.KIND for op in ops.bank()]
    assert kinds == [entry["kind"] for entry in design.OPERATORS]


def test_every_declaration_validates_and_names_its_libraries():
    for op in ops.bank():
        spec = contract.validate_spec(op.describe())
        assert spec["library_versions"]["numpy"]
        assert spec["fit_scope"] == design.operator_entry(op.KIND)["fit_scope"]


@pytest.mark.parametrize("kind", list(ops.OPERATORS))
def test_each_operator_is_review_ready_with_its_twin(kind, x):
    op = ops.build(kind)
    report = battery.run_battery(op, x, twin=ops.twin_of(op), resource_contract=CONTRACT_0S)
    assert report["failed"] == [], json.dumps(report["results"], default=str)[:1500]
    assert report["undecided"] == []
    assert report["review_ready"] is True


@pytest.mark.parametrize("kind", list(ops.TWINS))
def test_each_twin_fails_causality_on_its_own(kind, x):
    twin = ops.build(kind)
    train = battery.prefix(x, 240)
    p = battery.check_prefix_all_available(twin, train, x)
    f = battery.check_future_perturbation(twin, train, x)
    assert p["passed"] is False or f["passed"] is False


def test_the_two_pointwise_codecs_are_scoped_with_the_design_reason(x):
    for kind in ("uniform_decile_quantizer", "delta_run_length"):
        op = ops.build(kind)
        report = battery.run_battery(op, x, twin=None, resource_contract=CONTRACT_0S)
        assert report["scoped"] == ["non_causal_twin"]
        expected = design.operator_entry(kind)["twin_not_applicable_reason"]
        assert report["results"]["non_causal_twin"]["reason"] == expected


def test_the_stft_onset_of_one_was_measured_not_assumed(x):
    op = ops.build("stft_trailing")
    outcome = battery.check_response_probe(op, battery.prefix(x, 240), x)
    assert outcome["observed"] == 1 and outcome["passed"] is True
    import numpy as np

    assert np.hanning(16)[-1] == 0.0


def test_wavelet_support_comes_from_the_library():
    import pywt

    assert ops.wavelet_support("haar", 3) == 8
    assert ops.wavelet_support("db4", 3) == (pywt.Wavelet("db4").dec_len - 1) * 7 + 1
    spec = ops.build("wavelet_trailing").describe()
    assert spec["support"]["samples"] == 50 and spec["support"]["boundary_mode"] == "zero"
    assert "dec_len=8" in spec["support"]["derivation"]


def test_recursive_operators_declare_a_dependency_not_a_finite_memory():
    for kind in ("butterworth_causal", "cusum_causal"):
        spec = ops.build(kind).describe()
        assert spec["support"]["kind"] == "RECURSIVE" and spec["support"]["samples"] is None
        assert spec["chunk_restart"] == "STATEFUL_WITH_CHECKPOINT"


def test_butterworth_checkpoint_resume_reproduces_one_pass(x):
    op = ops.build("butterworth_causal")
    outcome = battery.check_chunk_restart(op, battery.prefix(x, 240), x)
    assert outcome["passed"] is True and outcome["mode"] == "STATEFUL_WITH_CHECKPOINT"


def test_cusum_checkpoint_resume_reproduces_one_pass(x):
    op = ops.build("cusum_causal")
    outcome = battery.check_chunk_restart(op, battery.prefix(x, 240), x)
    assert outcome["passed"] is True


@pytest.mark.parametrize("kind", list(ops.OPERATORS))
def test_a_missing_value_yields_unavailable_outputs_never_numbers(kind):
    values = series(120)
    values[60] = float("nan")
    x = battery.make_input(values)
    op = ops.build(kind)
    out = op.transform(x, op.fit(battery.prefix(x, 72)))
    assert out["available"][60] is False
    for i, flag in enumerate(out["available"]):
        if flag:
            assert out["values"][i] == out["values"][i]        # never NaN when available


def test_the_raw_branch_is_the_input_for_every_operator(x):
    for op in ops.bank():
        out = op.transform(x, op.fit(battery.prefix(x, 240)))
        assert out["raw"] == x["values"]


def test_an_unknown_parameter_is_refused():
    with pytest.raises(contract.SpecRefusal, match="unknown parameter"):
        ops.build("stft_trailing", width=3)


def test_twin_of_carries_the_operators_parameters():
    op = ops.build("variance_regime_trailing", w=20)
    twin = ops.twin_of(op)
    assert twin.KIND == "variance_regime_centred" and twin.params["w"] == 20
    assert ops.twin_of(ops.build("delta_run_length")) is None


def test_the_measured_cost_is_within_every_declaration(x):
    for op in ops.bank():
        outcome = battery.check_cost_pilot(op, battery.prefix(x, 240), x)
        assert outcome["passed"] is True, (op.KIND, outcome)
