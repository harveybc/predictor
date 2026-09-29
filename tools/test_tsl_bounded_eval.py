#!/usr/bin/env python3
"""Tests of the Traffic bounded evaluator's generators: what it derives, what it refuses, and what it is NOT allowed to be.

These tests do not run a model and do not read the dataset. They assert the properties that make the measured numbers mean what
they are reported to mean: that this module introduces no reduction of its own, that the bounded path's resident terms do not
contain the population size while its DISK term does, that the author-path figure is the sealed one rather than a second
derivation, that a chunked average is not silently accepted as the author's reduction, and that Traffic's clock is not Weather's.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

B = importlib.import_module("df_tsl_bounded_eval")
X = importlib.import_module("df_tsl_execute")
R = importlib.import_module("df_tsl_repro")
S = importlib.import_module("df_sota_repro")

LANE = Path.home() / ".local/state/crispdm-data-foundation/tsl_traffic_bounded_20260929"
DESIGN = LANE / "DESIGN.traffic.L96.json"
CHAR = LANE / "CHARACTERIZATION.traffic.json"
FALLBACK_DESIGN = Path.home() / ".local/state/crispdm-data-foundation/tsl_weather_rb02_20260928/DESIGN.traffic.L96.json"


def _design():
    for p in (DESIGN, FALLBACK_DESIGN):
        if p.is_file():
            return json.loads(p.read_text())
    pytest.skip("the sealed Traffic design is not retained on this host")


def _char():
    if not CHAR.is_file():
        pytest.skip("the Traffic characterization is not retained on this host")
    return json.loads(CHAR.read_text())


# --- what this module is not allowed to be ---------------------------------------------------------------------------------

def test_the_module_defines_no_reduction_of_its_own():
    """The bounded reduction is the EXISTING one. If this module ever grew its own mean, the number would stop being the
    author's, and no measured peak would make that acceptable."""
    src = inspect.getsource(B)
    body = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "S.author_metric_exact" in body, "the bounded reduction must be the existing author_metric_exact"
    assert "S.bounded_test" in body, "the bounded evaluator must be the existing bounded_test"
    # the only np.mean in this module is the counter-example inside reducer_parity, which is explicitly the thing it is NOT
    means = [ln for ln in body.splitlines() if "np.mean(" in ln]
    assert len(means) == 1, f"this module must not compute means outside the named counter-example: {means}"
    assert "chunk_means_mae" in means[0]
    for forbidden in ("astype(np.float16", "float16", "[::", "sub_sample", "subsample"):
        assert forbidden not in body, f"{forbidden!r} would change the population or the dtype"


def test_the_bounded_evaluator_it_calls_is_the_sealed_one():
    assert S.BOUNDED_ADAPTER_VERSION == "df_sota_bounded_eval.v1"
    assert S.AUTHOR_SCORER_ROUTE == "df_sota_author_metric_exact.v2"
    assert "bit_equal" in inspect.getsource(S.bounded_test)
    assert "REFUSED: the exact bounded route disagrees with utils.metrics.metric" in inspect.getsource(S.bounded_test)


def test_the_invariants_are_declared_and_carried():
    kinds = {i.split(":")[0] for i in B.INVARIANTS}
    assert kinds == {"input population", "model", "batch semantics", "target", "dtype", "reduction", "checkpoint rule"}


# --- the derivation, term by term ------------------------------------------------------------------------------------------

def test_the_author_path_figure_is_the_sealed_derivation_not_a_second_one():
    """`memory_split` must report the SAME author-path number `df_tsl_execute` already derives. A second derivation of the same
    quantity is how two documents come to carry two figures."""
    d, c = _design(), _char()
    split = B.memory_split(d, c, baseline_bytes=0, cap_bytes=1 << 34)
    for h in d["horizons"]:
        sealed = X.eval_path_memory_derivation(d, c, h, baseline_bytes=0)
        assert split["horizons"][f"h{h}"]["author_path"]["derived_peak_bytes"] == sealed["derived_peak_bytes"]


def test_the_bounded_resident_terms_do_not_contain_the_population_size():
    """The whole point: the bounded path's resident cost is a function of the batch, the leaf and the flush window, never of how
    many windows the test set has. If a term ever started scaling with `windows`, the path would not be bounded."""
    d, c = _design(), _char()
    for h in d["horizons"]:
        real = B.bounded_path_derivation(d, c, h, baseline_bytes=0)
        doubled = json.loads(json.dumps(c))
        key = f"L{d['seq_len']}_h{h}"
        w = doubled["sets"][key]["test"]["windows"] * 2
        doubled["sets"][key]["test"]["windows"] = w
        doubled["sets"][key]["elements_test"] = w * h * doubled["sets"][key]["test"]["channels"]
        twice = B.bounded_path_derivation(d, c if False else doubled, h, baseline_bytes=0)
        assert twice["arrays_peak_bytes"] == real["arrays_peak_bytes"], "a resident term scaled with the population"
        assert twice["disk"]["total_bytes"] == 2 * real["disk"]["total_bytes"], "the disk term must scale; it is the budget"


def test_the_author_path_terms_scale_with_the_population_and_the_bounded_ones_do_not():
    d, c = _design(), _char()
    h = max(d["horizons"])
    author = X.eval_path_memory_derivation(d, c, h, baseline_bytes=0)
    bounded = B.bounded_path_derivation(d, c, h, baseline_bytes=0)
    assert author["arrays_peak_bytes"] > 20 * bounded["arrays_peak_bytes"]


def test_every_term_is_classified_and_the_model_side_is_never_implementation():
    d, c = _design(), _char()
    split = B.memory_split(d, c, baseline_bytes=0, cap_bytes=1 << 34)
    for row in split["horizons"].values():
        for side in ("author_path", "bounded_path"):
            for name, term in row[side]["terms"].items():
                assert term["term_class"] == "EVALUATION_IMPLEMENTATION", f"{side}.{name} is not an implementation term"
    assert set(split["term_classes"]) == {"EVALUATION_IMPLEMENTATION", "MODEL"}
    assert split["model_side_unmeasured"]["activations"] == "UNMEASURED"


def test_the_disk_budget_is_stated_and_the_retention_rule_deletes():
    d, c = _design(), _char()
    for h in d["horizons"]:
        disk = B.bounded_path_derivation(d, c, h, baseline_bytes=0)["disk"]
        assert disk["total_bytes"] > 0
        assert disk["retained_after_metrics"] == 0
        assert "deleted" in disk["retention_rule"]


def test_an_element_count_that_does_not_reconcile_is_refused():
    d, c = _design(), _char()
    bad = json.loads(json.dumps(c))
    bad["sets"][f"L{d['seq_len']}_h720"]["elements_test"] += 1
    with pytest.raises(B.BoundedRefusal):
        B.bounded_path_derivation(d, bad, 720, baseline_bytes=0)


# --- the verdict -----------------------------------------------------------------------------------------------------------

def test_an_unread_number_never_admits():
    assert B.probe_verdict(measured_peak_bytes=None, cap_bytes=1 << 33, population_complete=True,
                           parity_holds=True, author_function_bit_equal=True) == "UNDETERMINED"
    assert B.probe_verdict(measured_peak_bytes=1, cap_bytes=None, population_complete=True,
                           parity_holds=True, author_function_bit_equal=True) == "UNDETERMINED"


def test_the_cap_boundary_is_inclusive_and_one_byte_over_is_a_deficit():
    cap = 8 * (1 << 30)
    assert B.probe_verdict(measured_peak_bytes=cap, cap_bytes=cap, population_complete=True,
                           parity_holds=True, author_function_bit_equal=None) == "ADMISSIBLE"
    assert B.probe_verdict(measured_peak_bytes=cap + 1, cap_bytes=cap, population_complete=True,
                           parity_holds=True, author_function_bit_equal=None) == "CAPACITY_DEFICIT"


def test_an_incomplete_population_is_not_a_measurement():
    assert B.probe_verdict(measured_peak_bytes=1, cap_bytes=1 << 33, population_complete=False,
                           parity_holds=True, author_function_bit_equal=True) == "REFUSED_INCOMPLETE_POPULATION"


def test_broken_parity_is_refused_even_when_it_fits():
    assert B.probe_verdict(measured_peak_bytes=1, cap_bytes=1 << 33, population_complete=True,
                           parity_holds=False, author_function_bit_equal=None) == "REFUSED_PARITY_BROKEN"
    assert B.probe_verdict(measured_peak_bytes=1, cap_bytes=1 << 33, population_complete=True,
                           parity_holds=None, author_function_bit_equal=False) == "REFUSED_PARITY_BROKEN"


def test_an_undemonstrated_parity_is_labelled_inherited_and_never_as_demonstrated():
    v = B.probe_verdict(measured_peak_bytes=1, cap_bytes=1 << 33, population_complete=True,
                        parity_holds=None, author_function_bit_equal=None)
    assert v == "ADMISSIBLE_PARITY_INHERITED"
    assert "inherited rather than measured here" in B.PROBE_VERDICTS[v]


# --- the target population's identity ---------------------------------------------------------------------------------------

def test_target_pairing_refuses_a_probe_from_another_design():
    d, c = _design(), _char()
    with pytest.raises(B.BoundedRefusal):
        B.target_pairing(d, c, data_path=Path("/nonexistent"), horizon=720,
                         probe={"design_sha256": "0" * 64, "horizon_steps": 720})


def test_target_pairing_refuses_a_probe_for_another_horizon():
    d, c = _design(), _char()
    with pytest.raises(B.BoundedRefusal):
        B.target_pairing(d, c, data_path=Path("/nonexistent"), horizon=720,
                         probe={"design_sha256": d["design_sha256"], "horizon_steps": 96})


def test_target_pairing_compares_two_independently_produced_digests():
    """It must read the bounded writer's digest out of the probe record and the author loader's digest out of a fresh pass, and
    compare them. A version that recomputed one from the other would prove nothing."""
    src = inspect.getsource(B.target_pairing)
    assert "S.naive_and_trues" in src
    assert 'probe["population"]["trues_sha256"]' in src


# --- the clock -------------------------------------------------------------------------------------------------------------

def test_traffic_96_steps_are_96_hours_and_weather_96_steps_are_16():
    """The same step count is not the same elapsed horizon. This is the guard against reporting one as the other."""
    assert R.horizon_seconds("traffic", 96) == 96 * 3600
    assert R.horizon_seconds("weather", 96) == 96 * 600
    assert R.horizon_seconds("traffic", 96) != R.horizon_seconds("weather", 96)
    assert R.horizon_seconds("traffic", 96) / 3600 == 96
    assert R.horizon_seconds("weather", 96) / 3600 == 16


def test_the_split_carries_the_elapsed_horizon_beside_the_step_count():
    d, c = _design(), _char()
    split = B.memory_split(d, c, baseline_bytes=0, cap_bytes=1 << 34)
    for h in d["horizons"]:
        row = split["horizons"][f"h{h}"]
        assert row["horizon_seconds"] == int(h) * 3600
        assert row["horizon_steps"] == int(h)


# --- the design's identity -------------------------------------------------------------------------------------------------

def test_traffics_margin_is_traffics_own_and_not_weathers():
    d = _design()
    std = d["lock"]["agreement"]["std_paper"]
    assert (std["mse"], std["mae"]) == (0.008, 0.004)
    assert "Traffic" in d["lock"]["agreement"]["source"]


def test_the_module_refuses_a_characterization_from_another_design():
    d, c = _design(), _char()
    foreign = json.loads(json.dumps(c))
    foreign["design_sha256"] = "0" * 64
    with pytest.raises(B.BoundedRefusal):
        B.bounded_probe(d, foreign, data_path=Path("/nonexistent"), work=Path("/nonexistent"), horizon=96)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
