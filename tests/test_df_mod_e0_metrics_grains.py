"""RP20: the metrics describe every grain the learner consumed and the whole planted signal."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("df_mod_e0")
M = _load("df_mod_e0_metrics")


def _dm(diagnostic=None):
    g = E.generate(3, 1, 1, diagnostic=diagnostic)
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(g["x"], g["oracle"], periods)
    return g, prep, M.data_metrics(g, prep, ["train", "validation", "test"], 2)


def test_RP20_composition_holds_with_the_deterministic_term_and_snr_is_versioned():
    g, prep, dm = _dm("trend_event")
    assert dm["composition"]["max_residual"] < 1e-9 and dm["composition"]["diagnostic"] == "trend_event"
    v = dm["splits"]["train"]["variables"][0]
    assert v["snr_planted_total_db"] > v["snr_planted_db"] and v["snr_total_state"] == "MEDIDO" and "v1" in v["snr_versions"]["snr_planted_db"]
    # the v1 formula omits the deterministic term: x - (v1 signal + noise) is NOT zero under the diagnostic
    x_without = g["s"] + g["periodic"] + g["cross"] + g["noise"]
    assert np.max(np.abs(g["x"] - x_without)) > 1.0
    g0, prep0, dm0 = _dm(None)
    v0 = dm0["splits"]["train"]["variables"][0]
    assert v0["snr_planted_total_db"] == pytest.approx(v0["snr_planted_db"]) and dm0["composition"]["max_residual"] < 1e-9


def test_RP20_the_four_grains_have_their_own_identities_shapes_bytes_and_repetitions():
    g, prep, dm = _dm()
    b = prep["boundaries"]
    W, h = b["window"], b["horizon"]
    for part in ("train", "validation", "test"):
        lo, hi = b[part]
        gr = dm["splits"][part]["grains"]
        n = hi - lo
        assert gr["base_series"]["row_identity"] == [lo, hi] and gr["base_series"]["shape"] == [n, 8] and gr["base_series"]["bytes_raw"] == n * 8 * 8
        assert gr["inputs_unique"]["row_identity"] == [lo - W + 1, hi] and gr["inputs_unique"]["shape"] == [n + W - 1, 8]
        assert gr["window_tensor"]["shape"] == [n, W, 8] and gr["window_tensor"]["bytes_raw"] == n * W * 8 * 8
        assert gr["window_tensor"]["repetition_factor"] == pytest.approx(n * W / (n + W - 1))
        assert gr["targets"]["row_identity"] == [lo + h, hi + h] and gr["targets"]["shape"] == [n, 8] and gr["targets"]["horizon"] == h
        assert gr["targets"]["mase_denominator"] == [float(d) for d in prep["mase_denominator"]]
        assert gr["base_series"]["scale"]["applied"] is False and len(gr["base_series"]["scale"]["train_only"]["mean"]) == 8
        assert all(gr[k]["mask_non_finite"] == 0 for k in gr)
        # the window tensor is a repetition of the unique inputs: it compresses better per value
        assert gr["window_tensor"]["compressed_bits_per_value_zlib9"] < gr["inputs_unique"]["compressed_bits_per_value_zlib9"]
    rows, states = M.terminal_rows({"data_metrics": dm, "model_metrics": None, "parameters": None})
    names = {r[0] for r in rows}
    assert "mod_e0.data.grain.window_tensor.bytes_raw" in names and "mod_e0.data.grain.targets.bytes_raw" in names
    assert "mod_e0.data.snr_planted_total_db_mean" in names and "mod_e0.data.var0.snr_planted_total_db" in names
    assert all(np.isfinite(r[1]) for r in rows)


def test_RP20_grain_descriptors_do_not_impersonate_each_other():
    g, prep, dm = _dm()
    gr = dm["splits"]["validation"]["grains"]
    assert gr["base_series"]["bytes_raw"] != gr["window_tensor"]["bytes_raw"] != gr["targets"]["bytes_raw"] or gr["targets"]["row_identity"] != gr["base_series"]["row_identity"]
    assert "not descriptors of X or Y" in M.DECLARATIONS["grains"]
