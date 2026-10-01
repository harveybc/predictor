"""Lane H: cross-worker replay comparison (exact layer vs numeric layer; non-causal control excluded from the exact layer)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cmp_ = _load("h_kalman_replay_compare")


def doc(kalman_digest="k1", smoother="s1", pred="p1"):
    return {"inputs": {"view_sha256": "v"},
            "kalman": {"v": {"local_level": {"artifact_sha256": "a", "fitted_state_digest": "f", "output_digest": kalman_digest}}},
            "arms_exact_inputs": {"v|lags1|B": {"train": "t", "validation": "x"},
                                  "v|lags1|C_SMOOTHER_NONCAUSAL": {"train": smoother, "validation": smoother}},
            "arms_numeric_predictions": {"v|lags1|B": pred, "v|lags1|C_SMOOTHER_NONCAUSAL": pred}}


def test_identical_exact_layer_even_when_the_noncausal_control_and_predictions_differ():
    r = cmp_.compare(doc(), doc(smoother="s2", pred="p2"))
    assert r["exact_core_equal"] is True and r["noncausal_control_excluded"] == ["v|lags1|C_SMOOTHER_NONCAUSAL"]
    assert r["numeric_predictions_equal"] is False and r["numeric_differing"] == ["v|lags1|B", "v|lags1|C_SMOOTHER_NONCAUSAL"]


def test_a_kalman_digest_difference_breaks_the_exact_layer():
    r = cmp_.compare(doc(), doc(kalman_digest="k2"))
    assert r["exact_core_equal"] is False and r["exact_differences"] == [["kalman", "v", "local_level", "output_digest"]]


def test_input_identity_mismatch_is_reported():
    a, b = doc(), doc()
    b["inputs"]["view_sha256"] = "other"
    r = cmp_.compare(a, b)
    assert r["exact_core_equal"] is False and ["inputs", "view_sha256"] in r["exact_differences"]
