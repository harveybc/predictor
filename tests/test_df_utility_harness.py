"""L5: the utility harness on fabricated data of known truth, with negative controls, and
without ever scoring the reserved holdout. Nothing here is a scientific result."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


H = _load("df_utility_harness")

PROTO = H.Protocol(target="return", horizon=1, model="ridge", window=4, n_blocks=5,
                   margin=0.0, seed=7, comparisons=3, min_rows_per_block=40)


def trailing_mean(x: np.ndarray, w: int = 8) -> dict:
    """A causal representation with a known reach of zero: the trailing mean of w samples."""
    n = x.size
    values = np.zeros(n)
    available = np.zeros(n, dtype=bool)
    for t in range(w - 1, n):
        seg = x[t - w + 1:t + 1]
        if not np.isnan(seg).any():
            values[t] = seg.mean()
            available[t] = True
    return {"values": values, "available": available, "reach_right": 0, "accepted": True}


def fabricated(n=3000, seed=3, truth="mean"):
    """x whose next step depends on the trailing mean (truth='mean') or on nothing ('noise')."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        drive = x[max(0, t - 8):t].mean() if truth == "mean" and t > 8 else 0.0
        x[t] = x[t - 1] + (-0.5 * drive if truth == "mean" else 0.0) + rng.normal(0, 1.0)
    return x


def test_the_protocol_is_sealed_and_names_its_loss_as_a_loss():
    doc = PROTO.sealed()
    assert doc["protocol_sha256"] and doc["schema"] == "df_utility_protocol.v1"
    out = H.contrast(fabricated(), trailing_mean(fabricated()), PROTO)
    assert out["loss_name"] == "mae" and "not information" in out["note"]


def test_a_representation_that_carries_the_truth_advances_on_fabricated_data():
    x = fabricated(truth="mean")
    out = H.contrast(x, trailing_mean(x), PROTO)
    assert out["outcome"] == H.ADVANCES, out
    assert out["delta_lower"] > 0 and out["blocks_used"] == 5


def test_a_noise_target_does_not_advance_negative_control():
    x = fabricated(truth="noise")
    out = H.contrast(x, trailing_mean(x), PROTO)
    assert out["outcome"] == H.DOES_NOT_ADVANCE, out
    assert abs(out["delta_mean"]) < 0.2


def test_a_representation_not_mechanically_accepted_is_refused_before_any_scoring():
    x = fabricated()
    rep = dict(trailing_mean(x), accepted=False)
    out = H.contrast(x, rep, PROTO)
    assert out["outcome"] == H.REFUSED and "not MECHANICALLY_ACCEPTED" in out["why"]


def test_both_branches_are_scored_on_the_same_paired_rows():
    x = fabricated()
    x[100:140] = np.nan                       # a gap: the raw and transformed rows differ
    rep = trailing_mean(x)
    out = H.contrast(x, rep, PROTO)
    cov = out["coverage"]
    assert cov["rows_paired"] <= min(cov["rows_a"], cov["rows_b"])
    assert cov["inputs_missing"] == 40


def test_the_purge_covers_the_label_horizon_the_reach_and_the_window():
    x = fabricated()
    rep = dict(trailing_mean(x), reach_right=5)
    proto = H.Protocol(target="return", horizon=3, model="ridge", window=4, n_blocks=5,
                       margin=0.0, seed=7, comparisons=3, min_rows_per_block=40)
    out = H.contrast(x, rep, proto)
    assert out["purge"] == 3 + 5 + 4
    for b in out["blocks"]:
        if "delta" in b:
            assert b["train_rows"] > 0
    scheme = H.blocks(np.arange(1000), proto, purge=12)
    for b in scheme:
        assert b["train"].max() <= b["validation"][0] - 12


def test_an_exhausted_budget_is_a_recorded_outcome_not_a_silent_drop():
    x = fabricated()
    out = H.contrast(x, trailing_mean(x), PROTO, cpu_seconds=999.0)
    assert out["outcome"] == H.BUDGET_EXHAUSTED and out["budget"] == 60.0


def test_too_few_rows_is_insufficient_never_a_verdict():
    x = fabricated(n=120)
    out = H.contrast(x, trailing_mean(x), PROTO)
    assert out["outcome"] == H.INSUFFICIENT_ROWS


def test_the_augmented_branch_is_a_separate_declared_contrast():
    x = fabricated(truth="mean")
    out = H.contrast(x, trailing_mean(x), PROTO, branch_a="raw", branch_b="augmented")
    assert out["branch_b"] == "augmented" and out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE)


def test_the_reserved_holdout_is_adjudicated_once_and_refuses_a_second_look(tmp_path):
    calls = []
    out = H.adjudicate_holdout(tmp_path, PROTO, lambda: calls.append(1) or {"scored": True})
    assert out == {"scored": True} and calls == [1]
    marker = json.loads((tmp_path / "HOLDOUT_USED.json").read_text())
    assert marker["protocol_sha256"] == PROTO.sealed()["protocol_sha256"]
    with pytest.raises(SystemExit, match="second look"):
        H.adjudicate_holdout(tmp_path, PROTO, lambda: calls.append(2))
    assert calls == [1]


def test_the_direction_target_uses_log_loss_with_the_logistic_probe():
    proto = H.Protocol(target="direction", horizon=1, model="logistic", window=4, n_blocks=4,
                       margin=0.0, seed=1, comparisons=1, min_rows_per_block=40)
    x = fabricated(truth="mean")
    out = H.contrast(x, trailing_mean(x), proto)
    assert out["loss_name"] == "log_loss" and out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE)
    for b in out["blocks"]:
        if "loss_a" in b:
            assert 0 < b["loss_a"] < 2 and 0 < b["loss_b"] < 2
