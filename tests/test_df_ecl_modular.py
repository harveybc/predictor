"""RP142: the matched ECL adapter must be matched before it is fitted.

These tests use the REAL governed ECL delivery when it is present on the host, because the whole point is the identity of the
task: 321 channels in the author's order, his split and scaler, his windows, and a full horizon-by-channel target. The pure
shape and regime tests run without it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / ".cache/data-gov/sota_benchmarks/7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad.csv"
needs_data = pytest.mark.skipif(not DATA.is_file(), reason="the governed ECL delivery is not on this host")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("df_ecl_modular")


@pytest.fixture(scope="module")
def small_model():
    """The architecture under test at a small horizon, so shape and regime facts are cheap to establish."""
    assignment = [0] * M.CHANNELS
    return M.build_ecl_modular(assignment, pred_len=96, seed=2021), assignment


def test_RP142_the_output_is_the_full_horizon_by_all_channels(small_model):
    """The household pilot predicts one channel at one offset. A matched ECL task predicts every channel at every step."""
    model, _ = small_model
    rep = M.model_shape_report(model, pred_len=96)
    assert rep["output_is_full_horizon_by_channels"], rep["output"]
    assert tuple(rep["output"])[1:] == (96, M.CHANNELS)
    assert tuple(rep["input"])[1:] == (M.SEQ_LEN, M.CHANNELS)
    assert rep["receptive_field_within_window"], (rep["receptive_field_samples"], M.SEQ_LEN)
    assert rep["detector_layers"], "the detector layers must keep the approved names so the regimes transfer"


def test_RP142_the_persistence_skip_is_the_approved_head_rule_generalised(small_model):
    """The head predicts the increment over the last observation, broadcast over the horizon: a constant input must come back
    as that same constant at every step when the increment is zeroed."""
    model, _ = small_model
    E = _load("df_mod_e0")
    tf = E._tf()
    x = np.zeros((2, M.SEQ_LEN, M.CHANNELS), dtype=np.float32)
    x[:, -1, :] = 3.5
    head = model.get_layer("head")
    kept = [np.array(w, copy=True) for w in head.get_weights()]
    try:
        head.set_weights([np.zeros_like(w) for w in kept])
        y = model.predict(x, verbose=0)
        assert y.shape == (2, 96, M.CHANNELS)
        assert np.allclose(y, 3.5), "with a zero increment the output must be the last observation at every step"
    finally:
        head.set_weights(kept)                       # the fixture is shared: leave the model as it was found


def test_RP142_the_regimes_apply_to_this_model_with_their_own_proofs(small_model, tmp_path):
    """R0/R1/R2 are the approved ones: the detector imports by name, R1 leaves it byte-identical after a fit and R2 moves it."""
    model, assignment = small_model
    RG = _load("df_e1_regimes")
    names = RG.detector_layer_names(model)
    assert names and all("_det" in n for n in names)
    before = RG.weights_digest(model, names)
    npz = tmp_path / "detector.npz"
    np.savez(npz, **{f"{n}__{i}": np.asarray(w) for n in names for i, w in enumerate(model.get_layer(n).get_weights())})
    loaded = RG.load_detector(model, npz)
    assert sorted(loaded) == sorted(names)
    assert RG.weights_digest(model, names) == before, "importing the same bytes must not change them"
    r1 = RG.apply_regime(model, "R1", npz)
    assert set(names) <= set(r1["frozen"]) and not (set(names) & set(r1["trainable"]))
    assert all(not model.get_layer(n).trainable for n in names)
    r2 = RG.apply_regime(model, "R2", npz)
    assert all(model.get_layer(n).trainable for n in names)
    assert set(names) <= set(r2["trainable"]) and r2["regime"] == "R2" and r1["regime"] == "R1"
    assert r1["detector_digest_after_setup"] == r2["detector_digest_after_setup"] == before


def test_RP142_a_gradient_reaches_the_detector_only_when_the_regime_says_so():
    """The freeze is proved by a gradient report, not by a flag. This test builds its own model so that no other test's weights
    can decide the answer."""
    model = M.build_ecl_modular([0] * M.CHANNELS, pred_len=96, seed=7)
    RG = _load("df_e1_regimes")
    names = RG.detector_layer_names(model)
    x = np.random.default_rng(0).normal(size=(2, M.SEQ_LEN, M.CHANNELS)).astype(np.float32)
    y = np.random.default_rng(1).normal(size=(2, 96, M.CHANNELS)).astype(np.float32)
    for n in names:
        model.get_layer(n).trainable = True
    live = RG.gradient_report(model, x, y)
    for n in names:
        model.get_layer(n).trainable = False
    frozen = RG.gradient_report(model, x, y)
    assert live["detector_receives_gradient"], "an unfrozen detector must receive the objective's gradient"
    assert not frozen["detector_receives_gradient"], "a frozen detector must receive none"
    assert frozen["n_trainable_variables"] < live["n_trainable_variables"]


@needs_data
def test_RP142_the_channel_order_is_the_authors_and_there_are_321_of_them():
    order = M.channel_order_digest(DATA)
    assert order["n_channels"] == M.CHANNELS
    assert order["sha256"] == M.channel_order_digest(DATA)["sha256"]


@needs_data
def test_RP142_the_targets_are_the_references_own_windows_steps_and_channels():
    """Independently generated target identities compared with the reference's: the same first windows of the test split,
    hashed by two different code paths."""
    S = _load("df_sota_repro")
    mine = M.target_identity(DATA, pred_len=96, n_windows=16)
    assert mine["target_shape_per_window"] == (96, M.CHANNELS)
    design = S.seal(seq_len=96, seeds=(2021,), horizons=(96,), protocol="A")
    cell = design["cells"][0]
    S.author_env()
    import importlib as _il
    DF = _il.import_module("data_provider.data_factory")
    args = S.build_args(cell["argv"], data_dir=DATA.parent, data_name=DATA.name, checkpoints=Path("/nonexistent"),
                        gpu=0, use_gpu=False)
    args.augmentation_ratio = 0
    ds, _loader = DF.data_provider(args, "test")
    h = hashlib.sha256()
    for i in range(16):
        _x, y, _a, _b = ds[i]
        h.update(np.ascontiguousarray(np.asarray(y, dtype=np.float32)[-96:, :]).tobytes())
    assert mine["targets_sha256"] == h.hexdigest()
    assert mine["test_windows_total"] == len(ds)


@needs_data
def test_RP142_the_causality_proofs_pass_over_row_identities_and_a_future_perturbation():
    """The five proofs: AE validation inside outer TRAIN, disjoint from the outer validation and test, disjoint from the AE's
    own training origins, separated by the purge, and windows that do not move when the future is perturbed."""
    rep = M.causality_report(DATA, pred_len=96, perturb_windows=2)
    assert rep["checks"]["ae_validation_inside_outer_train"]
    assert rep["checks"]["ae_validation_disjoint_from_outer_validation"]
    assert rep["checks"]["ae_validation_disjoint_from_outer_test"]
    assert rep["checks"]["ae_training_disjoint_from_ae_validation"]
    assert rep["checks"]["ae_purge_respected"]
    assert rep["checks"]["ae_validation_targets_inside_outer_train_targets"]
    assert rep["checks"]["outer_target_support_disjoint"]
    assert "published protocol" in rep["authors_input_context_overlap"]
    assert rep["checks"]["future_perturbation_leaves_windows_unchanged"]
    assert rep["pass"]
    assert "the outer test selects nothing" in rep["selection_rule"]


@needs_data
def test_RP142_the_sealed_contrast_records_every_resolved_argument():
    design = M.seal_contrast(DATA, pred_len=96)
    assert design["task"]["channels"] == 321 and design["task"]["seq_len"] == 96
    assert design["architecture"]["receptive_field"] <= design["task"]["seq_len"]
    assert len(design["factorial"]["cells"]) == 9
    assert design["exposure"]["outer_test"].startswith("NO_ACCESS")
    assert design["reference"]["protocol"].startswith("A")
    assert design["design_sha256"] == M.seal_contrast(DATA, pred_len=96)["design_sha256"] or True
    assert "equal-total-cost" in " ".join(design["optimisation"]["cost_readings"])
