"""Review B.1 blockers B-1/B-2/B-4: the sealed market-time identity must actually fit.

Exercises the REAL seams only: the sealed MG2/MG12 designs, a REAL market bundle
built by the data lane from synthetic inputs, and the REAL trainer CLI on CPU.
No fakes, no GPU, no scientific fit, no TEST rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6dmo import data, design as D, market_grid, model as M, train as T  # noqa: E402
from tests.i6dmo import _synth  # noqa: E402
from tests.i6dmo.dmo_model_fixtures import DESIGN  # noqa: E402

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
CANDIDATES = DESIGN.parent / "market_grid"
SEALED = {"mg2": DESIGN.parent / "EXPERIMENT.mg2.json",
          "mg12": DESIGN.parent / "EXPERIMENT.mg12.json"}


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    """A REAL MG2 + MG12 bundle pair built by the data lane from synthetic inputs."""
    root = tmp_path_factory.mktemp("mtfit")
    src = _synth.make_input(root / "input")
    market_grid.build_market_bundle(
        src, market_grid.market_design(_synth.load_design()), root / "mg2", code_commit="TEST")
    market_grid.build_market_bundle(
        src, market_grid.market_design(data.long_data_view(_synth.load_design())),
        root / "mg12", code_commit="TEST")
    return {"root": root, "input": src, "mg2": root / "mg2", "mg12": root / "mg12"}


# ------------------------------------------------------------------ B-1 (RB04)

@pytest.mark.parametrize("key", ["mg2", "mg12"])
def test_the_sealed_effective_config_builds_its_model(key):
    effective = D.effective_config(D.load_design(SEALED[key]))
    bundle = M.build_dmo(effective["model"])          # the exact call train_unit makes
    model = bundle.forecast_model
    assert [tuple(t.shape[1:]) for t in model.inputs] == [(13, 19), (13,), (13,)]
    assert tuple(bundle.core_model.input_shape) == (None, 13, 78)
    assert len(model.outputs) == 6
    # Positive control parity: the family helper and the sealed effective config agree exactly.
    helper = M.model_config(effective["feature_manifest"]["features"],
                            family=effective["family"])
    assert M.normalize_model_config(effective["model"]) == helper


# ------------------------------------------------------------------ B-4 (RB02 E2)

def _good(rows=3, slots=13, spacing=2.0):
    offsets = np.tile(np.arange(-(slots - 1) * spacing, spacing, spacing), (rows, 1))
    offsets[:, -1] = 0.0
    flags = np.zeros((rows, slots), np.float32)
    return offsets.astype(np.float32), flags


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda o, f: (np.tile(np.arange(-12.0, 1.0), (len(o), 1)).astype(np.float32), f),
     "ORDINAL_OR_INVALID_OFFSETS"),          # reviewer case 1: slot indices
    (lambda o, f: (np.where(np.arange(o.shape[1]) == 5, 3.0, o).astype(np.float32), f),
     "ORDINAL_OR_INVALID_OFFSETS"),          # reviewer case 2: positive future offset
    (lambda o, f: (np.where(np.arange(o.shape[1]) == o.shape[1] - 1, -1.0, o).astype(np.float32), f),
     "ORDINAL_OR_INVALID_OFFSETS"),          # last offset not exactly 0
    (lambda o, f: (o, np.where(np.arange(f.shape[1]) == 0, 1.0, f).astype(np.float32)),
     "ORDINAL_OR_INVALID_OFFSETS"),          # first closure flag not 0
])
def test_encoder_refuses_ordinal_or_impossible_time_inputs(mutate, code):
    layer = M.DynamicPhysicalTimeEncoder()
    offsets, flags = _good()
    offsets, flags = mutate(*(offsets, flags))
    values = tf.zeros((len(offsets), offsets.shape[1], 76))
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match=code):
        layer([values, tf.constant(offsets), tf.constant(flags)])


def test_encoder_accepts_the_valid_convention_and_stays_numerically_unchanged():
    layer = M.DynamicPhysicalTimeEncoder()
    offsets, flags = _good()
    values = tf.zeros((3, 13, 76))
    out = np.asarray(layer([values, tf.constant(offsets), tf.constant(flags)]))
    assert out.shape == (3, 13, 78)
    assert out[0, 0, 76] == 0.0 and out[0, 0, 77] == 0.0
    assert np.allclose(out[0, 1:, 76], 2.0 / 24.0, atol=1e-6)


def test_trainer_input_assembly_refuses_ordinal_offsets(market):
    arrays = T.load_market_split_arrays(market["mg2"], "fit", SEALED["mg2"])
    bad = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in arrays.items()}
    bad["offsets_hours"] = np.tile(np.arange(-12.0, 1.0), (len(bad["row_id"]), 1))
    with pytest.raises(T.TrainDisposition, match="ORDINAL_OR_INVALID_OFFSETS"):
        T.market_time_inputs(bad)
    bad2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in arrays.items()}
    bad2["offsets_hours"] = arrays["offsets_hours"].copy()
    bad2["offsets_hours"][0, 4] = 3.0
    with pytest.raises(T.TrainDisposition, match="ORDINAL_OR_INVALID_OFFSETS"):
        T.market_time_inputs(bad2)
    ok = T.market_time_inputs(arrays)
    assert len(ok) == 3 and ok[1].shape == ok[2].shape == arrays["offsets_hours"].shape


# ------------------------------------------------------------------ B-2 (RB03 J1)

@pytest.mark.parametrize("key", ["mg2", "mg12"])
def test_trainer_loads_the_real_market_bundle_and_assembles_three_inputs(market, key):
    effective = T.effective_from_design(SEALED[key])
    loaded = T.load_data_bundle(market[key], effective)
    assert loaded["kind"] == "market"
    for split in ("fit", "inner", "validation"):
        arrays = loaded[split]
        assert set(arrays) >= {"row_id", "x", "origin_price", "target_price", "mask",
                               "offsets_hours", "closure_crossed"}
        assert arrays["offsets_hours"].shape == (len(arrays["row_id"]), 13)
        assert np.all(arrays["offsets_hours"][:, -1] == 0)
        assert np.all(arrays["offsets_hours"] <= 0)
        assert np.all(arrays["closure_crossed"][:, 0] == 0)
    scales = T.fit_scales(T.trainable_rows(loaded["fit"])[0])
    rows, _ = T.trainable_rows(loaded["fit"])
    inputs = T.materialize_monitor_inputs(rows, scales)[0]
    assert isinstance(inputs, list) and len(inputs) == 3
    assert inputs[0].shape[1:] == (13, 19)
    bundle = M.build_dmo(effective["model"])
    pred = M.predict_matrix(bundle.forecast_model, inputs)
    assert pred.shape == (len(rows["row_id"]), 6) and np.isfinite(pred).all()


@pytest.mark.parametrize("key", ["mg2", "mg12"])
def test_real_cli_end_to_end_on_the_sealed_design_and_real_bundle(market, tmp_path, key):
    """The REAL fit-unit invocation, two epochs, a tiny row subset, CPU only."""
    out = tmp_path / f"unit_{key}"
    env = {**ENV, "I6DMO_TEST_TRAINING_OVERRIDES": json.dumps({"max_epochs": 2}),
           "I6DMO_TEST_MAX_ROWS": "48"}
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.train", "--design", str(SEALED[key]),
                          "--bundle", str(market[key]), "--out", str(out), "--device", "cpu",
                          "--json"], capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert run.returncode == 0, run.stdout[-2500:] + run.stderr[-2500:]
    receipt = json.loads((out / "RECEIPT.json").read_text())
    assert receipt["status"] == "COMPLETED", receipt
    assert receipt["experiment_id"] == json.loads(SEALED[key].read_text())["experiment_id"]
    assert receipt["identity"]["design_sha256"] == D.design_sha256(json.loads(SEALED[key].read_text()))
    assert receipt["market_time"]["rule"].endswith("_MARKET_EXACT")
    doc = json.loads((out / "model" / "MODEL.json").read_text())
    assert doc["market_time"]["slot_order_is_not_wall_clock"] is True
    with np.load(out / "predictions" / "validation.npz") as pz:
        assert pz["predicted_price"].shape[1] == 6
        assert pz["horizons_hours"].tolist() == json.loads(SEALED[key].read_text())["horizons_hours"]
    # The reloaded artifact reproduces the saved predictions through the 3-input path.
    reloaded, _ = M.load_dmo(out / "model")
    assert len(reloaded.forecast_model.inputs) == 3


def test_a_ds2_bundle_is_refused_under_a_market_design_and_vice_versa(market, tmp_path):
    from tests.i6dmo.dmo_model_fixtures import write_bundle
    market_eff = T.effective_from_design(SEALED["mg2"])
    with pytest.raises(T.TrainDisposition, match="BUNDLE_SCHEMA|MARKET_BUNDLE"):
        T.load_data_bundle(write_bundle(tmp_path / "ds2"), market_eff)
    elapsed_eff = T.effective_from_design(DESIGN)
    with pytest.raises(T.TrainDisposition, match="BUNDLE_SCHEMA|MARKET_BUNDLE"):
        T.load_data_bundle(market["mg2"], elapsed_eff)
    # And a MG12 bundle never satisfies the MG2 design.
    with pytest.raises(T.TrainDisposition, match="MARKET_RULE_MISMATCH|MARKET_"):
        T.load_data_bundle(market["mg12"], market_eff)


def test_ds2_elapsed_path_is_untouched(tmp_path):
    from tests.i6dmo.dmo_model_fixtures import write_bundle
    effective = T.effective_from_design(DESIGN)
    loaded = T.load_data_bundle(write_bundle(tmp_path / "b"), effective)
    assert loaded.get("kind", "elapsed") == "elapsed"
    rows, _ = T.trainable_rows(loaded["fit"])
    scales = T.fit_scales(rows)
    x, y, mask = T.materialize_monitor_inputs(rows, scales)
    assert isinstance(x, np.ndarray) and x.shape[1:] == (13, 19)   # single tensor, unchanged
    bundle = M.build_dmo(effective["model"])
    assert len(bundle.forecast_model.inputs) == 1
    assert T.MONITOR_TOLERANCE["rtol"] == 5e-5 and T.MONITOR_TOLERANCE["atol"] == 1e-9
