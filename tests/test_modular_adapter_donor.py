"""Typed TRAIN-only NPZ adapter output as an engine donor (lane A2, order 2026-10-03 §3).

The fixture under tests/fixtures/a2_adapter_6fd7601 was emitted by the REAL adapter
(feature-extractor 6fd7601, tools/a2_make_adapter_fixture.py); nothing here re-implements it.
Checks: the converted donor passes the engine's OPERATIONAL contract; R0 ignores donors; R1 keeps the
donor weights bit-identical through a fit step; R2 updates them; a donor whose manifest row-id digest
is not the TRAIN split's is refused, as is a non-TRAIN NPZ; the carried trunk is the adapter's trunk
lagged two steps (causal).
"""
import json
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from predictor_plugins.modular_temporal import adapter_donor as ad

FIXTURE = Path(__file__).parent / "fixtures" / "a2_adapter_6fd7601"
FEATURES = ["a", "b", "c"]


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(2021)
    yield


def config(regime="R0", donor=None):
    c = mt.default_config(FEATURES)
    c["branches"] = [{"name": "branch_0", "features": FEATURES, "plugin": "npz_adapter_conv",
                      "params": {"filters": 8, "kernel_size": 3}, "regime": regime, "donor": donor}]
    c["core"]["regime"], c["core"]["donor"] = "R0", None
    return c


def convert(tmp_path):
    receipt = ad.convert_adapter_donor(FIXTURE / "adapter", FIXTURE / "train.npz", config(), "branch_0",
                                       tmp_path / "donor.keras")
    return tmp_path / "donor.keras", receipt


def donor_bytes(path):
    return [w.tobytes() for w in tf.keras.models.load_model(path, compile=False).get_weights()]


def branch_bytes(bundle):
    return [w.tobytes() for w in bundle.branch_models["branch_0"].get_weights()]


def fit_step(bundle, epochs=2):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(32, 24, 3)).astype("float32")
    y = rng.normal(size=(32,) + tuple(bundle.forecast_model.output_shape[1:])).astype("float32")
    model = bundle.forecast_model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mae")
    model.fit(x, y, epochs=epochs, batch_size=16, verbose=0)


def test_converted_donor_satisfies_the_operational_contract(tmp_path):
    path, receipt = convert(tmp_path)
    side = json.loads(path.with_suffix(".manifest.json").read_text())
    assert side["schema"] == 2 and side["provenance"]["conditioning_contract"] == "OPERATIONAL"
    corpus = side["provenance"]["learned_corpus"]
    assert corpus["kind"] == "TRAIN_ONLY" and corpus["data_sha256"] == receipt["train_split"]["sha256"]
    assert receipt["dropped"] == ["Flatten", "Dense(latent)"] and receipt["padding"]["lag_steps"] == 2
    bundle = mt.build_modular(config("R1", str(path)))          # engine enforces OPERATIONAL by default
    assert branch_bytes(bundle) == donor_bytes(path)


def test_carried_trunk_is_the_adapter_trunk_lagged_two_steps(tmp_path):
    path, _ = convert(tmp_path)
    encoder = tf.keras.models.load_model(FIXTURE / "adapter" / "encoder.keras", compile=False)
    convs = [l for l in encoder.layers if isinstance(l, tf.keras.layers.Conv1D)]
    x = np.random.default_rng(3).normal(size=(4, 24, 3)).astype("float32")
    same = x
    for conv in convs:
        same = conv(same)
    same = np.asarray(same)
    causal = np.asarray(tf.keras.models.load_model(path, compile=False)(x))
    np.testing.assert_allclose(causal[:, 4:], same[:, 2:-2], rtol=1e-5, atol=1e-5)
    mt.probe_alignment(tf.keras.models.load_model(path, compile=False), tuple(range(1, 25)),
                       tuple(range(1, 25)), label="adapter branch")


def test_r0_ignores_donors(tmp_path):
    path, _ = convert(tmp_path)
    tf.keras.utils.set_random_seed(2021)
    fresh = branch_bytes(mt.build_modular(config("R0")))
    assert fresh != donor_bytes(path)                       # R0 does not read the donor beside it
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(2021)
    assert branch_bytes(mt.build_modular(config("R0"))) == fresh
    with pytest.raises(ValueError, match="R0 forbids donors"):
        mt.build_modular(config("R0", str(path)))             # and a donor cannot be smuggled into R0


def test_r1_keeps_donor_weights_bit_identical_after_a_fit_step(tmp_path):
    path, _ = convert(tmp_path)
    bundle = mt.build_modular(config("R1", str(path)))
    core_before = [w.tobytes() for w in bundle.core_model.get_weights()]
    fit_step(bundle)
    assert branch_bytes(bundle) == donor_bytes(path)
    assert [w.tobytes() for w in bundle.core_model.get_weights()] != core_before   # the fit did run


def test_r2_updates_donor_weights(tmp_path):
    path, _ = convert(tmp_path)
    bundle = mt.build_modular(config("R2", str(path)))
    assert branch_bytes(bundle) == donor_bytes(path)
    fit_step(bundle)
    assert branch_bytes(bundle) != donor_bytes(path)


def test_donor_whose_row_ids_differ_from_the_train_split_is_refused(tmp_path):
    with np.load(FIXTURE / "train.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["row_ids"] = np.array([f"fx:row{i + 1}" for i in range(len(arrays["row_ids"]))])
    np.savez(tmp_path / "other_train.npz", **arrays)
    with pytest.raises(ValueError, match="ADAPTER_ROWS_NOT_TRAIN_SPLIT"):
        ad.convert_adapter_donor(FIXTURE / "adapter", tmp_path / "other_train.npz", config(), "branch_0",
                                 tmp_path / "donor.keras")
    assert not (tmp_path / "donor.keras").exists()


def test_non_train_split_and_tampered_adapter_are_refused(tmp_path):
    with np.load(FIXTURE / "train.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["split"] = np.array("validation")
    np.savez(tmp_path / "val.npz", **arrays)
    with pytest.raises(ValueError, match="ADAPTER_SPLIT_NOT_TRAIN"):
        ad.convert_adapter_donor(FIXTURE / "adapter", tmp_path / "val.npz", config(), "branch_0",
                                 tmp_path / "d.keras")
    shutil.copytree(FIXTURE / "adapter", tmp_path / "adapter")
    manifest = json.loads((tmp_path / "adapter" / "manifest.json").read_text())
    manifest["weight_sha256"] = "0" * 64
    (tmp_path / "adapter" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ADAPTER_WEIGHTS_MISMATCH"):
        ad.convert_adapter_donor(tmp_path / "adapter", FIXTURE / "train.npz", config(), "branch_0",
                                 tmp_path / "d.keras")
