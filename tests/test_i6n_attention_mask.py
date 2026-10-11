"""CPU mask regression and opt-in hard-placement GPU probe, synthetic only.

GPU: I6N_GPU_MASK_PROBE=1 I6N_GPU_PROBE_OUTPUT=<new-absolute-directory>
run pytest on this file with -k gpu under explicit GPU resource admission.
"""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import pytest


@pytest.mark.parametrize("length", [1, 4, 13])
def test_cpu_attention_exact_legacy_parity_without_band_part(length):
    import tensorflow as tf
    from tools.i6e_masked_model import ClockView
    from tools.i6n_contracts.temporal_ops import I6NAttentionMask
    presence = np.random.default_rng(81).random((3, length, 1)) > .3
    presence[1] = False
    with tf.device("/CPU:0"):
        layer = I6NAttentionMask()
        traced = tf.function(layer, input_signature=[tf.TensorSpec((None, None, 1), tf.bool)])
        actual = traced(presence)
        expected = ClockView("attention")(tf.constant(presence))
    np.testing.assert_array_equal(actual, expected)
    positions = np.arange(length)
    reference = presence[:, :, 0, None] & presence[:, None, :, 0] & (positions[:, None] >= positions[None, :])
    np.testing.assert_array_equal(actual, reference)
    assert actual.dtype == tf.bool
    ops = {op.type for op in traced.get_concrete_function().graph.get_operations()}
    assert "MatrixBandPart" not in ops and "GreaterEqual" in ops


def test_cpu_attention_roundtrip_and_future_prefix(tmp_path):
    import tensorflow as tf
    from tools.i6n_contracts.temporal_ops import I6NAttentionMask
    with tf.device("/CPU:0"):
        inp = tf.keras.Input((None, 1), dtype="bool")
        model = tf.keras.Model(inp, I6NAttentionMask()(inp))
        mask = np.array([[[True], [False], [True], [True]]])
        before = model(mask).numpy()
        extended = np.concatenate([mask, np.ones((1, 3, 1), bool)], axis=1)
        np.testing.assert_array_equal(model(extended)[:, :4, :4], before)
        model.save(tmp_path / "mask.keras")
        restored = tf.keras.models.load_model(tmp_path / "mask.keras", compile=False)
        np.testing.assert_array_equal(restored(mask), before)


@pytest.mark.parametrize("entrance", ["transformer", "conv_transformer"])
def test_cpu_factory_uses_additive_mask_only(entrance):
    import tensorflow as tf
    from tools.i6e_masked_model import ClockView
    from tools.i6n_contracts.temporal_ops import I6NAttentionMask
    from tools.i6n_model import DEFAULT_CONFIG, build_model
    with tf.device("/CPU:0"):
        model = build_model(5, 2, 2, dict(DEFAULT_CONFIG, entrance=entrance, transformer_blocks=1))
    assert isinstance(model.get_layer("causal_market_attention"), I6NAttentionMask)
    assert isinstance(model.get_layer("market_presence"), ClockView)
    assert not any(isinstance(layer, ClockView) and layer.mode == "attention" for layer in model.layers)


@pytest.fixture(scope="module")
def hard_gpu():
    import tensorflow as tf
    devices = tf.config.list_physical_devices("GPU")
    assert len(devices) == 1, "exactly one explicitly visible GPU required; no CPU fallback"
    tf.config.experimental.set_memory_growth(devices[0], True)
    previous = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_op_determinism()
    yield tf
    tf.config.set_soft_device_placement(previous)


@pytest.mark.skipif(os.environ.get("I6N_GPU_MASK_PROBE") != "1", reason="explicit admitted GPU probe only")
@pytest.mark.parametrize("entrance,blocks", [("conv", 0), ("transformer", 1), ("transformer", 2),
                                            ("conv_transformer", 1), ("conv_transformer", 2)])
@pytest.mark.parametrize("adapter", ["none", "causalconv", "lstm"])
def test_gpu_hard_placement_six_updates_all_variants(hard_gpu, entrance, blocks, adapter):
    tf = hard_gpu
    from tools.i6m_queue import atomic_json
    from tools.i6n_model import DEFAULT_CONFIG, build_model
    from tools.i6n_contracts.temporal_ops import masked_huber
    root = Path(os.environ["I6N_GPU_PROBE_OUTPUT"])
    assert root.is_absolute(), "retained absolute probe output required"
    root = root / f"{entrance}-blocks{blocks}-{adapter}"
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    pins = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in
            ("tools/i6n_model.py", "tools/i6n_contracts/temporal_ops.py", "tools/i6e_masked_model.py", "tests/test_i6n_attention_mask.py")}
    state = dict(state="RUNNING", scope="SYNTHETIC_GPU_MASK_REGRESSION", historical_data_read=False,
        test_data_read=False, production_cost_evidence=False, source_sha256=pins,
        started_at=datetime.now(timezone.utc).isoformat(), updates=0, soft_placement=False)
    atomic_json(root / "STATUS.json", state)
    started = time.monotonic()
    try:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(6102026)
        cfg = dict(DEFAULT_CONFIG, entrance=entrance, transformer_blocks=blocks,
                   adapter=adapter, adapter_width=None if adapter == "none" else 8,
                   reducer_strides=[2, 1])
        rng = np.random.default_rng(6102026)
        x = rng.normal(size=(4, 13, 41)).astype("float32")
        x[:, :, 1::2] = 1
        x[:, :, -1] = 1
        x[:, [2, 5, 9], -1] = 0
        x[3, :, -1] = 0
        x[3, :, :-1] = np.nan
        with tf.device("/GPU:0"):
            model = build_model(13, 20, 2, cfg)
            assert all(v.value.handle.device.endswith("/device:GPU:0") for v in model.trainable_variables)
            optimizer = tf.keras.optimizers.Adam(1e-3)
            targets = tf.constant([[.1], [float("nan")], [-.2], [float("nan")]])
            valid = tf.constant([[True], [False], [True], [False]])
            before = [v.numpy().copy() for v in model.trainable_variables]
            for update in range(6):
                with tf.GradientTape() as tape:
                    prediction = model(x, training=True)
                    loss = masked_huber(targets, prediction, valid, [1.])
                gradients = tape.gradient(loss, model.trainable_variables)
                assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
                assert prediction.device.endswith("/device:GPU:0") and loss.device.endswith("/device:GPU:0")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                state.update(updates=update+1, loss=float(loss.numpy()), elapsed_seconds=time.monotonic()-started)
                atomic_json(root / "STATUS.json", state)
            assert any(not np.array_equal(a, v.numpy()) for a, v in zip(before, model.trainable_variables))
            expected = model(x, training=False)
            model.save(root / "model.keras")
            restored = tf.keras.models.load_model(root / "model.keras", compile=False)
            actual = restored(x, training=False)
            assert actual.device.endswith("/device:GPU:0")
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        state.update(state="COMPLETED", config=cfg, samples=13, features=20,
            output_device=actual.device, tensorflow_version=tf.__version__,
            model_sha256=hashlib.sha256((root / "model.keras").read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic()-started)
        atomic_json(root / "RESULT.json", state)
        atomic_json(root / "STATUS.json", state)
    except BaseException as exc:
        state.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        atomic_json(root / "STATUS.json", state)
        raise
