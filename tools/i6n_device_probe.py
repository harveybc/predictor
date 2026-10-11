"""Finite GPU integration smoke: synthetic tensors, no market-data training.

Run inside the existing resource-admission launcher, with exactly one GPU
visible. Six updates check framework placement, masked gradients and saved
model parity. These measurements are NOT a production memory envelope, a
cost pilot on historical TRAIN, or predictive-quality evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def run(output):
    import numpy as np
    import tensorflow as tf
    from tools.i6m_queue import atomic_json
    from tools.i6n_model import DEFAULT_CONFIG, build_model
    from tools.i6n_contracts.temporal_ops import masked_huber

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "RESULT.json"
    if terminal.exists():
        raise ValueError("terminal exists; do not repeat this probe")
    with (output / "CLAIM").open("x") as claim:
        claim.write(str(os.getpid()))
    state = dict(schema="predictor.i6n.gpu_smoke.v1", state="RUNNING",
                 seed=6102026, historical_data_read=False, test_read=False,
                 scientific_measurement=False, production_cost_evidence=False)
    atomic_json(output / "STATUS.json", state)
    started = time.monotonic()
    try:
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise ValueError("exactly one visible TensorFlow GPU required; no CPU fallback")
        tf.config.experimental.set_memory_growth(devices[0], True)
        tf.keras.utils.set_random_seed(state["seed"])
        tf.config.experimental.enable_op_determinism()
        with tf.device("/GPU:0"):
            operation = tf.linalg.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
        if "GPU:0" not in operation.device:
            raise ValueError("GPU operation placement not verified")
        configs = [dict(DEFAULT_CONFIG), dict(DEFAULT_CONFIG, entrance="transformer",
                   transformer_blocks=1, reducer_filters=[8, 4], reducer_strides=[2, 2])]
        rows = []
        for index, config in enumerate(configs):
            tf.keras.backend.clear_session()
            config["n_horizons"] = 12
            with tf.device("/GPU:0"):
                model = build_model(55, 20, 8, config)
                rng = np.random.default_rng(state["seed"])
                x = rng.normal(size=(4, 55, 41)).astype("float32")
                x[:, :, 1::2] = 1
                x[:, :, -1] = 1
                x[:, 10:13] = 0  # masked clock gaps, not prices
                targets = tf.constant(rng.normal(size=(4, 12)), dtype=tf.float32)
                mask = tf.constant(rng.random((4, 12)) > .25)
                targets = tf.where(mask, targets, float("nan"))
                optimizer = tf.keras.optimizers.Adam(1e-3)
                before = [w.numpy().copy() for w in model.weights]
                for _ in range(6):
                    with tf.GradientTape() as tape:
                        prediction = model(x, training=True)
                        loss = masked_huber(targets, prediction, mask, [1.] * 12)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    if any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g)))
                           for g in gradients):
                        raise ValueError("missing/nonfinite training gradients")
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if "GPU:0" not in prediction.device:
                    raise ValueError("model output was not on GPU")
            if not any(not np.array_equal(a, b.numpy()) for a, b in zip(before, model.weights)):
                raise ValueError("updates did not change model weights")
            expected = model(x, training=False).numpy()
            path = output / f"model_{index}.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            actual = restored(x, training=False).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            with path.open("rb") as stream:
                artifact_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            rows.append(dict(config=config, parameters=model.count_params(), updates=6,
                             output_device=prediction.device, model_sha256=artifact_hash,
                             max_reload_difference=float(np.max(np.abs(actual - expected)))))
            state.update(completed_models=len(rows), total_models=len(configs))
            atomic_json(output / "STATUS.json", state)
        result = dict(state, state="COMPLETED", models=rows,
                      elapsed_seconds=time.monotonic() - started,
                      tensorflow_version=tf.__version__, gpu_operation_device=operation.device,
                      device_details=tf.config.experimental.get_device_details(devices[0]))
        atomic_json(terminal, result)
        atomic_json(output / "STATUS.json", result)
        return result
    except Exception as exc:
        atomic_json(output / "STATUS.json", dict(state, state="FAILED", error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
