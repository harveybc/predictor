"""Lane H: flatten + MLP control learner on CPU, selection on an inner chronological hold-out of TRAIN only.

Same architecture family and budget as lane F2's control (hidden widths, ReLU, MAE loss, AdamW, batch 64, learning rate
1e-3, at most 30 epochs, patience 5) but early stopping reads a purged inner hold-out of the TRAIN rows, never the
validation rows. Inputs are standardized with TRAIN statistics. TensorFlow is imported lazily; CPU only.
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np


def _check(X, name):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"{name} must be a finite 2-D array")
    return X


def mlp_fit_predict(Xtr, Ytr, Xev_list, seed, hidden=(44, 44), max_epochs=30, patience=5, batch_size=64,
                    learning_rate=1e-3, weight_decay=1e-4, holdout_frac=0.2, purge=30):
    Xtr, Ytr = _check(Xtr, "Xtr"), _check(Ytr, "Ytr")
    if Xtr.shape[0] != Ytr.shape[0]:
        raise ValueError("Xtr and Ytr rows differ")
    evs = [_check(X, "Xev") for X in Xev_list]
    if any(X.shape[1] != Xtr.shape[1] for X in evs):
        raise ValueError("evaluation width differs from the training width")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    N = Xtr.shape[0]
    cut = int(N * (1.0 - holdout_frac))
    mu, sd = Xtr[:cut - purge].mean(axis=0), Xtr[:cut - purge].std(axis=0)
    sd[sd == 0] = 1.0
    def prep(X):
        return ((X - mu) / sd).astype(np.float32)
    xin, yin = prep(Xtr[:cut - purge]), Ytr[:cut - purge].astype(np.float32)
    xho, yho = prep(Xtr[cut:]), Ytr[cut:].astype(np.float32)
    inputs = tf.keras.Input(shape=(Xtr.shape[1],))
    x = inputs
    for i, h in enumerate(hidden):
        x = tf.keras.layers.Dense(int(h), activation="relu", name=f"hidden_{i}")(x)
    outputs = tf.keras.layers.Dense(Ytr.shape[1], name="forecast")(x)
    model = tf.keras.Model(inputs, outputs)
    opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=opt, loss="mae")
    best, best_w, best_epoch, bad = float("inf"), None, 0, 0
    run = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.fit(xin, yin, batch_size=batch_size, epochs=1, shuffle=True, verbose=0)
        run = epoch
        loss = float(np.mean(np.abs(model.predict(xho, batch_size=1024, verbose=0) - yho)))
        if loss < best - 1e-9:
            best, best_w, best_epoch, bad = loss, model.get_weights(), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.set_weights(best_w)
    h = hashlib.sha256()
    for w in best_w:
        h.update(np.ascontiguousarray(w, dtype="<f4").tobytes())
    preds = [model.predict(prep(X), batch_size=1024, verbose=0).astype(np.float64) for X in evs]
    return {"predictions": preds, "selected_epoch": int(best_epoch), "epochs_run": int(run), "inner_best_loss": best,
            "weights_sha256": h.hexdigest(), "selection": "inner_chronological_holdout_of_train",
            "hidden": list(hidden), "trainable_parameters": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))}
