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
    X = np.asarray(X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"{name} must be a finite 2-D array")
    return X


def mlp_fit_predict(Xtr, Ytr, Xev_list, seed, hidden=(44, 44), max_epochs=30, patience=5, batch_size=64,
                    learning_rate=1e-3, weight_decay=1e-4, holdout_frac=0.2, purge=30):
    Xtr, Ytr = _check(Xtr, "Xtr"), _check(Ytr, "Ytr")
    if Xtr.shape[0] != Ytr.shape[0]:
        raise ValueError("Xtr and Ytr rows differ")
    # an evaluation element may be a zero-argument callable that builds its array on demand (memory: wide validation sets)
    evs = [X if callable(X) else _check(X, "Xev") for X in Xev_list]
    if any((not callable(X)) and X.shape[1] != Xtr.shape[1] for X in evs):
        raise ValueError("evaluation width differs from the training width")
    import tensorflow as tf

    tf.keras.backend.clear_session()          # free the graphs of earlier fits in the same process
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    N = Xtr.shape[0]
    cut = int(N * (1.0 - holdout_frac))
    mu, sd = _col_stats(Xtr[:cut - purge])
    sd[sd == 0] = 1.0
    def prep(X, block=2048):
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, X.shape[0], block):
            out[i:i + block] = ((X[i:i + block] - mu) / sd).astype(np.float32)
        return out
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
    # one fit call (a fit call per epoch re-builds the data adapter every epoch and grew memory on the ECL run);
    # the inner hold-out is Keras validation data, the loss is MAE, the best weights are kept by the callback below
    class _Keep(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_w, self.best_epoch, self.bad = float("inf"), None, 0, 0

        def on_epoch_end(self, epoch, logs=None):
            loss = float(logs["val_loss"])
            if loss < self.best - 1e-9:
                self.best, self.best_w, self.best_epoch, self.bad = loss, self.model.get_weights(), epoch + 1, 0
            else:
                self.bad += 1
                if self.bad >= patience:
                    self.model.stop_training = True

    keep = _Keep()
    hist = model.fit(xin, yin, batch_size=batch_size, epochs=int(max_epochs), shuffle=True, verbose=0,
                     validation_data=(xho, yho), validation_batch_size=4096, callbacks=[keep])
    best, best_w, best_epoch, run = keep.best, keep.best_w, keep.best_epoch, len(hist.history["val_loss"])
    model.set_weights(best_w)
    h = hashlib.sha256()
    for w in best_w:
        h.update(np.ascontiguousarray(w, dtype="<f4").tobytes())
    preds = []
    for X in evs:
        if callable(X):
            X = _check(X(), "Xev")
            if X.shape[1] != Xtr.shape[1]:
                raise ValueError("evaluation width differs from the training width")
        preds.append(model.predict(prep(X), batch_size=1024, verbose=0).astype(np.float64))
        del X
    return {"predictions": preds, "selected_epoch": int(best_epoch), "epochs_run": int(run), "inner_best_loss": best,
            "weights_sha256": h.hexdigest(), "selection": "inner_chronological_holdout_of_train",
            "hidden": list(hidden), "trainable_parameters": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))}


def _col_stats(X, block=2048):
    n = X.shape[0]
    s = np.zeros(X.shape[1])
    for i in range(0, n, block):
        s += X[i:i + block].sum(axis=0, dtype=np.float64)
    mu = s / n
    v = np.zeros(X.shape[1])
    for i in range(0, n, block):
        d = X[i:i + block].astype(np.float64) - mu
        v += (d * d).sum(axis=0)
    sd = np.sqrt(v / n)
    sd[sd == 0] = 1.0
    return mu, sd
