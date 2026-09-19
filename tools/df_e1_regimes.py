#!/usr/bin/env python3
"""RP29: the detector regimes of the proposal (Table `tab:regimenes`) on the modular model.

  R0  from scratch: detector and the rest of the model start from a random initialisation and train.
  R1  pre-trained and frozen: the detector's weights are imported from the pre-training and kept fixed
      (trainable weights AND non-trainable states, e.g. normalisation statistics: none exist in the
      Conv1D detector, and that is asserted); the rest trains.
  R2  pre-trained and adjustable: the SAME initial detector weights as R1 are imported and adjusted by
      the full model's objective (gradients reach the detector and updates are observed).

Grouping, fusion, readout, architecture and preprocessing are OTHER factors and are held constant across
the three regimes; ARCH-0 is an architecture control, never the definition of R0. The detector is the
per-branch detector stack (`g{k}_det*` layers); the integrator, adapter, fusion, core and head are never
frozen by module name.

Pre-training (as in the proposal): a masked autoencoder reconstructs the preprocessed inputs of the
TRAIN windows only (mask ratio declared; the masked positions are zeroed and a mask channel is NOT given
to the encoder), encoder = the detector stack of every branch, decoder = a separate 1x1 convolution stack
saved apart; loss = mse on the masked positions (declared); early stopping on the internal validation
subset of TRAIN windows; the test split and the future targets never enter. The reconstruction error is
a diagnostic only, never a substitute for the forecasting score.

The regimes share one initial checkpoint: `initial_checkpoint(seed)` builds the full model once, saves
its weights; R1/R2 then load the pre-trained detector into a copy of that checkpoint; R0 keeps it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("df_mod_e0")
REGIMES = {"R0": "from scratch: detector trainable from the shared random initial checkpoint",
           "R1": "pre-trained and frozen: detector weights imported and fixed; rest trainable",
           "R2": "pre-trained and adjustable: the same imported detector weights, adjusted with the full model"}
DETECTOR_TAGS = ("_det",)


def detector_layer_names(model) -> list:
    return [l.name for l in model.layers if any(t in l.name for t in DETECTOR_TAGS) and l.weights]


def non_detector_weighted_layer_names(model) -> list:
    return [l.name for l in model.layers if l.weights and not any(t in l.name for t in DETECTOR_TAGS)]


def weights_digest(model, names: list) -> str:
    h = hashlib.sha256()
    for n in names:
        for w in model.get_layer(n).get_weights():
            h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


def initial_checkpoint(assignment: list, window: int, p: int, *, arch: str, fusion: str, seed: int, out_dir: Path) -> dict:
    """The shared initial state of the FULL model for one replicate (one seed): saved once, reused by every regime."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = E.build_modular(assignment, window, p, fusion=fusion, seed=seed, arch=arch)
    path = out_dir / f"initial_seed{seed}.weights.h5"
    model.save_weights(str(path))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": seed,
            "detector_layers": detector_layer_names(model), "detector_digest": weights_digest(model, detector_layer_names(model)),
            "params": E.count_params(model)}


def build_autoencoder(assignment: list, window: int, p: int, *, arch: str, seed: int, mask_ratio: float):
    """Encoder = the branch detectors (same layer names as the modular model, so weights transfer by name);
    decoder = per branch a separate 1x1 Conv1D stack back to the branch's channels. Returns (ae, decoder_names)."""
    tf = E._tf()
    tf.keras.utils.set_random_seed(int(seed))
    inp = tf.keras.Input(shape=(window, p), name="x")
    groups = sorted(set(assignment))
    outs, dec_names = [], []
    for g in groups:
        idx = [k for k in range(p) if assignment[k] == g]
        sub = tf.keras.layers.Lambda(lambda t, idx=idx: tf.gather(t, idx, axis=2), name=f"g{g}_select")(inp)
        h = E._tcn_block(tf, sub, f"g{g}_det1", 1)
        h = E._tcn_block(tf, h, f"g{g}_det2", 1)
        d = tf.keras.layers.Conv1D(16, 1, activation=E.ACTIVATION, name=f"g{g}_dec1")(h)
        d = tf.keras.layers.Conv1D(len(idx), 1, name=f"g{g}_dec2")(d)
        dec_names += [f"g{g}_dec1", f"g{g}_dec2"]
        outs.append(d)
    # reassemble the branches into the input order
    order = [k for g in groups for k in range(p) if assignment[k] == g]
    inv = np.argsort(order)
    cat = tf.keras.layers.Concatenate(axis=2, name="dec_concat")(outs) if len(outs) > 1 else outs[0]
    rec = tf.keras.layers.Lambda(lambda t: tf.gather(t, inv.tolist(), axis=2), name="dec_reorder")(cat)
    return tf.keras.Model(inp, rec, name=f"ae_{arch}"), dec_names


def masked_pretrain(assignment: list, window: int, p: int, *, arch: str, seed: int, Xtr: np.ndarray, Xval: np.ndarray, out_dir: Path,
                    mask_ratio: float = 0.3, max_updates: int = 600, patience: int = 5, batch: int = 64, learning_rate: float = 3e-3) -> dict:
    """Masked reconstruction on TRAIN windows (internal validation = Xval, a subset of train windows chosen
    by the caller BEFORE any score). Saves encoder (detector) weights and decoder weights apart."""
    tf = E._tf()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ae, dec_names = build_autoencoder(assignment, window, p, arch=arch, seed=seed, mask_ratio=mask_ratio)

    def mask(X):
        m = rng.random(X.shape) < mask_ratio                      # masked POSITIONS (time x channel), zeroed; the encoder gets no mask channel
        return np.where(m, 0.0, X).astype(np.float32), m
    Xm, Mtr = mask(Xtr)
    Xvm, Mva = mask(Xval)
    # target = [x, mask] on the channel axis; the loss averages the squared error over MASKED positions only
    Ytr = np.concatenate([Xtr.astype(np.float32), Mtr.astype(np.float32)], axis=2)
    Yva = np.concatenate([Xval.astype(np.float32), Mva.astype(np.float32)], axis=2)

    def masked_mse(y_true, y_pred):
        x, m = y_true[..., :p], y_true[..., p:]
        return tf.reduce_sum(m * tf.square(x - y_pred)) / (tf.reduce_sum(m) + 1e-8)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=masked_mse)
    steps = math.ceil(Xtr.shape[0] / batch)
    max_epochs = max(1, math.ceil(max_updates / steps))
    counter = {"updates": 0}

    class Stop(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, b, logs=None):
            counter["updates"] += 1
            if counter["updates"] >= max_updates:
                self.model.stop_training = True
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    t0 = time.process_time()
    hist = ae.fit(Xm, Ytr, validation_data=(Xvm, Yva), epochs=max_epochs, batch_size=batch,
                  shuffle=True, verbose=0, callbacks=[Stop(), es])
    cpu = time.process_time() - t0
    enc_names = detector_layer_names(ae)
    enc_path, dec_path = out_dir / f"detector_pretrained_seed{seed}.weights.h5", out_dir / f"decoder_seed{seed}.weights.h5"
    ae.save_weights(str(out_dir / f"ae_seed{seed}.weights.h5"))
    # separate files: detector layers only, decoder layers only (by name, via two throwaway models)
    det_weights = {n: ae.get_layer(n).get_weights() for n in enc_names}
    dec_weights = {n: ae.get_layer(n).get_weights() for n in dec_names}
    np.savez(out_dir / f"detector_pretrained_seed{seed}.npz", **{f"{n}__{i}": w for n, ws in det_weights.items() for i, w in enumerate(ws)})
    np.savez(out_dir / f"decoder_seed{seed}.npz", **{f"{n}__{i}": w for n, ws in dec_weights.items() for i, w in enumerate(ws)})
    va = [float(v) for v in hist.history["val_loss"]]
    rec_val = float(ae.evaluate(Xvm, Yva, verbose=0))
    return {"schema": "df_e1_pretrain.v1", "arch": arch, "seed": seed, "mask_ratio": mask_ratio, "loss": "mse on masked positions (mask carried in the target tensor; unmasked positions weigh zero)",
            "decoder": "separate 1x1 Conv1D stack per branch, saved apart, never connected at inference", "normalisation": "inputs already scaled train-only by the loader",
            "what_the_decoder_sees": "the detector's output of the masked input; it reconstructs the unmasked preprocessed input",
            "updates": counter["updates"], "epochs": len(va), "curve": {"train": [float(v) for v in hist.history["loss"]], "validation": va},
            "best_epoch": int(np.argmin(va)) + 1, "reconstruction_val_mse_masked": rec_val, "cpu_seconds": round(cpu, 3),
            "detector_layers": enc_names, "detector_file": str(out_dir / f"detector_pretrained_seed{seed}.npz"), "decoder_file": str(out_dir / f"decoder_seed{seed}.npz"),
            "detector_digest": weights_digest(ae, enc_names), "diagnostic_only": True}


def load_detector(model, npz_path: Path) -> list:
    z = np.load(npz_path)
    names = sorted({k.split("__")[0] for k in z.files})
    for n in names:
        ws = [z[f"{n}__{i}"] for i in range(len([k for k in z.files if k.startswith(n + "__")]))]
        model.get_layer(n).set_weights(ws)
    return names


def apply_regime(model, regime: str, pretrained_npz: Path | None) -> dict:
    """Configure the model for a regime; returns what was imported and what is trainable, with digests."""
    det = detector_layer_names(model)
    info = {"regime": regime, "description": REGIMES[regime], "detector_layers": det, "imported": [], "frozen": [], "trainable": []}
    if regime in ("R1", "R2"):
        if pretrained_npz is None:
            raise ValueError(f"{regime} needs the pre-trained detector")
        info["imported"] = load_detector(model, pretrained_npz)
        if sorted(info["imported"]) != sorted(det):
            raise ValueError(f"pre-trained layers {info['imported']} are not the detector layers {det}")
    for l in model.layers:
        if not l.weights:
            continue
        l.trainable = not (regime == "R1" and l.name in det)
        (info["frozen"] if not l.trainable else info["trainable"]).append(l.name)
    info["detector_digest_after_setup"] = weights_digest(model, det)
    info["non_trainable_states_in_detector"] = int(sum(len([w for w in model.get_layer(n).non_trainable_weights]) for n in det if model.get_layer(n).trainable))
    info["params"] = E.count_params(model)
    return info


def gradient_report(model, Xb: np.ndarray, yb: np.ndarray, loss: str = "mse") -> dict:
    """Which trainable variables receive a non-zero gradient of the final objective on one batch."""
    tf = E._tf()
    loss_fn = tf.keras.losses.MeanSquaredError() if loss == "mse" else tf.keras.losses.MeanAbsoluteError()
    with tf.GradientTape() as tape:
        pred = model(tf.constant(Xb, dtype=tf.float32), training=True)
        value = loss_fn(tf.constant(yb, dtype=tf.float32), pred)
    grads = tape.gradient(value, model.trainable_variables)
    per = {}
    for v, g in zip(model.trainable_variables, grads):
        per[v.path if hasattr(v, "path") else v.name] = None if g is None else float(np.linalg.norm(np.asarray(g).ravel()))
    det = detector_layer_names(model)
    det_vars = [k for k in per if any(k.startswith(n + "/") or k.startswith(n + "_") or n in k for n in det)]
    return {"loss_on_batch": float(value), "per_variable": per, "detector_variables_with_gradient": [k for k in det_vars if per[k]],
            "detector_receives_gradient": any(per[k] for k in det_vars), "n_trainable_variables": len(per)}
