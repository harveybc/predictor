"""Branch training objectives as versioned components, separate from the architecture.

PS3-R (subplan section 6, FS17). An objective trains an EXISTING branch encoder
in place; it never changes the encoder's architecture, so the branch donor
manifest is the same whatever objective produced the weights. The objective has
its own entry-point group (``modular.objective``), version, effective
parameters and identity, which ``save_donor(..., objective=identity)`` records
in the donor's provenance beside the unchanged manifest.

Built-ins:

``autoencoder_reconstruction`` (1.0.0)
    The AE control: a learned decoder reconstructs the branch input; the shared
    early-stopping loop fits it. Reconstruction is MEASURED.
``ts2vec_contrastive`` (1.0.0)
    TS2Vec-style objective (Yue et al., AAAI 2022, arXiv:2106.10466; reference
    code zhihanyue/ts2vec @ b0088e14): hierarchical instance-wise + temporal
    contrastive loss over max-pooled scales, with contextual consistency from
    two random crops and independent timestamp masking. Decoder-free:
    reconstruction is NOT_APPLICABLE. Declared deviations from the reference:
    (1) the branch input is fixed at the 24-step grid, so a crop is applied by
    zeroing the history before the crop start (the causal encoder then sees a
    different context) and the loss is computed on the overlapping suffix,
    instead of feeding shorter sequences; (2) timestamp masking zeroes INPUT
    steps, because the branch is an external component without an exposed
    input projection to mask. Crops never cross a fold boundary: the caller
    passes windows from one fold only.

Every fit has max_epochs / patience / min_delta / max_updates, restores the
best validation checkpoint and reports observed optimizer updates.
"""
from __future__ import annotations

import json
import math
import time
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import tensorflow as tf

from .common import _copy, _digest

keras = tf.keras
GROUP = "modular.objective"


def objective(name, version, parameters, contract, defaults):
    def mark(cls):
        cls.objective_name, cls.objective_version = name, version
        cls.objective_parameters = tuple(sorted(parameters))
        cls.objective_contract, cls.objective_defaults = contract, dict(defaults)
        return cls
    return mark


# --------------------------------------------------------------- TS2Vec loss
def _check(z):
    if len(z.shape) != 3:
        raise ValueError("contrastive loss needs rank-three (batch, time, channels) latents")


def instance_contrastive_loss(z1, z2):
    b = tf.shape(z1)[0]
    z = tf.transpose(tf.concat([z1, z2], axis=0), (1, 0, 2))            # T x 2B x C
    sim = tf.matmul(z, z, transpose_b=True)                              # T x 2B x 2B
    logits = _off_diagonal(sim)
    logits = -tf.nn.log_softmax(logits, axis=-1)
    i = tf.range(b)
    pos_a = tf.gather(tf.gather(logits, i, axis=1), b + i - 1, axis=2, batch_dims=0)
    pos_b = tf.gather(tf.gather(logits, b + i, axis=1), i, axis=2, batch_dims=0)
    return (tf.reduce_mean(tf.linalg.diag_part(pos_a)) + tf.reduce_mean(tf.linalg.diag_part(pos_b))) / 2


def temporal_contrastive_loss(z1, z2):
    t = tf.shape(z1)[1]
    z = tf.concat([z1, z2], axis=1)                                     # B x 2T x C
    sim = tf.matmul(z, z, transpose_b=True)                              # B x 2T x 2T
    logits = -tf.nn.log_softmax(_off_diagonal(sim), axis=-1)
    i = tf.range(t)
    pos_a = tf.gather(tf.gather(logits, i, axis=1), t + i - 1, axis=2)
    pos_b = tf.gather(tf.gather(logits, t + i, axis=1), i, axis=2)
    return (tf.reduce_mean(tf.linalg.diag_part(pos_a)) + tf.reduce_mean(tf.linalg.diag_part(pos_b))) / 2


def _off_diagonal(sim):
    """Drop the self-similarity column of every row: (..., N, N) -> (..., N, N-1), as in TS2Vec."""
    lower = tf.linalg.band_part(sim, -1, 0) - tf.linalg.band_part(sim, 0, 0)
    upper = tf.linalg.band_part(sim, 0, -1) - tf.linalg.band_part(sim, 0, 0)
    return lower[..., :-1] + upper[..., 1:]


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    _check(z1)
    _check(z2)
    loss, depth = 0.0, 0
    while int(z1.shape[1]) > 1:
        if alpha:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if depth >= temporal_unit and 1 - alpha:
            loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        depth += 1
        z1 = tf.nn.max_pool1d(z1, 2, 2, "VALID")
        z2 = tf.nn.max_pool1d(z2, 2, 2, "VALID")
    if int(z1.shape[1]) == 1:
        if alpha:
            loss += alpha * instance_contrastive_loss(z1, z2)
        depth += 1
    return loss / depth


# ------------------------------------------------------------------- objectives
@objective("ts2vec_contrastive", "1.0.0", {"alpha", "mask_probability", "min_overlap", "temporal_unit"},
           "(B,T,F) branch input -> (B,T,C) latent; hierarchical instance+temporal contrastive loss on two "
           "cropped, masked views; decoder-free (reconstruction NOT_APPLICABLE)",
           {"alpha": 0.5, "mask_probability": 0.5, "min_overlap": 8, "temporal_unit": 0})
class TS2VecContrastive:
    def __init__(self, params):
        self.p = params

    def _views(self, x, rng):
        n, t = x.shape[0], x.shape[1]
        overlap = int(rng.integers(self.p["min_overlap"], t + 1))
        start = t - overlap                                     # the shared suffix [start, t)
        a1, a2 = (int(rng.integers(0, start + 1)) for _ in range(2))
        views = []
        for a in (a1, a2):
            v = np.array(x, copy=True)
            v[:, :a] = 0.0                                      # crop: history before a is not seen
            keep = rng.random((n, t, 1)) >= self.p["mask_probability"]
            views.append((v * keep).astype("float32"))
        return views[0], views[1], start

    def _pair_loss(self, encoder, v1, v2, start, training):
        z1 = encoder(v1, training=training)[:, start:]
        z2 = encoder(v2, training=training)[:, start:]
        return hierarchical_contrastive_loss(z1, z2, self.p["alpha"], self.p["temporal_unit"])

    def loss(self, encoder, x, rng, training):
        v1, v2, start = self._views(x, rng)
        return self._pair_loss(encoder, v1, v2, start, training)

    def _compiled(self, encoder, optimizer, variables):
        """One traced graph per overlap start (at most T - min_overlap + 1); same math as eager."""
        @tf.function(reduce_retracing=False)
        def step(v1, v2, start):
            with tf.GradientTape() as tape:
                value = self._pair_loss(encoder, v1, v2, start, True)
            optimizer.apply_gradients(zip(tape.gradient(value, variables), variables))
            return value

        @tf.function(reduce_retracing=False)
        def evaluate(v1, v2, start):
            return self._pair_loss(encoder, v1, v2, start, False)
        return step, evaluate

    def fit(self, encoder, x, vx, settings):
        s = _settings(settings)
        rng = np.random.default_rng(s["seed"])
        optimizer = keras.optimizers.AdamW(learning_rate=s["learning_rate"], weight_decay=s["weight_decay"])
        variables = encoder.trainable_variables
        if not variables:
            raise ValueError("objective fit needs a trainable encoder (an R1-frozen branch cannot be pretrained)")

        step, evaluate = self._compiled(encoder, optimizer, variables)

        def validation_loss():
            vrng = np.random.default_rng(s["seed"] + 10_000)              # same augmentations every epoch
            total = 0.0
            for i in range(0, len(vx), s["batch_size"]):
                part = vx[i:i + s["batch_size"]]
                v1, v2, start = self._views(part, vrng)
                total += float(evaluate(v1, v2, start)) * len(part)
            return total / len(vx)

        initial_val = validation_loss()
        best, best_weights, stale, updates, history = initial_val, None, 0, 0, []
        initial_iterations = int(optimizer.iterations.numpy())
        started, stop = time.monotonic(), "max_epochs"
        for epoch in range(1, s["max_epochs"] + 1):
            order = rng.permutation(len(x))
            train_total, epoch_started, epoch_updates = 0.0, time.monotonic(), 0
            for i in range(0, len(x), s["batch_size"]):
                if updates >= s["max_updates"] or time.monotonic() - started >= s["max_seconds"]:
                    stop = "max_updates" if updates >= s["max_updates"] else "max_seconds"
                    break
                part = x[order[i:i + s["batch_size"]]]
                v1, v2, start = self._views(part, rng)
                value = step(v1, v2, start)
                updates += 1
                epoch_updates += 1
                if not math.isfinite(float(value)):
                    raise ValueError("nonfinite contrastive loss")
                train_total += float(value) * len(part)
            else:
                val = validation_loss()
                history.append({"epoch": epoch, "train_loss": train_total / len(x), "validation_loss": val,
                                "updates": epoch_updates, "seconds": time.monotonic() - epoch_started})
                if val < best - s["min_delta"]:
                    best, stale = val, 0
                    best_weights = [w.copy() for w in encoder.get_weights()]
                else:
                    stale += 1
                if stale >= s["patience"]:
                    stop = "patience"
                    break
                continue
            break
        if best_weights is None:
            raise ValueError("no epoch improved on the untrained validation loss; nothing to restore")
        encoder.set_weights(best_weights)
        iterations = int(optimizer.iterations.numpy())
        if iterations - initial_iterations != updates:
            raise ValueError("optimizer update count changed outside training")
        return {"history": history, "epochs_completed": len(history), "observed_updates": updates,
                "initial_optimizer_iterations": initial_iterations, "optimizer_iterations": iterations,
                "initial_validation_loss": initial_val, "best_validation_loss": best,
                "restored_best_weights": True, "stop_reason": stop,
                "elapsed_seconds": time.monotonic() - started,
                "reconstruction": {"state": "NOT_APPLICABLE"}}


@objective("autoencoder_reconstruction", "1.0.0", {"decoder_channels", "loss"},
           "(B,T,F) branch input -> (B,T,C) latent -> learned decoder -> (B,T,F); reconstruction MEASURED",
           {"decoder_channels": 32, "loss": "mse"})
class AutoencoderReconstruction:
    def __init__(self, params):
        self.p = params

    def fit(self, encoder, x, vx, settings):
        from tools.modular_candidate_evaluator import fit_with_early_stopping
        from .pretraining import build_autoencoder
        s = _settings(settings)
        ae = build_autoencoder(encoder, channels=self.p["decoder_channels"])
        fit = {k: s[k] for k in ("max_epochs", "patience", "batch_size", "learning_rate", "weight_decay",
                                 "min_delta", "seed", "max_updates", "max_seconds")}
        fit["loss"] = self.p["loss"]
        result = fit_with_early_stopping(ae, x, x, vx, vx, fit)
        recon = np.asarray(ae.predict(vx, batch_size=s["batch_size"], verbose=0))
        result = {k: v for k, v in result.items() if k != "settings"}
        result["reconstruction"] = {"state": "MEASURED", "mse_z": float(np.mean((recon - vx) ** 2)),
                                    "mae_z": float(np.mean(np.abs(recon - vx)))}
        return result


BUILTINS = {"ts2vec_contrastive": TS2VecContrastive, "autoencoder_reconstruction": AutoencoderReconstruction}
_SETTINGS = {"max_epochs": 20, "patience": 5, "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
             "min_delta": 0.0, "seed": 42, "max_updates": 100000, "max_seconds": 3600.0}


def _settings(settings):
    unknown = set(settings) - set(_SETTINGS)
    if unknown:
        raise ValueError(f"unknown objective fit settings {sorted(unknown)}")
    s = {**_SETTINGS, **settings}
    for k in ("max_epochs", "patience", "batch_size", "max_updates"):
        if isinstance(s[k], bool) or not isinstance(s[k], int) or s[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    return s


def resolve_objective(spec):
    """Deterministic resolution: built-ins by name (an entry point may only point at the same class)."""
    if not isinstance(spec, dict) or not isinstance(spec.get("plugin"), str):
        raise ValueError("objective spec needs a plugin name")
    name = spec["plugin"]
    matches = list(entry_points(group=GROUP, name=name))
    cls = BUILTINS.get(name)
    if cls is not None:
        if any(ep.load() is not cls for ep in matches):
            raise ValueError(f"entry point {GROUP}:{name} shadows the built-in objective")
        implementation = "predictor_plugins.modular_temporal.objectives:" + cls.__name__
    else:
        if len(matches) != 1:
            raise ValueError(f"expected exactly one objective {GROUP}:{name}; got {len(matches)}")
        cls, implementation = matches[0].load(), matches[0].value
    if getattr(cls, "objective_name", None) != name:
        raise ValueError(f"objective {name} lacks an objective() declaration")
    params = spec.get("params", {})
    if not isinstance(params, dict) or set(params) - set(cls.objective_parameters):
        raise ValueError(f"objective params must be within {list(cls.objective_parameters)}")
    effective = {**cls.objective_defaults, **_copy(params)}
    for key, value in effective.items():
        if isinstance(cls.objective_defaults.get(key), float) and type(value) is int:
            effective[key] = float(value)
    return cls, {"group": GROUP, "name": name, "implementation": implementation,
                 "version": cls.objective_version, "params": effective}


def objective_identity(spec):
    _, identity = resolve_objective(spec)
    return {**identity, "sha256": _digest(identity)}


def describe_objective(name):
    cls, identity = resolve_objective({"plugin": name})
    return {**identity, "parameters": list(cls.objective_parameters), "contract": cls.objective_contract,
            "defaults": dict(cls.objective_defaults)}


def fit_objective(spec, encoder, x, vx, settings):
    """Train ``encoder`` in place with the declared objective; return its receipt with the identity."""
    cls, _ = resolve_objective(spec)
    identity = objective_identity(spec)
    for name, arr in (("x", x), ("vx", vx)):
        if np.ndim(arr) != 3 or tuple(arr.shape[1:]) != tuple(encoder.input_shape[1:]):
            raise ValueError(f"{name} must be (N, {encoder.input_shape[1]}, {encoder.input_shape[2]}) windows")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
    receipt = cls(identity["params"]).fit(encoder, np.asarray(x, "float32"), np.asarray(vx, "float32"), settings)
    receipt["identity"] = identity
    return receipt


def donor_objective(path):
    """The objective identity recorded in a donor's sidecar provenance (None if not recorded)."""
    sidecar = Path(path).with_suffix(".manifest.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    return (document.get("provenance") or {}).get("objective")
