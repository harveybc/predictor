#!/usr/bin/env python3
"""R0 modular temporal predictor for the phase-4 subset comparison (plan section 4).

One causal Conv1D branch per feature that preserves the time axis (24 -> 12 -> 6 by strided causal
convolutions), channel fusion on the common 6-step grid, a compact dilated causal temporal core
with residual connections, and a head that reads ONLY the last valid position. There is no layer
that flattens time before fusion (no Flatten, no global pooling, no MLP over the window).

Inputs are either RAW standardised feature windows ``(B, 24, F)`` or FROZEN encoder latents
``(B, 6, F * D)`` produced per feature by ``FrozenEncoder`` (the phase-4 runner's chosen weights
for TRAINED_ENCODER, or a seed-0 initialisation that is never updated for RANDOM_ENCODER). The
encoder is applied as a numpy preprocessing step: it cannot receive gradients.

The architecture and the training budget are identical for every subset; only the number of
branches follows the subset size. Feature names are sorted before any tensor is built, so a column
permutation changes nothing (FS4-02). TensorFlow is imported lazily; every pure-python helper in
this module is usable without it.
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

INPUT_MODES = ("RAW", "TRAINED_ENCODER", "RANDOM_ENCODER")
FAMILY = "R0_TEMPORAL_CONV1D"
CORE_OUTPUT_NAME = "core_out"
HEAD_INPUT_NAME = "last_valid_position"
ENCODER_WEIGHT_KEYS = ("down1_kernel", "down1_bias", "down2_kernel", "down2_bias", "latent_kernel", "latent_bias")


class Refusal(ValueError):
    """A contract violation that must not become a fitted model."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def arrays_sha256(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype="float32")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class PredictorSpec:
    window: int = 24
    latent_steps: int = 6
    branch_filters: int = 4
    branch_kernel: int = 3
    fuse_filters: int = 16
    core_filters: int = 16
    core_kernel: int = 3
    core_dilations: tuple = (1, 2)
    max_epochs: int = 30
    batch_size: int = 256
    patience: int = 5
    learning_rate: float = 1e-3
    loss: str = "mse"

    def __post_init__(self):
        if self.window != 4 * self.latent_steps:
            raise Refusal("SPEC_WINDOW_MUST_BE_FOUR_TIMES_LATENT_STEPS (two strided causal halvings)")
        if self.fuse_filters != self.core_filters:
            raise Refusal("SPEC_FUSE_AND_CORE_FILTERS_MUST_MATCH (residual core)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["core_dilations"] = list(self.core_dilations)
        d["family"] = FAMILY
        return d

    def budget(self) -> dict:
        return {"max_epochs": self.max_epochs, "batch_size": self.batch_size, "patience": self.patience,
                "learning_rate": self.learning_rate, "loss": self.loss, "optimizer": "adam",
                "early_stopping": "internal chronological tail of the fit window; best checkpoint restored"}

    def sha256(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class EncoderSpec:
    window: int = 24
    latent_steps: int = 6
    latent_dim: int = 8
    filters: int = 8
    kernel: int = 3

    def to_dict(self) -> dict:
        return asdict(self) | {"family": "CAUSAL_CONV1D_24_12_6"}

    def sha256(self) -> str:
        return digest(self.to_dict())


def budget_sha256(spec: PredictorSpec) -> str:
    return hashlib.sha256(canonical(spec.budget()).encode()).hexdigest()


def canonical_features(features) -> tuple:
    feats = tuple(sorted(features))
    if len(feats) != len(set(feats)) or not feats:
        raise Refusal("FEATURES_MUST_BE_UNIQUE_AND_NONEMPTY")
    return feats


def input_identity(spec: PredictorSpec, features, input_mode: str, encoder_sha256: str | None) -> str:
    if input_mode not in INPUT_MODES:
        raise Refusal(f"UNKNOWN_INPUT_MODE: {input_mode}")
    if input_mode != "RAW" and not (isinstance(encoder_sha256, str) and len(encoder_sha256) == 64):
        raise Refusal("ENCODER_IDENTITY_REQUIRED for encoder input modes")
    return digest({"spec": spec.to_dict(), "features": list(canonical_features(features)), "input_mode": input_mode,
                   "encoder_sha256": encoder_sha256 if input_mode != "RAW" else None})


def architecture_identity(spec: PredictorSpec, input_mode: str, latent_dim: int | None) -> str:
    return digest({"family": FAMILY, "spec": spec.to_dict(), "input_mode": input_mode,
                   "latent_dim": latent_dim if input_mode != "RAW" else None})


# ----------------------------------------------------------------------------- windows and scaling
def make_windows(X: np.ndarray, origin_idx, window: int):
    """Windows of the ``window`` rows ending at each origin (origin included). Origins without a full
    history are dropped and reported through the returned index array. Rows after an origin never
    enter its window."""
    origin_idx = np.asarray(origin_idx, dtype="int64")
    keep = origin_idx >= window - 1
    kept = origin_idx[keep]
    if kept.size == 0:
        return np.empty((0, window, X.shape[1]), dtype="float64"), kept
    offsets = np.arange(-(window - 1), 1)
    idx = kept[:, None] + offsets[None, :]
    return X[idx], kept


@dataclass(frozen=True)
class Standardiser:
    median: np.ndarray
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, Xf: np.ndarray) -> "Standardiser":
        Xf = np.asarray(Xf, dtype="float64")
        med = np.nanmedian(Xf, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Z = np.where(np.isfinite(Xf), Xf, med)
        mean = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return cls(median=med, mean=mean, sd=sd)

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        Z = np.where(np.isfinite(X), X, self.median)
        return (Z - self.mean) / self.sd

    def sha256(self) -> str:
        return arrays_sha256([self.median, self.mean, self.sd])


# ----------------------------------------------------------------------------- tensorflow
_TF = None


def _tf():
    global _TF
    if _TF is None:
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
        _TF = tf
    return _TF


def _keras():
    _tf()
    import keras

    return keras


def _branch(L, x, spec: PredictorSpec, name: str):
    """Causal temporal branch of one feature: (B, 24, 1) -> (B, 6, branch_filters), time preserved."""
    h = L.Conv1D(spec.branch_filters, spec.branch_kernel, padding="causal", activation="relu", name=f"{name}_stem")(x)
    h = L.Conv1D(spec.branch_filters, spec.branch_kernel, strides=2, padding="causal", activation="relu", name=f"{name}_down1")(h)
    h = L.Conv1D(spec.branch_filters, spec.branch_kernel, strides=2, padding="causal", activation="relu", name=f"{name}_down2")(h)
    return h


def build_predictor(spec: PredictorSpec, n_features: int, input_mode: str, latent_dim: int | None):
    if input_mode not in INPUT_MODES:
        raise Refusal(f"UNKNOWN_INPUT_MODE: {input_mode}")
    if n_features < 1:
        raise Refusal("AT_LEAST_ONE_FEATURE")
    keras = _keras()
    L = keras.layers
    if input_mode == "RAW":
        inp = keras.Input((spec.window, n_features), name="raw_windows")
        branches = []
        for i in range(n_features):
            xi = L.Lambda(lambda t, i=i: t[:, :, i:i + 1], output_shape=(spec.window, 1), name=f"branch_{i}_slice")(inp)
            branches.append(_branch(L, xi, spec, f"branch_{i}"))
    else:
        if not latent_dim:
            raise Refusal("LATENT_DIM_REQUIRED for encoder input modes")
        inp = keras.Input((spec.latent_steps, n_features * latent_dim), name="latent_windows")
        branches = []
        for i in range(n_features):
            xi = L.Lambda(lambda t, i=i: t[:, :, i * latent_dim:(i + 1) * latent_dim], output_shape=(spec.latent_steps, latent_dim),
                          name=f"branch_{i}_slice")(inp)
            branches.append(L.Conv1D(spec.branch_filters, 1, padding="causal", activation="relu", name=f"branch_{i}_proj")(xi))
    fused = branches[0] if len(branches) == 1 else L.Concatenate(name="channel_fusion")(branches)
    h = L.Conv1D(spec.fuse_filters, 1, padding="causal", activation="relu", name="fuse")(fused)
    for j, d in enumerate(spec.core_dilations):
        u = L.Conv1D(spec.core_filters, spec.core_kernel, padding="causal", dilation_rate=d, activation="relu", name=f"core{j}_a")(h)
        u = L.Conv1D(spec.core_filters, spec.core_kernel, padding="causal", dilation_rate=d, name=f"core{j}_b")(u)
        h = L.Add(name=f"core{j}_residual")([h, u])
    h = L.Activation("relu", name=CORE_OUTPUT_NAME)(h)
    last = L.Lambda(lambda t: t[:, -1, :], output_shape=(spec.core_filters,), name=HEAD_INPUT_NAME)(h)
    out = L.Dense(1, name="head")(last)
    model = keras.Model(inp, out, name=FAMILY)
    model.fs4_architecture = architecture_identity(spec, input_mode, latent_dim)
    return model


def architecture_sha256(model) -> str:
    return model.fs4_architecture


def count_params(model) -> int:
    return int(model.count_params())


def model_weights_sha256(model) -> str:
    return arrays_sha256([np.asarray(w) for w in model.get_weights()])


# ----------------------------------------------------------------------------- frozen encoder
class FrozenEncoder:
    """Per-feature causal Conv1D encoder 24 -> 12 -> 6 with a D-dimensional temporal latent.

    Weights come either from a seed-0 initialisation that is never updated (RANDOM_ENCODER control)
    or from a saved ``.npz`` of chosen weights (TRAINED_ENCODER: the phase-4 runner's chosen weights,
    one key set per feature ``f{i}_<name>``). The encoder is only ever applied with numpy inputs
    through ``transform``; it has no optimizer and ``optimizer_steps`` stays 0 by construction.
    """

    def __init__(self, spec: EncoderSpec, n_features: int, weights: dict[str, np.ndarray], source: str):
        self.spec = spec
        self.n_features = n_features
        self.weights = {k: np.ascontiguousarray(v, dtype="float32") for k, v in weights.items()}
        self.source = source
        self.optimizer_steps = 0
        for i in range(n_features):
            for key in ENCODER_WEIGHT_KEYS:
                if f"f{i}_{key}" not in self.weights:
                    raise Refusal(f"ENCODER_WEIGHTS_INCOMPLETE: f{i}_{key}")
        self._model = None

    @property
    def weights_sha256(self) -> str:
        return arrays_sha256([self.weights[k] for k in sorted(self.weights)])

    @property
    def latent_dim(self) -> int:
        return self.spec.latent_dim

    @classmethod
    def from_random(cls, spec: EncoderSpec, n_features: int, seed: int = 0) -> "FrozenEncoder":
        rng = np.random.default_rng(seed)
        k, F, D = spec.kernel, spec.filters, spec.latent_dim

        def glorot(shape):
            fan_in, fan_out = np.prod(shape[:-1]), shape[-1] * (shape[0] if len(shape) == 3 else 1)
            lim = math.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, size=shape)

        weights = {}
        for i in range(n_features):
            weights[f"f{i}_down1_kernel"] = glorot((k, 1, F))
            weights[f"f{i}_down1_bias"] = np.zeros(F)
            weights[f"f{i}_down2_kernel"] = glorot((k, F, F))
            weights[f"f{i}_down2_bias"] = np.zeros(F)
            weights[f"f{i}_latent_kernel"] = glorot((1, F, D))
            weights[f"f{i}_latent_bias"] = np.zeros(D)
        return cls(spec, n_features, weights, source=f"RANDOM_SEED_{seed}")

    @classmethod
    def from_npz(cls, spec: EncoderSpec, path, expected_sha256: str | None = None) -> "FrozenEncoder":
        with np.load(Path(path)) as z:
            weights = {k: z[k] for k in z.files}
        n = len({k.split("_", 1)[0] for k in weights if k.startswith("f")})
        enc = cls(spec, n, weights, source=f"NPZ:{Path(path).name}")
        if expected_sha256 is not None and enc.weights_sha256 != expected_sha256:
            raise Refusal("ENCODER_IDENTITY_MISMATCH: the chosen weights do not match the declared digest")
        return enc

    @classmethod
    def from_feature_files(cls, spec: EncoderSpec, feature_paths: dict[str, Path], features) -> "FrozenEncoder":
        """Assemble one bank from per-feature chosen-weight files (keys = ENCODER_WEIGHT_KEYS)."""
        weights = {}
        for i, f in enumerate(canonical_features(features)):
            if f not in feature_paths:
                raise Refusal(f"ENCODER_WEIGHTS_MISSING_FOR_FEATURE: {f}")
            with np.load(Path(feature_paths[f])) as z:
                for key in ENCODER_WEIGHT_KEYS:
                    if key not in z.files:
                        raise Refusal(f"ENCODER_WEIGHTS_INCOMPLETE: {f} {key}")
                    weights[f"f{i}_{key}"] = z[key]
        return cls(spec, len(feature_paths), weights, source="FEATURE_FILES")

    def save(self, path) -> str:
        np.savez(Path(path), **self.weights)
        return self.weights_sha256

    def _build(self):
        keras = _keras()
        L = keras.layers
        s = self.spec
        inp = keras.Input((s.window, self.n_features))
        outs = []
        for i in range(self.n_features):
            xi = L.Lambda(lambda t, i=i: t[:, :, i:i + 1], output_shape=(s.window, 1))(inp)
            h = L.Conv1D(s.filters, s.kernel, strides=2, padding="causal", activation="relu", name=f"f{i}_down1")(xi)
            h = L.Conv1D(s.filters, s.kernel, strides=2, padding="causal", activation="relu", name=f"f{i}_down2")(h)
            outs.append(L.Conv1D(s.latent_dim, 1, padding="causal", name=f"f{i}_latent")(h))
        out = outs[0] if len(outs) == 1 else L.Concatenate()(outs)
        model = keras.Model(inp, out)
        for i in range(self.n_features):
            for layer in ("down1", "down2", "latent"):
                model.get_layer(f"f{i}_{layer}").set_weights([self.weights[f"f{i}_{layer}_kernel"], self.weights[f"f{i}_{layer}_bias"]])
        model.trainable = False
        return model

    def transform(self, windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows, dtype="float32")
        if windows.ndim != 3 or windows.shape[1] != self.spec.window or windows.shape[2] != self.n_features:
            raise Refusal(f"ENCODER_INPUT_SHAPE: expected (B, {self.spec.window}, {self.n_features}), got {windows.shape}")
        if windows.shape[0] == 0:
            return np.empty((0, self.spec.latent_steps, self.n_features * self.spec.latent_dim), dtype="float32")
        if self._model is None:
            self._model = self._build()
        return np.asarray(self._model.predict(windows, batch_size=1024, verbose=0), dtype="float32")


# ----------------------------------------------------------------------------- fit / predict
@dataclass
class FitReport:
    spec: PredictorSpec
    input_mode: str
    features: tuple
    standardiser: Standardiser
    encoder: FrozenEncoder | None
    model: object
    weights_sha256: str
    initial_weights_sha256: str
    epochs_run: int
    best_epoch: int
    updates: int
    fit_seconds: float
    peak_rss_bytes: int
    n_params: int
    budget_sha256: str
    architecture_sha256: str
    encoder_sha256: str | None
    input_identity: str | None
    fit_windows: int
    inner_windows: int
    seed: int
    history: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray, idx) -> np.ndarray:
        return predict(self, X, idx)


def _inputs_for(rep_or_mode, spec: PredictorSpec, encoder: FrozenEncoder | None, Z: np.ndarray, idx):
    windows, kept = make_windows(Z, idx, spec.window)
    if encoder is not None:
        windows = encoder.transform(windows)
    return np.asarray(windows, dtype="float32"), kept


def fit_predictor(spec: PredictorSpec, X: np.ndarray, y: np.ndarray, fit_idx, inner_idx, *, input_mode: str,
                  encoder: FrozenEncoder | None, seed: int, features=None) -> FitReport:
    if input_mode not in INPUT_MODES:
        raise Refusal(f"UNKNOWN_INPUT_MODE: {input_mode}")
    if (input_mode == "RAW") != (encoder is None):
        raise Refusal("ENCODER_PRESENCE_MUST_MATCH_INPUT_MODE")
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    fit_idx = np.asarray(fit_idx, dtype="int64")
    inner_idx = np.asarray(inner_idx, dtype="int64")
    if fit_idx.size == 0 or inner_idx.size == 0:
        raise Refusal("FIT_AND_INNER_ROWS_REQUIRED")
    if encoder is not None and encoder.n_features != X.shape[1]:
        raise Refusal("ENCODER_FEATURE_COUNT_MISMATCH")
    encoder_sha = encoder.weights_sha256 if encoder is not None else None
    standardiser = Standardiser.fit(X[fit_idx])
    Z = standardiser.apply(X)
    Wf, kept_f = _inputs_for(None, spec, encoder, Z, fit_idx)
    Wi, kept_i = _inputs_for(None, spec, encoder, Z, inner_idx)
    if kept_f.size < spec.batch_size or kept_i.size == 0:
        raise Refusal(f"TOO_FEW_WINDOWS: fit {kept_f.size} inner {kept_i.size}")
    yf = y[kept_f].astype("float32")
    yi = y[kept_i].astype("float32")
    if not (np.all(np.isfinite(yf)) and np.all(np.isfinite(yi))):
        raise Refusal("TARGET_NOT_FINITE_ON_FIT_OR_INNER_ROWS")
    keras = _keras()
    keras.utils.set_random_seed(int(seed))
    model = build_predictor(spec, X.shape[1], input_mode, encoder.latent_dim if encoder is not None else None)
    initial_sha = model_weights_sha256(model)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=spec.learning_rate), loss=spec.loss)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=spec.patience, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(Wf, yf, validation_data=(Wi, yi), epochs=spec.max_epochs, batch_size=spec.batch_size,
                     shuffle=True, verbose=0, callbacks=[stop])
    elapsed = time.time() - t0
    val = [float(v) for v in hist.history.get("val_loss", [])]
    epochs_run = len(val)
    best_epoch = int(np.argmin(val)) + 1 if val else 0
    steps_per_epoch = int(math.ceil(kept_f.size / spec.batch_size))
    return FitReport(
        spec=spec, input_mode=input_mode, features=tuple(features) if features else tuple(f"c{i}" for i in range(X.shape[1])),
        standardiser=standardiser, encoder=encoder, model=model,
        weights_sha256=model_weights_sha256(model), initial_weights_sha256=initial_sha,
        epochs_run=epochs_run, best_epoch=best_epoch, updates=epochs_run * steps_per_epoch, fit_seconds=elapsed,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, n_params=count_params(model),
        budget_sha256=budget_sha256(spec), architecture_sha256=architecture_sha256(model), encoder_sha256=encoder_sha,
        input_identity=None, fit_windows=int(kept_f.size), inner_windows=int(kept_i.size), seed=int(seed),
        history={"loss": [float(v) for v in hist.history.get("loss", [])], "val_loss": val},
    )


def fit_named(spec: PredictorSpec, X: np.ndarray, names, y, fit_idx, inner_idx, *, input_mode: str,
              encoder: FrozenEncoder | None, seed: int) -> FitReport:
    """Sort the columns by feature name before anything is built (FS4-02)."""
    names = list(names)
    if len(names) != X.shape[1]:
        raise Refusal("NAMES_AND_COLUMNS_MISMATCH")
    order = sorted(range(len(names)), key=lambda i: names[i])
    feats = canonical_features(names)
    rep = fit_predictor(spec, np.asarray(X)[:, order], y, fit_idx, inner_idx, input_mode=input_mode, encoder=encoder,
                        seed=seed, features=feats)
    rep.input_identity = input_identity(spec, feats, input_mode, rep.encoder_sha256)
    return rep


def predict(rep: FitReport, X: np.ndarray, idx) -> np.ndarray:
    """Predictions for origins ``idx`` (NaN where an origin lacks a full window history)."""
    idx = np.asarray(idx, dtype="int64")
    Z = rep.standardiser.apply(np.asarray(X, dtype="float64"))
    W, kept = _inputs_for(None, rep.spec, rep.encoder, Z, idx)
    out = np.full(idx.shape, np.nan, dtype="float64")
    if kept.size:
        pred = np.asarray(rep.model.predict(W, batch_size=1024, verbose=0), dtype="float64").reshape(-1)
        pos = {int(k): i for i, k in enumerate(idx)}
        for k, p in zip(kept, pred):
            out[pos[int(k)]] = p
    return out
