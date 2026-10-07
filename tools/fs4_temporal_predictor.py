#!/usr/bin/env python3
"""R0 modular temporal predictor for the phase-4 subset comparison (plan section 4).

One causal Conv1D branch per feature that preserves the time axis (24 -> 12 -> 6 by strided causal
convolutions), channel fusion on the common 6-step grid, a compact dilated causal temporal core
with residual connections, and a head that reads ONLY the last valid position. There is no layer
that flattens time before fusion (no Flatten, no global pooling, no MLP over the window).

Inputs are either RAW standardised feature windows ``(B, 24, F)`` or FROZEN encoder latents
``(B, 6, F * D)`` produced per feature by ``RunnerEncoderBank``: the phase-4 runner's own
``build_models`` loaded with its ``chosen.weights.h5`` (TRAINED_ENCODER) or re-initialised with
``seed_weights(0)`` and never updated (RANDOM_ENCODER), each checked against the digests the runner
recorded. The encoder is applied as a numpy preprocessing step: it cannot receive gradients.

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

INPUT_MODES = ("RAW", "RAW_LAG3", "TRAINED_ENCODER", "RANDOM_ENCODER")
RAW_MODES = ("RAW", "RAW_LAG3")        # same architecture and inputs; RAW_LAG3 only ends its window earlier (the encoder's own support)
FAMILY = "R0_TEMPORAL_CONV1D"
CORE_OUTPUT_NAME = "core_out"
HEAD_INPUT_NAME = "last_valid_position"


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
    """Identity of the frozen per-feature encoder: it MUST equal the phase-4 runner's architecture.

    The runner (feature-extractor ``app/fs4_task_runner.py``, ``app.fs4_extractibility.build_models``)
    is the only builder: causal Conv1D 24 -> 12 -> 6, latent_dim 8, fed by (signal, observed_mask,
    delta_time, calendar) windows on an hourly grid. ``fold_id`` names the TRAIN fold whose chosen
    weights are frozen for every weekly fit (inner_2023: fitted on TRAIN rows before 2023 only)."""
    window: int = 24
    latent_steps: int = 6
    latent_dim: int = 8
    filters: int = 16
    kernel: int = 3
    fold_id: str = "inner_2023"
    architecture_id: str = "fs4_causal_conv_24_12_6_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return digest(self.to_dict())


def budget_sha256(spec: PredictorSpec) -> str:
    return hashlib.sha256(canonical(spec.budget()).encode()).hexdigest()


def canonical_features(features) -> tuple:
    feats = tuple(sorted(features))
    if len(feats) != len(set(feats)) or not feats:
        raise Refusal("FEATURES_MUST_BE_UNIQUE_AND_NONEMPTY")
    return feats


def input_identity(spec: PredictorSpec, features, input_mode: str, encoder_sha256: str | None, support_lag_hours: int | None = None) -> str:
    if input_mode not in INPUT_MODES:
        raise Refusal(f"UNKNOWN_INPUT_MODE: {input_mode}")
    if input_mode == "RAW_LAG3" and not (type(support_lag_hours) is int and support_lag_hours >= 1):
        raise Refusal("SUPPORT_LAG_HOURS_REQUIRED for RAW_LAG3")
    if input_mode not in RAW_MODES and not (isinstance(encoder_sha256, str) and len(encoder_sha256) == 64):
        raise Refusal("ENCODER_IDENTITY_REQUIRED for encoder input modes")
    return digest({"spec": spec.to_dict(), "features": list(canonical_features(features)), "input_mode": input_mode,
                   "encoder_sha256": encoder_sha256 if input_mode not in RAW_MODES else None,
                   "support_lag_hours": support_lag_hours if input_mode == "RAW_LAG3" else None})


def architecture_identity(spec: PredictorSpec, input_mode: str, latent_dim: int | None) -> str:
    # the NETWORK is the same for RAW and RAW_LAG3 (only the window's last row differs), so they share one architecture identity
    return digest({"family": FAMILY, "spec": spec.to_dict(), "input_mode": "RAW" if input_mode in RAW_MODES else input_mode,
                   "latent_dim": latent_dim if input_mode not in RAW_MODES else None})


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


def _grouped_layer_class():
    """GroupedCausalConv1D: per-group causal temporal convolution as ONE batched matmul over groups.

    ``groups`` independent branches (group g sees only its own ``cin`` channels), kernel ``k`` over time, strictly causal
    (left zero padding). With ``strides > 1`` the outputs are taken at t = strides-1, 2*strides-1, ... so the LAST output
    always covers the newest input row (the origin): 24 -> 12 -> 6 never discards the most recent observation.
    Equivalent to Conv1D(groups=G) but without the slow CPU grouped-convolution kernel."""
    keras = _keras()
    tf = _tf()

    class GroupedCausalConv1D(keras.layers.Layer):
        def __init__(self, groups, filters_per_group, kernel_size, strides=1, activation=None, **kw):
            super().__init__(**kw)
            self.groups, self.fpg, self.k, self.strides = groups, filters_per_group, kernel_size, strides
            self.act = keras.activations.get(activation)

        def build(self, input_shape):
            c = int(input_shape[-1])
            if c % self.groups:
                raise Refusal("GROUPS_MUST_DIVIDE_CHANNELS")
            self.cin = c // self.groups
            fan_in, fan_out = self.k * self.cin, self.fpg
            self.w = self.add_weight(name="kernel", shape=(self.groups, self.k * self.cin, self.fpg), initializer=keras.initializers.GlorotUniform())
            self.b = self.add_weight(name="bias", shape=(self.groups, self.fpg), initializer="zeros")

        def call(self, x):
            t = tf.shape(x)[1]
            x = tf.reshape(x, (-1, x.shape[1], self.groups, self.cin))
            xp = tf.pad(x, [[0, 0], [self.k - 1, 0], [0, 0], [0, 0]])
            cols = tf.concat([xp[:, i:i + x.shape[1]] for i in range(self.k)], axis=-1)        # (B, T, G, k*cin)
            if self.strides > 1:
                cols = cols[:, self.strides - 1::self.strides]
            y = tf.einsum("btgi,gio->btgo", cols, self.w) + self.b
            y = tf.reshape(y, (-1, y.shape[1], self.groups * self.fpg))
            return self.act(y)

        def compute_output_shape(self, input_shape):
            t = input_shape[1]
            t = None if t is None else (t // self.strides)
            return (input_shape[0], t, self.groups * self.fpg)

    return GroupedCausalConv1D


def build_predictor(spec: PredictorSpec, n_features: int, input_mode: str, latent_dim: int | None):
    """One causal temporal branch per feature as a grouped causal convolution (``GroupedCausalConv1D``, G = F): group i sees
    only channel/latent block i, so each feature has its own filters and the time axis is preserved (24 -> 12 -> 6). Two
    measured alternatives were refused: 365 separate slice+Conv1D branches exhausted an 8G cap, and Conv1D(groups=F) was
    over 25 minutes per fit on CPU."""
    if input_mode not in INPUT_MODES:
        raise Refusal(f"UNKNOWN_INPUT_MODE: {input_mode}")
    if n_features < 1:
        raise Refusal("AT_LEAST_ONE_FEATURE")
    keras = _keras()
    L = keras.layers
    F, bf = n_features, spec.branch_filters
    if input_mode in RAW_MODES:
        inp = keras.Input((spec.window, F), name="raw_windows")
        G = _grouped_layer_class()
        h = G(F, bf, spec.branch_kernel, 1, "relu", name="branch_stem")(inp)
        h = G(F, bf, spec.branch_kernel, 2, "relu", name="branch_down1")(h)
        h = G(F, bf, spec.branch_kernel, 2, "relu", name="branch_down2")(h)
    else:
        if not latent_dim:
            raise Refusal("LATENT_DIM_REQUIRED for encoder input modes")
        inp = keras.Input((spec.latent_steps, F * latent_dim), name="latent_windows")
        h = _grouped_layer_class()(F, bf, 1, 1, "relu", name="branch_proj")(inp)
    h = L.Conv1D(spec.fuse_filters, 1, padding="causal", activation="relu", name="channel_fusion")(h)
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
_EXTRACTOR_MODS = None


def load_extractor(code_dir):
    """Import the PINNED feature-extractor code (the runner's own model builders). The process must not
    have another top-level ``app`` package loaded."""
    global _EXTRACTOR_MODS
    if _EXTRACTOR_MODS is None:
        import importlib
        import sys

        code_dir = Path(code_dir)
        if not (code_dir / "app" / "fs4_extractibility.py").is_file():
            raise Refusal(f"EXTRACTOR_CODE_MISSING: {code_dir} has no app/fs4_extractibility.py")
        if "app" in sys.modules and not str(getattr(sys.modules["app"], "__file__", "")).startswith(str(code_dir)):
            raise Refusal("EXTRACTOR_APP_PACKAGE_CONFLICT: another top-level 'app' package is already imported")
        sys.path.insert(0, str(code_dir))
        _EXTRACTOR_MODS = (importlib.import_module("app.fs4_extractibility"), importlib.import_module("app.univariate_temporal"))
    return _EXTRACTOR_MODS


def index_runner_results(results_root) -> dict:
    """(population_id, identity, feature_id, fold_id, arm) -> (result dict, task directory) for every
    ``<root>/<task_id>/result.json`` the phase-4 runner retained."""
    out = {}
    for path in sorted(Path(results_root).glob("*/result.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "COMPLETE" or rec.get("schema") != "fs4.extractibility.result.v1":
            continue
        key = (rec["population_id"], rec["identity"], rec["feature_id"], rec["fold_id"], rec["arm"])
        if key in out:
            raise Refusal(f"DUPLICATE_RUNNER_RESULT: {key}")
        out[key] = (rec, path.parent)
    return out


class RunnerEncoderBank:
    """Per-feature frozen encoders, built by the runner's own code and verified against its recorded digests.

    TRAINED_ENCODER: ``chosen.weights.h5`` of the runner's TRAINED task for (feature, fold) is loaded into
    ``build_models`` and its file digest and weights digest must equal the terminal's.
    RANDOM_ENCODER: the same architecture re-initialised with ``seed_weights(seed)`` and NEVER updated; the
    weights digest must equal the RANDOM terminal's (so it is the same control the extractibility run used).
    The bank is bound to one (ts, raw columns) pair in canonical (sorted) feature order and cannot be trained.
    """

    optimizer_steps = 0
    # the runner's own strided layers applied to a window that ends at the origin: weights are used at the phase they were trained at
    # (EVEN: output j reads rows 2j-(k-1)..2j). The LAST latent step reads rows <= origin-3: it never reads the origin row.
    alignment = "EVEN_AS_TRAINED"
    last_step_lag_rows = 3

    def __init__(self, spec: EncoderSpec, features, ts_rows, X_raw, *, arm: str, population_id: str, identity: str,
                 results_root, code_dir, seed: int = 0):
        if arm not in ("TRAINED_ENCODER", "RANDOM_ENCODER"):
            raise Refusal(f"UNKNOWN_INPUT_MODE: {arm}")
        self.spec, self.arm, self.seed = spec, arm, int(seed)
        self.features = canonical_features(features)
        X_raw = np.asarray(X_raw, dtype="float64")
        if X_raw.shape[1] != len(self.features):
            raise Refusal("ENCODER_COLUMNS_MUST_FOLLOW_SORTED_FEATURES")
        self.n_features = len(self.features)
        self.latent_dim = spec.latent_dim
        self._X, self._U = load_extractor(code_dir)
        self._ts = np.asarray(ts_rows, dtype="int64")
        self._raw = X_raw
        self._grid = {}
        self._parts = []
        index = index_runner_results(results_root)
        self._enc = []
        for i, name in enumerate(self.features):
            rec_key = (population_id, identity, name, spec.fold_id, arm)
            if rec_key not in index:
                raise Refusal(f"RUNNER_RESULT_MISSING: {rec_key}")
            rec, directory = index[rec_key]
            self._enc.append(self._build_one(name, i, rec, directory))
        self.weights_sha256 = digest({"arm": arm, "fold": spec.fold_id, "spec": spec.to_dict(), "features": [
            [n, p] for n, p in zip(self.features, self._parts)]})

    def _build_one(self, name, i, rec, directory):
        X, U = self._X, self._U
        if rec["hyper"]["window"] != self.spec.window or rec["hyper"]["latent_dim"] != self.spec.latent_dim \
                or rec["architecture"]["id"] != self.spec.architecture_id:
            raise Refusal(f"ENCODER_ARCHITECTURE_MISMATCH: {name}")
        hp = X.Hyper(**rec["hyper"])
        encoder, decoder, training = X.build_models(hp, calendar_dim=len(U.CALENDAR_SPEC))
        want = rec["weights"]["chosen_weights_sha256"]
        if self.arm == "TRAINED_ENCODER":
            h5 = Path(directory) / "chosen.weights.h5"
            if not h5.is_file():
                raise Refusal(f"CHOSEN_WEIGHTS_FILE_MISSING: {h5}")
            if U.sha256_file(str(h5)) != rec["artifacts"].get("chosen_weights_file_sha256"):
                raise Refusal(f"ENCODER_FILE_IDENTITY_MISMATCH: {name}")
            training.load_weights(str(h5))
        else:
            X.seed_weights([encoder, decoder], self.seed)
        if X.weights_digest([encoder, decoder]) != want:
            raise Refusal(f"ENCODER_IDENTITY_MISMATCH: {name} weights digest differs from the runner terminal")
        encoder.trainable = False
        nm = rec["normalization"]
        norm = U.Normalization(float(nm["mean"]), float(nm["std"]), int(nm["n"]), bool(nm["constant"]))
        self._parts.append(want)
        return {"encoder": encoder, "norm": norm, "col": i}

    def _feature_grid(self, i):
        if i not in self._grid:
            self._grid[i] = self._X.to_grid(self._ts, self._raw[:, i])
        return self._grid[i]

    def latents(self, idx):
        """(latents (n, 6, F*D) float32, kept row indices). Only rows <= each origin enter its window."""
        idx = np.asarray(idx, dtype="int64")
        W = self.spec.window
        g0 = self._feature_grid(0)
        anchors = g0.row_index[idx]
        keep = anchors >= W - 1
        kept = idx[keep]
        if kept.size == 0:
            return np.empty((0, self.spec.latent_steps, self.n_features * self.latent_dim), dtype="float32"), kept
        cal = self._U.calendar_features(g0.ts)
        outs = []
        for i, e in enumerate(self._enc):
            g = self._feature_grid(i)
            batch = self._U.make_windows(g.ts, g.x, g.observed, anchors[keep], W, e["norm"], calendar=cal)
            outs.append(np.asarray(e["encoder"].predict(batch.as_inputs(), batch_size=512, verbose=0), dtype="float32"))
        return np.concatenate(outs, axis=-1), kept


# ----------------------------------------------------------------------------- fit / predict
@dataclass
class FitReport:
    spec: PredictorSpec
    input_mode: str
    features: tuple
    standardiser: Standardiser
    encoder: object
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
    window_end: object = None
    support_lag_hours: int | None = None

    def predict(self, X: np.ndarray, idx) -> np.ndarray:
        return predict(self, X, idx)


def _inputs_for(spec: PredictorSpec, encoder, Z: np.ndarray, idx, window_end=None):
    """(windows, kept origin rows). ``window_end[row]`` is the last row a window may contain for an origin at ``row`` (RAW_LAG3: the last
    row at or before origin minus the support lag; -1 when none); None means the window ends at the origin itself (RAW)."""
    if encoder is not None:
        windows, kept = encoder.latents(idx)
        return np.asarray(windows, dtype="float32"), kept
    idx = np.asarray(idx, dtype="int64")
    end = idx if window_end is None else np.asarray(window_end, dtype="int64")[idx]
    if np.any(end > idx):
        raise Refusal("WINDOW_END_AFTER_ORIGIN")
    ok = end >= spec.window - 1
    windows, _ = make_windows(Z, end[ok], spec.window)
    return np.asarray(windows, dtype="float32"), idx[ok]


def fit_predictor(spec: PredictorSpec, X: np.ndarray, y: np.ndarray, fit_idx, inner_idx, *, input_mode: str,
                  encoder, seed: int, features=None, window_end=None, support_lag_hours: int | None = None) -> FitReport:
    if input_mode not in INPUT_MODES:
        raise Refusal(f"UNKNOWN_INPUT_MODE: {input_mode}")
    if (input_mode in RAW_MODES) != (encoder is None):
        raise Refusal("ENCODER_PRESENCE_MUST_MATCH_INPUT_MODE")
    if (input_mode == "RAW_LAG3") != (window_end is not None):
        raise Refusal("WINDOW_END_IS_REQUIRED_FOR_RAW_LAG3_AND_FORBIDDEN_OTHERWISE")
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
    Wf, kept_f = _inputs_for(spec, encoder, Z, fit_idx, window_end)
    Wi, kept_i = _inputs_for(spec, encoder, Z, inner_idx, window_end)
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
        window_end=None if window_end is None else np.asarray(window_end, dtype="int64"), support_lag_hours=support_lag_hours,
    )


def fit_named(spec: PredictorSpec, X: np.ndarray, names, y, fit_idx, inner_idx, *, input_mode: str,
              encoder, seed: int, window_end=None, support_lag_hours: int | None = None) -> FitReport:
    """Sort the columns by feature name before anything is built (FS4-02)."""
    names = list(names)
    if len(names) != X.shape[1]:
        raise Refusal("NAMES_AND_COLUMNS_MISMATCH")
    order = sorted(range(len(names)), key=lambda i: names[i])
    feats = canonical_features(names)
    rep = fit_predictor(spec, np.asarray(X)[:, order], y, fit_idx, inner_idx, input_mode=input_mode, encoder=encoder,
                        seed=seed, features=feats, window_end=window_end, support_lag_hours=support_lag_hours)
    rep.input_identity = input_identity(spec, feats, input_mode, rep.encoder_sha256, support_lag_hours)
    return rep


def predict(rep: FitReport, X: np.ndarray, idx) -> np.ndarray:
    """Predictions for origins ``idx`` (NaN where an origin lacks a full window history)."""
    idx = np.asarray(idx, dtype="int64")
    Z = rep.standardiser.apply(np.asarray(X, dtype="float64"))
    W, kept = _inputs_for(rep.spec, rep.encoder, Z, idx, rep.window_end)
    out = np.full(idx.shape, np.nan, dtype="float64")
    if kept.size:
        pred = np.asarray(rep.model.predict(W, batch_size=1024, verbose=0), dtype="float64").reshape(-1)
        pos = {int(k): i for i, k in enumerate(idx)}
        for k, p in zip(kept, pred):
            out[pos[int(k)]] = p
    return out
