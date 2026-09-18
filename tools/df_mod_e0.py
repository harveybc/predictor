#!/usr/bin/env python3
"""MOD-E0-DEV: multivariate mechanism pilot of the modular proposal (P-MOD) — generator, profiles,
grouping, modular learners, H2/H3 arms, metrics. DEVELOPMENT only; nothing here confirms H2/H3.

Generator (equations reproduced by ML01):
    p variables in two latent groups A, B (p/2 each). Each variable k has an AR(1) part with
    persistence phi_g and a periodic part with period P_g, plus white noise:
        s_k[t] = phi_g * s_k[t-1] + eps_k[t]                      eps ~ N(0, sigma_ar^2)
        x_k[t] = s_k[t] + a * sin(2*pi*t/P_g + theta_k) + n_k[t]  n ~ N(0, sigma_n^2)
    H2 heterogeneity level h in {0,1,2,3} widens the gap between the groups' persistence and
    period: phi_A = 0.5 + 0.12 h, phi_B = 0.5 - 0.12 h; P_B = P_A * (1 + 0.5 h). Sampling,
    p, N, amplitude and noise distribution are the same at every level.
    H3 lagged dependence (r = 1): x_B[t] += beta * x_A_partner[t - tau] (a causal cross term with
    lag tau). r = 0: the SAME marginal process in distribution — the cross term is beta times the
    lagged value of an independent PHANTOM partner generated with group A's parameters (same AR,
    period, amplitude, noise draws of its own), so x_B keeps its persistence, periodicity and
    variance while no observed variable carries information about it (verified by ML04); never
    a temporal shuffle. Target: every variable one step ahead (H = 1, direct). Generator oracle:
    E[x[t+1] | all latent states up to t] (AR propagated, periodic exact, cross term known —
    under r = 0 it uses the phantom, which no observer can see: a floor with full generator
    knowledge, reported as such).

Profiles (train only): ACF at declared lags, Welch band powers, trend/seasonality strength;
descriptors scaled by their development SD, constants excluded; average-linkage clustering.

Learner (per branch j = group): detector = two residual causal TCN blocks (16 filters, kernel 3) ->
integrator = residual dilated blocks (d = 2,4,8,16; branch reach 65 >= context 48) -> linear adapter
TimeDistributed Dense(8) -> fusion (H3-sequence: concatenate the
aligned sequences on the channel axis; H3-summary: global average over time per branch, then
concatenate vectors) -> core (Conv1D(16,3) on the joint sequence | Dense on the joint vector)
-> head Dense(p) from the last position. H2 arms change only which variables reach each
branch (profiles vs predefined random assignments). H3 arms consume the SAME frozen extractor
activations; only fusion, core and head are trained.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent

# --- frozen numbers of the design (derivations in df_mod_e0_design) ---------------------------------
P_VARS = 8
GROUPS = ("A", "B")
P_A = 24                      # samples per cycle of group A's periodic part
AMPLITUDE = 1.0
SIGMA_AR = 0.5
SIGMA_NOISE = 0.3
PHI_BASE, PHI_STEP = 0.5, 0.12
PERIOD_STEP = 0.5
BETA = 0.8                    # cross-dependence weight (H3, r = 1)
TAU = 3                       # cross-dependence lag (samples)
LEVELS = (0, 1, 2, 3)
N_TOTAL = 3000
WINDOW = 48                   # context: 2 cycles of P_A (sensitivity 24 / 96 declared)
HORIZON = 1
ACF_LAGS = (1, 2, 3, 6, 12, 24)
WELCH_BANDS = ((0.0, 0.02), (0.02, 0.06), (0.06, 0.15), (0.15, 0.5))    # cycles per sample
SPLITS = {"train": (0, 2000), "validation": None, "test": None}         # derived in boundaries()
VALIDATION_ROWS = 400
TEST_ROWS = 400
TRAINING = {"optimizer": "adam", "learning_rate": 3e-3, "batch": 64, "max_epochs": 200, "max_updates": 6000, "loss": "mse",
            "early_stopping": {"monitor": "validation loss (mse)", "patience": 30, "restore_best": True},
            "tuning_allowance": "NONE after the ML07 instrument diagnostic (sealed before any pilot cell)",
            "ml07_diagnostic": "plain ReLU stack + mae + lr 1e-3 + 3000 updates stayed at the naive level (val MASE 0.49 vs naive 0.50, "
                               "linear 48x8 reference 0.43); residual TCN blocks with ELU + mse + lr 3e-3 + 6000 updates + patience 30 reach "
                               "0.43-0.46 on 3/3 seeds (relu at lr 1e-2 collapsed on 3/3); metrics reported are MAE/MASE, the optimisation loss is mse"}
CELL_SCHEMA = "df_mod_e0_cell.v1"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# --- generator -------------------------------------------------------------------------------------------

def group_params(level: int) -> dict:
    return {"A": {"phi": PHI_BASE + PHI_STEP * level, "period": float(P_A)},
            "B": {"phi": PHI_BASE - PHI_STEP * level, "period": float(P_A) * (1.0 + PERIOD_STEP * level)}}


def latent_groups(p: int = P_VARS) -> list:
    """Variable k belongs to A for k < p/2, else B (diagnostic only; never fed to the grouper)."""
    return ["A" if k < p // 2 else "B" for k in range(p)]


def generate(level: int, r: int, seed: int, n: int = N_TOTAL, p: int = P_VARS) -> dict:
    """One multivariate trajectory. Returns x (n, p), the components, the causal one-step oracle,
    and the exact parameters used."""
    rng = np.random.default_rng(seed)
    gp = group_params(level)
    groups = latent_groups(p)
    thetas = rng.uniform(0, 2 * math.pi, p)
    s = np.zeros((n, p))
    eps = rng.normal(0, SIGMA_AR, (n, p))
    noise = rng.normal(0, SIGMA_NOISE, (n, p))
    periodic = np.zeros((n, p))
    t = np.arange(n)
    for k in range(p):
        phi = gp[groups[k]]["phi"]
        for i in range(1, n):
            s[i, k] = phi * s[i - 1, k] + eps[i, k]
        periodic[:, k] = AMPLITUDE * np.sin(2 * math.pi * t / gp[groups[k]]["period"] + thetas[k])
    base = s + periodic + noise
    half = p // 2
    x = base.copy()
    cross = np.zeros((n, p))
    partner = {k: k - half for k in range(half, p)}          # B variable k depends on A variable k - half
    # phantom partners (r = 0): independent trajectories with group A's parameters
    phantom = np.zeros((n, p))
    ph_eps = rng.normal(0, SIGMA_AR, (n, p))
    ph_noise = rng.normal(0, SIGMA_NOISE, (n, p))
    ph_theta = rng.uniform(0, 2 * math.pi, p)
    for k in range(half, p):
        ps = np.zeros(n)
        for i in range(1, n):
            ps[i] = gp["A"]["phi"] * ps[i - 1] + ph_eps[i, k]
        phantom[:, k] = ps + AMPLITUDE * np.sin(2 * math.pi * t / gp["A"]["period"] + ph_theta[k]) + ph_noise[:, k]
    source = np.zeros((n, p))
    for k in range(half, p):
        a = base[:, partner[k]] if r == 1 else phantom[:, k]
        source[:, k] = a
        lagged = np.zeros(n)
        lagged[TAU:] = a[:-TAU]
        cross[:, k] = BETA * lagged
        x[:, k] += cross[:, k]
    # the causal one-step oracle: E[x[t+1] | everything up to t] under the generator
    oracle = np.full((n, p), np.nan)
    for k in range(p):
        phi = gp[groups[k]]["phi"]
        per_next = AMPLITUDE * np.sin(2 * math.pi * (t + 1) / gp[groups[k]]["period"] + thetas[k])
        o = phi * s[:, k] + per_next                          # E[s[t+1]] + periodic[t+1]; noise has mean 0
        if k >= half:
            lagged_next = np.zeros(n)
            lagged_next[TAU - 1:] = source[:n - TAU + 1, k]           # a[t+1-tau] is in the past for tau >= 1 (phantom under r = 0)
            o = o + BETA * lagged_next
        oracle[:-1, k] = o[:-1]
    return {"x": x, "s": s, "periodic": periodic, "noise": noise, "cross": cross, "oracle": oracle, "phantom": phantom,
            "params": {"level": level, "r": r, "seed": seed, "n": n, "p": p, "groups": gp, "latent_groups": groups,
                       "thetas": thetas.tolist(), "beta": BETA, "tau": TAU, "amplitude": AMPLITUDE,
                       "sigma_ar": SIGMA_AR, "sigma_noise": SIGMA_NOISE, "partner": {str(k): v for k, v in partner.items()}}}


# --- profiles and grouping (train only) -------------------------------------------------------------------

def acf(x: np.ndarray, lags) -> list:
    x = x - x.mean()
    denom = float(np.dot(x, x)) or 1.0
    return [float(np.dot(x[:-lag], x[lag:]) / denom) for lag in lags]


def welch_bands(x: np.ndarray, bands=WELCH_BANDS, nperseg: int = 256) -> list:
    from scipy.signal import welch
    f, pxx = welch(x - x.mean(), fs=1.0, nperseg=min(nperseg, x.size))
    total = float(pxx.sum()) or 1.0
    return [float(pxx[(f >= lo) & (f < hi)].sum() / total) for lo, hi in bands]


def trend_seasonal_strength(x: np.ndarray, period: int = P_A) -> list:
    """Hyndman's strength: 1 - Var(remainder)/Var(trend+remainder), via a simple moving-average
    decomposition with the declared period (a common rule for every method compared)."""
    n = x.size
    k = period
    trend = np.convolve(x, np.ones(k) / k, mode="same")
    detr = x - trend
    seasonal = np.array([detr[i::k].mean() for i in range(k)])
    seas = np.tile(seasonal, n // k + 1)[:n]
    rem = detr - seas
    ft = max(0.0, 1 - np.var(rem) / (np.var(detr + trend - trend.mean() + 0) or 1.0))
    fs = max(0.0, 1 - np.var(rem) / (np.var(detr) or 1.0))
    return [float(ft), float(fs)]


def profiles(x_train: np.ndarray) -> dict:
    """Descriptor vector per variable, computed on TRAINING rows only."""
    desc = []
    for k in range(x_train.shape[1]):
        v = x_train[:, k]
        desc.append(acf(v, ACF_LAGS) + welch_bands(v) + trend_seasonal_strength(v))
    d = np.asarray(desc)
    sd = d.std(axis=0)
    keep = sd > 1e-12                                                   # constant descriptors excluded
    scaled = (d[:, keep] - d[:, keep].mean(axis=0)) / sd[keep]
    return {"raw": d, "scaled": scaled, "kept": keep.tolist(),
            "names": [f"acf{l}" for l in ACF_LAGS] + [f"band{i}" for i in range(len(WELCH_BANDS))] + ["trend", "seasonal"]}


def average_linkage(scaled: np.ndarray, n_groups: int) -> list:
    """Agglomerative average-linkage clustering on Euclidean distances; returns group index per variable."""
    p = scaled.shape[0]
    clusters = [[k] for k in range(p)]
    dist = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=2)
    while len(clusters) > n_groups:
        best, pair = None, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(np.mean([dist[a, b] for a in clusters[i] for b in clusters[j]]))
                if best is None or d < best:
                    best, pair = d, (i, j)
        i, j = pair
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    labels = [0] * p
    for g, members in enumerate(sorted(clusters, key=min)):
        for k in members:
            labels[k] = g
    return labels


def adjusted_rand(a: list, b: list) -> float:
    from itertools import combinations
    n = len(a)
    same_a = {(i, j): a[i] == a[j] for i, j in combinations(range(n), 2)}
    same_b = {(i, j): b[i] == b[j] for i, j in combinations(range(n), 2)}
    n11 = sum(1 for k in same_a if same_a[k] and same_b[k])
    n00 = sum(1 for k in same_a if not same_a[k] and not same_b[k])
    n10 = sum(1 for k in same_a if same_a[k] and not same_b[k])
    n01 = sum(1 for k in same_a if not same_a[k] and same_b[k])
    tot = n11 + n00 + n10 + n01
    expected = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / tot
    maxi = 0.5 * ((n11 + n10) + (n11 + n01) + (n01 + n00) + (n10 + n00))
    return float((n11 + n00 - expected) / (maxi - expected)) if maxi != expected else 1.0


def random_assignments(p: int, n_groups: int, sizes: list, n_perm: int, seed: int) -> list:
    """Predefined random redistributions with the SAME group sizes (never a renaming of a
    profile partition: a permutation equal to the profile partition up to relabeling is skipped)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_perm:
        perm = rng.permutation(p).tolist()
        labels = [0] * p
        start = 0
        for g, size in enumerate(sizes):
            for k in perm[start:start + size]:
                labels[k] = g
            start += size
        out.append(labels)
    return out


def same_partition(a: list, b: list) -> bool:
    return adjusted_rand(a, b) >= 1.0 - 1e-12


# --- windows, splits, scaling, MASE ---------------------------------------------------------------------

def boundaries(n: int = N_TOTAL, window: int = WINDOW, horizon: int = HORIZON) -> dict:
    purge = window + horizon
    test_end = n
    test_start = test_end - TEST_ROWS
    val_end = test_start - purge
    val_start = val_end - VALIDATION_ROWS
    train_end = val_start - purge
    train_start = window - 1
    return {"train": [train_start, train_end], "validation": [val_start, val_end], "test": [test_start, test_end - horizon],
            "purge": purge, "window": window, "horizon": horizon}


def make_windows(x: np.ndarray, rows: np.ndarray, window: int) -> np.ndarray:
    return np.stack([x[t - window + 1:t + 1] for t in rows])            # (rows, W, p); the past up to t only


def prepare(x: np.ndarray, oracle: np.ndarray, periods: list, window: int = WINDOW, horizon: int = HORIZON,
            *, test_access: bool = True) -> dict:
    b = boundaries(x.shape[0], window, horizon)
    parts = {}
    for name in ("train", "validation") + (("test",) if test_access else ()):
        lo, hi = b[name]
        rows = np.arange(lo, hi)
        parts[name] = {"rows": rows, "X": make_windows(x, rows, window), "y": x[rows + horizon],
                       "naive": x[rows], "oracle": oracle[rows]}
    tr = parts["train"]
    mean, sd = tr["X"].reshape(-1, x.shape[1]).mean(axis=0), tr["X"].reshape(-1, x.shape[1]).std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    # MASE denominators: train MAE of the seasonal naive (repeat one period earlier) per variable,
    # with the declared period; zero denominators flagged before any scoring (policy: MASE NO_APLICA)
    denom = []
    lo, hi = b["train"]
    for k in range(x.shape[1]):
        m = int(round(periods[k]))
        v = x[lo:hi, k]
        d = float(np.mean(np.abs(v[m:] - v[:-m]))) if v.size > m else 0.0
        denom.append(d)
    return {"boundaries": b, "parts": parts, "scale": {"mean": mean.tolist(), "sd": sd.tolist()},
            "mase_denominator": denom, "mase_policy": {"zero": "NO_APLICA (MAE reported only)", "shared": "same denominators for every arm"},
            "test_access": test_access}


def mase(pred: np.ndarray, y: np.ndarray, denom: list) -> dict:
    """Per-variable MASE with the shared train denominator; NO_APLICA where the denominator is 0."""
    out = {}
    for k in range(y.shape[1]):
        m = float(np.mean(np.abs(pred[:, k] - y[:, k])))
        out[k] = {"mae": m, "mase": (m / denom[k]) if denom[k] > 0 else None, "status": "MEDIDO" if denom[k] > 0 else "NO_APLICA"}
    vals = [v["mase"] for v in out.values() if v["mase"] is not None]
    return {"per_variable": out, "mase_mean": float(np.mean(vals)) if vals else None,
            "mae_mean": float(np.mean([v["mae"] for v in out.values()]))}


# --- models ------------------------------------------------------------------------------------------------

def _tf():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def _tcn_block(tf, h, name: str, dilation: int, filters: int = 16):
    """Residual dilated causal block after Bai et al. (2018): conv -> ReLU, added to a 1x1
    projection of the input when channels differ; keeps the temporal axis."""
    y = tf.keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation, activation=ACTIVATION, name=f"{name}_conv")(h)
    skip = h if h.shape[-1] == filters else tf.keras.layers.Conv1D(filters, 1, padding="causal", name=f"{name}_proj")(h)
    return tf.keras.layers.Add(name=f"{name}_res")([y, skip])


def branch_extractor(tf, inp, name: str):
    """Detector (two residual blocks, dilation 1) -> integrator (residual blocks with dilations
    INTEGRATOR_DILATIONS) -> linear adapter; the temporal axis is kept (no Flatten)."""
    h = _tcn_block(tf, inp, f"{name}_det1", 1)
    h = _tcn_block(tf, h, f"{name}_det2", 1)
    for i, d in enumerate(INTEGRATOR_DILATIONS, start=1):
        h = _tcn_block(tf, h, f"{name}_int{i}", d)
    z = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8), name=f"{name}_adapt")(h)
    return z


ACTIVATION = "elu"                        # ML07 seed-robustness diagnostic: relu at lr 1e-2 collapsed to the naive level for 3/3 seeds;
                                          # elu reached 0.43-0.46 for 3/3 seeds at lr 1e-2 and 3e-3 (see RP3_ML07_RECEIVER_DIAGNOSTIC.json)
INTEGRATOR_DILATIONS = (2, 4, 8, 16)      # ML07 diagnostic: the plain ReLU stack (no residual path) collapsed to near-zero
                                          # activations and stayed at the naive level below a linear 48x8 reference; residual
                                          # blocks (the TCN of Bai et al.) and a reach (65) covering the context (48) are required
RECEPTIVE_FIELD = 1 + 2 * (1 + 1 + sum(INTEGRATOR_DILATIONS))   # kernel 3 -> (k-1) x dilation: 1,1,2,4,8,16 -> 65 samples


def build_modular(assignment: list, window: int, p: int, *, fusion: str = "sequence", seed: int = 1):
    """One branch per group of `assignment`; fusion 'sequence' concatenates the aligned sequences
    and keeps the temporal axis into the core; 'summary' averages each branch over time first."""
    tf = _tf()
    tf.keras.utils.set_random_seed(int(seed))
    inp = tf.keras.Input(shape=(window, p), name="x")
    groups = sorted(set(assignment))
    branches = []
    for g in groups:
        idx = [k for k in range(p) if assignment[k] == g]
        sub = tf.keras.layers.Lambda(lambda t, idx=idx: tf.gather(t, idx, axis=2), name=f"g{g}_select")(inp)
        branches.append(branch_extractor(tf, sub, f"g{g}"))
    # the head predicts the increment over the last observation (persistence skip): identical in
    # every arm, so it changes no comparison; the ML07 diagnostic showed the plain head needs more
    # than the sealed update allowance to pass the naive forecaster
    last_x = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_observation")(inp)
    if fusion == "sequence":
        joint = tf.keras.layers.Concatenate(axis=2, name="fusion_seq")(branches) if len(branches) > 1 else branches[0]
        core = tf.keras.layers.Conv1D(16, 3, padding="causal", activation=ACTIVATION, name="core_conv")(joint)
        last = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="core_last")(core)
        delta = tf.keras.layers.Dense(p, name="head")(last)
    elif fusion == "summary":
        pooled = [tf.keras.layers.GlobalAveragePooling1D(name=f"summary_g{g}")(b) for g, b in zip(groups, branches)]
        joint = tf.keras.layers.Concatenate(axis=1, name="fusion_vec")(pooled) if len(pooled) > 1 else pooled[0]
        # capacity matched to the sequence core within the declared tolerance (15 %): 808 vs 920 trainable
        core = tf.keras.layers.Dense(32, activation=ACTIVATION, name="core_dense")(joint)
        delta = tf.keras.layers.Dense(p, name="head")(core)
    else:
        raise ValueError(fusion)
    out = tf.keras.layers.Add(name="persistence_skip")([last_x, delta])
    return tf.keras.Model(inp, out, name=f"modular_{fusion}")


def extractor_layer_names(model) -> list:
    return [l.name for l in model.layers if any(tag in l.name for tag in ("_det", "_int", "_adapt"))]


def freeze_extractor(model) -> dict:
    frozen, trainable = [], []
    for l in model.layers:
        if any(tag in l.name for tag in ("_det", "_int", "_adapt")):
            l.trainable = False
            frozen.append(l.name)
        elif l.weights:
            trainable.append(l.name)
    return {"frozen": frozen, "trainable": trainable}


def count_params(model) -> dict:
    tf = _tf()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def fit(model, Xtr, ytr, Xva, yva, *, training: dict = TRAINING, seed: int = 1) -> dict:
    tf = _tf()
    tf.keras.utils.set_random_seed(int(seed))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training["learning_rate"]), loss=training.get("loss", "mae"))
    before = {l.name: [w.copy() for w in l.get_weights()] for l in model.layers if l.weights}
    batch = int(training["batch"])
    steps = math.ceil(Xtr.shape[0] / batch)
    max_epochs = min(int(training["max_epochs"]), max(1, int(training["max_updates"]) // steps))

    class Counter(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.updates = 0

        def on_train_batch_end(self, batch, logs=None):
            self.updates += 1
    counter = Counter()
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(training["early_stopping"]["patience"]),
                                          restore_best_weights=True)
    t0 = time.process_time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=max_epochs, batch_size=batch, shuffle=True, verbose=0,
                     callbacks=[counter, es])
    fit_seconds = time.process_time() - t0
    after = {l.name: [w for w in l.get_weights()] for l in model.layers if l.weights}
    changed = {n: float(sum(np.linalg.norm(a - b) for a, b in zip(after[n], before[n]))) for n in before}
    va = [float(v) for v in hist.history["val_loss"]]
    return {"updates": int(counter.updates), "epochs": len(va), "curve": {"train": [float(v) for v in hist.history["loss"]], "validation": va},
            "stop_reason": "EARLY_STOPPING" if es.stopped_epoch else ("UPDATE_BUDGET" if len(va) >= max_epochs else "EPOCH_BUDGET"),
            "restored_checkpoint_epoch": int(np.argmin(va)) + 1, "weight_change_by_layer": changed,
            "steps_per_epoch": steps, "max_epochs_allowed": max_epochs, "fit_seconds": round(fit_seconds, 3)}


def _sx(X, s):
    return (X - np.asarray(s["mean"])) / np.asarray(s["sd"])


def _sy(y, s):
    return (y - np.asarray(s["mean"])) / np.asarray(s["sd"])


def _uy(z, s):
    return z * np.asarray(s["sd"]) + np.asarray(s["mean"])


# --- one cell ------------------------------------------------------------------------------------------------

def run_cell(job: dict, out_dir: Path) -> dict:
    """A governed unit: one (hypothesis, condition, replicate, arm). H2 arms train a full modular
    model under an assignment (profiles or a predefined random one). H3 first trains the shared
    extractor under the profile assignment with the sequence fusion (arm 'extractor'), then each
    arm loads and FREEZES it and trains only fusion/core/head."""
    t0 = time.process_time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    level, r, seed = int(job["level"]), int(job["r"]), int(job["seed"])
    gen = generate(level, r, seed)
    x, oracle, params = gen["x"], gen["oracle"], gen["params"]
    if not np.isfinite(x).all():
        raise ValueError("non-finite generator output")
    periods = [params["groups"][g]["period"] for g in params["latent_groups"]]
    test_access = job.get("role", "CELL") != "COST_PILOT"
    prep = prepare(x, oracle, periods, int(job.get("window", WINDOW)), HORIZON, test_access=test_access)
    P, s = prep["parts"], prep["scale"]
    prof = profiles(P["train"]["X"][:, -1, :] if False else x[prep["boundaries"]["train"][0]:prep["boundaries"]["train"][1]])
    profile_labels = average_linkage(prof["scaled"], 2)
    sizes = [profile_labels.count(g) for g in sorted(set(profile_labels))]
    arm = job["arm"]
    if job["hypothesis"] == "H2":
        if arm == "profiles":
            assignment = profile_labels
        else:
            k = int(arm.split("_")[-1])
            cands = random_assignments(x.shape[1], 2, sizes, 8, seed=1000 + seed)
            cands = [c for c in cands if not same_partition(c, profile_labels)]
            assignment = cands[k]
        fusion = "sequence"
    else:
        assignment = profile_labels
        fusion = {"extractor": "sequence", "sequence": "sequence", "summary": "summary"}[arm]
    training = dict(job.get("training") or TRAINING)
    if job.get("max_updates_override"):
        training["max_updates"] = int(job["max_updates_override"])
    model = build_modular(assignment, int(job.get("window", WINDOW)), x.shape[1], fusion=fusion, seed=seed)
    frozen_info = None
    if job["hypothesis"] == "H3" and arm in ("sequence", "summary"):
        ext_path = Path(job["extractor_weights"])
        donor = build_modular(assignment, int(job.get("window", WINDOW)), x.shape[1], fusion="sequence", seed=seed)
        donor.load_weights(str(ext_path))
        for name in extractor_layer_names(donor):
            model.get_layer(name).set_weights(donor.get_layer(name).get_weights())
        frozen_info = freeze_extractor(model)
        frozen_info["extractor_sha256"] = hashlib.sha256(ext_path.read_bytes()).hexdigest()
    params_count = count_params(model)
    Xtr, ytr = _sx(P["train"]["X"], s), _sy(P["train"]["y"], s)
    Xva, yva = _sx(P["validation"]["X"], s), _sy(P["validation"]["y"], s)
    ext_before = {n: [w.copy() for w in model.get_layer(n).get_weights()] for n in extractor_layer_names(model)}
    fitrec = fit(model, Xtr, ytr, Xva, yva, training=training, seed=seed)
    ext_after = {n: model.get_layer(n).get_weights() for n in extractor_layer_names(model)}
    extractor_moved = float(sum(np.linalg.norm(a - b) for n in ext_before for a, b in zip(ext_after[n], ext_before[n])))
    weights_path = out_dir / "weights.weights.h5"
    model.save_weights(str(weights_path))
    reloaded = build_modular(assignment, int(job.get("window", WINDOW)), x.shape[1], fusion=fusion, seed=seed)
    reloaded.load_weights(str(weights_path))
    preds = {n: _uy(model.predict(_sx(P[n]["X"], s), verbose=0, batch_size=256), s) for n in P}
    parity = bool(np.allclose(_uy(reloaded.predict(Xva, verbose=0, batch_size=256), s), preds["validation"], atol=1e-5))
    denom = prep["mase_denominator"]
    # the observer-attainable linear reference (ridge on the same window of all variables, train only)
    Ftr = Xtr.reshape(Xtr.shape[0], -1)
    A = np.hstack([Ftr, np.ones((Ftr.shape[0], 1))])
    w_lin = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)
    linear = {n: _uy(np.hstack([_sx(P[n]["X"], s).reshape(P[n]["X"].shape[0], -1), np.ones((P[n]["X"].shape[0], 1))]) @ w_lin, s) for n in P}
    scores = {}
    for n in P:
        scores[n] = {"model": mase(preds[n], P[n]["y"], denom), "naive": mase(P[n]["naive"], P[n]["y"], denom),
                     "oracle": mase(P[n]["oracle"], P[n]["y"], denom), "linear_window": mase(linear[n], P[n]["y"], denom),
                     "rows": int(P[n]["y"].shape[0])}
    np.savez(out_dir / "arrays.npz", **{f"{n}_rows": P[n]["rows"] for n in P}, **{f"{n}_y": P[n]["y"] for n in P},
             **{f"{n}_pred": preds[n] for n in P}, **{f"{n}_naive": P[n]["naive"] for n in P},
             **{f"{n}_oracle": P[n]["oracle"] for n in P}, **{f"{n}_linear": linear[n] for n in P}, denominator=np.asarray(denom))
    record = {"schema": CELL_SCHEMA, "cell_id": job["cell_id"], "hypothesis": job["hypothesis"], "level": level, "r": r,
              "seed": seed, "arm": arm, "role": job.get("role", "CELL"), "window": int(job.get("window", WINDOW)),
              "horizon": HORIZON, "receptive_field": RECEPTIVE_FIELD, "generator": {k: v for k, v in params.items() if k != "thetas"},
              "generator_sha256": sha_obj(params), "x_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
              "boundaries": prep["boundaries"], "rows": {n: int(P[n]["rows"].size) for n in P},
              "profiles": {"names": prof["names"], "kept": prof["kept"], "labels": profile_labels,
                           "ari_vs_latent": adjusted_rand(profile_labels, [0 if g == "A" else 1 for g in params["latent_groups"]])},
              "assignment": assignment, "assignment_is_profile": same_partition(assignment, profile_labels),
              "fusion": fusion, "parameters": params_count, "frozen": frozen_info, "extractor_weight_change": extractor_moved,
              "training": {**fitrec, "rule": training}, "prediction_parity_after_reload": parity,
              "mase_denominator": denom, "mase_policy": prep["mase_policy"], "scores": scores,
              "exposure": "TEST_SCORED_DESCRIPTIVE" if test_access else "NO_TEST_ACCESS",
              "arrays_sha256": hashlib.sha256((out_dir / "arrays.npz").read_bytes()).hexdigest(),
              "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
              "cost": {"cpu_seconds": round(time.process_time() - t0, 3), "fit_seconds": fitrec["fit_seconds"]}}
    body = json.dumps(record, sort_keys=True, default=float).encode()
    (out_dir / "cell.json").write_bytes(body)
    return record


def worker_main(job_file: Path) -> int:
    job = json.loads(Path(job_file).read_text())
    adir = Path(job["attempt_dir"])
    record = run_cell(job, adir)
    body = (adir / "cell.json").read_bytes()
    result = {"status": "COMPLETED", "reason": "", "output_file": "cell.json", "output_sha256": hashlib.sha256(body).hexdigest(),
              "rows_written": 1, "outcome": "COMPLETED"}
    tmp = adir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, adir / "result.json")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    raise SystemExit(worker_main(parser.parse_args().worker))
