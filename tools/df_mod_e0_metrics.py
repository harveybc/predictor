#!/usr/bin/env python3
"""RP13: the D/Y/M/G metrics of the programme's metrics contract (v1) for MOD-E0 units.

Data / targets (per variable and split): serialization (bytes, dtype, endian, shape, mask), lossless
compressed length with a fixed coder/level (zlib 9 and lzma 6, headers included; a ratio above 1 is
possible and is NOT information), H0 plug-in entropy with a TRAIN-frozen 16-quantile quantizer
(reused from df_profile_information), lag-1 conditional redundancy/surprisal, the planted SNR of the
generator (signal = AR + periodic + planted cross term, noise = the iid term; power = variance over
the split rows), ACF/Welch/trend/seasonal profiles (descriptor v2), the planted lagged dependence
and the temporal support. Every quantity carries a state (MEDIDO / NO_APLICA / NO_MEDIDO /
INCONCLUSIVE) and a reason; nothing missing becomes zero.

Model / graph: parameters total/trainable/frozen (counted apart), serialized weight bytes (the
saved file) and raw float32 bytes, per-layer Frobenius norms, and per weight matrix (Dense kernels
and Conv1D kernels reshaped to (kernel x in, out)) the singular-value descriptors of the contract:
effective_rank_entropy = exp(-sum p ln p) with p = s / sum(s) (Roy & Vetterli 2007, reused from
df_profile_information.effective_rank), stable_rank = sum(s^2)/max(s)^2, nuclear_ratio =
sum(s)/max(s), spectral_norm = max(s); a zero matrix is NOT_DEFINED by those formulas (state, not
zero). Gradient norms on ONE fixed declared mini-batch and adapter activation statistics on ONE
fixed declared validation batch, when measured; otherwise NO_MEDIDO with the reason. Graph: nodes =
layers, edges = inbound links, directed; density = E / (N (N - 1)); spectral radius NO_APLICA (a DAG
adjacency has radius 0 by structure). Checkpoints: initial, epochs of a geometric schedule fixed
before the run, final (last epoch, before restore) and best (restored).
"""

from __future__ import annotations

import importlib.util
import lzma
import math
import os
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PI = _load("df_profile_information")

SCHEMA = "df_mod_e0_metrics.v2"        # v2 (RP20): grains (base series, unique inputs, window tensor, targets) and SNR total
MEDIDO, NO_APLICA, NO_MEDIDO, INCONCLUSIVE, NOT_DEFINED = "MEDIDO", "NO_APLICA", "NO_MEDIDO", "INCONCLUSIVE", "NOT_DEFINED"
CHECKPOINT_EPOCHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)         # fixed before any run; geometric, cheap
CODERS = {"zlib9": lambda b: len(zlib.compress(b, PI.ZLIB_LEVEL)), "lzma6": lambda b: len(lzma.compress(b, preset=PI.LZMA_PRESET))}
DECLARATIONS = {
    "bytes_raw": "little-endian float64 of the split's rows of the variable (the series the windows and targets are cut from)",
    "compression": f"zlib level {PI.ZLIB_LEVEL} and lzma preset {PI.LZMA_PRESET} (xz container), headers included; operational length, not information",
    "symbols": f"train-frozen {PI.N_BINS}-quantile quantizer of df_profile_information (edges fitted on the train rows of the same variable)",
    "h0": "plug-in Shannon entropy of the symbol counts, bits; biased low; not differential entropy, not a rate with memory",
    "conditional": f"H(X_t) - H(X_t | X_(t-1)) on symbols; INCONCLUSIVE when a conditioning state has fewer than {PI.COND_MIN_COUNT} counts",
    "snr_planted": "v1 (kept under its name, NOT the whole signal of a diagnostic condition): 10 log10(Var(s + periodic + cross) / Var(noise)) over the split rows",
    "snr_planted_total": "v2: 10 log10(Var(s + periodic + cross + deterministic) / Var(noise)) over the split rows, power = variance over the rows; "
                         "equals v1 when the deterministic term is zero; the components are the generator's (identified), never estimated",
    "composition": "x = s + periodic + cross + deterministic + noise exactly; the maximum residual is reported per split",
    "grains": "base_series = the split's rows of x; inputs_unique = the rows any window of the split consumes (rows - W + 1 .. rows); "
              "window_tensor = X (rows x W x p, WITH repetitions: every base row appears in up to W windows); targets = y = x[rows + h]; "
              "descriptors of the base series are not descriptors of X or Y",
    "dependence_planted": "Pearson correlation between x_B[t] and its lagged partner a[t - tau] over the split rows (r = 1: partner; r = 0: phantom)",
    "effective_rank_entropy": "exp(-sum p ln p), p = s / sum s (Roy & Vetterli 2007, definition 1)",
    "stable_rank": "sum(s^2) / max(s)^2", "nuclear_ratio": "sum(s) / max(s) (NOT the effective rank)", "spectral_norm": "max(s) of the rectangular kernel (not a radius)",
    "gradients": "global and per-layer L2 norm of the loss gradient on ONE fixed mini-batch (the first 64 training windows), measured at the checkpoints",
    "activations": "mean, std and near-zero fraction (|a| < 1e-6) of every adapter output on ONE fixed batch (the first 32 validation windows)",
    "graph": "nodes = Keras layers, edges = inbound links (directed); density = E / (N (N - 1)); spectral radius NO_APLICA for a DAG",
}


# --- data / targets ------------------------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray):
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def grain_metrics(x: np.ndarray, prep: dict, part: str, scale: dict) -> dict:
    """RP20: the four grains a split really consumes, each with row identities, shape, bytes, mask, scale
    and (for targets) the denominator; compression of the window tensor counts its repetitions."""
    b = prep["boundaries"]
    lo, hi = b[part]
    W, h = int(b["window"]), int(b["horizon"])
    rows = np.arange(lo, hi)
    P = prep["parts"][part]
    X, y = P["X"], P["y"]
    p = x.shape[1]
    g = {}
    base = np.ascontiguousarray(x[lo:hi], dtype="<f8")
    uniq = np.ascontiguousarray(x[lo - W + 1:hi], dtype="<f8")
    xt = np.ascontiguousarray(X, dtype="<f8")
    yt = np.ascontiguousarray(y, dtype="<f8")
    for name, arr, ident in (("base_series", base, [int(lo), int(hi)]), ("inputs_unique", uniq, [int(lo - W + 1), int(hi)]),
                             ("window_tensor", xt, {"origins": [int(lo), int(hi)], "window": W, "rows_per_window": W}),
                             ("targets", yt, [int(lo + h), int(hi + h)])):
        raw = arr.tobytes()
        g[name] = {"row_identity": ident, "shape": list(arr.shape), "dtype": "<f8", "bytes_raw": len(raw), "mask_non_finite": int((~np.isfinite(arr)).sum()),
                   "compressed_bytes_zlib9": CODERS["zlib9"](raw), "compressed_bits_per_value_zlib9": 8.0 * CODERS["zlib9"](raw) / arr.size,
                   "scale": {"applied": False, "train_only": {"mean": [float(v) for v in scale["mean"]], "sd": [float(v) for v in scale["sd"]]},
                             "note": "the model consumes (X - mean) / sd and predicts in the scaled space; arrays here are raw"}}
    g["window_tensor"]["repetition_factor"] = float(xt.size / max(uniq.size, 1))
    g["window_tensor"]["note"] = "every base row of the split appears in up to W windows; the tensor is what the learner consumed, not new information"
    g["targets"]["mase_denominator"] = [float(d) for d in prep["mase_denominator"]]
    g["targets"]["horizon"] = h
    return g


def data_metrics(gen: dict, prep: dict, parts: list, descriptor_version: int) -> dict:
    """Per split and variable; the states are explicit."""
    E = _load("df_mod_e0")
    x, params = gen["x"], gen["params"]
    p = x.shape[1]
    half = p // 2
    clean = gen["s"] + gen["periodic"] + gen["cross"]
    det = gen.get("deterministic")
    det = np.zeros(x.shape[0]) if det is None else np.asarray(det)
    clean_total = clean + det[:, None]
    noise = gen["noise"]
    b = prep["boundaries"]
    lo_tr, hi_tr = b["train"]
    out = {"schema": SCHEMA, "declarations": DECLARATIONS, "descriptor_version": int(descriptor_version), "splits": {}, "support": {},
           "composition": {"formula": "x = s + periodic + cross + deterministic + noise",
                           "max_residual": float(np.max(np.abs(x - (clean_total + noise)))), "diagnostic": params.get("diagnostic", "none")}}
    edges = [PI.frozen_edges(x[lo_tr:hi_tr, k]) for k in range(p)]
    for part in parts:
        lo, hi = b[part]
        rows = np.arange(lo, hi)
        entry = {"rows": int(rows.size), "range": [int(lo), int(hi)], "variables": {}, "grains": grain_metrics(x, prep, part, prep["scale"])}
        for k in range(p):
            v = np.ascontiguousarray(x[lo:hi, k], dtype="<f8")
            raw = v.tobytes()
            sym = PI.quantize(v, edges[k])
            var = {"bytes_raw": len(raw), "dtype": "<f8", "endian": "little", "shape": [int(v.size)], "mask_non_finite": int((~np.isfinite(v)).sum()),
                   "serialization_precision_bits": 64}
            for coder, fn in CODERS.items():
                var[f"compressed_bytes_{coder}_raw_float64"] = fn(raw)
                var[f"compressed_bits_per_sample_{coder}_raw_float64"] = 8.0 * fn(raw) / v.size
            var["compressed_bytes_zlib9_symbols"] = CODERS["zlib9"](sym.tobytes())
            var["compressed_bits_per_sample_zlib9_symbols"] = 8.0 * var["compressed_bytes_zlib9_symbols"] / v.size
            var["compression_ratio_zlib9_raw"] = var["compressed_bytes_zlib9_raw_float64"] / len(raw)
            counts = np.bincount(sym[sym != PI.MISSING_SYMBOL], minlength=PI.N_BINS)
            var["h0_bits"] = PI.entropy_bits(counts) if counts.sum() > 0 else None
            var["h0_state"] = MEDIDO if var["h0_bits"] is not None else NO_MEDIDO
            var["symbols_used"] = int((counts > 0).sum())
            cr = PI.conditional_redundancy(sym)
            if cr is None:
                var.update(conditional_redundancy_bits_lag1=None, conditional_surprisal_bits_lag1=None, conditional_state=NO_MEDIDO)
            elif cr[2] < PI.COND_MIN_COUNT:
                var.update(conditional_redundancy_bits_lag1=None, conditional_surprisal_bits_lag1=None, conditional_state=INCONCLUSIVE,
                           conditional_reason=f"a conditioning state has {cr[2]} < {PI.COND_MIN_COUNT} counts")
            else:
                var.update(conditional_redundancy_bits_lag1=cr[0] - cr[1], conditional_surprisal_bits_lag1=cr[1], conditional_state=MEDIDO)
            sv, nv = float(clean[lo:hi, k].var()), float(noise[lo:hi, k].var())
            st = float(clean_total[lo:hi, k].var())
            if nv > 0 and sv > 0:
                var.update(snr_planted_db=float(10 * np.log10(sv / nv)), snr_state=MEDIDO, signal_variance=sv, noise_variance=nv)
            else:
                var.update(snr_planted_db=None, snr_state=NO_APLICA, signal_variance=sv, noise_variance=nv)
            if nv > 0 and st > 0:
                var.update(snr_planted_total_db=float(10 * np.log10(st / nv)), snr_total_state=MEDIDO, signal_total_variance=st)
            else:
                var.update(snr_planted_total_db=None, snr_total_state=NO_APLICA, signal_total_variance=st)
            var["snr_versions"] = {"snr_planted_db": "v1 components s+periodic+cross (kept)", "snr_planted_total_db": "v2 total incl. deterministic"}
            var["acf"] = E.acf(v, E.ACF_LAGS)
            var["welch_bands"] = E.welch_bands(v)
            var["trend_seasonal_strength"] = E.DESCRIPTORS[int(descriptor_version)](v)
            var["quantiles"] = [float(q) for q in np.quantile(v, [0.0, 0.05, 0.5, 0.95, 1.0])]
            var["constant"] = bool(v.std() == 0)
            var["latent_group"] = params["latent_groups"][k]
            var["period_declared"] = params["groups"][params["latent_groups"][k]]["period"]
            var["phi_declared"] = params["groups"][params["latent_groups"][k]]["phi"]
            if k >= half:
                src = gen["source"][:, k]
                tau = int(params["tau"])
                c = _pearson(x[lo:hi, k], src[lo - tau:hi - tau])
                var.update(dependence_planted_lag=tau, dependence_planted_corr=c, dependence_state=MEDIDO if c is not None else NO_MEDIDO,
                           dependence_partner="latent partner" if params["r"] == 1 else "phantom (independent)")
            else:
                var.update(dependence_planted_lag=None, dependence_planted_corr=None, dependence_state=NO_APLICA, dependence_partner=None)
            entry["variables"][k] = var
        vs = list(entry["variables"].values())
        entry["aggregate"] = {"bytes_raw_total": int(sum(v["bytes_raw"] for v in vs)),
                              "compressed_bits_per_sample_zlib9_raw_float64_mean": float(np.mean([v["compressed_bits_per_sample_zlib9_raw_float64"] for v in vs])),
                              "compressed_bits_per_sample_lzma6_raw_float64_mean": float(np.mean([v["compressed_bits_per_sample_lzma6_raw_float64"] for v in vs])),
                              "compressed_bits_per_sample_zlib9_symbols_mean": float(np.mean([v["compressed_bits_per_sample_zlib9_symbols"] for v in vs])),
                              "h0_bits_mean": float(np.mean([v["h0_bits"] for v in vs if v["h0_bits"] is not None])) if any(v["h0_bits"] is not None for v in vs) else None,
                              "snr_planted_db_mean": float(np.mean([v["snr_planted_db"] for v in vs if v["snr_planted_db"] is not None])) if any(v["snr_planted_db"] is not None for v in vs) else None,
                              "snr_planted_total_db_mean": float(np.mean([v["snr_planted_total_db"] for v in vs if v["snr_planted_total_db"] is not None])) if any(v["snr_planted_total_db"] is not None for v in vs) else None,
                              "conditional_redundancy_bits_lag1_mean": float(np.mean([v["conditional_redundancy_bits_lag1"] for v in vs if v["conditional_state"] == MEDIDO]))
                                                                        if any(v["conditional_state"] == MEDIDO for v in vs) else None,
                              "conditional_measured_variables": int(sum(v["conditional_state"] == MEDIDO for v in vs)),
                              "dependence_planted_corr_mean_B": float(np.mean([v["dependence_planted_corr"] for v in vs if v["dependence_planted_corr"] is not None]))
                                                                 if any(v["dependence_planted_corr"] is not None for v in vs) else None}
        out["splits"][part] = entry
    out["support"] = {"boundaries": b, "n_total": int(x.shape[0]), "p": int(p), "sampling": "unit step, shared", "targets": "x[t + horizon], every variable"}
    return out


# --- model / graph -------------------------------------------------------------------------------------------

def matrix_descriptors(w: np.ndarray) -> dict:
    """Singular-value descriptors of a 2-D matrix; a zero (or empty) matrix is NOT_DEFINED."""
    m = np.asarray(w, dtype=float)
    if m.ndim == 3:                                        # Conv1D kernel (k, in, out) -> (k * in, out)
        m = m.reshape(-1, m.shape[-1])
    if m.ndim != 2 or m.size == 0:
        return {"state": NO_APLICA, "reason": f"not a matrix: shape {list(np.asarray(w).shape)}"}
    er, nr, s = PI.effective_rank(m)
    if er is None or s.max() <= 0:
        return {"state": NOT_DEFINED, "reason": "zero matrix: p = s / sum(s) undefined", "shape": list(m.shape), "frobenius": 0.0}
    return {"state": MEDIDO, "shape": list(m.shape), "effective_rank_entropy": er, "numerical_rank": nr,
            "stable_rank": float((s ** 2).sum() / s.max() ** 2), "nuclear_ratio": float(s.sum() / s.max()), "spectral_norm": float(s.max()),
            "frobenius": float(np.sqrt((s ** 2).sum())), "min_singular": float(s.min()), "singular_values": [float(v) for v in s[:8]]}


def graph_descriptors(model) -> dict:
    layers = list(model.layers)
    names = {l.name for l in layers}
    edges = 0
    for l in layers:
        for node in getattr(l, "_inbound_nodes", []):
            for t in (node.input_tensors if hasattr(node, "input_tensors") else []):
                hist = getattr(t, "_keras_history", None)
                if hist is not None and getattr(hist, "operation", None) is not None and hist.operation.name in names:
                    edges += 1
    n = len(layers)
    return {"nodes": n, "edges": int(edges), "directed": True, "density": (edges / (n * (n - 1))) if n > 1 else None,
            "spectral_radius": None, "spectral_radius_state": NO_APLICA, "spectral_radius_reason": "DAG adjacency: radius 0 by structure (uninformative)",
            "declaration": DECLARATIONS["graph"]}


def adapter_probe(model, adapter_suffix: str = "_adapt"):
    """One sub-model (built once) that returns every adapter output; None when there is none."""
    E = _load("df_mod_e0")
    tf = E._tf()
    outs = [l.output for l in model.layers if l.name.endswith(adapter_suffix)]
    names = [l.name for l in model.layers if l.name.endswith(adapter_suffix)]
    return (tf.keras.Model(model.input, outs), names) if outs else None


def model_metrics(model, *, grad_batch=None, act_batch=None, loss: str = "mse", adapter_suffix: str = "_adapt", probe=None,
                  serialized_bytes: bool = True) -> dict:
    """Parameters, bytes, per-layer norms, matrix descriptors, optional gradients/activations, graph.
    `probe`: a cached adapter_probe(model) (the callback builds it once); `serialized_bytes`: save the
    weights to a temporary file to measure their serialized length (skipped at intermediate checkpoints)."""
    E = _load("df_mod_e0")
    tf = E._tf()
    t0 = time.process_time()
    params = E.count_params(model)
    layers = {}
    frob_total_sq = 0.0
    matrices = []
    for l in model.layers:
        ws = l.get_weights()
        if not ws:
            continue
        entry = {"trainable": bool(l.trainable), "tensors": []}
        for w in ws:
            f = float(np.linalg.norm(np.asarray(w, dtype=float).ravel()))
            frob_total_sq += f * f
            t = {"shape": list(w.shape), "frobenius": f}
            if np.asarray(w).ndim >= 2:
                t["matrix"] = matrix_descriptors(w)
                if t["matrix"]["state"] == MEDIDO:
                    matrices.append(t["matrix"])
            entry["tensors"].append(t)
        layers[l.name] = entry
    out = {"schema": SCHEMA, "parameters": params, "bytes_raw_float32": int(4 * params["total"]),
           "layers": layers, "weight_frobenius_total": float(math.sqrt(frob_total_sq)),
           "matrices_measured": len(matrices), "matrices_not_defined": int(sum(1 for l in layers.values() for t in l["tensors"] if t.get("matrix", {}).get("state") == NOT_DEFINED)),
           "effective_rank_entropy_mean": float(np.mean([m["effective_rank_entropy"] for m in matrices])) if matrices else None,
           "stable_rank_mean": float(np.mean([m["stable_rank"] for m in matrices])) if matrices else None,
           "nuclear_ratio_mean": float(np.mean([m["nuclear_ratio"] for m in matrices])) if matrices else None,
           "spectral_norm_max": float(max(m["spectral_norm"] for m in matrices)) if matrices else None,
           "graph": graph_descriptors(model)}
    if serialized_bytes:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.weights.h5")
            model.save_weights(path)
            out["bytes_serialized_weights_h5"] = int(os.path.getsize(path))
    else:
        out["bytes_serialized_weights_h5"] = None
    if grad_batch is not None:
        Xb, yb = grad_batch
        loss_fn = tf.keras.losses.MeanSquaredError() if loss == "mse" else tf.keras.losses.MeanAbsoluteError()
        with tf.GradientTape() as tape:
            pred = model(tf.constant(Xb, dtype=tf.float32), training=True)
            value = loss_fn(tf.constant(yb, dtype=tf.float32), pred)
        grads = tape.gradient(value, model.trainable_variables)
        per = {}
        total = 0.0
        for var, g in zip(model.trainable_variables, grads):
            if g is None:
                continue
            n = float(np.linalg.norm(np.asarray(g).ravel()))
            per[var.path if hasattr(var, "path") else var.name] = n
            total += n * n
        out["gradients"] = {"state": MEDIDO, "global_norm": float(math.sqrt(total)), "loss_on_batch": float(value), "batch_rows": int(len(Xb)),
                            "per_variable": per, "declaration": DECLARATIONS["gradients"]}
    else:
        out["gradients"] = {"state": NO_MEDIDO, "reason": "no fixed batch supplied (backfill from weights cannot recover gradients of the past)"}
    if act_batch is not None:
        acts = {}
        probe = probe if probe is not None else adapter_probe(model, adapter_suffix)
        if probe is not None:
            sub, names = probe
            values = sub(tf.constant(act_batch, dtype=tf.float32), training=False)
            values = values if isinstance(values, (list, tuple)) else [values]
            for name, a in zip(names, values):
                a = np.asarray(a)
                acts[name] = {"mean": float(a.mean()), "std": float(a.std()), "near_zero_fraction": float((np.abs(a) < 1e-6).mean()),
                              "shape": list(a.shape)}
        out["activations"] = {"state": MEDIDO if acts else NO_APLICA, "adapters": acts, "batch_rows": int(len(act_batch)), "declaration": DECLARATIONS["activations"]}
    else:
        out["activations"] = {"state": NO_MEDIDO, "reason": "no fixed batch supplied"}
    out["cpu_seconds"] = round(time.process_time() - t0, 4)
    return out


def checkpoint_callback(model, *, grad_batch, act_batch, loss: str, schedule=CHECKPOINT_EPOCHS):
    """A Keras callback that measures the model descriptors at the initial state, at the scheduled
    epochs and at the final epoch (before any restore). Place it BEFORE EarlyStopping so that its
    on_train_end sees the last weights, not the restored ones."""
    E = _load("df_mod_e0")
    tf = E._tf()

    class Descriptors(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.checkpoints = {}
            self.schedule = tuple(int(s) for s in schedule)
            self.seconds = 0.0
            self.probe = adapter_probe(model)

        def _measure(self, key):
            t0 = time.process_time()
            self.checkpoints[key] = model_metrics(model, grad_batch=grad_batch, act_batch=act_batch, loss=loss, probe=self.probe,
                                                  serialized_bytes=key in ("initial", "final"))
            self.seconds += time.process_time() - t0

        def on_train_begin(self, logs=None):
            self._measure("initial")

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) in self.schedule:
                self._measure(f"epoch_{epoch + 1}")

        def on_train_end(self, logs=None):
            self._measure("final")
    return Descriptors()


def terminal_rows(rec: dict) -> tuple:
    """Governed metric rows (name, value, unit, split) for the cube from a cell record's data and
    model metrics; only finite values become rows, every other state travels in the tags."""
    rows, states = [], {}
    dm = rec.get("data_metrics") or {}
    for split, entry in (dm.get("splits") or {}).items():
        agg = entry.get("aggregate") or {}
        for gname, gm in (entry.get("grains") or {}).items():
            for name, unit in (("bytes_raw", "bytes"), ("compressed_bits_per_value_zlib9", "bits_per_value"), ("mask_non_finite", "count")):
                key = f"mod_e0.data.grain.{gname}.{name}"
                v = gm.get(name)
                if v is not None and np.isfinite(v):
                    rows.append((key, float(v), unit, split))
                    states[f"{key}@{split}"] = MEDIDO
                else:
                    states[f"{key}@{split}"] = NO_MEDIDO
            if gname == "window_tensor" and gm.get("repetition_factor") is not None:
                rows.append((f"mod_e0.data.grain.window_tensor.repetition_factor", float(gm["repetition_factor"]), "ratio", split))
                states[f"mod_e0.data.grain.window_tensor.repetition_factor@{split}"] = MEDIDO
        for name, unit in (("bytes_raw_total", "bytes"), ("compressed_bits_per_sample_zlib9_raw_float64_mean", "bits_per_sample"),
                           ("compressed_bits_per_sample_lzma6_raw_float64_mean", "bits_per_sample"), ("compressed_bits_per_sample_zlib9_symbols_mean", "bits_per_sample"),
                           ("h0_bits_mean", "bits"), ("snr_planted_db_mean", "dB"), ("snr_planted_total_db_mean", "dB"), ("conditional_redundancy_bits_lag1_mean", "bits"),
                           ("dependence_planted_corr_mean_B", "corr")):
            key = f"mod_e0.data.{name}"
            v = agg.get(name)
            if v is not None and np.isfinite(v):
                rows.append((key, float(v), unit, split))
                states[f"{key}@{split}"] = MEDIDO
            elif name.startswith("conditional") and agg.get("conditional_measured_variables") == 0:
                states[f"{key}@{split}"] = INCONCLUSIVE
            else:
                states[f"{key}@{split}"] = NO_APLICA if name.startswith("dependence") or name.startswith("snr") else NO_MEDIDO
        for k, var in (entry.get("variables") or {}).items():
            for name, unit, st in (("h0_bits", "bits", var.get("h0_state")), ("snr_planted_db", "dB", var.get("snr_state")),
                                   ("snr_planted_total_db", "dB", var.get("snr_total_state")),
                                   ("compressed_bits_per_sample_zlib9_raw_float64", "bits_per_sample", MEDIDO),
                                   ("conditional_redundancy_bits_lag1", "bits", var.get("conditional_state")),
                                   ("dependence_planted_corr", "corr", var.get("dependence_state"))):
                key = f"mod_e0.data.var{k}.{name}"
                v = var.get(name)
                if v is not None and np.isfinite(v):
                    rows.append((key, float(v), unit, split))
                    states[f"{key}@{split}"] = MEDIDO
                else:
                    states[f"{key}@{split}"] = st or NO_MEDIDO
    mm = rec.get("model_metrics") or {}
    for ck, m in (mm.get("checkpoints") or {}).items():
        if not isinstance(m, dict) or m.get("state") == NO_MEDIDO:
            states[f"mod_e0.model.checkpoint@{ck}"] = NO_MEDIDO
            continue
        for name, unit in (("weight_frobenius_total", "norm"), ("effective_rank_entropy_mean", "rank"), ("stable_rank_mean", "rank"),
                           ("nuclear_ratio_mean", "ratio"), ("spectral_norm_max", "norm"), ("bytes_serialized_weights_h5", "bytes")):
            key = f"mod_e0.model.{name}"
            v = m.get(name)
            if v is not None and np.isfinite(v):
                rows.append((key, float(v), unit, ck))
                states[f"{key}@{ck}"] = MEDIDO
            else:
                states[f"{key}@{ck}"] = NO_MEDIDO
        g = m.get("gradients") or {}
        if g.get("state") == MEDIDO:
            rows.append(("mod_e0.model.gradient_global_norm", float(g["global_norm"]), "norm", ck))
            states[f"mod_e0.model.gradient_global_norm@{ck}"] = MEDIDO
        else:
            states[f"mod_e0.model.gradient_global_norm@{ck}"] = g.get("state", NO_MEDIDO)
        a = m.get("activations") or {}
        if a.get("state") == MEDIDO and a.get("adapters"):
            rows.append(("mod_e0.model.activation_near_zero_fraction_mean", float(np.mean([v["near_zero_fraction"] for v in a["adapters"].values()])), "fraction", ck))
            states[f"mod_e0.model.activation_near_zero_fraction_mean@{ck}"] = MEDIDO
        else:
            states[f"mod_e0.model.activation_near_zero_fraction_mean@{ck}"] = a.get("state", NO_MEDIDO)
    prm = (mm.get("checkpoints") or {}).get("best", {}).get("parameters") or rec.get("parameters") or {}
    for name in ("total", "trainable", "frozen"):
        if prm.get(name) is not None:
            rows.append((f"mod_e0.model.params_{name}", float(prm[name]), "count", "model"))
            states[f"mod_e0.model.params_{name}@model"] = MEDIDO
    g = (mm.get("checkpoints") or {}).get("best", {}).get("graph") or {}
    for name, unit in (("nodes", "count"), ("edges", "count"), ("density", "ratio")):
        if g.get(name) is not None:
            rows.append((f"mod_e0.model.graph_{name}", float(g[name]), unit, "model"))
            states[f"mod_e0.model.graph_{name}@model"] = MEDIDO
    states["mod_e0.model.graph_spectral_radius@model"] = NO_APLICA
    return rows, states
