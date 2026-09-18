#!/usr/bin/env python3
"""Real temporal learners for the adequacy pilot (S2): ridge baseline, explicit multilayer causal
Conv1D/TCN, explicit LSTM, plus the task baselines and the analytic known-frequency oracle.

Every cell reads the SAME rows and labels for every model (prepared once from the frozen
boundaries), scales with train-only statistics, fits on training rows only, early-stops on the
inner validation, and predicts the untouched test. The record of a cell carries the model graph
(layers, parameters, receptive field, input support, state policy), the optimizer, the observed
updates (count and the weight change they produced), train/validation curves, a diagnosis
(FITTED / UNDERFIT / OVERFIT / OPTIMIZATION_FAILURE), predictions, labels, row ids, baseline and
oracle predictions, losses recomputed from the arrays, and cost. CPU only.

    python tools/df_adequacy_models.py --worker JOB.json      (child protocol: result.json at the end)
"""

from __future__ import annotations

import argparse
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


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load("df_adequacy_design")

CELL_SCHEMA = "df_adequacy_cell.v1"
FITTED, UNDERFIT, OVERFIT, OPT_FAIL = "FITTED", "UNDERFIT", "OVERFIT", "OPTIMIZATION_FAILURE"


# --- data: tasks, windows, boundaries, train-only scaling ----------------------------------------------

def array_digest(a: np.ndarray) -> str:
    """The bank's own scheme: sha256 over {dtype, shape} + raw C-order bytes (df_synthetic_bank)."""
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True).encode())
    h.update(b"\n")
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_unit_arrays(bank: Path, unit: str) -> dict:
    """The consumed arrays, bound to the frozen metadata by RECOMPUTING their digest with the
    bank's scheme (T2): one changed byte under unchanged metadata refuses before anything else."""
    d = Path(bank) / unit
    observed_raw = np.load(d / "observed_signal.npy", allow_pickle=False)
    clean_raw = np.load(d / "clean_signal.npy", allow_pickle=False)
    meta = D.unit_metadata(bank, unit)
    binding = {"observed": array_digest(observed_raw), "clean": array_digest(clean_raw)}
    for k in ("observed", "clean"):
        if binding[k] != meta["digests"][k]:
            raise ValueError(f"{unit}: {k} array digest {binding[k][:12]} is not the frozen {meta['digests'][k][:12]} (digest binding)")
    observed = observed_raw.astype(float)
    clean = clean_raw.astype(float)
    if observed.ndim == 2:
        observed, clean = observed[0], clean[0]
    if not (np.isfinite(observed).all() and np.isfinite(clean).all()):
        raise ValueError(f"{unit}: arrays carry non-finite values")
    return {"observed": observed, "clean": clean, "meta": meta, "binding": binding}


def task_series_full(arrays: dict, task: str) -> dict:
    """Input series, label at t+1 per row t, baseline, the CONDITIONAL oracle, the old oracle (history)
    and the noise floor, all declared per task (T1)."""
    spec = D.TASKS[task]
    clean = np.asarray(arrays["clean"], dtype=float)
    observed = np.asarray(arrays["observed"], dtype=float)
    if not (np.isfinite(clean).all() and np.isfinite(observed).all()):
        raise ValueError("non-finite values in the consumed arrays")
    x = clean if spec["input"] == "clean" else observed
    P = arrays["meta"]["period"]
    c = 2.0 * math.cos(2.0 * math.pi / P)
    n = x.size
    label = np.full(n, np.nan)
    baseline = np.full(n, np.nan)
    oracle = np.full(n, np.nan)
    oracle_old = None
    rec_level = np.full(n, np.nan)
    rec_level[1:n - 1] = c * clean[1:n - 1] - clean[0:n - 2]          # the recurrence's clean[t+1] at row t
    if task == "clean_next_level":
        label[:-1] = clean[1:]
        baseline[:-1] = clean[:-1]
        oracle = rec_level
    elif task == "clean_increment":
        label[:-1] = clean[1:] - clean[:-1]
        baseline[:] = 0.0
        oracle[1:n - 1] = rec_level[1:n - 1] - clean[1:n - 1]
    elif task == "observed_increment":
        label[:-1] = observed[1:] - observed[:-1]
        baseline[:] = 0.0
        oracle[1:n - 1] = rec_level[1:n - 1] - observed[1:n - 1]        # conditional on the CURRENT observation: residual = next noise
        oracle_old = np.full(n, np.nan)
        oracle_old[1:n - 1] = rec_level[1:n - 1] - clean[1:n - 1]       # the S1 oracle, kept for history only
    else:
        raise ValueError(task)
    sd = float(arrays["meta"].get("noise_sd") or 0.0)
    floor = {"expected_gaussian_mae": sd * math.sqrt(2.0 / math.pi), "noise_sd_declared": sd,
             "note": "neither the expected noise-only MAE nor the realised oracle MAE is a samplewise lower bound; "
                     "both fluctuate; reported apart"}
    return {"x": x, "label": label, "baseline": baseline, "oracle": oracle, "oracle_old": oracle_old, "noise_floor": floor}


def task_series(arrays: dict, task: str) -> tuple:
    """(input series, label, baseline, conditional oracle) — see task_series_full."""
    f = task_series_full(arrays, task)
    return f["x"], f["label"], f["baseline"], f["oracle"]


def windows(x: np.ndarray, rows: np.ndarray, window: int) -> np.ndarray:
    """Row t consumes x[t-W+1 .. t] (the past only); shape (len(rows), W)."""
    return np.stack([x[t - window + 1:t + 1] for t in rows])


class TestAccessDenied(PermissionError):
    """The test split is not available to this role (cost pilots never score it)."""


def part(prep: dict, name: str) -> dict:
    if name == "test" and "test" not in prep["parts"]:
        raise TestAccessDenied("test rows are not accessible in this role (NO_TEST_ACCESS)")
    return prep["parts"][name]


def prepare(arrays: dict, task: str, window: int, train_length: int, horizon: int = D.HORIZON, *, test_access: bool = True) -> dict:
    """Model-free preparation: shared decision rows asserted against the design's geometry
    identity (T1), train-only scaling, and the test split only when the role allows it (T2)."""
    b = D.boundaries(window, train_length, horizon)
    ident = D.geometry_identity()
    if tuple(b["validation"]) != ident["validation"] or tuple(b["test"]) != ident["test"] or b["cutoff"] != ident["cutoff"]:
        raise ValueError("prepared rows differ from the shared decision-row geometry")
    full = task_series_full(arrays, task)
    x, label, baseline, oracle, oracle_old = full["x"], full["label"], full["baseline"], full["oracle"], full["oracle_old"]
    parts = {}
    for name in ("train", "validation") + (("test",) if test_access else ()):
        lo, hi = b[name]
        rows = np.arange(lo, hi)
        rows = rows[~np.isnan(label[rows])]
        parts[name] = {"rows": rows, "X": windows(x, rows, window), "y": label[rows],
                       "baseline": baseline[rows], "oracle": oracle[rows],
                       "oracle_old": oracle_old[rows] if oracle_old is not None else None}
    xm, xs = parts["train"]["X"].mean(), parts["train"]["X"].std() or 1.0          # TRAIN-ONLY statistics
    ym, ys = parts["train"]["y"].mean(), parts["train"]["y"].std() or 1.0
    last = parts["test"]["rows"][-1] if test_access else parts["validation"]["rows"][-1]
    return {"boundaries": b, "parts": parts, "scale": {"x_mean": float(xm), "x_sd": float(xs), "y_mean": float(ym), "y_sd": float(ys)},
            "consumed": {"first_row": int(parts["train"]["rows"][0] - window + 1), "last_row": int(last)},
            "noise_floor": full["noise_floor"], "test_access": test_access}


def _scale_x(X, s):
    return (X - s["x_mean"]) / s["x_sd"]


def _scale_y(y, s):
    return (y - s["y_mean"]) / s["y_sd"]


def _unscale_y(z, s):
    return z * s["y_sd"] + s["y_mean"]


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# --- models -----------------------------------------------------------------------------------------------

class Ridge:
    KIND = "ridge"

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.w = None

    def fit(self, X, y, Xv, yv, *, seed, training):
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        reg = self.lam * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        self.w = np.linalg.solve(A.T @ A + reg, A.T @ y)
        curve = {"train": [mae(self.predict(X), y)], "validation": [mae(self.predict(Xv), yv)]}
        return {"updates": 1, "epochs": 1, "curve": curve, "weight_change_norm": float(np.linalg.norm(self.w)),
                "parameters": int(self.w.size), "layers": [{"type": "linear", "inputs": int(X.shape[1]), "lambda": self.lam}],
                "receptive_field": int(X.shape[1]), "optimizer": "closed_form", "state_policy": "none",
                "stop_reason": "CLOSED_FORM", "restored_checkpoint_epoch": 1, "deterministic_predict": True}

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.w

    def save(self, path: Path):
        np.save(path, self.w)

    def load(self, path: Path):
        self.w = np.load(path)


def _tf():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


class KerasModel:
    KIND = "keras"

    def __init__(self, window: int):
        self.window = window
        self.model = None
        self.graph = None

    def build(self, seed: int):
        raise NotImplementedError

    def fit(self, X, y, Xv, yv, *, seed, training):
        tf = _tf()
        tf.keras.utils.set_random_seed(int(seed))
        self.model = self.build(seed)
        opt = tf.keras.optimizers.Adam(learning_rate=training["learning_rate"])
        self.model.compile(optimizer=opt, loss="mae")
        before = np.concatenate([w.ravel() for w in self.model.get_weights()])
        batch = int(training["batch"])
        steps_per_epoch = math.ceil(X.shape[0] / batch)
        max_epochs = min(int(training["max_epochs"]), max(1, int(training["max_updates"]) // steps_per_epoch))
        counter = _UpdateCounter()
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(training["early_stopping"]["patience"]),
                                              restore_best_weights=bool(training["early_stopping"]["restore_best"]))
        hist = self.model.fit(X[..., None], y, validation_data=(Xv[..., None], yv), epochs=max_epochs, batch_size=batch,
                              shuffle=True, verbose=0, callbacks=[counter, es])
        after = np.concatenate([w.ravel() for w in self.model.get_weights()])
        epochs = len(hist.history["loss"])
        va = [float(v) for v in hist.history["val_loss"]]
        if es.stopped_epoch:
            stop_reason = "EARLY_STOPPING"
        elif epochs >= max_epochs and int(training["max_updates"]) // steps_per_epoch <= int(training["max_epochs"]):
            stop_reason = "UPDATE_BUDGET"
        else:
            stop_reason = "EPOCH_BUDGET"
        best_epoch = int(np.argmin(va)) + 1 if va else None
        parity = bool(np.allclose(self.model.predict(Xv[..., None], verbose=0).ravel(), self.model.predict(Xv[..., None], verbose=0).ravel()))
        return {"updates": int(counter.updates), "epochs": epochs,
                "curve": {"train": [float(v) for v in hist.history["loss"]], "validation": va},
                "weight_change_norm": float(np.linalg.norm(after - before)),
                "parameters": int(self.model.count_params()), "layers": self.graph["layers"],
                "receptive_field": self.graph["receptive_field"], "optimizer": f"adam(lr={training['learning_rate']})",
                "state_policy": self.graph["state_policy"], "steps_per_epoch": steps_per_epoch, "max_epochs_allowed": max_epochs,
                "stop_reason": stop_reason, "restored_checkpoint_epoch": best_epoch if training["early_stopping"]["restore_best"] else None,
                "budget_vs_early_stopping": {"max_updates": int(training["max_updates"]), "max_epochs": int(training["max_epochs"]),
                                             "patience": int(training["early_stopping"]["patience"])},
                "deterministic_predict": parity}

    def predict(self, X):
        return self.model.predict(X[..., None], verbose=0, batch_size=256).ravel()

    def save(self, path: Path):
        self.model.save_weights(str(path))

    def load(self, path: Path, seed: int = 0):
        if self.model is None:
            self.model = self.build(seed)
        self.model.load_weights(str(path))


def _UpdateCounter():
    """A Keras callback counting real optimizer updates (train batches)."""
    tf = _tf()

    class Counter(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.updates = 0

        def on_train_batch_end(self, batch, logs=None):
            self.updates += 1
    return Counter()


class CausalConv1D(KerasModel):
    KIND = "causal_conv1d"

    def build(self, seed: int):
        tf = _tf()
        dilations = D.conv_dilations_fixed()                       # the same graph at every W (T1)
        inp = tf.keras.Input(shape=(self.window, 1))
        h = inp
        layers = []
        for d in dilations:
            h = tf.keras.layers.Conv1D(16, 3, padding="causal", dilation_rate=d, activation="relu")(h)
            layers.append({"type": "Conv1D", "filters": 16, "kernel": 3, "dilation": d, "padding": "causal", "activation": "relu"})
        last = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(h)
        out = tf.keras.layers.Dense(1)(last)
        layers += [{"type": "last_position"}, {"type": "Dense", "units": 1}]
        rf = D.receptive_field(3, dilations)
        self.graph = {"layers": layers, "receptive_field": rf, "effective_support": min(rf, self.window), "state_policy": "none",
                      "padding": "causal", "fixed_across_W": True}
        return tf.keras.Model(inp, out)


class LSTMModel(KerasModel):
    KIND = "lstm"

    def build(self, seed: int):
        tf = _tf()
        inp = tf.keras.Input(shape=(self.window, 1))
        h = tf.keras.layers.LSTM(32, stateful=False)(inp)
        out = tf.keras.layers.Dense(1)(h)
        self.graph = {"layers": [{"type": "LSTM", "units": 32, "stateful": False}, {"type": "Dense", "units": 1}],
                      "receptive_field": self.window, "effective_support": self.window, "state_policy": "RESET_PER_WINDOW"}
        return tf.keras.Model(inp, out)


def build_model(kind: str, window: int):
    if kind == "ridge":
        return Ridge(D.MODELS["ridge"]["lambda"])
    if kind == "causal_conv1d":
        return CausalConv1D(window)
    if kind == "lstm":
        return LSTMModel(window)
    raise ValueError(kind)


# --- diagnosis -----------------------------------------------------------------------------------------

def diagnose(fit: dict, baseline_train_mae_scaled: float) -> dict:
    """Heuristic fit classes, distinct from scientific adequacy (T2). OPTIMIZATION_FAILURE needs
    hard evidence (non-finite loss, or no update / no weight change); a single loss trend is a
    flag, never a failure verdict."""
    note = "heuristic fit class from curves and updates; NOT a scientific adequacy verdict"
    tr, va = fit["curve"]["train"], fit["curve"]["validation"]
    flags = []
    if not tr or not all(math.isfinite(v) for v in tr + va):
        return {"class": OPT_FAIL, "why": "non-finite loss", "flags": ["NON_FINITE"], "note": note}
    if fit["updates"] <= 0 or fit["weight_change_norm"] == 0.0:
        return {"class": OPT_FAIL, "why": "no optimizer update changed the weights", "flags": ["NO_UPDATE"], "note": note}
    if len(tr) >= 10 and tr[9] >= tr[0]:
        flags.append("NO_EARLY_PROGRESS")
    if fit.get("stop_reason") in ("UPDATE_BUDGET", "EPOCH_BUDGET"):
        flags.append("STOPPED_BY_BUDGET")
    best_va = min(va)
    if va[-1] > 1.2 * best_va and tr[-1] < tr[max(0, va.index(best_va))]:
        return {"class": OVERFIT, "why": f"validation rose {va[-1] / best_va:.2f}x over its minimum while training kept falling",
                "flags": flags, "note": note}
    if tr[-1] > 0.9 * baseline_train_mae_scaled:
        return {"class": UNDERFIT, "why": f"final training MAE {tr[-1]:.3f} is within 10 % of the baseline's {baseline_train_mae_scaled:.3f}",
                "flags": flags, "note": note}
    return {"class": FITTED, "why": "training loss fell below the baseline and validation did not diverge", "flags": flags, "note": note}


# --- one cell ------------------------------------------------------------------------------------------

def run_cell(job: dict, out_dir: Path) -> dict:
    t0 = time.process_time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_unit_arrays(Path(job["bank"]), job["unit"])                   # digest binding + finiteness first (T2)
    test_access = job.get("role", "CELL") != "COST_PILOT"
    prep = prepare(arrays, job["task"], int(job["window"]), int(job["train_length"]), int(job.get("horizon", D.HORIZON)),
                   test_access=test_access)
    s = prep["scale"]
    P = prep["parts"]
    names = ("train", "validation") + (("test",) if test_access else ())
    training = dict(job["training"])
    if job.get("max_updates_override"):
        training["max_updates"] = int(job["max_updates_override"])
    model = build_model(job["model"], int(job["window"]))
    np.random.seed(int(job["seed"]))
    fit = model.fit(_scale_x(P["train"]["X"], s), _scale_y(P["train"]["y"], s),
                    _scale_x(P["validation"]["X"], s), _scale_y(P["validation"]["y"], s),
                    seed=int(job["seed"]), training=training)
    baseline_train_scaled = mae(_scale_y(P["train"]["baseline"], s), _scale_y(P["train"]["y"], s))
    diagnosis = diagnose(fit, baseline_train_scaled)
    preds = {name: _unscale_y(model.predict(_scale_x(P[name]["X"], s)), s) for name in names}
    weights_path = out_dir / ("weights.npy" if job["model"] == "ridge" else "weights.weights.h5")
    model.save(weights_path)
    reloaded = build_model(job["model"], int(job["window"]))
    reloaded.load(weights_path, **({} if job["model"] == "ridge" else {"seed": int(job["seed"])}))
    parity = bool(np.allclose(_unscale_y(reloaded.predict(_scale_x(P["validation"]["X"], s)), s), preds["validation"], atol=1e-6))
    fit["prediction_parity_after_reload"] = parity
    losses = {}
    for name in names:
        y = P[name]["y"]
        losses[name] = {"model": mae(preds[name], y), "baseline": mae(P[name]["baseline"], y),
                        "oracle": mae(P[name]["oracle"], y) if not np.isnan(P[name]["oracle"]).any() else None,
                        "oracle_old": (mae(P[name]["oracle_old"], y) if P[name]["oracle_old"] is not None
                                       and not np.isnan(P[name]["oracle_old"]).any() else None),
                        "rows": int(y.size)}
    val = losses["validation"]
    skill_validation = 1.0 - val["model"] / val["baseline"] if val["baseline"] > 0 else None
    skill = block_skill = None
    if test_access:
        test = losses["test"]
        skill = 1.0 - test["model"] / test["baseline"] if test["baseline"] > 0 else None
        yt, pt, bt = P["test"]["y"], preds["test"], P["test"]["baseline"]
        blocks = np.array_split(np.arange(yt.size), 4)
        block_skill = [1.0 - mae(pt[b], yt[b]) / mae(bt[b], yt[b]) if mae(bt[b], yt[b]) > 0 else None for b in blocks]
    saved = {}
    for n in names:
        saved.update({f"{n}_rows": P[n]["rows"], f"{n}_y": P[n]["y"], f"{n}_pred": preds[n], f"{n}_baseline": P[n]["baseline"],
                      f"{n}_oracle": P[n]["oracle"]})
        if P[n]["oracle_old"] is not None:
            saved[f"{n}_oracle_old"] = P[n]["oracle_old"]
    np.savez(out_dir / "arrays.npz", **saved)
    meta = arrays["meta"]
    record = {"schema": CELL_SCHEMA, "cell_id": job["cell_id"], "unit": job["unit"], "task": job["task"], "model": job["model"],
              "window": int(job["window"]), "train_length": int(job["train_length"]), "seed": int(job["seed"]), "horizon": int(job.get("horizon", D.HORIZON)),
              "period": meta["period"], "W_over_P": int(job["window"]) / meta["period"], "consumed_span_over_P": (int(job["window"]) - 1) / meta["period"],
              "boundaries": prep["boundaries"], "geometry_identity": {k: (list(v) if isinstance(v, tuple) else v) for k, v in D.geometry_identity().items()},
              "consumed": prep["consumed"], "scale": s, "binding": arrays["binding"],
              "graph": {k: fit[k] for k in ("layers", "parameters", "receptive_field", "optimizer", "state_policy")} | {"padding": "causal" if job["model"] == "causal_conv1d" else None},
              "effective_support": min(fit["receptive_field"], int(job["window"])),
              "training": {k: fit[k] for k in ("updates", "epochs", "weight_change_norm", "stop_reason", "restored_checkpoint_epoch",
                                               "budget_vs_early_stopping", "deterministic_predict", "prediction_parity_after_reload") if k in fit} | {"rule": training},
              "curve": fit["curve"], "diagnosis": diagnosis, "losses": losses, "skill_test": skill, "skill_validation": skill_validation,
              "block_skill_test": block_skill, "noise_floor": prep["noise_floor"],
              "exposure": "TEST_SCORED_DESCRIPTIVE" if test_access else "NO_TEST_ACCESS", "role": job.get("role", "CELL"),
              "oracle_role": "diagnostic, truth-derived (known period); conditional on the current observation for the observed task; never a learner input",
              "arrays_sha256": hashlib.sha256((out_dir / "arrays.npz").read_bytes()).hexdigest(),
              "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(), "weights_file": weights_path.name,
              "unit_digests": meta["digests"], "cost": {"cpu_seconds": round(time.process_time() - t0, 3)}}
    body = json.dumps(record, sort_keys=True, default=float).encode()
    (out_dir / "cell.json").write_bytes(body)
    return record


def worker_main(job_file: Path) -> int:
    job = json.loads(Path(job_file).read_text())
    adir = Path(job["attempt_dir"])
    record = run_cell(job, adir)
    body = (adir / "cell.json").read_bytes()
    result = {"status": "COMPLETED", "reason": "", "output_file": "cell.json", "output_sha256": hashlib.sha256(body).hexdigest(),
              "rows_written": 1, "outcome": record["diagnosis"]["class"]}
    tmp = adir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    os.replace(tmp, adir / "result.json")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args(argv)
    return worker_main(args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
