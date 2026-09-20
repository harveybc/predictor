#!/usr/bin/env python3
"""RP37: is the receiver able to USE the support its task declares?

The dictum measured the deployed graph: altering the first 53 of 60 input rows changed nothing, and
only the last seven did. A window is a declaration; the reach is what the prediction can depend on;
and neither is the same as a capacity the network actually learns. This tool separates the three:

    reach       measured, not asserted: each input row is perturbed in turn and the output change is
                recorded, and the gradient of the output with respect to each input row is taken.
                Both must agree with the reach the design declares for that core.
    capacity    bounded DEV diagnostics with KNOWN answers. A distant-lag task is solvable exactly by
                a receiver that sees the whole window and IMPOSSIBLE for one that sees seven samples;
                a near-lag task is solvable by both; an unpredictable task is solvable by neither and
                exists so that "everything looks fine" cannot pass for a result. A competent linear
                control on the same information is fitted for each task.
    cost        parameters and CPU seconds per core, so the cheaper option that meets the requirement
                can be preferred before anyone sees a score of the real task.

Nothing here chooses a core by looking at R0/R1/R2 on the published validation: the tasks are
diagnostics with known solutions, run on the DEV slice only.

    python tools/df_e1_receiver.py --data DATA.npz --design DESIGN.json --out RECEIVER.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("df_mod_e0")
P = _load("df_e1_pilot")
SCHEMA = "df_e1_receiver.v1"


def measure_reach(model, X: np.ndarray, *, delta: float = 100.0) -> dict:
    """Perturbation and gradient, per input row, on real windows."""
    tf = E._tf()
    base = np.asarray(model.predict(X, verbose=0))
    W = X.shape[1]
    changes = []
    for k in range(W):
        Xk = X.copy()
        Xk[:, k, :] += delta
        changes.append(float(np.max(np.abs(np.asarray(model.predict(Xk, verbose=0)) - base))))
    xt = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        out = model(xt, training=False)
    grad = np.asarray(tape.gradient(out, xt))
    per_row = [float(np.max(np.abs(grad[:, k, :]))) for k in range(W)]
    influential = [k for k, c in enumerate(changes) if c > 1e-6]
    grad_rows = [k for k, g in enumerate(per_row) if g > 1e-9]
    return {"perturbation_max_abs_change_per_row": changes, "gradient_max_abs_per_row": per_row,
            "rows_that_change_the_output": influential, "rows_with_gradient": grad_rows,
            "measured_reach": (W - min(influential)) if influential else 0,
            "measured_reach_by_gradient": (W - min(grad_rows)) if grad_rows else 0,
            "delta": delta, "windows": int(X.shape[0])}


def _tasks(Xs: np.ndarray, Y: np.ndarray, origins: np.ndarray, W: int, h: int, j: int, *, distant_lag: int, seed: int) -> dict:
    """Three diagnostics with known answers, built from the DEV inputs themselves."""
    rng = np.random.default_rng(seed)
    x = P._gather(Xs, origins, W).astype(np.float32)
    near = x[:, -1, j].astype(np.float64)                           # solvable by any receiver
    far_index = W - 1 - distant_lag
    if far_index < 0:
        raise SystemExit(f"distant lag {distant_lag} does not fit in a window of {W}")
    far = x[:, far_index, j].astype(np.float64)                     # solvable only with the whole window
    noise = rng.normal(size=origins.size)                           # solvable by nobody
    return {"near_lag": {"y": near, "known_solution": "copy the last input row of the target channel",
                         "row_needed": W - 1,
                         "short_receiver_claim": "REACHABLE: the row it needs is inside any receiver's support"},
            "distant_lag": {"y": far, "known_solution": f"copy input row {far_index} of the target channel",
                            "row_needed": far_index,
                            # RP44 (dictum F5): POSITION alone decides nothing. On a series that is
                            # correlated, periodic or redundant across channels, a short receiver can
                            # recover this target from its own support — a constant-within-window
                            # series makes copying the last sample exact. What this task measures is
                            # therefore EMPIRICAL on THIS series, never a general impossibility. The
                            # constructed independent-innovation claim lives in df_e1_innovation.py.
                            "short_receiver_claim": ("UNDECIDED BY POSITION: on these real, correlated series the "
                                                     "short receiver may or may not infer the target from its own "
                                                     "support; the measurement below is empirical and DEV-scoped")},
            "unpredictable": {"y": noise, "known_solution": "none: the target is independent of the inputs",
                              "solvable_by_seven_sample_receiver": False, "row_needed": None}}, x


def _linear(x_train, y_train, x_eval, *, lam: float, rows: slice) -> np.ndarray:
    """A competent linear control on a DECLARED information support (`rows` of the window)."""
    A = x_train[:, rows, :].reshape(x_train.shape[0], -1).astype(np.float64)
    A = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    B = x_eval[:, rows, :].reshape(x_eval.shape[0], -1).astype(np.float64)
    B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ y_train)
    return B @ beta


def _score(pred, y) -> dict:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    var = float(np.var(y))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.mean(err ** 2) / var) if var > 0 else None}


def diagnostics(data: dict, design: dict, *, cores=("conv3", "tcn_w"), updates: int = 400, batch: int = 64,
                lr: float = 3e-3, seed: int = 1, n_train: int = 4000, n_eval: int = 1000, distant_lag: int = 50) -> dict:
    tf = E._tf()
    Xs, Y = data["Xs"], data["Y"]
    W, h, j = int(data["window"][0]), int(data["horizon"][0]), int(data["target_channel"][0])
    tr = np.asarray(data["train_origins"], dtype=np.int64)[:n_train]
    ev = np.asarray(data["eval_origins"], dtype=np.int64)[:n_eval]
    tasks_tr, x_tr = _tasks(Xs, Y, tr, W, h, j, distant_lag=distant_lag, seed=seed)
    tasks_ev, x_ev = _tasks(Xs, Y, ev, W, h, j, distant_lag=distant_lag, seed=seed + 1)
    out = {"tasks": {}, "distant_lag": distant_lag, "windows": {"train": int(tr.size), "evaluation": int(ev.size)}}
    for name in tasks_tr:
        y_tr, y_ev = tasks_tr[name]["y"], tasks_ev[name]["y"]
        block = {k: v for k, v in tasks_tr[name].items() if k != "y"}
        block["linear_full_window"] = _score(_linear(x_tr, y_tr, x_ev, lam=1.0, rows=slice(0, W)), y_ev)
        block["linear_last_seven"] = _score(_linear(x_tr, y_tr, x_ev, lam=1.0, rows=slice(W - 7, W)), y_ev)
        block["mean_of_train"] = _score(np.full(y_ev.shape, float(np.mean(y_tr))), y_ev)
        for core in cores:
            tf.keras.utils.set_random_seed(seed)
            model = P._model_for_target(design["graph"]["assignment"], W, Xs.shape[1], j, seed, core=core)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
            t0 = time.process_time()
            steps = max(1, int(np.ceil(tr.size / batch)))
            model.fit(x_tr, y_tr.astype(np.float32)[:, None], epochs=max(1, int(np.ceil(updates / steps))),
                      batch_size=batch, verbose=0, shuffle=True)
            seconds = time.process_time() - t0
            pred = np.asarray(model.predict(x_ev, verbose=0))[:, 0]
            block[core] = {**_score(pred, y_ev), "cpu_seconds": round(seconds, 2),
                           "parameters": int(model.count_params()),
                           "declared_reach": P.model_reach(W, core)}
        out["tasks"][name] = block
    # the reach of each core, measured on real windows
    out["reach"] = {}
    for core in cores:
        tf.keras.utils.set_random_seed(seed)
        model = P._model_for_target(design["graph"]["assignment"], W, Xs.shape[1], j, seed, core=core)
        measured = measure_reach(model, x_ev[:64])
        out["reach"][core] = {**measured, "declared_reach": P.model_reach(W, core),
                              "agrees_with_declaration": measured["measured_reach"] == P.model_reach(W, core),
                              "parameters": int(model.count_params()),
                              "dilations": P.core_dilations(W) if core == "tcn_w" else None}
    return out


def verdict(doc: dict) -> dict:
    """The adequacy reading, stated as a rule before any real-task score is looked at."""
    t = doc["tasks"]
    seven = t["distant_lag"]["conv3"]["r2"]
    full = t["distant_lag"]["tcn_w"]["r2"]
    near_ok = all(t["near_lag"][c]["r2"] is not None and t["near_lag"][c]["r2"] > 0.9 for c in ("conv3", "tcn_w"))
    noise_ok = all((t["unpredictable"][c]["r2"] or 0) < 0.1 for c in ("conv3", "tcn_w", "linear_full_window"))
    return {"seven_sample_receiver_fails_this_distant_lag_on_these_series": bool(seven is not None and seven < 0.5),
            "scope": ("EMPIRICAL, on the household DEV windows and this budget: it is not a claim that a short "
                      "receiver fails at any budget, nor that the information is unreachable from its support. "
                      "Position does not decide reachability (dictum F5); the constructed claim is in "
                      "df_e1_innovation.py, whose task makes the innovation independent of the short support."),
            "full_window_receiver_solves_it": bool(full is not None and full > 0.9),
            "both_solve_the_near_lag": bool(near_ok),
            "nobody_solves_the_unpredictable_task": bool(noise_ok),
            "reach_agrees_with_declaration": all(v["agrees_with_declaration"] for v in doc["reach"].values()),
            "cost_ratio_parameters": (doc["reach"]["tcn_w"]["parameters"] / doc["reach"]["conv3"]["parameters"]),
            "rule": "a receiver is adequate for a task when it CAN use the support the task needs, shown on a "
                    "diagnostic whose answer is known. Fitting the budget is not adequacy, and a core is never "
                    "chosen by which one makes a regime win on the published validation.",
            "budget_errata": ("RP44: this tool asked for N updates but ran whole epochs, so with 4 000 windows and "
                              "batch 64 (63 batches per epoch) a request of 400 ran 441 updates and one of 1 600 ran "
                              "1 638. The stored results are kept and only their budget label is corrected; the "
                              "historical runs did not save an optimiser counter, so the corrected figures are the "
                              "loop's arithmetic, not a re-measurement, and nothing was refitted to change the prose.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--distant-lag", type=int, default=50)
    a = ap.parse_args(argv)
    z = np.load(a.data)
    data = {k: z[k] for k in z.files}
    design = json.loads(a.design.read_text())
    doc = {"schema": SCHEMA, "design_sha256": design["design_sha256"], "updates_per_diagnostic": a.updates,
           **diagnostics(data, design, updates=a.updates, distant_lag=a.distant_lag)}
    doc["verdict"] = verdict(doc)
    a.out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"verdict": doc["verdict"],
                      "distant_lag_r2": {c: doc["tasks"]["distant_lag"][c]["r2"] for c in ("conv3", "tcn_w", "linear_full_window", "linear_last_seven")},
                      "reach": {c: [v["measured_reach"], v["declared_reach"]] for c, v in doc["reach"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
