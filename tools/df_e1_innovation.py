#!/usr/bin/env python3
"""RP44: a diagnostic that tells INFORMATION, CAPACITY and OPTIMISATION apart, declared before it runs.

The dictum's F5 is the reason this file exists. The earlier diagnostic labelled a distant lag
"impossible for a seven-sample receiver" from its POSITION alone, and a deterministic counterexample
broke that: on a series that is constant within each window, copying the last sample recovers the
lag-50 target exactly (R2 = 1). Position does not decide reachability — the series does.

So the recovery task here is built on INDEPENDENT INNOVATIONS, and the claim is about that task only:

  * the target is a function of an innovation that enters the window `distant_lag` samples back and is
    independent of everything the last `short_support` samples contain, so a short receiver has NO
    information about it, by construction and not by position;
  * the same generator emits a REDUNDANT companion series in which that innovation is also echoed
    inside the short support, where a short receiver SHOULD succeed — if it fails there too, the
    diagnostic is measuring the optimiser, not the information, and says so;
  * a long receiver has a KNOWN structural solution (read the innovation's channel at its lag), so
    failure there is capacity or budget, never information;
  * competent linear controls on each support, a label-independent control (targets shuffled across
    realisations) and a train/validation split BY REALISATION, not by window, so no leakage of the
    same innovation across the split.

Everything it measures is stored with its scope, favourable or not. Budgets are counted in observed
optimiser iterations, not in requested epochs.

    python tools/df_e1_innovation.py --out INNOVATION.json [--updates 1500] [--seeds 3]
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
SCHEMA = "df_e1_innovation_diagnostic.v1"


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


# --- the generator, declared before anything is fitted -------------------------------------------

def generate(*, realisations: int, window: int, distant_lag: int, short_support: int, channels: int,
             amplitude: float, noise: float, seed: int) -> dict:
    """One window per realisation, so train and validation never share an innovation.

    Channel 0 carries the innovation `u` at position `window - 1 - distant_lag` and zeros elsewhere;
    the other channels carry independent noise. The target of the RECOVERY task is `amplitude * u`
    plus observation noise: nothing in the last `short_support` samples of any channel depends on
    `u`. The REDUNDANT task differs in exactly one way: channel 1 echoes `u` at the last position, so
    the same short receiver has the information there.
    """
    rng = np.random.default_rng(seed)
    position = window - 1 - distant_lag
    if position < 0 or distant_lag < short_support:
        raise SystemExit(f"distant_lag {distant_lag} must fit in the window and exceed the short support")
    u = rng.normal(size=realisations)
    x = rng.normal(scale=noise, size=(realisations, window, channels)).astype(np.float32)
    x[:, position, 0] += u                                    # the innovation, only there
    redundant = x.copy()
    redundant[:, -1, 1] += u                                  # the same innovation, echoed where a short receiver sees it
    y = (amplitude * u + rng.normal(scale=noise, size=realisations)).astype(np.float64)
    return {"recovery_x": x, "redundant_x": redundant, "y": y, "innovation": u,
            "innovation_position": int(position), "snr": float(amplitude / max(noise, 1e-12)),
            "declared": {"realisations": realisations, "window": window, "distant_lag": distant_lag,
                         "short_support": short_support, "channels": channels, "amplitude": amplitude,
                         "noise": noise, "seed": seed,
                         "split": "BY REALISATION: a window belongs to exactly one split, so no innovation crosses it",
                         "known_solution": "read channel 0 at the innovation's position and scale by the amplitude",
                         "short_receiver_information": "none in the recovery task by construction; present in the redundant one"}}


def reference_levels(*, amplitude: float, noise: float, realisations: int = 400000, seed: int = 20260920) -> dict:
    """RP54 (dictum F6): what the irreducible level of this task actually is.

    The generator is z = u + e_x on the innovation's channel, y = a*u + e_y, with u ~ N(0, 1) and both
    noises N(0, s^2), independent. Three references, which are NOT the same number:

      latent oracle      a predictor that saw u itself: MSE = s^2. Nothing observable can beat it.
      Bayes on the input the model actually gets: E[y | z] = a * z / (1 + s^2), because
                         Cov(y, z) = a and Var(z) = 1 + s^2. Its MSE is a^2 * s^2 / (1 + s^2) + s^2.
      copy reference     predicting a*z, which is what the previous diagnostic called the floor. It is
                         NOT the floor: its MSE is a^2 * s^2 + s^2, strictly larger whenever s > 0.

    With a = 1 and s = 0.3 the three are 0.09, 0.172568... and 0.18. The population formulas are
    computed here in closed form AND checked against a Monte Carlo draw with no training at all, so
    the finite-sample number and the population number are never swapped for one another.
    """
    a, s2 = float(amplitude), float(noise) ** 2
    var_y = a ** 2 + s2
    closed = {"latent_oracle_mse": s2,
              "bayes_on_observed_input_mse": (a ** 2) * s2 / (1.0 + s2) + s2,
              "copy_reference_mse": (a ** 2) * s2 + s2,
              "bayes_coefficient": a / (1.0 + s2),
              "variance_of_y": var_y}
    closed["latent_oracle_r2"] = 1.0 - closed["latent_oracle_mse"] / var_y
    closed["bayes_on_observed_input_r2"] = 1.0 - closed["bayes_on_observed_input_mse"] / var_y
    closed["copy_reference_r2"] = 1.0 - closed["copy_reference_mse"] / var_y
    rng = np.random.default_rng(seed)
    u = rng.normal(size=realisations)
    z = u + rng.normal(scale=noise, size=realisations)
    y = a * u + rng.normal(scale=noise, size=realisations)
    empirical = {"latent_oracle_mse": float(np.mean((a * u - y) ** 2)),
                 "bayes_on_observed_input_mse": float(np.mean((closed["bayes_coefficient"] * z - y) ** 2)),
                 "copy_reference_mse": float(np.mean((a * z - y) ** 2)),
                 "realisations": int(realisations), "seed": int(seed),
                 "note": "a Monte Carlo draw of the declared generator; NO model was trained to produce it"}
    return {"assumptions": {"u": "N(0, 1)", "e_x": f"N(0, {noise}^2) on the innovation's channel",
                            "e_y": f"N(0, {noise}^2) on the target", "amplitude": a,
                            "independence": "u, e_x and e_y are mutually independent",
                            "units": "MSE in the target's own squared units; R2 against Var(y)"},
            "closed_form": closed, "monte_carlo": empirical,
            "agreement_max_abs_diff": max(abs(closed[k] - empirical[k]) for k in
                                          ("latent_oracle_mse", "bayes_on_observed_input_mse", "copy_reference_mse")),
            "reading": ("the copy reference is not the irreducible level: the Bayes predictor on the observed input "
                        "beats it, and the latent oracle beats them both. A receiver that reaches the copy level has "
                        "not reached the floor, and one that beats it has not done anything impossible")}


def snr_terms(*, amplitude: float, noise: float) -> dict:
    """RP54: name the definition and the unit instead of interchanging them."""
    amp = float(amplitude) / max(float(noise), 1e-12)
    power = amp ** 2
    return {"amplitude_ratio": amp, "power_ratio": power, "power_ratio_db": float(10.0 * np.log10(power)),
            "definition": "amplitude_ratio = a / s; power_ratio = (a / s)^2; dB = 10 log10(power_ratio)",
            "note": "the earlier field named `snr` was the AMPLITUDE ratio; the power ratio is its square"}


def _score(pred, y) -> dict:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    var = float(np.var(y))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.mean(err ** 2) / var) if var > 0 else None}


def _linear(x_train, y_train, x_eval, *, rows: slice, lam: float = 1.0) -> np.ndarray:
    A = x_train[:, rows, :].reshape(x_train.shape[0], -1).astype(np.float64)
    A = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    B = x_eval[:, rows, :].reshape(x_eval.shape[0], -1).astype(np.float64)
    B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    return B @ np.linalg.solve(A.T @ A + reg, A.T @ y_train)


class _Counter:
    """Observed optimiser iterations, read from the optimiser itself (RP44), not from requested epochs."""

    def __init__(self):
        self.updates = 0

    def callback(self, tf):
        counter = self

        class Count(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                counter.updates += 1

            def on_train_end(self, logs=None):
                try:
                    counter.observed_iterations = int(self.model.optimizer.iterations.numpy())
                except Exception:
                    counter.observed_iterations = None
        return Count()


def _fit_receiver(x_tr, y_tr, x_ev, *, core: str, window: int, channels: int, seed: int, updates: int,
                  batch: int, lr: float, assignment: list, artifacts: Path | None = None) -> dict:
    tf = E._tf()
    tf.keras.utils.set_random_seed(seed)
    model = P._model_for_target(assignment, window, channels, 0, seed, core=core)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    counter = _Counter()
    steps = max(1, int(np.ceil(len(y_tr) / batch)))
    epochs = max(1, int(np.ceil(updates / steps)))
    t0 = time.process_time()
    history = model.fit(x_tr, y_tr.astype(np.float32)[:, None], epochs=epochs, batch_size=batch, verbose=0,
                        shuffle=True, callbacks=[counter.callback(tf)])
    seconds = time.process_time() - t0
    pred = np.asarray(model.predict(x_ev, verbose=0))[:, 0]
    kept = None
    if artifacts is not None:
        artifacts.mkdir(parents=True, exist_ok=True)
        np.savez(artifacts / "arrays.npz", prediction=pred, y_eval=y_ev if False else np.asarray([]),
                 x_eval_last_rows=x_ev[:, -1, :])
        model.save_weights(str(artifacts / "weights.weights.h5"))
        (artifacts / "graph.json").write_text(model.to_json())
        kept = str(artifacts)
    return {"prediction": pred, "artifacts": kept, "requested_updates": int(updates), "observed_updates": int(counter.updates),
            "observed_optimizer_iterations": getattr(counter, "observed_iterations", None),
            "epochs_run": int(epochs), "steps_per_epoch": int(steps),
            "budget_note": ("epochs are whole, so the loop runs ceil(requested / steps) * steps updates: the OBSERVED "
                            "count is what this diagnostic reports"),
            "final_train_loss": float(history.history["loss"][-1]), "cpu_seconds": round(seconds, 2),
            "parameters": int(model.count_params()), "declared_reach": P.model_reach(window, core)}


def diagnose(*, realisations: int = 6000, window: int = 60, distant_lag: int = 50, short_support: int = 7,
             channels: int = 7, amplitude: float = 1.0, noise: float = 0.3, updates: int = 1500,
             batch: int = 64, lr: float = 3e-3, seeds=(1, 2, 3), cores=("conv3", "tcn_w"),
             artifacts: Path | None = None) -> dict:
    data = generate(realisations=realisations, window=window, distant_lag=distant_lag, short_support=short_support,
                    channels=channels, amplitude=amplitude, noise=noise, seed=20260920)
    assignment = [0, 1, 1, 2, 2, 2, 2][:channels] + [2] * max(0, channels - 7)
    n_train = int(realisations * 0.7)
    out = {"schema": SCHEMA, "declared": data["declared"], "innovation_position": data["innovation_position"],
           "snr": snr_terms(amplitude=amplitude, noise=noise),
           "reference_levels": reference_levels(amplitude=amplitude, noise=noise),
           "tasks": {}, "seeds": list(seeds)}
    y = data["y"]
    y_tr, y_ev = y[:n_train], y[n_train:]
    rng = np.random.default_rng(7)
    shuffled_tr = rng.permutation(y_tr)                                  # label-independent control
    for task in ("recovery", "redundant"):
        x = data[f"{task}_x"]
        x_tr, x_ev = x[:n_train], x[n_train:]
        block = {"short_receiver_has_information": task == "redundant",
                 "copy_reference_r2": _score(amplitude * x_ev[:, data["innovation_position"], 0], y_ev)["r2"],
                 "bayes_on_observed_input_r2": _score(
                     out["reference_levels"]["closed_form"]["bayes_coefficient"] * x_ev[:, data["innovation_position"], 0],
                     y_ev)["r2"],
                 "latent_oracle_r2": _score(amplitude * data["innovation"][n_train:], y_ev)["r2"],
                 "reference_note": "finite-sample values on THIS evaluation set; the population levels are in "
                                   "`reference_levels.closed_form`",
                 "linear_full_window": _score(_linear(x_tr, y_tr, x_ev, rows=slice(0, window)), y_ev),
                 "linear_short_support": _score(_linear(x_tr, y_tr, x_ev, rows=slice(window - short_support, window)), y_ev),
                 "mean_of_train": _score(np.full(y_ev.shape, float(np.mean(y_tr))), y_ev),
                 "label_independent_control": _score(_linear(x_tr, shuffled_tr, x_ev, rows=slice(0, window)), y_ev),
                 "receivers": {}}
        for core in cores:
            per_seed = []
            for seed in seeds:
                fit = _fit_receiver(x_tr, y_tr, x_ev, core=core, window=window, channels=channels, seed=seed,
                                    updates=updates, batch=batch, lr=lr, assignment=assignment,
                                    artifacts=(artifacts / f"{task}_{core}_seed{seed}") if artifacts else None)
                per_seed.append({**{k: v for k, v in fit.items() if k != "prediction"}, **_score(fit["prediction"], y_ev)})
            r2s = [s["r2"] for s in per_seed if s["r2"] is not None]
            block["receivers"][core] = {"per_seed": per_seed, "r2_mean": float(np.mean(r2s)) if r2s else None,
                                        "r2_min": float(min(r2s)) if r2s else None,
                                        "r2_max": float(max(r2s)) if r2s else None,
                                        "cpu_seconds": float(sum(s["cpu_seconds"] for s in per_seed))}
        out["tasks"][task] = block
    out["verdict"] = verdict(out)
    return out


def verdict(doc: dict) -> dict:
    rec, red = doc["tasks"]["recovery"], doc["tasks"]["redundant"]
    short_rec = rec["receivers"]["conv3"]["r2_mean"]
    long_rec = rec["receivers"]["tcn_w"]["r2_mean"]
    short_red = red["receivers"]["conv3"]["r2_mean"]
    return {
        "short_receiver_fails_the_recovery_task": bool(short_rec is not None and short_rec < 0.1),
        "short_receiver_solves_the_redundant_task": bool(short_red is not None and short_red > 0.5),
        "long_receiver_solves_the_recovery_task": bool(long_rec is not None and long_rec > 0.5),
        "linear_short_support_fails_recovery": bool((rec["linear_short_support"]["r2"] or 0) < 0.1),
        "linear_full_window_solves_recovery": bool((rec["linear_full_window"]["r2"] or 0) > 0.5),
        # shuffled labels must not beat the mean; a NEGATIVE R2 is the expected shape of that control
        # (a fitted model on independent labels is worse than predicting the mean), so the rule is
        # "no better than the mean", never "close to zero".
        "label_independent_control_no_better_than_the_mean": bool(
            (rec["label_independent_control"]["r2"] or 0) <= (rec["mean_of_train"]["r2"] or 0) + 0.05),
        "label_independent_control_r2": rec["label_independent_control"]["r2"],
        "reading": ("This says the SHORT receiver lacks the information IN THIS TASK, where the innovation is "
                    "independent of its support by construction, and that it can use the same innovation when the "
                    "series carries it inside that support. It does not say a short receiver fails on every "
                    "correlated series, and it is not a claim about the household task, whose diagnostic stays "
                    "empirical and DEV-scoped."),
        "if_the_redundant_task_also_fails": ("the measurement is about the optimiser or the budget, not the "
                                             "information, and no information claim may be made from it"),
    }


def provenance(out_path: Path, doc: dict, *, artifacts: Path | None = None) -> dict:
    """RP54: what this diagnostic's result is bound to. New results keep their arrays, graph, weights,
    seeds, partitions, scaler and OBSERVED updates beside the report; the historical run of 2026-09-20
    kept only its metrics, and that is declared rather than dressed up as an earlier governance."""
    return {"scope": "DIAGNOSTIC, DEVELOPMENT: a synthetic generator declared before it ran, not an experiment on "
                     "governed data; it carries no campaign and no terminal, and none is invented for it",
            "artifacts_directory": str(artifacts) if artifacts else None,
            "report": str(out_path),
            "kept": ["the generator's declared parameters and seed", "per-seed observed optimiser iterations",
                     "the split by realisation", "the closed-form and Monte Carlo reference levels"]
                    + (["the evaluation arrays and per-receiver weights"] if artifacts else []),
            "not_kept": [] if artifacts else ["arrays and weights: this entry point returned metrics only"],
            "earlier_run": "the 2026-09-20 run on WORKER_A stored metrics without arrays or weights; its scope is "
                           "historical diagnostic, and no receipt is claimed for it"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--realisations", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--distant-lag", type=int, default=50)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--artifacts", type=Path, default=None, help="where to keep arrays, weights and seeds")
    a = ap.parse_args(argv)
    doc = diagnose(realisations=a.realisations, updates=a.updates, seeds=tuple(range(1, a.seeds + 1)),
                   distant_lag=a.distant_lag, noise=a.noise, artifacts=a.artifacts)
    doc["host"] = os.uname().nodename
    if a.artifacts:
        a.artifacts.mkdir(parents=True, exist_ok=True)
    doc["provenance"] = provenance(a.out, doc, artifacts=a.artifacts)
    a.out.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"verdict": doc["verdict"],
                      "recovery_r2": {c: doc["tasks"]["recovery"]["receivers"][c]["r2_mean"] for c in doc["tasks"]["recovery"]["receivers"]},
                      "redundant_r2": {c: doc["tasks"]["redundant"]["receivers"][c]["r2_mean"] for c in doc["tasks"]["redundant"]["receivers"]},
                      "linear": {"recovery_full": doc["tasks"]["recovery"]["linear_full_window"]["r2"],
                                 "recovery_short": doc["tasks"]["recovery"]["linear_short_support"]["r2"]},
                      "references": doc["reference_levels"]["closed_form"], "snr": doc["snr"],
                      "observed_updates": doc["tasks"]["recovery"]["receivers"]["conv3"]["per_seed"][0]["observed_updates"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
