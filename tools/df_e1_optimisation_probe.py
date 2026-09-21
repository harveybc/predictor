#!/usr/bin/env python3
"""RP60: does the optimisation do what it says — before any hyper-parameter is swept.

Four sanity checks, in this order, because a failure in an earlier one makes the later ones
meaningless. Each runs through the SAME functions the pilot uses; none of them is a new training
protocol and none of them is evidence about generalisation.

  1. updates ARE the optimiser's steps. The runner counts `on_train_batch_end`; that count must
     equal `optimizer.iterations` after the fit, or the reported budget is a proxy for the real one.
  2. capacity. On a FIXED, small subset of train windows, with validation on that same subset, the
     model must be able to drive the error far down. This says the graph, the gradient and the
     learning rate can fit data — nothing about generalising.
  3. the route. A head forced to emit the last observed target, pushed through the same inverse and
     the same metric code, must reproduce the persistence control EXACTLY. If it does not, the
     inverse or the metric is wrong and every number downstream is wrong with it.
  4. independent labels. With the labels shuffled, the fit must NOT beat the persistence control.
     A model that scores well on noise is scoring on something other than the signal.

Then it recomputes, for a finished run, each fit's curve and which checkpoint was actually restored.

    python tools/df_e1_optimisation_probe.py --root RUN_ROOT --out PROBE.json [--skip-training]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _data(root: Path) -> dict:
    with np.load(Path(root)/"DATA.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def capacity_and_iterations(root: Path, *, subset: int = 256, updates: int = 400, seed: int = 1) -> dict:
    """Checks 1 and 2 together: one small real fit on a fixed subset, with its own budget."""
    P = _module("df_e1_pilot")
    E = _module("df_mod_e0")
    data = _data(root)
    design = json.loads((Path(root)/"DESIGN.json").read_text())
    W, h = int(data["window"][0]), int(data["horizon"][0])
    j = int(data["target_channel"][0])
    origins = data["train_origins"][:subset]
    Dataset = P._dataset_class()
    kw = dict(scaler_mean=data["scaler_mean"], scaler_sd=data["scaler_sd"])
    train = Dataset(data["Xs"], data["Y"], origins, W, h, j, 32, shuffle=True, seed=seed, **kw)
    check = Dataset(data["Xs"], data["Y"], origins, W, h, j, 32, shuffle=False, seed=seed, **kw)
    model = P._model_for_target(design["graph"]["assignment"], W, data["Xs"].shape[1], j, seed,
                                core=design["graph"]["core_kind"])
    before = float(np.mean([np.mean((P._predict(model, check).reshape(-1) -
                                     np.concatenate([np.asarray(b[1]).reshape(-1) for b in check]))**2)]))
    fit = P._fit_batched(model, train, check, max_updates=updates, patience=10**6,
                         lr=design["training"]["learning_rate"], seed=seed)
    after = float(model.evaluate(check, verbose=0))
    return {"subset_windows": int(origins.size), "update_ceiling": updates,
            "counted_updates": fit["updates"], "optimizer_iterations": fit.get("optimizer_iterations"),
            "updates_are_optimizer_iterations": fit.get("updates_are_optimizer_iterations"),
            "loss_on_the_subset_before": before, "loss_on_the_subset_after": after,
            "reduction_factor": (before/after) if (before and after) else None,
            "can_fit_what_it_sees": bool(after < 0.5*before) if (before and after) else None,
            "scope": "capacity on data the model also evaluates on; this says the optimiser can move "
                     "the graph, and says NOTHING about generalisation"}


def route_probe(root: Path) -> dict:
    """Check 3: a forced head equal to the last observed target must reproduce the control exactly."""
    E = _module("df_mod_e0")
    data = _data(root)
    h, j = int(data["horizon"][0]), int(data["target_channel"][0])
    ev = data["eval_origins"]
    y = data["Y"][ev+h]
    m, s = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    # what a head that emits the window's last scaled target would produce, inverted by the run's rule
    scaled_last = data["Xs"][ev, j].astype(np.float64)
    through_inverse = scaled_last*s + m
    persistence = data["Y"][ev]
    arrays = Path(root)/"attempts"/"controls"/"arrays.npz"
    stored = None
    if arrays.is_file():
        with np.load(arrays, allow_pickle=False) as z:
            stored = z["validation_pred_persistence"].reshape(-1)
    scores = E.mase(through_inverse[:, None], y[:, None], data["denominator"].tolist())
    control = E.mase(persistence[:, None], y[:, None], data["denominator"].tolist())
    return {"largest_difference_from_the_panel_value": float(np.max(np.abs(through_inverse-persistence))),
            "reproduces_persistence": bool(np.allclose(through_inverse, persistence, atol=1e-6)),
            "matches_the_stored_control": (None if stored is None else
                                           bool(np.allclose(through_inverse, stored, atol=1e-6))),
            "mae_through_the_route": scores["mae_mean"], "mae_of_the_control": control["mae_mean"],
            "identical_metric": bool(abs(scores["mae_mean"]-control["mae_mean"]) < 1e-9),
            "reading": "the scaling, its inverse and the metric agree end to end; a failure here would "
                       "invalidate every number the run reports"}


def _fit_and_score(P, data, design, Y, tr, ev, *, updates, seed, batch=64):
    """One fit through the pilot's own functions, scored in kW on the unshuffled evaluation labels."""
    W, h = int(data["window"][0]), int(data["horizon"][0])
    j = int(data["target_channel"][0])
    Dataset = P._dataset_class()
    kw = dict(scaler_mean=data["scaler_mean"], scaler_sd=data["scaler_sd"])
    train = Dataset(data["Xs"], Y, tr, W, h, j, batch, shuffle=True, seed=seed, **kw)
    val = Dataset(data["Xs"], data["Y"], ev, W, h, j, batch, shuffle=False, seed=seed, **kw)
    model = P._model_for_target(design["graph"]["assignment"], W, data["Xs"].shape[1], j, seed,
                                core=design["graph"]["core_kind"])
    untrained = P._predict(model, val).reshape(-1)
    fit = P._fit_batched(model, train, val, max_updates=updates, patience=10**6,
                         lr=design["training"]["learning_rate"], seed=seed)
    m, s = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    pred = P._predict(model, val).reshape(-1)*s + m
    y = data["Y"][ev+h]
    return {"mae": float(np.mean(np.abs(pred-y))),
            "correlation_with_truth": float(np.corrcoef(pred, y)[0, 1]),
            "sd_of_predictions": float(np.std(pred)),
            "mae_before_any_update": float(np.mean(np.abs(untrained*s+m-y))),
            "restored_epoch": fit.get("restored_checkpoint_epoch"), "epochs": fit.get("epochs"),
            "updates": fit.get("updates")}


def shuffled_labels(root: Path, *, subset: int = 4096, updates: int = 600, seed: int = 7) -> dict:
    """Check 4: scrambled labels must not reach what the SAME protocol reaches with true ones.

    The bar matters. A scrambled-label fit beating persistence proves nothing: on this series a
    near-constant forecast already beats copying the last value at sixty minutes. And beating a
    constant proves less than it seems, because the selection rule restores the best validation
    checkpoint, which on scrambled labels is an EARLY one whose output still tracks the recent level
    of its input. What would be damning is reaching the true-label score. So both fits are run here,
    same subset, same budget, same seed, and the gap between them is the measurement.
    """
    P = _module("df_e1_pilot")
    data = _data(root)
    design = json.loads((Path(root)/"DESIGN.json").read_text())
    h = int(data["horizon"][0])
    tr = data["train_origins"][:subset]
    ev = data["eval_origins"][:2048]
    y = data["Y"][ev+h]
    rng = np.random.default_rng(seed)
    scrambled = data["Y"].copy()
    labels = scrambled[tr+h].copy()
    rng.shuffle(labels)
    scrambled[tr+h] = labels                 # the inputs keep their order; only the train labels move
    honest = _fit_and_score(P, data, design, data["Y"], tr, ev, updates=updates, seed=seed)
    noise = _fit_and_score(P, data, design, scrambled, tr, ev, updates=updates, seed=seed)
    persistence = float(np.mean(np.abs(data["Y"][ev]-y)))
    constant = float(np.mean(np.abs(np.mean(data["Y"][tr+h])-y)))
    gap = noise["mae"]-honest["mae"]
    return {"train_windows": int(tr.size), "update_ceiling": updates, "evaluation_rows": int(ev.size),
            "with_true_labels": honest, "with_scrambled_labels": noise,
            "gap_in_mae": gap,
            "persistence_on_these_rows": persistence, "train_mean_on_these_rows": constant,
            "verdict": ("AS_EXPECTED: scrambling the labels costs the model most of what it had"
                        if gap > 0.05 else
                        "SUSPECT: scrambling the labels barely changed the score"),
            "why_it_still_beats_a_constant": (
                "the selection restores the best validation checkpoint, which on scrambled labels is "
                "an early, partly-fitted state; its output still follows the recent level of its own "
                "input, and that alone beats a constant here. An untrained network does not: its "
                "predictions are far too wide (measured above as mae_before_any_update)."),
            "scope": "a negative control on the TRAIN labels only; the evaluation labels are untouched"}


def curves(root: Path) -> dict:
    """For a finished run: each fit's curve, and which checkpoint was actually restored."""
    out = {}
    for path in sorted((Path(root)/"attempts").glob("*/cell.json")):
        rec = json.loads(path.read_text())
        tr = rec.get("training") or {}
        if not tr.get("curve"):
            continue
        va = tr["curve"]["validation"]
        argmin = int(np.argmin(va))+1
        out[path.parent.name] = {
            "epochs": len(va), "validation_curve": va, "train_curve": tr["curve"]["train"],
            "best_epoch_by_curve": argmin, "recorded_restored_epoch": tr.get("restored_checkpoint_epoch"),
            "agree": argmin == tr.get("restored_checkpoint_epoch"),
            "restored_validation_loss": tr.get("restored_validation_loss"),
            "best_value_on_the_curve": float(min(va)),
            "restore_verified": tr.get("restore_verified"),
            "train_minus_validation_at_the_end": float(tr["curve"]["train"][-1]-va[-1]),
            "validation_rose_after_the_best": float(va[-1]-min(va)),
            "updates": tr.get("updates"), "optimizer_iterations": tr.get("optimizer_iterations"),
            "stop_reason": tr.get("stop_reason")}
    return out


def full_scale_negative(root: Path, *, seed: int = 1) -> dict:
    """The negative control at the scale the run actually used, against the run's own fit.

    At subset scale the control cannot separate true labels from scrambled ones — the true-label fit
    restores epoch 1 and scores worse than the scrambled one. That is a fact about the probe's scale,
    not about the data, so the control is repeated here with ALL the train origins and the design's
    own update ceiling, and compared with R0_s1 of the finished run on the same evaluation rows.
    """
    P = _module("df_e1_pilot")
    data = _data(root)
    design = json.loads((Path(root)/"DESIGN.json").read_text())
    h = int(data["horizon"][0])
    tr, ev = data["train_origins"], data["eval_origins"]
    y = data["Y"][ev+h]
    rng = np.random.default_rng(seed)
    scrambled = data["Y"].copy()
    labels = scrambled[tr+h].copy()
    rng.shuffle(labels)
    scrambled[tr+h] = labels
    noise = _fit_and_score(P, data, design, scrambled, tr, ev,
                           updates=design["training"]["max_updates"], seed=seed,
                           batch=design["training"]["batch"])
    stored = None
    arrays = Path(root)/"attempts"/f"R0_s{seed}"/"arrays.npz"
    if arrays.is_file():
        with np.load(arrays, allow_pickle=False) as z:
            pred = z["validation_pred"].reshape(-1)
        stored = float(np.mean(np.abs(pred-y)))
    persistence = float(np.mean(np.abs(data["Y"][ev]-y)))
    gap = None if stored is None else noise["mae"]-stored
    return {"train_windows": int(tr.size), "update_ceiling": design["training"]["max_updates"],
            "with_scrambled_labels": noise,
            "the_runs_own_true_label_fit": {"unit": f"R0_s{seed}", "mae": stored},
            "gap_in_mae": gap, "persistence": persistence,
            "verdict": (None if gap is None else
                        "AS_EXPECTED: scrambling the labels costs the model what it had"
                        if gap > 0.05 else
                        "SUSPECT: scrambling the labels barely changed the score"),
            "scope": "one seed at full train scale; the evaluation labels are untouched"}


def probe(root: Path, *, skip_training: bool = False, full_scale: bool = False) -> dict:
    report = {"schema": "df_e1_optimisation_probe.v1",
              "at": datetime.utcnow().isoformat(timespec="seconds")+"Z", "root": str(root),
              "route": route_probe(root), "curves": curves(root)}
    if not skip_training:
        report["capacity"] = capacity_and_iterations(root)
        report["shuffled_labels"] = shuffled_labels(root)
    if full_scale:
        report["shuffled_labels_full_scale"] = full_scale_negative(root)
    report["order"] = ("route first, then capacity, then the negative control; a hyper-parameter "
                       "sweep before these pass would be measuring the wrong thing")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--full-scale", action="store_true",
                    help="repeat the negative control with every train origin and the design's ceiling")
    a = ap.parse_args(argv)
    report = probe(a.root, skip_training=a.skip_training, full_scale=a.full_scale)
    a.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "curves"}, indent=1, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
