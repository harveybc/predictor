"""R3 warm regime: frozen donor components for ``freeze_epochs``, then unfrozen at a declared learning rate.

Phase 1 trains with every R3 component frozen (only R0/R2 components and the head move) for exactly
``freeze_epochs`` epochs (patience = freeze_epochs, so the warm phase is never cut short). Phase 2 makes
the R3 components trainable and continues with a NEW AdamW at ``unfreeze_learning_rate`` for the rest of
``max_epochs``. Optimizer state is RESET at the boundary on purpose: phase-1 moments belong to a
different set of trainable variables, and carrying them would mix learning rates silently. The receipt
says so (``optimizer_reset_at_unfreeze``).

The best checkpoint is chosen ACROSS the boundary: if phase 2 never beats phase 1's best validation loss,
phase 1's selected weights are restored (``selected_phase`` 1). Observed optimizer updates of both phases
are counted. Every R3 component must declare the same freeze_epochs and unfreeze_learning_rate.
"""
from __future__ import annotations

from .common import weights_hash


def _r3_components(bundle):
    c = bundle.config
    out = {spec["name"]: (bundle.branch_models[spec["name"]], spec)
           for spec in c["branches"] if spec["regime"] == "R3"}
    if c["core"]["regime"] == "R3":
        out["core"] = (bundle.core_model, c["core"])
    return out


def fit_warm(bundle, x, y, vx, vy, fit_config):
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    comps = _r3_components(bundle)
    if not comps:
        raise ValueError("fit_warm needs at least one R3 component")
    schedules = {(spec["freeze_epochs"], spec["unfreeze_learning_rate"]) for _, spec in comps.values()}
    if len(schedules) != 1:
        raise ValueError("all R3 components must share freeze_epochs and unfreeze_learning_rate")
    freeze, unfreeze_lr = schedules.pop()
    if "optimizer" in fit_config:
        raise ValueError("fit_warm builds its own optimizers; a custom optimizer is not accepted")
    max_epochs = int(fit_config.get("max_epochs", 20))
    if freeze >= max_epochs:
        raise ValueError("freeze_epochs must be smaller than max_epochs (phase 2 needs at least one epoch)")
    model = bundle.forecast_model
    hashes = lambda: {name: weights_hash(m) for name, (m, _) in comps.items()}
    start = hashes()
    for m, _ in comps.values():
        m.trainable = False
    p1 = fit_with_early_stopping(model, x, y, vx, vy, {**fit_config, "max_epochs": freeze, "patience": freeze})
    after1 = hashes()
    if after1 != start:
        raise ValueError("an R3 component moved while frozen")
    phase1_weights = [w.copy() for w in model.get_weights()]
    for m, _ in comps.values():
        m.trainable = True
    p2 = fit_with_early_stopping(model, x, y, vx, vy,
                                 {**fit_config, "max_epochs": max_epochs - freeze, "learning_rate": unfreeze_lr})
    after2 = hashes()
    selected = 2
    if not p2["best_validation_loss"] < p1["best_validation_loss"]:
        model.set_weights(phase1_weights)
        selected = 1
    strip = lambda r: {k: v for k, v in r.items() if k not in ("history", "settings")}
    return {"schema": "predictor.modular.warm_fit.v1", "freeze_epochs": freeze,
            "unfreeze_learning_rate": unfreeze_lr, "unfreeze_epoch": p1["epochs_completed"] + 1,
            "optimizer_reset_at_unfreeze": True, "r3_components": sorted(comps),
            "component_hashes_start": start,
            "phase_1": {**strip(p1), "component_hashes_after": after1,
                        "learning_rate": fit_config.get("learning_rate")},
            "phase_2": {**strip(p2), "component_hashes_after": after2, "learning_rate": unfreeze_lr},
            "observed_updates": p1["observed_updates"] + p2["observed_updates"],
            "selected_phase": selected,
            "best_validation_loss": min(p1["best_validation_loss"], p2["best_validation_loss"]),
            "history": {"phase_1": p1.get("history"), "phase_2": p2.get("history")}}
