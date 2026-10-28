"""Decoder-free validation: the Delta_probe battery of PS3-R (subplan section 7, FS17).

For each business target and horizon, with every encoder FROZEN and one probe
of the same form and budget for every arm:

* ``loss_trained`` - probe on the trained encoder's latent ``Z_trained``;
* ``loss_random``  - same probe on the SAME architecture with untrained weights
  (``untrained_twin``), so ``delta_probe = loss_random - loss_trained`` > 0
  means training added usable information about the target;
* ``loss_raw``     - same probe on the raw input window; ``preservation =
  loss_raw - loss_trained`` < 0 means the extractor degraded the input. The raw
  arm has a different dimension; ``raw_dimension_differs`` says so;
* ``naive``        - paired naive on the IDENTICAL evaluation rows: zero log
  return (price persistence) for Y_s / Y_l with squared error, the fit-rows base
  rate for Y_b with log loss.

The probe is a ridge regression (logistic regression for Y_b) on the flattened
per-timestamp latent, standardized on fit rows, with its penalty chosen on the
chronologically last 20 % of the fit rows and refitted on all fit rows; the
same grid and selection serve every arm. Targets are named ``Y_s@<h>h``,
``Y_l@<h>h`` or ``Y_b@<h>h``; anything else is a self-forecast of an input and
is refused (FS02). A pooled latent is refused: the temporal contract requires
(B, T, C) on the input grid. Rows are fixed by the caller (fit before eval,
disjoint); rows with a non-finite label are dropped for every arm alike.
"""
from __future__ import annotations

import re

import numpy as np

TARGETS = ("Y_s", "Y_l", "Y_b")
CARD_ROW_KEYS = ("target", "horizon", "fold", "seed", "loss_trained", "loss_random", "loss_raw", "naive",
                 "delta_probe", "preservation")
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _parse(name):
    m = re.fullmatch(r"(Y_[slb])@(\d+)h", name)
    if not m or m.group(1) not in TARGETS:
        raise ValueError(f"SELF_FORECAST_REFUSED: target {name!r} is not a declared business target "
                         f"Y_s@<h>h / Y_l@<h>h / Y_b@<h>h")
    return m.group(1), int(m.group(2))


def _latent(arm, x, label):
    z = np.asarray(arm(x, training=False), dtype="float64")
    if z.ndim != 3 or z.shape[1] != x.shape[1]:
        raise ValueError(f"POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT: {label} latent {z.shape} is not "
                         f"(batch, {x.shape[1]}, channels)")
    return z.reshape(len(z), -1)


def _standardize(a, fit):
    mu, sd = a[fit].mean(axis=0), a[fit].std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (a - mu) / sd


def _ridge(xf, yf, alpha):
    xm, ym = xf.mean(axis=0), yf.mean()
    xc = xf - xm
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (yf - ym))
    return lambda z: (z - xm) @ w + ym


def _logistic(xf, yf, c):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(yf)) < 2:
        p = float(np.clip(yf.mean(), 1e-6, 1 - 1e-6))
        return lambda z: np.full(len(z), p)
    model = LogisticRegression(C=c, max_iter=2000).fit(xf, yf)
    return lambda z: model.predict_proba(z)[:, 1]


def _loss(kind, y, pred):
    if kind == "binary":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((y - pred) ** 2))


def _probe_loss(features, y, fit, ev, kind):
    a = _standardize(features, fit)
    cut = int(len(fit) * 0.8)
    inner_fit, inner_val = fit[:cut], fit[cut:]
    grid, make = (LOGISTIC_CS, _logistic) if kind == "binary" else (RIDGE_ALPHAS, _ridge)
    scores = []
    for penalty in grid:
        model = make(a[inner_fit], y[inner_fit], penalty)
        scores.append(_loss(kind, y[inner_val], model(a[inner_val])))
    chosen = grid[int(np.argmin(scores))]
    model = make(a[fit], y[fit], chosen)
    return _loss(kind, y[ev], model(a[ev])), chosen


def probe_battery(*, trained, random, x, targets, fit_rows, eval_rows, fold, seed):
    """Return one card row per target/horizon (``representation_candidate_card.v1`` evaluation.probes)."""
    x = np.asarray(x, dtype="float32")
    fit_rows, eval_rows = np.asarray(fit_rows), np.asarray(eval_rows)
    if (len(np.intersect1d(fit_rows, eval_rows)) or not len(fit_rows) or not len(eval_rows)
            or fit_rows.max() >= eval_rows.min()):
        raise ValueError("fit and evaluation rows must be disjoint, nonempty and chronological (no overlap)")
    parsed = {name: _parse(name) for name in targets}
    feats = {"trained": _latent(trained, x, "trained"), "random": _latent(random, x, "random"),
             "raw": x.reshape(len(x), -1).astype("float64")}
    rows = []
    for name, (target, horizon) in parsed.items():
        y = np.asarray(targets[name], dtype="float64")
        ok = np.isfinite(y)
        fit, ev = fit_rows[ok[fit_rows]], eval_rows[ok[eval_rows]]
        kind = "binary" if target == "Y_b" else "regression"
        losses, penalties = {}, {}
        for arm, f in feats.items():
            losses[arm], penalties[arm] = _probe_loss(f, y, fit, ev, kind)
        if kind == "binary":
            base = float(np.clip(y[fit].mean(), 1e-6, 1 - 1e-6))
            naive = _loss(kind, y[ev], np.full(len(ev), base))
        else:
            naive = _loss(kind, y[ev], np.zeros(len(ev)))           # price persistence: zero log return
        rows.append({"target": target, "horizon": horizon, "fold": fold, "seed": int(seed),
                     "probe": ("logistic" if kind == "binary" else "ridge") + "_on_flattened_latent",
                     "loss_name": "log_loss" if kind == "binary" else "mse",
                     "loss_trained": losses["trained"], "loss_random": losses["random"],
                     "loss_raw": losses["raw"], "naive": naive,
                     "delta_probe": losses["random"] - losses["trained"],
                     "preservation": losses["raw"] - losses["trained"],
                     "raw_dimension_differs": feats["raw"].shape[1] != feats["trained"].shape[1],
                     "fit_rows": int(len(fit)), "eval_rows": int(len(ev)), "penalties": penalties})
    return rows


def untrained_twin(encoder, seed):
    """The same architecture with fresh, seeded initial weights (the Z_random arm)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))
    twin = tf.keras.models.clone_model(encoder)
    twin.build(encoder.input_shape)
    return twin


def latent_diagnostics(encoder, x):
    """Effective dimension (participation ratio over per-timestamp channels) and collapse flag."""
    z = np.asarray(encoder(np.asarray(x, "float32"), training=False), dtype="float64")
    flat = z.reshape(-1, z.shape[-1])
    eig = np.clip(np.linalg.eigvalsh(np.cov(flat, rowvar=False)), 0, None)
    total = float(eig.sum())
    effective = 0.0 if total < 1e-12 else float(total ** 2 / np.sum(eig ** 2))
    return {"effective_dimension": effective, "channels": int(z.shape[-1]),
            "per_channel_std_min": float(flat.std(axis=0).min()), "collapsed": effective < 1.0 + 1e-6}
