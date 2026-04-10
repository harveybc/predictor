"""Binary classification fitness computation for optimizer and DON evaluator.

Composite Binary Fitness (CBF):
    composite = 0.50*Accuracy + 0.20*AUC + 0.15*F1 + 0.15*(1-Brier)
    base = 0.4 * train_composite + 0.6 * val_composite
    fitness = -base + penalty

    Accuracy is the primary optimisation target.
    AUC and F1 provide secondary signal.
    Brier score penalises miscalibrated probabilities.

Lower fitness is better (more negative = better model).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    brier_score_loss,
)


def _safe_auc(y_true, y_prob):
    """Compute AUC-ROC, returning 0.5 on constant class or failure."""
    try:
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def _safe_f1(y_true, y_prob, threshold=0.5):
    """Compute F1, returning 0.0 on failure."""
    try:
        y_true = np.asarray(y_true).flatten().astype(int)
        y_hat = (np.asarray(y_prob).flatten() >= threshold).astype(int)
        return float(f1_score(y_true, y_hat, zero_division=0))
    except Exception:
        return 0.0


def compute_binary_metrics_for_split(y_true, y_prob, threshold=0.5):
    """Compute full binary classification metrics for a single split.

    Parameters
    ----------
    y_true : array-like, shape (N,)  — ground-truth binary labels (0/1)
    y_prob : array-like, shape (N,)  — predicted probabilities

    Returns
    -------
    dict with keys: auc_roc, f1, accuracy, precision, recall, mcc, brier, pos_rate_true, pos_rate_pred
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float32).flatten()
    n = min(len(y_true), len(y_prob))
    y_true = y_true[:n]
    y_prob = y_prob[:n]
    y_int = y_true.astype(int)
    y_hat = (y_prob >= threshold).astype(int)

    auc = _safe_auc(y_int, y_prob)
    f1 = _safe_f1(y_int, y_prob, threshold)

    try:
        acc = float(accuracy_score(y_int, y_hat))
    except Exception:
        acc = 0.0
    try:
        prec = float(precision_score(y_int, y_hat, zero_division=0))
    except Exception:
        prec = 0.0
    try:
        rec = float(recall_score(y_int, y_hat, zero_division=0))
    except Exception:
        rec = 0.0
    try:
        mcc = float(matthews_corrcoef(y_int, y_hat))
    except Exception:
        mcc = 0.0
    try:
        brier = float(brier_score_loss(y_int, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    except Exception:
        brier = 1.0

    return {
        "auc_roc": auc,
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "mcc": mcc,
        "brier": brier,
        "pos_rate_true": float(np.mean(y_int)),
        "pos_rate_pred": float(np.mean(y_hat)),
    }


def _composite_score(metrics):
    """Compute composite binary metric from a metrics dict. Range ~[0, 1]."""
    acc = metrics.get("accuracy", 0.0)
    auc = metrics.get("auc_roc", 0.5)
    f1 = metrics.get("f1", 0.0)
    brier = metrics.get("brier", 1.0)
    return 0.50 * acc + 0.20 * auc + 0.15 * f1 + 0.15 * (1.0 - brier)


def compute_binary_fitness(train_metrics, val_metrics):
    """Composite binary fitness (lower is better).

    Uses Accuracy + AUC + F1 + Brier with accuracy heavily weighted.

    Parameters
    ----------
    train_metrics : dict from compute_binary_metrics_for_split
    val_metrics   : dict from compute_binary_metrics_for_split

    Returns
    -------
    float : fitness value (lower is better, negative = good model)
    """
    train_comp = _composite_score(train_metrics)
    val_comp = _composite_score(val_metrics)

    if not np.isfinite(train_comp) or not np.isfinite(val_comp):
        return float("inf")

    # Base: weighted combination (higher composite = lower fitness via negation)
    fitness = -(0.4 * train_comp + 0.6 * val_comp)

    # Penalty 1: overfitting (train composite >> val composite)
    overfit = train_comp - val_comp
    if overfit > 0.03:
        fitness += overfit * 2.0

    # Penalty 2: worse than random on validation
    if val_metrics.get("auc_roc", 0.5) < 0.5:
        fitness += (0.5 - val_metrics["auc_roc"]) * 2.0

    return fitness


def compute_binary_val_only_fitness(val_metrics):
    """Val-only composite fitness for DON evaluator (no training data available).

    Returns -composite when available, else 0.
    """
    comp = _composite_score(val_metrics)
    if np.isfinite(comp):
        return -comp
    return 0.0
