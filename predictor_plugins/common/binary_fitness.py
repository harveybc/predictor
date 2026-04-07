"""Binary classification fitness computation for optimizer and DON evaluator.

Penalized Asymmetric AUC (PAA):
    base = 0.4 * train_auc + 0.6 * val_auc        (higher is better)
    fitness = -base + penalty                        (lower is better)

    penalty += (train_auc - val_auc) * 2   if overfitting (train >> val)
    penalty += (0.5 - val_auc) * 2         if val_auc < 0.5 (worse than random)

Lower fitness is better (more negative = beating random baseline).
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


def compute_binary_fitness(train_metrics, val_metrics):
    """Full penalized asymmetric AUC fitness (lower is better).

    Used by the optimizer's candidate_worker during training.

    Parameters
    ----------
    train_metrics : dict with at least 'auc_roc' key
    val_metrics   : dict with at least 'auc_roc' key

    Returns
    -------
    float : fitness value (lower is better, negative = beating random)
    """
    train_auc = train_metrics.get("auc_roc", 0.5)
    val_auc = val_metrics.get("auc_roc", 0.5)

    if not np.isfinite(train_auc) or not np.isfinite(val_auc):
        return float("inf")

    # Base: weighted combination (higher AUC = lower fitness via negation)
    base = 0.4 * train_auc + 0.6 * val_auc
    fitness = -base

    # Penalty 1: overfitting (train >> val)
    overfit = train_auc - val_auc
    if overfit > 0.02:  # Allow small gap; penalize large overfitting
        fitness += overfit * 2.0

    # Penalty 2: worse than random on validation
    if val_auc < 0.5:
        fitness += (0.5 - val_auc) * 2.0

    return fitness


def compute_binary_val_only_fitness(val_metrics):
    """Val-only fitness for DON evaluator (no training data available).

    Returns -val_auc when available, else 0.
    """
    val_auc = val_metrics.get("auc_roc", 0.5)
    if np.isfinite(val_auc):
        return -val_auc
    return 0.0
