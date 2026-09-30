"""
Metrics computation and aggregation for binary classification pipelines.

Provides:
- compute_binary_metrics  (train / val / test — single split)
- aggregate_and_save_binary_results
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

# Canonical metric list (order used everywhere)
BINARY_METRIC_NAMES = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC_ROC",
    "AUC_PR",
    "MCC",
    "Brier",
    "LogLoss",
    "Pos_Rate_True",
    "Pos_Rate_Pred",
    "Uncertainty",
]


def _safe_metric(fn, *args, default=np.nan, **kwargs):
    """Call *fn* and return default on any failure (constant-class, empty, etc.)."""
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_unc: Optional[np.ndarray],
    split_name: str,
    horizon: int,
    metrics_results: Dict,
) -> None:
    """Compute all binary metrics for one split / horizon and store in *metrics_results*.

    Parameters
    ----------
    y_true : (N,) or (N,1) float32  — ground-truth 0/1 labels
    y_prob : (N,) or (N,1) float32  — predicted probabilities
    y_unc  : (N,) or (N,1) float32 or None — MC uncertainty (std)
    split_name : "Train" | "Validation" | "Test"
    horizon : int — always 1 for binary predictors
    metrics_results : nested dict[split][metric][horizon] -> list of floats
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float32).flatten()

    n = min(len(y_true), len(y_prob))
    y_true = y_true[:n]
    y_prob = y_prob[:n]

    y_hat = (y_prob >= 0.5).astype(int)
    y_int = y_true.astype(int)

    acc   = _safe_metric(accuracy_score, y_int, y_hat)
    prec  = _safe_metric(precision_score, y_int, y_hat, zero_division=0)
    rec   = _safe_metric(recall_score, y_int, y_hat, zero_division=0)
    f1    = _safe_metric(f1_score, y_int, y_hat, zero_division=0)
    auc   = _safe_metric(roc_auc_score, y_int, y_prob)
    ap    = _safe_metric(average_precision_score, y_int, y_prob)
    mcc   = _safe_metric(matthews_corrcoef, y_int, y_hat)
    brier = _safe_metric(brier_score_loss, y_int, y_prob)
    ll    = _safe_metric(log_loss, y_int, np.clip(y_prob, 1e-7, 1 - 1e-7))
    pos_true = float(np.mean(y_int))
    pos_pred = float(np.mean(y_hat))

    unc_mean = np.nan
    if y_unc is not None:
        y_unc = np.asarray(y_unc, dtype=np.float32).flatten()[:n]
        unc_mean = float(np.mean(np.abs(y_unc)))

    values = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC_ROC": auc,
        "AUC_PR": ap,
        "MCC": mcc,
        "Brier": brier,
        "LogLoss": ll,
        "Pos_Rate_True": pos_true,
        "Pos_Rate_Pred": pos_pred,
        "Uncertainty": unc_mean,
    }

    for metric, val in values.items():
        metrics_results[split_name][metric][horizon].append(val)

    # Pretty-print one-liner
    print(
        f"  {split_name} H{horizon} | "
        f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} "
        f"AUC={auc:.4f} AP={ap:.4f} MCC={mcc:.4f} Brier={brier:.4f} "
        f"Unc={unc_mean:.6f}"
    )


def aggregate_and_save_binary_results(
    metrics_results: Dict,
    predicted_horizons: List[int],
    results_file: str,
) -> None:
    """Aggregate across iterations and save to CSV (same format as regression pipeline)."""
    print("\n--- Aggregating Binary Classification Results ---")
    data_sets = ["Train", "Validation", "Test"]

    results_list = []
    for ds in data_sets:
        for mn in BINARY_METRIC_NAMES:
            for h in predicted_horizons:
                values = metrics_results[ds][mn].get(h, [])
                valid = [v for v in values if not np.isnan(v)]
                if valid:
                    results_list.append({
                        "Metric": f"{ds} {mn} H{h}",
                        "Average": np.mean(valid),
                        "Std Dev": np.std(valid),
                        "Min": np.min(valid),
                        "Max": np.max(valid),
                    })
                else:
                    results_list.append({
                        "Metric": f"{ds} {mn} H{h}",
                        "Average": np.nan,
                        "Std Dev": np.nan,
                        "Min": np.nan,
                        "Max": np.nan,
                    })

    results_df = pd.DataFrame(results_list)
    try:
        results_df.to_csv(results_file, index=False, float_format="%.6f")
        print(f"Results saved: {results_file}")
        print(results_df.to_string())
    except Exception as e:
        print(f"ERROR saving results: {e}")


# --------------------------------------------------------------------------- #
# classification_metrics.v1 receipts, in this existing evaluation path
# --------------------------------------------------------------------------- #
# The results CSV above answers "what were the numbers". It does not answer
# "of what, on which rows, against what baseline, and with how much of the
# population abstained" - and those are the questions a later reader needs in
# order not to misread the numbers. The receipt below carries them, through the
# same contract the literature-replication runs use, so this pipeline and a
# BANKING77 evaluation are read the same way.


def _classification_digest(body) -> str:
    import hashlib as _hashlib
    import json as _json
    return _hashlib.sha256(_json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _label_population_digest(role: str, labels) -> str:
    """Bind a receipt to exactly these ordered reference labels."""
    return _classification_digest({"role": role, "n": int(len(labels)),
                                   "labels": [int(v) for v in labels]})


def _scorer_digest() -> str:
    """The bytes of the scorer that produced the number, not its name."""
    import hashlib as _hashlib
    from pathlib import Path as _Path
    return _hashlib.sha256(_Path(__file__).read_bytes()).hexdigest()


def _macro_f1_from_confusion(matrix, answered_only: bool) -> float:
    size = len(matrix)
    total = 0.0
    for index in range(size):
        true_positive = matrix[index][index]
        predicted = sum(matrix[row][index] for row in range(size))
        support = sum(matrix[index]) - (matrix[index][-1] if answered_only else 0)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        total += 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return total / size if size else 0.0


def build_binary_classification_receipt(
    y_true,
    y_prob,
    y_train_true,
    *,
    config: Dict,
    evaluation_split: str = "test",
    horizon: int = 1,
) -> Dict:
    """One `classification_metrics.v1` receipt for a binary predictor's evaluation.

    Uses the same arrays the metrics above are computed from, so the receipt and
    the CSV describe one evaluation rather than two. Abstention is real when the
    config declares `abstain_band`: probabilities within that band of the
    threshold are refusals, and they are counted in the ABSTAINED column instead
    of being dropped.
    """
    import sys as _sys
    from pathlib import Path as _Path
    root = str(_Path(__file__).resolve().parents[1])
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from app import classification_receipt as receipts

    threshold = float(config.get("classification_threshold", 0.5))
    band = float(config.get("abstain_band", 0.0))
    vocabulary = list(config.get("class_vocabulary") or ["down", "up"])
    if len(vocabulary) != 2:
        raise ValueError("a binary receipt needs exactly two class labels")

    truth = np.asarray(y_true, dtype=np.float32).flatten()
    probability = np.asarray(y_prob, dtype=np.float32).flatten()
    rows = min(len(truth), len(probability))
    truth, probability = truth[:rows].astype(int), probability[:rows]
    train_truth = np.asarray(y_train_true, dtype=np.float32).flatten().astype(int)

    # rows = reference class, columns = predicted class, last column = ABSTAINED
    matrix = [[0, 0, 0], [0, 0, 0]]
    for reference, score in zip(truth, probability):
        if band > 0.0 and abs(float(score) - threshold) < band:
            matrix[int(reference)][2] += 1
        else:
            matrix[int(reference)][1 if float(score) >= threshold else 0] += 1

    total = int(len(truth))
    abstained = sum(row[2] for row in matrix)
    answered_only = band > 0.0
    policy = "ANSWERED_ONLY" if answered_only else "FULL_POPULATION_ABSTENTION_WRONG"

    macro_f1 = _macro_f1_from_confusion(matrix, answered_only)
    correct = sum(matrix[index][index] for index in range(2))
    denominator = (total - abstained) if answered_only else total
    accuracy = correct / denominator if denominator else 0.0

    # the paired naive: the majority class of the TRAIN labels, on these rows
    majority = int(np.argmax(np.bincount(train_truth, minlength=2)))
    naive_matrix = [[0, 0, 0], [0, 0, 0]]
    for reference in truth:
        naive_matrix[int(reference)][majority] += 1
    naive_macro_f1 = _macro_f1_from_confusion(naive_matrix, False)

    # where these answers came from. This model is fitted and scored in THIS
    # process, so the path is observed rather than read off a configuration; and
    # the checkpoint file is not digested here, because the array the metrics
    # were computed from came from the in-process model and not from that file.
    # Before this round the field `checkpoint_sha256` carried a digest of four
    # hyperparameters - a 64-hex value shaped exactly like a checkpoint digest,
    # for a checkpoint nothing had read.
    answered = {"path_id": f"pipeline_plugins.binary.{config.get('predictor_plugin', 'unknown')}",
                "kind": "MODEL_IN_PROCESS_NOT_CHECKPOINTED",
                "weights_present": True,
                "served_checkpoint": str(config.get("save_model") or "in-process"),
                "served_checkpoint_sha256": receipts.provenance.NOT_DIGESTED,
                "attestation": "OBSERVED_FROM_ANSWERING_PATH"}

    document = {
        "task_id": str(config.get("olap_experiment_key")
                       or f"{config.get('predictor_plugin', 'unknown')}."
                          f"{config.get('signal_type', 'binary')}.h{horizon}"),
        "corpus_class": "PROGRAMME_INTERNAL_DATASET",
        "corpus_id": str(config.get("x_test_file") or config.get("x_train_file") or "unknown"),
        "corpus_sha256": _classification_digest({
            "x_train": str(config.get("x_train_file")),
            "x_test": str(config.get("x_test_file")),
            "y_train": str(config.get("y_train_file")),
            "y_test": str(config.get("y_test_file"))}),
        "evidence_class": "MEASUREMENT",
        "supervision_regime": "FULL_FINETUNE",
        "labelled_rows_fit_head": True,
        "provider": str(config.get("predictor_plugin", "unknown")),
        "checkpoint": answered["served_checkpoint"],
        "checkpoint_sha256": answered["served_checkpoint_sha256"],
        "answering_path": answered,
        "author_primary_metric": {"family": "MACRO_F1", "name": "macro-F1",
                                  "value": macro_f1, "denominator_policy": policy},
        "paired_naive": {"family": "MACRO_F1", "policy": "MAJORITY_CLASS_FROM_TRAIN",
                         "value": naive_macro_f1, "seed": None,
                         "train_label_population_sha256":
                             _label_population_digest("train", train_truth),
                         "evaluation_population_sha256":
                             _label_population_digest(evaluation_split, truth)},
        "class_vocabulary": vocabulary,
        "per_class_confusion": matrix,
        "secondary_metrics": {"ACCURACY": accuracy},
        "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED",
        "calibrated": False,
        "calibration_split": {"split_id": "NONE", "population_sha256": None, "rows": 0,
                              "fitted_parameters": None},
        "abstention": {
            "abstained": abstained, "answered": total - abstained,
            "coverage": abstained / total if total else 0.0,
            "denominator_policy": policy,
            "abstention_rule": (f"|p - {threshold}| < {band} is a refusal"
                                if band > 0.0
                                else "no abstention rule is configured; every row is answered "
                                     "and no row is removed from the denominator")},
        "population": {"total": total, "answered": total - abstained, "abstained": abstained,
                       "independent_units": total, "repeats": 1, "clustered_by": "NONE"},
        "evaluation_split": evaluation_split,
        "evaluation_population_sha256": _label_population_digest(evaluation_split, truth),
        "protocol_sha256": _classification_digest({
            "threshold": threshold, "abstain_band": band, "window_size": config.get("window_size"),
            "predicted_horizons": config.get("predicted_horizons"),
            "max_steps_train": config.get("max_steps_train"),
            "max_steps_test": config.get("max_steps_test")}),
        "scorer_sha256": _scorer_digest(),
        "seed": str(config.get("seed", "unset")),
        "declared_fields": ["checkpoint"],
        "limitations": ("a single-seed run of this repository's own binary pipeline on its own "
                        "market data; it is not a published-benchmark result, not a business "
                        "corpus result, and grants no deployment"),
    }
    return receipts.build_receipt(document)


def save_binary_classification_receipt(receipt: Dict, path: str) -> None:
    """Write the receipt beside the results CSV, and print primary against naive."""
    import json as _json
    with open(path, "w", encoding="utf-8") as handle:
        _json.dump(receipt, handle, indent=2, sort_keys=True)
    primary = receipt["author_primary_metric"]
    naive = receipt["paired_naive"]
    print(f"Classification receipt: {path}")
    print(f"  {primary['name']} ({primary['family']}, unit {primary['unit']}) = "
          f"{primary['value']:.6f}")
    print(f"  paired naive ({naive['policy']}, same rows) = {naive['value']:.6f}")
    print(f"  abstention coverage = {receipt['abstention']['coverage']:.6f} "
          f"under {receipt['abstention']['denominator_policy']}")
    print(f"  metric identity = {receipt['metric_identity_sha256'][:16]}…  "
          f"receipt = {receipt['receipt_sha256'][:16]}…")
