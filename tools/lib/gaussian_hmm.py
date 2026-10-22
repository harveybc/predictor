"""Gaussian hidden Markov model (diagonal covariances) in plain numpy: EM on a fit sequence and a CAUSAL FORWARD
FILTER for deployment.

Written for predictor's layer-2 pool because ``hmmlearn`` is not installed in the sanctioned environment and
because the legacy ``regime_labels`` worker fitted on the full matrix and emitted ``predict_proba`` of the whole
sequence (forward-backward, i.e. smoothed with the future). Here:

* ``fit`` runs bounded EM (scaled forward-backward) on the rows it is given -- the caller passes TRAIN rows only;
* ``forward_filter`` emits, for every t, the filtered posterior P(s_t | x_1..x_t) using emissions <= t only, the
  entropy of that posterior and the one-step predictive surprise -log p(x_t | x_1..x_{t-1}) in nats. Rows whose
  inputs are not finite propagate the state belief through the transition matrix only (prediction step) and emit
  NaN; nothing is interpolated, nothing after t is read;
* states are re-ordered after fitting by increasing fitted mean of the ``order_dim`` input (the trailing realized
  volatility in the layer-2 pool) so the state encoding is stable across fits.
"""
from __future__ import annotations

import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))


def _log_emission(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """(n, k) log N(x_t | mu_j, diag(var_j))."""
    d = X[:, None, :] - means[None, :, :]
    return -0.5 * (np.sum(np.log(variances) + d * d / variances, axis=2) + X.shape[1] * LOG2PI)


def _forward_backward(logB: np.ndarray, A: np.ndarray, pi: np.ndarray):
    n, k = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    alpha = np.zeros((n, k))
    beta = np.zeros((n, k))
    c = np.zeros(n)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    loglik = float(np.sum(np.log(c)) + np.sum(logB.max(axis=1)))
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((k, k))
    for t in range(n - 1):
        xi = (alpha[t][:, None] * A) * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
        xi_sum += xi
    return gamma, xi_sum, loglik


def fit(X: np.ndarray, k: int = 3, *, max_iter: int = 100, tol: float = 1e-5, var_floor: float = 1e-6,
        order_dim: int = 0, seed: int = 0) -> dict:
    """EM on the finite rows of X (n, D) treated as ONE sequence (the caller restricts X to the fit period).
    Deterministic initialisation: means from the k quantile groups of ``order_dim``, variances = global variance,
    sticky transitions. Returns a parameter dict with the states ordered by increasing mean of ``order_dim``."""
    X = np.asarray(X, float)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    n, D = X.shape
    if n < 10 * k:
        raise ValueError("not enough finite rows to fit the HMM")
    order = np.argsort(X[:, order_dim])
    groups = np.array_split(order, k)
    means = np.stack([X[g].mean(axis=0) for g in groups])
    variances = np.tile(np.maximum(X.var(axis=0), var_floor), (k, 1))
    A = np.full((k, k), 0.05 / max(k - 1, 1))
    np.fill_diagonal(A, 0.95)
    pi = np.full(k, 1.0 / k)
    history = []
    prev = -np.inf
    for it in range(max_iter):
        logB = _log_emission(X, means, variances)
        gamma, xi_sum, ll = _forward_backward(logB, A, pi)
        history.append(ll)
        # M step
        pi = gamma[0] / gamma[0].sum()
        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        w = gamma.sum(axis=0)[:, None]
        means = (gamma.T @ X) / np.maximum(w, 1e-300)
        d2 = (X[:, None, :] - means[None, :, :]) ** 2
        variances = np.maximum(np.einsum("tj,tjd->jd", gamma, d2) / np.maximum(w, 1e-300), var_floor)
        if ll - prev < tol * max(1.0, abs(ll)) and it > 0:
            break
        prev = ll
    perm = np.argsort(means[:, order_dim])
    means, variances = means[perm], variances[perm]
    A = A[np.ix_(perm, perm)]
    pi = pi[perm]
    return {"k": int(k), "dims": int(D), "means": means.tolist(), "variances": variances.tolist(), "transmat": A.tolist(),
            "startprob": pi.tolist(), "loglik": float(history[-1]), "iterations": int(len(history)),
            "converged": bool(len(history) < max_iter), "loglik_history": [float(v) for v in history],
            "order_dim": int(order_dim), "fit_rows": int(n), "var_floor": float(var_floor), "max_iter": int(max_iter), "tol": float(tol)}


def forward_filter(X: np.ndarray, params: dict):
    """Causal filter. Returns (filtered (n,k), entropy_nats (n,), surprise_nats (n,)). Row t uses x_1..x_t only; a
    row with a non-finite input propagates the belief by the transition matrix and emits NaN at that row."""
    X = np.asarray(X, float)
    means, variances = np.asarray(params["means"]), np.asarray(params["variances"])
    A, pi = np.asarray(params["transmat"]), np.asarray(params["startprob"])
    n, k = X.shape[0], len(pi)
    filt = np.full((n, k), np.nan)
    ent = np.full(n, np.nan)
    surprise = np.full(n, np.nan)
    finite = np.isfinite(X).all(axis=1)
    logB = np.full((n, k), np.nan)
    if finite.any():
        logB[finite] = _log_emission(X[finite], means, variances)
    belief = pi.copy()            # predicted state distribution for the current row
    for t in range(n):
        if finite[t]:
            lb = logB[t]
            m = lb.max()
            b = np.exp(lb - m)
            joint = belief * b
            c = joint.sum()
            post = joint / c
            filt[t] = post
            p = post[post > 0]
            ent[t] = float(-np.sum(p * np.log(p)))
            surprise[t] = float(-(np.log(c) + m))
            belief = post @ A
        else:
            belief = belief @ A
    return filt, ent, surprise
