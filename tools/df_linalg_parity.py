#!/usr/bin/env python3
"""C169 (order 2026-09-13): declared numeric parity for linear-algebra descriptors.

PCA explained-variance ratios, effective rank and PC1 loadings/shares come out
of LAPACK (eigh, svd). Two paths or two stacks (numpy 1.26 + MKL, numpy 2.5 +
OpenBLAS) agree to about 1e-16..1e-14, not bit for bit. Equality of these
values is therefore never claimed. Instead, declared before any POST:

* TOLERANCES: per metric, an absolute and a relative tolerance; two raw values
  agree when |a - b| <= max(abs, rel * max(|a|, |b|)) (math.isclose);
* EXACT_METRICS: integer descriptors of the same decompositions, compared
  exactly (a count is either equal or not);
* CANONICAL_SIGNIFICANT_DIGITS: the portable representation of a tolerance
  metric is its value formatted with that many significant digits, in a
  separate field (`value_canonical`); the raw value is always kept unchanged.
  Only the canonical string may enter a digest meant to be portable across
  hosts and stacks;
* PC1 sign rule: the loading vector is oriented so its sum is positive; when
  |sum| <= PC1_SIGN_TOL the first component with |u_i| > PC1_SIGN_TOL (in
  variable_id order) is made positive, so an almost balanced vector does not
  flip sign between stacks.

TOLERANCES_SHA256 binds all of it; the parity test pins that digest.

Provenance (`linalg_provenance()`): numpy version and the linear-algebra
backends threadpoolctl reports (internal API, user API, version; never file
paths), recorded in the estimator params of every such row.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "crispdm.data_foundation.linalg_parity_tolerance.v1"
CANONICAL_SIGNIFICANT_DIGITS = 9
PC1_SIGN_TOL = 1e-8
_TOL = {"abs": 1e-12, "rel": 1e-10}
# metric name (or prefix ending in "_pc") -> tolerance, per module
TOLERANCES = {
    "df_profile_multivariate": {
        "pca_explained_variance_ratio_pc": dict(_TOL, match="prefix", decomposition="numpy.linalg.eigh"),
        "effective_rank": dict(_TOL, match="exact", decomposition="numpy.linalg.eigh"),
        "pc1_loading": dict(_TOL, match="exact", decomposition="numpy.linalg.eigh"),
        "pc1_common_variance_share": dict(_TOL, match="exact", decomposition="numpy.linalg.eigh"),
        "private_residual_variance_share": dict(_TOL, match="exact", decomposition="numpy.linalg.eigh"),
    },
    "df_profile_information": {
        "effective_rank": dict(_TOL, match="exact", decomposition="numpy.linalg.svd"),
    },
}
EXACT_METRICS = {"df_profile_multivariate": ("negative_eigenvalue_count",),
                 "df_profile_information": ("numerical_rank",)}


def _declaration() -> dict:
    return {"schema": SCHEMA, "tolerances": TOLERANCES, "exact_metrics": {k: list(v) for k, v in EXACT_METRICS.items()},
            "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS, "pc1_sign_tol": PC1_SIGN_TOL,
            "agreement": "abs(a - b) <= max(abs, rel * max(abs(a), abs(b)))",
            "canonical": "format(value, '.{d-1}e') with -0 written as 0; raw value kept in `value`"}


TOLERANCES_SHA256 = hashlib.sha256(json.dumps(_declaration(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tolerance_for(module: str, metric: str):
    for name, t in TOLERANCES.get(module, {}).items():
        if (t["match"] == "prefix" and metric.startswith(name)) or metric == name:
            return t
    return None


def is_exact_linalg(module: str, metric: str) -> bool:
    return metric in EXACT_METRICS.get(module, ())


def canonical(value) -> str | None:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    if v == 0.0:
        v = 0.0
    return format(v, f".{CANONICAL_SIGNIFICANT_DIGITS - 1}e")


def agree(module: str, metric: str, a, b) -> bool:
    t = tolerance_for(module, metric)
    if t is None:
        raise KeyError(f"no declared tolerance for {module}.{metric}")
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=t["rel"], abs_tol=t["abs"])


def pc1_sign(u) -> float:
    s = float(sum(float(x) for x in u))
    if abs(s) > PC1_SIGN_TOL:
        return 1.0 if s > 0 else -1.0
    for x in u:
        if abs(float(x)) > PC1_SIGN_TOL:
            return 1.0 if float(x) > 0 else -1.0
    return 1.0


_PROVENANCE = None


def linalg_provenance() -> dict:
    """numpy version and linear-algebra backends, without file paths; computed once per process."""
    global _PROVENANCE
    if _PROVENANCE is None:
        import numpy as np
        try:
            from threadpoolctl import threadpool_info
            backends = sorted({(str(d.get("user_api")), str(d.get("internal_api")), str(d.get("version")))
                               for d in threadpool_info()})
            backends = [{"user_api": u, "internal_api": i, "version": v} for u, i, v in backends]
        except ImportError:  # pragma: no cover
            backends = "THREADPOOLCTL_UNAVAILABLE"
        _PROVENANCE = {"numpy_version": np.__version__, "linalg_backends": backends,
                       "tolerance_sha256": TOLERANCES_SHA256,
                       "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS,
                       "pc1_sign_rule": f"sum > 0; if |sum| <= {PC1_SIGN_TOL}, first |u_i| > {PC1_SIGN_TOL} positive"}
    return _PROVENANCE
