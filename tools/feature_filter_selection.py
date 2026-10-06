#!/usr/bin/env python3
"""Phase 3: reproducible filter rankings per asset, target and horizon.

Runs, on exactly the phase-2 population (alias representatives plus every non-alias
feature) and TRAIN rows only:

* ``SPEARMAN_CLUSTER``: average-linkage hierarchical clustering on 1-|Spearman| from the
  phase-2 TRAIN matrix; for each K the tree is cut into K clusters and one representative
  per cluster is chosen by relevance (MI with the target), availability clock, coverage and
  cost;
* ``MRMR``: greedy max-relevance (MI with the target) minus mean redundancy (phase-2 MI
  between features);
* ``JMI``: greedy joint mutual information, sum over the selected set of I(f, s; y);
* controls: ``ALL_ADMISSIBLE``, ``UNIVARIATE_MI``, ``CAUSAL_SUPPORTED`` (phase-1 causal-rule
  support, a label and not a final selection), ``RANDOM_K`` with a fixed seed;
* declared causal variants ``MRMR_CAUSAL`` and ``JMI_CAUSAL``: the base score is min-max
  normalised across candidates at every greedy step and a predeclared weight times the
  causal indicator is added; NOT_IDENTIFIED (no phase-1 support) counts as zero.

Every ranking keeps its full order, the per-step score and its terms (relevance,
redundancy, complementarity, causality, cost).  Subsets are reported for
K in {4, 8, 12, 16, 24, 32} capped by the admissible population; no K is declared a winner
and no validation or test row is opened.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.fs_phase23_manifest import canonical_bytes, sha256_file
from tools.feature_pairwise_worker import row_key

METHOD_NAMES = ("SPEARMAN_CLUSTER", "MRMR", "JMI", "MRMR_CAUSAL", "JMI_CAUSAL",
                "ALL_ADMISSIBLE", "UNIVARIATE_MI", "CAUSAL_SUPPORTED", "RANDOM_K")
CAUSAL_LABEL = "CAUSAL_SUPPORTED"


class SelectionError(RuntimeError):
    pass


def default_params(*, seed: int, k_grid: tuple[int, ...]) -> dict:
    return {"method": "filter_v1", "k_grid": list(k_grid), "mi_estimator": "quantile_bins_plugin", "mi_bins": 8, "mi_seed": 0,
            "mi_edges": "per-feature quantile edges fit once on the feature's finite rows within the target's finite TRAIN support; target edges fit once on its finite rows",
            "joint_mi_estimator": "quantile_bins_plugin_3d", "min_support": 100, "random_seed": int(seed),
            "cluster_linkage": "average", "cluster_distance": "1-|spearman_TRAIN|",
            "causal_weight": 0.1, "causal_score_normalisation": "min-max over candidates at each greedy step",
            "cost_normalisation": "source_bytes / max source_bytes over admissible features",
            "not_identified_causal_evidence": 0.0}


def params_sha256(params: dict) -> str:
    return hashlib.sha256(canonical_bytes(params)).hexdigest()


# ----------------------------------------------------------------------------- inputs

def load_phase2_matrices(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"features": [str(f) for f in z["features"]], "spearman": z["spearman"], "mi": z["mi"],
                "admissible": z["admissible"].astype(bool), "population_id": str(z["population_id"]),
                "identity": str(z["identity"]), "alias_representative": [str(f) for f in z["alias_representative"]],
                "params_sha256": str(z["params_sha256"])}


def load_train_matrix(manifest: dict, data_root: Path, target_id: str):
    import pandas as pd
    data_root = Path(data_root)
    for key in ("features_file", "targets_file"):
        if any(tok in manifest["data"][key].lower() for tok in ("validation", "test", "holdout")):
            raise SelectionError(f"refusing non-TRAIN file {manifest['data'][key]}")
    fpath = data_root / manifest["data"]["features_file"]
    tpath = data_root / manifest["data"]["targets_file"]
    if sha256_file(fpath) != manifest["data"]["features_sha256"] or sha256_file(tpath) != manifest["data"]["targets_sha256"]:
        raise SelectionError("TRAIN file digests differ from the manifest")
    feats = [f["feature_id"] for f in manifest["features"]]
    target = next((t for t in manifest["targets"] if t["target_id"] == target_id), None)
    if target is None:
        raise SelectionError(f"target {target_id} is not in the manifest")
    X = pd.read_parquet(fpath, columns=feats).to_numpy(dtype=np.float64)
    y = pd.read_parquet(tpath, columns=[target["column"]])[target["column"]].to_numpy(dtype=np.float64)
    n = int(manifest["data"]["train_rows"])
    if len(X) != n or len(y) != n:
        raise SelectionError("TRAIN row count differs from the manifest")
    return feats, X, y


# ----------------------------------------------------------------------------- estimators

def _codes(v: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(v, np.linspace(0.0, 1.0, bins + 1))
    inner = np.unique(edges[1:-1])
    return np.searchsorted(inner, v, side="right")


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def mi_codes(cx: np.ndarray, cy: np.ndarray, bins: int) -> float:
    joint = np.bincount(cx * bins + cy, minlength=bins * bins).astype(np.float64)
    return _entropy(np.bincount(cx, minlength=bins).astype(float)) + _entropy(np.bincount(cy, minlength=bins).astype(float)) - _entropy(joint)


def joint_mi_codes(ca: np.ndarray, cb: np.ndarray, cy: np.ndarray, bins: int) -> float:
    """I((a,b); y) from a 3-D plug-in histogram."""
    ab = ca * bins + cb
    h_ab = _entropy(np.bincount(ab, minlength=bins * bins).astype(float))
    h_y = _entropy(np.bincount(cy, minlength=bins).astype(float))
    h_aby = _entropy(np.bincount(ab * bins + cy, minlength=bins * bins * bins).astype(float))
    return h_ab + h_y - h_aby


# ----------------------------------------------------------------------------- selection

def _rep_key(meta: dict, relevance: float):
    cov = meta.get("train_coverage")
    cov = float(cov) if isinstance(cov, (int, float)) else 0.0
    cost = meta.get("source_bytes")
    cost = float(cost) if isinstance(cost, (int, float)) else float("inf")
    return (-relevance, 0 if meta.get("clock") == "OBSERVED" else 1, -cov, cost, meta["feature_id"])


def _subset_grid(k_grid: tuple[int, ...], p: int) -> list[int]:
    return [k for k in k_grid if k <= p]


def run_filter_methods(manifest: dict, mats: dict, feats: list[str], X: np.ndarray, y: np.ndarray, *, target_id: str,
                       seed: int, k_grid: tuple[int, ...] = (4, 8, 12, 16, 24, 32), params: dict | None = None) -> dict:
    t0 = time.time()
    params = dict(default_params(seed=seed, k_grid=tuple(k_grid)), **(params or {}))
    psha = params_sha256(params)
    identity = manifest["identity"]
    pop = manifest["population_id"]
    # --- population and matrix validation (mixed populations and incomplete matrices are refused)
    if mats["population_id"] != pop:
        raise SelectionError(f"phase-2 matrices belong to population {mats['population_id']}, manifest is {pop}")
    if mats["identity"] != identity:
        raise SelectionError("phase-2 matrices carry a foreign identity for this population")
    if list(mats["features"]) != list(feats):
        raise SelectionError("phase-2 matrices feature order differs from the manifest population")
    p_all = len(feats)
    adm = np.asarray(mats["admissible"], dtype=bool)
    if adm.shape != (p_all,):
        raise SelectionError("admissible mask shape mismatch")
    idx = np.nonzero(adm)[0]
    for name in ("spearman", "mi"):
        m = np.asarray(mats[name], dtype=np.float64)
        if m.shape != (p_all, p_all):
            raise SelectionError(f"{name} matrix is not {p_all}x{p_all}")
        sub = m[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, 0.0)
        if not np.all(np.isfinite(sub)):
            raise SelectionError(f"{name} matrix is incomplete for the admissible population")
    target = next(t for t in manifest["targets"] if t["target_id"] == target_id)
    n = int(manifest["data"]["train_rows"])
    if X.shape != (n, p_all) or y.shape != (n,):
        raise SelectionError("TRAIN arrays do not match the manifest population")
    bins = int(params["mi_bins"])
    meta = {f["feature_id"]: f for f in manifest["features"]}
    adm_feats = [feats[i] for i in idx]
    p = len(adm_feats)
    k_list = _subset_grid(tuple(params["k_grid"]), p)
    causal = {c["feature_id"] for c in manifest["causal_supported"] if c["target_id"] == target_id}
    # alias members inherit their representative's support so labelling is group-aware
    rep_of = dict(zip(feats, mats["alias_representative"]))
    causal_adm = {f for f in adm_feats if f in causal or any(rep_of[c] == f for c in causal)}
    # --- relevance: MI(f; y) on rows where both are finite.  Quantile edges for every feature are fit once
    # on its own finite rows inside the target's finite support (declared in params["mi_edges"]); the
    # target's edges are fit once on its finite rows.  Pairwise terms then reuse the cached codes on the
    # rows the pair shares, which keeps JMI on hundreds of features within minutes.
    yf = np.isfinite(y)
    relevance = np.zeros(p)
    support = np.zeros(p, dtype=int)
    feat_finite: list[np.ndarray] = []
    feat_codes: list[np.ndarray] = []
    cy_all = np.full(n, -1, dtype=np.int64)
    cy_all[yf] = _codes(y[yf], bins)
    for k, i in enumerate(idx):
        m = yf & np.isfinite(X[:, i])
        feat_finite.append(m)
        codes = np.full(n, -1, dtype=np.int64)
        support[k] = int(m.sum())
        if support[k] >= int(params["min_support"]):
            codes[m] = _codes(X[m, i], bins)
            relevance[k] = mi_codes(codes[m], cy_all[m], bins)
        feat_codes.append(codes)
    mi_ff = np.asarray(mats["mi"], dtype=np.float64)[np.ix_(idx, idx)]
    np.fill_diagonal(mi_ff, 0.0)
    sp_ff = np.abs(np.asarray(mats["spearman"], dtype=np.float64)[np.ix_(idx, idx)])
    np.fill_diagonal(sp_ff, 1.0)
    cost_raw = np.array([float(meta[f].get("source_bytes")) if isinstance(meta[f].get("source_bytes"), (int, float)) else 0.0 for f in adm_feats])
    cost = cost_raw / cost_raw.max() if cost_raw.max() > 0 else cost_raw
    causal_ind = np.array([1.0 if f in causal_adm else float(params["not_identified_causal_evidence"]) for f in adm_feats])
    k_max = max(k_list) if k_list else 0
    joint_cache: dict[tuple[int, int], float] = {}

    def joint(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key not in joint_cache:
            m = feat_finite[a] & feat_finite[b]
            if m.sum() < int(params["min_support"]) or support[a] < int(params["min_support"]) or support[b] < int(params["min_support"]):
                joint_cache[key] = 0.0
            else:
                joint_cache[key] = joint_mi_codes(feat_codes[a][m], feat_codes[b][m], cy_all[m], bins)
        return joint_cache[key]

    def terms(k: int, selected: list[int]) -> dict:
        red = float(mi_ff[k, selected].mean()) if selected else 0.0
        return {"relevance": float(relevance[k]), "redundancy": red, "complementarity": None,
                "causality": float(causal_ind[k]), "cost": float(cost[k]), "support_n": int(support[k])}

    def greedy(kind: str, causal_variant: bool) -> list[dict]:
        selected: list[int] = []
        ranking = []
        remaining = list(range(p))
        depth = p if kind == "MRMR" else min(p, max(k_max, 1))
        while remaining and len(selected) < depth:
            base = np.empty(len(remaining))
            comp = np.empty(len(remaining))
            for c, k in enumerate(remaining):
                if kind == "MRMR":
                    red = float(mi_ff[k, selected].mean()) if selected else 0.0
                    base[c] = relevance[k] - red
                    comp[c] = float("nan")
                else:
                    comp[c] = float(sum(joint(k, s) for s in selected)) if selected else float(relevance[k])
                    base[c] = comp[c]
            score = base.copy()
            if causal_variant:
                lo, hi = float(base.min()), float(base.max())
                norm = (base - lo) / (hi - lo) if hi > lo else np.zeros_like(base)
                score = norm + float(params["causal_weight"]) * causal_ind[remaining]
            order = sorted(range(len(remaining)), key=lambda c: (-score[c], adm_feats[remaining[c]]))
            c = order[0]
            k = remaining.pop(c)
            t = terms(k, selected)
            t["complementarity"] = None if math.isnan(comp[c]) else float(comp[c])
            ranking.append({"rank": len(selected) + 1, "feature_id": adm_feats[k], "score": float(score[c]), "base_score": float(base[c]), "terms": t})
            selected.append(k)
        for k in sorted(remaining, key=lambda k: (-relevance[k], adm_feats[k])):  # JMI tail: never ranked greedily beyond k_max
            ranking.append({"rank": len(ranking) + 1, "feature_id": adm_feats[k], "score": None, "base_score": None,
                            "terms": dict(terms(k, selected), complementarity=None), "note": "beyond greedy depth; ordered by relevance"})
        return ranking

    methods: dict[str, dict] = {}
    # SPEARMAN_CLUSTER
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = 1.0 - sp_ff
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)
    cluster_subsets: dict[int, list[str]] = {}
    cluster_order: list[str] = []
    if p >= 2:
        Z = linkage(squareform(dist, checks=False), method=params["cluster_linkage"])
        for K in range(1, p + 1):
            labels = fcluster(Z, t=K, criterion="maxclust")
            reps = []
            for lab in sorted(set(labels)):
                members = [k for k in range(p) if labels[k] == lab]
                rep = sorted(members, key=lambda k: _rep_key(meta[adm_feats[k]], relevance[k]))[0]
                reps.append(adm_feats[rep])
            cluster_subsets[K] = sorted(reps)
            for f in sorted(reps, key=lambda f: -relevance[adm_feats.index(f)]):
                if f not in cluster_order:
                    cluster_order.append(f)
    else:
        cluster_subsets = {1: list(adm_feats)}
        cluster_order = list(adm_feats)
    ranking = [{"rank": r + 1, "feature_id": f, "score": float(relevance[adm_feats.index(f)]), "base_score": None,
                "terms": dict(terms(adm_feats.index(f), []), complementarity=None)} for r, f in enumerate(cluster_order)]
    methods["SPEARMAN_CLUSTER"] = {"ranking": ranking, "subsets": {K: cluster_subsets[K] for K in k_list if K in cluster_subsets},
                                   "subset_rule": "cut tree into K clusters; representative per cluster"}
    methods["MRMR"] = {"ranking": greedy("MRMR", False), "subset_rule": "ranking prefix"}
    methods["JMI"] = {"ranking": greedy("JMI", False), "subset_rule": "ranking prefix"}
    methods["MRMR_CAUSAL"] = {"ranking": greedy("MRMR", True), "subset_rule": "ranking prefix", "variant": "causal evidence added"}
    methods["JMI_CAUSAL"] = {"ranking": greedy("JMI", True), "subset_rule": "ranking prefix", "variant": "causal evidence added"}
    uni = sorted(range(p), key=lambda k: (-relevance[k], adm_feats[k]))
    methods["UNIVARIATE_MI"] = {"ranking": [{"rank": r + 1, "feature_id": adm_feats[k], "score": float(relevance[k]), "base_score": None,
                                             "terms": dict(terms(k, []), complementarity=None)} for r, k in enumerate(uni)], "subset_rule": "ranking prefix"}
    caus = sorted(range(p), key=lambda k: (-causal_ind[k], -relevance[k], adm_feats[k]))
    methods["CAUSAL_SUPPORTED"] = {"ranking": [{"rank": r + 1, "feature_id": adm_feats[k], "score": float(causal_ind[k]), "base_score": None,
                                                "terms": dict(terms(k, []), complementarity=None),
                                                "label": CAUSAL_LABEL if causal_ind[k] > 0 else "NOT_IDENTIFIED"} for r, k in enumerate(caus)],
                                   "subset_rule": "causal-supported features first (label, not final selection), then relevance",
                                   "causal_supported_count": int(causal_ind.sum())}
    rng = np.random.default_rng(int(params["random_seed"]))
    perm = rng.permutation(p)
    methods["RANDOM_K"] = {"ranking": [{"rank": r + 1, "feature_id": adm_feats[k], "score": None, "base_score": None,
                                        "terms": dict(terms(k, []), complementarity=None)} for r, k in enumerate(perm)],
                           "subset_rule": f"fixed-seed permutation prefix (seed {params['random_seed']})"}
    methods["ALL_ADMISSIBLE"] = {"ranking": [{"rank": r + 1, "feature_id": f, "score": None, "base_score": None,
                                              "terms": dict(terms(r, []), complementarity=None)} for r, f in enumerate(adm_feats)],
                                 "subset_rule": "whole admissible population regardless of K"}
    for name, m in methods.items():
        if "subsets" not in m:
            order = [r["feature_id"] for r in m["ranking"]]
            if name == "ALL_ADMISSIBLE":
                m["subsets"] = {p: list(adm_feats)}
            else:
                m["subsets"] = {K: sorted(order[:K]) for K in k_list}
    # --- rows
    rank_rows, subset_rows = [], []
    for name, m in methods.items():
        for r in m["ranking"]:
            label = r.get("label", CAUSAL_LABEL if r["feature_id"] in causal_adm else "NOT_IDENTIFIED")
            rank_rows.append({"run_id": identity, "population_id": pop, "target_id": target_id, "horizon_hours": target["horizon_hours"],
                              "method": name, "rank": r["rank"], "feature_id": r["feature_id"], "score": r["score"], "base_score": r["base_score"],
                              "terms": r["terms"], "causal_label": label, "params_sha256": psha,
                              "row_key": row_key(identity, "rank", target_id, name, r["rank"])})
        for K, members in m["subsets"].items():
            subset_rows.append({"run_id": identity, "population_id": pop, "target_id": target_id, "horizon_hours": target["horizon_hours"],
                                "method": name, "k": int(K), "members": list(members), "subset_sha256": hashlib.sha256(canonical_bytes(sorted(members))).hexdigest(),
                                "label": CAUSAL_LABEL if name == "CAUSAL_SUPPORTED" else "FILTER_CANDIDATE", "is_final_selection": False,
                                "params_sha256": psha, "row_key": row_key(identity, "subset", target_id, name, int(K))})
    return {"population_id": pop, "identity": identity, "target_id": target_id, "horizon_hours": target["horizon_hours"],
            "methods": {name: {k: v for k, v in m.items() if k != "ranking"} for name, m in methods.items()},
            "admissible": adm_feats, "admissible_count": p, "k_list": k_list, "params": params, "params_sha256": psha,
            "causal_supported": sorted(causal_adm), "causal_supported_label": CAUSAL_LABEL, "final_selection": False,
            "relevance_support": {adm_feats[k]: int(support[k]) for k in range(p)},
            "rows": {"feature_filter_rankings": rank_rows, "feature_filter_subsets": subset_rows},
            "wall_seconds": time.time() - t0}
