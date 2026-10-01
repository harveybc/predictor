"""Lane H: compare the deterministic successor with the historical MLE ``local_level_kalman`` (reference only)."""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")


def _series(T, q, r, seed):
    rng = random.Random(seed)
    g = lambda: sum(rng.random() for _ in range(12)) - 6.0
    cur, y = 0.0, []
    for _ in range(T):
        cur += math.sqrt(q) * g()
        y.append(cur + math.sqrt(r) * g())
    return np.array(y)


def compare(T=20000, q=0.05, r=0.5, seed=3):
    ops = _load("df_operators")
    y = _series(T, q, r, seed)
    n_tr = int(0.7 * T)
    X = y.reshape(-1, 1)
    art = kf.fit(kf.default_spec(kf.LOCAL_LEVEL), X[:n_tr], {"dataset_id": "synthetic", "role": "TRAIN",
                                                              "row_range": [0, n_tr], "column_ids": ["y"]})
    succ = art["fitted"]["per_column"][0]
    out = kf.transform_batch(art, X)
    rep = {"declared_regime": bool(0.01 <= q / r <= 1.0 and T >= 5000), "true": {"q": q, "r": r, "ratio": q / r},
           "successor": {"status": art["status"], "ratio": succ["q"] / succ["r"], "q": succ["q"], "r": succ["r"],
                         "clipped": [succ["r_clipped"], succ["q_clipped"]]}}
    hist = ops._fit_kernel({"kind": "local_level_kalman", "params": {}}, X[:n_tr])
    if hist["status"] != "FITTED":
        rep["historical"] = {"status": "ABSTAIN", "reason": hist["abstain_reason"]}
        return rep
    h = hist["fitted"]["per_column"][0]
    rep["historical"] = {"status": "FITTED", "ratio": math.exp(h["log_ratio"]), "q": h["level_var"], "r": h["obs_var"]}
    hout, _, _ = ops._transform_kernel(hist, X)
    hl = np.asarray(hout).reshape(-1)
    rep["max_abs_level_difference_over_obs_std"] = float(np.nanmax(np.abs(hl[200:] - out.arrays["level"][200:, 0])) / y.std())
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(compare(), indent=1))
