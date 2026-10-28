#!/usr/bin/env python3
"""NON_GOVERNING CPU witnesses against two exact peer revisions; no fit or deployment."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from scipy.stats import skew


def source(repo, rev, path):
    data = subprocess.check_output(["git", "-C", str(repo), "show", rev + ":" + path])
    return data.decode(), hashlib.sha256(data).hexdigest()


def audit(predictor_repo, strategy_repo):
    code, scale_sha = source(predictor_repo, "46af988e", "tools/layer1_pool_h1.py")
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decide")
    bounded = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "BOUNDED" for t in n.targets))
    ns = {"np": np, "BOUNDED": ast.literal_eval(bounded)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "retained_decide", "exec"), ns)
    x = 1e-3 * np.exp(np.random.default_rng(5).normal(0, 0.8, 5000))
    tr = np.arange(len(x)) < 4000
    altered = x.copy()
    altered[4500] = 0
    diag = {"adf_p": 0., "kpss_p": .1, "min": float(x[tr].min()),
            "skew": float(skew(x[tr])), "median_pos": float(np.median(x[tr]))}
    before, _, y = ns["decide"]("px.rv_4", x, dict(diag), train=tr)
    after, _, z = ns["decide"]("px.rv_4", altered, dict(diag), train=tr)
    scale = {"same_TRAIN": bool(np.array_equal(x[tr], altered[tr])),
             "before": before, "after": after,
             "same_transformed_TRAIN": bool(np.array_equal(y[tr], z[tr]))}
    assert scale == {"same_TRAIN": True, "before": "LOG", "after": "ASINHS",
                     "same_transformed_TRAIN": False}, scale

    code, signal_sha = source(strategy_repo, "793595d", "app/short_horizon/expert_guided/signals.py")
    name = "_retained_signals_20261022"
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(code, "retained_signals", "exec"), mod.__dict__)
    n = 12
    o = c = np.full(n, 100.)
    h, l, sigma = c + .5, c - .1, np.ones(n)
    p = mod.SignalParams(cost_sigma=0.)
    args = (o, h, l, c, sigma)
    sa, ha = mod.hindsight_signals(*args, [(7, 1, 100.5)], np.full(n, 3), p)
    sb, hb = mod.hindsight_signals(*args, [(7, -1, 99.9)], np.full(n, 3), p)
    ca = mod.closer_labels(sa, 1, np.full(n, 2.), p)[0]
    cb = mod.closer_labels(sb, 1, np.full(n, 2.), p)[0]
    cap = {"declared_horizon_a": int(ha["close"][0]), "declared_horizon_b": int(hb["close"][0]),
           "changed_pivot_bar": 7, "closer_before": float(ca), "closer_after": float(cb)}
    assert cap == {"declared_horizon_a": 3, "declared_horizon_b": 3,
                   "changed_pivot_bar": 7, "closer_before": 0., "closer_after": 1.}, cap

    c = np.array([1., 1.0005, .999, 1.002, 1.003, 1.001, .999, .997, .998, .9985])
    o, h, l, sigma = np.r_[c[0], c[:-1]], c + .0002, c - .0002, np.full(len(c), .001)
    pivots = [(2, -1, .9988), (4, 1, 1.0032), (7, -1, .9968)]
    s, hz = mod.hindsight_signals(o, h, l, c, sigma, pivots, np.full(len(c), 9), p)
    conflict = {"open_long": float(mod.opener_labels(s, 1, p)[0]),
                "close_long_with_SL2": float(mod.closer_labels(s, 1, np.full(len(c), 2.), p)[0]),
                "opportunity_sigma": float(s["opportunity"][1][0]),
                "adverse_to_peak_sigma": float(s["adverse_to_peak"][1][0]),
                "favourable_peak_bar": int(hz["open"][1][0])}
    assert conflict["open_long"] == conflict["close_long_with_SL2"] == 1., conflict
    assert conflict["adverse_to_peak_sigma"] < 2 and conflict["favourable_peak_bar"] == 4
    return {"scope": "NON_GOVERNING synthetic dependency/semantic witnesses, not real-data impact or feature leakage certification",
            "source_sha256": {"predictor_46af988e": scale_sha, "strategy_793595d": signal_sha},
            "transform_selection_future_domain_dependency": scale,
            "declared_target_horizon_omits_future_pivot_kind": cap,
            "opener_closer_objective_mismatch": conflict}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictor-repo", required=True)
    p.add_argument("--strategy-repo", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    report = audit(a.predictor_repo, a.strategy_repo)
    if a.output:
        from expert_package_worker import atomic
        atomic(Path(a.output), report)
    print(json.dumps(report, indent=2, allow_nan=False))
