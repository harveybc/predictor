"""C132: multivariate profile inside TRAIN only: causal lead-lag, coherence null,
stable groups, pair cap. Synthetic arrays only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("test_df_profile_univariate_helpers_mv", ROOT / "tests/test_df_profile_univariate.py")
M = _load("df_profile_multivariate", ROOT / "tools/df_profile_multivariate.py")
MOD = ROOT / "tools/df_profile_multivariate.py"


def pair_rows(rows, a, b, metric):
    out = [r for r in rows if r.get("pair") == [a, b] and r["metric"] == metric]
    assert len(out) == 1, (metric, len(out))
    return out[0]


def test_known_three_sample_lead_is_recovered_with_direction():
    rng = np.random.default_rng(20)
    T = 5000
    lead = rng.standard_normal(T)
    lagged = np.empty(T)
    lagged[3:] = lead[:-3]
    lagged[:3] = rng.standard_normal(3)
    lagged += 0.3 * rng.standard_normal(T)
    ct = H.contract(T, 2)
    ids = [v["variable_id"] for v in ct["variables"]]
    rows = M.run_multivariate(ct, np.column_stack([lead, lagged]))
    a, b = sorted(ids)
    lag = pair_rows(rows, a, b, "best_signed_lag")["value"]
    # +k means the first pair member leads the second
    expected = 3.0 if a == ids[0] else -3.0
    assert lag == expected
    assert pair_rows(rows, a, b, "best_lag_stable")["value"] == 1.0
    assert pair_rows(rows, a, b, "best_lag_modal_fraction")["value"] >= 0.75
    assert abs(pair_rows(rows, a, b, "best_signed_lag_correlation")["value"]) > 0.9
    assert all(r["partition"] == "train" for r in rows)
    H.check_rows(rows, MOD)


def test_lagged_correlation_is_causal_past_with_present():
    rng = np.random.default_rng(21)
    a = rng.standard_normal(2000)
    b = np.roll(a, 2)                       # b_t = a_{t-2}
    assert M.causal_lagged_corr(a, b, 2) > 0.99   # a's past with b's present
    assert abs(M.causal_lagged_corr(b, a, 2)) < 0.1


def test_independent_noise_coherence_inside_null_interval():
    rng = np.random.default_rng(22)
    T = 8000
    ct = H.contract(T, 2)
    rows = M.run_multivariate(ct, rng.standard_normal((T, 2)))
    a, b = sorted(v["variable_id"] for v in ct["variables"])
    obs = pair_rows(rows, a, b, "coherence_band_mean")["value"]
    lo = pair_rows(rows, a, b, "coherence_null_q025")["value"]
    hi = pair_rows(rows, a, b, "coherence_null_q975")["value"]
    pct = pair_rows(rows, a, b, "coherence_null_percentile")["value"]
    assert lo <= obs <= hi and 2.5 <= pct <= 97.5
    mi = pair_rows(rows, a, b, "mi_excess_over_shuffle_bits")["value"]
    assert abs(mi) < 0.01


def test_two_correlated_groups_are_stable_and_random_data_is_not_identified():
    rng = np.random.default_rng(23)
    T = 3000
    f1, f2 = rng.standard_normal(T), rng.standard_normal(T)
    X = np.column_stack([f1 + 0.4 * rng.standard_normal(T) for _ in range(3)] +
                        [f2 + 0.4 * rng.standard_normal(T) for _ in range(3)])
    ct = H.contract(T, 6)
    ids = [v["variable_id"] for v in ct["variables"]]
    rows = M.run_multivariate(ct, X)
    groups = [r for r in rows if r["metric"] == "cluster_stability"]
    assert len(groups) == 2
    assert all(r["reason"] == "GROUP_IDENTIFIED" and r["value"] >= 0.75 for r in groups)
    members = sorted(sorted(r["estimator"]["params"]["members"]) for r in groups)
    assert members == sorted([sorted(ids[:3]), sorted(ids[3:])])
    assert H.pick(rows, "identified_group_count", "train")["value"] == 2.0

    rnd = M.run_multivariate(H.contract(T, 6), rng.standard_normal((T, 6)))
    cnt = H.pick(rnd, "identified_group_count", "train")
    assert cnt["value"] == 0.0 and cnt["reason"] == "GROUP_NOT_IDENTIFIED"
    assert all(r["reason"] == "GROUP_NOT_IDENTIFIED" for r in rnd if r["metric"] == "cluster_stability")
    H.check_rows(rows + rnd, MOD)


def test_only_train_is_read_and_candidates_are_labelled():
    rng = np.random.default_rng(24)
    T = 2000
    X = rng.standard_normal((T, 3))
    ct = H.contract(T, 3)
    s = ct["partitions"]["boundaries"]["calibration"][0]
    Y = X.copy()
    Y[s:] = rng.standard_normal((T - s, 3)) * 100
    strip = lambda rows: [{k: v for k, v in r.items() if k != "cpu_seconds"} for r in rows]
    rx = M.run_multivariate(ct, X)
    assert strip(rx) == strip(M.run_multivariate(ct, Y))
    cands = [r for r in rx if r["metric"] in ("pc1_loading", "pc1_common_variance_share",
                                               "private_residual_variance_share")]
    assert len(cands) == 9 and all(r["reason"] == "CANDIDATE_NOT_A_TRANSFORMATION" for r in cands)


def test_pair_cap_rule_is_deterministic_by_variable_id():
    vids = [f"{i:02d}" for i in range(60)][::-1]
    order, pvars, mvars = M.pair_cap(vids)
    assert len(pvars) == 50 and len(pvars) * (len(pvars) - 1) // 2 <= M.MAX_PAIRS
    assert [vids[i] for i in pvars] == sorted(vids)[:50]
    assert len(mvars) == 60


def test_effective_rank_of_rank_two_correlation():
    rng = np.random.default_rng(25)
    T = 3000
    X = rng.standard_normal((T, 2)) @ rng.standard_normal((2, 6))
    rows = M.run_multivariate(H.contract(T, 6), X)
    er = H.pick(rows, "effective_rank", "train")["value"]
    assert 1.0 < er <= 2.0 + 1e-3
    ratios = sum(H.pick(rows, f"pca_explained_variance_ratio_pc{i}", "train")["value"] for i in (1, 2))
    assert abs(ratios - 1.0) < 1e-9
