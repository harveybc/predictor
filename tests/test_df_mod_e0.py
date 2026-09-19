"""MOD-E0-DEV acceptance tests (RP3, ML01-ML10), declared before any pilot outcome. Each critical
block has a deliberate alteration of the productive path that must fail its test."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("df_mod_e0")
D = _load("df_mod_e0_design")
FAST = {**E.TRAINING, "max_updates": 40, "max_epochs": 5}
LINEAR_TOLERANCE = 0.03      # ML07 bar: within this MASE of the observer-attainable linear window reference


def test_ML01_the_generator_reproduces_its_equations_population_and_distribution():
    g = E.generate(2, 1, 11)
    p = g["params"]
    x, s, per, noise, cross = g["x"], g["s"], g["periodic"], g["noise"], g["cross"]
    assert np.allclose(x, s + per + noise + cross)                                            # composition
    eps = s[1:] - np.array([p["groups"][gg]["phi"] for gg in p["latent_groups"]]) * s[:-1]
    assert abs(eps.std() - E.SIGMA_AR) < 0.03 and abs(noise.std() - E.SIGMA_NOISE) < 0.02        # AR and noise draws
    for k, gg in enumerate(p["latent_groups"]):
        t = np.arange(p["n"])
        assert np.allclose(per[:, k], E.AMPLITUDE * np.sin(2 * math.pi * t / p["groups"][gg]["period"] + p["thetas"][k]))
    partner = int(p["partner"]["4"])
    assert np.allclose(cross[E.TAU:, 4], E.BETA * (s + per + noise)[:-E.TAU, partner])         # lagged cross term
    # deterministic in the seed; different seeds differ; the same seed at another level differs in phi/period only
    assert np.array_equal(E.generate(2, 1, 11)["x"], x) and not np.array_equal(E.generate(2, 1, 12)["x"], x)
    lat = E.latent_groups()
    assert lat == ["A"] * 4 + ["B"] * 4
    # labels/metadata/latent groups never enter the features: windows are built from x only
    W = E.make_windows(x, np.array([100]), 8)
    assert W.shape == (1, 8, 8) and np.array_equal(W[0], x[93:101])


def test_ML01_the_causal_oracle_uses_the_past_only_and_beats_the_naive():
    g = E.generate(3, 1, 5)
    x, o = g["x"], g["oracle"]
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(x, o, periods)
    P = prep["parts"]["validation"]
    oracle = E.mase(P["oracle"], P["y"], prep["mase_denominator"])["mase_mean"]
    naive = E.mase(P["naive"], P["y"], prep["mase_denominator"])["mase_mean"]
    assert oracle < naive
    # the oracle at row t must not change when the future beyond t is perturbed: the REAL callable is
    # re-run on components whose rows > t are altered (RP11 / review F4: the former check perturbed nothing)
    prm = g["params"]
    t_cut = 1500
    s2, src2 = g["s"].copy(), g["source"].copy()
    s2[t_cut + 1:] += 50.0
    src2[t_cut + 1:] -= 30.0
    o2 = E.causal_oracle(prm["groups"], prm["latent_groups"], prm["thetas"], s2, src2)
    assert np.array_equal(o2[:t_cut + 1], o[:t_cut + 1]) and not np.allclose(o2[t_cut + 1:-1], o[t_cut + 1:-1])
    # and a perturbation AT t changes the oracle at t (it does consume the present)
    s3 = g["s"].copy()
    s3[t_cut] += 1.0
    o3 = E.causal_oracle(prm["groups"], prm["latent_groups"], prm["thetas"], s3, g["source"])
    assert not np.allclose(o3[t_cut], o[t_cut]) and np.array_equal(o3[:t_cut], o[:t_cut])
    # the lagged partner enters through a[t + 1 - tau] <= t: perturbing source rows > t + 1 - tau leaves oracle[t]
    src4 = g["source"].copy()
    src4[t_cut + 2 - E.TAU:] += 7.0
    o4 = E.causal_oracle(prm["groups"], prm["latent_groups"], prm["thetas"], g["s"], src4)
    assert np.array_equal(o4[:t_cut + 1], o[:t_cut + 1]) and not np.allclose(o4[t_cut + 1:-1], o[t_cut + 1:-1])


def test_ML02_future_and_test_changes_do_not_change_past_outputs_or_the_fit():
    g = E.generate(2, 1, 3)
    x, o = g["x"], g["oracle"]
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(x, o, periods)
    b = prep["boundaries"]
    assert b["train"][1] + b["purge"] == b["validation"][0] and b["validation"][1] + b["purge"] == b["test"][0]
    x2 = x.copy()
    x2[b["test"][0]:] += 50.0                                                                # the test region is altered
    prep2 = E.prepare(x2, o, periods)
    assert np.array_equal(prep["parts"]["train"]["X"], prep2["parts"]["train"]["X"])            # windows, scaling and
    assert prep["scale"] == prep2["scale"] and prep["mase_denominator"] == prep2["mase_denominator"]   # denominators unchanged
    prof1 = E.profiles(x[b["train"][0]:b["train"][1]])
    prof2 = E.profiles(x2[b["train"][0]:b["train"][1]])
    assert np.array_equal(prof1["scaled"], prof2["scaled"])                                   # groups from training only
    # roles by identity: a window at row t consumes rows t-W+1..t
    rows = prep["parts"]["validation"]["rows"]
    assert np.array_equal(prep["parts"]["validation"]["X"][0], x[rows[0] - E.WINDOW + 1:rows[0] + 1])
    assert np.array_equal(prep["parts"]["validation"]["y"][0], x[rows[0] + 1])
    # cost pilots have no test split
    assert "test" not in E.prepare(x, o, periods, test_access=False)["parts"]


def test_ML03_H2_random_assignments_change_the_partition_but_keep_sizes_and_never_relabel():
    prof = [0, 0, 0, 0, 1, 1, 1, 1]
    cands = E.random_assignments(8, 2, [4, 4], 8, seed=5)
    assert all(sorted([c.count(0), c.count(1)]) == [4, 4] for c in cands)
    kept = [c for c in cands if not E.same_partition(c, prof)]
    assert len(kept) >= 3 and all(E.adjusted_rand(c, prof) < 1.0 for c in kept)
    assert E.same_partition([1, 1, 1, 1, 0, 0, 0, 0], prof)                                   # a relabeling IS the same partition
    m_prof = E.build_modular(prof, E.WINDOW, 8, fusion="sequence", seed=1)
    m_rand = E.build_modular(kept[0], E.WINDOW, 8, fusion="sequence", seed=1)
    assert E.count_params(m_prof) == E.count_params(m_rand)                                    # same branches, sizes, capacity
    assert [l.name for l in m_prof.layers] == [l.name for l in m_rand.layers]


def test_ML04_H3_pairs_preserve_marginals_in_distribution_and_change_the_lagged_dependence():
    var0, var1, xc0, xc1, acf0, acf1 = [], [], [], [], [], []
    for seed in (1, 2, 3, 4):
        g0, g1 = E.generate(2, 0, seed), E.generate(2, 1, seed)
        for k in range(4, 8):
            var0.append(np.var(g0["x"][:, k]))
            var1.append(np.var(g1["x"][:, k]))
            acf0.append(E.acf(g0["x"][:, k], (1,))[0])
            acf1.append(E.acf(g1["x"][:, k], (1,))[0])
            a = int(g1["params"]["partner"][str(k)])
            xc0.append(np.corrcoef(g0["x"][:-E.TAU, a], g0["x"][E.TAU:, k])[0, 1])
            xc1.append(np.corrcoef(g1["x"][:-E.TAU, a], g1["x"][E.TAU:, k])[0, 1])
    assert abs(np.mean(var0) - np.mean(var1)) / np.mean(var1) < 0.05                          # marginal variance preserved
    assert abs(np.mean(acf0) - np.mean(acf1)) < 0.05                                           # own persistence preserved (phantom partner)
    assert np.mean(xc1) > 0.5 and abs(np.mean(xc0)) < 0.1                                      # dependence present only under r = 1
    # the dependence changes the information available to the target: the oracle gains under r = 1
    g0, g1 = E.generate(2, 0, 9), E.generate(2, 1, 9)
    for g, expect_gain in ((g0, False), (g1, True)):
        periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
        prep = E.prepare(g["x"], g["oracle"], periods)
        P = prep["parts"]["validation"]
        # a marginal oracle ignoring the cross term: for r = 0 it IS the full oracle; for r = 1 it is worse
        full = E.mase(P["oracle"], P["y"], prep["mase_denominator"])["per_variable"]
        marg_pred = P["oracle"].copy()
        if expect_gain:
            marg_pred[:, 4:] -= E.BETA * np.vstack([g["x"][r - E.TAU + 1, :4] * 0 + (g["s"] + g["periodic"] + g["noise"])[r + 1 - E.TAU, :4] for r in P["rows"]])
        marg = E.mase(marg_pred, P["y"], prep["mase_denominator"])["per_variable"]
        gain = np.mean([marg[k]["mase"] - full[k]["mase"] for k in range(4, 8)])
        assert (gain > 0.05) == expect_gain
    # a temporal shuffle is NOT the control: it destroys the own autocorrelation
    sh = np.random.default_rng(0).permutation(g1["x"][:, 4])
    assert abs(E.acf(sh, (1,))[0]) < 0.1 < abs(E.acf(g1["x"][:, 4], (1,))[0])


def test_ML05_ML06_the_frozen_extractor_gives_identical_activations_and_only_fusion_head_train(tmp_path):
    tf = E._tf()
    g = E.generate(2, 1, 4)
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(g["x"], g["oracle"], periods)
    P, s = prep["parts"], prep["scale"]
    prof = E.profiles(g["x"][prep["boundaries"]["train"][0]:prep["boundaries"]["train"][1]])
    assign = E.average_linkage(prof["scaled"], 2)
    ext = E.build_modular(assign, E.WINDOW, 8, fusion="sequence", seed=1)
    E.fit(ext, E._sx(P["train"]["X"][:256], s), E._sy(P["train"]["y"][:256], s), E._sx(P["validation"]["X"], s), E._sy(P["validation"]["y"], s), training=FAST)
    ext.save_weights(str(tmp_path / "ext.weights.h5"))
    seq = E.build_modular(assign, E.WINDOW, 8, fusion="sequence", seed=2)
    summ = E.build_modular(assign, E.WINDOW, 8, fusion="summary", seed=3)
    for m in (seq, summ):
        for name in E.extractor_layer_names(ext):
            m.get_layer(name).set_weights(ext.get_layer(name).get_weights())
        info = E.freeze_extractor(m)
        assert set(info["frozen"]) == set(E.extractor_layer_names(ext)) and "head" in info["trainable"]
    # identical activations at the adapter output for both arms (ML05), temporal axis kept (ML06)
    X = E._sx(P["validation"]["X"][:16], s)
    act = {}
    for tag, m in (("seq", seq), ("summ", summ)):
        sub = tf.keras.Model(m.input, m.get_layer("g0_adapt").output)
        act[tag] = sub.predict(X, verbose=0)
    assert np.allclose(act["seq"], act["summ"]) and act["seq"].shape == (16, E.WINDOW, 8)
    # only fusion/core/head weights move under training; the extractor does not (ML05); gradients are real (ML06)
    before = {n: [w.copy() for w in seq.get_layer(n).get_weights()] for n in E.extractor_layer_names(seq)}
    rec = E.fit(seq, E._sx(P["train"]["X"][:256], s), E._sy(P["train"]["y"][:256], s), E._sx(P["validation"]["X"], s), E._sy(P["validation"]["y"], s), training=FAST)
    for n in before:
        assert all(np.array_equal(a, b) for a, b in zip(seq.get_layer(n).get_weights(), before[n]))
    assert rec["weight_change_by_layer"]["head"] > 0 and rec["weight_change_by_layer"]["core_conv"] > 0 and rec["updates"] > 0
    # summary arm: the temporal axis is reduced BEFORE fusion (declared), the sequence arm keeps it into the core
    assert seq.get_layer("fusion_seq").output.shape[1:] == (E.WINDOW, 16) and summ.get_layer("fusion_vec").output.shape[1:] == (16,)
    assert 0.85 <= E.count_params(summ)["trainable"] / E.count_params(seq)["trainable"] <= 1.15     # capacity tolerance
    # deliberate alteration: unfreezing the extractor makes ML05 fail
    for l in seq.layers:
        l.trainable = True
    rec2 = E.fit(seq, E._sx(P["train"]["X"][:256], s), E._sy(P["train"]["y"][:256], s), E._sx(P["validation"]["X"], s), E._sy(P["validation"]["y"], s), training=FAST)
    assert any(rec2["weight_change_by_layer"][n] > 0 for n in before)


def tmp_path_factory_dir():
    import tempfile
    return Path(tempfile.mkdtemp(prefix="ml07-"))


def test_ML07_positive_learning_with_the_real_receiver_and_grouping_recovers_latent_groups():
    g = E.generate(3, 1, 8)
    b = E.boundaries()
    prof = E.profiles(g["x"][b["train"][0]:b["train"][1]])
    labels = E.average_linkage(prof["scaled"], 2)
    assert E.adjusted_rand(labels, [0] * 4 + [1] * 4) == 1.0                                  # grouping positive control
    g0 = E.generate(0, 1, 8)
    prof0 = E.profiles(g0["x"][b["train"][0]:b["train"][1]])
    assert not prof0["kept"] == [] and sum(prof["kept"]) >= 10
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(g["x"], g["oracle"], periods)
    P, s = prep["parts"], prep["scale"]
    rec = E.run_cell({"cell_id": "ml07", "hypothesis": "H2", "level": 3, "r": 1, "seed": 8, "arm": "profiles", "role": "CELL"}, tmp_path_factory_dir())
    v = rec["scores"]["validation"]
    m, naive, oracle, linear = v["model"]["mase_mean"], v["naive"]["mase_mean"], v["oracle"]["mase_mean"], v["linear_window"]["mase_mean"]
    assert oracle < m < naive, (oracle, m, naive)                                             # learns; bounded by the generator oracle
    assert m <= linear + LINEAR_TOLERANCE, (m, linear)                                         # reaches the attainable linear bar
    assert rec["training"]["curve"]["train"][-1] < rec["training"]["curve"]["train"][0] and rec["training"]["updates"] > 100
    # deliberate alteration: the plain receiver of the first diagnostic (no residual path) would not pass;
    # the sealed rule is recorded in the cell
    assert rec["training"]["rule"]["loss"] == "mse" and rec["training"]["rule"]["learning_rate"] == 3e-3 and E.ACTIVATION == "elu"


def test_ML08_early_stopping_restores_the_best_validation_weights_and_reload_reproduces(tmp_path):
    g = E.generate(2, 1, 6)
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(g["x"], g["oracle"], periods)
    P, s = prep["parts"], prep["scale"]
    model = E.build_modular([0] * 4 + [1] * 4, E.WINDOW, 8, fusion="sequence", seed=1)
    rec = E.fit(model, E._sx(P["train"]["X"][:512], s), E._sy(P["train"]["y"][:512], s), E._sx(P["validation"]["X"], s), E._sy(P["validation"]["y"], s),
                training={**E.TRAINING, "max_updates": 200, "max_epochs": 25, "early_stopping": {"monitor": "validation MAE", "patience": 2, "restore_best": True}})
    va = rec["curve"]["validation"]
    restored = model.evaluate(E._sx(P["validation"]["X"], s), E._sy(P["validation"]["y"], s), verbose=0)
    assert abs(restored - min(va)) < 1e-5                                                      # the best checkpoint is really restored
    assert rec["restored_checkpoint_epoch"] == int(np.argmin(va)) + 1
    model.save_weights(str(tmp_path / "w.weights.h5"))
    again = E.build_modular([0] * 4 + [1] * 4, E.WINDOW, 8, fusion="sequence", seed=9)
    again.load_weights(str(tmp_path / "w.weights.h5"))
    assert np.allclose(again.predict(E._sx(P["validation"]["X"], s), verbose=0), model.predict(E._sx(P["validation"]["X"], s), verbose=0), atol=1e-6)
    assert rec["stop_reason"] in ("EARLY_STOPPING", "UPDATE_BUDGET", "EPOCH_BUDGET")


def test_ML09_metrics_from_arrays_with_shared_denominators_and_zero_policy():
    y = np.array([[1.0, 2.0], [2.0, 2.0], [4.0, 2.0]])
    pred = np.array([[1.5, 2.0], [2.0, 2.0], [3.0, 2.0]])
    out = E.mase(pred, y, [0.5, 0.0])
    assert out["per_variable"][0]["mase"] == pytest.approx((0.5 + 0 + 1.0) / 3 / 0.5)
    assert out["per_variable"][1]["status"] == "NO_APLICA" and out["per_variable"][1]["mase"] is None
    assert out["mase_mean"] == pytest.approx(out["per_variable"][0]["mase"])                    # NO_APLICA is not zero
    assert out["status"] == "MEDIDO_PARTIAL" and out["per_variable"][0]["mse"] == pytest.approx((0.25 + 0 + 1.0) / 3)
    # RP11 (review F4): the PRODUCTIVE metric refuses to call a non-finite or empty value measured
    nan = E.mase(np.array([[np.nan, 1.0]]), np.array([[1.0, 1.0]]), [1.0, 1.0])
    assert nan["status"] == E.NO_MEDIDO and nan["mase_mean"] is None and nan["mae_mean"] is None
    assert nan["per_variable"][0] == {"mae": None, "mse": None, "rmse": None, "mase": None, "status": E.NO_MEDIDO, "reason": "NON_FINITE"}
    assert nan["per_variable"][1]["status"] == E.MEDIDO
    inf = E.mase(np.array([[np.inf], [1.0]]), np.array([[1.0], [1.0]]), [1.0])
    assert inf["status"] == E.NO_MEDIDO and inf["rmse_mean"] is None
    empty = E.mase(np.zeros((0, 2)), np.zeros((0, 2)), [1.0, 1.0])
    assert empty["status"] == E.NO_MEDIDO and empty["per_variable"][0]["reason"] == "EMPTY"
    assert E.mase(np.zeros((3, 1)), np.zeros((3, 1)), [0.0])["status"] == E.NO_APLICA
    with pytest.raises(ValueError):                                                              # schema, not an observation
        E.mase(np.zeros((3, 2)), np.zeros((3, 3)), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        E.mase(np.zeros((3, 2)), np.zeros((3, 2)), [1.0])
    with pytest.raises(ValueError):
        E.mase(np.zeros(3), np.zeros(3), [1.0])


def test_RP12_trend_and_seasonal_strength_follow_var_T_plus_R_and_var_S_plus_R():
    t = np.arange(480)
    period = 24
    sine = np.sin(2 * np.pi * t / period)
    ramp = 0.01 * t
    noise = np.random.default_rng(0).normal(size=480)

    def reference(x):                                                     # independent formula on the SAME declared decomposition
        d = E.decompose_moving_average(x, period)
        T, S, R = d["trend"], d["seasonal"], d["remainder"]
        ft = 1 - np.var(R) / np.var(T + R) if np.var(T + R) > 0 else 0.0
        fs = 1 - np.var(R) / np.var(S + R) if np.var(S + R) > 0 else 0.0
        return [max(0.0, ft), max(0.0, fs)]
    for x in (sine, ramp, noise, sine + ramp, sine + 0.3 * noise, ramp + 0.3 * noise, np.ones(480), np.sin(2 * np.pi * t / 7)):
        assert np.allclose(E.trend_seasonal_strength(x, period), reference(x))
    ft, fs = E.trend_seasonal_strength(sine, period)
    assert ft < 0.05 and fs > 0.95                                        # pure sine of the declared period: seasonal, no trend
    ft, fs = E.trend_seasonal_strength(ramp, period)
    assert ft > 0.95 and fs < 0.1                                         # pure trend
    assert E.trend_seasonal_strength(np.ones(480), period)[0] == 0.0     # constant: zero denominators -> 0 by convention
    ft, fs = E.trend_seasonal_strength(noise, period)
    assert ft < 0.15 and fs < 0.15                                        # white noise: neither
    ft, fs = E.trend_seasonal_strength(sine + ramp, period)
    assert ft > 0.9 and fs > 0.8                                          # mixture: both
    # review F3: the v1 descriptor called a pure sine "trend-strong" (0.99); v2 does not; the v1 stays only for reanalysis
    assert E.trend_seasonal_strength_v1(sine, period)[0] > 0.9 and E.DESCRIPTOR_VERSION == 2
    # edges: short series (below one period) and a length that is not a multiple of the period do not fail
    assert len(E.trend_seasonal_strength(sine[:10], period)) == 2 and len(E.trend_seasonal_strength(sine[:101], period)) == 2
    assert E.decompose_moving_average(sine[:10], period)["period_truncated"] and not E.decompose_moving_average(sine, period)["period_truncated"]
    with pytest.raises(ValueError):
        E.trend_seasonal_strength(np.zeros(0), period)
    d = E.decompose_moving_average(sine, period)
    assert d["availability"] == "TRAIN_BATCH_CENTRED_MA_NOT_CAUSAL"       # declared: a centred window, not an online operator
    assert np.allclose(d["trend"] + d["seasonal"] + d["remainder"], sine)
    # the profiles record their descriptor version and are computed on train rows only
    g = E.generate(3, 1, 8)
    lo, hi = E.boundaries()["train"]
    prof = E.profiles(g["x"][lo:hi])
    assert prof["descriptor_version"] == 2 and E.profiles(g["x"][lo:hi], 1)["descriptor_version"] == 1


def test_ML10_effects_are_computed_with_the_declared_sign_and_replicates_do_not_inflate_units():
    V = _load("df_mod_e0_verify")
    cells = {}
    for h in (0, 1, 2, 3):
        for seed in (1, 2):
            cells[f"H2__h{h}__s{seed}__profiles"] = {"hypothesis": "H2", "level": h, "seed": seed, "arm": "profiles", "mase": 1.0 - 0.1 * h}
            for k in range(3):
                cells[f"H2__h{h}__s{seed}__random_{k}"] = {"hypothesis": "H2", "level": h, "seed": seed, "arm": f"random_{k}", "mase": 1.0}
    for r in (0, 1):
        for seed in (1, 2):
            cells[f"H3__r{r}__s{seed}__sequence"] = {"hypothesis": "H3", "r": r, "seed": seed, "arm": "sequence", "mase": 0.8 - 0.2 * r}
            cells[f"H3__r{r}__s{seed}__summary"] = {"hypothesis": "H3", "r": r, "seed": seed, "arm": "summary", "mase": 0.8}
    eff = V.effects(cells)
    assert eff["H2"]["e"] == {0: 0.0, 1: pytest.approx(-0.1), 2: pytest.approx(-0.2), 3: pytest.approx(-0.3)}
    assert eff["H2"]["slope"] == pytest.approx(-0.1) and eff["H2"]["replicates"] == 2 and eff["H2"]["random_assignments"] == 3
    assert eff["H3"]["d"] == {0: 0.0, 1: pytest.approx(-0.2)} and eff["H3"]["gamma"] == pytest.approx(-0.2)
    assert eff["H2"]["unit"] == "replicate (independent trajectory); random assignments averaged within the replicate"
    assert eff["H3"]["n_units"] == 2


def test_RP2_the_design_sheet_derives_every_number_and_enumerates_the_cells():
    doc = D.build()
    assert doc["cells_total"] == 4 * 3 * (1 + 3) + 2 * 3 * 3 == 66
    assert all("why" in v for k, v in doc["derivations"].items() if isinstance(v, dict) and k not in ("estimands", "controls", "mase", "architecture", "volume"))
    assert doc["derivations"]["context"]["receptive_field"] == 65 and doc["derivations"]["volume"]["purge"] == E.WINDOW + E.HORIZON
    assert [r["status"] for r in doc["review_table"]] == ["NOT_TESTED"] * 12
    dep = [c for c in doc["cells"] if c.get("depends_on")]
    assert len(dep) == 12 and all(c["depends_on"].endswith("extractor") for c in dep)
    assert doc["design_sha256"] == E.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
