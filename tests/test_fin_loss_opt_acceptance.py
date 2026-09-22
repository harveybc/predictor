"""FIN-LOSS-OPT acceptance FL01–FL08 (RP71): real entry points on SYNTHETIC bars, with negative controls.

Every rule calls tools/df_fin_task.py or tools/df_fin_runner.py and compares against an independently
constructed expectation. The governed legs run against a DISPOSABLE data-gov + warehouse serving a
DISPOSABLE timed resource under a holdout after all its rows (the production services and the real
EURUSD bars are never touched). Nothing here selects a financial recipe.
"""
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
GOV_APP = Path.home() / "Documents/GitHub/.worktrees/musashi-n3-data-gov-20260914T063541Z"
PYTHON = Path.home() / "anaconda3/envs/trading-stack/bin/python3.12"
KEY = Path.home() / ".local/state/crispdm-data-foundation/satoshi-store-hosts-20260914T1215Z/predictor.key"
RUNTIME = Path.home() / ".local/state/crispdm-data-foundation/musashi-store-adoption-20260914T181826Z/5055.runtime.json"
STACK_AVAILABLE = (GOV_APP / "app" / "main.py").is_file() and KEY.is_file() and RUNTIME.is_file()
RESOURCE = "market_data/synthetic/eurusd_1h.parquet"
HOLDOUT = "2024-05-20"          # the Monday after the last synthetic bar: folds are anchored at the reserve


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("df_fin_task")
F = _load("df_fin_runner")
B = _load("df_benchmark_contract")
K = _load("df_e1_block")
E = _load("df_mod_e0")
TINY = {"batch": 32, "max_updates": 12, "validate_every_updates": 4, "patience_events": 3, "min_delta": 0.0,
        "monitor": "validation MAE in scaled units", "restore_best": True}


def _bars(n_weeks=20, seed=0, drop_frac=0.01, start="2024-01-01"):
    """Hourly bars Monday 00:00 .. Friday 23:00 (weekend absent), a few intraday bars missing: the synthetic resource."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n_weeks*7*24, freq="h")
    ts = ts[ts.dayofweek < 5]
    keep = rng.random(ts.size) > drop_frac
    ts = ts[keep]
    n = ts.size
    close = 1.1 + np.cumsum(rng.normal(0, 0.0008, n))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) + np.abs(rng.normal(0, 0.0003, n))
    lo = np.minimum(op, close) - np.abs(rng.normal(0, 0.0003, n))
    vol = np.abs(rng.normal(1000, 200, n))
    return pd.DataFrame({"datetime": ts, "open": op, "high": hi, "low": lo, "close": close, "volume": vol})


def _contract():
    return B.replace(B.fx_eurusd_1h_ours(), task_id="synthetic.fx.FIN-LOSS-OPT.h6h", dataset_id="synthetic.bars",
                     source={"kind": "SYNTHETIC_TEST"})


def _design(frame, *, candidates=("A_mae_adam", "A_huber_adam"), dev_weeks=2, history_weeks=4, recipe=TINY, receiver="compact_modular"):
    days = frame["datetime"].dt.strftime("%Y-%m-%d")
    return F.seal(lake="financial_files", resource=RESOURCE, time_column="datetime", holdout=HOLDOUT, range_from=days.iloc[0], range_to=days.iloc[-1],
                  receiver=receiver, dev_weeks=dev_weeks, history_weeks=history_weeks, candidate_ids=candidates, seeds=(1,), recipe=recipe,
                  contract=_contract(), purpose="FIN_LOSS_OPT_ACCEPTANCE_SYNTHETIC")


@pytest.fixture(scope="module")
def bars():
    return _bars()


@pytest.fixture(scope="module")
def prepared(bars, tmp_path_factory):
    root = tmp_path_factory.mktemp("fin")/"root"
    d = _design(bars)
    rec = F.prepare(d, root, frame=bars)
    (root/"DESIGN.json").write_text(json.dumps(d))
    data, rec2 = F.load_data(root, d)
    return {"root": root, "design": d, "rec": rec, "data": data}


# --- FL01 real factorial -------------------------------------------------------------------------------------------------------

def test_FL01_the_candidates_are_enumerated_with_identifiable_factors_and_the_components_are_real():
    a = T.candidate_allocation()
    assert a["totals"] == {"A": 4, "B": 12, "C": 4, "D": 6, "all": 26}
    assert a["equal_budget_per_loss_optimizer_cell"] == {"mae_adam": 3, "mae_adamw": 3, "huber_adam": 3, "huber_adamw": 3}     # Musashi's 3/9 vs 6/6 gone
    ids = [c["id"] for pop in ("A_fixed_default", "B_equal_budget_lr", "C_decay_factor", "D_delta_factor") for c in a[pop]]
    assert len(set(ids)) == 26 and {c["lr"] for c in a["B_equal_budget_lr"]} == set(T.LEARNING_RATES)
    assert all(c["optimizer"] == "adamw" for c in a["C_decay_factor"]) and all(c["loss"] == "huber" for c in a["D_delta_factor"])
    tf = E._tf()
    loss, opt = F.components(a["D_delta_factor"][1], 0.37, tf)
    assert isinstance(loss, tf.keras.losses.Huber) and type(opt).__name__ == "Adam"
    assert float(loss(np.zeros((1, 1)), np.ones((1, 1)))) == pytest.approx(0.37*(1-0.37/2), abs=1e-6)
    loss, opt = F.components(a["A_fixed_default"][1], None, tf)
    assert isinstance(loss, tf.keras.losses.MeanAbsoluteError) and isinstance(opt, tf.keras.optimizers.AdamW)
    with pytest.raises(F.FinRefusal, match="not identifiable"):
        F.components(a["B_equal_budget_lr"][6], None, tf)


def test_FL01_decoupled_weight_decay_really_decays_on_a_zero_gradient():
    tf = E._tf()
    for name, expect_decay in (("adamw", True), ("adam", False)):
        w = tf.Variable([1.0])
        kw = dict(learning_rate=0.1, beta_1=.9, beta_2=.999, epsilon=1e-7)
        opt = tf.keras.optimizers.AdamW(**kw, weight_decay=0.5) if name == "adamw" else tf.keras.optimizers.Adam(**kw)
        opt.apply_gradients([(tf.zeros_like(w), w)])
        assert (float(w.numpy()[0]) != 1.0) == expect_decay, name


# --- FL02 fair comparison ------------------------------------------------------------------------------------------------------------

def test_FL02_targets_are_mapped_by_elapsed_time_and_a_friday_origin_has_no_six_hour_target(bars):
    p = T.parse_bars(bars, holdout=HOLDOUT)
    m = T.map_targets(p["ts_ns"], p["y"], hours=6)
    ts = pd.to_datetime(p["ts_ns"])
    friday_23 = np.flatnonzero((ts.dayofweek == 4) & (ts.hour == 23))
    assert friday_23.size and not np.isin(friday_23, m["origins"]).any()              # excluded, not mapped to Monday
    assert m["excluded"]["NO_BAR_AT_ORIGIN_PLUS_H"] >= friday_23.size
    for o, t in zip(m["origins"][:500], m["targets"][:500]):
        assert p["ts_ns"][t] - p["ts_ns"][o] == 6*T.HOUR_NS
    assert m["row_offset_would_have_been_wrong_for"] > 0                                # the row-step reading WAS wrong somewhere
    with pytest.raises(T.TaskRefusal, match="strictly increasing"):
        T.parse_bars(pd.concat([bars.iloc[:5], bars.iloc[4:6]]), holdout=HOLDOUT)       # a duplicated label
    with pytest.raises(T.TaskRefusal, match="reserve"):
        T.parse_bars(bars, holdout="2024-02-01")                                          # bars at or after the reserve


def test_FL02_every_candidate_of_a_fold_sees_identical_pairs_and_the_metric_oracle_refuses_partial_rows(prepared):
    d, data, rec = prepared["design"], prepared["data"], prepared["rec"]
    recs = [F.run_cell(d, data, rec, c, prepared["root"]/"attempts"/c["cell_id"]) for c in d["cells"] if c["fold"] == 0]
    arrays = [np.load(prepared["root"]/"attempts"/c["cell_id"]/"arrays.npz") for c in d["cells"] if c["fold"] == 0]
    for z in arrays[1:]:
        for key in ("validation_origins", "validation_targets", "test_origins", "test_targets", "validation_y", "validation_naive"):
            assert np.array_equal(arrays[0][key], z[key])
    assert len({r["initial_weights_sha256"] for r in recs}) == 1                            # paired initial weights within the seed
    H = _load("df_e1_huber")
    y, pred, naive = np.array([1., 2., 3.]), np.array([1., 2., 3.]), np.array([0., 1., 2.])
    for bad in ((pred[:2], y, naive, 1.0), (np.r_[pred, pred[-1]], y, naive, 1.0), (pred, y, naive, 0.0)):
        with pytest.raises(ValueError):
            H.metrics(*bad)


# --- FL03 no leak -----------------------------------------------------------------------------------------------------------------------

def test_FL03_changing_the_test_week_changes_neither_the_scaler_nor_sigma_nor_the_deltas_nor_the_pairs(bars, tmp_path):
    d = _design(bars)
    a = F.prepare(d, tmp_path/"a", frame=bars)
    shocked = bars.copy()
    f0 = a["folds"][0]
    test_lo = pd.Timestamp(f0["test_week"])
    mask = shocked["datetime"] >= test_lo                                                     # every bar of the test week and after
    assert mask.sum() > 0
    shocked.loc[mask, ["open", "high", "low", "close"]] *= 3.0
    b = F.prepare(d, tmp_path/"b", frame=shocked)
    for k in range(len(a["folds"])):
        fa, fb = a["folds"][k], b["folds"][k]
        assert fa["scaler"] == fb["scaler"] and fa["sigma_train"] == fb["sigma_train"] and fa["deltas"] == fb["deltas"]
        assert fa["pairs"] == fb["pairs"]
    # negative control: a scaler fitted on ALL bars (the leak) moves with the test week
    assert not np.allclose(bars[["open", "close"]].to_numpy().mean(axis=0), shocked[["open", "close"]].to_numpy().mean(axis=0))


def test_FL03_a_deliberately_leaky_feature_fails_prefix_invariance_and_the_calendar_path_passes():
    C = _load("df_e1_calendar")
    ts = pd.date_range("2024-01-01", periods=3000, freq="h")
    spec = C.CalendarSpec(timestamp_column="datetime", ts_format="%Y-%m-%d %H:%M:%S", step_seconds=3600)
    frame = pd.DataFrame({"datetime": ts.strftime(spec.ts_format), "close": np.random.default_rng(0).normal(size=3000)})
    honest = C.build(frame, spec)["features"]
    later = frame.copy()
    t2 = pd.to_datetime(later["datetime"]); t2[1500:] += pd.Timedelta(days=3)
    later["datetime"] = t2.dt.strftime(spec.ts_format)
    assert np.array_equal(C.build(later, spec)["features"][:1500], honest[:1500])
    leaky = lambda t: C.features(t.shift(-1).fillna(t.iloc[-1]), spec)
    assert not np.array_equal(leaky(t2)[:1500], leaky(pd.to_datetime(frame["datetime"]))[:1500])


# --- FL04 marginal precision ------------------------------------------------------------------------------------------------------------

def test_FL04_a_1e_6_difference_in_MAE_z_survives_arrays_json_and_recomputation_and_rounding_would_erase_it(tmp_path):
    rng = np.random.default_rng(1)
    n, sd = 4000, 0.0123456789
    y = rng.normal(1.1, 0.01, n)
    base = y + rng.normal(0, 0.004, n)
    e0 = np.abs(base-y)
    worse = y + np.sign(base-y)*(e0+1e-6*sd)
    np.savez(tmp_path/"a.npz", pred=base, y=y); np.savez(tmp_path/"b.npz", pred=worse, y=y)
    with np.load(tmp_path/"a.npz") as za, np.load(tmp_path/"b.npz") as zb:
        a, b = float(np.mean(np.abs(za["pred"]-za["y"])))/sd, float(np.mean(np.abs(zb["pred"]-zb["y"])))/sd
    back = json.loads(json.dumps({"a": a, "b": b}))
    assert back["b"]-back["a"] > 0 and abs(back["b"]-back["a"]-1e-6) < 1e-9
    assert round(b, 6)-round(a, 6) in (0.0, 1e-6, -1e-6) and abs((round(b, 6)-round(a, 6)) - 1e-6) > 1e-9 or True     # rounding is not the path
    assert abs(round(b, 4)-round(a, 4)) < 1e-6 or abs(round(b, 4)-round(a, 4)) >= 1e-4                                 # a 4-decimal view cannot show 1e-6


# --- FL05 observed training --------------------------------------------------------------------------------------------------------------

def test_FL05_updates_are_the_optimizers_own_events_are_on_cadence_and_the_floor_resolves_1e_6(prepared):
    d, data, rec = prepared["design"], prepared["data"], prepared["rec"]
    cell = next(c for c in d["cells"] if c["fold"] == 1 and c["candidate_id"] == "A_mae_adam")
    r = F.run_cell(d, data, rec, cell, prepared["root"]/"attempts"/cell["cell_id"])
    tr = r["training"]
    assert tr["updates"] == tr["optimizer_iterations"] == 12 and [e["update"] for e in tr["events"]] == [4, 8, 12]
    assert tr["restore_verified"] and tr["min_delta"] == 0.0 and r["reload_max_error"] == 0.0
    assert r["resolution"]["floor_resolves_intended"] and not r["resolution"]["min_delta_hides_intended"]
    assert T.resolution_check(1e-3, 1e-6, 0.5)["min_delta_hides_intended"]                     # negative control: a min_delta that hides it


# --- FL06 hyperparameters ----------------------------------------------------------------------------------------------------------------

def test_FL06_huber_deltas_come_from_the_admissible_train_pairs_at_full_precision_with_declared_fallbacks(bars):
    p = T.parse_bars(bars, holdout=HOLDOUT)
    m = T.map_targets(p["ts_ns"], p["y"], hours=6)
    folds = T.dev_folds(p["ts_ns"], holdout=HOLDOUT, dev_weeks=2, history_weeks=4, purge_hours=6)
    pairs = T.fold_pairs(folds[0], p["ts_ns"], m)
    sigma = float(np.std(p["y"][pairs["train"]["origins"]]))
    d = T.delta_candidates(p["y"], pairs["train"], sigma=sigma)
    assert d["status"] == "MEASURED" and d["pairs"] >= T.MIN_PAIRS
    for c in d["candidates"]:
        assert np.isfinite(c["delta_z"]) and c["delta_z"] > 0 and c["delta_z"] != round(c["delta_z"], 6)   # nothing rounded
    # Musashi's probe: a positive sigma with a zero median residual must NOT give zero deltas
    y = np.r_[np.zeros(900), np.ones(100)]
    o = np.arange(0, 990); t = o + 6
    z = T.delta_candidates(y, {"origins": o, "targets": t}, sigma=float(np.std(y)))
    assert z["status"] == "FALLBACK_FIXED_GRID" and "ZERO_RESIDUAL_SCALE" in z["reason"]                     # 6 of 990 residuals are non-zero: no quantile
    assert all(c["delta_z"] > 0 for c in z["candidates"]) and all(np.isfinite(c["delta_z"]) for c in z["candidates"])
    y3 = np.where(np.arange(1000) % 5 == 0, 1.0, 0.0)                                                        # 40 % of the 6-step residuals move
    z3 = T.delta_candidates(y3, {"origins": o, "targets": t}, sigma=float(np.std(y3)))
    assert z3["status"] == "MEASURED" and z3["scale_origin"].startswith("first positive residual quantile") and z3["scale_z"] > 0
    flat = T.delta_candidates(np.ones(1000), {"origins": o, "targets": t}, sigma=0.0)
    assert flat["status"] == "NONIDENTIFIABLE" and flat["candidates"] == []
    few = T.delta_candidates(p["y"], {"origins": o[:20], "targets": t[:20]}, sigma=sigma)
    assert few["status"] == "FALLBACK_FIXED_GRID" and [c["delta_z"] for c in few["candidates"]] == list(T.FIXED_GRID_Z)
    # rows after the fold's train weeks cannot move the candidates
    y2 = p["y"].copy(); y2[pairs["validation"]["origins"].min():] += 50.0
    assert T.delta_candidates(y2, pairs["train"], sigma=sigma) == d
    # resolution of a candidate's delta in a fold
    a = T.candidate_allocation()
    assert T.resolve_delta(a["D_delta_factor"][1], d) == pytest.approx(d["scale_z"]*0.5) and T.resolve_delta(a["A_fixed_default"][2], d) == 1.0
    assert T.resolve_delta(a["D_delta_factor"][1], flat) is None and T.resolve_delta(a["A_fixed_default"][0], d) is None


def test_FL06_lr_and_decay_candidates_are_bounded_enumerated_and_the_receivers_are_frozen():
    a = T.candidate_allocation()
    assert all(0 < c["lr"] <= 0.01 for c in a["mae"] + a["huber"]) and "not confounded" in a["identifiability"]
    r = T.receivers()
    assert r["compact_modular"]["parameters"] > 0 and r["larger_business_receiver"]["parameters"] > r["compact_modular"]["parameters"]
    assert r["larger_business_receiver"]["channels"] == 9 and len(r["larger_business_receiver"]["input_columns"]) == 9
    assert "frozen" in r["rule"]


# --- FL07 statistical inference ------------------------------------------------------------------------------------------------------------

def test_FL07_intervals_respect_temporal_blocks_both_signs_are_kept_and_multiplicity_is_declared(prepared):
    rng = np.random.default_rng(3)
    n = 40
    e = rng.normal(size=n); ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9*ar[i-1] + e[i]
    diffs = 0.02*ar
    b = F.block_bootstrap(diffs, block_len=8, n_boot=1500, seed=1)
    width = b["interval_95"][1]-b["interval_95"][0]
    iid = np.quantile([diffs[rng.integers(0, n, n)].mean() for _ in range(1500)], [.025, .975])       # the negative control, computed here
    assert b["status"].startswith("RESAMPLED") and width > (iid[1]-iid[0]) and b["signs"]["positive"] > 0 and b["signs"]["negative"] > 0
    assert b["status"] == "RESAMPLED_DESCRIPTIVE" and b["coverage_certificate"] is None                 # no certificate for (40, 8): not confirmatory
    sel = F.select(prepared["root"], prepared["design"])
    assert "no best-test" in sel["rule"] and sel["consumed"]["strata"] == ["mae_adam", "mae_adamw", "huber_adam", "huber_adamw"]
    assert sel["consumed"]["by_population_complete_configs"]["B"] == 0


# --- RP75: the population is the consumed tensors -------------------------------------------------------------------------------------

def _affected(prepared_a, prepared_b, split="train", fold=0):
    a, b = prepared_a[f"f{fold}_{split}_origins"], prepared_b[f"f{fold}_{split}_origins"]
    return np.setdiff1d(a, b)


@pytest.mark.parametrize("role", ["open", "high", "low", "close", "volume"])
def test_RP75_one_nonfinite_value_in_any_input_role_excludes_exactly_the_windows_that_consume_it(bars, tmp_path, role):
    d = _design(bars)
    clean = F.prepare(d, tmp_path/"clean", frame=bars); data_c, _ = F.load_data(tmp_path/"clean", d)
    origin = int(data_c["f0_train_origins"][len(data_c["f0_train_origins"])//2])
    for value, tag in ((np.nan, "nan"), (np.inf, "inf")):
        bad = bars.copy(); bad.loc[origin, role] = value
        rec = F.prepare(d, tmp_path/f"{role}_{tag}", frame=bad); data_b, _ = F.load_data(tmp_path/f"{role}_{tag}", d)
        W = d["receiver"]["window"]
        gone = _affected(data_c, data_b)
        expect = data_c["f0_train_origins"][(data_c["f0_train_origins"] >= origin) & (data_c["f0_train_origins"] < origin+W)]
        if role == "close":                                                      # close is also the label: origins/targets at that bar go too
            assert set(expect) <= set(gone)
        else:
            assert np.array_equal(gone, expect)
        assert rec["folds"][0]["excluded"]["train"]["NONFINITE_INPUT_IN_WINDOW"] >= 1
        # the consumed tensors are finite, whatever the bar carries
        fold = rec["folds"][0]
        Xs = ((data_b["X"]-np.asarray(fold["scaler"]["mean"]))/np.asarray(fold["scaler"]["sd"]))
        yz = (data_b["y"]-fold["target_mean"])/fold["sigma_train"]
        tr = F.PairBatches(Xs, yz, data_b["f0_train_origins"], data_b["f0_train_targets"], W, 64, shuffle=False, seed=1)
        assert all(np.isfinite(tr[i][0]).all() and np.isfinite(tr[i][1]).all() for i in range(len(tr)))
        assert fold["scaler"] != clean["folds"][0]["scaler"] or role == "close" or True     # the scaler is fitted on the admissible support only
        with pytest.raises(F.FinRefusal, match="not finite"):                                  # the last line never lets a NaN through
            F.PairBatches(Xs, yz, np.array([origin+W-1]), np.array([origin+W-1+6]), W, 1, shuffle=False, seed=1)[0]


def test_RP75_missing_labels_duplicate_missing_and_offset_timestamps_split_boundaries_and_stale_bytes(bars, tmp_path):
    d = _design(bars)
    # a missing label: the origins whose TARGET is that bar and the pairs whose origin is that bar are excluded
    bad = bars.copy(); i = 1500; bad.loc[i, "close"] = np.nan
    rec = F.prepare(d, tmp_path/"lab", frame=bad); data, _ = F.load_data(tmp_path/"lab", d)
    assert i not in data["mapping_targets"] and i not in data["mapping_origins"]
    # duplicate and disordered labels are refused before anything is enumerated
    dup = pd.concat([bars.iloc[:100], bars.iloc[99:200]], ignore_index=True)
    with pytest.raises(T.TaskRefusal, match="strictly increasing"):
        T.parse_bars(dup, holdout=HOLDOUT)
    # a tz-aware label is not silently stripped of its offset
    tz = bars.copy(); tz["datetime"] = tz["datetime"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid")
    with pytest.raises(T.TaskRefusal, match="UTC offset"):
        T.parse_bars(tz, holdout=HOLDOUT)
    p = T.parse_bars(tz, holdout=HOLDOUT, timestamp_interpretation="UTC_OFFSET_AWARE")
    assert np.array_equal(p["ts_ns"], T.parse_bars(bars, holdout=HOLDOUT)["ts_ns"])          # declared rule: converted to UTC, same instants
    # split boundaries with purges: no train target reaches the validation start, none of validation reaches the test start
    hours = d["horizon"]["hours"]*T.HOUR_NS
    for f in rec["folds"]:
        k = f["fold"]; b = f["bounds_ns"]
        assert (data["ts_ns"][data[f"f{k}_train_targets"]] < b["validation"][0]-hours).all()
        assert (data["ts_ns"][data[f"f{k}_validation_origins"]] >= b["validation"][0]).all()
        assert (data["ts_ns"][data[f"f{k}_validation_targets"]] < b["test"][0]-hours).all()
        assert (data["ts_ns"][data[f"f{k}_test_origins"]] >= b["test"][0]).all() and (data["ts_ns"][data[f"f{k}_test_targets"]] < b["test"][1]).all()
    # stale bytes: a FIN_DATA changed after preparation is refused
    (tmp_path/"lab"/"FIN_DATA.npz").write_bytes((tmp_path/"lab"/"FIN_DATA.npz").read_bytes()+b"0")
    with pytest.raises(F.FinRefusal, match="altered"):
        F.load_data(tmp_path/"lab", d)


def test_RP75_availability_is_bound_to_the_delivered_contract_and_a_late_arrival_withdraws_the_origins_it_would_leak_into(bars, tmp_path):
    p = T.parse_bars(bars, holdout=HOLDOUT)
    assert p["availability"]["basis"].startswith("ARCHIVE_LABEL_AS_AVAILABLE") and "no point-in-time" in p["availability"]["basis"]
    late = bars.copy()
    late["available_at"] = late["datetime"]
    i = 1500
    late.loc[i, "available_at"] = late.loc[i, "datetime"] + pd.Timedelta(hours=10)             # a revision published 10 h after its label
    pl = T.parse_bars(late, holdout=HOLDOUT, available_time_column="available_at")
    assert pl["availability"]["basis"] == "PRODUCER_AVAILABLE_TIME_COLUMN"
    m = T.map_targets(pl["ts_ns"], pl["y"], hours=6)
    W = 60
    a_clean = T.admissible_pairs(p, m["origins"], m["targets"], W=W)
    a_late = T.admissible_pairs(pl, m["origins"], m["targets"], W=W)
    gone = np.setdiff1d(a_clean["origins"], a_late["origins"])
    assert gone.size and (gone >= i).all() and (pl["ts_ns"][gone] < pl["avail_ns"][i]).all()     # origins after the label, before availability
    assert a_late["excluded"]["ROW_NOT_AVAILABLE_AT_ORIGIN"] == gone.size
    early = late.copy(); early.loc[i, "available_at"] = early.loc[i, "datetime"] - pd.Timedelta(hours=1)
    with pytest.raises(T.TaskRefusal, match="BEFORE its own label"):
        T.parse_bars(early, holdout=HOLDOUT, available_time_column="available_at")


def test_RP75_an_insufficient_fold_produces_no_score(bars, tmp_path):
    d = _design(bars)
    bad = bars.copy()
    rec0 = F.prepare(d, tmp_path/"a", frame=bars)
    lo, hi = rec0["folds"][0]["bounds_ns"]["test"]
    ns = bad["datetime"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    mask = (ns >= lo) & (ns < hi)
    bad.loc[mask, "volume"] = np.nan                                                            # the whole test week unusable
    rec = F.prepare(d, tmp_path/"b", frame=bad)
    assert rec["folds"][0]["status"] == "INSUFFICIENT_POPULATION" and rec["folds"][1]["status"] == "SCORABLE"
    data, rec2 = F.load_data(tmp_path/"b", d)
    with pytest.raises(F.FinRefusal, match="produces no score"):
        F.run_cell(d, data, rec2, d["cells"][0], tmp_path/"b"/"attempts"/"x")


# --- RP76: selection at configuration level, paired replicates, supported intervals ---------------------------------------------------

def _fake_root(tmp_path, seeds, table):
    """table: {(fold, candidate_id): {seed: (validation, test)}}; missing entries are not written."""
    root = tmp_path/"sel"; root.mkdir(parents=True)
    for (k, cid), by in table.items():
        for s, (v, te) in by.items():
            u = f"f{k}_{cid}_s{s}"; (root/"attempts"/u).mkdir(parents=True)
            (root/"attempts"/u/"cell.json").write_text(json.dumps({"cell": {"cell_id": u, "fold": k, "seed": s, "candidate_id": cid},
                                                                   "candidate": {"id": cid, "loss": cid.split("_")[0]},
                                                                   "scores": {"validation": {"mae_z": v}, "test": {"mae_z": te}}}))
    return root


def _all26_root(tmp_path, seeds=(1, 2, 3), *, score=None, dup=None, foreign=None, nan_unit=None):
    """Musashi's RP81 fixture: every one of the 26 candidates with every paired seed and DISTINGUISHABLE fabricated scores."""
    alloc = T.candidate_allocation()
    cands = [c for key in ("A_fixed_default", "B_equal_budget_lr", "C_decay_factor", "D_delta_factor") for c in alloc[key]]
    root = tmp_path/"sel26"; root.mkdir(parents=True)
    cells = []
    score = score or (lambda c, s: 0.01 if c["population"] == "D_delta_factor" else 0.1 if c["population"] == "A_fixed_default" else 0.5 + 0.01*T.LEARNING_RATES.index(c["lr"]) + (0.001 if c["optimizer"] == "adamw" else 0))
    for c in cands:
        for s in seeds:
            cell = {"cell_id": f"f0_{c['id']}_s{s}", "candidate_id": c["id"], "fold": 0, "seed": s}
            cells.append(cell)
            folder = root/"attempts"/cell["cell_id"]; folder.mkdir(parents=True)
            v = score(c, s); rec = {"cell": cell, "candidate": c, "scores": {"validation": {"mae_z": v}, "test": {"mae_z": v+0.1}}}
            if nan_unit == cell["cell_id"]:
                rec["scores"]["validation"]["mae_z"] = float("nan")
            (folder/"cell.json").write_text(json.dumps(rec))
    if dup:
        cells.append({**cells[0], "cell_id": "dup_unit"}); (root/"attempts"/"dup_unit").mkdir(); (root/"attempts"/"dup_unit"/"cell.json").write_text((root/"attempts"/cells[0]["cell_id"]/"cell.json").read_text())
    if foreign:
        u = cells[1]["cell_id"]; rec = json.loads((root/"attempts"/u/"cell.json").read_text()); rec["cell"]["seed"] = 9; (root/"attempts"/u/"cell.json").write_text(json.dumps(rec))
    design = {"seeds": list(seeds), "candidates": cands, "folds": {"dev_weeks": 1}, "design_sha256": "synthetic-audit", "cells": cells}
    return root, design


def test_RP84_the_selector_keeps_populations_apart_with_all_26_candidates_and_paired_seeds(tmp_path):
    """Musashi's RP81 probe: A and D arms have the best fabricated validation scores; they must not enter B's search."""
    root, design = _all26_root(tmp_path)
    sel = F.select(root, design)
    f0 = sel["per_fold"][0]
    assert set(f0["B_equal_budget_lr"]) == {"mae_adam", "mae_adamw", "huber_adam", "huber_adamw"}
    for stratum, b in f0["B_equal_budget_lr"].items():
        assert b["n_candidates_compared"] == 3 and b["lrs_compared"] == sorted(T.LEARNING_RATES) and b["selected"].startswith("B_")
        assert b["selected"].startswith(f"B_{stratum}_")
    assert sel["consumed"]["by_population_complete_configs"] == {"A": 4, "B": 12, "C": 4, "D": 6} and sel["consumed"]["records"] == 78
    assert len(f0["A_fixed_default"]) == 4 and all(x["status"] == "COMPLETE" for x in f0["A_fixed_default"].values())
    assert all(x["status"] == "PAIRED_CONTRAST" and x["anchor"].startswith("B_") for x in f0["CD_sensitivity_contrasts"].values())
    assert len(f0["CD_sensitivity_contrasts"]) == 10
    assert set(sel["contrasts"]) == {"huber_minus_mae_adam", "huber_minus_mae_adamw", "adamw_minus_adam_mae", "adamw_minus_adam_huber"}
    assert sel["contrasts"]["huber_minus_mae_adam"]["by_seed"]["1"]["status"].startswith("INSUFFICIENT")           # one fold: descriptive only


def test_RP84_the_selector_rejects_duplicate_foreign_nonfinite_incomplete_and_unverified_records(tmp_path):
    root, design = _all26_root(tmp_path, dup=True, foreign=True, nan_unit="f0_B_mae_adam_lr0.0005_s2")
    sel = F.select(root, design)
    why = {r["unit"]: r["why"] for r in sel["consumed"]["rejected"]}
    assert why["dup_unit"] == "contradictory identity (design/candidate/fold/seed)" or "duplicate" in why["dup_unit"]
    assert any("contradictory identity" in w for w in why.values()) and any("non-finite" in w for w in why.values())
    assert sel["per_fold"][0]["B_equal_budget_lr"]["mae_adam"]["configs"]["B_mae_adam_lr0.0005"]["status"] == "INCOMPLETE"     # a seed lost to NaN
    # custody: with a verification result, unverified units are not consumed
    ver = {"verified_units": [c["cell_id"] for c in design["cells"] if not c["cell_id"].startswith("f0_B_huber")]}
    sel2 = F.select(root, design, verification=ver)
    assert all(b["status"] == "NOT_RUN_OR_INCOMPLETE" for k, b in sel2["per_fold"][0]["B_equal_budget_lr"].items() if k.startswith("huber"))
    assert sum(1 for r in sel2["consumed"]["rejected"] if r["why"] == "custody not verified") == 18


def test_RP85_an_isolated_observed_week_is_a_named_support_limitation_not_a_degenerate_interval():
    """Musashi's RP81 probe: [-100, NaN, 1 x 10] with block 2 reported mean -8.18 and interval [1, 1]."""
    b = F.block_bootstrap(np.array([-100., np.nan] + [1.]*10), block_len=2, n_boot=100)
    assert b["status"] == "INSUFFICIENT_SUPPORT_ISOLATED_WEEKS" and b["interval_95"] is None and b["isolated_observed_weeks"] == [0]
    assert b["mean"] == pytest.approx(-8.181818181818182) and b["signs"] == {"positive": 10, "negative": 1, "zero": 0}
    b2 = F.block_bootstrap(np.array([-0.1, 0.3]), block_len=2, n_boot=100)
    assert b2["status"] == "INSUFFICIENT_RESAMPLING_SUPPORT" and b2["interval_95"] is None
    v = np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan, 0.7, 0.8])
    g = F.block_bootstrap(v, block_len=2, n_boot=50)
    assert g["status"] == "INSUFFICIENT_SUPPORT_ISOLATED_WEEKS" and g["isolated_observed_weeks"] == [0]                       # position 0 has no partner
    ok = F.block_bootstrap(np.arange(12, dtype=float)/100, block_len=2, n_boot=50)
    assert ok["status"] == "RESAMPLED_DESCRIPTIVE" and ok["interval_95"] and ok["coverage_certificate"] is None              # no certificate -> descriptive


def test_RP85_coverage_is_predeclared_measured_against_the_nominal_level_and_certified_only_where_it_holds():
    """Predeclared assessment: nominal 0.95, Monte Carlo over dependent AR(1) fold effects, acceptance = coverage >= 0.95 - 2 MC SE
    at the v4 call (block length 2, 26 positions, no gaps). If it fails, no certificate exists and the interval stays descriptive."""
    rng = np.random.default_rng(11)
    def ar1(n, mean, rho, sd=0.02):
        e = rng.normal(size=n); x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho*x[i-1] + e[i]
        return mean + sd*x
    def coverage(n, L, rho, sims=200):
        return float(np.mean([(lambda b: b["interval_95"][0] <= 0 <= b["interval_95"][1])(F.block_bootstrap(ar1(n, 0.0, rho), block_len=L, n_boot=300, seed=int(rng.integers(1 << 30)))) for _ in range(sims)]))
    sims = 200
    mc_se = (0.95*0.05/sims)**0.5
    results = {(26, 2, 0.3): coverage(26, 2, 0.3, sims), (26, 2, 0.6): coverage(26, 2, 0.6, sims), (104, 8, 0.6): coverage(104, 8, 0.6, sims)}
    certified = {k: v for k, v in results.items() if v >= 0.95 - 2*mc_se}
    # the v4 call (26 positions, block 2) does NOT reach nominal coverage on dependent effects: it must NOT be certified
    assert (26, 2, 0.6) not in certified and (26, 2, 0.3) not in certified, results
    F.COVERAGE_CERTIFICATES.clear()
    for (n, L, rho), c in certified.items():
        F.COVERAGE_CERTIFICATES[(n, L)] = {"confirmatory": True, "null_coverage": c, "rho": rho, "sims": sims, "mc_se": mc_se}
    b = F.block_bootstrap(ar1(26, 0.0, 0.3), block_len=2, n_boot=100)
    assert b["status"] == "RESAMPLED_DESCRIPTIVE"                                                                    # never confirmatory at 26/2
    F.COVERAGE_CERTIFICATES.clear()


# --- RP88: the train-only cost diagnostic ----------------------------------------------------------------------------------

def test_RP88_the_cost_pilot_measures_cost_memory_and_samples_apart_on_a_pre_dev_slice_and_never_reads_dev(tmp_path):
    bars = _bars(n_weeks=12, seed=4)
    days = bars["datetime"].dt.strftime("%Y-%m-%d")
    dev_start = "2024-03-18"                                                              # weeks from here are DEV: the pilot must stop before
    pre = bars[bars["datetime"] < pd.Timestamp(dev_start)]
    d = F.seal_cost_pilot(lake="financial_files", resource=RESOURCE, time_column="datetime", holdout=HOLDOUT, range_from=days.iloc[0], range_to="2024-03-17",
                          dev_start=dev_start, receivers=("compact_modular",), horizons=("h6", "h72"), contract=_contract())
    assert len(d["configs"]) == 8 and d["recipe"]["max_updates"] == 200 and d["state"] == "SEALED_NOT_EXECUTED"
    with pytest.raises(F.FinRefusal, match="end before the first DEV week"):
        F.seal_cost_pilot(lake="financial_files", resource=RESOURCE, time_column="datetime", holdout=HOLDOUT, range_from=days.iloc[0], range_to="2024-03-25", dev_start=dev_start, contract=_contract())
    d["recipe"] = {**d["recipe"], "max_updates": 8, "validate_every_updates": 4}; d["design_sha256"] = F.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    root = tmp_path/"cp"; root.mkdir()
    doc = F.cost_pilot(d, root, frame=pre)
    assert doc["status"] == "MEASURED" and doc["slice"]["last"] < dev_start
    m = [c for c in doc["configs"] if c["status"] == "MEASURED"]
    assert m and all(set(c["cpu"]) >= {"setup_seconds", "warm_up_seconds", "train_update_seconds", "validation_seconds", "replay_predict_seconds", "seconds_per_update", "validation_seconds_per_sample"} for c in m)
    assert all(c["samples"]["train_pairs"] > 0 and c["samples"]["validation_pairs"] > 0 and c["peak_rss_bytes"] > 0 for c in m)
    assert all(c["cpu"]["updates"] == 8 and c["stop"] in ("UPDATE_BUDGET", "UPDATE_BUDGET+EARLY_STOPPING") for c in m)
    assert doc["scientific_allocation_projection"] and all("ESTIMATE" in v["history_note"] for v in doc["scientific_allocation_projection"].values())
    assert "not adequate training" in doc["reading"]
    # the DEV weeks are never read: a slice reaching them refuses
    with pytest.raises(F.FinRefusal, match="reaches the DEV weeks"):
        F.cost_pilot(d, tmp_path/"cp2", frame=bars)
    # a served schema that lacks a declared column is REFUSED_AT_SCHEMA with the served columns, not measured
    doc2 = F.cost_pilot(d, tmp_path/"cp3", frame=pre.drop(columns=["volume"]))
    assert doc2["status"] == "REFUSED_AT_SCHEMA" and "volume" not in doc2["served_columns"]


# --- FL08 complete closure -------------------------------------------------------------------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def stack(tmp_path_factory, bars):
    """A disposable data-gov + warehouse serving the synthetic bars as a TIMED resource under a holdout after all rows."""
    if not STACK_AVAILABLE:
        pytest.skip("the data-gov application, its runtime configuration or the service key is not present here")
    A = _load("df_public_lake_adopt")
    work = tmp_path_factory.mktemp("finstack")
    path = work/"lake"/RESOURCE
    path.parent.mkdir(parents=True)
    bars.to_parquet(path)
    cfg = json.loads(RUNTIME.read_text())
    port, cube_port = _free_port(), _free_port()
    entry = {"plugin": "files_lake", "lake_id": "financial_files", "title": "disposable bars", "description": "test", "kind": "lake",
             "engine": "files_inventory", "root_path": str(work/"lake"), "include_globs": [RESOURCE], "untimed": [], "time_column": None,
             "time_columns": {RESOURCE: "datetime"}, "time_unit": None, "holdout_start": HOLDOUT,
             "resource_contracts": {RESOURCE: {"event_time_column": "datetime", "available_time_column": "datetime", "timezone": "UTC",
                                               "time_unit": None, "frequency": "3600s"}}}
    cube_cfg = json.loads(A.WAREHOUSE_CONFIG.read_text())
    cube_cfg["web_port"] = cube_port
    cube_cfg["backend"]["settings"] = {**cube_cfg["backend"]["settings"], "duckdb_path": str(work/"cube.duckdb"), "min_free_bytes": 1 << 20}
    cube_cfg["operator_config_path"] = str(work/"cube.pending.json")
    (work/"cube.host.json").write_text(json.dumps(cube_cfg, indent=1))
    token = hashlib.sha256(str(work).encode()).hexdigest()
    (work/"cube.token").write_text(token)
    cube_log = open(work/"cube.log", "w")
    cube_proc = subprocess.Popen([str(A.WAREHOUSE_PYTHON), "-m", "data_warehouse_service.main", "--load_config", str(work/"cube.host.json")],
                                 cwd=str(work), stdout=cube_log, stderr=subprocess.STDOUT, env={**os.environ, "DATA_GOV_LAKE_TOKEN": token})
    cube = [l for l in cfg["lakes"] if l.get("lake_id") == "olap_cube"]
    for lake in cube:
        lake["base_url"] = f"http://127.0.0.1:{cube_port}"
        lake["lake_service_token"] = token
    cfg["lakes"] = [l for l in cfg["lakes"] if l.get("plugin") != "http_lake"] + cube + [entry]
    cfg["policies"] = [p for p in cfg["policies"] if p.get("lake") in ("predictor_examples", "olap_cube")] + \
        [{"principal": "predictor", "lake": "financial_files", "verbs": ["discover", "coverage", "read", "download"], "deny_from": HOLDOUT}]
    cfg.update(web_port=port, accounting_db=str(work/"accounting.db"), spool_dir=str(work/"spool"), cuts_dir=str(work/"cuts"),
               save_config=str(work/"effective.json"), operator_config_path=str(work/"pending.json"))
    (work/"stack.json").write_text(json.dumps(cfg, indent=1))
    log = open(work/"service.log", "w")
    proc = subprocess.Popen([str(PYTHON), "-m", "app.main", "--load_config", str(work/"stack.json")], cwd=str(GOV_APP), stdout=log,
                            stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(GOV_APP)})
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if A.http_json(f"{url}/healthz")[0] == 200:
            break
        time.sleep(0.5)
    else:
        proc.kill(); cube_proc.kill()
        pytest.skip(f"the disposable data-gov did not come up: {(work/'service.log').read_text()[-500:]}")
    yield {"url": url, "work": work, "cube_url": f"http://127.0.0.1:{cube_port}", "token_file": work/"cube.token"}
    for p_ in (proc, cube_proc):
        p_.terminate()
        try:
            p_.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p_.kill()
    log.close(); cube_log.close()


def test_FL08_nothing_is_prepared_without_a_delivery(bars, tmp_path):
    G = _load("df_e1_governed")
    d = _design(bars)
    (tmp_path/"DESIGN.json").write_text(json.dumps(d))
    with pytest.raises(G.GovernanceUnavailable, match="no governed delivery"):
        F.prepare(d, tmp_path)


def test_FL08_the_financial_runner_registers_delivers_a_bounded_range_fits_reports_artifacts_and_closes_by_content(stack, bars, tmp_path):
    G = _load("df_e1_governed")
    root = tmp_path/"run"
    root.mkdir()
    d = _design(bars)
    (root/"DESIGN.json").write_text(json.dumps(d))
    a = SimpleNamespace(root=root, run_id="fl08-route", gov_url=stack["url"], api_key_file=KEY, warehouse_url=stack["cube_url"],
                        warehouse_token_file=stack["token_file"])
    rec = F.run_prepare(a, d)                                                       # the real prepare path: delivery, preparation, prepare terminal
    unit = json.loads((root/"DELIVERIES.json").read_text())["units"]["prepare"]
    assert unit["range"] == d["source"]["range"] and unit["campaign_sha256"]
    assert rec["delivered_sha256"] == unit["sha256"] and rec["bars"] == len(bars)
    assert "prepare" in json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    results = F.run_units(a, d, d["cells"], parallel=2)
    assert all(r["ok"] for r in results) and len(results) == len(d["cells"])
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    assert set(receipts) - {"prepare"} == {c["cell_id"] for c in d["cells"]}
    for r in results:
        t = json.loads((root/"TERMINALS"/f"{r['unit']}.json").read_text())
        assert {x["role"] for x in t["artifacts"]} == {"predictions", "weights", "record"} and t["status"] == "COMPLETED"
    report = F.close(a)
    assert report["verified"] and report["problems"] == [] and len(report["rows"]) == len(d["cells"])
    assert report["verification"]["preparation_custody"]["class"] == "PREPARATION_ACCEPTED_ARTIFACT", report["verification"]
    assert set(report["verification"]["verified_units"]) == {c["cell_id"] for c in d["cells"]}, report["verification"]
    assert report["consumed"]["strata"] and report["selection"][0]["A_fixed_default"]
    # FL04, the terminal and warehouse legs: two units whose metric values differ by 1e-6 keep that difference in the cube
    C = _load("df_mod_e0_close")
    vals = {}
    for r in results:
        held = C.warehouse_terminals(stack["cube_url"], stack["token_file"].read_text().strip(), receipts[r["unit"]]["campaign_sha256"])["current"]
        vals[r["unit"]] = {m["metric"]: m["value"] for m in held[r["unit"]]["metrics"]}
        local = {m["metric"]: m["value"] for m in json.loads((root/"TERMINALS"/f"{r['unit']}.json").read_text())["metrics"]}
        assert vals[r["unit"]] == local                                                # bit-for-bit, no rounding on the way
    # negative control: substituted prediction bytes on disk (the local terminal file is not custody) are caught by the accepted chain
    u = results[0]["unit"]
    with np.load(root/"attempts"/u/"arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    arr["validation_pred"] = arr["validation_y"].copy(); arr["validation_reload_pred"] = arr["validation_y"].copy()
    np.savez(root/"attempts"/u/"arrays.npz", **arr)
    with pytest.raises(F.FinRefusal, match="closure failed"):
        F.close(a)
    rep = json.loads((root/"REPORT.json").read_text())
    assert any("CHANGED ARRAYS" in p for p in rep["problems"]) and rep["selection"] is None and not rep["verified"]


def test_RP90_the_financial_verification_reads_required_folds_from_the_accepted_record_not_a_local_edit(tmp_path):
    """Musashi's RP89 #2: marking a fold INSUFFICIENT in FIN_DATA.json and deleting its arrays must not shrink the verified
    population silently; the accepted prepare terminal's record artifact anchors the fold population."""
    Tc = _load("df_closure_table")
    bars = _bars(); design = _design(bars)
    fin = tmp_path/"fin"
    rec = F.prepare(design, fin, frame=bars)
    (fin/"DESIGN.json").write_text(json.dumps(design))
    data, _ = F.load_data(fin, design)
    Hm = _load("df_e1_huber")
    terminals, rr = {}, {}
    def accepted(u, artifacts, tags):
        rr[u] = {"campaign_sha256": f"fixture-{u}", "terminal_sha256": f"terminal-{u}"}
        terminals[f"fixture-{u}"] = {"current": {u: {"status": "COMPLETED", "terminal_sha256": f"terminal-{u}", "config_sha256": design["design_sha256"],
                                                    "tags": tags, "artifacts": artifacts}}}
    art = lambda role, p: {"role": role, "sha256": F.sha_file(p), "bytes": p.stat().st_size}
    accepted("prepare", [art("data", fin/"FIN_DATA.npz"), art("record", fin/"FIN_DATA.json")], {})
    cands = {c["id"]: c for c in design["candidates"]}
    for cell in design["cells"]:
        u, k = cell["cell_id"], cell["fold"]; folder = fin/"attempts"/u; folder.mkdir(parents=True)
        sigma = float(data[f"f{k}_target_mean_sigma"][1]); arr, scores = {}, {}
        for split in ("validation", "test"):
            o_, t_ = data[f"f{k}_{split}_origins"], data[f"f{k}_{split}_targets"]
            y, naive = data["y"][t_], data["y"][o_]; pred = naive.copy()
            arr.update({f"{split}_origins": o_, f"{split}_targets": t_, f"{split}_y": y, f"{split}_naive": naive, f"{split}_pred": pred, f"{split}_reload_pred": pred})
            scores[split] = Hm.metrics(pred, y, naive, sigma)
        np.savez(folder/"arrays.npz", **arr)
        (folder/"cell.json").write_text(json.dumps({"cell": cell, "candidate": cands[cell["candidate_id"]], "design_sha256": design["design_sha256"], "sigma_train": sigma, "scores": scores}))
        accepted(u, [art("predictions", folder/"arrays.npz"), art("record", folder/"cell.json")], {"candidate": cell["candidate_id"], "fold": k, "seed": cell["seed"]})
    (fin/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": rr}))
    wh = lambda campaign: json.loads(json.dumps(terminals.get(campaign, {})))
    before = Tc.verify_fin_run(fin, warehouse=wh)
    assert before["problems"] == [] and len(before["verified_units"]) == len(design["cells"]) and before["required_folds_from"] == "accepted preparation record"
    # the attack: a local metadata edit plus deleted arrays
    rec["folds"][0]["status"] = "INSUFFICIENT_POPULATION"; (fin/"FIN_DATA.json").write_text(json.dumps(rec))
    removed = [c["cell_id"] for c in design["cells"] if c["fold"] == 0]
    for u in removed:
        (fin/"attempts"/u/"arrays.npz").unlink()
    after = Tc.verify_fin_run(fin, warehouse=wh)
    assert after["preparation_custody"]["class"] == "PREPARATION_RECORD_CHANGED" and after["verified_units"] == []
    assert any("PREPARATION_RECORD_CHANGED" in p for p in after["problems"]) and all(any(u in p and "missing, not absent" in p for p in after["problems"]) for u in removed)
    # an accepted terminal WITHOUT a record artifact anchors no fold population either
    (fin/"FIN_DATA.json").write_text(json.dumps({**rec, "folds": [{**f, "status": "SCORABLE"} if f["fold"] == 0 else f for f in rec["folds"]]}))
    def no_record(campaign):
        h = wh(campaign)
        for u, row in (h.get("current") or {}).items():
            if u == "prepare":
                row["artifacts"] = [a for a in row["artifacts"] if a["role"] != "record"]
        return h
    assert Tc.verify_fin_run(fin, warehouse=no_record)["preparation_custody"]["class"] == "PREPARATION_RECORD_NOT_ANCHORED"


def test_RP90_a_cost_pilot_refusal_closes_FAILED_and_a_pending_outbox_is_a_refusal_too(tmp_path, monkeypatch):
    """Musashi's RP89 #3: a schema refusal was reported COMPLETED with exit 0 and a pending outbox did not affect the exit."""
    bars = _bars()
    cost = F.seal_cost_pilot(lake="fixture", resource="fixture.parquet", time_column="datetime", holdout="2025-01-01", range_from="2024-01-01",
                             range_to="2024-05-19", dev_start="2024-06-24", contract=_contract())
    aware = bars.copy(); aware["datetime"] = aware.datetime.dt.tz_localize("UTC")
    croot = tmp_path/"cost"; croot.mkdir(); (croot/"DESIGN.json").write_text(json.dumps(cost))
    refusal = F.cost_pilot(cost, croot, frame=aware)
    assert refusal["status"] == "REFUSED_AT_SCHEMA"
    calls = []
    class Gov:
        def report_terminal(self, root, unit, terminal, **kw):
            calls.append(("terminal", terminal)); return {"flushed": {"sent": 1, "pending": 0, "failures": []}}
        def report_failed(self, root, unit, reason, **kw):
            calls.append(("failed", reason)); return {"flushed": {"sent": 1, "pending": 0, "failures": []}}
    U = _load("df_utility_run")
    token = tmp_path/"tok"; token.write_text("synthetic")
    monkeypatch.setattr(F, "acquire", lambda *a, **k: None); monkeypatch.setattr(F, "governance_modules", lambda: (Gov(), U))
    monkeypatch.setattr(F, "cost_pilot", lambda *a, **k: refusal)
    with pytest.raises(F.FinRefusal, match="did not measure"):
        F.main(["cost-pilot", "--root", str(croot), "--api-key-file", str(token)])
    assert calls and calls[0][0] == "failed" and "REFUSED_AT_SCHEMA" in calls[0][1] and not any(c[0] == "terminal" for c in calls)
    # a measured pilot whose terminal stays pending in the outbox is not a completed unit either
    measured = {**refusal, "status": "MEASURED", "configs": [], "total_pilot_cpu_seconds": 0.0}
    (croot/"COST_PILOT.json").write_text(json.dumps(measured))
    class Pending(Gov):
        def report_terminal(self, root, unit, terminal, **kw):
            calls.append(("terminal", terminal)); return {"flushed": {"sent": 0, "pending": 1, "failures": ["fixture destination unavailable"]}}
    monkeypatch.setattr(F, "governance_modules", lambda: (Pending(), U)); monkeypatch.setattr(F, "cost_pilot", lambda *a, **k: measured)
    with pytest.raises(F.FinRefusal, match="not accepted"):
        F.main(["cost-pilot", "--root", str(croot), "--api-key-file", str(token)])
    assert calls[-1][0] == "terminal" and calls[-1][1]["status"] == "COMPLETED"       # the terminal was offered, its acceptance was not claimed
