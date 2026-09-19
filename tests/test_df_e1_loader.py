"""RP27: the E1 loader end to end (contract -> enumerator -> tensors -> scaler -> eval set), including the
dictum's fixtures: a missing target withdraws one label, not 61 origins; an unforeseen numeric column is
refused (or ignored only when declared); a one-minute jump withdraws the windows whose support crosses
it; duplicates and disorder withdraw; timestamps are never coerced; the target's future never enters an
input; scaling is fitted on train windows only and validation/test changes cannot alter it; electricity
activation is causal; every arm scores one common evaluation set."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


L = _load("df_e1_loader")
N = 5000


def _household(n=N, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2007-01-01 00:00", periods=n, freq="min")
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = pd.DataFrame({"timestamp_label": ts.strftime("%d/%m/%Y %H:%M:%S"), **{c: rng.normal(size=n) + i for i, c in enumerate(cols)}})
    return df, ts


def _enum(df, c):
    return L.enumerate_windows(L.resolve(df, c), c)


def test_RP27_household_contract_binds_roles_and_counts_from_the_enumerator():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    r = L.resolve(df, c)
    assert r["input_columns"] == c.features + ["Global_active_power"] and r["targets"].shape == (N, 1)
    e = _enum(df, c)
    tr = e["splits"]["train"]
    assert tr["origins"] == tr["admissible"] == 3500 - 61 - 59                 # origins W-1 .. train_end - purge
    assert tr["targets_valid_per_column"]["Global_active_power"] == tr["admissible"]
    assert e["policy"]["gap"] == "withdraw" and e["grid_rows_not_ok"] == 0


def test_RP27_a_missing_target_withdraws_one_label_not_61_origins():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    base = _enum(df, c)["splits"]["train"]
    bad = df.copy()
    bad.loc[1000, "Global_active_power"] = np.nan                                # one missing TARGET value at row 1000
    e = _enum(bad, c)["splits"]["train"]
    # the target is also an input (history): windows whose 60 inputs include row 1000 lose their inputs (60 origins), and one label is lost
    assert base["admissible"] - e["admissible"] == 60 and e["withdrawn_non_finite_inputs"] == 60
    c2 = L.household_contract(window=60, horizon=1, history=False)             # without the target's history: only the label is lost
    b2, e2 = _enum(df, c2)["splits"]["train"], _enum(bad, c2)["splits"]["train"]
    assert b2["admissible"] == e2["admissible"] and b2["targets_valid_per_column"]["Global_active_power"] - e2["targets_valid_per_column"]["Global_active_power"] == 1


def test_RP27_an_unforeseen_numeric_column_is_refused_and_an_ignored_one_changes_nothing():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    extra = df.copy()
    extra["unused_numeric_metadata"] = 1.0
    extra.loc[800, "unused_numeric_metadata"] = np.nan
    with pytest.raises(L.ContractRefusal, match="not foreseen"):
        L.resolve(extra, c)
    c2 = L.TaskContract(**{**c.__dict__, "ignore": ["unused_numeric_metadata"]})
    assert _enum(extra, c2)["splits"]["train"]["admissible"] == _enum(df, c)["splits"]["train"]["admissible"]
    c3 = L.TaskContract(**{**c.__dict__, "metadata": ["unused_numeric_metadata"]})
    assert _enum(extra, c3)["splits"]["train"]["admissible"] == _enum(df, c)["splits"]["train"]["admissible"]
    with pytest.raises(L.ContractRefusal, match="absent"):
        L.resolve(df.drop(columns=["Voltage"]), c)


def test_RP27_a_one_minute_jump_withdraws_every_window_whose_support_crosses_it():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    base = _enum(df, c)["splits"]["train"]
    gap_time = ts.to_series().reset_index(drop=True).copy()
    gap_time.iloc[1000:] += pd.Timedelta(minutes=1)                             # values complete, the time axis jumps
    g = df.copy()
    g["timestamp_label"] = gap_time.dt.strftime("%d/%m/%Y %H:%M:%S")
    e = _enum(g, c)
    assert e["grid_rows_not_ok"] == 1
    # support t-W+1 .. t+h crossing row 1000: origins 940 .. 999 -> 60 windows withdrawn
    assert base["admissible"] - e["splits"]["train"]["admissible"] == 60 and e["splits"]["train"]["withdrawn_grid"] == 60
    # duplicates and disorder withdraw too; time is never compressed
    dup = pd.concat([df.iloc[:2000], df.iloc[1999:2000], df.iloc[2000:]]).reset_index(drop=True)
    ed = _enum(dup, c)
    assert ed["grid_rows_not_ok"] == 1 and ed["splits"]["train"]["withdrawn_grid"] > 0
    dis = pd.concat([df.iloc[:1500], df.iloc[1501:1502], df.iloc[1500:1501], df.iloc[1502:]]).reset_index(drop=True)
    assert _enum(dis, c)["grid_rows_not_ok"] >= 2
    # a declared ambiguous label (DST) withdraws the windows whose support contains it
    c_amb = L.TaskContract(**{**c.__dict__, "ambiguous_labels": [ts[3000].strftime("%Y-%m-%d %H:%M:%S")]})
    ea = _enum(df, c_amb)
    assert ea["ambiguous_rows"] == 1 and ea["splits"]["validation"]["withdrawn_grid"] + ea["splits"]["train"]["withdrawn_grid"] == 61


def test_RP27_timestamps_are_not_coerced_and_a_numeric_timestamp_is_refused():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    num = df.copy()
    num["timestamp_label"] = np.arange(N, dtype=float)
    with pytest.raises(L.ContractRefusal, match="timestamp"):
        L.resolve(num, c)
    r = L.resolve(df, c)
    assert pd.api.types.is_datetime64_any_dtype(r["timestamps"])


def test_RP27_the_tensor_never_contains_the_targets_future_and_the_prefix_control_holds():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=3)
    r = L.resolve(df, c)
    e = L.enumerate_windows(r, c)
    T = L.build_tensors(r, e, "train", c, None)
    t0 = T["origins"][0]
    assert np.array_equal(T["X"][0][:, -1], r["targets"][t0 - 59:t0 + 1, 0]) and T["target_ids"][0] == t0 + 3
    assert np.array_equal(T["y"][0], r["targets"][t0 + 3])
    # change the future (rows > origin) of the target and the inputs: the window's tensor is unchanged, the label changes
    df2 = df.copy()
    df2.loc[t0 + 1:, ["Global_active_power", "Voltage"]] += 1000.0
    T2 = L.build_tensors(L.resolve(df2, c), e, "train", c, None)
    assert np.array_equal(T2["X"][0], T["X"][0]) and not np.array_equal(T2["y"][0], T["y"][0])


def test_RP27_the_scaler_is_fitted_on_train_windows_only_and_later_splits_cannot_alter_it():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    r = L.resolve(df, c)
    e = L.enumerate_windows(r, c)
    sc = L.fit_scaler(L.build_tensors(r, e, "train", c, None))
    assert sc["fitted_on"] == "train windows only"
    df2 = df.copy()
    df2.loc[4000:, c.features] *= 100.0                                       # test rows altered
    r2 = L.resolve(df2, c)
    e2 = L.enumerate_windows(r2, c)
    sc2 = L.fit_scaler(L.build_tensors(r2, e2, "train", c, None))
    assert np.allclose(sc["mean"], sc2["mean"]) and np.allclose(sc["sd"], sc2["sd"])
    assert e["splits"]["train"]["origin_ids"] == e2["splits"]["train"]["origin_ids"] and e["splits"]["validation"]["origin_ids"] == e2["splits"]["validation"]["origin_ids"]
    Ttr = L.build_tensors(r2, e2, "train", c, sc2)
    assert abs(Ttr["X"].reshape(-1, Ttr["X"].shape[2]).mean()) < 1e-6           # scaled train inputs centred


def test_RP27_electricity_activation_is_causal_and_never_active_clients_are_named():
    n = 600
    ts = pd.date_range("2011-01-01 00:15", periods=n, freq="15min")
    x = np.ones((n, 3))
    x[:200, 1] = 0.0                                                             # client 1 appears at row 200
    x[:, 2] = 0.0                                                                # client 2 never active
    x[250:260, 1] = 0.0                                                          # measured zeros after activation
    df = pd.DataFrame({"timestamp_label": ts.strftime("%Y-%m-%d %H:%M:%S"), "MT_0": x[:, 0], "MT_1": x[:, 1], "MT_2": x[:, 2]})
    c = L.electricity_contract(["MT_0", "MT_1", "MT_2"], window=8, horizon=1)
    r = L.resolve(df, c)
    e = L.enumerate_windows(r, c)
    assert e["never_active_targets"] == ["MT_2"]
    tr = e["splits"]["train"]
    # activity is judged at the origin: an origin before row 200 has no valid label for MT_1 even if its target row is active
    ids = np.asarray(tr["origin_ids"])
    mask = np.asarray(tr["target_mask"])
    j = c.targets.index("MT_1")
    assert not mask[ids < 200, j].any() and mask[ids >= 200, j].all()             # measured zeros at 250..259 stay valid labels
    assert not mask[:, 2].any() and mask[:, 0].all()
    assert tr["targets_valid_per_column"]["MT_2"] == 0


def test_RP27_every_arm_scores_the_same_evaluation_set():
    df, ts = _household()
    c = L.household_contract(window=60, horizon=1)
    e = _enum(df, c)
    ev = L.eval_set(e, "validation")
    assert ev["origins"] == e["splits"]["validation"]["origin_ids"] and len(ev["mask"]) == len(ev["origins"]) and "ALL arms" in ev["rule"]
    assert set(ev["target_ids"]) == {o + 1 for o in ev["origins"]}
