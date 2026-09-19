"""RP23: data/loader controls of the E1 task pipeline against the real functions (df_e1_tasks): usable
windows after masks, purge and horizon; structural zeros vs measured zeros; the future/prefix control
(a change after the origin never changes a window's inputs); train-only scale; gaps; DST as a calendar
factor. Synthetic frames only; the governed panels are not read here."""
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


T = _load("df_e1_tasks")


def test_usable_windows_respect_masks_purge_and_horizon():
    n, W, h = 100, 5, 2
    ok = np.ones(n, dtype=bool)
    ok[20] = False                                              # one bad input row
    valid_t = np.ones(n, dtype=bool)
    valid_t[60] = False                                         # one bad target row
    splits = {"train": [0, 50], "validation": [50, 80], "test": [80, 100]}
    u = T.usable_windows(ok, valid_t, n, W, h, splits)
    # train origins 4..49: windows covering row 20 (origins 20..24) are lost -> 46 - 5 = 41; targets at +2 all valid in train
    assert u["train"]["origins"] == 46 and u["train"]["usable"] == 41
    # validation origins 50..79: the target row 60 kills origin 58 only
    assert u["validation"]["origins"] == 30 and u["validation"]["usable"] == 29
    # test origins 80..97 (n - h): every one usable
    assert u["test"]["origins"] == 18 and u["test"]["usable"] == 18
    # purge is derived: origins of a split end before its boundary minus purge in the family contract (see family_contract)


def test_structural_zeros_are_not_measurements_and_are_masked_from_targets():
    n = 200
    ts = pd.date_range("2011-01-01 00:15", periods=n, freq="15min")
    x = np.ones((n, 3))
    x[:80, 1] = 0.0                                             # client 1 does not exist for 80 rows, then measures 1.0
    x[120:125, 2] = 0.0                                         # client 2 measured zeros (after existing)
    df = pd.DataFrame({"timestamp_label": ts.strftime("%Y-%m-%d %H:%M:%S"), "c0": x[:, 0], "c1": x[:, 1], "c2": x[:, 2]})
    fam = dict(T.FAMILIES["uci_321"])
    cs = T.column_structure(df, fam, pd.Series(ts))
    assert cs["columns"]["c1"]["structural_zero_rows"] == 80 and cs["columns"]["c1"]["measured_zero_rows"] == 0 and cs["columns"]["c1"]["first_non_zero_row"] == 80
    assert cs["columns"]["c2"]["structural_zero_rows"] == 0 and cs["columns"]["c2"]["measured_zero_rows"] == 5
    fam235 = dict(T.FAMILIES["uci_235"])
    cs2 = T.column_structure(df, fam235, pd.Series(ts))
    assert cs2["columns"]["c1"]["structural_zero_rows"] == 0 and cs2["columns"]["c1"]["measured_zero_rows"] == 80   # no structural rule for the household


def test_future_and_prefix_control_a_window_never_consumes_rows_after_its_origin():
    n, W = 50, 6
    x = np.arange(n, dtype=float)[:, None] * np.ones((1, 2))
    rows = np.arange(W - 1, n - 1)
    E = _load("df_mod_e0")
    X = E.make_windows(x, rows, W)
    x2 = x.copy()
    x2[30:] += 1000.0
    X2 = E.make_windows(x2, rows, W)
    unaffected = rows < 30
    assert np.array_equal(X[unaffected], X2[unaffected]) and not np.array_equal(X[~unaffected], X2[~unaffected])
    assert np.array_equal(X[0, -1], x[W - 1]) and X.shape == (rows.size, W, 2)          # the last input of a window is its origin row


def test_time_structure_flags_gaps_duplicates_and_non_nominal_days():
    ts = pd.Series(pd.date_range("2012-03-24 00:00", periods=96 * 3, freq="15min"))
    t2 = pd.concat([ts.iloc[:100], ts.iloc[104:]]).reset_index(drop=True)              # a one-hour gap
    s = T.time_structure(t2, 900)
    assert s["irregular_steps"] == 1 and s["monotone"] and s["duplicates"] == 0
    assert s["days_with_row_count_not_nominal_total"] == 1 and list(s["days_with_row_count_not_nominal"].values()) == [92]
    dup = pd.concat([ts, ts.iloc[5:6]]).sort_values().reset_index(drop=True)
    assert T.time_structure(dup, 900)["duplicates"] == 1


def test_train_only_scale_and_split_purge_in_the_family_contract(monkeypatch, tmp_path):
    """A synthetic family through the real family_contract: purge = W + h removes origins from the train and
    validation ends; the scale is not applied to the arrays; contexts are reported in physical units."""
    n = 4000
    ts = pd.date_range("2011-01-01 00:15", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.stack([np.sin(2 * np.pi * t / 96) + 0.1 * rng.normal(size=n) + 3 for _ in range(6)], axis=1)
    x[:500, 3] = 0.0
    df = pd.DataFrame({"timestamp_label": ts.strftime("%Y-%m-%d %H:%M:%S"), **{f"MT_{i}": x[:, i] for i in range(6)}})
    panel = tmp_path / "uci_321_electricityloaddiagrams20112014"
    panel.mkdir()
    df.to_parquet(panel / "panel.parquet")
    (panel / "PARSE_RECEIPT.json").write_text('{"rows": %d}' % n)
    import hashlib
    fam = dict(T.FAMILIES["uci_321"], panel_sha256=hashlib.sha256((panel / "panel.parquet").read_bytes()).hexdigest())
    monkeypatch.setattr(T, "PANELS", tmp_path)
    monkeypatch.setitem(T.FAMILIES, "uci_321", fam)
    doc = T.family_contract("uci_321", [96], [1, 4])
    w = doc["windows"]["W96_h4"]
    assert w["purge"] == 100 and w["context_physical_hours"] == pytest.approx(95 * 0.25)
    tr = doc["splits_by_time"]["train"]
    assert w["origins"]["train"] == (tr[1] - 100) - 95                      # origins from W-1 to the train end minus the purge
    assert w["usable_windows_per_target_column_quantiles"]["train"][0] < w["usable_windows_per_target_column_quantiles"]["train"][4]   # the structural-zero client has fewer
    assert doc["train_only_periodicities"]["bands"]["daily_24h_share"] > 0.5                # the planted 24 h cycle is measured on train
    assert doc["eligibility"]["task"].startswith("TASK_CONTRACT_DECLARED") and "independent replicates" in doc["independence_note"]
