"""RP68: the block runner's feature paths, enumeration, scaling, cadence and closure — through the consumer.

Every rule here is exercised through tools/df_e1_block.py on a SYNTHETIC panel (declared as such) whose
source run is built by the SAME loader path the real preparation uses. Nothing here touches governance
hosts or the real panel; the governed legs are the same functions the phase-1 runner proved on the
disposable stack (tests/test_df_e1_governed_route.py).
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


K = _load("df_e1_block")
B = _load("df_benchmark_contract")
L = _load("df_e1_loader")
E = _load("df_mod_e0")
DAY, W, H = 1440, 60, 60
COLS = ["Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Global_active_power"]
FMT = "%d/%m/%Y %H:%M:%S"
ASSIGNMENT = [0, 1, 0, 2, 2, 2, 0]


def _panel(n_rows: int, seed: int = 3, start="2009-08-01 00:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    daily = np.sin(2*np.pi*t/DAY)
    base = 1.0 + 0.5*daily + 0.3*np.sin(2*np.pi*t/60) + 0.1*rng.normal(size=n_rows)
    df = pd.DataFrame({c: base*(i+1)*0.1 + i + 0.05*rng.normal(size=n_rows) for i, c in enumerate(COLS)})
    df["Global_active_power"] = np.abs(base)
    ts = pd.date_range(start, periods=n_rows, freq="min")
    df.insert(0, "timestamp_label", ts.strftime(FMT))
    return df


def _source(tmp: Path, frame: pd.DataFrame, lo: int) -> Path:
    """A source run root built through the REAL loader path (contract -> resolve -> enumerate -> scaler)."""
    hi = lo + 35*DAY
    root = tmp/"source"
    root.mkdir()
    panel = tmp/"panel.parquet"
    frame.to_parquet(panel)
    digest = K.sha_file(panel)
    c = L.household_contract(W, H, history=True)
    c.splits = {"train": 28/35, "validation": 7/35}
    resolved = L.resolve(frame.iloc[lo:hi].reset_index(drop=True), c)
    enum = L.enumerate_windows(resolved, c)
    scaler = L.fit_scaler(L.build_tensors(resolved, enum, "train", c, None))
    X = resolved["inputs"].astype(np.float64)
    Y = resolved["targets"][:, 0].astype(np.float64)
    Xs = ((X-scaler["mean"])/scaler["sd"]).astype(np.float32)
    j = c.input_columns().index("Global_active_power")
    tr, va = enum["splits"]["train"], enum["splits"]["validation"]
    tr_o = np.asarray(tr["origin_ids"], dtype=np.int64)[np.asarray(tr["target_mask"], dtype=bool)[:, 0]]
    va_o = np.asarray(va["origin_ids"], dtype=np.int64)
    va_m = np.asarray(va["target_mask"], dtype=bool)[:, 0]
    lookup = va_o + H - DAY
    ok = lookup >= 0
    ok[ok] = np.isfinite(Y[lookup[ok]])
    ev = va_o[va_m & ok]
    np.savez(root/"DATA.npz", Xs=Xs, Y=Y, train_origins=tr_o, eval_origins=ev, denominator=np.array([1.0]),
             scaler_mean=scaler["mean"], scaler_sd=scaler["sd"], target_channel=np.array([j]), window=np.array([W]), horizon=np.array([H]))
    design = {"schema": "synthetic_source.v1", "graph": {"assignment": ASSIGNMENT, "core_kind": "tcn_w"}}
    design["design_sha256"] = E.sha_obj(design)
    (root/"DESIGN.json").write_text(json.dumps(design))
    data = {"design_sha256": design["design_sha256"], "panel_sha256": digest, "slice_rows": [lo, hi], "input_columns": c.input_columns(),
            "target_channel": j, "window": W, "horizon": H, "p": 7, "scaler": {"mean": scaler["mean"].tolist(), "sd": scaler["sd"].tolist()},
            "enumerator": {"train": {"admissible": int(tr["admissible"])}, "validation": {"admissible": int(ev.size)}},
            "data_sha256": K.sha_file(root/"DATA.npz")}
    (root/"DATA.json").write_text(json.dumps(data))
    return root


def _contract(source: Path):
    data = json.loads((source/"DATA.json").read_text())
    return B.replace(B.household_ours(), task_id="synthetic.W60_h60.test", dataset_id="synthetic.test.panel",
                     source={"kind": "SYNTHETIC_TEST", "panel_sha256": data["panel_sha256"], "sd_train": data["scaler"]["sd"][6],
                             "evaluation_origins": data["enumerator"]["validation"]["admissible"]})


def _design(block: str, source: Path, *, seeds=(1,), recipe: dict | None = None) -> dict:
    d = K.seal(block, source_run=source, seeds=seeds, contract=_contract(source))
    if recipe:
        d = {k: v for k, v in d.items() if k != "design_sha256"}
        d["recipe"] = {**d["recipe"], **recipe}
        d["design_sha256"] = K.sha_obj(d)
    return d


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("block")
    lo = DAY - H                      # the widest pad any arm needs
    frame = _panel(lo + 35*DAY)
    return {"tmp": tmp, "frame": frame, "lo": lo, "source": _source(tmp, frame, lo)}


# --- enumeration and counts from row identities ---------------------------------------------------------------------

def test_the_block_reproduces_the_sources_populations_and_counts_rows_from_identities(world, tmp_path):
    d = _design("DEV_MATCHED", world["source"])
    rec = K.prepare(d, tmp_path/"r", frame=world["frame"])
    b = rec["binding_to_source"]
    assert b["train_origins_equal_source"] and b["common_evaluation_equals_source"]
    with np.load(tmp_path/"r"/"BLOCK_DATA.npz") as z:
        tr, ev, n = z["train_origins__modular_w60"], z["common_eval"], z["Y"].shape[0]
    c = rec["counts_from_identities"]["modular_w60"]
    assert c["distinct_windows"] == tr.size and c["labels"] == tr.size
    assert c["unique_support_rows"] == int(tr.max()-tr.min()+W)                # one contiguous span
    assert c["train_only_rows_excluding_validation_support"] < c["unique_train_rows_incl_targets"] or c["rows_shared_with_validation_support"] == 0
    assert c["mean_exposures_per_support_row"] == pytest.approx(W*tr.size/c["unique_support_rows"])
    assert rec["scaler"]["sd"] == json.loads((world["source"]/"DATA.json").read_text())["scaler"]["sd"]     # COMMON scaler
    assert rec["feasibility"]["gru_adapted_w60"]["train_admissible"] == tr.size


def test_nonfinite_rows_gaps_and_duplicates_withdraw_exactly_the_windows_that_touch_them():
    n = 5000
    ts = np.arange(n, dtype=np.int64)*60*10**9
    fin = np.ones(n, dtype=bool)
    lab = np.ones(n, dtype=bool)
    ref = K.admissible_origins(ts, fin, lab, W=W, h=H, lo=0, hi=n)
    assert ref[0] == W-1 and ref[-1] == n-H-1
    fin2 = fin.copy(); fin2[2000] = False                                     # a non-finite INPUT row
    got = K.admissible_origins(ts, fin2, lab, W=W, h=H, lo=0, hi=n)
    assert set(ref)-set(got) == set(range(2000, 2000+W))                        # the W windows containing it
    lab2 = lab.copy(); lab2[2000] = False                                      # a non-finite LABEL row
    got = K.admissible_origins(ts, fin, lab2, W=W, h=H, lo=0, hi=n)
    assert set(ref)-set(got) == {2000-H}
    ts2 = ts.copy(); ts2[2000:] += 60*10**9                                   # a one-minute gap before row 2000
    got = K.admissible_origins(ts2, fin, lab, W=W, h=H, lo=0, hi=n)
    assert set(ref)-set(got) == set(range(2000-H, 2000+W-1))                    # every support span crossing the gap
    ts3 = ts.copy(); ts3[2000] = ts3[1999]                                     # a duplicated label: two rows, one minute
    got = K.admissible_origins(ts3, fin, lab, W=W, h=H, lo=0, hi=n)
    assert set(ref)-set(got) == set(range(2000-H, 2000+W)) and not (set(got)-set(ref))   # two off-grid edges (2000, 2001)


def test_a_delivered_panel_whose_rows_differ_from_the_source_is_refused_before_any_score(world, tmp_path):
    d = _design("DEV_MATCHED", world["source"])
    frame2 = world["frame"].copy()
    frame2.loc[world["lo"] + 30*DAY, "Global_active_power"] += 1.0                  # one target row of the validation week changed
    with pytest.raises(K.BlockRefusal, match="target rows are not the source"):
        K.prepare(d, tmp_path/"b", frame=frame2)


def test_future_values_and_labels_do_not_reach_earlier_windows_features_or_labels_and_the_scaler_never_moves(tmp_path):
    """Two panels identical up to t* (inside the validation week) and different after it, each with its own source
    built through the loader: every window, calendar, lag and label at rows before t* is byte-identical, the train
    origins and the COMMON scaler are identical, and only the evaluation rows after t* leave the common set."""
    lo = DAY - H
    frame = _panel(lo + 35*DAY, seed=11)
    t_star = lo + 28*DAY + 3*DAY
    frame2 = frame.copy()
    frame2.loc[t_star:, COLS] += 100.0                                           # every value AFTER t* changes
    frame2.loc[t_star + 500:, "Global_active_power"] = np.nan                     # and later labels vanish
    (tmp_path/"a").mkdir(); (tmp_path/"b").mkdir()
    src_a, src_b = _source(tmp_path/"a", frame, lo), _source(tmp_path/"b", frame2, lo)
    da = K.seal("Q2_CONTEXT", source_run=src_a, seeds=(1,), contract=_contract(src_a))
    db = K.seal("Q2_CONTEXT", source_run=src_b, seeds=(1,), contract=_contract(src_b))
    a, b = K.prepare(da, tmp_path/"ra", frame=frame), K.prepare(db, tmp_path/"rb", frame=frame2)
    assert a["scaler"] == b["scaler"]                                              # train-only, and train did not change
    cut = t_star - da["rows"]["lo"]
    with np.load(tmp_path/"ra"/"BLOCK_DATA.npz") as za, np.load(tmp_path/"rb"/"BLOCK_DATA.npz") as zb:
        for arm in ("daily_lag", "long_window_own_depth", "long_window_crop60"):
            assert np.array_equal(za[f"train_origins__{arm}"], zb[f"train_origins__{arm}"])
        for key in ("Xs", "lag"):                                                  # rows before t* untouched, rows after moved
            assert np.array_equal(za[key][:cut], zb[key][:cut], equal_nan=True) and not np.array_equal(za[key][cut:], zb[key][cut:], equal_nan=True)
        for key in ("calendar", "calendar_randomised"):                            # a calendar reads no value at all
            assert np.array_equal(za[key], zb[key])
        assert np.array_equal(za["Y"][:cut], zb["Y"][:cut]) and np.isnan(zb["Y"][cut+500:]).all()
        ev_a, ev_b = za["common_eval"], zb["common_eval"]
    assert ev_b.size < ev_a.size and np.isin(ev_b, ev_a).all() and (ev_b + H < cut + 500).all()
    assert b["binding_to_source"]["common_evaluation_equals_source"]                  # its own source saw the same gaps


def test_a_context_block_trains_every_arm_on_the_common_train_intersection_and_still_binds_to_the_source(world, tmp_path):
    """A NaN inside the padded rows withdraws long-window (and lag) train origins but not W60 ones: the block trains
    EVERY arm, the baseline included, on the intersection, records the per-arm counts before it, and the baseline's
    own enumeration still reproduces the source (the binding)."""
    d = _design("Q2_CONTEXT", world["source"])
    assert d["train_population"] == "COMMON_INTERSECTION" and "modular_w60" in [a["arm"] for a in d["arms"]]
    frame2 = world["frame"].copy()
    frame2.loc[world["lo"] - 200, "Voltage"] = np.nan                                           # a non-finite row in the pad
    rec = K.prepare(d, tmp_path/"r", frame=frame2)
    with np.load(tmp_path/"r"/"BLOCK_DATA.npz") as z:
        trains = {a["arm"]: z[f"train_origins__{a['arm']}"] for a in d["arms"]}
    sizes = {k: v.size for k, v in trains.items()}
    assert len(set(sizes.values())) == 1                                                        # one train population for the block
    before = {a: rec["feasibility"][a]["train_admissible_before_intersection"] for a in trains}
    assert before["modular_w60"] > before["long_window_own_depth"] and before["modular_w60"] > sizes["modular_w60"]
    assert rec["binding_to_source"]["train_origins_equal_source"] and rec["binding_to_source"]["common_train_subset_of_source"]
    assert rec["binding_to_source"]["common_train_origins"] == sizes["modular_w60"]
    for v in trains.values():
        assert np.array_equal(v, trains["modular_w60"])


def test_the_common_evaluation_mask_is_the_intersection_across_arms(world, tmp_path):
    d = _design("Q2_CONTEXT", world["source"])
    frame2 = world["frame"].copy()
    r = world["lo"] + 28*DAY + 2*DAY + 700                                     # one non-finite input row in the validation week
    frame2.loc[r, "Voltage"] = np.nan
    rec = K.prepare(d, tmp_path/"r", frame=frame2)
    with np.load(tmp_path/"r"/"BLOCK_DATA.npz") as z:
        vals = {a["arm"]: z[f"validation_origins__{a['arm']}"] for a in d["arms"]}
        common = z["common_eval"]
    inter = None
    for v in vals.values():
        inter = v if inter is None else np.intersect1d(inter, v)
    assert np.array_equal(common, inter)
    assert vals["long_window_own_depth"].size < vals["daily_lag"].size            # the long window loses more windows to one row
    assert rec["common_evaluation"]["n"] == common.size


# --- feature paths ------------------------------------------------------------------------------------------------------

def test_calendar_and_its_randomised_control_are_deterministic_prefix_stable_and_distinct(world):
    frame = world["frame"]
    short, long_ = frame.iloc[:3000], frame.iloc[:4000]
    assert np.array_equal(K.calendar_channels(short, FMT), K.calendar_channels(long_, FMT)[:3000])
    ts_s = pd.to_datetime(short["timestamp_label"], format=FMT)
    ts_l = pd.to_datetime(long_["timestamp_label"], format=FMT)
    cs, cl = K.randomised_calendar_channels(ts_s, np.arange(3000)), K.randomised_calendar_channels(ts_l, np.arange(4000))
    assert np.array_equal(cs, cl[:3000]) and np.array_equal(cs, K.randomised_calendar_channels(ts_s, np.arange(3000)))
    assert cs.shape == (3000, 4) and np.allclose(cs[:, 0]**2+cs[:, 1]**2, 1.0)
    honest = K.calendar_channels(short, FMT)
    assert not np.allclose(cs, honest)
    assert not np.array_equal(cs, K.randomised_calendar_channels(ts_s, np.arange(3000), seed=7))   # the seed is part of the construction
    # the classic leak (next row's label) fails the same prefix test the honest path passes
    leaky = lambda ts: K.randomised_calendar_channels(ts.shift(-1).fillna(ts.iloc[-1]), np.arange(ts.size))
    later = ts_s.copy(); later[1500:] += pd.Timedelta(days=1)
    assert not np.array_equal(leaky(later)[:1500], leaky(ts_s)[:1500])


def test_the_daily_lag_reads_a_row_at_or_before_the_origin_and_refuses_beyond_a_day():
    Y = np.arange(5000, dtype=float)
    lag = K.daily_lag_channel(Y, h=H)
    r = 3000
    assert lag[r] == Y[r + H - DAY] and np.isnan(lag[:DAY-H]).all() and np.isfinite(lag[DAY-H:]).all()
    with pytest.raises(K.BlockRefusal, match="AFTER the origin"):
        K.daily_lag_channel(Y, h=1500)


def test_arm_inputs_append_the_declared_channels_with_declared_groups(world, tmp_path):
    d = _design("Q2_CONTEXT", world["source"])
    K.prepare(d, tmp_path/"r", frame=world["frame"])
    data = K.load_data(tmp_path/"r", d)
    X, asg = K.arm_inputs(data, K.arm_spec("daily_lag"), assignment=ASSIGNMENT)
    assert X.shape[1] == 8 and asg == ASSIGNMENT + [ASSIGNMENT[6]]
    assert np.allclose(X[DAY, 7], (data["lag_raw"][DAY]-data["scaler_mean"][6])/data["scaler_sd"][6], atol=1e-6)
    X, asg = K.arm_inputs(data, K.arm_spec("calendar"), assignment=ASSIGNMENT)
    assert X.shape[1] == 11 and asg[7:] == [3, 3, 3, 3]
    with pytest.raises(K.BlockRefusal):
        K.arm_inputs(data, {**K.arm_spec("calendar"), "features": "bogus"}, assignment=ASSIGNMENT)


# --- models: the exact crop null, the 67-sample arm, paired weights ------------------------------------------------------------

def _reach(model, X, rows):
    tf = E._tf()
    xt = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        out = model(xt, training=False)
    g = np.abs(np.asarray(tape.gradient(out, xt))).max(axis=(0, 2))
    grad_reach = int(X.shape[1] - np.flatnonzero(g > 1e-9).min())
    base = np.asarray(model.predict(X, verbose=0))
    pert = {}
    for k in rows:
        Xk = X.copy(); Xk[:, k, :] += 100.0
        pert[k] = float(np.max(np.abs(np.asarray(model.predict(Xk, verbose=0))-base)))
    return grad_reach, pert


def test_crop60_is_an_exact_information_null_with_paired_weights_and_local_support_reaches_67():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 1440, 7)).astype(np.float32)
    short = K.build_model(K.arm_spec("modular_w60"), ASSIGNMENT, 7, 6, seed=1)
    crop = K.build_model(K.arm_spec("long_window_crop60"), ASSIGNMENT, 7, 6, seed=1)
    local67 = K.build_model(K.arm_spec("long_window_local_support_67"), ASSIGNMENT, 7, 6, seed=1)
    assert K.weight_hash(short) == K.weight_hash(crop) and K.n_params(short) == K.n_params(crop) == 8127
    assert np.allclose(short.predict(X[:, -60:, :], verbose=0), crop.predict(X, verbose=0), atol=1e-6)   # same rows, same padding
    n = 1440
    reach, pert = _reach(crop, X, rows=[n-61, n-60, n-1])
    assert reach == 60 and pert[n-61] == 0.0 and pert[n-60] > 0 and pert[n-1] > 0
    reach, pert = _reach(local67, X, rows=[n-68, n-67, n-61])
    assert reach == 67 and pert[n-68] == 0.0 and pert[n-67] > 0 and pert[n-61] > 0
    assert K.n_params(local67) == 8127                                          # clamped depth: the W60 core's size, more support
    deep = K.build_model(K.arm_spec("short_window_deep_core"), ASSIGNMENT, 7, 6, seed=1)
    assert K.n_params(deep) > 8127 and K.n_params(deep) == K.n_params(K.build_model(K.arm_spec("long_window_own_depth"), ASSIGNMENT, 7, 6, 1))


# --- training in observed updates -----------------------------------------------------------------------------------------------

def _tiny_batches(n=640, seed=0):
    rng = np.random.default_rng(seed)
    Xs = rng.normal(size=(n+200, 2)).astype(np.float32)
    Y = Xs[:, 1].astype(np.float64)*2.0
    o = np.arange(W-1, n)
    return K.Batches(Xs, Y, o, W, 1, 1, 64, mean=0.0, sd=1.0, shuffle=True, seed=seed), K.Batches(Xs, Y, o[:128], W, 1, 1, 64, mean=0.0, sd=1.0, shuffle=False, seed=seed)


def test_fit_by_updates_counts_the_optimizers_own_steps_validates_on_cadence_and_censors_at_the_ceiling():
    tf = E._tf()
    tr, va = _tiny_batches()
    inp = tf.keras.Input(shape=(W, 2)); out = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(inp))
    model = tf.keras.Model(inp, out)
    r = K.fit_by_updates(model, tr, va, max_updates=30, validate_every=10, patience=3, lr=0.01, seed=1)
    assert r["updates"] == 30 == r["optimizer_iterations"] and r["updates_are_optimizer_iterations"]
    assert [e["update"] for e in r["events"]] == [10, 20, 30] and r["validation_events"] == 3
    assert r["stop_reason"] == "UPDATE_BUDGET" and r["censoring"]["verdict"] == "CENSORED_BY_BUDGET" and r["triggers"] == {"budget_reached": True, "patience_expired": False}
    assert r["cpu"]["train_update_seconds"] > 0 and r["cpu"]["validation_seconds"] >= 0 and r["cpu"]["restore_seconds"] >= 0
    assert r["restore_verified"] and r["best_val_mae_scaled"] == min(e["val_mae_scaled"] for e in r["events"])
    # a ceiling reached with the best checkpoint EARLY is still censored
    model2 = tf.keras.Model(inp, out)
    r2 = K.fit_by_updates(model2, tr, va, max_updates=30, validate_every=10, patience=5, lr=0.0, seed=1)
    assert r2["best_event"] == 1 and r2["censoring"]["verdict"] == "CENSORED_BY_BUDGET"
    # patience in EVENTS: with lr 0 nothing improves after the first event, so patience 2 stops at event 3
    model3 = tf.keras.Model(inp, out)
    r3 = K.fit_by_updates(model3, tr, va, max_updates=1000, validate_every=10, patience=2, lr=0.0, seed=1)
    assert r3["stop_reason"] == "EARLY_STOPPING" and r3["validation_events"] == 3 and r3["updates"] == 30
    # Musashi's RP73 probe: patience expiring ON the last allowed update is still the ceiling -> both triggers, CENSORED
    model4 = tf.keras.Model(inp, out)
    r4 = K.fit_by_updates(model4, tr, va, max_updates=30, validate_every=10, patience=2, lr=0.0, seed=1)
    assert r4["triggers"] == {"budget_reached": True, "patience_expired": True} and r4["stop_reason"] == "UPDATE_BUDGET+EARLY_STOPPING"
    assert r4["censoring"]["verdict"] == "CENSORED_BY_BUDGET"


# --- one cell, the cost pilot, the closure and the closure table -------------------------------------------------------------------

def test_a_pilot_never_reads_the_dev_validation_and_a_cell_scores_on_the_common_set_readable_by_the_table(world, tmp_path, monkeypatch):
    T = _load("df_closure_table")
    d = _design("DEV_MATCHED", world["source"], recipe={"max_updates": 6, "validate_every_updates": 3, "patience_events": 3})
    root = tmp_path/"root"
    K.prepare(d, root, frame=world["frame"])
    data = K.load_data(root, d)
    (root/"DESIGN.json").write_text(json.dumps(d))
    pilot = K.run_cell(d, data, {**d["pilots"][0], "max_updates": 4, "validate_every_updates": 2}, root/"attempts"/"pilot_modular_w60", pilot=True)
    train_end = int(data["train_end_local"][0])
    assert pilot["pilot"] and pilot["population"]["evaluation_is_common_dev_set"] is False
    with np.load(root/"attempts"/"pilot_modular_w60"/"arrays.npz") as z:
        assert (z["origins"] < train_end).all() and (z["origins"] >= train_end-7*DAY).all()     # inside train, last 7 days
    assert pilot["cost"]["seconds_per_update"] > 0 and pilot["cost"]["peak_rss_bytes"] > 0 and pilot["training"]["updates"] == 4
    receipts = {"units": {}}
    for cell in d["cells"]:
        rec = K.run_cell(d, data, cell, root/"attempts"/cell["cell_id"], pilot=False)
        assert rec["population"]["evaluation_is_common_dev_set"] and rec["training"]["updates"] == 6
        assert rec["reload_max_error"] == 0.0 and rec["training"]["restore_verified"]
        with np.load(root/"attempts"/cell["cell_id"]/"arrays.npz") as z:
            assert np.array_equal(z["origins"], data["common_eval"])
        (root/"TERMINALS").mkdir(exist_ok=True)
        (root/"TERMINALS"/f"{cell['cell_id']}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": [
            {"role": "predictions", "sha256": rec["arrays_sha256"], "bytes": 1}, {"role": "weights", "sha256": rec["weights_file_sha256"], "bytes": 1},
            {"role": "record", "sha256": K.sha_file(root/"attempts"/cell["cell_id"]/"cell.json"), "bytes": 1}]}))
        receipts["units"][cell["cell_id"]] = {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64}
    receipts["units"]["prepare"] = {"campaign_sha256": "c"*64, "terminal_sha256": "q"*64}
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps(receipts))
    # the accepted payloads: every cell's terminal artifacts plus the prepare terminal anchoring BLOCK_DATA (RP82)
    def held(campaign):
        cur = {c["cell_id"]: {"terminal_sha256": "t"*64, "status": "COMPLETED", "config_sha256": d["design_sha256"], "tags": {"arm": c["arm"], "seed": str(c["seed"])},
                              "artifacts": json.loads((root/"TERMINALS"/f"{c['cell_id']}.json").read_text())["artifacts"]} for c in d["cells"]}
        cur["prepare"] = {"terminal_sha256": "q"*64, "status": "COMPLETED", "config_sha256": d["design_sha256"],
                          "artifacts": [{"role": "data", "sha256": K.sha_file(root/"BLOCK_DATA.npz")}]}
        return {"current": cur}
    C = _load("df_mod_e0_close")
    monkeypatch.setattr(C, "warehouse_terminals", lambda url, token, campaign: held(campaign))
    token = tmp_path/"tok"; token.write_text("synthetic")
    report = K.close(SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="synthetic://", skip_replay=True))
    assert report["verified"] and set(report["summary"]) == {"modular_w60", "gru_adapted_w60"} and report["paired"]["values"]
    bl = report["baselines"]
    assert bl["persistence"]["skill_vs_naive"] == 0.0 and bl["daily_seasonal"]["rows"] == bl["train_constant"]["rows"] == data["common_eval"].size
    assert bl["daily_seasonal"]["mae_kw"] != bl["train_constant"]["mae_kw"]
    # the closure table reads the block's layout: DATA from the block's own arrays, roles from the design
    np.savez(root/"DATA.npz", Y=data["Y"], horizon=data["horizon"], target_channel=data["target_channel"], scaler_sd=data["scaler_sd"],
             eval_origins=data["common_eval"])
    (root/"DATA.json").write_text(json.dumps({"input_columns": COLS}))
    def warehouse(campaign):                                            # the accepted payload: what the terminals declared
        return {"current": {c["cell_id"]: {"terminal_sha256": "t"*64, "status": "COMPLETED",
                                            "artifacts": json.loads((root/"TERMINALS"/f"{c['cell_id']}.json").read_text())["artifacts"]}
                            for c in d["cells"]}}
    v = T.verify_run(root, label="synthetic_block", registry=B.registry(), warehouse=held)
    rows = v["rows"]
    assert len(rows) == 2 and all(r["verified"] and r["binding"]["level"] == "TERMINAL_ARTIFACT" and r["custody"]["class"] == "ACCEPTED_ARTIFACT_CHAIN" for r in rows)
    assert v["preparation_custody"]["class"] == "PREPARATION_ACCEPTED_ARTIFACT" and v["denominator"]["equal_to_contract"] is not False    # the synthetic contract carries its own sd_train
    unchecked = T.rows_from_run(root, label="synthetic_block", registry=B.registry())
    assert not any(r["verified"] for r in unchecked)                                     # nothing local verifies itself
    assert {r["unit"] for r in rows} == {c["cell_id"] for c in d["cells"]}
    assert rows[0]["units_not_scored_by_role"][0]["role"] in ("cost_pilot", "preparation")


def test_volume_tiers_grow_backwards_with_the_evaluation_and_scaler_fixed(tmp_path):
    lo = 84*DAY                                                     # 112 days of history before the 7-day validation week
    frame = _panel(lo + 35*DAY, seed=5)
    source = _source(tmp_path, frame, lo)
    d = K.seal("Q3_VOLUME", source_run=source, seeds=(1,), contract=_contract(source))
    assert d["rows"]["lo"] == 0 and d["rows"]["pad_rows"] == 0 and d["rows"]["widest_train_days"] == 112
    rec = K.prepare(d, tmp_path/"r", frame=frame)
    with np.load(tmp_path/"r"/"BLOCK_DATA.npz") as z:
        t56, t112, ev = z["train_origins__volume_56d"], z["train_origins__volume_112d"], z["common_eval"]
        src = np.load(source/"DATA.npz")
    assert np.isin(t56, t112).all() and t112.size > t56.size and t56.min() == t112.min() + 56*DAY
    assert np.array_equal(ev, src["eval_origins"] + (lo - d["rows"]["lo"]))                     # the SAME evaluation rows, byte-identical
    assert rec["scaler"]["mean"] == src["scaler_mean"].tolist()                                   # COMMON scaler, never refitted
    c56, c112 = rec["counts_from_identities"]["volume_56d"], rec["counts_from_identities"]["volume_112d"]
    assert c112["unique_support_rows"] > c56["unique_support_rows"] and c112["labels"] > c56["labels"]
    assert c56["rows_shared_with_validation_support"] == 0


def test_the_design_refuses_an_unknown_block_or_arm_and_binds_to_the_source(world):
    with pytest.raises(K.BlockRefusal):
        K.seal("Q9", source_run=world["source"], contract=_contract(world["source"]))
    d = _design("DEV_MATCHED", world["source"])
    K.validate(d)
    bad = {**d, "arms": [{**d["arms"][0], "window": 61}]}
    bad = {k: v for k, v in bad.items() if k != "design_sha256"}; bad["design_sha256"] = K.sha_obj(bad)
    with pytest.raises(K.BlockRefusal, match="differs from the registry"):
        K.validate(bad)
    stale = {**d, "seeds": [9]}
    with pytest.raises(K.BlockRefusal, match="digest"):
        K.validate(stale)


# --- RP79: architecture x calendar block, tier 2, per-host blocks and merge -------------------------------------------------------------

def test_RP79_the_arch_x_calendar_block_seals_the_15_cells_under_tier2_with_measured_parameter_deltas(world):
    d = _design("ARCH_X_CALENDAR", world["source"], seeds=(1, 2, 3))
    assert len(d["cells"]) == 15 and d["recipe"]["patience_events"] == 10 and d["recipe"]["validate_every_updates"] == 200 and d["recipe"]["max_updates"] == 4000
    assert d["tier"].startswith("TIER2") and "NOT confirmatory" in d["factorial"]["informed_by"]
    assert set(d["factorial"]["primary_factorial"]) == {"modular_w60", "gru_adapted_w60", "calendar", "gru_calendar_w60"}
    cap = d["capacity"]
    assert cap["calendar"]["parameter_delta_vs_base_inputs"] == cap["calendar"]["parameters"] - cap["modular_w60"]["parameters"] > 0
    assert cap["gru_calendar_w60"]["parameter_delta_vs_base_inputs"] == 3*50*4 and cap["gru_adapted_w60"]["parameters"] == 8901
    assert cap["randomised_calendar_control"]["parameters"] == cap["calendar"]["parameters"] and "paired by seed" in cap["initialization"]
    m = K.build_model(K.arm_spec("gru_calendar_w60"), ASSIGNMENT + [3]*4, 11, 6, 1)
    assert K.n_params(m) == cap["gru_calendar_w60"]["parameters"]


def test_RP79_a_host_block_runs_only_its_seeds_and_merge_verifies_artifacts_and_identical_prepared_data(world, tmp_path):
    d = _design("DEV_MATCHED", world["source"], seeds=(1, 2), recipe={"max_updates": 4, "validate_every_updates": 2, "patience_events": 3})
    roots = {}
    for host in ("coord", "worker"):
        root = tmp_path/host
        K.prepare(d, root, frame=world["frame"])
        (root/"DESIGN.json").write_text(json.dumps(d))
        roots[host] = root
    data = K.load_data(roots["worker"], d)
    receipts = {"units": {}}
    for cell in [c for c in d["cells"] if c["seed"] == 2]:                         # the worker's seed block
        rec = K.run_cell(d, data, cell, roots["worker"]/"attempts"/cell["cell_id"], pilot=False)
        (roots["worker"]/"TERMINALS").mkdir(exist_ok=True)
        (roots["worker"]/"TERMINALS"/f"{cell['cell_id']}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": [
            {"role": r, "sha256": K.sha_file(roots["worker"]/"attempts"/cell["cell_id"]/f), "bytes": 1} for r, f in (("predictions", "arrays.npz"), ("weights", "weights.weights.h5"), ("record", "cell.json"))]}))
        receipts["units"][cell["cell_id"]] = {"campaign_sha256": "w"*64, "terminal_sha256": "t"*64}
    (roots["worker"]/"TERMINAL_RECEIPTS.json").write_text(json.dumps(receipts))
    out = K.merge(roots["coord"], [roots["worker"]])
    assert out["problems"] == [] and set(out["units"]) == {"modular_w60_s2", "gru_adapted_w60_s2"}
    assert json.loads((roots["coord"]/"TERMINAL_RECEIPTS.json").read_text())["units"].keys() == out["units"].keys()
    # a tampered artifact is not merged, and different prepared data refuse the whole source
    (roots["worker"]/"attempts"/"modular_w60_s2"/"arrays.npz").write_bytes(b"0")
    (roots["coord"]/"attempts"/"modular_w60_s2").rename(roots["coord"]/"attempts"/"gone")
    out2 = K.merge(roots["coord"], [roots["worker"]])
    assert any("do not match the terminal" in p for p in out2["problems"])
    other = json.loads((roots["worker"]/"BLOCK_DATA.json").read_text()); other["data_sha256"] = "0"*64
    (roots["worker"]/"BLOCK_DATA.json").write_text(json.dumps(other))
    assert any("portability" in p for p in K.merge(roots["coord"], [roots["worker"]])["problems"])


# --- RP82/RP86: the block closure consumes the authoritative verification and replays checkpoints ---------------------------------

def test_RP82_the_block_closure_refuses_substituted_predictions_and_emits_nothing_when_verification_fails(world, tmp_path, monkeypatch):
    """Musashi's RP81 probe on a private copy: predictions replaced by truth and the local record updated; accepted payloads unchanged."""
    d = _design("DEV_MATCHED", world["source"], recipe={"max_updates": 4, "validate_every_updates": 2, "patience_events": 3})
    root = tmp_path/"root"
    K.prepare(d, root, frame=world["frame"])
    (root/"DESIGN.json").write_text(json.dumps(d))
    data = K.load_data(root, d)
    held = {"prepare": {"terminal_sha256": "p"*64, "status": "COMPLETED", "config_sha256": d["design_sha256"],
                        "artifacts": [{"role": "data", "sha256": K.sha_file(root/"BLOCK_DATA.npz")}, {"role": "record", "sha256": K.sha_file(root/"BLOCK_DATA.json")}]}}
    receipts = {"units": {"prepare": {"campaign_sha256": "c"*64, "terminal_sha256": "p"*64}}}
    for cell in d["cells"]:
        rec = K.run_cell(d, data, cell, root/"attempts"/cell["cell_id"], pilot=False)
        arts = [{"role": r, "sha256": K.sha_file(root/"attempts"/cell["cell_id"]/f), "bytes": 1} for r, f in (("predictions", "arrays.npz"), ("weights", "weights.weights.h5"), ("record", "cell.json"))]
        (root/"TERMINALS").mkdir(exist_ok=True)
        (root/"TERMINALS"/f"{cell['cell_id']}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": arts}))
        held[cell["cell_id"]] = {"terminal_sha256": "t"*64, "status": "COMPLETED", "artifacts": arts, "config_sha256": d["design_sha256"], "tags": {"arm": cell["arm"], "seed": str(cell["seed"])}}
        receipts["units"][cell["cell_id"]] = {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64}
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps(receipts))
    C = _load("df_mod_e0_close")
    monkeypatch.setattr(C, "warehouse_terminals", lambda url, token, campaign: {"current": json.loads(json.dumps(held))})
    token = tmp_path/"tok"; token.write_text("synthetic")
    args = SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="synthetic://", skip_replay=True)
    good = K.close(args)
    assert good["verified"] and good["summary"] and good["verification"]["preparation_custody"]["class"] == "PREPARATION_ACCEPTED_ARTIFACT"
    # the probe
    unit = d["cells"][0]["cell_id"]
    with np.load(root/"attempts"/unit/"arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    arr["pred"] = arr["y"].copy(); arr["reload_pred"] = arr["y"].copy()
    np.savez(root/"attempts"/unit/"arrays.npz", **arr)
    rec = json.loads((root/"attempts"/unit/"cell.json").read_text()); rec["arrays_sha256"] = K.sha_file(root/"attempts"/unit/"arrays.npz")
    H = _load("df_e1_huber"); rec["scores"] = H.metrics(arr["pred"], arr["y"], arr["naive"], rec["target_sd"])
    (root/"attempts"/unit/"cell.json").write_text(json.dumps(rec))
    with pytest.raises(K.BlockRefusal, match="closure failed"):
        K.close(args)
    rep = json.loads((root/"REPORT.json").read_text())
    assert not rep["verified"] and rep["summary"] is None and rep["paired"] is None and rep["baselines"] is None
    assert any("CHANGED ARRAYS" in p for p in rep["problems"])
    # no warehouse read -> nothing verified
    with pytest.raises(K.BlockRefusal):
        K.close(SimpleNamespace(root=root, warehouse_token_file=None, warehouse_url=None, skip_replay=True))


def test_RP86_a_fresh_process_checkpoint_replay_is_part_of_closure_for_new_cells(world, tmp_path):
    d = _design("DEV_MATCHED", world["source"], recipe={"max_updates": 4, "validate_every_updates": 2, "patience_events": 3})
    root = tmp_path/"root"
    K.prepare(d, root, frame=world["frame"]); (root/"DESIGN.json").write_text(json.dumps(d))
    data = K.load_data(root, d)
    cell = d["cells"][1]                                                    # the GRU cell
    K.run_cell(d, data, cell, root/"attempts"/cell["cell_id"], pilot=False)
    r = K.replay_cell(root, cell["cell_id"])
    assert r["allclose_1e_6"] and r["max_abs_prediction_difference"] < 1e-5 and abs(r["mae_z_replayed"]-r["mae_z_stored"]) < 1e-8
    (root/"attempts"/cell["cell_id"]/"weights.weights.h5").write_bytes(b"not a checkpoint")
    assert not K.replay_cell(root, cell["cell_id"])["allclose_1e_6"]
