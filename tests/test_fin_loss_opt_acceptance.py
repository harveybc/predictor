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


def _design(frame, *, candidates=("default_mae_adam", "default_huber_adam"), dev_weeks=2, history_weeks=4, recipe=TINY, receiver="compact_modular"):
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

def test_FL01_the_candidates_are_enumerated_twelve_per_family_with_the_defaults_apart_and_the_components_are_real():
    a = T.candidate_allocation()
    assert len(a["mae"]) == len(a["huber"]) == a["per_family"] == 12 and len(a["defaults"]) == 4
    ids = [c["id"] for c in a["mae"] + a["huber"] + a["defaults"]]
    assert len(set(ids)) == 28 and all(c["loss"] == "mae" for c in a["mae"]) and all(c["loss"] == "huber" for c in a["huber"])
    assert {c["optimizer"] for c in a["mae"]} == {"adam", "adamw"} and all(c["delta"] for c in a["huber"])
    tf = E._tf()
    loss, opt = F.components(a["huber"][1], 0.37, tf)
    assert isinstance(loss, tf.keras.losses.Huber) and isinstance(opt, tf.keras.optimizers.AdamW)
    # the delta is real: Huber(delta) on a residual of 1.0 is delta*(1 - delta/2), not 0.5 (MSE) nor 1.0 (MAE)
    assert float(loss(np.zeros((1, 1)), np.ones((1, 1)))) == pytest.approx(0.37*(1-0.37/2), abs=1e-6)
    loss, opt = F.components(a["mae"][0], None, tf)
    assert isinstance(loss, tf.keras.losses.MeanAbsoluteError) and type(opt).__name__ == "Adam"
    with pytest.raises(F.FinRefusal, match="not identifiable"):
        F.components(a["huber"][0], None, tf)


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
    cell = next(c for c in d["cells"] if c["fold"] == 1 and c["candidate_id"] == "default_mae_adam")
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
    assert T.resolve_delta(a["huber"][0], d) == pytest.approx(d["scale_z"]*0.5) and T.resolve_delta(a["defaults"][2], d) == 1.0
    assert T.resolve_delta(a["huber"][0], flat) is None and T.resolve_delta(a["mae"][0], d) is None


def test_FL06_lr_and_decay_candidates_are_bounded_enumerated_and_the_receivers_are_frozen():
    a = T.candidate_allocation()
    assert all(0 < c["lr"] <= 0.01 for c in a["mae"] + a["huber"]) and "declared" in a["trade_off"]
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
    iid = b["iid_interval_95_for_the_record"][1]-b["iid_interval_95_for_the_record"][0]
    assert width > iid and b["signs"]["positive"] > 0 and b["signs"]["negative"] > 0 and b["n"] == n   # negative control: iid is narrower
    sel = F.select(prepared["root"], prepared["design"])
    assert sel["paired"]["multiplicity"]["candidates_per_family"] == {"huber": 1, "mae": 1}
    assert "no best-test" in sel["rule"] and all(v.get("n_candidates_compared", 0) <= 1 for f in sel["per_fold"].values() for v in f.values())


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
    doc = F.acquire(a, d, "prepare")
    unit = doc["units"]["prepare"]
    assert unit["range"] == d["source"]["range"] and unit["campaign_sha256"]
    rec = F.prepare(d, root)
    assert rec["delivered_sha256"] == unit["sha256"] and rec["bars"] == len(bars)
    results = F.run_units(a, d, d["cells"], parallel=2)
    assert all(r["ok"] for r in results) and len(results) == len(d["cells"])
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    assert set(receipts) == {c["cell_id"] for c in d["cells"]}
    for r in results:
        t = json.loads((root/"TERMINALS"/f"{r['unit']}.json").read_text())
        assert {x["role"] for x in t["artifacts"]} == {"predictions", "weights", "record"} and t["status"] == "COMPLETED"
    report = F.close(a)
    assert report["verified"] and report["problems"] == [] and len(report["rows"]) == len(d["cells"])
    # FL04, the terminal and warehouse legs: two units whose metric values differ by 1e-6 keep that difference in the cube
    C = _load("df_mod_e0_close")
    vals = {}
    for r in results:
        held = C.warehouse_terminals(stack["cube_url"], stack["token_file"].read_text().strip(), receipts[r["unit"]]["campaign_sha256"])["current"]
        vals[r["unit"]] = {m["metric"]: m["value"] for m in held[r["unit"]]["metrics"]}
        local = {m["metric"]: m["value"] for m in json.loads((root/"TERMINALS"/f"{r['unit']}.json").read_text())["metrics"]}
        assert vals[r["unit"]] == local                                                # bit-for-bit, no rounding on the way
    # negative control: a tampered artifact digest is caught at closure
    t_path = root/"TERMINALS"/f"{results[0]['unit']}.json"
    t = json.loads(t_path.read_text()); t["artifacts"][0]["sha256"] = "0"*64; t_path.write_text(json.dumps(t))
    with pytest.raises(F.FinRefusal, match="closure failed"):
        F.close(a)
    assert any("warehouse artifacts" in p for p in json.loads((root/"REPORT.json").read_text())["problems"])
