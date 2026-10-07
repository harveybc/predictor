"""Weekly wrapper for the phase-4 subset comparison (FS4-11, FS4-12, BW01-BW04, BW13, BW15, BW16).

Written before the implementation. Synthetic inputs; the calendar comes from tools/fs_close_weekly
(build_protocol) and no second calendar exists. The TensorFlow trainer is replaced here by a
deterministic numpy stand-in injected through the Python API only; the CLI never accepts it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from tools import fs4_candidates as C
from tools import fs4_frontier as F
from tools import fs4_weekly_wrapper as WW
from tools import fs_close_weekly as W
from tools.business_weekly_protocol import EvaluationSplit

UTC = timezone.utc
POP = "EURUSD"
IDENT = "phase1-synthetic:0000000000000000"
NAMES = ["f_noise1", "f_noise2", "f_signal"]


# ----------------------------------------------------------------------------- fixtures
def _frame(start, end, rng, row_id_start=0, freq="6h"):
    import pandas as pd

    ts = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC")
    n = len(ts)
    X = rng.normal(size=(n, 3))
    y = 0.8 * X[:, 2] + rng.normal(scale=0.3, size=n)
    feats = pd.DataFrame(X, columns=NAMES)
    feats.insert(0, "row_id", np.arange(row_id_start, row_id_start + n))
    feats.insert(0, "t_decision_utc", ts)
    t = pd.DataFrame({"t_decision_utc": ts, "row_id": feats["row_id"], "Y_s_1h": y, "Y_s_2h": y * 0.5})
    return feats, t


def _inputs(tmp_path, val_end=datetime(2025, 1, 1, tzinfo=UTC), gap=False):
    rng = np.random.default_rng(0)
    tf, tt = _frame(datetime(2019, 6, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC), rng)
    vf, vt = _frame(datetime(2024, 1, 1, tzinfo=UTC), val_end, rng, row_id_start=len(tf))
    if gap:   # the third validation week has no rows at all
        keep = ~((vf["t_decision_utc"] >= "2024-01-15") & (vf["t_decision_utc"] < "2024-01-22"))
        vf, vt = vf[keep].reset_index(drop=True), vt[keep].reset_index(drop=True)
    d = tmp_path / "in"
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("train_features", tf), ("train_targets", tt), ("val_features", vf), ("val_targets", vt)):
        p = d / f"{name}.parquet"
        df.to_parquet(p)
        paths[name] = p
    return paths


def _consolidated():
    sets = [
        {"target_id": "Y_s_1h", "horizon_hours": 1, "methods": ["ALL_ADMISSIBLE"], "members": sorted(NAMES)},
        {"target_id": "Y_s_1h", "horizon_hours": 1, "methods": ["MRMR", "UNIVARIATE_MI"], "members": ["f_signal"]},
        {"target_id": "Y_s_1h", "horizon_hours": 1, "methods": ["RANDOM_K"], "members": ["f_noise1"]},
        {"target_id": "Y_s_2h", "horizon_hours": 2, "methods": ["ALL_ADMISSIBLE"], "members": sorted(NAMES)},
        {"target_id": "Y_s_2h", "horizon_hours": 2, "methods": ["JMI"], "members": ["f_noise2", "f_signal"]},
    ]
    out = []
    for s in sets:
        out.append(dict(s, set_id=C.set_identity(POP, IDENT, s["target_id"], s["members"]), population_id=POP, identity=IDENT,
                        k=len(s["members"]), declared_k=len(s["members"]), n_features=len(s["members"]),
                        control_methods=sorted(set(s["methods"]) & set(C.CONTROL_METHODS)), source_count=len(s["methods"]),
                        source_subset_sha256=[], phase3_closure_sha256="c" * 64, phase3_unit_id="u" * 64))
    body = {"schema": C.SCHEMA, "population_id": POP, "identity": IDENT, "phase3_closure_sha256": "c" * 64,
            "sets": sorted(out, key=lambda s: s["set_id"]), "consolidated_count": len(out)}
    body["consolidated_sha256"] = hashlib.sha256(json.dumps(body["sets"], sort_keys=True).encode()).hexdigest()
    return body


def _seal(tmp_path, cons):
    ranks = {t: {f: i + 1 for i, f in enumerate(["f_signal", "f_noise2", "f_noise1"])} for t in ("Y_s_1h", "Y_s_2h")}
    return F.seal_frontier(cons, ranks, None, tmp_path / "FRONTIER_SEAL.json")


def _plan(tmp_path, **kw):
    cons = _consolidated()
    seal = _seal(tmp_path, cons)
    return WW.build_plan([cons], [seal], validation_year=2024, input_modes=("RAW",), bar_hours={POP: 1}, **kw)


class LinearStandIn:
    """TEST-ONLY deterministic trainer: least squares on the last window step. Not an MLP, not production."""

    def __init__(self, spec, X, y, fit_idx, inner_idx, input_mode, encoder, seed):
        Xf = np.column_stack([X[fit_idx], np.ones(len(fit_idx))])
        self.W = np.linalg.lstsq(Xf, y[fit_idx], rcond=None)[0]
        self.weights_sha256 = hashlib.sha256(self.W.tobytes()).hexdigest()
        self.epochs_run, self.best_epoch, self.updates, self.fit_seconds = 1, 1, 1, 0.0
        self.peak_rss_bytes, self.n_params = 0, int(self.W.size)
        self.budget_sha256, self.architecture_sha256 = "b" * 64, "a" * 64
        self.encoder_sha256, self.input_identity = None, "i" * 64

    def predict(self, X, idx):
        return np.column_stack([X[idx], np.ones(len(idx))]) @ self.W


def _store(paths):
    return WW.DataStore.from_paths(POP, [paths["train_features"]], paths["train_targets"], [paths["val_features"]],
                                   paths["val_targets"], bar_hours=1)


# ----------------------------------------------------------------------------- plan
def test_plan_uses_the_existing_calendar_and_only_the_sealed_frontier(tmp_path):
    plan = _plan(tmp_path)
    protocol = W.build_protocol(2024, "a" * 64)
    expected = [W._iso(w.start) for w in protocol.weeks() if w.split is EvaluationSplit.VALIDATION]
    assert [w["start"] for w in plan["weeks"]] == expected and len(expected) == 52
    assert plan["evaluation_mode"] == "BUSINESS_WEEKLY_WALK_FORWARD" and plan["update_mode"] == "FULL_RETRAIN_ROLLING_4Y"
    assert plan["rolling_calendar_years"] == 4 and plan["seed"] == 0
    assert len(plan["sets"]) == 5 and plan["denominator"][POP] == {"sets": 5, "in_frontier": 5, "deferred": 0}
    assert plan["predictor_spec"]["window"] == 24 and plan["trainer"] == WW.PRODUCTION_TRAINER
    assert plan["aggregate_rule"] == WW.AGGREGATE_RULE and plan["tie_rule"] == WW.TIE_RULE
    assert plan["plan_sha256"] == WW.plan_digest(plan)
    with pytest.raises(WW.Refusal, match="LITERATURE_STATIC"):
        _plan(tmp_path, evaluation_mode="LITERATURE_STATIC")
    tasks = WW.enumerate_tasks(plan)
    assert len(tasks) == 5 * 52 and len({t["task_id"] for t in tasks}) == len(tasks)
    assert all(t["split"] == "validation" and t["plan_sha256"] == plan["plan_sha256"] for t in tasks)


# ----------------------------------------------------------------------------- one task
def test_task_fits_only_available_rows_before_the_cutoff_and_scores_the_week_with_a_paired_naive(tmp_path):
    plan = _plan(tmp_path)
    store = _store(_inputs(tmp_path))
    task = [t for t in WW.enumerate_tasks(plan) if t["week"]["ordinal"] == 5 and t["target_id"] == "Y_s_2h"][0]
    res = WW.run_task(task, store, trainer=LinearStandIn)
    assert res["status"] == "COMPLETE" and res["disposition"] == "COMPLETED" and res["task_id"] == task["task_id"]
    cutoff = datetime.strptime(task["week"]["cutoff"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    fit_max = datetime.strptime(res["fit_max_event_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert fit_max < cutoff and (cutoff - fit_max).total_seconds() >= 2 * 3600          # purge >= horizon
    assert res["fit_start"] == task["week"]["fit_start"] and res["fit_rows"] > 100 and res["inner_rows"] > 0
    ids = store.row_ids_between(task["week"]["start"], task["week"]["end"])
    assert res["n_scored"] == len(ids) and res["rows_sha256"] == WW.rows_digest(ids)
    y = store.target_values("Y_s_2h", ids)
    assert res["metrics"]["naive_mae"] == pytest.approx(float(np.mean(np.abs(y))))     # zero return, same rows
    assert res["metrics"]["mae"] < res["metrics"]["naive_mae"] and res["skill_mae"] > 0
    assert res["naive"]["rows_sha256"] == res["rows_sha256"] and res["naive"]["scale"] == "raw_log_return"
    for key in ("fit_seconds", "wall_seconds", "epochs", "best_epoch", "updates", "peak_rss_bytes", "n_params"):
        assert key in res["cost"]
    for key in ("model_sha256", "input_sha256", "code_sha256", "fit_population_digest", "standardiser_sha256"):
        assert len(res[key]) == 64
    assert res["seed"] == 0 and res["input_mode"] == "RAW" and res["update_mode"] == "FULL_RETRAIN_ROLLING_4Y"


def test_bytes_after_the_cutoff_do_not_change_the_week(tmp_path):
    plan = _plan(tmp_path)
    task = [t for t in WW.enumerate_tasks(plan) if t["week"]["ordinal"] == 3 and t["target_id"] == "Y_s_1h"][0]
    base = WW.run_task(task, _store(_inputs(tmp_path / "a")), trainer=LinearStandIn)
    paths = _inputs(tmp_path / "b")
    import pandas as pd

    vf = pd.read_parquet(paths["val_features"])
    vt = pd.read_parquet(paths["val_targets"])
    late = vf["t_decision_utc"] >= pd.Timestamp(task["week"]["end"])
    vf.loc[late, NAMES] = 1e6
    vt.loc[late, ["Y_s_1h", "Y_s_2h"]] = -1e6
    vf.to_parquet(paths["val_features"])
    vt.to_parquet(paths["val_targets"])
    perturbed = WW.run_task(task, _store(paths), trainer=LinearStandIn)
    assert perturbed["model_sha256"] == base["model_sha256"] and perturbed["metrics"] == base["metrics"]
    assert perturbed["rows_sha256"] == base["rows_sha256"] and perturbed["fit_population_digest"] == base["fit_population_digest"]
    assert perturbed["input_sha256"] != base["input_sha256"]                             # the bytes did change


def test_a_week_without_rows_is_a_failed_disposition_not_a_missing_one(tmp_path):
    plan = _plan(tmp_path)
    store = _store(_inputs(tmp_path, gap=True))
    task = [t for t in WW.enumerate_tasks(plan) if t["week"]["start"].startswith("2024-01-15") and t["target_id"] == "Y_s_1h"][0]
    res = WW.run_task(task, store, trainer=LinearStandIn)
    assert res["status"] == "COMPLETE" and res["disposition"] == "FAILED" and "no scored row" in res["reason"]
    assert "metrics" not in res and res["n_scored"] == 0 and res["task_id"] == task["task_id"]


def test_test_split_is_refused_until_the_procedure_is_frozen(tmp_path):
    plan = _plan(tmp_path)
    store = _store(_inputs(tmp_path))
    task = dict(WW.enumerate_tasks(plan)[0])
    task["split"] = "test"
    task["week"] = plan["test_weeks"][0]
    with pytest.raises(WW.Refusal, match="TEST_SEALED"):
        WW.run_task(task, store, trainer=LinearStandIn)
    with pytest.raises(WW.Refusal, match="TEST_SEALED"):
        WW.run_task(task, store, trainer=LinearStandIn, test_freeze={"freeze_sha256": "0" * 64})


def test_cli_never_accepts_a_stand_in_trainer():
    assert WW.PRODUCTION_TRAINER == "R0_TEMPORAL_CONV1D"
    with pytest.raises(SystemExit):
        WW.main(["run-task", "--trainer", "linear"])


# ----------------------------------------------------------------------------- aggregate and winner
def _result(set_id, week, mae, naive=1.0, n_features=2, fit_seconds=1.0, disposition="COMPLETED", input_mode="RAW"):
    r = {"status": "COMPLETE", "disposition": disposition, "set_id": set_id, "week_start": week, "input_mode": input_mode,
         "population_id": POP, "target_id": "Y_s_1h", "n_features": n_features, "cost": {"fit_seconds": fit_seconds, "epochs": 1},
         "rows_sha256": "r" * 64, "n_scored": 10}
    if disposition == "COMPLETED":
        r["metrics"] = {"mae": mae, "mse": mae ** 2, "naive_mae": naive, "naive_mse": naive ** 2}
        r["skill_mae"] = (naive - mae) / naive
    else:
        r["reason"] = "no scored row"
    return r


def test_winner_uses_the_mean_over_all_weeks_then_fewer_features_then_cost():
    weeks = ["2024-01-01T00:00:00Z", "2024-01-08T00:00:00Z", "2024-01-15T00:00:00Z"]
    results = []
    results += [_result("A", w, 0.5, n_features=3) for w in weeks]                    # mean skill 0.5
    results += [_result("B", w, m, n_features=2) for w, m in zip(weeks, (0.1, 0.9, 0.5))]   # mean skill 0.5, fewer features
    results += [_result("Cc", w, m, n_features=2, fit_seconds=5.0) for w, m in zip(weeks, (0.5, 0.5, 0.5))]   # tie -> cost
    results += [_result("D", w, 0.05) for w in weeks[:2]] + [_result("D", weeks[2], 0.0, disposition="FAILED")]  # best week, incomplete
    sets = {"A": 3, "B": 2, "Cc": 2, "D": 2}
    agg = WW.aggregate_results(results, weeks, sets_n_features=sets)
    assert agg["B"]["weeks_completed"] == 3 and agg["D"]["weeks_completed"] == 2 and agg["D"]["eligible"] is False
    assert agg["B"]["mean_weekly_skill_mae"] == pytest.approx(0.5) and agg["B"]["std_weekly_skill_mae"] > 0
    assert agg["A"]["weeks"] == 3 and all(d["disposition"] for d in agg["D"]["dispositions"])
    winner = WW.choose_winner(agg)
    assert winner["set_id"] == "B" and winner["tie_break"] == "fewer_features"
    agg2 = WW.aggregate_results([r for r in results if r["set_id"] in ("B", "Cc")], weeks, sets_n_features=sets)
    assert WW.choose_winner(agg2)["set_id"] == "B" and WW.choose_winner(agg2)["tie_break"] == "lower_cost"
    assert WW.choose_winner({"D": agg["D"]}) is None
    assert "mean" in WW.AGGREGATE_RULE and "ALL" in WW.AGGREGATE_RULE and "fewer features" in WW.TIE_RULE


def test_validation_batches_are_aligned_by_name_with_the_train_file(tmp_path):
    import pandas as pd

    paths = _inputs(tmp_path)
    vf = pd.read_parquet(paths["val_features"])
    ts_row = vf[["t_decision_utc", "row_id"]]
    # split the validation features in two batches, in another column order, as the PS1 batches are
    pd.concat([ts_row, vf[["f_signal"]]], axis=1).to_parquet(tmp_path / "in" / "vb1.parquet")
    pd.concat([ts_row, vf[["f_noise2", "f_noise1"]]], axis=1).to_parquet(tmp_path / "in" / "vb2.parquet")
    store = WW.DataStore.from_paths(POP, [paths["train_features"]], paths["train_targets"],
                                    [tmp_path / "in" / "vb1.parquet", tmp_path / "in" / "vb2.parquet"], paths["val_targets"], bar_hours=1)
    ref = _store(paths)
    assert store.names == ref.names and np.array_equal(store.X, ref.X)

