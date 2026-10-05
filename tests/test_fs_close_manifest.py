"""FS-CLOSE closure follower and paired-refit engine: fail-closed invariants on tiny fixtures.

Pure python + numpy/pandas/pyarrow (scikit-learn only for the barrier head test, skipped if absent).
No real evidence is read: every fixture is synthetic and built in a temporary directory.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M = _mod("fs_close_manifest", "tools/fs_close_manifest.py")
R = _mod("fs_close_refit", "tools/fs_close_refit.py")
G = _mod("selected_manifest_gate", "tools/selected_manifest_gate.py")


# ----------------------------------------------------------------------------- fixtures
def _pop(n=5):
    names = sorted(f"f{i:03d}" for i in range(n))
    return M.Population(names, {x: "batch_001" for x in names}, [], [], M.names_sha256(names), {"fixture": "a" * 64})


def _ev(pop, causal=None, rep=None, heavy=None):
    ev = M.LaneEvidence()
    ev.causal = causal or {}
    ev.rep = rep or {}
    ev.heavy = heavy or []
    ev.ps2_status = {f: {"statuses": {}, "oof_delta": {"Y_s|1": 0.001}, "redundant_with": set()} for f in pop.names}
    return ev


def _plan(pop, ev):
    plan, _ = M.build_plan(pop, ev)
    return plan


def _winner(plan):
    s = next((x for x in plan["sets"] if x["set_id"] == "RANDOM_K:24"), None) or \
        next((x for x in plan["sets"] if x["set_id"] == "RANDOM_K:8"), None) or plan["sets"][0]
    return {"set_id": s["set_id"], "set_kind": s["set_kind"], "set_sha256": s["set_sha256"]}


# ----------------------------------------------------------------------------- 366 invariant
def test_default_denominator_is_366_and_heavy_137():
    assert M.DENOMINATOR == 366 and M.HEAVY_CANDIDATES == 137 and M.SELECTOR_EPISODE_SOURCES == 37


def test_population_loader_refuses_wrong_count(tmp_path):
    ev_dir = tmp_path / "docs/audits/evidence/canonical_20261003"
    for batch, feats in (("batch_001", ["a", "b"]), ("batch_002", ["c"]), ("batch_003", ["d"])):
        d = ev_dir / "laneA" / batch
        d.mkdir(parents=True)
        (d / "admissible_features.csv").write_text("feature_id,role\n" + "".join(f"{f},feature\n" for f in feats))
    (ev_dir / "laneA/batch_002/role_overlay_batch_001.json").write_text(json.dumps(
        {"selector_episode_source_features": ["b"] * 0 + [f"ev{i}" for i in range(37)]}))
    paths = M.Paths(repo=tmp_path, state=tmp_path / "state")
    with pytest.raises(M.ClosureError, match="POPULATION_MISMATCH"):
        M.load_population(paths)                      # 4 != 366
    pop = M.load_population(paths, denominator=4)
    assert pop.names == ["a", "b", "c", "d"] and pop.digest == M.names_sha256(["a", "b", "c", "d"])


def test_dispositions_cover_every_candidate_exactly_once_and_check_c1_fails_otherwise():
    pop = _pop(5)
    ev = _ev(pop)
    plan = _plan(pop, ev)
    rows = M.build_dispositions(pop, ev, plan, None)
    assert [r["feature"] for r in rows] == pop.names
    assert all(r["state"] == "PENDING" for r in rows)
    cov = M.refit_coverage(plan, None)
    checks = M.run_checks(pop, ev, plan, [], rows, None, cov, None, ROOT / "tools/selected_manifest_gate.py")
    c1 = next(c for c in checks if c["id"] == "C1_DISPOSITIONS_366")
    assert c1["state"] == "FAIL"                     # 5 rows is not 366: fail-closed
    checks366 = M.run_checks(pop, ev, plan, [], rows + [dict(rows[0], feature=f"x{i}") for i in range(361)],
                             None, cov, None, ROOT / "tools/selected_manifest_gate.py")
    c1b = next(c for c in checks366 if c["id"] == "C1_DISPOSITIONS_366")
    assert c1b["state"] == "FAIL"                    # 366 rows but not the population: still fails


# ----------------------------------------------------------------------------- NOT_IDENTIFIED is neutral
def test_not_identified_never_rejects():
    pop = _pop(60)
    causal = {f: {"Y_s_1h": "NOT_IDENTIFIED"} for f in pop.names}
    ev = _ev(pop, causal=causal)
    plan = _plan(pop, ev)
    win = _winner(plan)
    rows = M.build_dispositions(pop, ev, plan, win)
    not_selected = [r for r in rows if r["state"] != "SELECTED"]
    assert not_selected, "fixture needs non-selected rows"
    for r in not_selected:
        assert r["state"] == "PENDING", r
        assert "CAUSAL_NOT_IDENTIFIED_NEUTRAL" in r["reason_codes"]
    checks = M.run_checks(pop, ev, plan, [], rows, None, M.refit_coverage(plan, None), None, ROOT / "tools/selected_manifest_gate.py")
    assert next(c for c in checks if c["id"] == "C5_NOT_IDENTIFIED_NEUTRAL")["state"] == "PASS"
    # a rejection needs positive evidence of its own (here: no OOF utility on any target), never the causal gap
    for f in pop.names:
        ev.ps2_status[f]["oof_delta"] = {"Y_s|1": -0.001, "Y_l|24": -0.002}
    rows2 = M.build_dispositions(pop, ev, plan, win)
    rejected = [r for r in rows2 if r["state"] == "REJECTED"]
    assert rejected and all("REJECTED_NO_OOF_UTILITY" in r["reason_codes"] for r in rejected)
    assert not any("CONTRADICTED" in r["reason_codes"] for r in rejected)


def test_reorder_causal_keeps_not_identified_in_place_and_demotes_only_robust_contradiction():
    order = ["a", "b", "c", "d"]
    out = M.reorder_causal(order, {"a": "CONTRADICTED", "c": "SUPPORTED", "b": "NOT_IDENTIFIED"})
    assert out == ["c", "b", "d", "a"]


# ----------------------------------------------------------------------------- gate acceptance / DRAFT refusal
def _sources():
    return {"laneA/x.csv": "b" * 64}


def test_final_manifest_passes_the_gate_and_draft_is_refused():
    pop = _pop(60)
    ev = _ev(pop)
    plan = _plan(pop, ev)
    win = _winner(plan)
    rows = M.build_dispositions(pop, ev, plan, win)
    checks = [{"id": "C1", "state": "PASS", "detail": "fixture"}]
    final, decision = M.build_manifest(pop, ev, plan, rows, checks, {"winner": win}, _sources(), True)
    selected = [f["name"] for f in final["features"] if f["state"] == "selected"]
    assert selected and final["schema"] == G.MANIFEST_SCHEMA
    rep = G.evaluate(final, selected, decision_record=decision, expected_dataset_id=M.DATASET_ID)
    assert rep["admitted"] is True, rep["reasons"]
    assert decision["decider_role"] != final["producer"]["role"]
    draft, none = M.build_manifest(pop, ev, plan, M.build_dispositions(pop, ev, plan, None), checks, None, _sources(), False)
    assert none is None and draft["status"] == "DRAFT" and draft["schema"] == M.DRAFT_SCHEMA
    rep2 = G.evaluate(draft, pop.names[:2], decision_record=decision)
    assert rep2["admitted"] is False
    assert any(r.startswith("WRONG_SCHEMA") for r in rep2["reasons"])
    # a DRAFT body relabelled as the real schema still refuses: all pending, no decision binding
    forged = copy.deepcopy(draft)
    forged["schema"] = G.MANIFEST_SCHEMA
    forged["manifest_sha256"] = G.canonical_sha256(forged, "manifest_sha256")
    rep3 = G.evaluate(forged, pop.names[:2], decision_record=decision)
    assert rep3["admitted"] is False and any(r.startswith("NO_SELECTED_FEATURE") for r in rep3["reasons"])


def test_vendored_gate_is_the_c0345f83_contract():
    assert M.sha256_file(ROOT / "tools/selected_manifest_gate.py") == M.GATE_SHA256_C0345F83


# ----------------------------------------------------------------------------- paired naive presence
def _metrics_frame(rows):
    import pandas as pd

    cols = list(R.METRIC_COLUMNS)
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


def _metric_row(pop, **kw):
    base = {"set_id": "ALL_ADMISSIBLE", "set_kind": "ALL_ADMISSIBLE", "set_sha256": "x", "k": -1, "head": "ridge",
            "target": "Y_s", "horizon": 1, "fold": "inner_2019", "metric": "mae", "value": 1.0, "naive_kind": "zero",
            "naive_value": 1.1, "skill": 0.09, "n_fit": 10, "n_rows": 5, "rows_sha256": "r", "population_sha256": pop.digest,
            "features_sha256": "f", "n_features": 5, "seed": 0, "fit_seconds": 0.0, "predict_ms_per_row": 0.0,
            "peak_rss_bytes": 0, "plan_sha256": "p", "code_sha256": "c", "state": "MEASURED", "reason": ""}
    base.update(kw)
    return base


def test_check_c6_fails_when_a_measured_metric_lacks_its_paired_naive():
    pop = _pop(5)
    ev = _ev(pop)
    plan = _plan(pop, ev)
    rows = M.build_dispositions(pop, ev, plan, None)
    good = _metrics_frame([_metric_row(pop)])
    bad = _metrics_frame([_metric_row(pop), _metric_row(pop, naive_kind="", naive_value=float("nan"))])
    gate = ROOT / "tools/selected_manifest_gate.py"
    ok = M.run_checks(pop, ev, plan, [], rows, good, M.refit_coverage(plan, good), None, gate)
    ko = M.run_checks(pop, ev, plan, [], rows, bad, M.refit_coverage(plan, bad), None, gate)
    assert next(c for c in ok if c["id"] == "C6_PAIRED_NAIVE")["state"] == "PASS"
    assert next(c for c in ko if c["id"] == "C6_PAIRED_NAIVE")["state"] == "FAIL"


def test_check_c2_fails_on_mixed_populations():
    pop = _pop(5)
    ev = _ev(pop)
    plan = _plan(pop, ev)
    rows = M.build_dispositions(pop, ev, plan, None)
    mixed = _metrics_frame([_metric_row(pop), _metric_row(pop, population_sha256="0" * 64)])
    checks = M.run_checks(pop, ev, plan, [], rows, mixed, M.refit_coverage(plan, mixed), None, ROOT / "tools/selected_manifest_gate.py")
    assert next(c for c in checks if c["id"] == "C2_SINGLE_POPULATION")["state"] == "FAIL"


def test_check_c3_fails_when_a_ranking_is_incomplete():
    pop = _pop(5)
    ev = _ev(pop)
    ev.pred_rankings = {"SPEARMAN_K": {"Y_s_1h|inner_2019": pop.names[:3]}}
    ev.pred_incomplete = {"SPEARMAN_K": "incomplete ranking: 1 partial, 69 missing cells"}
    plan, missing = M.build_plan(pop, ev)
    assert not any(s["set_kind"] == "PRED_METHOD" for s in plan["sets"])   # incomplete method yields no K set
    rows = M.build_dispositions(pop, ev, plan, None)
    checks = M.run_checks(pop, ev, plan, missing, rows, None, M.refit_coverage(plan, None), None, ROOT / "tools/selected_manifest_gate.py")
    assert next(c for c in checks if c["id"] == "C3_COMPLETE_RANKINGS")["state"] == "FAIL"


# ----------------------------------------------------------------------------- no-test-read guard
def test_refit_loader_refuses_rows_at_or_after_train_end():
    import datetime as dt

    end = dt.datetime.fromisoformat(R.TRAIN_END_UTC).timestamp()
    ok = np.array([end - 7200, end - 3600], dtype="int64")
    assert R.assert_train_only(ok)["rows"] == 2
    with pytest.raises(R.RefitError, match="TEST_OR_VALIDATION_READ_REFUSED"):
        R.assert_train_only(np.array([end - 3600, end], dtype="int64"))
    with pytest.raises(R.RefitError, match="TEST_OR_VALIDATION_READ_REFUSED"):
        R.assert_train_only(np.array([end + 86400 * 400], dtype="int64"))   # a 2025 row


def test_plan_loader_refuses_sets_outside_population_or_with_wrong_k(tmp_path):
    names = ["a", "b", "c"]
    plan = {"schema": R.PLAN_SCHEMA, "population": names, "population_sha256": R.names_sha256(names),
            "sets": [{"set_id": "S", "set_kind": "RANDOM_K", "k": 2, "features": ["a", "z"]}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    with pytest.raises(R.RefitError, match="OUTSIDE_POPULATION"):
        R.load_plan(p)
    plan["sets"][0]["features"] = ["a", "b", "c"]
    p.write_text(json.dumps(plan))
    with pytest.raises(R.RefitError, match="K_MISMATCH"):
        R.load_plan(p)


# ----------------------------------------------------------------------------- refit engine on synthetic data
def _synthetic_inputs(tmp_path, n=600):
    import pandas as pd

    rng = np.random.default_rng(0)
    ts = np.arange(n) * 3600 + 1_335_830_400          # hourly from 2012-05-01, all TRAIN
    X = rng.normal(size=(n, 4))
    y = 0.8 * X[:, 0] + rng.normal(scale=0.3, size=n)
    names = ["f_signal", "f_noise1", "f_noise2", "f_noise3"]
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "row_id", np.arange(n))
    df.insert(0, "t_decision_utc", pd.to_datetime(ts, unit="s", utc=True))
    feats = tmp_path / "features_train.parquet"
    df.to_parquet(feats)
    t = pd.DataFrame({"t_decision_utc": df["t_decision_utc"], "row_id": df["row_id"]})
    for h in (1, 2, 3, 4, 5, 6):
        t[f"Y_s_{h}h"] = y
    for h in (24, 48, 72, 96, 120, 144):
        t[f"Y_l_{h}h"] = y * 0.5
    bar = np.where(y > 0.5, 1.0, np.where(y < -0.5, -1.0, 0.0))
    t["Y_b_s6"] = bar
    t["Y_b_l144"] = bar
    tp = tmp_path / "targets_train.parquet"
    t.to_parquet(tp)
    folds = {"folds": [{"name": "inner_2019", "train_rows": [0, 400], "val_rows": [420, 600]},
                       {"name": "inner_2020", "train_rows": [0, 450], "val_rows": [470, 600]}]}
    fp = tmp_path / "folds.json"
    fp.write_text(json.dumps(folds))
    plan = {"schema": R.PLAN_SCHEMA, "population": sorted(names), "population_sha256": R.names_sha256(sorted(names)),
            "sets": [{"set_id": "ALL_ADMISSIBLE", "set_kind": "ALL_ADMISSIBLE", "k": None, "features": sorted(names)},
                     {"set_id": "SIGNAL:1", "set_kind": "PRED_BEST", "k": 1, "features_by": {"*": ["f_signal"]}},
                     {"set_id": "NOISE:1", "set_kind": "RANDOM_K", "k": 1, "features": ["f_noise1"]}]}
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan))
    return pp, [feats], tp, fp


def test_refit_engine_scores_every_set_on_identical_rows_with_paired_naive(tmp_path):
    pytest.importorskip("sklearn")
    import pandas as pd

    pp, feats, tp, fp = _synthetic_inputs(tmp_path)
    out = tmp_path / "out"
    receipt = R.run_plan(pp, feats, tp, fp, out, heads=("ridge",))
    df = pd.read_parquet(out / "paired_refit_metrics.parquet")
    m = df[df["state"] == "MEASURED"]
    assert receipt["cells_failed"] == 0 and len(m) > 0
    # identical rows per (target, horizon, fold) across every set
    per_cell = m.groupby(["target", "horizon", "fold"])["rows_sha256"].nunique()
    assert (per_cell == 1).all()
    assert (m["naive_kind"] != "").all() and np.isfinite(m["naive_value"]).all()
    assert (m["population_sha256"] == R.names_sha256(sorted(["f_signal", "f_noise1", "f_noise2", "f_noise3"]))).all()
    # the signal set beats the zero naive on Y_s; the noise set does not do better than the signal set
    sig = m[(m["set_id"] == "SIGNAL:1") & (m["target"] == "Y_s") & (m["metric"] == "mae") & (m["naive_kind"] == "zero")]
    noise = m[(m["set_id"] == "NOISE:1") & (m["target"] == "Y_s") & (m["metric"] == "mae") & (m["naive_kind"] == "zero")]
    assert (sig["skill"] > 0).all() and sig["value"].mean() < noise["value"].mean()
    # barrier cells carry the prior naive
    yb = m[m["target"] == "Y_b"]
    assert set(yb["naive_kind"]) == {"fit_prior"} and set(yb["metric"]) == {"log_loss", "brier"}
    # idempotent: a second run recomputes nothing
    receipt2 = R.run_plan(pp, feats, tp, fp, out, heads=("ridge",))
    assert receipt2["cells_done"] == 0 and receipt2["cells_skipped"] == receipt["cells_done"]
    assert receipt["train_bound"]["max_ts"] < 1_704_067_200


# ----------------------------------------------------------------------------- refit gain export (FS-REP G4)
def test_refit_gain_is_positive_for_a_feature_whose_removal_hurts():
    pop = _pop(3)
    ev = _ev(pop, heavy=[pop.names[0]])
    plan = _plan(pop, ev)
    sha = {s["set_id"]: s["set_sha256"] for s in plan["sets"]}
    rows = [_metric_row(pop, set_id="ALL_ADMISSIBLE", set_sha256=sha["ALL_ADMISSIBLE"], value=1.0),
            _metric_row(pop, set_id=f"ALL_MINUS:{pop.names[0]}", set_kind="REMOVAL", set_sha256=sha[f"ALL_MINUS:{pop.names[0]}"], value=1.2)]
    gains, cells = M.refit_gain_rows(_metrics_frame(rows), plan)
    assert len(gains) == 1 and gains[0]["feature_id"] == pop.names[0] and gains[0]["refit_gain"] > 0
    assert cells[0]["loss_without"] == 1.2 and cells[0]["naive_kind"] == "zero"


# ----------------------------------------------------------------------------- status is derived, not narrated
def test_status_and_checklist_are_computed_from_coverage():
    pop = _pop(4)
    ev = _ev(pop)
    plan = _plan(pop, ev)
    cov = M.refit_coverage(plan, None)
    rows = M.build_dispositions(pop, ev, plan, None)
    lanes = M.lane_coverage(pop, ev, cov, rows, None)
    assert lanes["PS1"]["fraction"] == 1.0 and lanes["FS-CLOSE"]["done"] == 0
    cl = M.checklist(lanes, final=False)
    assert cl["I5_joint_comparison_and_manifest"] == "NOT_STARTED"
    assert M.checklist(lanes, final=True)["I5_joint_comparison_and_manifest"] == "DONE"
