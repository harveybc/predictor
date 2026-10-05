"""BUSINESS weekly walk-forward closure for FS-CLOSE (Musashi corrections 2026-10-05 section 3).

Written before the implementation: every test here failed against the static ``run_closure``
(single fit on TRAIN, one score over all of 2024). Synthetic inputs only; no real evidence is read.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M = _mod("fs_close_manifest", "tools/fs_close_manifest.py")
W = _mod("fs_close_weekly", "tools/fs_close_weekly.py")
from tools import fs_close_refit as R  # noqa: E402  (the module instance the weekly closure raises from)
G = _mod("selected_manifest_gate", "tools/selected_manifest_gate.py")

from tools.business_weekly_protocol import EvaluationMode, EvaluationSplit, subtract_calendar_years  # noqa: E402

UTC = timezone.utc
NAMES = ["f_noise1", "f_noise2", "f_noise3", "f_signal"]


# ----------------------------------------------------------------------------- synthetic inputs
def _hourly_frame(start: datetime, end: datetime, rng, row_id_start=0):
    import pandas as pd

    ts = pd.date_range(start, end, freq="6h", inclusive="left", tz="UTC")   # 6-hourly keeps the fixture small
    n = len(ts)
    X = rng.normal(size=(n, 4))
    y = 0.8 * X[:, 3] + rng.normal(scale=0.3, size=n)
    feats = pd.DataFrame(X, columns=NAMES)
    feats.insert(0, "row_id", np.arange(row_id_start, row_id_start + n))
    feats.insert(0, "t_decision_utc", ts)
    t = pd.DataFrame({"t_decision_utc": ts, "row_id": feats["row_id"]})
    for h in (1, 2, 3, 4, 5, 6):
        t[f"Y_s_{h}h"] = y
    for h in (24, 48, 72, 96, 120, 144):
        t[f"Y_l_{h}h"] = y * 0.5
    bar = np.where(y > 0.5, 1.0, np.where(y < -0.5, -1.0, 0.0))
    t["Y_b_s6"] = bar
    t["Y_b_l144"] = bar
    return feats, t


def _inputs(tmp_path, val_end=datetime(2025, 1, 1, tzinfo=UTC), rogue_2025_row=False):
    rng = np.random.default_rng(0)
    tf, tt = _hourly_frame(datetime(2019, 6, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC), rng)
    vf, vt = _hourly_frame(datetime(2024, 1, 1, tzinfo=UTC), val_end, rng, row_id_start=len(tf))
    if rogue_2025_row:
        vf.loc[vf.index[-1], "t_decision_utc"] = datetime(2025, 1, 3, tzinfo=UTC)
        vt.loc[vt.index[-1], "t_decision_utc"] = datetime(2025, 1, 3, tzinfo=UTC)
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    paths = {}
    for name, df in (("train_features", tf), ("train_targets", tt), ("val_features", vf), ("val_targets", vt)):
        p = d / f"{name}.parquet"
        df.to_parquet(p)
        paths[name] = p
    plan = {"schema": R.PLAN_SCHEMA, "population": sorted(NAMES), "population_sha256": R.names_sha256(sorted(NAMES)),
            "k_primary": 1,
            "sets": [{"set_id": "ALL_ADMISSIBLE", "set_kind": "ALL_ADMISSIBLE", "k": None, "features": sorted(NAMES)},
                     {"set_id": "PRED_BEST:1", "set_kind": "PRED_BEST", "k": 1, "features_by": {"*": ["f_signal"]}},
                     {"set_id": "RANDOM_K:1", "set_kind": "RANDOM_K", "k": 1, "features": ["f_noise1"]}]}
    pp = d / "plan.json"
    pp.write_text(json.dumps(plan))
    paths["plan"] = pp
    return paths


def _run(tmp_path, out="out", **kw):
    p = _inputs(tmp_path, **{k: v for k, v in kw.items() if k in ("val_end", "rogue_2025_row")})
    return W.run_weekly_closure(p["plan"], [p["train_features"]], p["train_targets"], [p["val_features"]], p["val_targets"],
                                tmp_path / out, k_primary=1,
                                **{k: v for k, v in kw.items() if k not in ("val_end", "rogue_2025_row")})


# ----------------------------------------------------------------------------- 1. static single fit cannot be FINAL
def test_static_single_fit_record_cannot_satisfy_c9_or_the_business_gate():
    static = {"schema": "fs_close_closure_record.v1", "evaluation_mode": "LITERATURE_STATIC_VALIDATION_DIAGNOSTIC",
              "winner": {"set_id": "PRED_BEST:24", "set_kind": "PRED_BEST", "set_sha256": "x", "score": 0.1},
              "validation_bound": {"min_ts": 1704067200, "max_ts": 1735600000}, "train_bound": {"max_ts": 1703887200},
              "validation_read_count": 1}
    assert W.business_closure_or_none(static) is None
    with pytest.raises(W.WeeklyClosureError, match="BUSINESS_WEEKLY_WALK_FORWARD"):
        W.require_business_closure(static)
    pop = M.Population(NAMES, {n: "batch_001" for n in NAMES}, [], [], M.names_sha256(NAMES), {"fixture": "a" * 64})
    ev = M.LaneEvidence()
    plan, _ = M.build_plan(pop, ev)
    rows = M.build_dispositions(pop, ev, plan, None)
    checks = M.run_checks(pop, ev, plan, [], rows, None, M.refit_coverage(plan, None), static, ROOT / "tools/selected_manifest_gate.py")
    c9 = next(c for c in checks if c["id"] == "C9_VALIDATION_CHOICE")
    assert c9["state"] != "PASS" and "BUSINESS_WEEKLY_WALK_FORWARD" in c9["detail"]


def test_gate_wrapper_rejects_any_other_evaluation_mode():
    pop = M.Population([f"f{i:03d}" for i in range(60)], {f"f{i:03d}": "b" for i in range(60)}, [], [],
                       M.names_sha256([f"f{i:03d}" for i in range(60)]), {"fixture": "a" * 64})
    ev = M.LaneEvidence()
    plan, _ = M.build_plan(pop, ev)
    win = next(s for s in plan["sets"] if s["set_id"] == "RANDOM_K:24")
    winner = {"set_id": win["set_id"], "set_kind": win["set_kind"], "set_sha256": win["set_sha256"]}
    rows = M.build_dispositions(pop, ev, plan, winner)
    checks = [{"id": "C1", "state": "PASS", "detail": "fixture"}]
    static_closure = {"winner": winner, "evaluation_mode": "LITERATURE_STATIC_VALIDATION_DIAGNOSTIC"}
    manifest, decision = M.build_manifest(pop, ev, plan, rows, checks, static_closure, {"x": "b" * 64}, True)
    selected = [f["name"] for f in manifest["features"] if f["state"] == "selected"]
    assert G.evaluate(manifest, selected, decision_record=decision)["admitted"]      # the generic gate alone would admit
    rep = M.business_gate(manifest, selected, decision)
    assert rep["admitted"] is False and any("EVALUATION_MODE" in r for r in rep["reasons"])
    weekly_closure = {"winner": winner, "evaluation_mode": EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD.value,
                      "update_mode": "FULL_RETRAIN_ROLLING_4Y"}
    manifest2, decision2 = M.build_manifest(pop, ev, plan, rows, checks, weekly_closure, {"x": "b" * 64}, True)
    assert M.business_gate(manifest2, selected, decision2)["admitted"] is True


# ----------------------------------------------------------------------------- 2-4. weeks, cutoffs, four years, no leakage
def test_weeks_are_derived_from_the_calendar_never_hard_coded():
    p2024 = W.build_protocol(2024, procedure_digest="a" * 64)
    p2021 = W.build_protocol(2021, procedure_digest="a" * 64)
    v24 = [w for w in p2024.weeks() if w.split is EvaluationSplit.VALIDATION]
    v21 = [w for w in p2021.weeks() if w.split is EvaluationSplit.VALIDATION]
    assert v24[0].start == datetime(2024, 1, 1, tzinfo=UTC) and v24[-1].end == datetime(2024, 12, 30, tzinfo=UTC)
    assert len(v24) == 52
    # 2021: first Monday is Jan 4, last complete week ends Mon Dec 27 -> 51 weeks, computed, never a constant
    assert v21[0].start == datetime(2021, 1, 4, tzinfo=UTC) and v21[-1].end == datetime(2021, 12, 27, tzinfo=UTC) and len(v21) == 51
    assert all(w.cutoff == w.start and w.fit_start == subtract_calendar_years(w.cutoff, 4) for w in v24)
    assert p2024.evaluation_mode is EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD


def test_every_scored_week_has_spec_cutoff_four_years_fit_digest_and_own_weights(tmp_path):
    pytest.importorskip("sklearn")
    rec = _run(tmp_path)
    assert rec["evaluation_mode"] == "BUSINESS_WEEKLY_WALK_FORWARD" and rec["update_mode"] == "FULL_RETRAIN_ROLLING_4Y"
    weeks = rec["weeks"]
    assert len(weeks) == 52 and weeks[0]["start"].startswith("2024-01-01")
    for s in rec["sets"]:
        per_week = s["weeks"]
        assert len(per_week) == 52
        digests = set()
        for w in per_week:
            assert w["cutoff"] == w["start"]
            assert w["fit_start"] == W._iso(subtract_calendar_years(W._parse(w["cutoff"]), 4))
            for fam in ("short", "long", "barrier_s6", "barrier_l144"):
                unit = w["families"][fam]
                assert unit["fit_population_digest"] and unit["model_digest"] and unit["fit_rows"] > 0
                assert W._parse(unit["fit_end"]) <= W._parse(w["cutoff"])          # no row at/after the cutoff enters the fit
                assert W._parse(unit["fit_max_event_time"]) < W._parse(w["cutoff"])
                digests.add((fam, unit["model_digest"]))
        assert len(digests) == 4 * 52                                              # fresh weights every week and family
    assert rec["winner"]["set_id"] == "PRED_BEST:1"                                # the signal set wins by the predeclared aggregate


def test_same_row_naive_per_week_and_identical_origins_across_horizons(tmp_path):
    pytest.importorskip("sklearn")
    rec = _run(tmp_path)
    s = rec["sets"][0]
    for w in s["weeks"]:
        hz = w["horizon_scores"]
        counts = {h["sample_count"] for h in hz}
        assert len(counts) == 1 and all(h["naive_mae"] > 0 and h["model_mae"] >= 0 for h in hz)
        assert len(hz) == 12
        for fam in ("barrier_s6", "barrier_l144"):
            b = w["families"][fam]
            assert b["naive_kind"] == "fit_prior" and np.isfinite(b["naive_log_loss"]) and b["n_rows"] == b["naive_n_rows"]


# ----------------------------------------------------------------------------- 5. contract sealed before reading 2024
def test_contract_is_sealed_before_any_validation_read(tmp_path):
    pytest.importorskip("sklearn")
    rec = _run(tmp_path)
    assert rec["contract"]["sealed_utc"] <= rec["validation_first_read_utc"]
    sealed = json.loads((tmp_path / "out/weekly_contract.json").read_text())
    assert sealed["contract_sha256"] == rec["contract"]["contract_sha256"]
    assert rec["procedure_digest"] == sealed["contract_sha256"]


# ----------------------------------------------------------------------------- 6. TEST 2025 inaccessible
def test_validation_loader_refuses_a_2025_row_and_firewall_never_opens_test(tmp_path):
    pytest.importorskip("sklearn")
    with pytest.raises(R.RefitError, match="TEST_READ_REFUSED"):
        _run(tmp_path, out="out_bad", rogue_2025_row=True)
    rec = _run(tmp_path)
    assert rec["validation_bound"]["max_ts"] < 1735689600 and rec["firewall_phase"] == "VALIDATION_SELECTION"
    assert rec["test_weeks_opened"] == 0


# ----------------------------------------------------------------------------- 7. resume never repeats a terminal week
def test_resume_never_repeats_a_terminal_week(tmp_path):
    pytest.importorskip("sklearn")
    rec1 = _run(tmp_path)
    assert rec1["fits_performed"] == 2 * 52 * 4            # two candidate sets (RANDOM_K is a control, not a closure candidate)
    rec2 = _run(tmp_path)
    assert rec2["fits_performed"] == 0 and rec2["weeks_restored"] == 2 * 52
    assert rec2["winner"] == rec1["winner"]
    for a, b in zip(rec1["sets"], rec2["sets"]):
        assert a["ledger_digest"] == b["ledger_digest"]


# ----------------------------------------------------------------------------- 8. absence of TRAIN never defaults to RAW
def test_feature_without_train_observations_is_not_available_and_never_raw():
    pop = M.Population(sorted(NAMES), {n: "b" for n in NAMES}, [], [], M.names_sha256(sorted(NAMES)), {"fixture": "a" * 64})
    ev = M.LaneEvidence()
    ev.heavy = ["f_noise1", "f_noise2"]
    ev.rep = {"f_noise1": {"decision": "NOT_AVAILABLE_FOR_TRAIN", "flags": "NO_OBSERVED_TRAIN_VALUES"},
              "f_noise2": {"decision": "RAW", "flags": "DECIDED_WITH_FAILED_FAMILIES"}}   # raw has support, trained failed
    M.apply_availability(pop, ev)
    assert ev.not_available == {"f_noise1": "NO_OBSERVED_TRAIN_VALUES"}
    plan, _ = M.build_plan(pop, ev)
    for s in plan["sets"]:
        lists = s["features_by"].values() if "features_by" in s else [s["features"]]
        assert all("f_noise1" not in lst for lst in lists), s["set_id"]
    assert plan["fit_population_sha256"] != pop.digest and plan["denominator"] == 4
    rows = {r["feature"]: r for r in M.build_dispositions(pop, ev, plan, None)}
    assert rows["f_noise1"]["availability"] == "NOT_AVAILABLE_FOR_TRAIN" and rows["f_noise1"]["state"] == "REJECTED"
    assert rows["f_noise1"]["representation"] == "NOT_APPLICABLE" and "NO_OBSERVED_TRAIN_VALUES" in rows["f_noise1"]["reason_codes"]
    assert rows["f_noise2"]["representation"] == "RAW" and rows["f_noise2"]["availability"] == "AVAILABLE"
    assert len(rows) == 4
