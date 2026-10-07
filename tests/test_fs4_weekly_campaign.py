"""Set-level weekly controller, equivalent to tools/fs4_campaign.py (plan section 5, FS4-07/08/11/12/13).

Written before the implementation. Sealed plan, one terminal per set x week x input mode, resume,
warehouse readback, closure on the full denominator, generated STATUS.json, TEST opened once.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools import fs4_candidates as C
from tools import fs4_frontier as F
from tools import fs4_weekly_campaign as K
from pathlib import Path
from tools import fs4_weekly_wrapper as WW

POP = "EURUSD"
IDENT = "phase1-synthetic:0000000000000000"
NAMES = ["f_a", "f_b", "f_c"]


def _consolidated(tmp_path):
    sets = [{"target_id": "Y_s_1h", "horizon_hours": 1, "methods": ["ALL_ADMISSIBLE"], "members": NAMES},
            {"target_id": "Y_s_1h", "horizon_hours": 1, "methods": ["MRMR"], "members": ["f_a"]},
            {"target_id": "Y_s_1h", "horizon_hours": 1, "methods": ["RANDOM_K"], "members": ["f_c"]}]
    out = [dict(s, set_id=C.set_identity(POP, IDENT, s["target_id"], s["members"]), population_id=POP, identity=IDENT,
                k=len(s["members"]), declared_k=len(s["members"]), n_features=len(s["members"]),
                control_methods=sorted(set(s["methods"]) & set(C.CONTROL_METHODS)), source_count=1, source_subset_sha256=[],
                phase3_closure_sha256="c" * 64, phase3_unit_id="u" * 64) for s in sets]
    body = {"schema": C.SCHEMA, "population_id": POP, "identity": IDENT, "phase3_closure_sha256": "c" * 64,
            "sets": sorted(out, key=lambda s: s["set_id"]), "consolidated_count": 3}
    body["consolidated_sha256"] = hashlib.sha256(json.dumps(body["sets"], sort_keys=True).encode()).hexdigest()
    p = tmp_path / "CONSOLIDATED_CANDIDATES.json"
    p.write_text(json.dumps(body))
    return p


def _extractibility(tmp_path, state="EXTRACTIBILITY_COMPLETE", name="EXTRACTIBILITY_COMPLETE.json"):
    body = {"schema": F.EXTRACTIBILITY_SCHEMA, "state": state, "populations": {POP: {"features": {}}}}
    body["closure_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    p = tmp_path / name
    p.write_text(json.dumps(body))
    return p


def _seal(tmp_path, cons_path, ext_path):
    cons = json.loads(cons_path.read_text())
    ranks = {"Y_s_1h": {"f_a": 1, "f_b": 2, "f_c": 3}}
    F.seal_frontier(cons, ranks, json.loads(ext_path.read_text()), tmp_path / "FRONTIER_SEAL.json")
    return tmp_path / "FRONTIER_SEAL.json"


def _campaign(tmp_path, modes=("RAW",)):
    cons = _consolidated(tmp_path)
    ext = _extractibility(tmp_path)
    seal = _seal(tmp_path, cons, ext)
    db = tmp_path / "weekly.sqlite"
    out = K.initialize(db, [cons], [seal], ext, out_dir=tmp_path / "out", validation_year=2024, input_modes=modes)
    return db, out


def _result(task, mae=0.5, naive=1.0, rows="e" * 64, disposition="COMPLETED"):
    r = {"schema": WW.RESULT_SCHEMA, "status": "COMPLETE", "task_id": task["task_id"], "disposition": disposition,
         "set_id": task["set_id"], "week_start": task["week"]["start"], "input_mode": task["input_mode"],
         "population_id": POP, "target_id": task["target_id"], "seed": 0, "split": task["split"],
         "rows_sha256": rows, "n_scored": 10, "n_features": len(task["members"]), "fit_population_digest": "f" * 64,
         "model_sha256": "a" * 64, "input_sha256": "b" * 64, "code_sha256": "c" * 64,
         "cost": {"fit_seconds": 1.0, "wall_seconds": 1.5, "epochs": 2, "best_epoch": 1, "updates": 10, "peak_rss_bytes": 1, "n_params": 5}}
    if disposition == "COMPLETED":
        r["metrics"] = {"mae": mae, "mse": mae ** 2, "naive_mae": naive, "naive_mse": naive ** 2}
        r["skill_mae"] = (naive - mae) / naive
    else:
        r["reason"] = "no scored row with a finite target"
    return r


def _finish(db, now0, mae_for, rows="e" * 64):
    """Claim and complete every claimable task (RAW first)."""
    i = 0
    while (t := K.claim(db, "w", now=now0 + i)) is not None:
        K.complete(db, "w", _result(t, mae=mae_for(t), rows=rows), now=now0 + i + 0.5)
        i += 1
    return i


def _mae(t):
    base = 0.3 if t["members"] == ["f_a"] else (0.5 if len(t["members"]) == 3 else 0.9)
    return base * (0.9 if t["input_mode"] == "TRAINED_ENCODER" else 1.0)


def test_init_requires_extractibility_closure_and_matching_seal(tmp_path):
    cons = _consolidated(tmp_path)
    running = _extractibility(tmp_path, state="RUNNING", name="RUNNING.json")
    with pytest.raises(K.Refusal, match="EXTRACTIBILITY_NOT_COMPLETE"):
        seal = _seal(tmp_path, cons, _extractibility(tmp_path))
        K.initialize(tmp_path / "q.sqlite", [cons], [seal], running, out_dir=tmp_path / "o", validation_year=2024, input_modes=("RAW",))
    ext = _extractibility(tmp_path)
    seal = _seal(tmp_path, cons, ext)
    body = json.loads(cons.read_text())
    body["sets"][0]["members"] = ["f_b"]
    body["consolidated_sha256"] = "0" * 64
    other = tmp_path / "other.json"
    other.write_text(json.dumps(body))
    with pytest.raises(K.Refusal, match="FRONTIER_SEAL_MISMATCH"):
        K.initialize(tmp_path / "q2.sqlite", [other], [seal], ext, out_dir=tmp_path / "o2", validation_year=2024, input_modes=("RAW",))
    with pytest.raises(K.Refusal, match="MISSING_EXTRACTIBILITY"):
        K.initialize(tmp_path / "q3.sqlite", [cons], [seal], tmp_path / "nope.json", out_dir=tmp_path / "o3", validation_year=2024, input_modes=("RAW",))


def test_plan_is_sealed_tasks_are_sets_times_weeks_times_modes_and_init_is_idempotent(tmp_path):
    db, out = _campaign(tmp_path)
    assert out["tasks"] == 3 * 52 and out["plan_sha256"] == K.status(db)["plan"]["plan_sha256"]
    plan_file = json.loads((tmp_path / "out" / "WEEKLY_PLAN.json").read_text())
    assert plan_file["plan_sha256"] == out["plan_sha256"]
    again = K.initialize(db, [tmp_path / "CONSOLIDATED_CANDIDATES.json"], [tmp_path / "FRONTIER_SEAL.json"],
                         tmp_path / "EXTRACTIBILITY_COMPLETE.json", out_dir=tmp_path / "out", validation_year=2024, input_modes=("RAW",))
    assert again["plan_sha256"] == out["plan_sha256"] and K.status(db)["total"] == 156
    with pytest.raises(K.Refusal, match="PLAN_CHANGED"):
        K.initialize(db, [tmp_path / "CONSOLIDATED_CANDIDATES.json"], [tmp_path / "FRONTIER_SEAL.json"],
                     tmp_path / "EXTRACTIBILITY_COMPLETE.json", out_dir=tmp_path / "out", validation_year=2023, input_modes=("RAW",))


def test_claim_orders_raw_first_then_week_and_a_terminal_is_never_repeated(tmp_path):
    db, _ = _campaign(tmp_path)
    first = K.claim(db, "worker-a", now=100)
    assert first["input_mode"] == "RAW" and first["week"]["ordinal"] == 0 and first["split"] == "validation"
    second = K.claim(db, "worker-b", now=101)
    assert second["task_id"] != first["task_id"] and second["week"]["ordinal"] == 0
    K.heartbeat(db, "worker-a", first["task_id"], now=102)
    K.complete(db, "worker-a", _result(first), now=103)
    assert K.status(db)["complete"] == 1
    with pytest.raises(K.Refusal, match="NO_LIVE_LEASE"):
        K.complete(db, "worker-a", _result(first), now=104)
    seen = set()
    for i in range(400):
        t = K.claim(db, "worker-c", now=200 + i)
        if t is None:
            break
        seen.add(t["task_id"])
    assert first["task_id"] not in seen and len(seen) == 156 - 2


def test_complete_validates_identity_finite_metrics_and_paired_rows_across_input_modes(tmp_path):
    db, _ = _campaign(tmp_path)
    raw = K.claim(db, "w", now=100)
    with pytest.raises(K.Refusal, match="NO_LIVE_LEASE|RESULT_IDENTITY"):
        K.complete(db, "w", {**_result(raw), "task_id": "x" * 64}, now=101)
    with pytest.raises(K.Refusal, match="INVALID_METRIC"):
        K.complete(db, "w", {**_result(raw), "metrics": {"mae": float("nan"), "mse": 1, "naive_mae": 1, "naive_mse": 1}}, now=101)
    with pytest.raises(K.Refusal, match="SEED_MISMATCH"):
        K.complete(db, "w", {**_result(raw), "seed": 1}, now=101)
    with pytest.raises(K.Refusal, match="MISSING_COST"):
        K.complete(db, "w", {k: v for k, v in _result(raw).items() if k != "cost"}, now=101)
    K.complete(db, "w", _result(raw, rows="a" * 64), now=102)
    with pytest.raises(K.Refusal, match="STAGE1_INCOMPLETE"):
        K.stage2(db)                                                          # stage 2 needs every stage-1 task terminal
    _finish(db, 1000, _mae, rows="a" * 64)
    out = K.stage2(db)
    assert out["tasks"] == 3 * 52 * 2
    peer_id = [t["task_id"] for t in K.list_tasks(db, set_id=raw["set_id"], week_start=raw["week"]["start"])
               if t["input_mode"] == "RANDOM_ENCODER"][0]
    peer = K.claim(db, "w", task_id=peer_id, now=5000)
    with pytest.raises(K.Refusal, match="PAIRED_ROWS_OR_NAIVE_MISMATCH"):
        K.complete(db, "w", _result(peer, rows="b" * 64), now=5001)
    with pytest.raises(K.Refusal, match="PAIRED_ROWS_OR_NAIVE_MISMATCH"):
        K.complete(db, "w", _result(peer, rows="a" * 64, naive=2.0), now=5002)
    K.complete(db, "w", _result(peer, rows="a" * 64, mae=0.9), now=5003)
    failed = K.claim(db, "w", now=5004)
    K.complete(db, "w", _result(failed, disposition="FAILED", rows="a" * 64), now=5005)      # a failed WEEK is a terminal disposition
    assert K.status(db)["failed"] == 0


def test_technical_failures_retry_three_times_then_stop(tmp_path):
    db, _ = _campaign(tmp_path, modes=("RAW",))
    t = K.claim(db, "w", now=1)
    for i in range(K.MAX_ATTEMPTS):
        K.fail(db, "w", t["task_id"], "runner rc=1", now=2 + i)
        if i < K.MAX_ATTEMPTS - 1:
            t = K.claim(db, "w", task_id=t["task_id"], now=10 + i)
            assert t is not None and t["attempt"] == i + 2
    assert K.claim(db, "w", task_id=t["task_id"], now=50) is None
    assert K.status(db)["failed"] == 1


def test_status_json_is_generated_from_the_store_and_eta_is_null_without_throughput(tmp_path):
    db, _ = _campaign(tmp_path, modes=("RAW",))
    s = K.status(db, now=100, write=True)
    doc = json.loads((tmp_path / "out" / "STATUS.json").read_text())
    for key in ("state", "expected", "complete", "failed", "active", "last_heartbeat", "rate_tasks_per_hour", "eta_seconds", "eta_reason", "by_population"):
        assert key in doc, key
    assert doc["expected"] == 156 and doc["eta_seconds"] is None and doc["eta_reason"] and doc["state"] == "PENDING"
    assert doc["generated_from"] == "task_store" and doc["expected"] == s["total"]
    t = K.claim(db, "w", now=200)
    K.heartbeat(db, "w", t["task_id"], now=260)
    s = K.status(db, now=300, write=True)
    assert s["active"] == 1 and s["last_heartbeat"] == 260 and s["state"] == "RUNNING"
    K.complete(db, "w", _result(t), now=320)
    s = K.status(db, now=330)
    assert s["eta_seconds"] is None and "no active worker" in s["eta_reason"]
    K.claim(db, "w", now=331)
    s = K.status(db, now=332)
    assert isinstance(s["eta_seconds"], int) and s["rate_tasks_per_hour"] > 0


def test_stage2_list_is_mechanical_sealed_and_encoders_never_choose_the_winner(tmp_path):
    db, out = _campaign(tmp_path)
    plan = json.loads((tmp_path / "out" / "WEEKLY_PLAN.json").read_text())
    import hashlib
    assert plan["stage2_rule_sha256"] == hashlib.sha256(Path(WW.STAGE2_RULE_PATH).read_bytes()).hexdigest()
    assert plan["stage2_modes"] == ["RANDOM_ENCODER", "TRAINED_ENCODER"] and plan["input_modes"] == ["RAW"]
    with pytest.raises(K.Refusal, match="STAGE2_NOT_COMPUTED"):
        K.close(db, warehouse_path=tmp_path / "wh.duckdb")
    _finish(db, 1000, _mae)
    s2 = K.stage2(db)
    assert sorted(s2["set_ids"]) == sorted(s["set_id"] for s in plan["sets"])          # all three are top-3 or ALL_ADMISSIBLE
    assert s2["stage2_rule_sha256"] == plan["stage2_rule_sha256"] and s2["list_sha256"]
    assert any(r.get("family") == "ALL_ADMISSIBLE" and r["rank"] == 0 for v in s2["reasons"].values() for r in v)
    assert K.stage2(db)["list_sha256"] == s2["list_sha256"] and K.status(db)["total"] == 156 + 312   # idempotent
    with pytest.raises(K.Refusal, match="DENOMINATOR_INCOMPLETE"):
        K.close(db, warehouse_path=tmp_path / "wh.duckdb")
    # the TRAINED arm is better than RAW everywhere, yet the winner is the RAW winner
    _finish(db, 9000, _mae)
    closure = K.close(db, warehouse_path=tmp_path / "wh.duckdb")
    assert list(closure["winners"][POP]["Y_s_1h"]) == ["RAW"] and closure["winners"][POP]["Y_s_1h"]["RAW"]["members"] == ["f_a"]
    cmp_ = closure["encoder_comparison"][POP]["Y_s_1h"]
    assert all(v["trained_minus_raw"] > 0 and abs(v["trained_minus_random"] - v["trained_minus_raw"]) < 1e-9 for v in cmp_.values())
    assert closure["stage2"]["list_sha256"] == s2["list_sha256"]


def test_close_refuses_until_every_task_is_terminal_then_selects_and_reads_back(tmp_path):
    db, _ = _campaign(tmp_path)
    with pytest.raises(K.Refusal, match="STAGE2_NOT_COMPUTED|DENOMINATOR_INCOMPLETE"):
        K.close(db, warehouse_path=tmp_path / "wh.duckdb")
    _finish(db, 1000, _mae)
    K.stage2(db)
    _finish(db, 9000, _mae)
    closure = K.close(db, warehouse_path=tmp_path / "wh.duckdb")
    assert closure["state"] == "WEEKLY_SELECTION_COMPLETE" and closure["test_opened"] is False
    win = closure["winners"][POP]["Y_s_1h"]["RAW"]
    assert win["members"] == ["f_a"] and win["weeks"] == 52
    assert closure["denominator"]["tasks"] == 468 and closure["denominator"]["terminal"] == 468
    assert closure["warehouse"]["table"] == K.WAREHOUSE_TABLE and closure["warehouse"]["readback"]["count"] == 468
    assert closure["warehouse"]["readback"]["matches_store"] is True
    doc = json.loads((tmp_path / "out" / "WEEKLY_SELECTION_COMPLETE.json").read_text())
    assert doc["closure_sha256"] == closure["closure_sha256"] and doc["final_selection"] is False
    again = K.close(db, warehouse_path=tmp_path / "wh.duckdb")                       # idempotent, no double writes
    assert again["closure_sha256"] == closure["closure_sha256"] and again["warehouse"]["readback"]["count"] == 468


def test_test_opens_once_only_after_the_freeze(tmp_path):
    db, _ = _campaign(tmp_path, modes=("RAW",))
    with pytest.raises(K.Refusal, match="NOT_CLOSED"):
        K.freeze(db)
    with pytest.raises(K.Refusal, match="NOT_FROZEN"):
        K.open_test(db)
    _finish(db, 1000, _mae)
    K.stage2(db)
    _finish(db, 9000, _mae)
    K.close(db, warehouse_path=tmp_path / "wh.duckdb")
    frozen = K.freeze(db)
    doc = json.loads((tmp_path / "out" / "TEST_FREEZE.json").read_text())
    assert doc["freeze_sha256"] == frozen["freeze_sha256"]
    for key in ("selector", "model", "preprocessing", "tie_rule", "aggregate_rule", "winners", "firewall_phase"):
        assert key in doc
    assert doc["firewall_phase"] == "PROCEDURE_SEALED"
    opened = K.open_test(db)
    assert opened["test_tasks"] == 52 and opened["opened_once"] is True
    with pytest.raises(K.Refusal, match="TEST_ALREADY_OPENED"):
        K.open_test(db)
    t = K.claim(db, "w", now=5000)
    assert t["split"] == "test" and t["test_authorization"] == frozen["freeze_sha256"]
    with pytest.raises(K.Refusal, match="FROZEN"):
        K.freeze(db)


def test_claim_filters_by_population_and_size_class(tmp_path):
    db, _ = _campaign(tmp_path)
    assert K.claim(db, "w", population="ETH", now=1) is None
    big = K.claim(db, "w", min_features=3, now=2)
    assert len(big["members"]) == 3 and big["population_id"] == POP
    small = K.claim(db, "w", max_features=1, now=3)
    assert len(small["members"]) == 1
    assert K.claim(db, "w", min_features=4, now=4) is None and K.claim(db, "w", max_features=0, now=5) is None
