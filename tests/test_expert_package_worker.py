import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "tools/expert_package_worker.py"
    spec = importlib.util.spec_from_file_location("expert_package_worker", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cfg(root):
    return {"root": str(root), "python": "/env/python", "launcher": "/bin/admit",
            "gpu_uuid": "GPU-test", "memory": "4G", "seeds": [20261019, 20261020, 20261021],
            "pins": {}}


def artifacts(w, config, job):
    out = Path(config["root"]) / "out" / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    requested = ["x", "y"]
    lists = Path(config["root"]) / "in"
    lists.mkdir(exist_ok=True)
    (lists / job["list"]).write_text(json.dumps({"features": requested}))
    metrics = {"net_profit_usd": -1, "sharpe_daily_sqrt365": -0.1,
               "max_drawdown_pct": 2, "trades": 3, "win_rate": 0.3}
    trading = {p: metrics for p in w.periods(job)}
    trials = {str(s): {"trading": trading, "musashi_contract": trading}
              for s in config["seeds"]}
    result = {"device": "cuda", "verify": True, "scaling": "fixed_train",
              "arch": {"units": 2, "head": 16}, "variants": ["FULL"],
              "split_contract": {"arm": job["split"]},
              "agent_state": {"mode": job["state"]},
              "meta": {"support_hash": {p: "same" for p in w.periods(job)}},
              "trials": {"FULL": trials}, "wall_seconds": 1.0}
    probe = {"split_contract": job["split"], "scaling": "fixed_train",
             "agent_state": job["state"], "probe": {}}
    for s in config["seeds"]:
        for m in w.MODULES:
            name = f"FULL_{m}_{s}"
            (out / (name + ".pt")).write_bytes(b"synthetic weights")
            probe["probe"][name] = {"rows": [1], "logits": [[0.1, 0.2]]}
    for name, value in {"E3M.json": result, "SCALER.json": {
            "features": requested, "policy": "fixed_train", "mean": [0, 0],
            "std": [1, 1], "split_contract": {"arm": job["split"]}},
            "PROBE.json": probe}.items():
        (out / name).write_text(json.dumps(value))
    return out


def test_exact_twelve_jobs_and_supplied_factorial(tmp_path):
    w = module()
    jobs = w.jobs()
    assert len(jobs) == 12 and len({j["name"] for j in jobs}) == 12
    assert [j["name"] for j in jobs[:4]] == ["cons_core15", "cons_shortlist_strict",
                                                    "cons_shortlist", "cons_shortlist_strict_enc"]
    assert all(j["split"] == "consolidated" for j in jobs[:4])
    assert {(j["list"], j["state"], j["dropout"], j["wd"]) for j in jobs[4:]} == {
        (f"FEATURE_LIST_{f}.json", s, d, v)
        for f in ("core15", "shortlist_strict") for s in ("full", "none")
        for d, v in ((0.1, 1e-4), (0.3, 1e-2))}
    cmd = w.command(cfg(tmp_path), jobs[0])
    assert cmd[:2] == ["/bin/admit", "-q"]
    assert cmd[cmd.index("-m") + 1] == "4G"
    assert "--no-verify" not in cmd and "--cpu" not in cmd
    assert "--split-contract" in cmd and "--nonstationary" in cmd


def test_large_gpu_is_not_assumed_to_be_index_zero():
    w = module()
    assert w.choose_gpu("0, RTX 5070 Ti, GPU-small\n1, RTX 5090, GPU-large\n") == "GPU-large"
    with pytest.raises(ValueError):
        w.choose_gpu("0, RTX 5070 Ti, GPU-small")
    with pytest.raises(ValueError):
        w.choose_gpu("0, RTX 5090, GPU-a\n1, RTX 5090, GPU-b")


def test_pin_change_is_not_a_retryable_resource_problem(tmp_path):
    w = module()
    p = tmp_path / "input"
    p.write_bytes(b"original")
    c = cfg(tmp_path)
    c["pins"] = {str(p): w.digest(p)}
    w.verify_pins(c)
    p.write_bytes(b"other")
    with pytest.raises(ValueError, match="IDENTITY"):
        w.verify_pins(c)


def test_complete_negative_results_are_successful_execution(tmp_path):
    w = module()
    c, job = cfg(tmp_path), w.jobs()[0]
    artifacts(w, c, job)
    manifest = w.validate_result(c, job)
    assert len(manifest["hashes"]) == 15
    assert len(manifest["metrics"]) == 3


@pytest.mark.parametrize("fault", ["missing_seed", "missing_weights", "cpu", "state",
                                    "features", "missing_support", "bad_json", "nan_profit"])
def test_terminal_artifact_falsifications_rejected(tmp_path, fault):
    w = module()
    c, job = cfg(tmp_path), w.jobs()[0]
    out = artifacts(w, c, job)
    r = json.loads((out / "E3M.json").read_text())
    if fault == "missing_seed":
        r["trials"]["FULL"].pop(str(c["seeds"][0]))
    elif fault == "missing_weights":
        next(out.glob("*.pt")).unlink()
    elif fault == "cpu":
        r["device"] = "cpu"
    elif fault == "state":
        r["agent_state"]["mode"] = "none"
    elif fault == "features":
        p = out / "SCALER.json"
        d = json.loads(p.read_text()); d["features"].reverse(); p.write_text(json.dumps(d))
    elif fault == "missing_support":
        r["meta"]["support_hash"].pop("TRAIN")
    elif fault == "nan_profit":
        r["trials"]["FULL"][str(c["seeds"][0])]["trading"]["TRAIN"]["net_profit_usd"] = float("nan")
    (out / "E3M.json").write_text("{" if fault == "bad_json" else json.dumps(r))
    with pytest.raises((ValueError, KeyError)):
        w.validate_result(c, job)


def test_authenticated_completion_adopted_without_fit(tmp_path):
    w = module()
    c, job = cfg(tmp_path), w.jobs()[0]
    artifacts(w, c, job)
    receipt = w.seal_result(c, job)
    assert w.adopt(c, job) == receipt
    (tmp_path / "out" / job["name"] / "FULL_open_l_20261019.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="RECEIPT"):
        w.adopt(c, job)


def test_refusal_after_output_is_not_counted_as_unentered(tmp_path):
    w = module()
    out = tmp_path / "out"
    assert w.resource_refusal(75, "crispdm-run: REFUSED", out)
    out.mkdir(); (out / "SCALER.json").write_text("{}")
    assert not w.resource_refusal(75, "crispdm-run: REFUSED", out)
    assert not w.resource_refusal(1, "REFUSED", tmp_path / "missing")


def test_only_actual_package_child_prevents_duplicate(tmp_path):
    w = module()
    root = str(tmp_path)
    assert w.owns_child(["/env/python", "tools/expert_guided_e3m.py", "run", "--out", root + "/out/a"], root)
    assert not w.owns_child(["bash", "-c", "expert_guided_e3m.py --out " + root], root)
    assert not w.owns_child(["/env/python", "tools/expert_guided_e3m.py", "run", "--out", "/other/out/a"], root)


def test_failed_independent_arm_is_terminal_not_whole_queue_success():
    w = module()
    states = {j["name"]: {"state": "DONE"} for j in w.jobs()}
    states[w.jobs()[0]["name"]] = {"state": "FAILED", "cause": "synthetic"}
    summary = w.closure(states)
    assert summary["state"] == "COMPLETE_WITH_FAILURES"
    assert summary["done"] == 11 and summary["failed"] == 1 and summary["pending"] == 0


def test_real_worker_loop_continues_after_bounded_failure_and_closes(tmp_path, monkeypatch):
    w = module()
    c = cfg(tmp_path)
    (tmp_path / "code").mkdir()
    calls = []

    class Child:
        pid = 123

        def __init__(self, cmd, **kwargs):
            job = next(j for j in w.jobs() if str(tmp_path / "out" / j["name"]) in cmd)
            calls.append(job["name"])
            self.returncode = 1 if job == w.jobs()[0] else 0
            if self.returncode == 0:
                artifacts(w, c, job)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(w.subprocess, "Popen", Child)
    monkeypatch.setattr(w, "ready_gpu", lambda c: True)
    monkeypatch.setattr(w, "live_children", lambda root: [])
    monkeypatch.setattr(w.time, "sleep", lambda t: None)
    c["min_disk_bytes"] = 0
    w.run(c)
    closed = json.loads((tmp_path / "ops/CLOSE.json").read_text())
    assert closed["state"] == "COMPLETE_WITH_FAILURES"
    assert closed["done"] == 11 and closed["failed"] == 1
    assert calls.count(w.jobs()[0]["name"]) == 2
    before = list(calls)
    w.run(c)
    assert calls == before
    assert len(json.loads((tmp_path / "CONSUMERS.json").read_text())["results"]) == 11


def test_reboot_after_scientific_completion_adopts_without_second_fit(tmp_path, monkeypatch):
    w = module()
    c = cfg(tmp_path)
    for job in w.jobs():
        artifacts(w, c, job)
    states = {j["name"]: {"state": "RUNNING", "entered": 1} for j in w.jobs()}
    w.atomic(tmp_path / "ops/JOBS.json", states)
    monkeypatch.setattr(w, "live_children", lambda root: [])
    monkeypatch.setattr(w.subprocess, "Popen", lambda *a, **k: pytest.fail("completed job refitted"))
    w.run(c)
    assert json.loads((tmp_path / "ops/CLOSE.json").read_text())["done"] == 12


def test_runtime_drift_fails_before_fit(tmp_path, monkeypatch):
    w = module()
    c = cfg(tmp_path)
    c["runtime_versions"] = {"torch": "pinned"}
    monkeypatch.setattr(w, "runtime_versions", lambda python: {"torch": "changed"})
    with pytest.raises(ValueError, match="RUNTIME"):
        w.verify_pins(c)


def test_recovery_timer_targets_one_service_without_parallel_dispatch():
    root = Path(__file__).parents[1]
    timer = (root / "deploy/expert-package-worker.timer").read_text()
    service = (root / "deploy/expert-package-worker.service").read_text()
    assert "Unit=musashi-e3m-assigned-20261022.service" in timer
    assert "OnBootSec=" in timer and "OnUnitInactiveSec=" in timer
    assert "Restart=on-failure" in service
    assert "MemoryMax=256M" in service  # admission launches the science in its own 4GiB scope


def test_recovery_scope_contains_only_requested_failed_arms():
    w = module()
    names = ["cons_shortlist", "cons_shortlist_strict_enc"]
    selected = w.planned_jobs({"only_jobs": names})
    assert [j["name"] for j in selected] == names
    closed = w.closure({n: {"state": "DONE"} for n in names}, selected)
    assert closed == {"state": "COMPLETE", "done": 2, "failed": 0, "pending": 0, "total": 2}
    with pytest.raises(ValueError):
        w.planned_jobs({"only_jobs": ["made-up"]})


def test_scientific_recovery_scope_does_not_change_model_flags(tmp_path):
    w = module()
    c = cfg(tmp_path)
    selected = w.planned_jobs({"only_jobs": ["cons_shortlist"]})[0]
    previous = w.command(c, selected)
    c["memory"] = "4812M"
    current = w.command(c, selected)
    index = previous.index("-m") + 1
    assert previous[:index] == current[:index] and previous[index + 1:] == current[index + 1:]
