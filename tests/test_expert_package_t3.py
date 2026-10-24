import hashlib
import importlib.util
import shlex
import sys
from pathlib import Path

import pytest


def worker():
    p = Path(__file__).parents[1] / "tools/expert_package_worker.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("t3_worker", p)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w


def package():
    units, entries = [], []
    for i in range(43):
        name = f"t3_unit_{i:02d}"
        u = {"id": i, "name": name, "group": "group", "kind": "pointwise",
             "identity": f"identity-{i}", "groups_file": f"unit_{i:02d}.json",
             "groups_file_sha256": "group-digest", "worker": "musashi" if i % 2 else "satoshi"}
        entries.append(u)
        if not i % 2:
            continue
        c = ("tools/expert_guided_e3m.py run --admitted <root>/data/admitted --csv-dir <root>/data/sel15 "
             "--store <root>/data/full_pool_s/full_pool_eurusd_h1.parquet "
             "--feature-list <root>/in/FEATURE_LIST_shortlist_strict.json "
             f"--groups <root>/t3/groups/unit_{i:02d}.json --out <root>/out/{name} "
             "--variants FULL --units 2 --depth 2 --head 16 --dropout 0.1 --wd 0.0001 --max-epochs 1000 "
             "--scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none "
             "--window 13 --stride 2 --exec-contract sac --min-orders-per-week 2.0")
        units.append(dict(u, command=c, command_identity=hashlib.sha256(c.encode()).hexdigest()[:16]))
    return {"schema": "t3_manifest.v2", "identity": "t3v2_8875abf84a63", "units": entries}, {
        "manifest": "t3v2_8875abf84a63", "worker": "musashi", "units": units}


def test_t3_uses_exact_peer_commands_and_only_odd_units(tmp_path):
    w = worker()
    manifest, units = package()
    cfg = {"campaign": "v4_t3", "root": str(tmp_path), "t3_manifest": manifest, "t3_units": units,
           "python": "/python", "launcher": "/admit", "memory": "4G"}
    jobs = w.planned_jobs(cfg)
    assert len(jobs) == 21
    assert [j["unit_id"] for j in jobs] == list(range(1, 43, 2))
    for j, u in zip(jobs, units["units"]):
        command = w.command(cfg, j)
        assert command[command.index("--") + 2:] == shlex.split(u["command"].replace("<root>", str(tmp_path)))


def package_v5():
    import copy
    manifest, assigned = package()
    manifest["identity"] = assigned["manifest"] = "t3v2_7db1953f99ad"
    for i in (43, 44):
        u = copy.deepcopy(manifest["units"][1 if i % 2 else 2])
        u.update(id=i, name=f"t3_unit_{i:02d}", groups_file=f"unit_{i:02d}.json", identity=f"identity-{i}")
        manifest["units"].append(u)
        if i % 2:
            new = dict(u, command=assigned["units"][0]["command"].replace("unit_01", f"unit_{i:02d}"))
            assigned["units"].append(new)
    for u in assigned["units"]:
        u["command"] = u["command"].replace("full_pool_s/", "full_pool_s_a1/").replace("--dropout 0.1", "--dropout 0.3").replace("--wd 0.0001", "--wd 0.01")
        u["command_identity"] = hashlib.sha256(u["command"].encode()).hexdigest()[:16]
    return manifest, assigned


def test_v5_manifest_keeps_new_science_and_disjoint_fast_slow_shards():
    w = worker()
    manifest, assigned = package_v5()
    cfg = {"campaign": "v4_t3", "t3_manifest": manifest, "t3_units": assigned}
    all_jobs = w.planned_jobs(cfg)
    assert len(all_jobs) == 22
    assert all(j["dropout"] == .3 and j["wd"] == .01 for j in all_jobs)
    slow = [j["name"] for j in all_jobs[:3]]
    fast = [j["name"] for j in all_jobs[3:]]
    assert len(w.planned_jobs(dict(cfg, only_jobs=slow))) == 3
    assert len(w.planned_jobs(dict(cfg, only_jobs=fast))) == 19
    assert not set(slow) & set(fast)
    for u in assigned["units"]:
        assert next(j["peer_command"] for j in all_jobs if j["name"] == u["name"]) == u["command"]


def test_v5_refuses_old_regularization_even_with_updated_command_hash():
    w = worker()
    manifest, assigned = package_v5()
    u = assigned["units"][0]
    u["command"] = u["command"].replace("--dropout 0.3", "--dropout 0.1")
    u["command_identity"] = hashlib.sha256(u["command"].encode()).hexdigest()[:16]
    with pytest.raises(ValueError, match="T3_COMMAND_CONTRACT"):
        w.planned_jobs({"campaign": "v4_t3", "t3_manifest": manifest, "t3_units": assigned})


def test_v5_deployment_partition_and_pin_failure(tmp_path):
    from prepare_expert_package_t3_v5 import shard, verify
    assert len(shard("fast")) == 19 and len(shard("slow")) == 3
    assert sorted(shard("fast") + shard("slow")) == [f"t3_unit_{i:02d}" for i in range(1, 45, 2)]
    with pytest.raises(ValueError, match="UNKNOWN_ROLE"):
        shard("other")
    (tmp_path / "PINS.sha256").write_text("modified")
    with pytest.raises(ValueError, match="PACKAGE_PIN_CHANGED"):
        verify(tmp_path)


@pytest.mark.parametrize("fault", ["hash", "duplicate", "feedback", "split", "window", "groups", "extra"])
def test_t3_rejects_changed_or_resealed_commands(fault):
    w = worker()
    manifest, units = package()
    unit = units["units"][0]
    if fault == "hash":
        unit["command_identity"] = "bad"
    elif fault == "duplicate":
        units["units"][1] = unit
    else:
        unit["command"] = {"feedback": unit["command"].replace("--agent-state none", "--agent-state full"),
                           "split": unit["command"].replace("consolidated", "dev3y"),
                           "window": unit["command"].replace("--window 13", "--window 25"),
                           "groups": unit["command"].replace("unit_01.json", "unit_02.json"),
                           "extra": unit["command"] + " --labels opportunity"}[fault]
        unit["command_identity"] = hashlib.sha256(unit["command"].encode()).hexdigest()[:16]
    with pytest.raises(ValueError):
        w.planned_jobs({"campaign": "v4_t3", "t3_manifest": manifest, "t3_units": units})


def test_t3_group_layout_is_checked_against_consumed_tensor(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from expert_package_t3 import validate_extra
    w = worker()
    cfg = {"root": str(tmp_path)}
    spec = {"groups": {"a": ["x", "y"]}, "units": {"a": 2}, "kinds": {"a": "pointwise"}}
    path = tmp_path / "t3/groups/unit_01.json"
    w.atomic(path, spec)
    job = {"groups_file": "t3/groups/unit_01.json", "groups_sha256": w.digest(path)}
    result = {"arch": {"depth": 2, "group_names": ["a"], "groups": [[0, 1]],
                       "group_units": [2], "group_kinds": ["pointwise"]},
              "window": {"observations": 13, "stride_rows": 2}, "labels": {"mode": "zigzag"}}
    validate_extra(cfg, job, result, ["x@diff", "y"])
    result["arch"]["group_kinds"] = ["conv"]
    with pytest.raises(ValueError, match="T3_GROUP_LAYOUT"):
        validate_extra(cfg, job, result, ["x@diff", "y"])


def test_t3_mismatched_support_is_not_sealed(tmp_path, monkeypatch):
    w = worker()
    manifest, units = package()
    cfg = {"campaign": "v4_t3", "root": str(tmp_path), "pins": {},
           "t3_manifest": manifest, "t3_units": units}
    jobs = w.planned_jobs(cfg)
    w.atomic(tmp_path / "ops/old.RESULT.json", {"job": jobs[0], "support_hash": {"TRAIN": "old"}})
    monkeypatch.setattr(w, "validate_result", lambda *a: {"job": jobs[1], "support_hash": {"TRAIN": "new"}})
    with pytest.raises(ValueError, match="PAIRED_SUPPORT"):
        w.seal_result(cfg, jobs[1])


def test_t3_shard_is_an_explicit_subset_without_changed_commands(tmp_path):
    w = worker()
    manifest, units = package()
    c = {"campaign": "v4_t3", "t3_manifest": manifest, "t3_units": units,
         "only_jobs": ["t3_unit_15", "t3_unit_17"]}
    assert [j["name"] for j in w.planned_jobs(c)] == c["only_jobs"]
    for bad in ([], ["t3_unit_15", "t3_unit_15"], ["t3_unit_02"]):
        c["only_jobs"] = bad
        with pytest.raises(ValueError):
            w.planned_jobs(c)


def test_peer_even_assignment_requires_explicit_owner_and_exact_commands():
    import copy
    from expert_package_t3 import jobs
    manifest, original = package()
    template = original["units"][0]["command"]
    assigned = {"manifest": manifest["identity"], "worker": "satoshi", "units": []}
    manifest["units"][0]["kind"] = "conv"
    for u in manifest["units"][::2]:
        command = template.replace("unit_01", f"unit_{u['id']:02d}")
        assigned["units"].append(dict(u, command=command,
            command_identity=hashlib.sha256(command.encode()).hexdigest()[:16]))
    result = jobs(manifest, assigned, worker="satoshi")
    assert [j["unit_id"] for j in result] == list(range(0, 43, 2))
    assert result[0]["kind"] == "conv"
    with pytest.raises(ValueError):
        jobs(manifest, assigned)
    bad = copy.deepcopy(assigned)
    bad["units"][1]["command"] += " --agent-state full"
    bad["units"][1]["command_identity"] = hashlib.sha256(bad["units"][1]["command"].encode()).hexdigest()[:16]
    with pytest.raises(ValueError):
        jobs(manifest, bad, worker="satoshi")
    with pytest.raises(ValueError):
        jobs(manifest, assigned, worker="other")


def test_combination_changes_only_groups_and_output_and_binds_the_decision(tmp_path):
    import json
    from expert_package_t3 import confirmation_job
    w = worker()
    manifest, assigned = package()
    base = {"groups": {"group": ["x", "y"]}, "units": {"group": 2}, "kinds": {"group": "conv"}}
    decision = {"manifest": manifest["identity"], "groups": [{"group": "group", "chosen": "pointwise"}]}
    combined = {"name": "t3_unit_C", "rule": "combination", **base, "kinds": {"group": "pointwise"}}
    w.atomic(tmp_path / "in/T3_DECISION.json", decision)
    w.atomic(tmp_path / "in/T3_CONTROL.json", {})
    ready = {"state": "READY_FOR_PER_GROUP_DECISION_AUDIT", "count": 43,
             "units": {"t3_unit_00": {"result_sha256": w.digest(tmp_path / "in/T3_CONTROL.json")}}}
    w.atomic(tmp_path / "in/T3_READY.json", ready)
    w.atomic(tmp_path / "t3/groups/unit_00.json", base)
    w.atomic(tmp_path / "t3/groups/unit_C.json", combined)
    c = {"root": str(tmp_path), "campaign": "v4_t3_confirmation", "t3_manifest": manifest, "t3_units": assigned,
         "confirmation": {"decision_sha256": w.digest(tmp_path / "in/T3_DECISION.json"),
                          "readiness_sha256": w.digest(tmp_path / "in/T3_READY.json")}}
    job = confirmation_job(c)
    assert job["name"] == "t3_unit_C"
    expected = shlex.split(assigned["units"][0]["command"])
    actual = shlex.split(job["peer_command"])
    changed = [i for i,(a,b) in enumerate(zip(expected,actual)) if a != b]
    assert changed == [expected.index("--groups") + 1, expected.index("--out") + 1]
    c.update(python="/python", launcher="/admit", memory="5G")
    assert w.planned_jobs(c) == [job]
    assert "--agent-state" in w.command(c, job)
    combined["kinds"]["group"] = "conv"
    w.atomic(tmp_path / "t3/groups/unit_C.json", combined)
    with pytest.raises(ValueError, match="COMBINATION_GROUPS"):
        confirmation_job(c)
    w.atomic(tmp_path / "in/T3_DECISION.json", dict(decision, extra="changed"))
    with pytest.raises(ValueError, match="COMBINATION_DECISION"):
        confirmation_job(c)
