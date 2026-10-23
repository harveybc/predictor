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
