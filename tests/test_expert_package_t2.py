import importlib.util
import hashlib
import shlex
from pathlib import Path

import pytest


def module():
    p = Path(__file__).parents[1] / "tools/expert_package_worker.py"
    s = importlib.util.spec_from_file_location("t2_worker", p)
    w = importlib.util.module_from_spec(s)
    s.loader.exec_module(w)
    return w


def manifest():
    units = []
    for name in ("core15", "shortlist_strict"):
        for state in ("none", "full"):
            for reg, drop, wd in (("base", "0.1", "1e-4"), ("strong", "0.3", "1e-2")):
                n = f"gap_{name}_{state}_{reg}"
                c = ("tools/expert_guided_e3m.py run --admitted <musashi-root>/data/admitted "
                     "--csv-dir <musashi-root>/data/sel15 --store <musashi-root>/data/full_pool_s/full_pool_eurusd_h1.parquet "
                     f"--feature-list <musashi-root>/in/FEATURE_LIST_{name}.json --out <musashi-root>/out/{n} "
                     "--units 2 --head 16 --variants FULL --max-epochs 1000 --scaling fixed_train --nonstationary diff "
                     f"--agent-state {state} --dropout {drop} --wd {wd} --exec-contract sac "
                     "--min-orders-per-week 2.0 --split-contract consolidated")
                units.append({"order": len(units) + 1, "name": n, "list": name, "agent_state": state,
                              "dropout": float(drop), "wd": float(wd), "regularization": reg,
                              "command": c, "identity": hashlib.sha256(c.encode()).hexdigest()[:16]})
    return {"units": units}


def test_t2_exact_eight_supplied_commands_and_no_legacy_defaults(tmp_path):
    w = module()
    c = {"campaign": "v4_t2", "root": str(tmp_path), "python": "/python", "launcher": "/admit",
         "memory": "4G", "t2_manifest": manifest()}
    selected = w.planned_jobs(c)
    assert len(selected) == 8
    assert selected[0]["name"] == "gap_core15_none_base"
    for j, u in zip(selected, c["t2_manifest"]["units"]):
        cmd = w.command(c, j)
        assert cmd[cmd.index("--") + 2:] == shlex.split(u["command"].replace("<musashi-root>", str(tmp_path)))
        assert j["split"] == "consolidated"


@pytest.mark.parametrize("fault", ["hash", "duplicates", "native", "old_split", "extra_flag", "escape"])
def test_t2_rejects_changed_or_resealed_command_contract(fault):
    w = module()
    m = manifest()
    if fault == "hash":
        m["units"][0]["identity"] = "bad"
    elif fault == "duplicates":
        m["units"][1] = m["units"][0]
    else:
        u = m["units"][0]
        u["command"] = {"native": u["command"].replace("--exec-contract sac", "--exec-contract native"),
                         "old_split": u["command"].replace("--split-contract consolidated", "--split-contract dev3y"),
                         "extra_flag": u["command"] + " --no-verify true",
                         "escape": u["command"].replace("<musashi-root>/out/", "/outside/")}[fault]
        u["identity"] = hashlib.sha256(u["command"].encode()).hexdigest()[:16]
    with pytest.raises(ValueError):
        w.planned_jobs({"campaign": "v4_t2", "t2_manifest": m})


@pytest.mark.parametrize("field,value", [("exec_contract", "native"), ("min_orders_per_week", 3),
                                         ("labels", {"mode": "opportunity"})])
def test_t2_native_results_cannot_silently_change_execution_or_labels(tmp_path, field, value):
    w = module()
    job = w.t2_jobs(manifest())[0]
    out = tmp_path / "out" / job["name"]
    r = {"exec_contract": "sac", "min_orders_per_week": 2.0}
    r[field] = value
    for name, data in (("E3M.json", r), ("SCALER.json", {}), ("PROBE.json", {})):
        w.atomic(out / name, data)
    with pytest.raises(ValueError, match="T2_"):
        w.validate_result({"campaign": "v4_t2", "root": str(tmp_path)}, job)


def test_t2_support_mismatch_never_gets_a_success_receipt(tmp_path, monkeypatch):
    w = module()
    job = w.t2_jobs(manifest())[1]
    receipt = {"job": job, "support_hash": {"TRAIN": "new", "INNER_A": "same"}}
    old = {"job": w.t2_jobs(manifest())[0], "support_hash": {"TRAIN": "old", "INNER_A": "same"}}
    w.atomic(tmp_path / "ops/previous.RESULT.json", old)
    monkeypatch.setattr(w, "validate_result", lambda *a: receipt)
    with pytest.raises(ValueError, match="PAIRED_SUPPORT"):
        w.seal_result({"campaign": "v4_t2", "root": str(tmp_path), "pins": {}}, job)
    assert not (tmp_path / "ops" / (job["name"] + ".RESULT.json")).exists()
