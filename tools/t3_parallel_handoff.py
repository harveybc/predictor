#!/usr/bin/env python3
"""Preserve one in-flight T3 unit, hand off its tail and index disjoint shards."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import expert_package_worker as w
from collect_expert_package import validate_mirror


def partition():
    return ([f"t3_unit_{i:02d}" for i in range(1, 15, 2)],
            [f"t3_unit_{i:02d}" for i in range(15, 43, 2)])


def owns_manager(argv, root):
    root = Path(root)
    return (len(argv) == 5 and Path(argv[0]).name.startswith("python")
            and Path(argv[1]) == root / "expert_package_worker.py" and argv[2:4] == ["run", "--root"]
            and Path(argv[4]) == root)


def checked_rows(rows):
    expected = set(sum(partition(), []))
    result = {}
    for row in rows:
        name = row["unit"]
        if name not in expected:
            raise ValueError("UNEXPECTED_PARALLEL_UNIT")
        if name in result:
            raise ValueError("DUPLICATE_PARALLEL_UNIT")
        result[name] = row
    return result


def argv_of(pid):
    return [v.decode() for v in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0") if v]


def pause(cfg):
    root = Path(cfg["source"])
    scientific = w.read(root / "ops/CONFIG.private.json")
    w.verify_pins(scientific)
    status = w.read(root / "ops/STATUS.json")
    if status.get("current_job") != "t3_unit_01":
        raise ValueError("SOURCE_CURSOR_CHANGED_REPLAN_REQUIRED")
    pid = int(subprocess.check_output(["systemctl", "--user", "show", cfg["source_service"],
                                      "-p", "MainPID", "--value"], text=True))
    if not owns_manager(argv_of(pid), root):
        raise ValueError("NOT_THE_OWNED_MANAGER")
    context = {"manager_pid": pid, "manager_identity": w.process_identity(pid),
               "launcher_pid": status.get("launcher_pid"), "source_config_sha256": w.identity(scientific),
               "paused_at": time.time(), "native_children_not_signalled": w.live_children(root)}
    if context["launcher_pid"] and Path(f"/proc/{context['launcher_pid']}").exists():
        context["launcher_identity"] = w.process_identity(context["launcher_pid"])
    os.kill(pid, signal.SIGSTOP)
    subprocess.run(["systemctl", "--user", "disable", "--now", scientific["timer_unit"]], check=True)
    w.atomic(Path(cfg["control"]) / "SOURCE_PAUSED.private.json", context)
    print(json.dumps({"manager_paused": True, "scientific_children_preserved": context["native_children_not_signalled"]}))


def launcher_live(context):
    pid = context.get("launcher_pid")
    if not pid or not Path(f"/proc/{pid}").exists():
        return False
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] == "Z":
        return False
    return w.process_identity(pid) == context.get("launcher_identity")


def handoff(cfg):
    control, root, target = Path(cfg["control"]), Path(cfg["source"]), Path(cfg["slow_root"])
    if (control / "HANDOFF.json").exists():
        return True
    context = w.read(control / "SOURCE_PAUSED.private.json")
    if not context:
        raise ValueError("SOURCE_NOT_PAUSED")
    if w.live_children(root) or launcher_live(context):
        w.atomic(control / "STATUS.json", {"state": "WAITING_FOR_INFLIGHT_UNIT", "source": "t3_unit_01"})
        return False
    original = w.read(root / "ops/CONFIG.private.json")
    if w.identity(original) != context["source_config_sha256"]:
        raise ValueError("SOURCE_CONFIG_CHANGED")
    job = next(j for j in w.planned_jobs(original) if j["name"] == "t3_unit_01")
    states = w.read(root / "ops/JOBS.json")
    before = dict(states)
    try:
        w.seal_result(original, job)
        states[job["name"]] = dict(states[job["name"]], state="DONE", operator_harvested_after_native_exit=True)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        states[job["name"]] = dict(states[job["name"]], state="FAILED", cause="INFLIGHT_NOT_COMPLETE:" + str(exc))
    pid = context["manager_pid"]
    if Path(f"/proc/{pid}").exists() and w.process_identity(pid) == context["manager_identity"]:
        if not owns_manager(argv_of(pid), root):
            raise ValueError("MANAGER_OWNERSHIP_CHANGED")
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    subprocess.run(["systemctl", "--user", "stop", cfg["source_service"]], check=True)
    if w.live_children(root):
        raise ValueError("SOURCE_CHILD_STILL_ALIVE")
    w.atomic(control / "JOBS_BEFORE_HANDOFF.private.json", before)
    w.atomic(root / "ops/JOBS.json", states)
    counts = w.closure(states, w.planned_jobs(original))
    w.publish_status(original, states, "STOPPED_FOR_RELOCATION", cause="DISJOINT_PARALLEL_HANDOFF")
    w.atomic(root / "ops/RELOCATION_STOP.private.json", {
        "state": "STOPPED_FOR_OWNER_REQUESTED_RELOCATION", "parent_config_sha256": w.identity(original),
        "prior_status": counts, "remaining_scientific_children": [], "reason": "DISJOINT_PARALLEL_HANDOFF"})
    shutil.copytree(root / "ops", target / "source/ops", dirs_exist_ok=True)
    slow, _ = partition()
    completed = {w.read(p)["job"]["name"] for p in (root / "ops").glob("*.RESULT.json")}
    selected = [n for n in slow if n not in completed]
    cmd = ["python3", str(target / "prepare_expert_package_t3.py"), "--root", str(target),
           "--python", cfg["python"], "--launcher", cfg["launcher"], "--templates", cfg["templates"],
           "--relocate", "--gpu-model", "4070", "--memory", "5G", "--jobs", ",".join(selected),
           "--unit-stem", "musashi-e3m-t3-parallel-slow-20261023"]
    if cfg.get("desktop_pid") and Path(f"/proc/{cfg['desktop_pid']}").exists():
        cmd += ["--desktop-pid", str(cfg["desktop_pid"])]
    subprocess.run(cmd, check=True)
    subprocess.run(["systemctl", "--user", "start", cfg["source_collector"] + ".service"], check=True)
    subprocess.run(["systemctl", "--user", "disable", "--now", cfg["source_collector"] + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", cfg["slow_collector"] + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", cfg["slow_collector"] + ".service"], check=True)
    w.atomic(control / "HANDOFF.json", {"source_completed": sorted(completed), "slow_selected": selected,
                                       "source_configuration": w.identity(original), "timestamp": time.time()})
    return True


def index(cfg):
    rows, failures = [], set()
    slow, fast = partition()
    tail_cfg = w.read(Path(cfg["slow_root"]) / "ops/CONFIG.private.json", {})
    tail_repairs_first = "t3_unit_01" in tail_cfg.get("only_jobs", [])
    for origin, root, allowed in (("inflight", Path(cfg["source"]), {"t3_unit_01"}),
                                 ("slow", Path(cfg["slow_root"]), set(slow)),
                                 ("fast", Path(cfg["fast_mirror"]), set(fast))):
        if not (root / "ops/CONFIG.private.json").exists():
            continue
        validate_mirror(root)
        for p in (root / "ops").glob("*.RESULT.json"):
            receipt = w.read(p)
            name = receipt["job"]["name"]
            if name not in allowed:
                raise ValueError("RESULT_OUTSIDE_ITS_SHARD")
            rows.append({"unit": name, "origin": origin, "receipt": str(p), "receipt_sha256": w.digest(p),
                         "config_sha256": receipt["config_sha256"], "output": str(root / "out" / name)})
        for name, rec in w.read(root / "ops/JOBS.json", {}).items():
            if name in allowed and rec.get("state") == "FAILED":
                if origin != "inflight" or not tail_repairs_first:
                    failures.add(name)
    checked = checked_rows(rows)
    failures -= set(checked)
    pending = 21 - len(checked) - len(failures)
    doc = {"done": len(checked), "failed": len(failures), "pending": pending, "total": 21,
           "failed_units": sorted(failures), "units": checked, "metrics_recomputed": False,
           "scientific_selection": False}
    coverage = Path(cfg["control"]) / "COVERAGE.json"
    w.atomic(coverage, doc)
    if cfg.get("coverage_destination"):
        published = Path(cfg["control"]) / "COVERAGE_PUBLISHED.json"
        current = w.digest(coverage)
        if w.read(published, {}).get("sha256") != current:
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                            str(coverage), cfg["coverage_destination"]], check=True, timeout=45)
            w.atomic(published, {"sha256": current})
    if not pending:
        subprocess.run(["systemctl", "--user", "disable", "--now", cfg["timer"]], check=True)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("pause", "tick"))
    p.add_argument("--config", required=True)
    a = p.parse_args()
    config = w.read(a.config)
    if a.mode == "pause":
        pause(config)
    else:
        ready = handoff(config)
        print(json.dumps({"handoff_complete": ready, "coverage": index(config)}))
