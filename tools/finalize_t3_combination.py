#!/usr/bin/env python3
"""Apply the frozen combined-representation rule and replay its saved probes."""
import argparse
import math
from pathlib import Path
import statistics
import subprocess
import shlex

from collect_expert_package import validate_mirror
import expert_package_worker as w


SEEDS = {"20261019", "20261020", "20261021"}


def arm(result):
    trials = result["trials"]["FULL"]
    if set(trials) != SEEDS:
        raise ValueError("COMBINATION_FINAL_SEEDS")
    sh = [trials[s]["trading"]["INNER_A"]["sharpe_daily_sqrt365"] for s in sorted(SEEDS)]
    train = [trials[s]["trading"]["TRAIN"]["sharpe_daily_sqrt365"] for s in sorted(SEEDS)]
    if any(v is None or not math.isfinite(v) for v in sh + train):
        raise ValueError("COMBINATION_FINAL_NONFINITE")
    valid = all(trials[s]["trading"]["INNER_A"]["valid"] for s in SEEDS)
    if valid != result["decision"]["means"]["FULL"]["all_valid"]:
        raise ValueError("COMBINATION_FINAL_VALIDITY")
    return {"inner_sharpe_mean": statistics.mean(sh), "inner_sharpe_std": statistics.pstdev(sh),
            "train_sharpe_mean": statistics.mean(train),
            "train_inner_gap_mean": statistics.mean(a - b for a, b in zip(train, sh)), "all_valid": valid}


def decide(control, combined):
    if control["meta"]["support_hash"] != combined["meta"]["support_hash"]:
        raise ValueError("COMBINATION_FINAL_SUPPORT")
    a, b = arm(control), arm(combined)
    pooled = math.sqrt((a["inner_sharpe_std"] ** 2 + b["inner_sharpe_std"] ** 2) / 2)
    selected = a["all_valid"] and b["all_valid"] and b["inner_sharpe_mean"] >= a["inner_sharpe_mean"] - pooled
    return {"state": "COMBINATION_SELECTED" if selected else "CONTROL_RETAINED", "control": a, "combined": b,
            "pooled_seed_std": pooled, "rule": "all-valid; combined INNER_A mean >= control mean minus pooled seed std",
            "selection_reads": ["TRAIN", "INNER_A"], "test_read": False, "scientific_release": False}


def tick(cfg):
    root = Path(cfg["mirror"])
    closed = w.read(root / "ops/CLOSE.json")
    if not closed:
        return {"state": "WAITING_FOR_COMBINATION"}
    status = validate_mirror(root)
    if closed.get("failed"):
        failure = {"state": "COMBINATION_EXECUTION_FAILED", "failed": closed["failed"], "scientific_release": False}
        w.atomic(cfg["output"], failure)
        subprocess.run(["scp", "-q", cfg["output"], cfg["destination"]], check=True, timeout=60)
        subprocess.run(["systemctl", "--user", "disable", "--now", cfg["timer"]], check=True)
        return failure
    if not status["closed"] or status["count"] != 1:
        raise ValueError("COMBINATION_FINAL_CLOSURE")
    replica = w.read(root / "REMOTE_REPLICA_STATUS.json", {})
    if not replica.get("verified") or not replica.get("closed") or replica.get("count") != 1:
        return {"state": "WAITING_FOR_REPLICA"}
    producer = w.read(root / "ops/CONFIG.private.json")
    if w.identity(producer) != cfg["expected_config_sha256"]:
        raise ValueError("COMBINATION_FINAL_PRODUCER")
    for relative in ("in/T3_READY.json", "in/T3_DECISION.json", "in/T3_CONTROL.json", "t3/groups/unit_C.json"):
        if w.digest(root / relative) != producer["pins"][str(Path(producer["root"]) / relative)]:
            raise ValueError("COMBINATION_FINAL_PIN")
    run = root / "out/t3_unit_C"
    control, combined = w.read(root / "in/T3_CONTROL.json"), w.read(run / "E3M.json")
    result = decide(control, combined)
    result["native_metrics_reported_after_decision"] = {
        label: {p: {s: e["trials"]["FULL"][s]["trading"][p] for s in sorted(SEEDS)}
                for p in ("TRAIN", "INNER_A", "OBS_2024")}
        for label, e in (("control", control), ("combined", combined))}
    receipt = w.read(root / "ops/t3_unit_C.RESULT.json")
    result.update(schema="t3_combination_decision.v1", producer_config_sha256=w.identity(producer),
                  receipt_sha256=w.digest(root / "ops/t3_unit_C.RESULT.json"),
                  combined_result_sha256=receipt["hashes"]["E3M.json"], full_trading_metrics_replayed=False)
    replay_file = Path(cfg["replay_output"])
    plan = {"source_root": cfg["source_root"], "trading_replay": True,
            "units": {"t3_unit_C": {"run": str(run), "hashes": receipt["hashes"]}}}
    if not replay_file.exists():
        if cfg.get("replay_host"):
            plan["source_root"] = producer["root"]
            plan["units"]["t3_unit_C"]["run"] = str(Path(producer["root"]) / "out/t3_unit_C")
        w.atomic(cfg["replay_plan"], plan)
        if cfg.get("replay_host"):
            host = cfg["replay_host"]
            subprocess.run(["scp", "-q", cfg["replay_plan"], host + ":" + cfg["remote_replay_plan"]], check=True, timeout=60)
            command = ["systemd-run", "--user", "--wait", "--collect", "--quiet",
                       "--unit=musashi-t3-combination-probe-replay-20261023", "-p", "MemoryMax=2G",
                       "-p", "MemorySwapMax=0", "-p", "CPUQuota=100%", "-p", "Nice=15",
                       "--setenv=CUDA_VISIBLE_DEVICES=", "--setenv=OMP_NUM_THREADS=1",
                       "--setenv=OPENBLAS_NUM_THREADS=1", "--setenv=MKL_NUM_THREADS=1",
                       cfg["python"], cfg["remote_replay_script"], "--plan", cfg["remote_replay_plan"], "--out", cfg["remote_replay_output"]]
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(command)],
                                  check=False, timeout=1850)
            subprocess.run(["scp", "-q", host + ":" + cfg["remote_replay_output"], str(replay_file)], check=False, timeout=60)
        else:
            proc = subprocess.run([cfg["python"], cfg["replay_script"], "--plan", cfg["replay_plan"], "--out", str(replay_file)],
                                  check=False, timeout=1800)
        result["checkpoint_replay_returncode"] = proc.returncode
    replay = w.read(replay_file, {})
    if replay and replay.get("units", {}).get("t3_unit_C", {}).get("result_sha256") != receipt["hashes"]["E3M.json"]:
        raise ValueError("COMBINATION_FINAL_STALE_REPLAY")
    if replay.get("state") == "PASS":
        unit = replay["units"]["t3_unit_C"]
        expected_modules = {f"FULL_{m}_{s}" for m in w.MODULES for s in SEEDS}
        if (replay.get("schema") != "t3_checkpoint_probe_replay.v1" or replay.get("count") != 1
                or replay.get("checkpoints") != 12 or replay.get("torch_cpu_threads") != 1
                or not unit.get("exact") or set(unit.get("modules", {})) != expected_modules
                or not all(v[k].get("exact") for v in unit["modules"].values() for k in ("logits", "reg"))):
            raise ValueError("COMBINATION_FINAL_REPLAY_CONTRACT")
    result["checkpoint_probe_replay"] = replay.get("state", "FAILED_WITHOUT_REPORT")
    result["full_trading_metrics_replayed"] = replay.get("full_trading_metrics_replayed", False)
    result["full_trading_metrics_exact"] = replay.get("units", {}).get("t3_unit_C", {}).get("trading_exact", False)
    result["frozen_rule_decision"] = result["state"]
    result["engineering_verified"] = replay.get("state") == "PASS" and result["full_trading_metrics_exact"]
    if not result["engineering_verified"]:
        result["state"] = "REPLAY_CORRECTION_REQUIRED"
    w.atomic(cfg["output"], result)
    for source, destination in ((cfg["output"], cfg["destination"]), (str(replay_file), cfg["replay_destination"])):
        if Path(source).exists():
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", source, destination], check=True, timeout=60)
    subprocess.run(["systemctl", "--user", "disable", "--now", cfg["timer"]], check=True)
    return {"state": result["state"], "checkpoint_probe_replay": result["checkpoint_probe_replay"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    print(tick(w.read(a.config)), flush=True)
