#!/usr/bin/env python3
"""Durable operations wrapper for the owner's twelve pinned E3-M arms."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
import time


MODULES = ("open_l", "open_s", "close_l", "close_s")
ARCHIVE = "efa23475193865fc6594f7af4f6ee61a7d2cbb41d96756af89a8ed3cd04e202a"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def jobs():
    result = []
    for name in ("core15", "shortlist_strict", "shortlist", "shortlist_strict_enc"):
        result.append({"name": "cons_" + name, "list": f"FEATURE_LIST_{name}.json",
                       "encoded": name.endswith("_enc"), "split": "consolidated",
                       "state": "full", "dropout": 0.1, "wd": 1e-4})
    for name in ("core15", "shortlist_strict"):
        for state in ("full", "none"):
            for reg, drop, wd in (("base", 0.1, 1e-4), ("strong", 0.3, 1e-2)):
                result.append({"name": f"gap_{name}_{state}_{reg}",
                               "list": f"FEATURE_LIST_{name}.json", "encoded": False,
                               "split": "dev3y", "state": state, "dropout": drop, "wd": wd})
    return result


def planned_jobs(cfg):
    all_jobs = jobs()
    selected = cfg.get("only_jobs")
    if selected is None:
        return all_jobs
    known = {j["name"]: j for j in all_jobs}
    if not selected or len(selected) != len(set(selected)) or any(n not in known for n in selected):
        raise ValueError("INVALID_JOB_SUBSET")
    return [known[n] for n in selected]


def choose_gpu(text):
    rows = list(csv.reader(text.splitlines()))
    choices = [r[2].strip() for r in rows if len(r) == 3 and "5090" in r[1]]
    if len(choices) != 1:
        raise ValueError("GPU_5090_MISSING_OR_AMBIGUOUS")
    return choices[0]


def command(cfg, job):
    root = Path(cfg["root"])
    subdir, parquet = (("full_pool_enc", "full_pool_enc_eurusd_h1.parquet") if job["encoded"]
                       else ("full_pool", "full_pool_eurusd_h1.parquet"))
    cmd = [cfg["launcher"], "-q", "-m", cfg["memory"], "-t", "43200s", "-n", "e3m-" + job["name"],
           "--", cfg["python"], "tools/expert_guided_e3m.py", "run",
           "--admitted", str(root / "data/admitted"), "--csv-dir", str(root / "data/sel15"),
           "--store", str(root / "data" / subdir / parquet), "--feature-list", str(root / "in" / job["list"]),
           "--out", str(root / "out" / job["name"]), "--units", "2", "--head", "16",
           "--variants", "FULL", "--max-epochs", "1000", "--scaling", "fixed_train", "--nonstationary", "diff"]
    if job["split"] == "consolidated":
        cmd += ["--split-contract", "consolidated"]
    else:
        cmd += ["--agent-state", job["state"], "--dropout", str(job["dropout"]), "--wd", str(job["wd"])]
    return cmd


def verify_pins(cfg):
    for filename, expected in cfg["pins"].items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise ValueError("IDENTITY_CHANGED:" + filename)
    if "runtime_versions" in cfg and cfg["runtime_versions"] != runtime_versions(cfg["python"]):
        raise ValueError("RUNTIME_VERSIONS_CHANGED")


def runtime_versions(python):
    script = ("import importlib.metadata as m,json;print(json.dumps({n:m.version(n) "
              "for n in ['torch','numpy','pandas','scipy','scikit-learn']}))")
    return json.loads(subprocess.check_output([python, "-c", script], text=True, timeout=30))


def verify_archive_code(root):
    expected = {}
    with tarfile.open(root / "hs_code_final.tar.gz", "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isfile() and name.startswith(("app/", "tools/")) and name.endswith(".py"):
                expected[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    actual = {str(p.relative_to(root / "code")): digest(p) for directory in ("app", "tools")
              for p in (root / "code" / directory).rglob("*.py")}
    if not expected or expected != actual:
        raise ValueError("EXTRACTED_CODE_DIFFERS_FROM_PINNED_ARCHIVE")


def periods(job):
    return ("TRAIN", "INNER_A", "OBS_2024") if job["split"] == "consolidated" else (
        "TRAIN", "INNER_A", "INNER_B", "OBS_2024")


def validate_result(cfg, job):
    out = Path(cfg["root"]) / "out" / job["name"]
    result, scaler, probe = [read(out / n) for n in ("E3M.json", "SCALER.json", "PROBE.json")]
    if any(not isinstance(v, dict) for v in (result, scaler, probe)):
        raise ValueError("INCOMPLETE_TERMINAL_ARTIFACTS")
    if (result["device"] != "cuda" or result["verify"] is not True
            or result["scaling"] != "fixed_train" or result["variants"] != ["FULL"]
            or result["split_contract"]["arm"] != job["split"]
            or result["agent_state"]["mode"] != job["state"]
            or result["arch"]["units"] != 2 or result["arch"]["head"] != 16):
        raise ValueError("RESULT_CONTRACT_MISMATCH")
    requested = read(Path(cfg["root"]) / "in" / job["list"])["features"]
    features = scaler["features"]
    if (len(features) != len(requested) or len(features) != len(set(features))
            or any(actual not in (want, want + "@diff") for want, actual in zip(requested, features))
            or scaler["policy"] != "fixed_train" or scaler["split_contract"]["arm"] != job["split"]):
        raise ValueError("SCALER_FEATURE_OR_POLICY_MISMATCH")
    for field in ("mean", "std"):
        values = scaler[field]
        if len(values) != len(features) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError("SCALER_NONFINITE_OR_SHAPE")
    if any(float(v) <= 0 for v in scaler["std"]):
        raise ValueError("SCALER_INVALID_STD")
    if (probe["split_contract"] != job["split"] or probe["scaling"] != "fixed_train"
            or probe["agent_state"] != job["state"]):
        raise ValueError("PROBE_CONTRACT_MISMATCH")
    trials = result["trials"]["FULL"]
    if set(trials) != {str(s) for s in cfg["seeds"]}:
        raise ValueError("SEED_SET_MISMATCH")
    support = result["meta"]["support_hash"]
    if any(not isinstance(support.get(p), str) or not support[p] for p in periods(job)):
        raise ValueError("SUPPORT_MISSING")
    files = [out / n for n in ("E3M.json", "SCALER.json", "PROBE.json")]
    metrics = {}
    for seed in cfg["seeds"]:
        rec = trials[str(seed)]
        for source in ("trading", "musashi_contract"):
            for period in periods(job):
                m = rec[source][period]
                for key in ("net_profit_usd", "max_drawdown_pct", "trades", "win_rate"):
                    if m[key] is not None and not math.isfinite(float(m[key])):
                        raise ValueError("NONFINITE_NATIVE_METRIC")
                if m["sharpe_daily_sqrt365"] is not None and not math.isfinite(float(m["sharpe_daily_sqrt365"])):
                    raise ValueError("NONFINITE_SHARPE")
        metrics[str(seed)] = {k: rec[k] for k in ("trading", "musashi_contract")}
        for name in MODULES:
            key = f"FULL_{name}_{seed}"
            path = out / (key + ".pt")
            if not path.is_file() or path.stat().st_size == 0 or key not in probe["probe"]:
                raise ValueError("CHECKPOINT_OR_PROBE_MISSING")
            files.append(path)
    return {"job": job, "config_sha256": identity(cfg), "hashes": {p.name: digest(p) for p in files},
            "requested_features": requested, "consumed_features": features, "support_hash": support,
            "metrics": metrics, "wall_seconds": result["wall_seconds"], "no_ranking": True}


def receipt_path(cfg, job):
    return Path(cfg["root"]) / "ops" / (job["name"] + ".RESULT.json")


def seal_result(cfg, job):
    verify_pins(cfg)
    receipt = validate_result(cfg, job)
    atomic(receipt_path(cfg, job), receipt)
    return receipt


def adopt(cfg, job):
    path = receipt_path(cfg, job)
    if not path.exists():
        return None
    expected = read(path)
    if expected != validate_result(cfg, job):
        raise ValueError("RECEIPT_CHANGED:" + job["name"])
    return expected


def owns_child(argv, root):
    return (len(argv) > 3 and Path(argv[0]).name.startswith("python")
            and argv[1].endswith("tools/expert_guided_e3m.py") and argv[2] == "run"
            and "--out" in argv and argv.index("--out") + 1 < len(argv)
            and Path(argv[argv.index("--out") + 1]).is_relative_to(Path(root) / "out"))


def live_children(root):
    result = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s]
            if owns_child(argv, root):
                result.append(int(proc.name))
        except (OSError, UnicodeError):
            continue
    return result


def resource_refusal(rc, log, output):
    return (rc == 75 and "crispdm-run: REFUSED" in log
            and (not Path(output).exists() or not any(Path(output).iterdir())))


def closure(states, selected=None):
    selected = jobs() if selected is None else selected
    done = sum(states.get(j["name"], {}).get("state") == "DONE" for j in selected)
    failed = sum(states.get(j["name"], {}).get("state") == "FAILED" for j in selected)
    pending = len(selected) - done - failed
    return {"state": "RUNNING" if pending else ("COMPLETE" if not failed else "COMPLETE_WITH_FAILURES"),
            "done": done, "failed": failed, "pending": pending, "total": len(selected)}


def publish_status(cfg, states, state=None, **fields):
    value = {**closure(states, planned_jobs(cfg)), "updated": time.time(), "config_sha256": identity(cfg),
             "jobs": states, "gpu_uuid": cfg["gpu_uuid"], "memory": cfg["memory"], **fields}
    if state:
        value["state"] = state
    atomic(Path(cfg["root"]) / "ops/STATUS.json", value)


def ready_gpu(cfg):
    output = subprocess.check_output(["nvidia-smi", "-i", cfg["gpu_uuid"],
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    return not any(line.strip().isdigit() for line in output.splitlines())


def run(cfg):
    root, ops = Path(cfg["root"]), Path(cfg["root"]) / "ops"
    ops.mkdir(exist_ok=True)
    with (ops / "LOCK").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        states = read(ops / "JOBS.json", {})
        owner = read(ops / "IDENTITY.json")
        if owner and owner["config_sha256"] != identity(cfg):
            raise ValueError("CONFIG_IDENTITY_CHANGED")
        atomic(ops / "IDENTITY.json", {"config_sha256": identity(cfg)})
        while True:
            verify_pins(cfg)
            active = live_children(root)
            if active:
                publish_status(cfg, states, "WAIT_EXISTING_CHILD", child_pids=active)
                time.sleep(30)
                continue
            selected = planned_jobs(cfg)
            remaining = [j for j in selected if states.get(j["name"], {}).get("state") != "FAILED"]
            job = None
            for candidate in remaining:
                receipt = adopt(cfg, candidate)
                if receipt:
                    states[candidate["name"]] = {**states.get(candidate["name"], {}), "state": "DONE"}
                else:
                    job = candidate
                    break
            atomic(ops / "JOBS.json", states)
            if job is None:
                receipts = [adopt(cfg, j) for j in selected if states[j["name"]]["state"] == "DONE"]
                atomic(root / "CONSUMERS.json", {"config_sha256": identity(cfg), "results": receipts,
                      "dispositions": states, "no_ranking_or_promotion": True})
                closed = {**closure(states, selected), "closed": time.time(), "config_sha256": identity(cfg),
                          "consumer_sha256": digest(root / "CONSUMERS.json")}
                atomic(ops / "CLOSE.json", closed)
                publish_status(cfg, states)
                if cfg.get("timer_unit"):
                    subprocess.run(["systemctl", "--user", "disable", "--now", cfg["timer_unit"]], check=True)
                return
            name, out = job["name"], root / "out" / job["name"]
            record = states.get(name, {})
            if (out / "E3M.json").exists() and record.get("state") == "RUNNING":
                seal_result(cfg, job)
                continue
            if record.get("state") == "RUNNING":
                record.update(state="RETRY_PENDING", cause="INTERRUPTED_ENTERED_ATTEMPT")
                states[name] = record
                atomic(ops / "JOBS.json", states)
            if record.get("entered", 0) >= cfg.get("max_entered_attempts", 2):
                record.update(state="FAILED", cause=record.get("cause", "ATTEMPTS_EXHAUSTED"))
                states[name] = record
                atomic(ops / "JOBS.json", states)
                continue
            if not ready_gpu(cfg) or shutil.disk_usage(root).free < cfg.get("min_disk_bytes", 8 * 1024**3):
                publish_status(cfg, states, "WAIT_RESOURCES", current_job=name)
                time.sleep(60)
                continue
            launch = record.get("launches", 0) + 1
            if out.exists() and any(out.iterdir()):
                if not record:
                    raise ValueError("UNOWNED_OUTPUT:" + name)
                archive = root / "attempts" / name / ("before-launch-" + str(launch))
                archive.parent.mkdir(parents=True, exist_ok=True)
                out.rename(archive)
            (ops / "logs").mkdir(exist_ok=True)
            log_path = ops / "logs" / f"{name}-{launch}.log"
            rec = {**record, "state": "RUNNING", "launches": launch, "entered": record.get("entered", 0) + 1,
                   "started": time.time(), "command": command(cfg, job), "log": str(log_path)}
            states[name] = rec
            atomic(ops / "JOBS.json", states)
            with log_path.open("a") as log:
                child = subprocess.Popen(rec["command"], cwd=root / "code", stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": cfg["gpu_uuid"], "PYTHONUNBUFFERED": "1"})
                while child.poll() is None:
                    publish_status(cfg, states, "RUNNING", current_job=name, launcher_pid=child.pid)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            rc = child.returncode
            verify_pins(cfg)
            rec.update(returncode=rc, finished=time.time())
            if rc == 0:
                try:
                    seal_result(cfg, job)
                    rec["state"] = "DONE"
                except (ValueError, KeyError, TypeError) as exc:
                    rec.update(state="RETRY_PENDING", cause="TERMINAL_VALIDATION:" + str(exc))
            elif resource_refusal(rc, log_path.read_text(), out):
                rec.update(state="WAIT_RESOURCES", entered=rec["entered"] - 1, cause="ADMISSION_REFUSED")
            else:
                rec.update(state="RETRY_PENDING", cause="CHILD_EXIT:" + str(rc))
                if rc in (124, 137):
                    rec["entered"] = cfg.get("max_entered_attempts", 2)
            states[name] = rec
            atomic(ops / "JOBS.json", states)
            atomic(ops / f"{name}.LAUNCH_{launch}.json", rec)
            publish_status(cfg, states, current_job=name)
            if rec["state"] != "DONE":
                time.sleep(60)


def initialize(root, python, launcher, *, only_jobs=None, memory="4G", timer_unit=None, recovery_source=None):
    root = Path(root).resolve()
    if digest(root / "hs_code_final.tar.gz") != ARCHIVE:
        raise ValueError("ARCHIVE_IDENTITY_CHANGED")
    verify_archive_code(root)
    gpu = choose_gpu(subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"], text=True))
    paths = [root / "hs_code_final.tar.gz", root / "gamma_queue_musashi.sh", Path(__file__).resolve()]
    for name in ("code/app", "code/tools", "data", "in"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and (name in ("data", "in") or p.suffix == ".py"))
    paths.extend(p for p in root.glob("*.py") if p.is_file())
    cfg = {"root": str(root), "python": str(Path(python).expanduser().resolve()),
           "launcher": str(Path(launcher).expanduser().resolve()), "memory": memory, "gpu_uuid": gpu,
           "seeds": [20261019, 20261020, 20261021], "max_entered_attempts": 2,
           "min_disk_bytes": 8 * 1024**3, "timer_unit": timer_unit or "musashi-e3m-assigned-20261022.timer",
           "pins": {str(p): digest(p) for p in paths}}
    cfg["runtime_versions"] = runtime_versions(cfg["python"])
    if only_jobs is not None:
        cfg["only_jobs"] = only_jobs
    if recovery_source:
        cfg["recovery_source"] = recovery_source
    for j in planned_jobs(cfg):
        features = read(root / "in" / j["list"])["features"]
        if "raw.log_close" in features or len(features) != len(set(features)):
            raise ValueError("INVALID_FEATURE_LIST")
    (root / "ops").mkdir(exist_ok=True)
    config_path = root / "ops/CONFIG.private.json"
    if config_path.exists() and read(config_path) != cfg:
        raise ValueError("EXISTING_CONFIG_DIFFERS")
    atomic(config_path, cfg)
    config_path.chmod(0o600)
    atomic(root / "ops/PLAN.json", {"config_sha256": identity(cfg), "jobs": planned_jobs(cfg), "no_ranking": True})
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("init", "run", "status"))
    p.add_argument("--root", required=True)
    p.add_argument("--python")
    p.add_argument("--launcher")
    args = p.parse_args()
    root = Path(args.root)
    if args.mode == "init":
        cfg = initialize(root, args.python, args.launcher)
        print(json.dumps({"pins": len(cfg["pins"]), "jobs": len(jobs()), "gpu_uuid": cfg["gpu_uuid"],
                          "config_sha256": identity(cfg)}))
    elif args.mode == "status":
        print(json.dumps(read(root / "ops/STATUS.json", {"state": "NOT_STARTED"})))
    else:
        cfg = read(root / "ops/CONFIG.private.json")
        try:
            run(cfg)
        except Exception as exc:
            publish_status(cfg, read(root / "ops/JOBS.json", {}), "BLOCKED_INTEGRITY_OR_WORKER_ERROR", cause=str(exc))
            raise


if __name__ == "__main__":
    main()
