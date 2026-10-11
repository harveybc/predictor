"""Unattended work-stealing scheduler over SSH and existing governed admission.

No worker needs an SSH key into the coordinator or warehouse credentials. The
coordinator deploys sealed jobs, supervises durable remote units, reads back
artifacts, verifies price metrics and moves the queue forward automatically.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from tools.i6m_data import file_hash, load_input, metrics, partitions, targets
from tools.i6m_queue import Queue, atomic_json, digest, seal, verify


def shell(command, timeout=40, check=True):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=check)


def remote_call(worker, command, **kwargs):
    return shell(command if worker.get("local") else ["ssh", "-o", "BatchMode=yes", worker["host"], *command], **kwargs)


def collect(root, worker, job, design, arrays):
    """Read artifacts, then independently reconcile complete scored populations."""
    remote = worker["root"] + "/cells/" + job["id"] + "/"
    local = root/"cells"/job["id"]
    local.mkdir(parents=True, exist_ok=True)
    if not worker.get("local"):
        shell(["rsync", "-a", "--timeout=30", worker["host"]+":"+remote, str(local)+"/"], timeout=90)
    cell = verify(json.loads((local/"CELL.json").read_text()))
    if cell["status"] == "COMPLETED":
        import numpy as np
        ts, x, price = arrays
        h = job["arm"]["horizon_hours"]
        y, endpoints = targets(ts, price, h)
        _, _, si = partitions(ts, y, job["week"], h)
        if file_hash(local/"PREDICTIONS.npz") != cell["predictions_sha256"]:
            raise ValueError("prediction file digest mismatch")
        with np.load(local/"PREDICTIONS.npz", allow_pickle=False) as saved:
            for key, expected in (("timestamps", ts[si]), ("target_timestamps", ts[endpoints[si]]),
                                  ("current", price[si]), ("actual", price[endpoints[si]])):
                if not np.array_equal(saved[key], expected):
                    raise ValueError(f"original price/population mismatch: {key}")
            recalculated = metrics(saved["current"], saved["actual"], saved["prediction"], h)
        if recalculated != cell["metrics"] or digest(ts[si].tolist()) != cell["rows_sha256"]:
            raise ValueError("MAE/naive/skill not derived from retained original prices")
        if job["model"] == "NEURAL" and file_hash(local/"MODEL.keras") != cell["model_sha256"]:
            raise ValueError("model checkpoint digest mismatch")
    return cell


def launch(root, worker, job):
    """Transient unit is the durable owner; worker restart never duplicates a fit."""
    body = seal(job)
    job_path = root/"jobs"/(job["id"]+".json")
    atomic_json(job_path, body)
    remote_job = worker["root"]+"/jobs/"+job_path.name
    if not worker.get("local"):
        shell(["rsync", "-a", str(job_path), worker["host"]+":"+remote_job])
    unit = "predictor-i6m-"+job["lease"][:16]
    invocation = [worker["light_python"], "-m", "tools.i6m_worker", "--job", remote_job,
                  "--python", worker["fit_python"], "--admission", worker["admission"], "--role", worker["role"],
                  "--pilot-cap", str(worker.get("pilot_cap", 4*1024**3)),
                  "--maximum-cap", str(worker.get("maximum_cap", 7936*1024**2))]
    env = dict(worker.get("environment", {}), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1", PYTHONNOUSERSITE="1")
    command = "cd "+shlex.quote(worker["worktree"])+"; exec env "+" ".join(shlex.quote(k+"="+v) for k,v in env.items())+" "+shlex.join(invocation)
    remote_call(worker, ["systemd-run", "--user", "--unit="+unit,
           "--property=MemoryMax=268435456", "--property=Restart=on-failure", "--property=RestartSec=30",
           "/bin/bash", "-lc", command if worker.get("local") else shlex.quote(command)])


def cycle(root, workers, q, arrays):
    """One bounded cycle; terminal failures advance other independent jobs."""
    by_role = {w["role"]: w for w in workers}
    errors = []
    for job in q.running():
        worker = by_role[job["slot"]]
        remote = worker["root"]+"/cells/"+job["id"]+"/CELL.json"
        try:
            exists = remote_call(worker, ["test", "-f", remote], check=False).returncode == 0
            if exists:
                cell = collect(root, worker, job, q.design, arrays)
                q.complete(job, cell)
                if not worker.get("local"):
                    shell(["rsync", "-a", "--timeout=20", worker["host"]+":"+worker["root"]+"/cost/", str(root/"cost")+"/"], timeout=40, check=False)
            else:
                unit = "predictor-i6m-"+job["lease"][:16]+".service"
                state = remote_call(worker, ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"], check=False).stdout.strip()
                if state not in ("active", "activating", "reloading"):
                    # Dispatch may have been interrupted before the durable owner
                    # existed. Only launch if neither terminal nor owner exists.
                    partial = remote_call(worker, ["test", "-d", worker["root"]+"/cells/"+job["id"]], check=False).returncode == 0
                    if not partial:
                        launch(root, worker, job)
                    else:
                        cell = seal(dict(job_id=job["id"], lease=job["lease"], design_sha256=q.identity,
                                    status="FAILED", model=job["model"], arm=job["arm"], week=job["week"], seed=job["seed"],
                                    reason="INTERRUPTED_OWNER_NO_TERMINAL: partial retained; no automatic refit",
                                    elapsed_seconds=time.time()-job["claimed"], metrics=None, test_read=False))
                        atomic_json(root/"cells"/job["id"]/"CELL.json", cell)
                        q.complete(job, cell)
        except Exception as exc:
            errors.append(dict(job_id=job["id"], reason=str(exc)))
    for worker in workers:
        try:
            measured = []
            for path in (root/"cost").glob(worker["role"]+"_n*.json"):
                cost = verify(json.loads(path.read_text()))
                if cost["design_sha256"] == q.identity:
                    measured.append(cost["samples"])
            job = q.claim(worker["role"], worker["model"], worker.get("max_samples"),
                          measured if worker["model"] == "NEURAL" else None)
            if job:
                launch(root, worker, job)
        except Exception as exc:
            errors.append(dict(role=worker["role"], reason=str(exc)))
    status = q.status()
    status["transport_errors"] = errors
    with q.connect() as conn:
        status["published"] = conn.execute("SELECT count(*) FROM jobs WHERE published=1").fetchone()[0]
    atomic_json(root/"STATUS.json", status)
    if not (root/"PROGRESS.png").exists() or time.time()-(root/"PROGRESS.png").stat().st_mtime > 300:
        from tools.i6m_results import plot
        plot(root, q)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "status"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=Path)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    if a.command == "status":
        print((a.root/"STATUS.json").read_text())
        return
    design = verify(json.loads((a.root/"DESIGN.json").read_text()))
    from tools.i6m_prepare import code_identity
    if code_identity() != design["code_sha256"]:
        raise ValueError("coordinator runtime is not the sealed code")
    q = Queue(a.root/"QUEUE.db", design)
    arrays = load_input(a.root/"INPUT.npz", design["input_sha256"])
    workers = json.loads(a.workers.read_text())
    while True:
        try:
            state = cycle(a.root, workers, q, arrays)
            if not state["counts"].get("QUEUED") and not state["counts"].get("RUNNING"):
                from tools.i6m_results import close
                close(a.root, q)
        except Exception as exc:
            atomic_json(a.root/"COORDINATOR_ERROR.json", dict(timestamp=time.time(), reason=str(exc)))
        if a.once:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
