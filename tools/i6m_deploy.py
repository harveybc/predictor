"""One-command pinned deployment and durable unattended D -> H execution.

Host names, GPU identities and library paths are discovered at deployment and
kept ONLY in a private local worker configuration, never committed to Git.
No broker, desktop application or unrelated running unit is stopped.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from tools.i6m_coordinator import shell
from tools.i6m_prepare import prepare
from tools.i6m_queue import atomic_json


def deploy(a):
    base = Path(__file__).resolve().parents[1]
    pin = shell(["git", "-C", str(base), "rev-parse", "HEAD"]).stdout.strip()
    if shell(["git", "-C", str(base), "status", "--porcelain"]).stdout:
        raise ValueError("deployment requires a clean, committed worktree")
    prepare(a.parent, a.root)
    workers = []
    for role, host, device, max_samples in (("preferred", a.preferred_host, "RTX 5090", 241),
                                            ("secondary", a.secondary_host, "RTX 4090", 9)):
        home = shell(["ssh", host, "printenv", "HOME"]).stdout.strip()
        worktree = home+"/Documents/GitHub/.worktrees/predictor-i6m-"+pin[:12]
        repo = home+"/Documents/GitHub/predictor"
        branch = shell(["git", "-C", str(base), "branch", "--show-current"]).stdout.strip()
        command = "set -e; git -C "+shlex.quote(repo)+" fetch origin "+shlex.quote(branch)+"; "
        command += "if [ ! -d "+shlex.quote(worktree)+" ]; then git -C "+shlex.quote(repo)+" worktree add --detach "+shlex.quote(worktree)+" "+pin+"; fi; "
        command += "test \"$(git -C "+shlex.quote(worktree)+" rev-parse HEAD)\" = "+pin+"; test -z \"$(git -C "+shlex.quote(worktree)+" status --porcelain)\""
        shell(["ssh", host, command], timeout=180)
        gpu_rows = shell(["ssh", host, "nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"]).stdout.splitlines()
        matches = [line.split(",", 1)[1].strip() for line in gpu_rows if device in line]
        if len(matches) != 1:
            raise ValueError(f"{role}: required device unavailable or ambiguous")
        tf = home+"/anaconda3/envs/tensorflow"
        ld = shell(["ssh", host, "cat", home+"/.local/state/scratch/m07/ld_library_path.txt"]).stdout.strip()
        remote_root = home+"/.local/state/canonical_20261003/i6e/multihorizon_2024_v1"
        shell(["ssh", host, "mkdir", "-p", remote_root+"/jobs", remote_root+"/cells", remote_root+"/cost"])
        shell(["rsync", "-a", str(a.root/"DESIGN.json"), str(a.root/"TRAIN.npz"), str(a.root/"INPUT.npz"), host+":"+remote_root+"/"], timeout=180)
        workers.append(dict(role=role, host=host, root=remote_root, worktree=worktree, model="NEURAL",
                            max_samples=max_samples, light_python=home+"/anaconda3/bin/python", fit_python=tf+"/bin/python",
                            admission=home+"/.local/bin/crispdm-run", pilot_cap=4*1024**3,
                            maximum_cap=7936*1024**2, environment=dict(CUDA_VISIBLE_DEVICES=matches[0], LD_LIBRARY_PATH=ld,
                            XLA_FLAGS="--xla_gpu_cuda_data_dir="+tf+"/lib/python3.12/site-packages/nvidia/cuda_nvcc", TF_CPP_MIN_LOG_LEVEL="2")))
    home = str(Path.home())
    workers.append(dict(role="control", local=True, host="local", model="ARIMA", root=str(a.root), worktree=str(base),
                        light_python=home+"/anaconda3/bin/python", fit_python=home+"/anaconda3/bin/python",
                        admission=home+"/.local/bin/crispdm-run", environment=dict(CUDA_VISIBLE_DEVICES="")))
    atomic_json(a.root/"WORKERS.private.json", workers)
    os.chmod(a.root/"WORKERS.private.json", 0o600)
    py = home+"/anaconda3/bin/python"
    units = [("predictor-i6m-coordinator", [py, "-m", "tools.i6m_coordinator", "run", "--root", str(a.root), "--workers", str(a.root/"WORKERS.private.json")]),
             ("predictor-i6m-olap", [py, "-m", "tools.i6m_publish", "--root", str(a.root), "--url", a.url,
                                     "--token-file", str(a.token_file), "--pin", pin, "--follow"])]
    for unit, command in units:
        active = shell(["systemctl", "--user", "is-active", unit+".service"], check=False)
        if active.returncode == 0:
            continue
        shell(["systemd-run", "--user", "--unit="+unit, "--property=MemoryMax=536870912",
               "--property=Restart=on-failure", "--property=RestartSec=30", "--property=WorkingDirectory="+str(base),
               "--setenv=CUDA_VISIBLE_DEVICES=", "--setenv=OPENBLAS_NUM_THREADS=1", *command])
    atomic_json(a.root/"DEPLOYMENT.json", dict(status="DURABLE_SERVICES_STARTED", code_commit=pin,
                role_devices={w["role"]:"preferred GPU only" if w["role"]=="preferred" else "secondary GPU" for w in workers if not w.get("local")},
                supervisor_memory_bytes=536870912, test_read=False))
    print(json.dumps(dict(status="DURABLE_SERVICES_STARTED", code_commit=pin, root=str(a.root))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--preferred-host", required=True)
    p.add_argument("--secondary-host", required=True)
    p.add_argument("--url", default="http://127.0.0.1:5057")
    p.add_argument("--token-file", type=Path, required=True)
    deploy(p.parse_args())


if __name__ == "__main__":
    main()
