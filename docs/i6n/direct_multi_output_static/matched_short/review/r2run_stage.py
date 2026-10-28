"""R2 runner review driver (copied from p7_stage.py): stage the EXACT pinned tree + a private copy of the frozen population/input onto the
secondary worker's scratch, then run p7_gpu_replay.py there on the pinned RTX 4090 under that
host's crispdm-run. Role facts (ssh alias, python, GPU UUID, launcher path, environment) are read
from the closed campaigns' existing private deployment spec by PATH and are never printed or
committed: outputs name the role ("secondary") and the GPU class only.

  python p7_stage.py check | stage | measure | fit | replay | fetch
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PIN = "a5ac7607"
ROLE = "secondary"
REMOTE = "/var/tmp/retsu-r2-replay-20261015"
LOCAL = Path("/var/tmp/retsu-r2run-scratch/p7")


def role():
    spec = json.loads(C._SPEC.read_text())
    r = spec["roles"][ROLE]
    gpu = next(g for g in r["gpus"] if g.get("heavy"))
    return dict(alias=r["ssh_alias"], python=r["python"], crispdm=(r.get("launcher") or {}).get("crispdm_run"),
                uuid=gpu["uuid"], cls=gpu.get("class"), env=r.get("environment") or {})


def ssh(alias, command, timeout=600, input_bytes=None):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", alias, "--", command], input=input_bytes,
                          capture_output=True, timeout=timeout)


def redact(text, r):
    for secret, ph in ((r["uuid"], "<GPU_UUID>"), (r["alias"], "<SECONDARY>"), (r["python"], "<PY>"),
                       (r["crispdm"] or "\x00", "<CRISPDM_RUN>")):
        text = text.replace(secret, ph)
    return text


def check(r):
    cmd = ("df -B1 --output=avail /var/tmp | tail -1; "
           "nvidia-smi --query-gpu=uuid,name,memory.used,memory.total --format=csv,noheader,nounits; "
           "nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits; "
           "free -b | sed -n 2p; test -x %s && echo CRISPDM_OK; "
           "%s -c 'import tensorflow as tf, keras; print(tf.__version__, keras.__version__)' 2>/dev/null"
           % (shlex.quote(r["crispdm"] or "/nonexistent"), shlex.quote(r["python"])))
    out = ssh(r["alias"], cmd, timeout=180)
    text = out.stdout.decode()
    lines = text.strip().splitlines()
    gpus = [ln for ln in lines if ln.startswith("GPU-")]
    pinned = [ln for ln in gpus if ln.startswith(r["uuid"])]
    apps = [ln for ln in lines if ln.count(",") == 1 and ln.startswith("GPU-")]
    doc = {"rc": out.returncode, "var_tmp_free_bytes": int(lines[0]) if lines and lines[0].isdigit() else None,
           "pinned_gpu_present": bool(pinned), "pinned_gpu_class": r["cls"],
           "pinned_gpu_row": redact(pinned[0], r) if pinned else None,
           "compute_apps_on_pinned_gpu": [redact(a, r) for a in apps if a.startswith(r["uuid"])],
           "crispdm_run_present": "CRISPDM_OK" in text,
           "runtime": lines[-1] if lines else None}
    return doc


def stage(r):
    LOCAL.mkdir(parents=True, exist_ok=True)
    archive = LOCAL / f"tree-{PIN}.tar"
    subprocess.run(["git", "-C", str(C.REPO), "archive", "--format=tar", "-o", str(archive), PIN], check=True)
    tree = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", f"{PIN}^{{tree}}"], capture_output=True,
                          text=True, check=True).stdout.strip()
    out = ssh(r["alias"], f"mkdir -p {REMOTE}/code {REMOTE}/population {REMOTE}/input {REMOTE}/out && "
                          f"tar -x -C {REMOTE}/code", input_bytes=archive.read_bytes(), timeout=600)
    res = {"archive_rc": 0, "untar_rc": out.returncode, "pin": PIN, "tree": tree}
    # the review probe itself is not in the pinned tree: copy it beside the code (not into it)
    for src, dst in ((C.REVIEW / "r2run_gpu_replay.py", f"{REMOTE}/r2run_gpu_replay.py"),):
        res["probe_copy_rc"] = subprocess.run(["rsync", "-a", str(src), f"{r['alias']}:{dst}"],
                                              capture_output=True, timeout=300).returncode
    for src, dst in ((C.POPULATION, f"{REMOTE}/population"), (C.INPUT, f"{REMOTE}/input")):
        res[f"rsync_{Path(dst).name}_rc"] = subprocess.run(
            ["rsync", "-a", "--delete", f"{src}/", f"{r['alias']}:{dst}/"], capture_output=True, timeout=1800).returncode
    # verify the staged tree is byte-identical to the pin: git-style digest of every file
    verify = ssh(r["alias"], f"cd {REMOTE}/code && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum; "
                             f"cd {REMOTE}/population && sha256sum POPULATION.json; "
                             f"cd {REMOTE}/input && sha256sum INPUT.npz CALENDAR.json")
    local = subprocess.run(f"mkdir -p {LOCAL}/tree && tar -x -C {LOCAL}/tree -f {archive} && cd {LOCAL}/tree && "
                           "find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum", shell=True,
                           capture_output=True, text=True).stdout.split()[0]
    lines = verify.stdout.decode().split("\n")
    res["remote_tree_manifest_sha256"] = lines[0].split()[0] if lines and lines[0] else None
    res["local_tree_manifest_sha256"] = local
    res["tree_identical"] = res["remote_tree_manifest_sha256"] == local
    res["remote_population_json_sha256_ok"] = lines[1].split()[0] == C.SEAL["population_json_sha256"] if len(lines) > 1 else False
    res["remote_input_sha256_ok"] = lines[2].split()[0] == C.SEAL["identities"]["source"]["inputs"]["INPUT.npz"] \
        if len(lines) > 2 else False
    return res


def run_remote(r, mode, cap, name, variant="original", gpu=True):
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in r["env"].items() if k != "CUDA_VISIBLE_DEVICES")
    cuda = shlex.quote(r["uuid"]) if gpu else "''"
    cmd = (f"cd {REMOTE}/code && env {env} CUDA_VISIBLE_DEVICES={cuda} "
           f"RETSU_CODE_ROOT={REMOTE}/code PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 "
           f"{shlex.quote(r['crispdm'])} -q -W 1800 -m {cap} -t 1800s -n {name} -- "
           f"{shlex.quote(r['python'])} {REMOTE}/r2run_gpu_replay.py --mode {mode} --variant {variant} "
           f"--population {REMOTE}/population --input-dir {REMOTE}/input --work {REMOTE}/work")
    out = ssh(r["alias"], cmd, timeout=3600)
    return {"rc": out.returncode, "stdout_tail": redact(out.stdout.decode()[-600:], r),
            "stderr_tail": redact(out.stderr.decode()[-1500:], r)}


def fetch(r):
    dst = LOCAL / "remote_out"
    dst.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["rsync", "-a", "--include=*.json", "--exclude=*", f"{r['alias']}:{REMOTE}/work/", f"{dst}/"],
                        capture_output=True,
                        timeout=1800).returncode
    return {"rc": rc}


if __name__ == "__main__":
    r = role()
    what = sys.argv[1]
    if what == "check":
        doc = check(r)
    elif what == "stage":
        doc = stage(r)
    elif what in ("leak", "pretrain", "importfit", "replay"):
        variant = sys.argv[3] if len(sys.argv) > 3 else "original"
        doc = run_remote(r, what, sys.argv[2], f"i6dmo-r2review-{what}-{variant}", variant=variant,
                         gpu=what != "leak")
    elif what == "fetch":
        doc = fetch(r)
    else:
        raise SystemExit(what)
    print(json.dumps(doc, indent=1, sort_keys=True))
