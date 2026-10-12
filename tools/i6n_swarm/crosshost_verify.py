"""Cross-host byte verification: every file of the shared data root, and the code pin, on each consuming host.

Identical paths do not prove identical bytes: each host hashes its own files and the result is compared with
the coordinator's. Hosts come from the private worker inventory and are never printed.

    python -m tools.i6n_swarm.crosshost_verify --pin <predictor commit> [--data-root D] [--inventory PATH]
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
_p = argparse.ArgumentParser()
_p.add_argument("--pin", required=True)
_p.add_argument("--data-root", default="/var/tmp/i6n-swarm-data-20261012")
_p.add_argument("--inventory", default=str(Path.home() / ".local/state/canonical_20261003/i6e/multihorizon_2024_v2/WORKERS.private.json"))
_a = _p.parse_args()
INV, DATA, PIN = Path(_a.inventory), _a.data_root, _a.pin
def local_manifest():
    out = {}
    for p in sorted(Path(DATA).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(DATA))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
REMOTE = r'''
import hashlib, json, os, subprocess, sys
from pathlib import Path
D = sys.argv[1]; pin = sys.argv[2]; work = sys.argv[3]
m = {}
if Path(D).is_dir():
    for p in sorted(Path(D).rglob("*")):
        if p.is_file():
            m[str(p.relative_to(D))] = hashlib.sha256(p.read_bytes()).hexdigest()
code = None
for repo in [work, os.path.expanduser("~/Documents/GitHub/predictor")]:
    if repo and Path(repo, ".git").exists():
        r = subprocess.run(["git", "-C", repo, "cat-file", "-e", pin + "^{commit}"], capture_output=True)
        code = dict(repo_present=True, pin_present=r.returncode == 0)
        break
print(json.dumps(dict(data_root_present=Path(D).is_dir(), manifest=m, code=code or dict(repo_present=False))))
'''
local = local_manifest()
report = dict(data_root=DATA, coordinator_files=len(local), pin=PIN, roles={})
for w in json.loads(INV.read_text()):
    role = w["role"]
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", w["host"], "python3", "-",
                            DATA, PIN, w.get("worktree", "")], input=REMOTE, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            report["roles"][role] = dict(reachable=False, error=r.stderr.strip()[-200:].replace(w["host"], "<host>"))
            continue
        remote = json.loads(r.stdout)
    except Exception as exc:
        report["roles"][role] = dict(reachable=False, error=type(exc).__name__)
        continue
    m = remote["manifest"]
    missing = sorted(set(local) - set(m))
    differ = sorted(k for k in set(local) & set(m) if local[k] != m[k])
    report["roles"][role] = dict(reachable=True, data_root_present=remote["data_root_present"],
                                 files_present=len(set(local) & set(m)), files_missing=len(missing),
                                 files_differing=len(differ), missing_examples=missing[:5], differing_examples=differ[:5],
                                 identical=not missing and not differ, code=remote["code"])
print(json.dumps(report, indent=1))
