"""Collect pilot status and mirror completed artifacts once, without a resident agent."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from tools.i6n_swarm import common


def collect(inventory, root):
    root = Path(root).resolve()
    workers = json.loads(Path(inventory).read_text())
    report = {}
    for role in ("preferred", "secondary"):
        worker = next(w for w in workers if w["role"] == role)
        remote_root = root / role / "run"
        code = ("import json,pathlib; r=pathlib.Path("+repr(str(remote_root))+"); "
                "s=json.loads((r/'STATUS.json').read_text()); "
                "p=list(r.glob('binding/candidates/*/STATUS.json')); "
                "c=json.loads(p[-1].read_text()) if p else {}; "
                "s['training']={k:c[k] for k in ('state','progress','eta_seconds','elapsed_seconds') if k in c}; "
                "print(json.dumps(s))")
        try:
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                                     worker["host"], "python3 -c "+shlex.quote(code)],
                                    check=True, capture_output=True, text=True, timeout=25)
            state = json.loads(result.stdout)
            if state["state"] in ("COMPLETED", "FAILED"):
                destination = root / "mirror" / role
                destination.mkdir(parents=True, exist_ok=True)
                marker = destination / "MIRRORED.json"
                digest = common.sha256_json(state)
                if not marker.exists() or json.loads(marker.read_text())["terminal_sha256"] != digest:
                    subprocess.run(["rsync", "-a", "--timeout=60", worker["host"]+":"+str(remote_root)+"/",
                                    str(destination)+"/"], check=True, capture_output=True, timeout=120)
                    if state["state"] == "COMPLETED":
                        expected = state["result"]["sha256"]
                        if common.file_sha256(destination / "RESULT.json") != expected:
                            raise ValueError("mirrored result digest mismatch")
                    common.atomic_json(marker, dict(terminal_sha256=digest), mode=0o600)
                state["mirror"] = str(destination)
            report[role] = state
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
            report[role] = dict(state="COLLECT_ERROR", reason=type(exc).__name__)
    common.atomic_json(root / "STATUS.json", report, mode=0o600)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    result = collect(args.inventory, args.root)
    print(json.dumps({role: value["state"] for role, value in result.items()}))


if __name__ == "__main__":
    main()
