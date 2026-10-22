#!/usr/bin/env python3
"""Install only the assigned worker; preserve the existing scientific jobs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def ssh(host, code, *args):
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                                    "python3", "-", *args], input=code.encode(), timeout=120).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-commit", default="38ca383")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--protocol-version", choices=("2", "3"), default="2")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    v3 = args.protocol_version == "3"
    code_subdir = "code_v3" if v3 else "code"
    config_name = "shared-selection-v3-20261022.json" if v3 else "shared-selection-20261022.json"
    unit_name = "musashi-psel-v3-20261022" if v3 else "musashi-psel-20261022"
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.source_commit], cwd=args.source_repo).decode().splitlines()
    paths = [p for p in paths if p.endswith(".py") and p.startswith(("app/", "tools/"))]
    expected = {p: hashlib.sha256(subprocess.check_output(["git", "show", f"{args.source_commit}:{p}"], cwd=args.source_repo)).hexdigest() for p in paths}
    probe = '''import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); c=sys.argv[2]; files=list((r/c/'app').rglob('*.py'))+list((r/c/'tools').rglob('*.py'))
files += [p for p in (r/'data').rglob('*') if p.is_file()]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
print(json.dumps({str(p.relative_to(r)):digest(p) for p in files}))
'''
    own = json.loads(ssh(args.host, probe, args.root, code_subdir))
    peer = json.loads(ssh(args.peer, probe, args.root, code_subdir))
    for p, digest in expected.items():
        if own.get(f"{code_subdir}/{p}") != digest or peer.get(f"{code_subdir}/{p}") != digest:
            raise ValueError(f"SOURCE_IDENTITY_MISMATCH:{p}")
    if own != peer:
        raise ValueError("INPUT_OR_CODE_COPIES_DIFFER")
    source = subprocess.check_output(["git", "show", f"{args.source_commit}:tools/pool_selection_v2.py"], cwd=args.source_repo).decode()
    tasks = next(ast.literal_eval(n.value) for n in ast.parse(source).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets))
    home = ssh(args.host, "from pathlib import Path;print(Path.home())").strip()
    config = {"root": args.root, "source_commit": args.source_commit,
              "python": f"{home}/anaconda3/envs/trading-stack/bin/python",
              "admission": f"{home}/.local/bin/crispdm-run",
              "pins": {f"{args.root}/{p}": h for p, h in own.items()},
              "base_keys": [f"base|{t}|{m}|{k}" for t in tasks for m in ("hgb", "temporal") for k in range(6)]}
    with tempfile.TemporaryDirectory() as tmp:
        ssh(args.host, "from pathlib import Path\nh=Path.home()\nfor p in ['.local/libexec','.config','.config/systemd/user']:(h/p).mkdir(parents=True,exist_ok=True)\n")
        if v3:
            details = {}
            for line in subprocess.check_output(["ssh", "-G", args.peer], text=True).splitlines():
                k, _, value = line.partition(" ")
                if k in ("hostname", "port", "user", "hostkeyalias"):
                    details[k] = value
            alias = details.get("hostkeyalias", f"[{details['hostname']}]:{details['port']}")
            trusted = subprocess.check_output(["ssh-keygen", "-F", alias], text=True)
            known = Path(tmp) / "known_hosts"
            known.write_text(''.join(s + '\n' for s in trusted.splitlines() if s and not s.startswith('#')))
            known_dest = f"{home}/.config/musashi-psel-v3-peer-known-hosts"
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(known), f"{args.host}:{known_dest}"], check=True, timeout=60)
            config.update({"code_subdir": code_subdir, "output_subdir": "out_v3", "ops_subdir": "musashi_ops_v3",
                           "base_log": "LOG_v3", "stage_log": "LOG_v3", "run_log": "psel_v3.log",
                           "stages": ["base", "contrast", "encoding", "rescue"], "unit_prefix": "musashi-psel3",
                           "timer_unit": unit_name + ".timer",
                           "peer_ssh": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
                                        "-o", f"HostKeyAlias={alias}", "-o", f"UserKnownHostsFile={known_dest}",
                                        "-p", details["port"], f"{details['user']}@{details['hostname']}"]})
            ssh(args.host, "import subprocess\nsubprocess.run(" + repr(config["peer_ssh"] + ["true"]) + ",check=True)\n")
        cfg = Path(tmp) / "config.json"
        cfg.write_text(json.dumps(config, indent=2))
        service = Path(tmp) / "worker.service"
        timer = Path(tmp) / "worker.timer"
        service.write_text((repo / "deploy/shared-selection-worker.service").read_text().replace("shared-selection-20261022.json", config_name))
        timer.write_text((repo / "deploy/shared-selection-worker.timer").read_text().replace("musashi-psel-20261022", unit_name))
        for src, dest in [(repo / "tools/shared_selection_worker.py", f"{home}/.local/libexec/shared_selection_worker.py"),
                          (cfg, f"{home}/.config/{config_name}"),
                          (service, f"{home}/.config/systemd/user/{unit_name}.service"),
                          (timer, f"{home}/.config/systemd/user/{unit_name}.timer")]:
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(src), f"{args.host}:{dest}"], check=True, timeout=60)
    ssh(args.host, "import subprocess\nfrom pathlib import Path\nPath.home().joinpath(" + repr('.config/' + config_name) + ").chmod(0o600)\nsubprocess.run(['systemctl','--user','daemon-reload'],check=True)\nsubprocess.run(['systemctl','--user','enable','--now'," + repr(unit_name + '.timer') + "],check=True)\nsubprocess.run(['systemctl','--user','start'," + repr(unit_name + '.service') + "],check=True)\n")
    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"config": config, "verified_copies": 2, "pinned_files": len(own)}, indent=2))
    receipt.chmod(0o600)
    print(json.dumps({"verified_copies": 2, "pinned_files": len(own), "base_units": len(config["base_keys"]), "timer_enabled": True}))


if __name__ == "__main__":
    main()
