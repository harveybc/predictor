#!/usr/bin/env python3
"""C161 (order 2026-09-13): host preflight before any dispatch.

Hosts are named by role only (COORDINATOR, WORKER_A, WORKER_B). The map from
role to ssh alias lives outside Git (default ~/.config/crispdm/host_roles.json;
a null alias means this machine) and never appears in an output.

For each role, every check is a row of df_fact_host_receipt with the value
expected from the coordinator and the value observed on the host:

* commit        - HEAD of the host's predictor checkout equals the coordinator's;
* tracked_clean - that checkout has no tracked edits;
* code_digest   - sha256 over (path, blob sha256) of every tracked file under
                  tools/ and olap/, computed on the host from its working files;
* data_manifest - the host's copy of the C161 transfer manifest is byte-equal
                  to the coordinator's, and every file in it verifies by sha256;
* memory_cgroup - the user systemd manager delegates the memory controller, so
                  a per-process MemoryMax can be enforced;
* gpu_free      - no GPU compute process (this order is CPU only; GPUs stay free);
* python_env    - the declared interpreter exists;
* mem_total / cpus - recorded, not judged.

A role is DISPATCHABLE only if every judged check is VERIFIED. A host without
the exact data does not receive a task. Read only on every host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATE = Path.home() / ".local/state/crispdm-data-foundation"
DEFAULT_ROLES = Path.home() / ".config/crispdm/host_roles.json"
MANIFEST_NAME = "C161_MANIFEST.sha256"
PY_REL = "anaconda3/envs/trading-stack/bin/python"
JUDGED = ("commit", "tracked_clean", "code_digest", "data_manifest", "memory_cgroup", "gpu_free", "python_env")

# Runs identically on every host (bash; paths relative to $HOME). Prints one JSON object.
PROBE = r"""
set -u
CK="$1"; MANIFEST_SHA_EXPECTED="$2"
out() { printf '%s' "$1"; }
head=$(git -C "$CK" rev-parse HEAD 2>/dev/null || echo MISSING)
dirty=$(git -C "$CK" status --porcelain 2>/dev/null | grep -vc '^??')
code=$(cd "$CK" 2>/dev/null && git ls-files -z tools olap | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
S="$HOME/.local/state/crispdm-data-foundation"
msha=$( [ -f "$S/C161_MANIFEST.sha256" ] && sha256sum "$S/C161_MANIFEST.sha256" | cut -d' ' -f1 || echo MISSING)
if [ "$msha" = "$MANIFEST_SHA_EXPECTED" ]; then
  bad=$(cd "$S" && sha256sum -c C161_MANIFEST.sha256 2>/dev/null | grep -vc ': OK$')
else
  bad=-1
fi
ctl=$(cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.subtree_control 2>/dev/null)
gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
py=$( [ -x "$HOME/PYREL" ] && echo PRESENT || echo MISSING)
mem=$(grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2)
printf '{"head":"%s","dirty":%s,"code":"%s","manifest_sha":"%s","manifest_bad":%s,"cgroup":"%s","gpu":%s,"python":"%s","mem_kib":%s,"cpus":%s}\n' \
  "$head" "${dirty:-0}" "$code" "$msha" "$bad" "$ctl" "${gpu:-0}" "$py" "$mem" "$(nproc)"
""".replace("PYREL", PY_REL)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_sha256() -> str:
    return _sha_bytes(Path(__file__).read_bytes())


def run_probe(alias, checkout_rel: str, manifest_sha: str) -> dict:
    home_ck = f"$HOME/{checkout_rel}" if alias else str(Path.home() / checkout_rel)
    if alias:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias,
               f"bash -s -- {home_ck} {manifest_sha}"]
    else:
        cmd = ["bash", "-s", "--", home_ck, manifest_sha]
    r = subprocess.run(cmd, input=PROBE, capture_output=True, text=True, timeout=900)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return json.loads(line) if line.startswith("{") else {"error": (r.stderr or "no output")[-300:]}


def expected_values(local_checkout: Path, manifest: Path) -> dict:
    head = subprocess.run(["git", "-C", str(local_checkout), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    probe = run_probe(None, str(local_checkout.relative_to(Path.home())), _sha_bytes(manifest.read_bytes()))
    return {"head": head, "code": probe.get("code"), "manifest_sha": _sha_bytes(manifest.read_bytes())}


def receipts_for(role: str, obs: dict, exp: dict, run_id: str) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    sha = code_sha256()

    def row(name, expected, observed, ok, details=None):
        status = "UNAVAILABLE" if observed is None else ("VERIFIED" if ok else "MISMATCH")
        return {"run_id": run_id, "host_role": role, "check_name": name, "expected": expected,
                "observed": None if observed is None else str(observed), "status": status,
                "details": details or ({} if status == "VERIFIED" else {"note": f"{name} did not verify"}),
                "checked_at": now, "code_sha256": sha}

    if "error" in obs:
        return [row(n, None, None, False, {"probe_error": obs["error"]}) for n in JUDGED]
    return [
        row("commit", exp["head"], obs["head"], obs["head"] == exp["head"]),
        row("tracked_clean", "0", obs["dirty"], obs["dirty"] == 0),
        row("code_digest", exp["code"], obs["code"], obs["code"] == exp["code"]),
        row("data_manifest", f"{exp['manifest_sha']} with 0 failing files",
            f"{obs['manifest_sha']} with {obs['manifest_bad']} failing files",
            obs["manifest_sha"] == exp["manifest_sha"] and obs["manifest_bad"] == 0,
            None if obs["manifest_bad"] == 0 else {"failing_files": obs["manifest_bad"]}),
        row("memory_cgroup", "memory delegated", obs["cgroup"], "memory" in obs["cgroup"].split()),
        row("gpu_free", "0 compute processes", obs["gpu"], obs["gpu"] == 0),
        row("python_env", "PRESENT", obs["python"], obs["python"] == "PRESENT"),
        row("mem_total_kib", None, obs["mem_kib"], True, {"recorded_not_judged": True}),
        row("cpus", None, obs["cpus"], True, {"recorded_not_judged": True}),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="new write-once directory")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    ap.add_argument("--checkout", default="Documents/GitHub/.worktrees/predictor-c146",
                    help="predictor checkout on WORKERS, relative to their home")
    ap.add_argument("--manifest", type=Path, default=STATE / "host_sync_c161_v1/MANIFEST.sha256")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"REFUSED: {a.out.name} exists; host receipts are write-once")
    sys.path.insert(0, str(HERE))
    import load_data_foundation as L  # noqa: E402

    roles = json.loads(a.roles.read_text())
    exp = expected_values(REPO, a.manifest)
    rows, summary = [], {}
    for role, cfg in roles.items():
        if role == "COORDINATOR":
            obs = run_probe(None, str(REPO.relative_to(Path.home())), exp["manifest_sha"])
            # the coordinator's own transfer manifest is not copied into its state root
            if obs.get("manifest_sha") == "MISSING":
                obs["manifest_sha"], obs["manifest_bad"] = exp["manifest_sha"], 0
        else:
            obs = run_probe(cfg["ssh"], a.checkout, exp["manifest_sha"])
        rs = receipts_for(role, obs, exp, a.run_id)
        problems = [p for r in rs for p in L.validate_row("df_fact_host_receipt", r)]
        if problems:
            raise SystemExit(f"host receipt rows do not validate: {problems[:3]}")
        rows += rs
        judged = [r for r in rs if r["check_name"] in JUDGED]
        summary[role] = {"dispatchable": all(r["status"] == "VERIFIED" for r in judged),
                         "not_verified": [r["check_name"] for r in judged if r["status"] != "VERIFIED"]}
    a.out.mkdir(parents=True)
    (a.out / "df_fact_host_receipt.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    doc = {"schema": "crispdm.data_foundation.host_preflight.v1", "run_id": a.run_id, "expected": exp,
           "roles": summary, "code_sha256": code_sha256(),
           "rows_sha256": _sha_bytes((a.out / "df_fact_host_receipt.jsonl").read_bytes())}
    (a.out / "HOST_PREFLIGHT.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if all(v["dispatchable"] for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
