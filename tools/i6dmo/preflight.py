"""Bounded preflight: pinned paths, design, real host/slice/disk/GPU headroom, legacy units, strategy pin.

``probe`` runs on each host (locally, or as ``ssh <alias> -- cd <workdir> && <python> -m tools.i6dmo
probe --json``) and only reads /proc, cgroup files, ``nvidia-smi`` queries, ``systemctl --user
list-units`` and file digests. It launches no GPU process and reads no credential.
``preflight`` aggregates the probes into PRECHECK.json with a typed state per check; it never
claims a launch. Checks carry ``blocks``: LAUNCH (first fit), PAIR (strategy stage) or TEST.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from tools.i6n_swarm.common import atomic_json, boot_id, sha256_json

GIB = 1024 ** 3
HEAVY_GPU_PROCESS = GIB
LEGACY_PREFIXES = ("i6n-", "doin-")


def _meminfo(path="/proc/meminfo"):
    values = {}
    with open(path) as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    return values


def _cgroup_available():
    uid = os.getuid()
    base = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice")
    candidates = [base, base / f"user@{uid}.service" / "crispdm-batch.slice"]
    best = None
    for path in candidates:
        try:
            limit = (path / "memory.max").read_text().strip()
            current = int((path / "memory.current").read_text().strip())
        except (OSError, ValueError):
            continue
        if limit == "max":
            continue
        free = int(limit) - current
        best = free if best is None else min(best, free)
    return best


def _nvidia(run):
    try:
        gpus = run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return dict(gpus=[], gpu_free_bytes={}, heavy_processes=[], nvidia_smi="UNAVAILABLE")
    listed, free = [], {}
    for line in gpus.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            listed.append(dict(uuid=parts[0], name=parts[1], total_bytes=int(parts[2]) * 1024 ** 2,
                               used_bytes=int(parts[3]) * 1024 ** 2))
            free[parts[0]] = int(parts[4]) * 1024 ** 2
    heavy = []
    for line in apps.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit() and int(parts[2]) * 1024 ** 2 >= HEAVY_GPU_PROCESS:
            heavy.append(dict(gpu_uuid=parts[0], pid=int(parts[1]), used_bytes=int(parts[2]) * 1024 ** 2))
    return dict(gpus=listed, gpu_free_bytes=free, heavy_processes=heavy,
                nvidia_smi="OK" if gpus.returncode == 0 else f"RC_{gpus.returncode}")


def _legacy_units(run):
    try:
        out = run(["systemctl", "--user", "list-units", "--state=active", "--plain", "--no-legend"],
                  capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [line.split()[0] for line in out.stdout.splitlines() if line.strip()]
    return sorted(n for n in names if n.startswith(LEGACY_PREFIXES))


def _strategy_pin():
    try:
        from tools.i6dmo import strategy_adapter
        policy = strategy_adapter.load_pinned_policy()
        return dict(state="PASS", identity=strategy_adapter.policy_identity(policy))
    except ImportError:
        return dict(state="FAIL", code="STRATEGY_ADAPTER_UNAVAILABLE")
    except Exception as exc:  # typed StrategyAdapterError and git failures
        return dict(state="FAIL", code="STRATEGY_PIN_UNRESOLVED", detail=str(exc)[:200])


def _file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_local(*, run_root=None, data_root=None, files=(), check_strategy=False, run=subprocess.run):
    mem = _meminfo()
    slice_free = _cgroup_available()
    target = Path(run_root or Path.home())
    while not target.exists() and target != target.parent:
        target = target.parent
    gpu = _nvidia(run)
    facts = dict(schema="i6dmo.probe.v1", at=time.time(), boot_id=boot_id(), mem_total_bytes=mem.get("MemTotal", 0),
                 mem_available_bytes=mem.get("MemAvailable", 0), sunreclaim_bytes=mem.get("SUnreclaim", 0),
                 slice_available_bytes=slice_free if slice_free is not None else mem.get("MemAvailable", 0),
                 slice_limited=slice_free is not None, disk_free_bytes=shutil.disk_usage(target).free,
                 gpus=gpu["gpus"], gpu_free_bytes=gpu["gpu_free_bytes"], nvidia_smi=gpu["nvidia_smi"],
                 heavy_gpu_processes=gpu["heavy_processes"], foreign_heavy_gpu_processes=len(gpu["heavy_processes"]),
                 legacy_active_units=_legacy_units(run))
    if files:
        found = {}
        for rel in files:
            path = Path(data_root or "") / rel
            found[rel] = _file_sha(path) if path.is_file() else None
        facts["files"] = found
    if check_strategy:
        facts["strategy_pin"] = _strategy_pin()
    return facts


def probe_role(spec, role, *, files=(), check_strategy=None, run=subprocess.run):
    role_spec = spec["roles"][role]
    check_strategy = bool(role_spec.get("cpu_units")) if check_strategy is None else check_strategy
    if not role_spec.get("ssh_alias"):
        return probe_local(run_root=role_spec.get("run_root"), data_root=role_spec.get("data_root"), files=files,
                           check_strategy=check_strategy, run=run)
    args = [role_spec["python"], "-m", "tools.i6dmo", "probe", "--json", "--run-root", role_spec["run_root"],
            "--data-root", role_spec.get("data_root") or ""]
    for rel in files:
        args += ["--file", rel]
    if check_strategy:
        args.append("--check-strategy")
    command = f"cd {shlex.quote(role_spec['workdir'])} && " + " ".join(shlex.quote(a) for a in args)
    out = run(["ssh", "-o", "BatchMode=yes", role_spec["ssh_alias"], "--", command], capture_output=True, text=True,
              timeout=180)
    if out.returncode != 0:
        raise RuntimeError(f"PROBE_FAILED role={role} rc={out.returncode}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _check(checks, name, ok, blocks, detail=None, owner="Satoshi integration", state=None):
    checks.append(dict(check=name, state=state or ("PASS" if ok else "FAIL"), blocks=blocks, owner=owner,
                       detail=detail))


def preflight(design_path, spec_path, *, adapter=None, probe=None, data_pins=None, out=None):
    from tools.i6dmo import campaign as C
    from tools.i6dmo import worker as W
    checks = []
    design = json.loads(Path(design_path).read_text())
    try:
        verdict = C._adapter(adapter).validate_design(design)
        _check(checks, "design", verdict.get("ok"), "LAUNCH", verdict, owner="model lane")
    except C.Refused as exc:
        _check(checks, "design", False, "LAUNCH", str(exc), owner="model lane")
    spec = json.loads(Path(spec_path).read_text())
    try:
        C._validate_spec(spec)
        _check(checks, "deployment_spec", True, "LAUNCH")
    except C.Refused as exc:
        _check(checks, "deployment_spec", False, "LAUNCH", str(exc))
        spec = None
    envfile = Path((spec or {}).get("warehouse", {}).get("environment_file", "/nonexistent"))
    _check(checks, "warehouse_environment_file_exists", envfile.exists(), "PAIR",
           "existence only; contents are never read", owner="existing fleet EnvironmentFile")
    pins = data_pins or {}
    for role, role_spec in sorted((spec or {}).get("roles", {}).items()):
        files = [p["relpath"] for p in pins.values() if "relpath" in p] if role_spec.get("gpus") else []
        try:
            facts = (probe or (lambda r, f: probe_role(spec, r, files=f)))(role, files)
        except Exception as exc:
            _check(checks, f"probe:{role}", False, "LAUNCH", str(exc)[:200])
            continue
        _check(checks, f"probe:{role}", True, "LAUNCH", dict(
            boot_id=facts.get("boot_id"), mem_available_bytes=facts.get("mem_available_bytes"),
            slice_available_bytes=facts.get("slice_available_bytes"), disk_free_bytes=facts.get("disk_free_bytes"),
            gpus=len(facts.get("gpus", [])), nvidia_smi=facts.get("nvidia_smi")))
        legacy = facts.get("legacy_active_units")
        _check(checks, f"legacy_units_stopped:{role}", legacy == [], "LAUNCH",
               dict(active=legacy) if legacy else None, state=None if legacy is not None else "UNVERIFIED")
        heavy = next((g for g in role_spec.get("gpus", []) if g.get("heavy")), None)
        for gpu in role_spec.get("gpus", []):
            visible = gpu["uuid"] in (facts.get("gpu_free_bytes") or {})
            _check(checks, f"gpu_visible:{role}:{gpu['class']}", visible, "LAUNCH" if gpu.get("heavy") else "TEST")
            if not gpu.get("heavy"):
                _check(checks, f"non_parallel_slot:{role}:{gpu['class']}", gpu.get("parallel_slot") is False, "LAUNCH",
                       "a second GPU on a heavy host is never a parallel slot")
        if heavy is not None:
            cap = int(spec["launcher"]["initial_cap_bytes"])
            ok, cause, evidence = W.admit(facts, cap=cap, gpu_uuid=heavy["uuid"],
                                          reserve=int(role_spec.get("reserve_bytes", W.HOST_RESERVE)))
            first = role_spec.get("order", 0) == min(v.get("order", 0) for v in spec["roles"].values()
                                                    if any(g.get("heavy") for g in v.get("gpus", [])))
            _check(checks, f"admission_initial_cap:{role}", ok, "LAUNCH" if first else "TEST",
                   dict(evidence, cause=cause, preferred=first))
        for key, pin in pins.items():
            if role_spec.get("gpus") and "relpath" in pin:
                got = (facts.get("files") or {}).get(pin["relpath"])
                _check(checks, f"data_pin:{role}:{key}", got == pin["sha256"], "LAUNCH",
                       dict(found=got is not None), owner="data lane")
        if role_spec.get("cpu_units"):
            sp = facts.get("strategy_pin") or dict(state="FAIL", code="STRATEGY_PIN_UNCHECKED")
            _check(checks, f"strategy_pin:{role}", sp["state"] == "PASS", "PAIR", sp, owner="strategy lane")
    if not pins:
        _check(checks, "data_pins", False, "LAUNCH", "no frozen bundle pins supplied (--data-pins)",
               owner="data lane", state="UNVERIFIED")
    launch_blockers = [c["check"] for c in checks if c["blocks"] == "LAUNCH" and c["state"] != "PASS"]
    doc = dict(schema="i6dmo.precheck.v1", state="PASS" if not launch_blockers else "FAIL",
               launch_blockers=launch_blockers, checks=checks, design_sha256=sha256_json(design),
               note="PRECHECK is evidence of headroom at probe time, not a launch")
    out = Path(out) if out else Path(spec_path).resolve().parent / "PRECHECK.json"
    atomic_json(out, doc, mode=0o600)
    return doc
