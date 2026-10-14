"""``worker --once``: admit, claim and supervise exactly one I6-DMO unit for one host role.

Runs on the coordinator (one systemd timer per role); the child runs on the role's host, locally
or through ``ssh <alias> systemd-run --user`` so that it survives a coordinator restart and can be
re-attached by its deterministic unit name. Per role at most one worker runs (file lock).

Admission (PLAN section 6, handoff C/D): live RAM, slice and disk headroom, the GPU's free memory
and no foreign heavy GPU process on the host; the cap is 1.25 x the measured full-tree peak when one
exists, else the declared initial cap (<= 7936 MiB on the preferred host); a refused cap is
remembered and never shrunk. The preferred role is tried first; a later role takes a GPU unit only
after every earlier GPU role refused it or holds another heavy lease. Only the role's single heavy
GPU is ever used: the preferred host's second GPU is not a parallel slot. No CPU fallback.
"""
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from tools.i6n_swarm.common import atomic_json, boot_id, file_sha256

from tools.i6dmo import campaign as C

GIB = 1024 ** 3
MIB = 1024 ** 2
HOST_RESERVE = 2 * GIB
MIN_DISK = 20 * GIB
MIN_VRAM = 2 * GIB
CPU_CAP = 4 * GIB
TTL = 900
POLL = 30.0
TERMINAL_STATUSES = ("COMPLETE", "FAILED", "NOT_ELIGIBLE", "PAIR_NOT_ELIGIBLE")


def attempt_dir(root, unit_id, index):
    return Path(root) / "attempts" / unit_id / f"a{index:02d}"


def cap_for(q, unit_id, device_class, *, initial_cap, measured_peak=None):
    cap = math.ceil(1.25 * measured_peak) if measured_peak else int(initial_cap)
    return max(cap, q.min_cap(unit_id))


def admit(facts, *, cap, gpu_uuid, reserve=HOST_RESERVE):
    evidence = dict(cap_bytes=cap, mem_available_bytes=facts.get("mem_available_bytes"),
                    slice_available_bytes=facts.get("slice_available_bytes"), disk_free_bytes=facts.get("disk_free_bytes"),
                    reserve_bytes=reserve)
    if facts.get("foreign_heavy_gpu_processes", 0):
        return False, "FOREIGN_HEAVY_GPU_PROCESS", evidence
    if facts.get("nv_err_no_memory", 0):
        return False, "NV_ERR_NO_MEMORY", evidence
    if facts.get("mem_available_bytes", 0) < cap + reserve:
        return False, "HOST_MEMORY_BELOW_GATE", evidence
    if facts.get("slice_available_bytes", 0) < cap:
        return False, "SLICE_BELOW_CAP", evidence
    if facts.get("disk_free_bytes", 0) < MIN_DISK:
        return False, "DISK_BELOW_GATE", evidence
    if gpu_uuid is not None:
        free = (facts.get("gpu_free_bytes") or {}).get(gpu_uuid)
        evidence["gpu_free_bytes"] = free
        if free is None:
            return False, "GPU_NOT_VISIBLE", evidence
        if free < MIN_VRAM:
            return False, "GPU_MEMORY_BELOW_GATE", evidence
    return True, None, evidence


# ------------------------------------------------------------------ executors
def _environ_has(pid, marker):
    try:
        return marker.encode() in Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False


class LocalExecutor:
    """Child on this host. Re-attach works while the process lives (pid + boot + attempt marker)."""

    def start(self, *, name, argv, env, cwd, folder, cap_bytes=None):
        folder = Path(folder)
        out = open(folder / "child.stdout", "ab")
        err = open(folder / "child.stderr", "ab")
        env = dict(env, I6DMO_ATTEMPT=name)
        proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        out.close()
        err.close()
        launch = dict(executor="local", name=name, pid=proc.pid, boot_id=boot_id(), started_at=time.time())
        return dict(launch=launch, proc=proc)

    def attach(self, launch):
        return dict(launch=launch, proc=None)

    def alive(self, launch):
        if launch.get("executor") != "local" or launch.get("boot_id") != boot_id():
            return False
        return _environ_has(launch["pid"], f"I6DMO_ATTEMPT={launch['name']}")

    def poll(self, handle):
        if handle["proc"] is not None:
            return handle["proc"].poll()
        return None if self.alive(handle["launch"]) else "GONE"

    def stop(self, handle):
        proc = handle.get("proc")
        pid = handle["launch"]["pid"]
        try:
            os.killpg(pid, 15)
        except ProcessLookupError:
            return
        if proc is not None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(pid, 9)
                proc.wait()

    def fetch(self, handle, folder):
        return None

    def progress(self, handle, folder):
        try:
            return json.loads((Path(folder) / "progress.json").read_text())
        except (OSError, ValueError):
            return None


class SshSystemdExecutor:
    """Child on another host as a transient user unit with a deterministic name (survives our restart)."""

    def __init__(self, alias, local_root, remote_root, run=subprocess.run):
        self.alias, self.local_root, self.remote_root, self.run = alias, Path(local_root), Path(remote_root), run

    def _remote(self, folder):
        return self.remote_root / Path(folder).resolve().relative_to(self.local_root.resolve())

    def _ssh(self, command, check=True):
        out = self.run(["ssh", "-o", "BatchMode=yes", self.alias, "--", command], capture_output=True, text=True,
                       timeout=120)
        if check and out.returncode != 0:
            raise RuntimeError(f"SSH_FAILED rc={out.returncode}")
        return out

    def start(self, *, name, argv, env, cwd, folder, cap_bytes=None):
        remote = self._remote(folder)
        self._ssh(f"mkdir -p {shlex.quote(str(remote))}")
        sync = self.run(["rsync", "-a", f"{folder}/", f"{self.alias}:{remote}/"], capture_output=True, timeout=600)
        if sync.returncode != 0:
            raise RuntimeError("RSYNC_TO_HOST_FAILED")
        argv = [str(a).replace(str(Path(folder)), str(remote)) for a in argv]
        props = ["--user", f"--unit={name}", "--collect", "-p", f"WorkingDirectory={cwd}", "-p", "KillMode=mixed",
                 "-E", f"I6DMO_ATTEMPT={name}"]
        if cap_bytes:
            props += ["-p", f"MemoryMax={int(cap_bytes)}", "-p", "MemorySwapMax=0"]
        for key in ("CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "TF_CPP_MIN_LOG_LEVEL", "I6DMO_ATTEMPT",
                    *env.get("I6DMO_FORWARD_ENV", "").split(",")):
            if key and key in env and key != "I6DMO_FORWARD_ENV":
                props += ["-E", f"{key}={env[key]}"]
        # The child's own exit code is written next to its outputs: a collected transient unit forgets it.
        exit_file = str(remote / "EXIT_CODE")
        wrapper = ['/bin/sh', '-c', '"$@"; rc=$?; echo "$rc" > ' + shlex.quote(exit_file) + '; exit "$rc"', "i6dmo"]
        command = " ".join(shlex.quote(p) for p in ["systemd-run", *props, "--", *wrapper, *argv])
        self._ssh(command)
        remote_boot = self._ssh("cat /proc/sys/kernel/random/boot_id").stdout.strip()
        launch = dict(executor="ssh-systemd", name=name, alias_role=None, boot_id=remote_boot,
                      remote_folder=str(remote), started_at=time.time())
        return dict(launch=launch, proc=None)

    def attach(self, launch):
        return dict(launch=launch, proc=None)

    def _show(self, name):
        out = self._ssh(f"systemctl --user show {shlex.quote(name)} -p ActiveState -p ExecMainStatus", check=False)
        values = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
        return values

    def alive(self, launch):
        if launch.get("executor") != "ssh-systemd":
            return False
        boot = self._ssh("cat /proc/sys/kernel/random/boot_id", check=False).stdout.strip()
        return boot == launch.get("boot_id") and self._show(launch["name"]).get("ActiveState") in (
            "active", "activating", "reloading")

    def poll(self, handle):
        values = self._show(handle["launch"]["name"])
        if values.get("ActiveState") in ("active", "activating", "reloading", "deactivating"):
            return None
        remote = handle["launch"]["remote_folder"]
        out = self._ssh(f"cat {shlex.quote(remote + '/EXIT_CODE')}", check=False)
        text = out.stdout.strip()
        if out.returncode == 0 and text.lstrip("-").isdigit():
            return int(text)
        return "GONE"  # no recorded exit code: never read as success

    def stop(self, handle):
        self._ssh(f"systemctl --user stop {shlex.quote(handle['launch']['name'])}", check=False)

    def fetch(self, handle, folder):
        remote = handle["launch"]["remote_folder"]
        out = self.run(["rsync", "-a", f"{self.alias}:{remote}/", f"{folder}/"], capture_output=True, timeout=3600)
        if out.returncode != 0:
            raise RuntimeError("RSYNC_FROM_HOST_FAILED")

    def progress(self, handle, folder):
        remote = handle["launch"]["remote_folder"]
        out = self._ssh(f"cat {shlex.quote(remote + '/progress.json')}", check=False)
        try:
            return json.loads(out.stdout)
        except ValueError:
            return None


def executor_for(root, spec, role, run=subprocess.run):
    role_spec = spec["roles"][role]
    if not role_spec.get("ssh_alias"):
        return LocalExecutor()
    return SshSystemdExecutor(role_spec["ssh_alias"], Path(root), Path(role_spec["run_root"]) /
                              C.load_plan(root)["campaign_id"], run=run)


# ------------------------------------------------------------------ terminals
def _verify_stored(folder, unit):
    """Verify a stored terminal envelope against the unit's identity (adoption and late collection)."""
    path = Path(folder) / "TERMINAL.json"
    doc = json.loads(path.read_text())
    if doc.get("schema") != "i6dmo.stored_terminal.v1":
        raise ValueError("schema")
    if doc.get("identity_sha256") != unit["identity_sha256"]:
        raise ValueError("IDENTITY_MISMATCH")
    if C.unit_identity(doc["identity"]) != unit["identity_sha256"]:
        raise ValueError("IDENTITY_COMPONENTS_MISMATCH")
    child = Path(folder) / "CHILD_TERMINAL.json"
    if hashlib.sha256(child.read_bytes()).hexdigest() != doc["child_terminal_sha256"]:
        raise ValueError("CHILD_TERMINAL_DIGEST")
    for name, art in doc["terminal"].get("artifacts", {}).items():
        if file_sha256(Path(folder) / "out" / art["path"]) != art["sha256"]:
            raise ValueError(f"ARTIFACT_DIGEST:{name}")
    return doc


def _validate_child(doc, unit):
    if doc.get("schema") != "i6dmo.terminal.v1":
        raise ValueError("TERMINAL_SCHEMA")
    if doc.get("unit_id") != unit["unit_id"] or doc.get("identity_sha256") != unit["identity_sha256"]:
        raise ValueError("TERMINAL_IDENTITY_MISMATCH")
    if doc.get("status") not in TERMINAL_STATUSES:
        raise ValueError("TERMINAL_STATUS")
    if unit["kind"] in ("report", "pair", "final_test") and doc.get("costs", {}).get("fit_seconds"):
        raise ValueError("REPORT_UNIT_FITTED: a report/pair/test unit must not fit")


def _enqueue_reports(root, unit_id, doc, q):
    from tools.i6dmo.outbox import Outbox
    box = Outbox(Path(root) / "state" / "outbox.sqlite3")
    rejected = 0
    try:
        for report in doc.get("reports", []):
            try:
                box.enqueue(report, source_unit=unit_id)
            except ValueError as exc:
                rejected += 1
                q.event("REPORT_REJECTED", unit_id, dict(error=str(exc)[:200]))
    finally:
        box.close()
    return rejected


def _store(root, unit, folder, attempt_info):
    out = Path(folder) / "out"
    raw = (out / "TERMINAL.json").read_bytes()
    doc = json.loads(raw)
    _validate_child(doc, unit)
    for name, art in doc.get("artifacts", {}).items():
        if file_sha256(out / art["path"]) != art["sha256"]:
            raise ValueError(f"ARTIFACT_DIGEST:{name}")
    target = Path(root) / "terminals" / unit["identity_sha256"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(out, target / "out")
    (target / "CHILD_TERMINAL.json").write_bytes(raw)
    child_sha = hashlib.sha256(raw).hexdigest()
    atomic_json(target / "TERMINAL.json", dict(
        schema="i6dmo.stored_terminal.v1", unit_id=unit["unit_id"], identity=unit["identity"],
        identity_sha256=unit["identity_sha256"], child_terminal_sha256=child_sha, terminal=doc, attempt=attempt_info,
        stored_at=time.time()))
    return doc, child_sha


def _finish(root, q, unit, token, rc, folder, attempt_info):
    unit_id = unit["unit_id"]
    rel = str(Path(folder).relative_to(root))
    if rc in (137, -9):
        q.fail(unit_id, token, "FAILED_OOM", f"rc={rc}")
        return dict(result="FAILED", unit_id=unit_id, cause="FAILED_OOM", attempt_dir=rel)
    if rc == 75:  # the launcher refused admission before starting anything: wait, never a failure
        q.interrupt(token, "LAUNCHER_ADMISSION_REFUSED")
        attempt = q.attempt_by_token(token)
        q.record_admission(unit_id, attempt["role"], False, "LAUNCHER_ADMISSION_REFUSED", attempt["cap_bytes"])
        return dict(result="WAIT_RESOURCE", unit_id=unit_id, cause="LAUNCHER_ADMISSION_REFUSED", attempt_dir=rel)
    if rc == 124:
        q.fail(unit_id, token, "FAILED_TIMEOUT", f"rc={rc}")
        return dict(result="FAILED", unit_id=unit_id, cause="FAILED_TIMEOUT", attempt_dir=rel)
    terminal = Path(folder) / "out" / "TERMINAL.json"
    if rc == "GONE" and not terminal.exists():
        q.interrupt(token, "CHILD_VANISHED")
        return dict(result="INTERRUPTED", unit_id=unit_id, cause="CHILD_VANISHED", attempt_dir=rel)
    if rc not in (0, "GONE"):
        q.fail(unit_id, token, "FAILED", f"EXIT_{rc}")
        return dict(result="FAILED", unit_id=unit_id, cause=f"EXIT_{rc}", attempt_dir=rel)
    try:
        doc, child_sha = _store(root, unit, folder, attempt_info)
    except (OSError, ValueError, KeyError) as exc:
        q.fail(unit_id, token, "FAILED", f"TERMINAL_INVALID:{exc}"[:200])
        return dict(result="FAILED", unit_id=unit_id, cause=f"TERMINAL_INVALID:{exc}", attempt_dir=rel)
    costs = doc.get("costs", {})
    if doc["status"] == "FAILED":  # typed scientific/contract failure: evidence kept, never retried
        q.fail(unit_id, token, "FAILED", f"TYPED:{doc.get('disposition')}"[:200],
               peak_bytes=costs.get("cgroup_peak_bytes"), costs=costs)
        return dict(result="FAILED", unit_id=unit_id, cause=f"TYPED:{doc.get('disposition')}", attempt_dir=rel)
    q.complete(unit_id, token, doc, child_sha, identity_sha256=unit["identity_sha256"],
               peak_bytes=costs.get("cgroup_peak_bytes"), costs=costs)
    rejected = _enqueue_reports(root, unit_id, doc, q)
    return dict(result="COMPLETE", unit_id=unit_id, status=doc["status"], attempt_dir=rel, reports_rejected=rejected)


def _adoptable(root, unit, prior_roots, q):
    for base in [Path(root), *map(Path, prior_roots)]:
        folder = base / "terminals" / unit["identity_sha256"]
        if not (folder / "TERMINAL.json").exists():
            continue
        try:
            doc = _verify_stored(folder, unit)
            if doc["terminal"].get("status") == "FAILED":
                raise ValueError("FAILED_TERMINAL_NOT_ADOPTED")
        except (OSError, ValueError, KeyError) as exc:
            q.event("PRIOR_REJECTED", unit["unit_id"], dict(source=str(base.name), reason=str(exc)[:200]))
            continue
        target = Path(root) / "terminals" / unit["identity_sha256"]
        if folder.resolve() != target.resolve():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(folder, target)
        return doc, base
    return None, None


# ------------------------------------------------------------------- picking
def _role_gpu(role_spec):
    return next((g for g in role_spec.get("gpus", []) if g.get("heavy")), None)


def _pick(q, plan, spec, role):
    role_spec = spec["roles"][role]
    gpu = _role_gpu(role_spec)
    order = role_spec.get("order", 0)
    earlier = [r for r, v in spec["roles"].items() if _role_gpu(v) and v.get("order", 0) < order]
    rows = {u["unit_id"]: u for u in q.units()}
    running_hosts = {a["host_key"] for a in q.running() if a["heavy"]}
    for unit in plan["units"]:
        row = rows.get(unit["unit_id"])
        if row is None or row["state"] not in ("QUEUED", "WAIT_RESOURCE", "INTERRUPTED", "WAIT_ARTIFACT",
                                               "WAIT_OWNER_RULING") \
                or not row["identity_sha256"]:
            continue
        if any(rows[d]["state"] != "COMPLETE" for d in row["spec"]["depends_on"]):
            continue
        if row["device"] == "gpu":
            if gpu is None:
                continue
            refused = set(q.refused_roles(unit["unit_id"]))
            blocked = [r for r in earlier if r not in refused and spec["roles"][r]["host_key"] not in running_hosts]
            if blocked:
                continue
        elif not role_spec.get("cpu_units"):
            continue
        return row
    return None


# ------------------------------------------------------------------- recovery
def recover_role(root, role, *, executor=None, facts_provider=None, clock=time.time):
    """Orphaned attempts of this role: re-attach a live child, collect a finished one, else INTERRUPTED."""
    root = Path(root)
    spec = C.load_deployment(root)
    executor = executor or executor_for(root, spec, role)
    q = C.open_queue(root, clock=clock)
    out = dict(interrupted=[], reattached=[], collected=[])
    try:
        for attempt in q.running(role):
            launch = attempt["launch"]
            if launch and executor.alive(launch):
                out["reattached"].append(dict(unit_id=attempt["unit_id"], token=attempt["token"], launch=launch))
                continue
            folder = attempt_dir(root, attempt["unit_id"], attempt["idx"])
            if launch:
                try:
                    executor.fetch(executor.attach(launch), folder)
                except Exception:  # unreachable host: the attempt is interrupted, not complete
                    pass
            if launch and (folder / "out" / "TERMINAL.json").exists():
                unit = q.unit(attempt["unit_id"])
                info = dict(index=attempt["idx"], kind=attempt["kind"], role=role,
                            device_class=attempt["device_class"], token=attempt["token"])
                spec_unit = dict(unit["spec"], identity_sha256=unit["identity_sha256"])
                _finish(root, q, spec_unit, attempt["token"], 0, folder, info)
                out["collected"].append(attempt["unit_id"])
                continue
            q.interrupt(attempt["token"], "SUPERVISOR_LOST" if launch else "NOT_LAUNCHED")
            out["interrupted"].append(attempt["unit_id"])
    finally:
        q.close()
    return out


# -------------------------------------------------------------------- worker
def _child_argv(entries, kind, python, folder):
    argv = [python if a == "{python}" else a for a in entries[kind]]
    return argv + ["run-unit", "--spec", str(folder / "UNIT_SPEC.json"), "--out", str(folder / "out"),
                   "--progress", str(folder / "progress.json")]


def _wrap(launcher, name, cap, argv):
    crispdm = launcher.get("crispdm_run")
    if not crispdm:
        return argv
    return [crispdm, "-m", str(int(cap)), "-t", f"{int(launcher['wall_seconds'])}s", "-n", name, "-L", name,
            "--", *argv]


def _supervise(root, q, unit, token, handle, executor, folder, ttl, poll_interval, attempt_info):
    while True:
        rc = executor.poll(handle)
        if rc is not None:
            break
        if not q.heartbeat(unit["unit_id"], token, ttl, executor.progress(handle, folder)):
            executor.stop(handle)
            return dict(result="LOST_LEASE", unit_id=unit["unit_id"])
        time.sleep(poll_interval)
    progress = executor.progress(handle, folder)
    if progress:
        q.heartbeat(unit["unit_id"], token, ttl, progress)
    executor.fetch(handle, folder)
    return _finish(root, q, unit, token, rc, folder, attempt_info)


def worker_once(root, role, *, facts_provider=None, executor=None, poll_interval=POLL, ttl=TTL, extra_env=None,
                clock=time.time):
    root = Path(root)
    effective = C.load_effective(root)
    spec = C.load_deployment(root)
    if role not in spec["roles"]:
        raise C.Refused(f"UNKNOWN_ROLE: {role}")
    lock_path = root / "state" / f"worker-{role}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return dict(result="BUSY", role=role)
    try:
        return _worker_locked(root, role, effective, spec, facts_provider, executor, poll_interval, ttl,
                              extra_env or {}, clock)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _worker_locked(root, role, effective, spec, facts_provider, executor, poll_interval, ttl, extra_env, clock):
    from tools.i6dmo import preflight
    executor = executor or executor_for(root, spec, role)
    facts_provider = facts_provider or (lambda r: preflight.probe_role(spec, r))
    recovered = recover_role(root, role, executor=executor, clock=clock)
    q = C.open_queue(root, clock=clock)
    try:
        if recovered["reattached"]:
            item = recovered["reattached"][0]
            attempt = q.attempt_by_token(item["token"])
            unit = q.unit(item["unit_id"])
            unit = dict(unit["spec"], identity_sha256=unit["identity_sha256"])
            folder = attempt_dir(root, unit["unit_id"], attempt["idx"])
            info = dict(index=attempt["idx"], kind=attempt["kind"], role=role, device_class=attempt["device_class"],
                        token=attempt["token"], reattached=True)
            return _supervise(root, q, unit, item["token"], executor.attach(item["launch"]), executor, folder, ttl,
                              poll_interval, info)
        plan = C.load_plan(root)
        row = _pick(q, plan, spec, role)
        if row is None:
            return dict(result="IDLE", role=role)
        unit = dict(row["spec"], identity_sha256=row["identity_sha256"])
        doc, source = _adoptable(root, unit, effective.get("prior_roots", []), q)
        if doc is not None:
            q.adopt(unit["unit_id"], doc["terminal"], doc["child_terminal_sha256"],
                    identity_sha256=unit["identity_sha256"], source_detail=dict(source_root_name=source.name))
            _enqueue_reports(root, unit["unit_id"], doc["terminal"], q)
            return dict(result="ADOPTED", unit_id=unit["unit_id"])
        role_spec = spec["roles"][role]
        bundle, missing = _artifacts(root, effective, unit, role_spec)
        if missing:
            state = "WAIT_OWNER_RULING" if missing.startswith("WAIT_OWNER_RULING") else "WAIT_ARTIFACT"
            q.set_state(unit["unit_id"], state, missing)
            return dict(result=state, unit_id=unit["unit_id"], cause=missing)
        gpu = _role_gpu(role_spec) if unit["device"] == "gpu" else None
        device_class = gpu["class"] if gpu else "cpu"
        launcher = dict(spec["launcher"], **role_spec.get("launcher", {}))
        if gpu:
            # The resource class is the model family+kind, not the GPU: a full-tree peak measured on
            # one admitted GPU carries to another of the same class. The launcher may seed a class
            # peak measured in a prior campaign; a local measurement, once it exists, wins.
            class_key = f"{unit['family']}.{unit['kind']}"
            measured = q.measured_peak(kind=unit["kind"], family=unit["family"]) \
                or (launcher.get("measured_class_peaks") or {}).get(class_key)
            cap = cap_for(q, unit["unit_id"], device_class, initial_cap=launcher["initial_cap_bytes"],
                          measured_peak=measured)
        else:
            cap = max(int(launcher.get("cpu_cap_bytes", CPU_CAP)), q.min_cap(unit["unit_id"]))
        facts = facts_provider(role)
        ok, cause, evidence = admit(facts, cap=cap, gpu_uuid=gpu["uuid"] if gpu else None,
                                    reserve=int(role_spec.get("reserve_bytes", HOST_RESERVE)))
        q.record_admission(unit["unit_id"], role, ok, cause, cap, evidence)
        if not ok:
            return dict(result="WAIT_RESOURCE", unit_id=unit["unit_id"], cause=cause, cap_bytes=cap)
        claim = q.claim(unit["unit_id"], owner=f"worker:{role}", role=role, host_key=role_spec["host_key"],
                        boot_id=facts.get("boot_id", "unknown"), device_class=device_class, heavy=bool(gpu), ttl=ttl,
                        cap_bytes=cap)
        if claim["result"] != "CLAIMED":
            return dict(claim, unit_id=unit["unit_id"])
        attempt = claim["attempt"]
        folder = attempt_dir(root, unit["unit_id"], attempt["index"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "out").mkdir(exist_ok=True)
        unit_spec = dict(unit, schema="i6dmo.unit_spec.v1", campaign_id=plan["campaign_id"],
                         effective_config=effective, effective_config_sha256=plan["effective_config_sha256"],
                         device=dict(cls=device_class, gpu_uuid=gpu["uuid"] if gpu else None), cap_bytes=cap,
                         data_root=role_spec.get("data_root"), bundle=bundle,
                         pins=resolved_pins(effective, role_spec), attempt=dict(index=attempt["index"], kind=attempt["kind"]),
                         upstream={d: _upstream(root, q, d) for d in unit["depends_on"]})
        atomic_json(folder / "UNIT_SPEC.json", unit_spec)
        name = f"{plan['campaign_id']}-{unit['unit_id'].replace('.', '-')}-a{attempt['index']:02d}"
        role_env = {k: str(v) for k, v in (role_spec.get("environment") or {}).items() if k != "CUDA_VISIBLE_DEVICES"}
        env = dict(os.environ, **extra_env, **role_env, CUDA_VISIBLE_DEVICES=gpu["uuid"] if gpu else "",
                   PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2", I6DMO_FORWARD_ENV=",".join(sorted(role_env)))
        argv = _wrap(launcher, name, cap, _child_argv(effective["entrypoints"], unit["kind"],
                                                     role_spec.get("python", "python3"), folder))
        info = dict(index=attempt["index"], kind=attempt["kind"], role=role, device_class=device_class,
                    token=claim["token"], cap_bytes=cap)
        try:
            handle = executor.start(name=name, argv=argv, env=env, cwd=role_spec.get("workdir"), folder=folder,
                                    cap_bytes=cap)
        except Exception as exc:  # nothing started: interrupted, not a scientific failure
            q.interrupt(claim["token"], f"LAUNCH_FAILED:{str(exc)[:120]}")
            return dict(result="INTERRUPTED", unit_id=unit["unit_id"], cause="LAUNCH_FAILED")
        q.set_launch(claim["token"], folder.relative_to(root), handle["launch"])
        return _supervise(root, q, unit, claim["token"], handle, executor, folder, ttl, poll_interval, info)
    finally:
        q.close()


def _uses_unit_runner(argv):
    return "-m" in argv and argv[argv.index("-m") + 1:argv.index("-m") + 2] == ["tools.i6dmo.unit_runner"]


def bundle_pin(root, effective, unit):
    """Data-lane bundle pin: sealed data_pins for validation fits; DATA_PROVENANCE evidence for final refits."""
    family = unit["family"]
    if unit["kind"] == "fit":
        key, pins = f"{family}_bundle", effective["data_pins"]
    elif unit["kind"] == "final_refit":
        key = f"final_{family}_bundle"
        gate = Path(root) / "gates" / "DATA_PROVENANCE.json"
        pins = json.loads(gate.read_text()).get("evidence", {}).get("final_bundles", {}) if gate.exists() else {}
    else:
        return None, None
    return key, pins.get(key)


def _artifacts(root, effective, unit, role_spec):
    import importlib.util
    from tools.i6dmo import unit_runner
    argv = effective["entrypoints"][unit["kind"]]
    key, pin = bundle_pin(root, effective, unit)
    bundle = None
    if pin is not None:
        bundle = dict(key=key, path=str(Path(role_spec.get("data_root") or "") / pin["relpath"]), sha256=pin["sha256"],
                      bundle_sha256=pin.get("bundle_sha256"))
    if unit["kind"] in ("pair", "final_test"):
        problem = C.gate_problem(root, "STRATEGY_OWNER_RULINGS")
        if problem:
            return bundle, (f"WAIT_OWNER_RULING:{problem}: " + ",".join(C.OWNER_RULINGS) + " owner=owner")
        cause = _pair_inputs_missing(root, effective, unit, role_spec)
        if cause:
            return bundle, cause
    if not _uses_unit_runner(argv):
        return bundle, None
    module = unit_runner.hook_module(unit["kind"])
    owner = unit_runner.HOOK_OWNERS.get(module, "model lane")
    if importlib.util.find_spec(module) is None:
        return bundle, f"ENTRY_UNAVAILABLE:{module} owner={owner}"
    if unit["kind"] not in unit_runner.FIT_KINDS:
        try:
            ready = callable(getattr(importlib.import_module(module), "run_unit", None))
        except Exception:  # an import error is the same missing artifact
            ready = False
        if not ready:
            return bundle, f"ENTRY_UNAVAILABLE:{module}.run_unit owner={owner}"
    if key is not None and pin is None:
        return bundle, f"BUNDLE_PIN_MISSING:{key} owner=data lane (tools.i6dmo.data.build_frozen_bundle)"
    if pin is not None and pin.get("data_contract_sha256") and unit["family"] in ("short", "long"):
        design = effective["long"]["design"] if unit["family"] == "long" else effective["design"]
        from tools.i6dmo import data as data_lane
        if data_lane.data_contract_sha256(design) != pin["data_contract_sha256"]:
            return bundle, (f"DATA_CONTRACT_MISMATCH:{key} sealed {unit['family']} design differs from the bundle's "
                            "data view owner=data lane / model lane")
    hook = importlib.import_module(module)
    if hasattr(hook, "preconditions"):
        cause = hook.preconditions(dict(unit=unit, effective_config=effective, data_root=role_spec.get("data_root")))
        if cause:
            return bundle, cause
    return bundle, None


STRATEGY_PINS = ("strategy_bars", "strategy_calendar")
PREDICTION_ARTIFACTS = ("validation_predictions", "predictions/validation.npz")


def _pair_inputs_missing(root, effective, unit, role_spec):
    """Keep a pair/test unit unclaimed while its hook would only return a final FAILED terminal."""
    pins = effective.get("data_pins") or {}
    data_root = Path(role_spec.get("data_root") or "")
    for key in STRATEGY_PINS:
        pin = pins.get(key)
        if not pin or "relpath" not in pin:
            return f"STRATEGY_INPUT_PIN_MISSING:{key} owner=data lane / Satoshi (private pins)"
        path = data_root / pin["relpath"]
        if not path.is_file() or file_sha256(path) != pin.get("sha256"):
            return f"STRATEGY_INPUT_DIGEST_MISMATCH:{key} owner=Satoshi (replicate to the executing role)"
    q = C.open_queue(root)
    try:
        for dep in unit["depends_on"]:
            term = q.terminal(dep)
            if term is None:
                return f"UPSTREAM_NOT_TERMINAL:{dep}"
            stored = Path(root) / "terminals" / (term["identity_sha256"] or "none") / "TERMINAL.json"
            arts = (term["terminal"].get("artifacts") or {}) if stored.exists() else {}
            if dep.endswith(".report") and not any(n in arts for n in PREDICTION_ARTIFACTS):
                return f"UPSTREAM_ARTIFACT_MISSING:{dep}:validation_predictions owner=ops report unit"
    finally:
        q.close()
    return None


def resolved_pins(effective, role_spec):
    """Every sealed data pin as an absolute path on the executing role's data root."""
    root = Path(role_spec.get("data_root") or "")
    return {key: dict(pin, path=str(root / pin["relpath"])) for key, pin in (effective.get("data_pins") or {}).items()
            if "relpath" in pin}


def _upstream(root, q, unit_id):
    term = q.terminal(unit_id)
    if term is None or not term["identity_sha256"]:
        return None
    path = Path(root) / "terminals" / term["identity_sha256"] / "TERMINAL.json"
    return dict(path=str(path), sha256=file_sha256(path), terminal_sha256=term["terminal_sha256"])
