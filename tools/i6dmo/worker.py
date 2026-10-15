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
#: Units that produce large immutable artifacts; their claim re-checks experiment storage.
ARTIFACT_WRITING_KINDS = ("fit", "final_refit", "report", "pretrain")
# A terminal may only be adopted when its own recorded identity components match this unit's.
# Shape or horizon similarity is never sufficient: an elapsed-grid (DS2/ASOF) terminal must not be
# reused for a market-time identity even though both carry six heads and the same window shape.
IDENTITY_COMPONENT_KEYS = ("unit", "family", "kind", "cadence", "effective_config_sha256",
                           "data_sha256", "code_commit", "seed", "upstream")


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
    stored = doc.get("identity") or {}
    if set(stored) != set(IDENTITY_COMPONENT_KEYS):
        raise ValueError("IDENTITY_COMPONENTS_SCHEMA")
    if C.unit_identity(stored) != unit["identity_sha256"]:
        raise ValueError("IDENTITY_COMPONENTS_MISMATCH")
    if unit.get("identity") and stored != unit["identity"]:
        # same digest claimed, different components: a forged or cross-treatment terminal
        raise ValueError("IDENTITY_COMPONENTS_DIFFER_FROM_THIS_UNIT")
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
    if unit["kind"] in ("report", "pair", "strategy", "comparison", "final_test", "evaluation") and \
            doc.get("costs", {}).get("fit_seconds"):
        raise ValueError("REPORT_UNIT_FITTED: a report/pair/strategy/comparison/test/probe unit must not fit")


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


INFRA_PREFIT = "INFRA_PREFIT:"
#: Codes a child returns when the HOST/infrastructure refused before any training or scoring
#: started (wrong replica, missing GPU, wrong runtime). They are waits, never scientific terminals.
INFRA_PREFIT_CODES = frozenset({
    "BUNDLE_PIN_MISMATCH", "MATCHED_PIN_UNRESOLVED", "POPULATION_PIN_MISMATCH", "INPUT_PIN_MISMATCH",
    "PIN_MISSING", "NO_GPU_VISIBLE", "RUNTIME_PIN_MISMATCH", "POPULATION_DIGEST_MISMATCH",
    "FILE_DIGEST_MISMATCH", "SOURCE_DIGEST_MISMATCH", "CALENDAR_DIGEST_MISMATCH"})


def _infra_prefit_refusal(unit, terminal_path):
    """The child's typed pre-fit infrastructure refusal code, or None (anything else stays terminal)."""
    if unit["kind"] not in ("fit", "final_refit", "report", "pretrain"):
        return None
    try:
        doc = json.loads(Path(terminal_path).read_text())
    except (OSError, ValueError):
        return None
    if doc.get("status") != "FAILED":
        return None
    code = str(doc.get("disposition") or "").split(":", 1)[0]
    costs = doc.get("costs") or {}
    started = bool(costs.get("fit_seconds")) or bool((doc.get("training") or {}).get("epochs_completed"))
    return code if code in INFRA_PREFIT_CODES and not started else None


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


def _left_no_child_output(folder):
    """True only when an attempt left no trace of a started child (pre-execution refusal proof)."""
    folder = Path(folder)
    out = folder / "out"
    if (out / "TERMINAL.json").exists() or (folder / "progress.json").exists():
        return False
    if out.is_dir() and any(out.iterdir()):
        return False
    stdout = folder / "child.stdout"
    return not (stdout.exists() and stdout.stat().st_size > 0)


def _finish(root, q, unit, token, rc, folder, attempt_info):
    unit_id = unit["unit_id"]
    rel = str(Path(folder).relative_to(root))
    if rc in (137, -9):
        q.fail(unit_id, token, "FAILED_OOM", f"rc={rc}")
        return dict(result="FAILED", unit_id=unit_id, cause="FAILED_OOM", attempt_dir=rel)
    if rc == 75 and _left_no_child_output(folder):
        # The launcher refused admission before starting anything: wait, never a failure. Exit 75
        # alone is NOT that proof: any child output (a terminal, progress, stdout, anything under
        # out/) means something ran, and the ordinary failure path below applies (budget spent).
        q.interrupt(token, "LAUNCHER_ADMISSION_REFUSED")
        attempt = q.attempt_by_token(token)
        q.record_admission(unit_id, attempt["role"], False, "LAUNCHER_ADMISSION_REFUSED", attempt["cap_bytes"])
        return dict(result="WAIT_RESOURCE", unit_id=unit_id, cause="LAUNCHER_ADMISSION_REFUSED", attempt_dir=rel)
    if rc == 124:
        q.fail(unit_id, token, "FAILED_TIMEOUT", f"rc={rc}")
        return dict(result="FAILED", unit_id=unit_id, cause="FAILED_TIMEOUT", attempt_dir=rel)
    terminal = Path(folder) / "out" / "TERMINAL.json"
    infra = _infra_prefit_refusal(unit, terminal) if rc in (0, "GONE") else None
    if infra:
        # Infrastructure refused BEFORE any training: never a terminal. The attempt is interrupted
        # (not counted against the attempt budget), this role is recorded as refused so another can
        # take it, and status turns it into a typed failure after a bounded threshold.
        cause = f"{INFRA_PREFIT}{infra}@{attempt_info.get('role')}"
        q.interrupt(token, cause[:200])
        q.record_admission(unit_id, attempt_info.get("role"), False, cause[:120], attempt_info.get("cap_bytes"))
        q.set_state(unit_id, "WAIT_RESOURCE", cause[:200])
        return dict(result="WAIT_RESOURCE", unit_id=unit_id, cause=cause, attempt_dir=rel)
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
    effective = None
    for base in [Path(root), *map(Path, prior_roots)]:
        folder = base / "terminals" / unit["identity_sha256"]
        if not (folder / "TERMINAL.json").exists():
            continue
        if base.resolve() != Path(root).resolve():
            effective = effective or C.load_effective(root)
            problem = C.donor_problem(base, effective)
            if problem:  # the fit procedure binds adoption, not the identity alone
                q.event("PRIOR_REJECTED", unit["unit_id"], dict(source=str(base.name), reason=problem[:200]))
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
    units = plan["units"]
    if plan.get("mode") in C.MATCHED_FAMILY_MODES:
        # Persisted pre-execution refusals (INFRA_PREFIT and proven launcher admission refusals)
        # rotate arms instead of repeatedly starving the next one; ties keep plan order.
        refusals = {}
        for attempt in q.attempts():
            cause = str(attempt.get("cause") or "")
            if attempt["outcome"] == "INTERRUPTED" and (cause.startswith(INFRA_PREFIT)
                                                         or cause == "LAUNCHER_ADMISSION_REFUSED"):
                key = attempt["unit_id"]
                refusals[key] = refusals.get(key, 0) + 1
        units = sorted(units, key=lambda unit: refusals.get(unit["unit_id"], 0))
    for unit in units:
        row = rows.get(unit["unit_id"])
        if row is None or row["state"] not in ("QUEUED", "WAIT_RESOURCE", "INTERRUPTED", "WAIT_ARTIFACT") \
                or not row["identity_sha256"]:
            continue
        # A comparison runs once every arm is terminal: a failed arm is recorded as missing evidence.
        ready_states = ("COMPLETE", "FAILED", "BLOCKED") if row["kind"] == "comparison" else ("COMPLETE",)
        if any(rows[d]["state"] not in ready_states for d in row["spec"]["depends_on"]):
            continue
        if row["device"] == "gpu":
            if gpu is None:
                continue
            hold = fit_gpu_class_problem(row["spec"], gpu)
            if hold:
                # C1: never fail, never relabel; the fit waits typed for its sealed GPU class
                if row["state"] != "WAIT_RESOURCE" or row["cause"] != hold:
                    q.set_state(unit["unit_id"], "WAIT_RESOURCE", hold)
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
        if attempt_info.get("role"):
            from tools.i6dmo import services
            services.alive(root, f"worker-{attempt_info['role']}")
        time.sleep(poll_interval)
    progress = executor.progress(handle, folder)
    if progress:
        q.heartbeat(unit["unit_id"], token, ttl, progress)
    executor.fetch(handle, folder)
    return _finish(root, q, unit, token, rc, folder, attempt_info)


def worker_once(root, role, *, facts_provider=None, executor=None, poll_interval=POLL, ttl=TTL, extra_env=None,
                clock=time.time):
    from tools.i6dmo import services
    root = Path(root)
    if C.closed_doc(root) is not None:  # a CLOSED campaign is inert: nothing is claimed or written
        return dict(result="CLOSED", role=role)
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
        with services.run(root, f"worker-{role}") as outcome:
            out = _worker_locked(root, role, effective, spec, facts_provider, executor, poll_interval, ttl,
                                 extra_env or {}, clock)
            outcome["result"] = out.get("result")
        return out
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
            adopted = q.adopt(unit["unit_id"], doc["terminal"], doc["child_terminal_sha256"],
                              identity_sha256=unit["identity_sha256"],
                              source_detail=dict(source_root_name=source.name))
            if adopted["result"] == "ALREADY_TERMINAL":
                # The queue already holds this unit's verified terminal: the row is repaired to
                # COMPLETE and nothing is launched or re-adopted (no repeat, no spin).
                q.set_state(unit["unit_id"], "COMPLETE", "ALREADY_TERMINAL_IN_QUEUE")
                return dict(result="ALREADY_TERMINAL", unit_id=unit["unit_id"])
            _enqueue_reports(root, unit["unit_id"], doc["terminal"], q)
            return dict(result="ADOPTED", unit_id=unit["unit_id"])
        if unit.get("execution_policy") == "ADOPT_ONLY":
            cause = "REUSED_FIT_ARTIFACT_MISSING_OR_INVALID"
            q.set_state(unit["unit_id"], "WAIT_ARTIFACT", cause)
            return dict(result="WAIT_ARTIFACT", unit_id=unit["unit_id"], cause=cause)
        role_spec = spec["roles"][role]
        bundle, missing = _artifacts(root, effective, unit, role_spec)
        if missing:
            q.set_state(unit["unit_id"], "WAIT_ARTIFACT", missing)
            return dict(result="WAIT_ARTIFACT", unit_id=unit["unit_id"], cause=missing)
        gpu = _role_gpu(role_spec) if unit["device"] == "gpu" else None
        device_class = gpu["class"] if gpu else "cpu"
        launcher = dict(spec["launcher"], **role_spec.get("launcher", {}))
        if gpu:
            # The resource class is the model family+kind, not the GPU: a full-tree peak measured on
            # one admitted GPU carries to another of the same class. The launcher may seed a class
            # peak measured in a prior campaign; a local measurement, once it exists, wins.
            # A matched arm carries its own resource class: a 13-slot peak is never the 25-slot need.
            resource = unit.get("resource_class") or unit["family"]
            class_key = f"{resource}.{unit['kind']}"
            measured = (q.measured_peak(kind=unit["kind"], family=unit["family"], resource_class=unit["resource_class"])
                        if unit.get("resource_class") else q.measured_peak(kind=unit["kind"], family=unit["family"])) \
                or (launcher.get("measured_class_peaks") or {}).get(class_key)
            # A campaign may seal a PROVISIONAL cap per class with its basis (R2-17): used only until
            # the first unit of that class has its own measured peak (then 1.25x that peak governs).
            provisional = (effective.get("class_caps") or {}).get(class_key) or {}
            cap = cap_for(q, unit["unit_id"], device_class,
                          initial_cap=provisional.get("cap_bytes") or launcher["initial_cap_bytes"],
                          measured_peak=measured)
        else:
            cap = max(int(launcher.get("cpu_cap_bytes", CPU_CAP)), q.min_cap(unit["unit_id"]))
        if unit["kind"] in ARTIFACT_WRITING_KINDS:
            try:
                C._storage_gate(spec, f"claim:{unit['unit_id']}")
            except C.Refused as exc:
                q.record_admission(unit["unit_id"], role, False, str(exc)[:120], None)
                return dict(result="WAIT_RESOURCE", unit_id=unit["unit_id"], cause=str(exc)[:160])
        facts = facts_provider(role)
        ok, cause, evidence = admit(facts, cap=cap, gpu_uuid=gpu["uuid"] if gpu else None,
                                    reserve=int(role_spec.get("reserve_bytes", HOST_RESERVE)))
        runtime = matched_runtime_problem(effective, unit, facts) or \
            test_override_problem(effective, extra_env, role_spec) or scientific_device_problem(effective, unit) \
            or fit_gpu_class_problem(unit, gpu)
        if ok and runtime:  # a refusal routes to another role; never a reseal, never a retry with other caps
            ok, cause = False, runtime[:200]
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
        unit_spec = build_unit_spec(root, q, plan, effective, unit, role_spec, device_class=device_class,
                                    gpu=gpu, cap=cap, bundle=bundle, attempt=attempt)
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


def child_effective(effective, unit):
    """What a child sees as its effective config: the arm VIEW for a matched arm, else the campaign's."""
    return C.arm_view(effective, unit["arm"]) if unit.get("arm") else effective


def build_unit_spec(root, q, plan, effective, unit, role_spec, *, device_class, gpu, cap, bundle, attempt):
    """The UNIT_SPEC.json a claimed child receives (one definition for the worker and its tests)."""
    view = child_effective(effective, unit)
    unit_spec = dict(unit, schema="i6dmo.unit_spec.v1", campaign_id=plan["campaign_id"],
                     campaign_root=str(root),
                     effective_config=view, effective_config_sha256=plan["effective_config_sha256"],
                     device=dict(cls=device_class, gpu_uuid=gpu["uuid"] if gpu else None), cap_bytes=cap,
                     data_root=role_spec.get("data_root"), bundle=bundle,
                     pins=resolved_pins(view, role_spec), attempt=dict(index=attempt["index"], kind=attempt["kind"]),
                     upstream={d: _upstream(root, q, d) for d in unit["depends_on"]})
    if unit.get("arm"):
        # The model lane's matched report contract (reporting.run_unit switches on these keys);
        # a scientific campaign's report refuses any fit that is not scientific on a GPU.
        pop, inp = unit_spec["pins"].get("common_population"), unit_spec["pins"].get("matched_input")
        if pop and pop.get("path"):
            unit_spec["common_population"] = dict(path=pop["path"], sha256=pop["population_sha256"])
        if inp and inp.get("path"):
            unit_spec["matched"] = dict(arm=unit["arm"], fit_unit=f"{unit['arm']}.fit",
                                        input=dict(path=inp["path"], sha256=inp["sha256"]),
                                        require_scientific_fit=not effective.get("engineering_cpu_fixture"))
    if C.mode_of(effective) == C.MODE_R2:
        unit_spec.update(r2_unit_spec(root, q, plan, effective, unit))
        fit_unit = unit_spec.pop("regime_fit_unit", None)
        if fit_unit and unit_spec.get("matched"):
            unit_spec["matched"]["fit_unit"] = fit_unit
    if unit["unit_id"] == "raw.pair" and C.mode_of(effective) == C.MODE_MATCHED_LONG:
        for arm in effective["arms"]:
            unit_spec["upstream"][f"{arm}.report"] = _upstream(root, q, f"{arm}.report")
    if unit["kind"] == "strategy":
        # The trading stage's forecast inputs are authenticated digests, not assumptions: the
        # hook checks them against the digests the pair decision recorded for the same units.
        unit_spec["forecast_upstream"] = {d: _upstream(root, q, d) for d in forecast_dependencies(q, unit)}
    return unit_spec


def r2_unit_spec(root, q, plan, effective, unit):
    """The R2 keys of a child's UNIT_SPEC: the ruled objective + this campaign's provenance for every
    unit, the regime's own fit unit for a report, and for ``r2.fit``/``r2.probe`` the donor references
    the STORE retained when each pretraining unit completed (``campaign.r2_donor_references``)."""
    r2 = dict(effective["r2"], provenance=C.r2_provenance(effective, plan))
    extra = dict(r2=r2)
    if unit["kind"] == "report" and unit.get("regime") in ("R0", "R2"):
        extra["regime_fit_unit"] = f"{unit['regime'].lower()}.fit"
    if unit["unit_id"] in R2_DONOR_CONSUMERS:
        refs = C.r2_donor_references(root, q, effective)
        extra["r2_donors"] = None if refs is None else dict(donors=refs, objective=r2["objective"],
                                                             provenance=r2["provenance"])
    return extra


#: The only units that receive donors; the R0 fit never does (no import binding on R0).
R2_DONOR_CONSUMERS = ("r2.fit", "r2.probe")


FIT_GPU_CLASS_HOLD = "FIT_GPU_CLASS_HOLD"


def fit_gpu_class_problem(unit, gpu):
    """C1: an R0/R2 fit sealed to a GPU class is claimed only on a role of that class. Anything else
    is a typed wait (WAIT_RESOURCE FIT_GPU_CLASS_HOLD), never a failure and never a relabel."""
    wanted = unit.get("gpu_class")
    if not wanted or unit.get("kind") != "fit":
        return None
    actual = (gpu or {}).get("class")
    if actual == wanted:
        return None
    return f"{FIT_GPU_CLASS_HOLD}:{unit['unit_id']} needs {wanted}, role offers {actual}"


def scientific_device_problem(effective, unit):
    """A scientific R2 campaign never admits a CPU fit or pretraining unit (only the engineering
    CPU fixture may, and its receipts are non-scientific)."""
    if C.mode_of(effective) != C.MODE_R2 or effective.get("engineering_cpu_fixture"):
        return None
    if unit["kind"] in ("fit", "pretrain") and unit["device"] != "gpu":
        return f"R2_SCIENTIFIC_CPU_FIT_REFUSED:{unit['unit_id']}"
    return None


def _uses_unit_runner(argv):
    return "-m" in argv and argv[argv.index("-m") + 1:argv.index("-m") + 2] == ["tools.i6dmo.unit_runner"]


def bundle_pin(root, effective, unit):
    """Data-lane bundle pin: sealed data_pins for validation fits; DATA_PROVENANCE evidence for final refits."""
    if unit.get("arm"):
        return None, None  # matched arms read the frozen common population, not a per-arm bundle
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
    # A pin that IS present is verified whatever entrypoint runs the unit: a swapped or
    # re-materialized bundle must never reach a claim because of the entrypoint in use.
    if pin is not None and pin.get("data_contract_sha256") and unit["family"] in ("short", "long"):
        cause = _data_identity_problem(effective, unit, bundle, key, pin)
        if cause:
            return bundle, cause
    if unit["kind"] in ("pair", "strategy", "final_test"):
        problem = C._execution_policy_problem(root)
        if problem in ("INSTRUMENT_CONFIG_MISSING", "ABSENT"):
            return bundle, (f"EXECUTION_POLICY_{problem}: sealed at init/deploy from "
                            "effective_config.strategy_execution.instrument owner=Satoshi")
        # EXECUTION_POLICY_TAMPERED / MISMATCH stay claimable: the pinned hook refuses them FINALLY.
        if unit["kind"] == "final_test":
            try:
                from tools.i6dmo import strategy_adapter as SA
                exported = callable(getattr(SA, "run_final_backtest", None))
            except ImportError:
                exported = False
            if not exported:
                # Claimed now, the final hook would write a FINAL FAILED terminal; so it waits instead.
                return bundle, ("FINAL_BACKTEST_EXPORT_MISSING owner=strategy lane "
                                "(strategy_adapter.run_final_backtest)")
        cause = _pair_inputs_missing(root, effective, unit, role_spec)
        if cause:
            return bundle, cause
    cause = matched_unit_problem(effective, unit, role_spec, argv) or r2_donor_problem(root, effective, unit)
    if cause:
        return bundle, cause
    if not _uses_unit_runner(argv):
        return bundle, None
    module = unit_runner.hook_module(unit["kind"], unit)
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
    hook = importlib.import_module(module)
    # A matched arm's inputs (frozen population + retained input) were verified by
    # matched_unit_problem above; the hook's bundle-based preconditions describe the
    # per-family bundle path and do not apply to it.
    if hasattr(hook, "preconditions") and not unit.get("arm"):
        cause = hook.preconditions(dict(unit=unit, effective_config=child_effective(effective, unit),
                                        data_root=role_spec.get("data_root")))
        if cause:
            return bundle, cause
    return bundle, None


STRATEGY_PINS = ("strategy_bars", "strategy_calendar")
PREDICTION_ARTIFACTS = ("validation_predictions", "predictions/validation.npz")


def trading_stage(effective, unit):
    """Does this unit read OHLC bars and trade? Only those units need the strategy source pins.

    The market-time ``pair`` unit (pair.eligibility) decides eligibility from authenticated report
    metrics and the sealed minimum-head policy only; its trading stage is ``strategy``. The elapsed
    ``pair.validation`` unit both decides and backtests, so it keeps the pin precondition.
    """
    if unit["kind"] in ("strategy", "final_test"):
        return True
    return unit["kind"] == "pair" and C.mode_of(effective) != C.MODE_MARKET_TIME


def forecast_dependencies(q, unit):
    """The forecast report units a strategy unit trades on: the inputs of its pair decision."""
    deps = []
    for dep in unit["depends_on"]:
        row = q.unit(dep)
        if row is not None and row["kind"] == "pair":
            deps.extend(d for d in row["spec"]["depends_on"] if d not in deps)
    return deps


def _pair_inputs_missing(root, effective, unit, role_spec):
    """Keep a pair/strategy/test unit unclaimed while its hook would only return a final FAILED terminal."""
    trading = trading_stage(effective, unit)
    if trading:
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
        deps = list(unit["depends_on"])
        if unit["kind"] == "strategy":
            for dep in deps:
                term = q.terminal(dep)
                if term is not None and not C.pair_eligible(term):
                    return f"PAIR_NOT_QUALIFIED:{dep} (the tick closes this unit PAIR_NOT_ELIGIBLE)"
            deps += [d for d in forecast_dependencies(q, unit) if d not in deps]
        for dep in deps:
            term = q.terminal(dep)
            if term is None:
                return f"UPSTREAM_NOT_TERMINAL:{dep}"
            stored = Path(root) / "terminals" / (term["identity_sha256"] or "none") / "TERMINAL.json"
            arts = (term["terminal"].get("artifacts") or {}) if stored.exists() else {}
            if trading and dep.endswith(".report") and not any(n in arts for n in PREDICTION_ARTIFACTS):
                return f"UPSTREAM_ARTIFACT_MISSING:{dep}:validation_predictions owner=ops report unit"
    finally:
        q.close()
    return None


def _data_identity_problem(effective, unit, bundle, key, pin):
    """The pinned data identity must match the treatment's own declaration, or the unit waits."""
    design = C.unit_design(effective, unit)
    if design.get("market_time"):
        # A market-time bundle declares its identity in its own sealed manifest (the data-lane
        # market design document is not carried in the campaign).
        declared = None
        manifest = Path(bundle["path"], "BUNDLE.json") if bundle else None
        if manifest is not None and manifest.is_file():
            declared = json.loads(manifest.read_text()).get("data_identity_sha256")
        if declared != pin["data_contract_sha256"]:
            return (f"DATA_IDENTITY_MISMATCH:{key} the pinned data identity is not the one the bundle "
                    "manifest declares owner=data lane")
        return None
    from tools.i6dmo import data as data_lane
    if data_lane.data_contract_sha256(design) != pin["data_contract_sha256"]:
        return (f"DATA_CONTRACT_MISMATCH:{key} sealed {unit['family']} design differs from the bundle's "
                "data view owner=data lane / model lane")
    return None


def resolved_pins(effective, role_spec):
    """Every sealed data pin as an absolute path on the executing role: relative to its data root, or
    to its experiment storage root for pins declared ``base: experiment_storage`` (the frozen common
    population). A storage pin the role cannot resolve gets no path (the child refuses it typed)."""
    root = Path(role_spec.get("data_root") or "")
    out = {}
    for key, pin in (effective.get("data_pins") or {}).items():
        if "relpath" not in pin:
            continue
        if pin.get("base") == "experiment_storage":
            storage = role_spec.get("experiment_storage_root")
            if storage:
                out[key] = dict(pin, path=str(Path(storage) / pin["relpath"]))
            continue
        out[key] = dict(pin, path=str(root / pin["relpath"]))
    return out


#: Real capability of the model lane's adapters, checked statically (no TensorFlow import here):
#: the trainer's argparse must register exactly the flags unit_runner passes, and the report module
#: must define the matched report and the single decision rule.
MATCHED_CAPABILITIES = {
    "fit": ("tools/i6dmo/train.py", ("--common-population", "--common-population-digest", "--input-dir"), (),
            "model lane (train.py matched CLI flags)"),
    "report": ("tools/i6dmo/reporting.py", (), ("run_matched_unit", "decide_matched"),
               "model lane (reporting.run_matched_unit / decide_matched)"),
}
#: R2 contrast: the trainer must accept the donor import flag; the R2 bodies must exist.
R2_CAPABILITIES = {
    "fit": ("tools/i6dmo/train.py", ("--common-population", "--common-population-digest", "--input-dir",
                                     "--r2-donors"), ("r2_import_into",), "model lane (train.py --r2-donors)"),
    "report": MATCHED_CAPABILITIES["report"],
    "pretrain": ("tools/i6dmo/r2_runner.py", (), ("run_pretrain", "run_unit"), "ops lane (r2_runner.run_pretrain)"),
    "evaluation": ("tools/i6dmo/r2_runner.py", (), ("run_probe",), "ops lane (r2_runner.run_probe)"),
    "comparison": ("tools/i6dmo/r2_runner.py", (), ("run_comparison",), "ops lane (r2_runner.run_comparison)"),
}


def _capability_missing(relpath, flags, functions):
    """Missing flags/functions of a lane module, read from its source (never imported here)."""
    import ast
    path = Path(__file__).resolve().parents[2] / relpath
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return ["UNREADABLE"]
    registered = {node.args[0].value for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                  and node.args and isinstance(node.args[0], ast.Constant)}
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    return [f for f in flags if f not in registered] + [f for f in functions if f not in defined]


def matched_unit_problem(effective, unit, role_spec, argv):
    """Typed WAIT_ARTIFACT cause for a matched arm unit that cannot be claimed yet, else None."""
    table = R2_CAPABILITIES if C.mode_of(effective) == C.MODE_R2 else MATCHED_CAPABILITIES
    if not unit.get("arm") or unit["kind"] not in table:
        return None
    if _uses_unit_runner(argv):
        relpath, flags, functions, owner = table[unit["kind"]]
        missing = _capability_missing(relpath, flags, functions)
        if missing:
            return f"MATCHED_{unit['kind'].upper()}_ADAPTER_UNAVAILABLE:{','.join(missing)} owner={owner}"
    pins = resolved_pins(effective, role_spec)
    for key, name in (("common_population", "POPULATION.json"), ("matched_input", "INPUT.npz")):
        pin = (effective.get("data_pins") or {}).get(key)
        if pin is None:
            return f"MATCHED_PIN_MISSING:{key} owner=Satoshi (matched-short prepare)"
        if key not in pins:
            return f"MATCHED_PIN_UNRESOLVED:{key} (role declares no experiment_storage_root) owner=Satoshi"
        if role_spec.get("ssh_alias"):
            continue  # a remote role's files are verified by its preflight probe and by the child
        path = Path(pins[key]["path"]) / name
        if not path.is_file() or file_sha256(path) != pin["sha256"]:
            return f"MATCHED_INPUT_DIGEST_MISMATCH:{key} owner=Satoshi (replicate to the executing role)"
    return None


def r2_donor_problem(root, effective, unit):
    """``r2.fit``/``r2.probe`` wait until every pretraining terminal is stored with its donor digest:
    an R2 fit is never launched without its 19 authenticated donor references (it would be an R0)."""
    if C.mode_of(effective) != C.MODE_R2 or unit["unit_id"] not in R2_DONOR_CONSUMERS:
        return None
    q = C.open_queue(root)
    try:
        refs = C.r2_donor_references(root, q, effective)
    finally:
        q.close()
    return None if refs is not None else "R2_DONOR_REFERENCES_INCOMPLETE owner=ops (pretraining store)"


def test_override_problem(effective, extra_env, role_spec):
    """A scientific matched campaign never launches a child with an I6DMO_TEST_* override in reach
    (worker environment, injected env or the role's forwarded environment)."""
    if C.mode_of(effective) not in C.MATCHED_FAMILY_MODES or effective.get("engineering_cpu_fixture"):
        return None
    keys = sorted(k for k in {*os.environ, *(extra_env or {}), *(role_spec.get("environment") or {})}
                  if k.startswith("I6DMO_TEST_"))
    return f"TEST_OVERRIDE_IN_SCIENTIFIC_ENVIRONMENT:{','.join(keys)}" if keys else None


def matched_runtime_problem(effective, unit, facts):
    """A matched arm fit runs only on a worker whose actual TF/Keras equal the sealed runtime pin."""
    if not unit.get("arm") or unit["kind"] not in ("fit", "pretrain"):
        return None
    from tools.i6dmo import matched_runner as MR
    return MR.runtime_problem(MR.runtime_pin(C.unit_design(effective, unit)), facts)


def _upstream(root, q, unit_id):
    term = q.terminal(unit_id)
    if term is None or not term["identity_sha256"]:
        return None
    path = Path(root) / "terminals" / term["identity_sha256"] / "TERMINAL.json"
    return dict(path=str(path), sha256=file_sha256(path), terminal_sha256=term["terminal_sha256"])
