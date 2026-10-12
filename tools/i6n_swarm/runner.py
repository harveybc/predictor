"""One admitted evaluator attempt in a child process, supervised without process-wide timers.

The supervisor only waits on the child with ``Popen.wait(timeout=...)``; it
installs no signal handler and arms no interval timer, so it is safe in the
DOIN node's executor thread. While the child runs it renews the ledger lease;
when renewal is refused, or has not succeeded for ``ttl - margin`` seconds, it
stops its own child BEFORE the lease can expire (self-fencing), so a peer that
reclaims the fit never trains concurrently with this one.

Outcome kinds: OK, FAILED, FAILED_OOM, FAILED_TIMEOUT, INTERRUPTED, NOT_STARTED.
OOM and timeout are terminal: nothing here retries, lowers a cap or changes device.
"""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ADMISSION_REFUSED = 75
PRESSURE_MARK = "STOPPED under sustained memory pressure"
GPU_CLASSES = ("rtx5090", "rtx4090", "rtx4070", "rtx5070ti")


class NoFallback(ValueError):
    """Raised instead of silently running a GPU fit on the CPU."""


def device_env(device, base_env):
    """Environment for the child: exactly one explicit GPU, or a DECLARED CPU fixture."""
    cls = device.get("cls")
    visible = device.get("cuda_visible_devices")
    if cls == "cpu_fixture":
        if not device.get("fixture") or visible != "":
            raise NoFallback("cpu_fixture requires fixture=true and hidden GPUs")
    elif cls in GPU_CLASSES:
        if not isinstance(visible, str) or not visible or "," in visible:
            raise NoFallback("explicit single GPU required; no CPU fallback")
    else:
        raise NoFallback(f"device class {cls!r} not authorized; no CPU fallback")
    env = dict(base_env)
    env.update(CUDA_VISIBLE_DEVICES=visible, TF_CPP_MIN_LOG_LEVEL="2", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
               TF_NUM_INTEROP_THREADS="1")
    return env


def build_argv(launcher, label, child_argv):
    """crispdm-run argv with ONE fixed memory cap and wall limit (never lowered on retry)."""
    memory, wall = launcher.get("memory_bytes"), launcher.get("wall_seconds")
    if type(memory) is not int or memory <= 0 or type(wall) is not int or wall <= 0:
        raise ValueError("positive integer memory_bytes and wall_seconds required")
    name = f"{launcher.get('label_prefix', 'i6n-swarm')}-{label}"
    argv = [launcher["crispdm_run"], "-m", str(memory), "-t", f"{wall}s", "-n", name, "-L", name]
    if launcher.get("queue_wait_seconds"):
        argv += ["-q", "-W", str(int(launcher["queue_wait_seconds"]))]
    relay = str(Path(__file__).with_name("request_exec.py"))
    return argv + ["--", sys.executable, relay, "--", *child_argv]


def _tail(path, limit=4000):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode(errors="replace")


def _stop(process, grace=20.0):
    """Ask the launcher to stop its own scope (it forwards TERM), then kill our process group."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def classify(returncode, stdout_text, stderr_text, elapsed, wall_seconds):
    if returncode == 0:
        try:
            lines = [line for line in stdout_text.splitlines() if line.strip()]
            receipt = json.loads(lines[-1])
        except (ValueError, IndexError):
            return dict(kind="FAILED", cause="RECEIPT_NOT_JSON")
        if not isinstance(receipt, dict):
            return dict(kind="FAILED", cause="RECEIPT_NOT_OBJECT")
        return dict(kind="OK", cause=None, receipt=receipt)
    if returncode == ADMISSION_REFUSED and "REFUSED" in stderr_text:
        return dict(kind="NOT_STARTED", cause="ADMISSION_REFUSED")
    if PRESSURE_MARK in stderr_text:
        return dict(kind="INTERRUPTED", cause="HOST_MEMORY_PRESSURE")
    if returncode == 124 or (wall_seconds and elapsed >= wall_seconds):
        return dict(kind="FAILED_TIMEOUT", cause=f"WALL_LIMIT rc={returncode}")
    if returncode in (137, -9):
        return dict(kind="FAILED_OOM", cause=f"KILLED rc={returncode}")
    return dict(kind="FAILED", cause=f"EXIT_{returncode}")


def guarded(argv, python=None):
    """Prefix the orphan guard so the child dies with its supervising process."""
    import sys
    script = str(Path(__file__).with_name("childguard.py"))  # stdlib-only; independent of the child's cwd
    return [python or sys.executable, script, "--parent-pid", str(os.getpid()), "--", *argv]


def run_attempt(argv, env, stdin_text, *, wall_seconds, renew, renew_interval, ttl, log_dir,
                stop_event=None, progress_reader=None, cwd=None, margin=None, clock=time.monotonic, guard=False):
    """Run one attempt. ``renew(progress)`` returns True while the lease is ours."""
    if guard:
        argv = guarded(argv)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
    env = dict(env)
    # The relay restores stdin inside the admitted scope, not before the launcher.
    if stdin_text:
        data = stdin_text.encode()
        request_path = log_dir.resolve() / f"request-{stamp}.json"
        fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        env.update(I6N_SWARM_REQUEST_FILE=str(request_path),
                   I6N_SWARM_REQUEST_SHA256=hashlib.sha256(data).hexdigest())
    else:
        env.pop("I6N_SWARM_REQUEST_FILE", None)
        env.pop("I6N_SWARM_REQUEST_SHA256", None)
    out_path, err_path = log_dir / f"attempt-{stamp}.stdout", log_dir / f"attempt-{stamp}.stderr"
    margin = max(renew_interval * 2, ttl * 0.25) if margin is None else margin
    started = clock()
    last_ok = started
    kind = None
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err, env=env, cwd=cwd,
                                   start_new_session=True)
        try:
            process.stdin.write(stdin_text.encode())
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            while True:
                try:
                    process.wait(timeout=renew_interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = clock()
                if stop_event is not None and stop_event.is_set():
                    kind = dict(kind="INTERRUPTED", cause="NODE_SHUTDOWN")
                elif wall_seconds and now - started > wall_seconds + min(60.0, max(float(wall_seconds), 1.0)):
                    kind = dict(kind="FAILED_TIMEOUT", cause="SUPERVISOR_WALL_OVERRUN")
                else:
                    progress = None
                    if progress_reader is not None:
                        try:
                            progress = progress_reader()
                        except (OSError, ValueError):
                            progress = None
                    try:
                        if renew(progress):
                            last_ok = clock()
                        else:
                            kind = dict(kind="INTERRUPTED", cause="LEASE_LOST")
                    except Exception:
                        if clock() - last_ok >= ttl - margin:
                            kind = dict(kind="INTERRUPTED", cause="LEASE_UNRENEWABLE")
                if kind is not None:
                    _stop(process)
                    break
        except BaseException:
            _stop(process)
            raise
    elapsed = clock() - started
    stdout_text, stderr_text = _tail(out_path, 10_000_000), _tail(err_path)
    if kind is None:
        kind = classify(process.returncode, stdout_text, stderr_text, elapsed, wall_seconds)
    kind.update(returncode=process.returncode, elapsed_seconds=elapsed, stderr_tail=stderr_text[-2000:],
                stdout_path=str(out_path), stderr_path=str(err_path))
    return kind
