"""Lane B admitted-subprocess runner: caps, classification, self-fencing, no signals in threads."""
import json
import os
import sys
import threading
import time

import pytest

from tools.i6n_swarm import runner as R


def script(tmp_path, body, name="child.py"):
    path = tmp_path / name
    path.write_text("import sys, json, time, os\n" + body)
    return [sys.executable, str(path)]


def fake_launcher(tmp_path):
    """Stands in for crispdm-run: records its argv, then runs everything after '--'."""
    path = tmp_path / "fake-crispdm-run"
    path.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/launcher.argv\"\n"
                    "while [ \"$1\" != \"--\" ]; do shift; done; shift\nexec \"$@\"\n")
    path.chmod(0o755)
    return dict(crispdm_run=str(path), memory_bytes=2 * 1024**3, wall_seconds=30, label_prefix="i6n-test")


def run(tmp_path, argv, renew=lambda progress=None: True, **kw):
    params = dict(wall_seconds=30, renew_interval=0.2, ttl=5.0, log_dir=tmp_path / "logs")
    params.update(kw)
    return R.run_attempt(argv, dict(os.environ, CUDA_VISIBLE_DEVICES=""), json.dumps({"x": 1}),
                         renew=renew, **params)


def test_launcher_argv_carries_caps_and_label(tmp_path):
    launcher = fake_launcher(tmp_path)
    argv = R.build_argv(launcher, "abc", ["python", "-m", "tools.i6n_historical_evaluator", "evaluate"])
    assert argv[:argv.index("--")] == [launcher["crispdm_run"], "-m", str(2 * 1024**3), "-t", "30s",
                                       "-n", "i6n-test-abc", "-L", "i6n-test-abc"]
    child = argv[argv.index("--") + 1:]
    assert child[0] == sys.executable and child[1].endswith("request_exec.py")
    assert child[2:] == ["--", "python", "-m", "tools.i6n_historical_evaluator", "evaluate"]
    with pytest.raises(ValueError):
        R.build_argv(dict(launcher, memory_bytes=0), "abc", ["x"])


def test_ok_receipt_from_stdout_through_launcher(tmp_path):
    launcher = fake_launcher(tmp_path)
    child = script(tmp_path, "req=json.load(sys.stdin); print(json.dumps({'status':'ok','echo':req}))")
    out = run(tmp_path, R.build_argv(launcher, "ok", child))
    assert out["kind"] == "OK" and out["receipt"]["echo"] == {"x": 1}
    assert (tmp_path / "launcher.argv").exists()


def test_request_survives_launcher_without_inherited_stdin(tmp_path):
    launcher = fake_launcher(tmp_path)
    path = tmp_path / "fake-crispdm-run"
    path.write_text(path.read_text().replace('exec "$@"', 'exec "$@" </dev/null'))
    child = script(tmp_path, "req=json.load(sys.stdin); print(json.dumps({'echo':req}))")
    out = run(tmp_path, R.build_argv(launcher, "closed-stdin", child))
    assert out["kind"] == "OK", out
    assert out["receipt"]["echo"] == {"x": 1}


def test_nonzero_exit_is_failed(tmp_path):
    out = run(tmp_path, script(tmp_path, "sys.stderr.write('refused'); sys.exit(1)"))
    assert out["kind"] == "FAILED" and "refused" in out["stderr_tail"]


def test_admission_refusal_is_not_started(tmp_path):
    out = run(tmp_path, script(tmp_path, "sys.stderr.write('crispdm-run: REFUSED x'); sys.exit(75)"))
    assert out["kind"] == "NOT_STARTED"


def test_oom_kill_is_terminal_oom(tmp_path):
    out = run(tmp_path, script(tmp_path, "import signal; os.kill(os.getpid(), signal.SIGKILL)"))
    assert out["kind"] == "FAILED_OOM"


def test_pressure_stop_is_external_interruption(tmp_path):
    body = ("sys.stderr.write('crispdm-run: this job\\'s own scope was STOPPED under sustained memory pressure');"
            "sys.exit(143)")
    out = run(tmp_path, script(tmp_path, body))
    assert out["kind"] == "INTERRUPTED" and out["cause"] == "HOST_MEMORY_PRESSURE"


def test_wall_timeout_is_terminal_timeout(tmp_path):
    out = run(tmp_path, script(tmp_path, "time.sleep(30)"), wall_seconds=0.5)
    assert out["kind"] == "FAILED_TIMEOUT"


def test_lease_loss_self_fences_child_before_expiry(tmp_path):
    marker = tmp_path / "alive"
    child = script(tmp_path, f"open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(30)")
    calls = []

    def renew(progress=None):
        calls.append(1)
        return len(calls) < 2

    started = time.monotonic()
    out = run(tmp_path, child, renew=renew)
    assert out["kind"] == "INTERRUPTED" and out["cause"] == "LEASE_LOST"
    assert time.monotonic() - started < 5.0  # well inside ttl
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_unreachable_ledger_self_fences_before_ttl(tmp_path):
    def renew(progress=None):
        raise ConnectionError("ledger unreachable")

    out = run(tmp_path, script(tmp_path, "time.sleep(30)"), renew=renew, ttl=1.5, renew_interval=0.2)
    assert out["kind"] == "INTERRUPTED" and out["cause"] == "LEASE_UNRENEWABLE"
    assert out["elapsed_seconds"] < 1.5


def test_shutdown_event_interrupts(tmp_path):
    stop = threading.Event()
    threading.Timer(0.4, stop.set).start()
    out = run(tmp_path, script(tmp_path, "time.sleep(30)"), stop_event=stop)
    assert out["kind"] == "INTERRUPTED" and out["cause"] == "NODE_SHUTDOWN"


def test_runs_inside_a_non_main_thread(tmp_path):
    """The DOIN node evaluates in an executor thread; nothing here may need SIGALRM."""
    box = {}
    child = script(tmp_path, "print(json.dumps({'status':'ok'}))")
    thread = threading.Thread(target=lambda: box.update(out=run(tmp_path, child)))
    thread.start()
    thread.join(20)
    assert box["out"]["kind"] == "OK"


def test_progress_from_status_file_reaches_renew(tmp_path):
    status = tmp_path / "STATUS.json"
    child = script(tmp_path, f"json.dump({{'state':'RUNNING','eta_seconds':12.0,'progress':{{'epoch':2}}}}, open({str(status)!r},'w'));"
                             "time.sleep(1); print(json.dumps({'status':'ok'}))")
    seen = []
    out = run(tmp_path, child, renew=lambda progress=None: seen.append(progress) or True,
              progress_reader=lambda: json.loads(status.read_text()) if status.exists() else None)
    assert out["kind"] == "OK"
    assert any(p and p.get("eta_seconds") == 12.0 for p in seen)


def test_gpu_device_env_refuses_cpu_fallback():
    with pytest.raises(R.NoFallback):
        R.device_env(dict(cls="rtx5090", cuda_visible_devices="", required_gpu_name="NVIDIA GeForce RTX 5090"), {})
    with pytest.raises(R.NoFallback):
        R.device_env(dict(cls="rtx5090", cuda_visible_devices="GPU-a,GPU-b", required_gpu_name="x"), {})
    env = R.device_env(dict(cls="rtx5090", cuda_visible_devices="GPU-0000", required_gpu_name="x"), {"PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-0000" and env["TF_CPP_MIN_LOG_LEVEL"] == "2"


def test_cpu_only_allowed_for_declared_fixture():
    with pytest.raises(R.NoFallback):
        R.device_env(dict(cls="cpu", cuda_visible_devices=""), {})
    env = R.device_env(dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True), {})
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_runner_module_does_not_use_alarm_signals():
    source = open(R.__file__).read()
    assert "SIGALRM" not in source and "setitimer" not in source and "signal.alarm" not in source


def test_guard_keeps_receipt_and_signal_exit_codes(tmp_path):
    ok = run(tmp_path, script(tmp_path, "req=json.load(sys.stdin); print(json.dumps({'status':'ok','echo':req}))"), guard=True)
    assert ok["kind"] == "OK" and ok["receipt"]["echo"] == {"x": 1}
    oom = run(tmp_path, script(tmp_path, "import signal; os.kill(os.getpid(), signal.SIGKILL)", "k.py"), guard=True)
    assert oom["kind"] == "FAILED_OOM" and oom["returncode"] == 137


def test_guard_stops_child_when_supervisor_dies(tmp_path):
    """Node loss: the supervising process is SIGKILLed; its admitted child must not keep training."""
    marker = tmp_path / "child.pid"
    child = script(tmp_path, f"open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(60)", "sleeper.py")
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import json, os, sys\n"
        "from tools.i6n_swarm import runner as R\n"
        f"R.run_attempt({child!r}, dict(os.environ), '{{}}', wall_seconds=60, renew=lambda p=None: True,\n"
        f"              renew_interval=0.2, ttl=30, log_dir={str(tmp_path / 'plogs')!r}, guard=True)\n")
    import subprocess
    proc = subprocess.Popen([sys.executable, str(parent)], cwd=os.getcwd(),
                            env=dict(os.environ, PYTHONPATH=os.getcwd()))
    deadline = time.monotonic() + 15
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    pid = int(marker.read_text())
    proc.kill()
    proc.wait()
    deadline = time.monotonic() + 10
    alive = True
    while alive and time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            alive = False
    assert not alive
