"""The weekly installer refuses before touching systemd (extends tools/fs4_deploy/install_host.sh)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/fs4_deploy/weekly/install_weekly_host.sh"


def _run(tmp_path, env_text=None, mode=0o600):
    home = tmp_path / "home"
    (home / ".config/fs4").mkdir(parents=True)
    if env_text is not None:
        f = home / ".config/fs4/s1.env"
        f.write_text(env_text)
        f.chmod(mode)
    return subprocess.run(["bash", str(SCRIPT), "s1", "--check-only"], env={**os.environ, "HOME": str(home)}, text=True,
                          capture_output=True)


def _env(**over):
    base = {"FS4_PYTHON": "/usr/bin/python3", "FS4_CODE": "/nonexistent", "FS4_COORDINATOR": "c", "FS4_CONTROLLER": "/c/x.py",
            "FS4_DB": "/c/q", "FS4_OWNER": "worker_a-weekly-raw-1", "FS4_CAP": "2G", "FS4_INPUT_MODE": "RAW",
            "FS4_OUTPUT_ROOT": "/var/fs4", "FS4_POPULATION": "EURUSD", "FS4_BAR_HOURS": "1",
            "FS4_TRAIN_FEATURES": "/a", "FS4_TRAIN_TARGETS": "/b"}
    base.update(over)
    return "".join(f"{k}={v}\n" for k, v in base.items())


@pytest.mark.parametrize("text,mode,needle", [
    (None, 0o600, "missing"),
    (_env(), 0o644, "must be mode 0600"),
    (_env(FS4_CAP="PENDING_MEASURED_CAP"), 0o600, "placeholder"),
    (_env(FS4_CAP="lots"), 0o600, "FS4_CAP must be a size"),
    (_env(FS4_OWNER="somebody"), 0o600, "FS4_OWNER must be"),
    (_env(FS4_OUTPUT_ROOT="/tmp/x"), 0o600, "not durable"),
    (_env(FS4_INPUT_MODE="MLP"), 0o600, "FS4_INPUT_MODE"),
    (_env(), 0o600, "no tools/fs4_weekly_worker.py"),
])
def test_installer_refuses(tmp_path, text, mode, needle):
    r = _run(tmp_path, text, mode)
    assert r.returncode == 2 and needle in r.stderr, r.stderr
