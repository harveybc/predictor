"""C174: a job's argv may carry {role}; it is bound to the placed role only in the launched command, while the job
identity and unit name stay those of the template, so resume finds the same job wherever it ran."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("df_dispatch", ROOT / "tools/df_dispatch.py")
DP = importlib.util.module_from_spec(spec)
sys.modules["df_dispatch"] = DP
spec.loader.exec_module(DP)

JOB = {"job_id": "d2-shard00", "argv": ["env", "-u", "PYTHONPATH", "~/py", "-B", "tools/w.py", "--out",
                                         "~/.local/state/x/{role}/shard_00", "--host-role", "{role}"],
       "cpu_bytes": 1 << 30, "gpu_bytes": 0, "cpus": 1, "wall": 600, "roles": ["WORKER_A", "WORKER_B"],
       "gpu_index": None, "split": None}


def test_bind_role_replaces_every_placeholder_and_keeps_the_template():
    b = DP.bind_role(JOB, "WORKER_B")
    assert b["argv"][-3] == "~/.local/state/x/WORKER_B/shard_00" and b["argv"][-1] == "WORKER_B"
    assert "{role}" not in " ".join(b["argv"]) and "{role}" in " ".join(JOB["argv"])
    assert DP.identity(JOB) == DP.identity(dict(JOB)) and DP.unit_name(JOB) == DP.unit_name(dict(JOB))


def test_the_launched_script_carries_the_role_and_expands_home():
    decision = {"role": "WORKER_A", "request_bytes": 2 << 30, "gpu": None}
    script = DP.build_start_script(DP.bind_role(JOB, "WORKER_A"), decision, "Documents/GitHub/.worktrees/x")
    assert '"$HOME"/.local/state/x/WORKER_A/shard_00' in script and "--host-role WORKER_A" in script
    assert "{role}" not in script and "/home/" not in script


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="unknown role"):
        DP.bind_role(JOB, "omega")
