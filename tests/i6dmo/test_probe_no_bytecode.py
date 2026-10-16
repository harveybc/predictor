"""Remote probe argv and real local module execution without SSH, services or bytecode."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from tools.i6dmo import preflight as P


def spec(root):
    return dict(roles=dict(worker=dict(ssh_alias="fixture-only", python=sys.executable,
        workdir=str(root), run_root=str(root), data_root=str(root))))


def remote_args(argv):
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert argv[3:5] == ["fixture-only", "--"]
    command = shlex.split(argv[5])
    assert command[0] == "cd" and command[2] == "&&"
    return command[1], command[3:]


@pytest.mark.parametrize("check_strategy", [False, True])
def test_remote_probe_has_B_before_module_and_preserves_arguments(tmp_path, check_strategy):
    def run(argv, **kwargs):
        cwd, args = remote_args(argv)
        assert cwd == str(tmp_path)
        assert args[:4] == [sys.executable, "-B", "-m", "tools.i6dmo"]
        assert args[-2:] == (["artifact with spaces.json", "--check-strategy"] if check_strategy
                             else ["--file", "artifact with spaces.json"])
        return subprocess.CompletedProcess(argv, 0, '{"schema":"fixture"}\n', "")
    assert P.probe_role(spec(tmp_path), "worker", files=["artifact with spaces.json"],
                        check_strategy=check_strategy, run=run) == dict(schema="fixture")


def test_actual_read_only_module_probe_leaves_no_bytecode_without_environment_flag(tmp_path):
    runtime = tmp_path / "isolated runtime"
    repo = Path(P.__file__).resolve().parents[2]
    for package in ("tools", "tools/i6dmo", "tools/i6n_swarm"):
        directory = runtime / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    shutil.copyfile(P.__file__, runtime / "tools/i6dmo/preflight.py")
    shutil.copyfile(repo / "tools/i6n_swarm/common.py", runtime / "tools/i6n_swarm/common.py")
    (runtime / "tools/i6dmo/__main__.py").write_text(
        "import argparse,json,subprocess,sys\n"
        "from .preflight import probe_local\n"
        "p=argparse.ArgumentParser(); p.add_argument('action',choices=['probe'])\n"
        "p.add_argument('--json',action='store_true'); p.add_argument('--run-root')\n"
        "p.add_argument('--data-root'); p.add_argument('--file',action='append',default=[])\n"
        "a=p.parse_args()\n"
        "def controlled_run(argv,**kwargs):\n"
        " assert argv[0] in ('nvidia-smi','systemctl')\n"
        " return subprocess.CompletedProcess(argv,0,'','')\n"
        "doc=probe_local(run_root=a.run_root,data_root=a.data_root,files=a.file,run=controlled_run)\n"
        "doc['tensorflow_loaded']='tensorflow' in sys.modules\n"
        "print(json.dumps(doc))\n")
    before = {str(p.relative_to(runtime)): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in runtime.rglob("*") if p.is_file()}
    def local_transport(argv, **kwargs):
        cwd, args = remote_args(argv)
        env = dict(os.environ, PYTHONPATH=str(runtime))
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env.pop("PYTHONPYCACHEPREFIX", None)
        return subprocess.run(args, cwd=cwd, env=env, **kwargs)
    result = P.probe_role(spec(runtime), "worker", check_strategy=False, run=local_transport)
    assert result["schema"] == "i6dmo.probe.v1" and result["tensorflow_loaded"] is False
    assert not list(runtime.rglob("*.pyc"))
    assert not list(runtime.rglob("__pycache__"))
    after = {str(p.relative_to(runtime)): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in runtime.rglob("*") if p.is_file()}
    assert after == before
