#!/usr/bin/env python3
"""Raise a disposable governance stack whose input lake is the **new host**, and prove it.

N4 asks for the route `consumer -> data-gov -> data-lake/external provider -> warehouse` and
for a negative control: removing the entry host must make the run fail for the expected
reason, which is how this is told apart from a local adapter that keeps working.

Everything here is disposable: a data-gov on a free port, a warehouse host over a throwaway
SQLite file, and the lake host serving the fixture root. No production service is touched and
no production database is opened.

usage:
  disposable_route_stack.py --fixtures DIR --work DIR [--hold SECONDS] [--print-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GITHUB = REPO.parent
DATA_GOV = GITHUB / "data-gov"
HOSTS_PYTHON = Path.home() / ".venvs" / "store-hosts" / "bin" / "python"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url: str, processes, deadline: float = 60.0) -> None:
    start = time.monotonic()
    while time.monotonic() - start < deadline:
        for process in processes:
            if process.poll() is not None:
                raise SystemExit(f"a service exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise SystemExit(f"service did not become ready: {url}")


def start(argv, cwd, env, log: Path):
    handle = log.open("wb")
    process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    process._log = handle
    return process


def stop(process):
    if process.poll() is None:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    process._log.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, required=True, help="the fixture root to serve")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--hold", type=float, default=0.0,
                        help="keep the stack up this long (seconds) after printing its ports")
    args = parser.parse_args(argv)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    fixtures = args.fixtures.resolve()
    manifest = json.loads((fixtures / "MANIFEST.json").read_text(encoding="utf-8"))

    lake_port, warehouse_port, gov_port = free_port(), free_port(), free_port()
    token = "disposable-lake-" + os.urandom(8).hex()
    actor = "disposable-actor-" + os.urandom(8).hex()
    salt = "disposable-salt"
    key_file = work / "actor.key"
    key_file.write_text(actor + "\n")
    key_file.chmod(0o600)

    (work / "lake.json").write_text(json.dumps({
        "store_id": "synthetic_fixtures", "title": "consumer fixtures", "kind": "lake",
        "web_host": "127.0.0.1", "web_port": lake_port,
        "backend": {"entry_point": "financial_files", "distribution": "financial-data-store",
                    "settings": {"root_path": str(fixtures),
                                 "include_globs": ["*.csv"],
                                 "resource_contracts": manifest["contracts"],
                                 "holdout_start": None,
                                 "spool_dir": str(work / "lake-spool"),
                                 "cuts_dir": str(work / "lake-cuts")}}}, indent=1))
    (work / "warehouse.json").write_text(json.dumps({
        "store_id": "olap_cube", "title": "disposable cube", "kind": "warehouse",
        "web_host": "127.0.0.1", "web_port": warehouse_port,
        "backend": {"entry_point": "predictor_olap", "distribution": "predictor-olap-store",
                    "settings": {"sqlite_path": str(work / "cube.sqlite"),
                                 "holdout_start": None, "lake_id": "olap_cube"}}}, indent=1))
    (work / "governance.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web",
        "access_plugin": "default_access", "accounting_plugin": "default_accounting",
        "role_plugin": "default_role", "web_host": "127.0.0.1", "web_port": gov_port,
        "accounting_db": str(work / "accounting.sqlite"), "spool_dir": str(work / "gov-spool"),
        "cuts_dir": str(work / "gov-cuts"), "save_config": None, "password_salt": salt,
        "secret_key": "disposable",
        "principals": {"predictor": {"kind": "service", "role": "service",
                                     "api_key_hash": hashlib.sha256(
                                         f"{salt}:{actor}".encode()).hexdigest()}},
        "policies": [
            {"principal": "*", "lake": "synthetic_fixtures",
             "verbs": ["discover", "coverage", "read", "download"]},
            {"principal": "*", "lake": "olap_cube",
             "verbs": ["discover", "query", "write_metrics", "write_terminal"]}],
        "lakes": [
            {"plugin": "http_lake", "lake_id": "synthetic_fixtures", "kind": "lake",
             "base_url": f"http://127.0.0.1:{lake_port}", "holdout_start": None},
            {"plugin": "http_lake", "lake_id": "olap_cube", "kind": "warehouse",
             "base_url": f"http://127.0.0.1:{warehouse_port}", "holdout_start": None}]}, indent=1))

    env = dict(os.environ, DATA_GOV_LAKE_TOKEN=token, PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    for name in ("PGDATABASE", "PGUSER", "PGPASSWORD", "PGHOST", "PGPORT"):
        env.pop(name, None)      # a disposable stack must not be able to reach the real cube
    processes = []
    try:
        processes.append(start([str(HOSTS_PYTHON), "-m", "data_lake_service.main",
                                "--load_config", str(work / "lake.json")], work, env,
                               work / "lake.log"))
        processes.append(start([str(HOSTS_PYTHON), "-m", "data_warehouse_service.main",
                                "--load_config", str(work / "warehouse.json")], work, env,
                               work / "warehouse.log"))
        wait_for(f"http://127.0.0.1:{lake_port}/healthz", processes)
        wait_for(f"http://127.0.0.1:{warehouse_port}/healthz", processes)
        gov_env = dict(env, PYTHONPATH=str(DATA_GOV))
        processes.append(start([sys.executable, "-m", "app.main", "--load_config",
                                str(work / "governance.json")], DATA_GOV, gov_env,
                               work / "governance.log"))
        wait_for(f"http://127.0.0.1:{gov_port}/healthz", processes)
        state = {"schema": "disposable_route_stack.v1",
                 "gov_url": f"http://127.0.0.1:{gov_port}",
                 "lake_url": f"http://127.0.0.1:{lake_port}",
                 "warehouse_url": f"http://127.0.0.1:{warehouse_port}",
                 "lake": "synthetic_fixtures", "metrics_lake": "olap_cube",
                 "key_file": str(key_file), "fixtures": str(fixtures),
                 "cube": str(work / "cube.sqlite"),
                 "lake_pid": processes[0].pid, "warehouse_pid": processes[1].pid,
                 "gov_pid": processes[2].pid}
        (work / "STACK.json").write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(state, indent=1))
        if args.hold:
            time.sleep(args.hold)
        return 0
    finally:
        if args.hold:
            for process in reversed(processes):
                stop(process)


if __name__ == "__main__":
    raise SystemExit(main())
