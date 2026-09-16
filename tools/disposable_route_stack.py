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
#: the data-gov checkout that provides the governance service for the disposable stack;
#: overridable so a worker whose layout differs can still run this path (R4)
DATA_GOV = Path(os.environ.get("DATA_GOV_CHECKOUT") or GITHUB / "data-gov").expanduser()
#: The interpreter that runs the two store hosts. Overridable so a worker can run this stack
#: against a CANDIDATE provider without touching the venv the live hosts use (R4).
HOSTS_PYTHON = Path(os.environ.get("STORE_HOSTS_PYTHON")
                    or Path.home() / ".venvs" / "store-hosts" / "bin" / "python")


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
    # Its own session, so the stack owns a process GROUP nobody else is in: teardown can then
    # signal that group instead of matching a module name that production shares.
    process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    process._log = handle
    return process


def cmdline_of(pid: int) -> str | None:
    """The process's own argv, or None when the PID no longer denotes a running process.

    A ZOMBIE still has a `/proc/<pid>` directory - the entry survives until its parent reaps it
    - but its `cmdline` is empty. Reading only the directory's existence reports a process that
    has already exited as still running, which made teardown escalate to SIGKILL and then
    report STILL_RUNNING for a process that was gone. Both signs are checked here.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    if not raw.strip(b"\x00"):
        return None                      # zombie, or a process with no argv: not running
    return raw.decode("utf-8", "replace").replace("\x00", " ")


def reap(pid: int) -> None:
    """Reap a child we started, so it does not linger as a zombie. Harmless for non-children."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


def teardown(state: dict, *, grace: float = 20.0) -> dict:
    """Stop exactly the processes this stack started, and nothing else.

    The rule that matters: a recorded PID alone never authorizes a signal. Between writing
    STACK.json and tearing it down the PID may have been reused by an unrelated process — and
    matching on a module name instead is how a disposable teardown reached the PRODUCTION
    warehouse host on 2026-09-15. So every PID is checked against the marker this stack put in
    its own command line (its unique work directory) before anything is signalled, and a
    mismatch is REFUSED and reported rather than resolved in favour of killing.
    """
    marker = str(state["work"])
    report = {"marker": marker, "processes": []}
    for name in ("lake_pid", "warehouse_pid", "gov_pid"):
        pid = state.get(name)
        entry = {"role": name, "pid": pid}
        if not pid:
            entry["outcome"] = "NOT_RECORDED"
            report["processes"].append(entry)
            continue
        pid = int(pid)
        cmdline = cmdline_of(pid)
        if cmdline is None:
            entry["outcome"] = "ALREADY_GONE"
            report["processes"].append(entry)
            continue
        if marker not in cmdline:
            # The PID is alive and is NOT ours. This is the case that must never be a kill.
            entry.update(outcome="PID_REUSED_REFUSED", observed_cmdline=cmdline[:200])
            report["processes"].append(entry)
            continue
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            entry["outcome"] = "ALREADY_GONE"
            report["processes"].append(entry)
            continue
        entry["process_group"] = group
        os.killpg(group, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and cmdline_of(pid) is not None:
            reap(pid)
            time.sleep(0.1)
        reap(pid)
        if cmdline_of(pid) is None:
            entry["outcome"] = "TERMINATED"
        else:
            os.killpg(group, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and cmdline_of(pid) is not None:
                reap(pid)
                time.sleep(0.1)
            reap(pid)
            entry["outcome"] = ("KILLED" if cmdline_of(pid) is None else "STILL_RUNNING")
        report["processes"].append(entry)
    outcomes = [entry["outcome"] for entry in report["processes"]]
    report["stopped"] = sum(1 for o in outcomes if o in ("TERMINATED", "KILLED"))
    report["refused"] = sum(1 for o in outcomes if o == "PID_REUSED_REFUSED")
    report["complete"] = all(o != "STILL_RUNNING" for o in outcomes)
    return report


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
    parser.add_argument("--teardown", type=Path, metavar="STACK.json",
                        help="stop exactly the processes that stack started and report what "
                             "happened; refuses any PID whose command line is not this "
                             "stack's. Takes no other argument.")
    parser.add_argument("--fixtures", type=Path, help="the fixture root to serve")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--hold", type=float, default=0.0,
                        help="keep the stack up this long (seconds) after printing its ports")
    # S2 asks for the proof on PostgreSQL, because that is what production runs and because
    # retaining bytes and resolving them through a view is where dialects differ.
    parser.add_argument("--cube-postgres", metavar="DBNAME",
                        help="use a DISPOSABLE PostgreSQL database for the cube instead of "
                             "SQLite; PG* come from the environment and the database must "
                             "already exist and be throwaway")
    parser.add_argument("--holdout", metavar="YYYY-MM-DD",
                        help="declare a holdout start on the lake, so a range that crosses "
                             "it can be refused for that reason rather than untested")
    parser.add_argument("--host-pythonpath", action="append", default=[],
                        help="paths importable by the store hosts, so a CANDIDATE provider "
                             "can be exercised without installing it into a live venv")
    args = parser.parse_args(argv)
    if args.teardown is not None:
        state = json.loads(args.teardown.read_text(encoding="utf-8"))
        report = teardown(state)
        print(json.dumps(report, indent=1))
        return 0 if report["complete"] and not report["refused"] else 1
    for required in ("fixtures", "work"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required unless --teardown is given")
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
                                 "holdout_start": args.holdout,
                                 "spool_dir": str(work / "lake-spool"),
                                 "cuts_dir": str(work / "lake-cuts")}}}, indent=1))
    (work / "warehouse.json").write_text(json.dumps({
        "store_id": "olap_cube", "title": "disposable cube", "kind": "warehouse",
        "web_host": "127.0.0.1", "web_port": warehouse_port,
        "backend": {"entry_point": "predictor_olap", "distribution": "predictor-olap-store",
                    "settings": {"sqlite_path": None if args.cube_postgres
                                 else str(work / "cube.sqlite"),
                                 "schema": "public",
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
            # data-gov refuses to start when a lake declares a holdout and the policy does
            # not: the governance will not serve a boundary the access rules do not know.
            {"principal": "*", "lake": "synthetic_fixtures",
             "verbs": ["discover", "coverage", "read", "download"],
             **({"deny_from": args.holdout} if args.holdout else {})},
            {"principal": "*", "lake": "olap_cube",
             "verbs": ["discover", "query", "write_metrics", "write_terminal"]}],
        "lakes": [
            {"plugin": "http_lake", "lake_id": "synthetic_fixtures", "kind": "lake",
             "base_url": f"http://127.0.0.1:{lake_port}", "holdout_start": args.holdout},
            {"plugin": "http_lake", "lake_id": "olap_cube", "kind": "warehouse",
             "base_url": f"http://127.0.0.1:{warehouse_port}", "holdout_start": None}]}, indent=1))

    env = dict(os.environ, DATA_GOV_LAKE_TOKEN=token, PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    for name in ("PGDATABASE", "PGUSER", "PGPASSWORD", "PGHOST", "PGPORT"):
        env.pop(name, None)      # a disposable stack must not be able to reach the real cube
    if args.host_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(Path(item).resolve())
                                            for item in args.host_pythonpath)
    warehouse_env = dict(env)
    if args.cube_postgres:
        # The ONLY process given database credentials, and only for the throwaway database
        # this run names. Nothing else in the stack can open any PostgreSQL at all.
        # The guard names what must never be written by a disposable stack. Comparing
        # against PGDATABASE was wrong: the caller must export the throwaway name for
        # `createdb`, so that check refused exactly the correct invocation.
        protected = {name.strip() for name in
                     (os.environ.get("PROTECTED_DATABASES") or "predictor_olap").split(",")
                     if name.strip()}
        if args.cube_postgres in protected:
            raise SystemExit("refusing to run a disposable stack against a protected database "
                             f"{args.cube_postgres!r} (PROTECTED_DATABASES={sorted(protected)})")
        warehouse_env.update(
            PGDATABASE=args.cube_postgres,
            PGHOST=os.environ.get("PGHOST", "127.0.0.1"),
            PGPORT=os.environ.get("PGPORT", "5432"),
            PGUSER=os.environ.get("PGUSER", ""),
            PGPASSWORD=os.environ.get("PGPASSWORD", ""))
    processes = []
    try:
        processes.append(start([str(HOSTS_PYTHON), "-m", "data_lake_service.main",
                                "--load_config", str(work / "lake.json")], work, env,
                               work / "lake.log"))
        processes.append(start([str(HOSTS_PYTHON), "-m", "data_warehouse_service.main",
                                "--load_config", str(work / "warehouse.json")], work,
                               warehouse_env, work / "warehouse.log"))
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
                 "holdout_start": args.holdout,
                 "key_file": str(key_file), "fixtures": str(fixtures),
                 "cube": (f"postgresql:{args.cube_postgres}" if args.cube_postgres
                          else str(work / "cube.sqlite")),
                 "cube_postgres": args.cube_postgres,
                 "lake_pid": processes[0].pid, "warehouse_pid": processes[1].pid,
                 "gov_pid": processes[2].pid,
                 # disposable secrets of a disposable stack: recorded so a test can stop and
                 # restart one of these services exactly as it was started
                 "lake_token": token,
                 "warehouse_command": [str(HOSTS_PYTHON), "-m", "data_warehouse_service.main",
                                       "--load_config", str(work / "warehouse.json")],
                 "lake_command": [str(HOSTS_PYTHON), "-m", "data_lake_service.main",
                                  "--load_config", str(work / "lake.json")],
                 "work": str(work)}
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
