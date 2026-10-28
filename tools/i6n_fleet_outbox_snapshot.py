#!/usr/bin/env python3
"""Collect four terminal-bound candidate outboxes without stopping live nodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from tools.i6n_outbox_snapshot import SnapshotPending, SnapshotRefusal, snapshot


class FleetSpecError(ValueError):
    pass


def _absolute(value, name) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise FleetSpecError(f"{name} must be an absolute path")
    return Path(value)


def load_spec(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "i6n.fleet_outbox_snapshot.spec.v1":
        raise FleetSpecError("unsupported spec schema")
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise FleetSpecError("nodes must contain exactly four entries")
    normalized = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != {
            "island_id", "host", "source", "terminal_receipt", "remote_terminal",
            "remote_snapshot", "remote_seal",
        }:
            raise FleetSpecError(f"nodes[{index}] has invalid keys")
        host = node["host"]
        if host is not None and (not isinstance(host, str) or not host):
            raise FleetSpecError(f"nodes[{index}].host is invalid")
        normalized.append({
            "island_id": node["island_id"], "host": host,
            **{name: _absolute(node[name], f"nodes[{index}].{name}")
               for name in ("source", "terminal_receipt", "remote_terminal",
                            "remote_snapshot", "remote_seal")},
        })
    if len({node["island_id"] for node in normalized}) != 4:
        raise FleetSpecError("island ids must be distinct")
    if len({str(node["terminal_receipt"]) for node in normalized}) != 4:
        raise FleetSpecError("terminal receipt paths must be distinct")
    return {
        "python": _absolute(raw["python"], "python"),
        "working_directory": _absolute(raw["working_directory"], "working_directory"),
        "collected_root": _absolute(raw["collected_root"], "collected_root"),
        "nodes": normalized,
    }


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def _snapshot_command(spec: dict, node: dict) -> list[str]:
    return [
        str(spec["python"]), "-m", "tools.i6n_outbox_snapshot",
        "--source", str(node["source"]),
        "--destination", str(node["remote_snapshot"]),
        "--terminal-receipt", str(node["remote_terminal"]),
        "--receipt", str(node["remote_seal"]),
    ]


def _copy_remote(host: str, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        result = _run(["scp", "-q", f"{host}:{source}", str(tmp)])
        if result.returncode:
            raise SnapshotPending(f"REMOTE_COPY_PENDING[{host}]: {result.stderr.strip()[:200]}")
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def collect(spec: dict) -> dict:
    completed, waiting = {}, {}
    for node in spec["nodes"]:
        island = node["island_id"]
        terminal = node["terminal_receipt"]
        if not terminal.is_file():
            waiting[island] = "TERMINAL_NOT_AVAILABLE"
            continue
        collected_dir = spec["collected_root"] / island
        collected_db = collected_dir / "outbox.sqlite"
        collected_seal = collected_dir / "SEAL.json"
        try:
            if node["host"] is None:
                result = snapshot(
                    node["source"], collected_db,
                    terminal_receipt=terminal, receipt=collected_seal,
                )
            else:
                host = node["host"]
                if not (collected_db.is_file() and collected_seal.is_file()):
                    mkdir = _run(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p",
                                  str(node["remote_terminal"].parent),
                                  str(node["remote_snapshot"].parent)])
                    if mkdir.returncode:
                        raise SnapshotPending(f"REMOTE_UNREACHABLE[{host}]")
                    upload = _run(["scp", "-q", str(terminal),
                                   f"{host}:{node['remote_terminal']}"])
                    if upload.returncode:
                        raise SnapshotPending(f"TERMINAL_UPLOAD_PENDING[{host}]")
                    command = "cd {} && PYTHONPATH=. {}".format(
                        shlex.quote(str(spec["working_directory"])),
                        shlex.join(_snapshot_command(spec, node)),
                    )
                    remote = _run(["ssh", "-o", "BatchMode=yes", host, "bash", "-lc", command])
                    if remote.returncode == 4:
                        raise SnapshotPending(f"REMOTE_OUTBOX_PENDING[{host}]")
                    if remote.returncode:
                        raise SnapshotRefusal(
                            f"REMOTE_SNAPSHOT_REFUSED[{host}]: {remote.stdout[-300:]} {remote.stderr[-300:]}")
                    _copy_remote(host, node["remote_snapshot"], collected_db)
                    _copy_remote(host, node["remote_seal"], collected_seal)
                result = snapshot(
                    collected_db, collected_db,
                    terminal_receipt=terminal, receipt=collected_seal,
                )
            completed[island] = {
                "state": result["state"], "items": result["item_count"],
                "candidate_ledger_sha256": result["candidate_ledger_sha256"],
                "items_sha256": result["items_sha256"],
            }
        except SnapshotPending as exc:
            waiting[island] = str(exc)
    state = "COMPLETE" if len(completed) == 4 else "WAIT"
    return {"schema": "i6n.fleet_outbox_snapshot.status.v1", "state": state,
            "completed": completed, "waiting": waiting}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = collect(load_spec(args.spec))
    except (FleetSpecError, SnapshotRefusal, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"state": "REFUSED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["state"] == "COMPLETE" else 4


if __name__ == "__main__":
    sys.exit(main())
