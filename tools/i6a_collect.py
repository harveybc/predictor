"""Wait for sharded I6-A workers, then close and publish one complete target.

Each source is LABEL=SSH_HOST:/absolute/results/path. The worker status files
are authoritative; partial rows are never summarized or published.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from tools.i6a_campaign import atomic_json
from tools.i6a_close import close
from tools.i6a_publish_olap import main as publish


def source_spec(value):
    label, sep, remote = value.partition("=")
    host, colon, path = remote.partition(":")
    if (not sep or not colon or not re.fullmatch(r"[a-zA-Z0-9_-]+", label)
            or not re.fullmatch(r"[a-zA-Z0-9_.-]+", host) or not path.startswith("/")):
        raise ValueError("INVALID_REMOTE_SOURCE")
    return label, host, path.rstrip("/")


def complete_status(status):
    return (status.get("state") == "COMPLETE"
            and type(status.get("completed")) is int
            and type(status.get("total")) is int
            and status["total"] > 0
            and status["completed"] == status["total"])


def remote_status(label, host, path):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
               "cat", f"{path}/STATUS_{label}.json"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"REMOTE_STATUS_UNAVAILABLE: {label}: {result.returncode}")
    status = json.loads(result.stdout)
    if status.get("host") != label:
        raise ValueError(f"REMOTE_STATUS_IDENTITY_MISMATCH: {label}")
    if status.get("failed"):
        raise RuntimeError(f"REMOTE_CELL_FAILED: {label}: {status['failed']}")
    return status


def collect(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--destination", required=True, type=Path)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--closure", required=True, type=Path)
    ap.add_argument("--warehouse-url", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--timeout-seconds", type=int, default=7200)
    args = ap.parse_args(argv)
    if args.poll_seconds < 1 or args.timeout_seconds < args.poll_seconds:
        raise ValueError("INVALID_POLL_BUDGET")
    sources = [source_spec(value) for value in args.source]
    if len({label for label, _, _ in sources}) != len(sources):
        raise ValueError("DUPLICATE_SOURCE_LABEL")
    args.destination.mkdir(parents=True, exist_ok=True)
    status_file = args.destination / "STATUS.json"
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        statuses = {}
        for label, host, path in sources:
            statuses[label] = remote_status(label, host, path)
        total = sum(status["total"] for status in statuses.values())
        atomic_json(status_file, {"state": "WAITING", "target": args.target,
                                  "workers": statuses, "expected_cells": total})
        if all(complete_status(status) for status in statuses.values()):
            if total != 208:
                raise ValueError(f"CAMPAIGN_DENOMINATOR_MISMATCH: {total}")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("CAMPAIGN_TIMEOUT")
        time.sleep(args.poll_seconds)

    roots = []
    for label, host, path in sources:
        destination = args.destination / label
        destination.mkdir(exist_ok=True)
        subprocess.run(["rsync", "-a", "--include=ARCH_*.json", "--exclude=*",
                        f"{host}:{path}/", str(destination) + "/"], check=True)
        roots.append(destination)
    freeze = json.loads(args.freeze.read_text())
    closure = close(freeze, roots, args.target)
    atomic_json(args.closure, closure)
    if closure["state"] != "COMPLETE":
        atomic_json(status_file, {"state": "INCOMPLETE_EVIDENCE", "target": args.target,
                                  "closure_sha256": closure["sha256"], "problems": closure["problems"]})
        return 2
    atomic_json(status_file, {"state": "CLOSED", "target": args.target,
                              "closure_sha256": closure["sha256"]})
    publish_args = ["--freeze", str(args.freeze), "--target", args.target,
                    "--url", args.warehouse_url, "--token-file", args.token_file,
                    "--code-commit", args.code_commit]
    for root in roots:
        publish_args += ["--result-dir", str(root)]
    publish(publish_args)
    atomic_json(status_file, {"state": "PUBLISHED", "target": args.target,
                              "closure_sha256": closure["sha256"], "reports": 208})
    return 0


if __name__ == "__main__":
    raise SystemExit(collect())
