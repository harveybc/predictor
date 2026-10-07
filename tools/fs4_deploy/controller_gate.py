#!/usr/bin/env python3
"""Forced-command gate for a worker's SSH key on the coordinator (authorized_keys `command=`).

A worker may do exactly one thing over this key: call the FS4 controller's claim, complete, fail,
heartbeat, status or list verbs against the one queue. The gate parses $SSH_ORIGINAL_COMMAND with
shlex (no shell is ever involved), requires

    <python> <controller> --db <db> <verb> [--owner OWNER] [verb options]

with python, controller and db equal to the values the gate was started with, an allowed verb, an
owner that starts with the key's role prefix, and then replaces itself with that exact argv. Anything
else exits 126 without running. Install (owner action on the coordinator), one line per worker key:

    command="python3 <state>/code/PREDICTOR_CURRENT/tools/fs4_deploy/controller_gate.py --role worker_b \
 --python python3 --controller <state>/code/PREDICTOR_CURRENT/tools/fs4_campaign.py --db <state>/queue_v2.sqlite",\
restrict ssh-ed25519 AAAA... worker_b-fs4
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys

VERBS = {"claim", "complete", "fail", "heartbeat", "status", "list"}


def validate(original: str, role: str, python: str, controller: str, db: str) -> list[str]:
    """Return the argv to execute, or raise PermissionError."""
    try:
        argv = shlex.split(original)
    except ValueError as exc:
        raise PermissionError(f"UNPARSEABLE_COMMAND: {exc}")
    if len(argv) < 5 or argv[:5] != [python, controller, "--db", db, argv[4]] or argv[4] not in VERBS:
        raise PermissionError("COMMAND_NOT_THE_FS4_CONTROLLER")
    verb, rest = argv[4], argv[5:]
    if verb in {"claim", "complete", "fail", "heartbeat"}:
        if len(rest) < 2 or rest[0] != "--owner" or not rest[1].startswith(f"{role}-"):
            raise PermissionError("OWNER_NOT_THIS_ROLE")
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    try:
        argv = validate(os.environ.get("SSH_ORIGINAL_COMMAND", ""), args.role, args.python, args.controller, args.db)
    except PermissionError as exc:
        print(f"fs4 gate refused: {exc}", file=sys.stderr)
        raise SystemExit(126)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
