#!/usr/bin/env python3
"""Launch ONE heavy cell through `crispdm-run` only when this host's live MemAvailable allows it, and record every reading.

The owner's standing placement rule (2026-09-25, and the 2026-09-26 grant for this lane) is that a job is placed by the
LIVE memory of the host, that the reading is taken immediately before the launch, and that a job whose measured pilot peak
does not fit is NOT started - it waits and the reading is taken again. This tool is that rule as code, so the readings are
evidence instead of a claim:

  * `--peak-bytes` is the arm's OWN measured pilot peak RSS, read from a retained pilot record by the caller, never typed;
  * a launch is REFUSED while MemAvailable < peak + `--margin-bytes` (1 GiB by default);
  * a launch is also held while MemAvailable - 3 GiB < the requested cap, because that is `crispdm-run`'s own refusal and
    burning it as a failed launch teaches nothing;
  * every poll is appended to MEMORY_GATE.jsonl in the run root with its timestamp, the reading and the verdict;
  * the command runs under `$HOME/.local/bin/crispdm-run -m CAP -t WALL -n NAME --`, never bypassed: its cgroup MemoryMax
    with MemorySwapMax=0 contains an out-of-memory kill to this job alone. The owner's machine has been OOM-killed before
    by an uncapped run; that is why the cap is the only sanctioned path and why this tool does not offer a way around it.

It starts at most one job and it never starts a second while the first is alive: one cell resident at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1 << 30
HOST_RESERVE = 3 * GIB          # crispdm-run's own reserve above which it refuses a request


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise SystemExit("REFUSED: /proc/meminfo has no MemAvailable")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the run root; MEMORY_GATE.jsonl is appended there")
    ap.add_argument("--label", required=True, help="what is being launched, for the log")
    ap.add_argument("--peak-bytes", type=int, required=True, help="the arm's measured pilot peak RSS, from its retained record")
    ap.add_argument("--margin-bytes", type=int, default=GIB)
    ap.add_argument("--cap", required=True, help="the crispdm-run -m value, e.g. 11G")
    ap.add_argument("--wall", required=True, help="the crispdm-run -t value, e.g. 7200")
    ap.add_argument("--name", default="q2deep")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-seconds", type=int, default=3600)
    ap.add_argument("--cap-bytes", type=int, required=True, help="the --cap value in bytes, so the hold is exact")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = [c for c in a.command if c != "--"]
    if not cmd:
        raise SystemExit("REFUSED: no command")
    log = a.root / "MEMORY_GATE.jsonl"
    a.root.mkdir(parents=True, exist_ok=True)
    need = a.peak_bytes + a.margin_bytes
    t0 = time.monotonic()
    while True:
        avail = mem_available_bytes()
        headroom_ok = avail >= need
        crispdm_ok = a.cap_bytes <= avail - HOST_RESERVE
        rec = {"at": _now(), "label": a.label, "mem_available_bytes": avail,
               "measured_pilot_peak_bytes": a.peak_bytes, "margin_bytes": a.margin_bytes,
               "required_bytes": need, "cap": a.cap, "cap_bytes": a.cap_bytes,
               "fits_measured_peak_plus_margin": headroom_ok,
               "crispdm_run_would_accept_the_cap": crispdm_ok,
               "verdict": "LAUNCH" if (headroom_ok and crispdm_ok) else "HELD_WAITING_FOR_MEMORY",
               "waited_seconds": round(time.monotonic() - t0, 1)}
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if headroom_ok and crispdm_ok:
            break
        if time.monotonic() - t0 >= a.max_wait_seconds:
            with open(log, "a") as fh:
                fh.write(json.dumps({**rec, "verdict": "NOT_STARTED_MEMORY_NEVER_ALLOWED_IT"}) + "\n")
            print(json.dumps({"label": a.label, "started": False,
                              "why": "MemAvailable never reached the measured pilot peak plus the margin, and the cap was "
                                     "never one crispdm-run would accept, within the declared wait; the cell was NOT started "
                                     "and the guard was NOT bypassed"}), flush=True)
            return 75
        time.sleep(a.poll_seconds)
    launcher = str(Path(os.path.expanduser("~/.local/bin/crispdm-run")))
    full = [launcher, "-m", a.cap, "-t", a.wall, "-n", a.name, "--", *cmd]
    started = _now()
    proc = subprocess.run(full, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    with open(log, "a") as fh:
        fh.write(json.dumps({"at": _now(), "label": a.label, "verdict": "FINISHED", "started_at": started,
                             "exit_code": proc.returncode, "mem_available_bytes_after": mem_available_bytes(),
                             "command": full}) + "\n")
    print(json.dumps({"label": a.label, "exit_code": proc.returncode}), flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
