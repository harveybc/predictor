"""Orphan guard: stop the admitted child when the supervising process disappears.

The evaluator runs inside crispdm-run's own systemd scope, outside the DOIN
node's cgroup, so stopping or killing the node would not stop it. This tiny
process sits between the supervisor and the launcher: when its parent pid is
gone it sends TERM to the launcher (which forwards it to its scope and
releases the reservation), then KILL after a grace period. A lost node can
therefore never leave a fit running beside a peer's recovery attempt.

    python -m tools.i6n_swarm.childguard --parent-pid <pid> [--poll 2] -- <launcher argv...>
"""
import argparse
import os
import signal
import subprocess
import sys
import time


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: childguard --parent-pid PID -- command...")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--grace", type=float, default=30.0)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    child = subprocess.Popen(command)  # stdin/stdout/stderr inherited: the receipt flows through unchanged
    # This guard is its own process (main thread); TERM from the supervisor is forwarded to the
    # launcher, which stops its scope and releases its reservation before exiting.
    signal.signal(signal.SIGTERM, lambda signum, frame: child.send_signal(signal.SIGTERM))
    while True:
        try:
            code = child.wait(timeout=args.poll)
            return 128 - code if code < 0 else code  # shell convention: SIGKILL -> 137
        except subprocess.TimeoutExpired:
            pass
        if os.getppid() != args.parent_pid:
            child.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + args.grace
            while child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if child.poll() is None:
                child.kill()
            child.wait()
            return 70


if __name__ == "__main__":
    sys.exit(main())
