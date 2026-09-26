#!/usr/bin/env python3
"""DR01 follow-on (order 2026-09-26): a per-cell fit runner PROVES it is inside a reserved scope.

Why this exists, and why it is not a second reservation
-------------------------------------------------------
The per-cell fit runners spawn their children as plain subprocesses **inside their own cgroup**.
Those children therefore share the parent's ``MemoryMax``: the bytes are already accounted for by
the parent's reservation, and a runner that took a reservation of its own would count the same
bytes twice and, once the slice's aggregate budget is exhausted by its own double entry, queue
against itself -- a deadlock.  So the correct obligation is not "reserve" but **"prove somebody
reserved for you"**: the runner refuses to fit unless a live reservation covers the cgroup it is
running in.

Before this guard, such a runner was capped and reserved only when the *invocation* happened to go
through ``crispdm-run``.  Invoked bare it ran uncapped and unreserved, invisible to admission --
exactly the silent bypass DR01 left open (SATOSHI_DR01_ATOMIC_ADMISSION_2026_09_26.md §3(a)).

The proof is ``crispdm_admission.py inside-scope``, whose **exit code** is the verdict: 0 covered,
1 not covered.  A lease armed with the holder's pid rather than a cgroup (the ``PRLIMIT_AS``
fallback, where a child shares its parent's scope and the scope is deliberately *not* recorded as
the witness) is honoured as well, by checking whether a live lease's holder pid is an ancestor of
this process.  Both are reads; nothing is reserved, started, signalled or changed here.

A missing or unreadable admission module REFUSES.  That is the same failure direction the fleet
dispatcher takes: an unenforceable cap is not a cap.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

GUARD_EXIT = 75                 # EX_TEMPFAIL, the code crispdm-run has always used for a refusal
COVERED_BY_CGROUP = "COVERED_BY_CGROUP"
COVERED_BY_HOLDER_PID = "COVERED_BY_HOLDER_PID"
NOT_COVERED = "NOT_COVERED"
NO_ADMISSION_MODULE = "NO_ADMISSION_MODULE"
ADMISSION_UNREADABLE = "ADMISSION_UNREADABLE"


def admission_module() -> Path | None:
    """The one authority, looked up the way crispdm-run looks it up: an explicit override, the
    deployed copy, then this checkout's own tracked copy."""
    for candidate in (os.environ.get("CRISPDM_ADMISSION_MODULE"),
                      Path.home() / ".local/libexec/crispdm/crispdm_admission.py",
                      HERE / "crispdm_admission.py"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _run(module: Path, *args, timeout: float = 60.0):
    return subprocess.run([sys.executable, str(module), *args], capture_output=True, text=True,
                          timeout=timeout)


def _ancestors(pid: int | None = None) -> list:
    """This process and its ancestors, so a reservation held by a launcher above us is seen."""
    out, cur, seen = [], int(pid or os.getpid()), set()
    while cur and cur not in seen:
        out.append(cur)
        seen.add(cur)
        try:
            stat = Path(f"/proc/{cur}/stat").read_text()
            cur = int(stat.rsplit(")", 1)[-1].split()[1])      # ppid
        except (OSError, IndexError, ValueError):
            break
    return out


def coverage() -> dict:
    """Is a live reservation covering this process?  Reads only; never writes a lease.

    -> {"covered": bool, "code": ..., "cgroup": str|None, "covering_lease_ids": [...],
        "live_lease_ids": [...], "module": str|None, "detail": str}
    """
    module = admission_module()
    out = {"covered": False, "code": NOT_COVERED, "cgroup": None, "covering_lease_ids": [],
           "live_lease_ids": [], "module": None if module is None else str(module), "detail": ""}
    if module is None:
        return dict(out, code=NO_ADMISSION_MODULE,
                    detail="the shared admission module is not installed; run "
                           "tools/install_crispdm_launcher.sh")
    try:
        r = _run(module, "inside-scope")
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(out, code=ADMISSION_UNREADABLE, detail=f"{type(exc).__name__}: {exc}")
    try:
        doc = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        doc = {}
    out["cgroup"] = doc.get("cgroup")
    out["live_lease_ids"] = doc.get("live_lease_ids") or []
    out["covering_lease_ids"] = doc.get("covering_lease_ids") or []
    if r.returncode == 0:                                   # the exit code IS the verdict
        return dict(out, covered=True, code=COVERED_BY_CGROUP,
                    detail=f"reservation(s) {', '.join(out['covering_lease_ids'])} cover this cgroup")
    if r.returncode != 1:
        return dict(out, code=ADMISSION_UNREADABLE,
                    detail=f"inside-scope exited {r.returncode}: {(r.stderr or r.stdout).strip()[-200:]}")
    # A lease armed with its holder's pid and no cgroup (the PRLIMIT_AS fallback: the child shares
    # its parent's scope, so the scope must NOT be the witness) still covers this process when that
    # holder is one of our ancestors.
    try:
        st = json.loads(_run(module, "state").stdout or "{}")
    except (ValueError, OSError, subprocess.SubprocessError):
        st = {}
    mine = set(_ancestors())
    holders = [l.get("lease_id") for l in st.get("live", []) if l.get("pid") in mine]
    if holders:
        return dict(out, covered=True, code=COVERED_BY_HOLDER_PID, covering_lease_ids=holders,
                    detail=f"reservation(s) {', '.join(holders)} are held by an ancestor process")
    return dict(out, detail="no live reservation covers this process")


def refusal_text(tool: str, action: str, cov: dict, *, mem: str = "<MEM>", wall: str = "<WALL>",
                 name: str = "job", argv=None) -> str:
    cmd = " ".join(shlex.quote(str(a)) for a in (argv if argv is not None else sys.argv))
    return "\n".join([
        f"REFUSED: {tool} will not {action} outside a reserved scope ({cov['code']}).",
        f"  why      this runner spawns its children inside its OWN cgroup, so they share its",
        f"           MemoryMax; it must NOT take a second reservation (the same bytes would be",
        f"           counted twice and the request would queue against itself).  It is capped and",
        f"           reserved only when the invocation itself went through the launcher, so an",
        f"           invocation that cannot show a covering reservation is uncapped and invisible",
        f"           to admission -- it is refused instead.",
        f"  detail   {cov['detail']}",
        f"  cgroup   {cov['cgroup'] or 'unreadable'}",
        f"  leases   live: {', '.join(cov['live_lease_ids']) or 'none'}",
        f"  module   {cov['module'] or 'ABSENT'}",
        f"  start it as:",
        f"    $HOME/.local/bin/crispdm-run -m {mem} -t {wall} -n {name} -- {cmd}",
        "  nothing was started, and no limit, slice or kernel setting was changed.",
    ])


def require_reserved_scope(tool: str, action: str = "fit", *, mem: str = "<MEM>",
                           wall: str = "<WALL>", name: str = "job", argv=None,
                           stream=None) -> dict:
    """Prove coverage or refuse.  On refusal: print why and exit 75; start nothing."""
    cov = coverage()
    if cov["covered"]:
        return cov
    print(refusal_text(tool, action, cov, mem=mem, wall=wall, name=name, argv=argv),
          file=stream or sys.stderr, flush=True)
    raise SystemExit(GUARD_EXIT)


def main(argv=None) -> int:
    """`python3 tools/df_admission_guard.py [--require TOOL]` -- 0 covered, 1 not, 75 refused."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--require"]:
        require_reserved_scope(args[1] if len(args) > 1 else "this tool")
        return 0
    cov = coverage()
    print(json.dumps(cov, sort_keys=True))
    return 0 if cov["covered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
