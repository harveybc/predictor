#!/usr/bin/env python3
"""Whose memory is it: slice membership as EVIDENCE, not a host-wide resident set as a guess.

The correction this exists for.  The QRM02/F3 telemetry lane attributed a batch slice's residual
charge to the MT5 paper-trading virtual machine on the strength of the VM's **host-wide resident
set** (`qemu-system-x86`, one process, 5 785 092 096 B).  A resident set is a figure about a
process's own address space in HOST RAM.  It says nothing whatever about which cgroup the kernel
charges that memory to, so it is **not evidence of slice membership** and cannot explain a slice's
own charge.  Two scopes were added that must never be added.

Membership is a fact with a reading:

  * a cgroup charges the tasks in it and in its descendants -- `cgroup.procs`, transitively;
  * `/proc/<pid>/cgroup` says, for one task, which cgroup that is;
  * so a process is a member of a slice **iff** its own cgroup path is the slice or is under it.

This module answers three separate questions and never merges their answers:

  1. **SLICE_CHARGE**   `memory.current` / `memory.peak` of the slice, as the KERNEL charges it.
  2. **SLICE_MEMBERS**  the processes actually inside the slice, with their resident sets -- which
     are still HOST RAM per process, are still not the slice's accounting, and are reported only to
     say *who is in there*.
  3. **HOST_WIDE**      named holders that are NOT in the slice.  Their resident sets are host-wide
     and are `NOT_A_SLICE_CHARGE` by membership, whatever their size.

The residual -- the part of the slice's charge no member accounts for -- is attributed to a member
or to nobody.  `UNDETERMINED` is the answer when membership cannot be established, and it is a
finding, not a gap to be filled with the largest process on the host.

Nothing here writes, kills, reclaims, restarts or changes a limit.  Every read is of `/proc` and
`/sys/fs/cgroup`, and no host name, account identifier or VM disk path is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SCHEMA = "df_slice_membership.v1"

MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"
UNDETERMINED = "UNDETERMINED"

NOT_A_SLICE_CHARGE = ("NOT_A_SLICE_CHARGE: this holder is not a task of the slice or of any "
                      "descendant of it, so the kernel does not charge its memory to the slice. "
                      "Its resident set is HOST-WIDE and may not be offered as an explanation of "
                      "the slice's own charge")

SLICE_SCOPE = ("SLICE_CGROUP_CHARGE: memory.current and memory.peak of the named slice, as the "
               "kernel charges the whole subtree. This is the only figure in this record that is "
               "the slice's accounting, and it is never added to any resident set")
RSS_SCOPE = ("PROCESS_RESIDENT_SET: one process's VmRSS in HOST RAM. It is not a cgroup charge, it "
             "double-counts shared pages between processes, and it is recorded here only to say "
             "which processes are inside a scope -- never to size or explain that scope's charge")

CGROUP_ROOT = Path(os.environ.get("CRISPDM_CGROUP_ROOT", "/sys/fs/cgroup"))


def _int(path: Path):
    try:
        v = path.read_text().strip()
    except OSError:
        return None
    return int(v) if v.isdigit() else None


def _own_user_slice() -> str:
    try:
        line = Path("/proc/self/cgroup").read_text().strip().splitlines()[0]
    except OSError:
        return ""
    rel = line.split("::", 1)[1].lstrip("/") if "::" in line else ""
    m = re.match(r"(user\.slice/user-\d+\.slice/user@\d+\.service)", rel)
    return m.group(1) if m else ""


def find_slice(name: str = "crispdm-batch.slice") -> dict:
    """Locate one slice under this user's manager, by search rather than by a hardcoded path."""
    base = CGROUP_ROOT / _own_user_slice() if _own_user_slice() else CGROUP_ROOT
    for d in [base, CGROUP_ROOT]:
        if not d.is_dir():
            continue
        for cand in d.rglob(name):
            if cand.is_dir():
                return {"status": MEASURED, "rel": str(cand.relative_to(CGROUP_ROOT)),
                        "path": str(cand)}
    return {"status": UNKNOWN, "rel": None, "path": None,
            "why": f"no cgroup directory named {name} exists under {CGROUP_ROOT}"}


def slice_charge(path) -> dict:
    p = Path(path)
    cur, peak, mx = _int(p / "memory.current"), _int(p / "memory.peak"), None
    raw = (p / "memory.max").read_text().strip() if (p / "memory.max").is_file() else ""
    mx = None if raw in ("", "max") else int(raw)
    return {"scope": SLICE_SCOPE, "status": MEASURED if cur is not None else UNKNOWN,
            "current_bytes": cur, "peak_bytes": peak, "max_bytes": mx,
            "basis": "memory.current / memory.peak / memory.max of the slice subtree"}


def _procs(path: Path) -> list:
    out = []
    for f in [path / "cgroup.procs"] + [d / "cgroup.procs" for d in path.rglob("*") if d.is_dir()]:
        try:
            out += [int(x) for x in f.read_text().split() if x.isdigit()]
        except OSError:
            continue
    return sorted(set(out))


def _proc_facts(pid: int) -> dict:
    st = Path(f"/proc/{pid}/status")
    try:
        text = st.read_text()
    except OSError:
        return {"pid": pid, "status": UNKNOWN, "why": "the process ended before it was read"}
    rss = next((int(l.split()[1]) * 1024 for l in text.splitlines()
                if l.startswith("VmRSS:")), None)
    name = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                 if l.startswith("Name:")), None)
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[0]
        cg = cg.split("::", 1)[1].lstrip("/") if "::" in cg else None
    except OSError:
        cg = None
    # The executable NAME only.  No argv: a VM's command line carries disk paths and an account
    # identifier, and AGENTS.md forbids writing either into this repository.
    return {"pid": pid, "status": MEASURED, "name": name, "rss_bytes": rss, "cgroup": cg}


def membership(pid_cgroup: str | None, slice_rel: str | None) -> dict:
    """Is this process charged to the slice?  A path comparison, which is the whole evidence."""
    if not pid_cgroup or not slice_rel:
        return {"member": None, "verdict": UNDETERMINED,
                "why": ("the process's own cgroup or the slice's path could not be read, so "
                        "membership is UNDETERMINED -- which is a finding, not a licence to "
                        "attribute the charge to the largest process on the host")}
    a, b = pid_cgroup.strip("/"), slice_rel.strip("/")
    inside = a == b or a.startswith(b + "/")
    return {"member": inside, "verdict": MEASURED,
            "why": (f"the process's cgroup is {'inside' if inside else 'outside'} the slice, so the "
                    f"kernel {'charges' if inside else 'does not charge'} its memory to it")}


def attribute(slice_name: str = "crispdm-batch.slice", *, holders=("qemu-system-x86",)) -> dict:
    """The slice's charge, its members, the named host-wide holders, and the residual's owner."""
    loc = find_slice(slice_name)
    out = {"schema": SCHEMA, "slice": slice_name, "location": loc,
           "scopes_are_never_merged": (
               "three scopes appear in this record -- the slice's cgroup charge, each member's "
               "resident set and each non-member's resident set. Each carries its own name and "
               "basis, and no two of them are ever added")}
    if loc["status"] != MEASURED:
        out["status"] = UNKNOWN
        out["residual"] = {"bytes": None, "owner": UNDETERMINED, "why": loc["why"]}
        return out
    charge = slice_charge(loc["path"])
    out["slice_charge"] = charge
    members = [_proc_facts(p) for p in _procs(Path(loc["path"]))]
    out["members"] = {"scope": RSS_SCOPE, "count": len(members),
                      "processes": [m for m in members if m.get("status") == MEASURED],
                      "sum_of_member_resident_sets_bytes": sum(
                          m.get("rss_bytes") or 0 for m in members if m.get("status") == MEASURED),
                      "why_the_sum_is_not_the_charge": (
                          "resident sets double-count shared pages and exclude kernel memory the "
                          "cgroup is charged for, so their sum neither equals nor bounds "
                          "memory.current; it is printed to be compared, never substituted")}

    # the named host-wide holders, and whether membership can be established for them at all
    named = []
    for pid in _pids_named(holders):
        f = _proc_facts(pid)
        if f.get("status") != MEASURED:
            continue
        m = membership(f.get("cgroup"), loc["rel"])
        f["membership"] = m
        f["disposition"] = (NOT_A_SLICE_CHARGE if m["member"] is False else
                            ("A_SLICE_MEMBER" if m["member"] else UNDETERMINED))
        named.append(f)
    out["named_host_wide_holders"] = {"scope": RSS_SCOPE, "processes": named,
                                      "searched_for": list(holders)}

    unexplained = None
    if charge.get("current_bytes") is not None:
        unexplained = int(charge["current_bytes"]) - int(
            out["members"]["sum_of_member_resident_sets_bytes"])
    non_members = [h for h in named if h["membership"]["member"] is False]
    undetermined = [h for h in named if h["membership"]["member"] is None]
    if not named:
        owner, why = (UNDETERMINED,
                      "none of the named holders is running on this host, so no attribution of "
                      "this slice's residual to any of them is even available here")
    elif non_members and not undetermined:
        owner, why = (UNDETERMINED,
                      "every named holder is OUTSIDE the slice by membership evidence, so none of "
                      "them can account for the slice's charge. The residual's owner is "
                      "UNDETERMINED: it belongs to the slice's own members, to kernel memory "
                      "charged to the slice, or to shared pages -- and saying which would need a "
                      "per-member measurement this record does not have")
    elif undetermined:
        owner, why = (UNDETERMINED,
                      "membership could not be established for at least one named holder, so its "
                      "charge is UNDETERMINED rather than assumed either way")
    else:
        owner, why = ("SLICE_MEMBERS",
                      "the named holders are inside the slice, so their charge is the slice's; "
                      "their resident sets still do not sum to memory.current and are not offered "
                      "as if they did")
    out["residual"] = {
        "bytes": unexplained, "owner": owner, "why": why,
        "definition": ("the slice's memory.current minus the sum of its members' resident sets. "
                       "Because those two quantities have different bases, this difference is an "
                       "INDICATOR of how much of the charge the member list does not explain -- it "
                       "is not itself a measured quantity of anything"),
        "host_wide_rss_is_not_evidence_of_membership": (
            "a holder's host-wide resident set was the basis of the withdrawn attribution. It is "
            "not evidence of slice membership and it is not used as such anywhere in this record")}
    out["status"] = MEASURED
    out["mutations"] = ("NONE: every read is of /proc and /sys/fs/cgroup. No process was signalled, "
                        "no memory reclaimed, no shared memory cleared, no limit moved and no "
                        "service touched")
    return out


def _pids_named(names) -> list:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            comm = (d / "comm").read_text().strip()
        except OSError:
            continue
        if any(n.startswith(comm) or comm.startswith(n[:15]) for n in names):
            out.append(int(d.name))
    return sorted(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", default="crispdm-batch.slice")
    p.add_argument("--holder", action="append", default=None,
                   help="an executable name to test for slice membership (repeatable)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rec = attribute(a.slice, holders=tuple(a.holder or ("qemu-system-x86",)))
    text = json.dumps(rec, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
