#!/usr/bin/env python3
"""Exclusive shard claims for the phase 2/3 selection campaign (§E.6: every host claims shards
exclusively from the plan; workers cannot reach each other).

Three roles (coordinator, worker_a, worker_b) share one plan and one assignment
(tools/fs_phase23_deploy/shard_policy.py). Each host has a local state directory with:

    claims/<shard_id>/claim.<own_role>.json        written ONLY by this host
    peer_claims/<shard_id>/claim.<other_role>.json  relayed in by the coordinator loop
    terminals/<shard_id>/...                        this host's terminal outputs
    peer_terminals/<shard_id>/...                   relayed terminals of the other roles
    RELAY_HEARTBEAT.json                            monotonically increasing relay cycle

Protocol for one worker slot (several slots may run on one host):

1. ``next``: walk the shards the assignment gives THIS role, in plan order, and claim the
   first one that is not terminal anywhere visible and not claimed live by anyone. Local
   exclusivity between slots of the same host is the atomic O_EXCL creation of the claim
   file; a live own claim whose pid is gone on this host (a slot that died) is an orphan
   and is re-claimed, reported in the claim's ``reclaimed_orphan`` field. When every own
   shard is terminal or claimed, and ``--steal`` is given, claim a shard assigned to
   another role that is free everywhere visible, walking the others' lists backwards so
   the stealer and the home role meet as late as possible; a stale live peer claim (older than ``--stale-after-seconds`` with no terminal) is not a
   blocker.
2. ``settle``: an own-assigned shard needs no wait, the home role always wins. A STOLEN
   shard waits until the relay heartbeat has advanced ``min_cycle_advance`` cycles past
   the cycle seen at claim time (so every claim the home role wrote before seeing ours has
   reached us), then arbitrates: the home role wins when it holds a non-ABANDONED claim;
   otherwise the earliest ``claimed_at_utc``. The loser marks ABANDONED before computing.
3. ``mark`` RUNNING, then COMPLETED (with the terminal's results digest) or FAILED with the
   exact reason. A valid terminal is never recomputed; a FAILED shard is never retried
   blindly (the status shows its reason, a human decides).

A terminal is valid when ``<dir>/run_manifest.json`` (or ``terminal.json``) says
``status: COMPLETED`` and its ``results_sha256`` equals the digest of the results file it
names (``results_file``, default ``results.jsonl``). Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "fs_phase23_shard_claim.v1"
ROLES = ("coordinator", "worker_a", "worker_b")
LIVE_STATES = ("CLAIMED", "RUNNING")
TERMINAL_STATES = ("COMPLETED", "FAILED")
MANIFEST_NAMES = ("run_manifest.json", "terminal.json")
FAILED_MARKERS = ("FAILED.json", "FAILED.reason.txt")

EXIT_OK = 0
EXIT_NONE = 10      # nothing claimable now (all own shards terminal/claimed, nothing to steal)
EXIT_LOSE = 11
EXIT_WAIT = 12
EXIT_ALL_DONE = 13  # every shard of the plan is terminal somewhere visible
EXIT_ERROR = 2


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: Any) -> _dt.datetime:
    return _dt.datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def create_exclusive_json(path: Path, payload: Dict[str, Any]) -> bool:
    """Create ``path`` only if it does not exist (O_EXCL): the local slot race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return True


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def pid_alive(pid: Any) -> bool:
    """Liveness of a LOCAL pid, read from /proc (no signal is ever sent)."""
    try:
        return Path("/proc", str(int(pid))).is_dir()
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------ terminals


def valid_terminal(directory: Path) -> Optional[str]:
    for name in MANIFEST_NAMES:
        manifest = read_json(directory / name)
        if not manifest:
            continue
        if manifest.get("status") != "COMPLETED":
            return None
        expected = manifest.get("results_sha256") or manifest.get("results_digest")
        results = directory / str(manifest.get("results_file") or "results.jsonl")
        try:
            actual = hashlib.sha256(results.read_bytes()).hexdigest()
        except OSError:
            return None
        return actual if isinstance(expected, str) and expected.lower() == actual else None
    return None


def failed_marker(directory: Path) -> Optional[Path]:
    for name in FAILED_MARKERS:
        if (directory / name).is_file():
            return directory / name
    return None


def terminal_state(shard: str, roots: Iterable[Path]) -> Optional[Dict[str, Any]]:
    for root in roots:
        d = Path(root) / shard
        digest = valid_terminal(d)
        if digest:
            return {"kind": "COMPLETED", "path": str(d), "results_sha256": digest}
        marker = failed_marker(d)
        if marker:
            return {"kind": "FAILED", "path": str(marker)}
    return None


# --------------------------------------------------------------------------------- claims


class State:
    """All the paths one host needs; ``role`` is this host's role."""

    def __init__(self, state_dir: Path, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        self.role = role
        self.state_dir = Path(state_dir)
        self.claims = self.state_dir / "claims"
        self.peer_claims = self.state_dir / "peer_claims"
        self.terminals = self.state_dir / "terminals"
        self.peer_terminals = self.state_dir / "peer_terminals"
        self.heartbeat = self.state_dir / "RELAY_HEARTBEAT.json"

    def own_claim_path(self, shard: str) -> Path:
        return self.claims / shard / f"claim.{self.role}.json"

    def all_claims(self, shard: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        own = read_json(self.own_claim_path(shard))
        if own and own.get("host_role") == self.role:
            out[self.role] = own
        for role in ROLES:
            if role == self.role:
                continue
            peer = read_json(self.peer_claims / shard / f"claim.{role}.json")
            if peer and peer.get("host_role") == role:
                out[role] = peer
        return out

    def terminal(self, shard: str) -> Optional[Dict[str, Any]]:
        return terminal_state(shard, [self.terminals, self.peer_terminals])

    def heartbeat_cycle(self) -> Optional[int]:
        beat = read_json(self.heartbeat)
        cycle = beat.get("cycle") if beat else None
        return cycle if isinstance(cycle, int) else None


def is_stale(claim: Dict[str, Any], stale_after_seconds: Optional[int], now: _dt.datetime) -> bool:
    if stale_after_seconds is None or claim.get("state") not in LIVE_STATES:
        return False
    try:
        when = parse_utc(claim.get("claimed_at_utc"))
    except (TypeError, ValueError):
        return False
    return (now - when).total_seconds() > stale_after_seconds


def arbitrate(claims: Dict[str, Dict[str, Any]], home_role: str) -> Optional[str]:
    home = claims.get(home_role)
    if home and home.get("state") != "ABANDONED":
        return home_role
    contenders: List[Tuple[_dt.datetime, str]] = []
    for role, claim in claims.items():
        if claim.get("state") == "ABANDONED":
            continue
        try:
            contenders.append((parse_utc(claim.get("claimed_at_utc")), role))
        except (TypeError, ValueError):
            continue
    return min(contenders)[1] if contenders else None


def home_role_of(assignment: Dict[str, Any], shard: str) -> Optional[str]:
    for role, body in assignment["roles"].items():
        if shard in body.get("shards", ()):
            return role
    return None


def plan_order(plan: Dict[str, Any]) -> List[str]:
    shards = plan.get("shards")
    if shards is None and isinstance(plan.get("populations"), list):
        shards = [s for pop in plan["populations"] for s in pop.get("shards", [])]
    return [str(s.get("shard_id", s.get("id", s.get("shard")))) for s in (shards or [])]


def new_claim(state: State, shard: str, home_role: str, slot: str, extra: Optional[Dict[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": SCHEMA, "shard_id": shard, "host_role": state.role, "home_role": home_role, "slot": slot,
        "state": "CLAIMED", "claimed_at_utc": utc_now(), "pid": os.getpid(),
        "heartbeat_cycle_at_claim": state.heartbeat_cycle(), "history": [],
    }
    payload.update({k: v for k, v in fields.items() if v is not None})
    payload.update(extra or {})
    return payload


def shard_blocked(state: State, shard: str, stale_after_seconds: Optional[int], now: _dt.datetime) -> Optional[Dict[str, Any]]:
    """Why ``shard`` cannot be claimed now (terminal, or live claim by someone), else None."""
    terminal = state.terminal(shard)
    if terminal:
        return {"reason": "TERMINAL", "terminal": terminal}
    claims = state.all_claims(shard)
    for role, claim in claims.items():
        st = claim.get("state")
        if st in TERMINAL_STATES:
            return {"reason": "TERMINAL_CLAIM", "role": role, "state": st}
        if st in LIVE_STATES:
            if role == state.role:
                if pid_alive(claim.get("pid")):
                    return {"reason": "OWN_LIVE_CLAIM", "slot": claim.get("slot"), "pid": claim.get("pid")}
                continue  # own orphan: reclaimable
            if is_stale(claim, stale_after_seconds, now):
                continue
            return {"reason": "PEER_LIVE_CLAIM", "role": role, "state": st}
    return None


def next_shard(state: State, plan: Dict[str, Any], assignment: Dict[str, Any], slot: str, steal: bool = False,
               stale_after_seconds: Optional[int] = None, extra: Optional[Dict[str, Any]] = None,
               now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    order = plan_order(plan)
    own = [s for s in order if home_role_of(assignment, s) == state.role]
    # a stealer walks the other roles' shards BACKWARDS: the home role walks forwards, so the
    # two meet as late as possible and most steals settle without a contest
    others = [s for s in reversed(order) if home_role_of(assignment, s) not in (None, state.role)]
    candidates = [(s, False) for s in own] + ([(s, True) for s in others] if steal else [])
    for shard, stolen in candidates:
        if shard_blocked(state, shard, stale_after_seconds, now):
            continue
        own_prev = read_json(state.own_claim_path(shard))
        if own_prev and own_prev.get("state") == "ABANDONED" and stolen:
            continue  # we already lost this one once; the home role has it
        reclaimed = None
        if own_prev and own_prev.get("state") in LIVE_STATES:
            reclaimed = {"slot": own_prev.get("slot"), "pid": own_prev.get("pid"), "claimed_at_utc": own_prev.get("claimed_at_utc")}
        payload = new_claim(state, shard, home_role_of(assignment, shard) or state.role, slot, extra,
                            stolen=stolen or None, reclaimed_orphan=reclaimed)
        if own_prev:
            payload["history"] = list(own_prev.get("history", [])) + [{"state": own_prev.get("state"), "until_utc": utc_now()}]
            atomic_write_json(state.own_claim_path(shard), payload)
        elif not create_exclusive_json(state.own_claim_path(shard), payload):
            continue  # another local slot won the O_EXCL race this instant
        return {"action": "CLAIMED", "shard_id": shard, "stolen": stolen, "claim": payload}
    # ALL_DONE when every shard of the plan is terminal somewhere visible: a terminal output,
    # or a relayed COMPLETED/FAILED claim (workers receive claims, not the results themselves)
    all_terminal = all((shard_blocked(state, s, None, now) or {}).get("reason") in ("TERMINAL", "TERMINAL_CLAIM") for s in order)
    return {"action": "ALL_DONE" if all_terminal else "NONE"}


def settle(state: State, shard: str, min_cycle_advance: int = 3, stale_after_seconds: Optional[int] = None,
           now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    claims = state.all_claims(shard)
    own = claims.get(state.role)
    if not own:
        return {"action": "LOSE", "reason": "OWN_CLAIM_MISSING"}
    home = own.get("home_role") or state.role
    if home == state.role:
        return {"action": "WIN", "home": True}
    cycle = state.heartbeat_cycle()
    at_claim = own.get("heartbeat_cycle_at_claim")
    advanced = None if cycle is None or not isinstance(at_claim, int) else cycle - at_claim
    if advanced is None or advanced < min_cycle_advance:
        return {"action": "WAIT", "cycles_advanced": advanced, "min_cycle_advance": min_cycle_advance}
    for role in list(claims):
        if role != state.role and is_stale(claims[role], stale_after_seconds, now):
            claims[role] = dict(claims[role], state="ABANDONED")
    winner = arbitrate(claims, home)
    if winner == state.role:
        return {"action": "WIN", "home": False, "competitors": sorted(r for r in claims if r != state.role)}
    mark(state, shard, "ABANDONED", lost_to=winner)
    return {"action": "LOSE", "winner": winner}


def mark(state: State, shard: str, new_state: str, **extra: Any) -> Dict[str, Any]:
    path = state.own_claim_path(shard)
    payload = read_json(path)
    if not payload:
        raise FileNotFoundError(f"no own claim for {shard}")
    payload.setdefault("history", []).append({"state": payload.get("state"), "until_utc": utc_now()})
    payload["state"] = new_state
    payload[f"{new_state.lower()}_at_utc"] = utc_now()
    if new_state == "COMPLETED":
        digest = valid_terminal(state.terminals / shard)
        if not digest:
            raise ValueError(f"terminal for {shard} is not valid; refusing to mark COMPLETED")
        payload["results_sha256"] = digest
    payload.update({k: v for k, v in extra.items() if v is not None})
    atomic_write_json(path, payload)
    return payload


def ledger(plan: Dict[str, Any], assignment: Dict[str, Any], claim_roots: Dict[str, Path], terminal_roots: Dict[str, Path]) -> Dict[str, Any]:
    """Coordinator view over relayed roots: per shard, home role, claims by role, terminal."""
    shards: Dict[str, Dict[str, Any]] = {}
    for sid in plan_order(plan):
        cell: Dict[str, Any] = {"home_role": home_role_of(assignment, sid), "claims": {}, "terminal": None}
        for root in claim_roots.values():
            d = Path(root) / sid
            for path in (sorted(d.glob("claim.*.json")) if d.is_dir() else []):
                c = read_json(path)
                if c and c.get("host_role") in ROLES:
                    prev = cell["claims"].get(c["host_role"])
                    if prev is None or len(c.get("history", [])) >= len(prev.get("history", [])):
                        cell["claims"][c["host_role"]] = c
        cell["terminal"] = terminal_state(sid, terminal_roots.values())
        shards[sid] = cell
    counts: Dict[str, int] = {}
    contested: List[str] = []
    for sid, cell in shards.items():
        if cell["terminal"]:
            key = "terminal:" + cell["terminal"]["kind"]
        else:
            live = [r for r, c in cell["claims"].items() if c.get("state") in LIVE_STATES]
            key = "live" if live else "free"
            if len(live) > 1:
                contested.append(sid)
        counts[key] = counts.get(key, 0) + 1
    return {"schema": "fs_phase23_claims_ledger.v1", "generated_at_utc": utc_now(), "counts": counts, "contested": contested, "shards": shards}


# ------------------------------------------------------------------------------------ CLI


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--state-dir", required=True, type=Path)
    p.add_argument("--role", required=True, choices=ROLES)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("next", help="claim the next free shard; prints its id; exit 0 claimed, 10 none now, 13 all done")
    _common(p)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--assignment", required=True, type=Path)
    p.add_argument("--slot", required=True)
    p.add_argument("--steal", action="store_true")
    p.add_argument("--stale-after-seconds", type=int)
    p.add_argument("--extra", help="JSON object merged into the claim (code_commit, cap, ...)")
    p = sub.add_parser("settle", help="exit 0 win, 11 lose (ABANDONED), 12 wait")
    _common(p)
    p.add_argument("--shard", required=True)
    p.add_argument("--min-cycle-advance", type=int, default=3)
    p.add_argument("--stale-after-seconds", type=int)
    p = sub.add_parser("mark")
    _common(p)
    p.add_argument("--shard", required=True)
    p.add_argument("--state", required=True, choices=["RUNNING", "COMPLETED", "FAILED", "ABANDONED"])
    p.add_argument("--extra", help="JSON object merged into the claim (exit_code, reason, wall_seconds)")
    p = sub.add_parser("ledger")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--assignment", required=True, type=Path)
    p.add_argument("--claims", action="append", default=[], help="source=path")
    p.add_argument("--terminals", action="append", default=[], help="source=path")
    p.add_argument("--out", type=Path)
    a = parser.parse_args(argv)

    if a.command == "ledger":
        doc = ledger(read_json(a.plan) or {}, read_json(a.assignment) or {"roles": {}},
                     dict(x.split("=", 1) for x in a.claims), dict(x.split("=", 1) for x in a.terminals))
        if a.out:
            atomic_write_json(a.out, doc)
        else:
            sys.stdout.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return 0 if not doc["contested"] else 1

    state = State(a.state_dir, a.role)
    if a.command == "next":
        plan, assignment = read_json(a.plan), read_json(a.assignment)
        if not plan or not assignment:
            print("plan or assignment unreadable", file=sys.stderr)
            return EXIT_ERROR
        extra = json.loads(a.extra) if a.extra else None
        r = next_shard(state, plan, assignment, a.slot, a.steal, a.stale_after_seconds, extra)
        if r["action"] == "CLAIMED":
            print(r["shard_id"])
            return EXIT_OK
        print(r["action"], file=sys.stderr)
        return EXIT_ALL_DONE if r["action"] == "ALL_DONE" else EXIT_NONE
    if a.command == "settle":
        r = settle(state, a.shard, a.min_cycle_advance, a.stale_after_seconds)
        print(json.dumps(r), file=sys.stderr)
        return {"WIN": EXIT_OK, "LOSE": EXIT_LOSE, "WAIT": EXIT_WAIT}[r["action"]]
    if a.command == "mark":
        extra = json.loads(a.extra) if a.extra else {}
        try:
            mark(state, a.shard, a.state, **extra)
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
