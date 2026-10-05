#!/usr/bin/env python3
"""Atomic shared claims for PS3-R batch_002 baseline cells (lane FS-GPU, order §3.3).

Two GPU workers (roles ``worker_a`` = RTX 5090 host, ``worker_b`` = RTX 4090 host) share
one queue. They cannot reach each other; the coordinator relays each worker's claims
directory to the other within its sync cadence (60 s + rsync time). Every claim is a
small JSON file named after the host ROLE, so relaying never overwrites the peer's file:

    <claims_root>/<batch>/<feature>/claim.<role>.json

Protocol (one cell, one worker):

1. SKIP if the cell is terminal anywhere visible (own output valid, peer terminal relayed,
   or a FAILED marker) -- a valid terminal is never repeated, a failure is never retried
   blindly.
2. SKIP if a peer claim is visible and not ABANDONED.
3. Otherwise write own claim with state CLAIMED and ``claimed_at_utc`` (UTC seconds).
4. Settle: wait until the relay heartbeat (pushed to every worker at the end of each
   coordinator cycle) shows at least THREE cycles after the one visible at claim time.
   Cycle order is pull a, pull b, push to a, push to b, heartbeat; so after three more
   cycles every claim the peer wrote before it could see ours has reached us. Then
   re-read: the batch's HOME role (``worker_b`` owns batch_002) wins any conflict; between
   non-home claims the earliest ``claimed_at_utc`` wins. The home role never waits: a
   stealer's claim it cannot yet see is still inside the stealer's gate, and the stealer
   will see the home claim and abandon. The loser marks its claim ABANDONED and moves on
   BEFORE any training starts. Nobody is ever killed. If the heartbeat never advances
   (relay down) the stealer abandons on timeout and cannot claim again until it is fresh.
   A live peer claim older than ``stale_after_seconds`` with no terminal anywhere is a
   crashed worker's orphan and may be re-claimed (reported, never silent).
5. The winner marks RUNNING, then COMPLETED (with results_sha256) or FAILED.

A ``claim.json`` without ``host_role`` is a legacy artifact of lanes C/E and is reported
but ignored by arbitration (its cells are all terminal). Only the standard library is
used so the module runs unchanged on the workers' system python.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = "ps3r_shared_claim.v1"
ROLES = ("worker_a", "worker_b")
DEFAULT_HOME_ROLE = "worker_b"
LIVE_STATES = ("CLAIMED", "RUNNING")
TERMINAL_STATES = ("COMPLETED", "FAILED")
FAILED_MARKERS = ("FAILED.codex.json", "FAILED.json")

EXIT_CLAIMED = 0
EXIT_WIN = 0
EXIT_SKIP = 10
EXIT_LOSE = 11
EXIT_WAIT = 12
EXIT_ERROR = 2


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> _dt.datetime:
    return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)


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


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------- terminals


def valid_terminal(directory: Path) -> Optional[str]:
    """Return results_sha256 when ``directory`` holds a COMPLETED manifest whose digest
    matches results.jsonl (the same rule every driver's ``valid_terminal`` applies)."""

    manifest = read_json(directory / "run_manifest.json")
    if not manifest or manifest.get("status") != "COMPLETED":
        return None
    expected = manifest.get("results_sha256") or manifest.get("results_digest")
    try:
        actual = hashlib.sha256((directory / "results.jsonl").read_bytes()).hexdigest()
    except OSError:
        return None
    return actual if isinstance(expected, str) and expected.lower() == actual else None


def failed_marker(directory: Path) -> Optional[Path]:
    for name in FAILED_MARKERS:
        if (directory / name).is_file():
            return directory / name
    return None


def terminal_state(directories: Iterable[Path]) -> Optional[Dict[str, Any]]:
    """First terminal evidence among candidate directories (own output, peer mirrors)."""

    for directory in directories:
        digest = valid_terminal(directory)
        if digest:
            return {"kind": "COMPLETED", "path": str(directory), "results_sha256": digest}
        marker = failed_marker(directory)
        if marker:
            return {"kind": "FAILED", "path": str(marker)}
    return None


# ------------------------------------------------------------------------------ claims


def claim_dir(root: Path, batch: str, feature: str) -> Path:
    return root / batch / feature


def claim_path(root: Path, batch: str, feature: str, role: str) -> Path:
    return claim_dir(root, batch, feature) / f"claim.{role}.json"


def load_claims(root: Path, batch: str, feature: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Return ``{role: claim}`` for role-named claims plus a list of legacy files."""

    claims: Dict[str, Dict[str, Any]] = {}
    legacy: List[str] = []
    directory = claim_dir(root, batch, feature)
    if not directory.is_dir():
        return claims, legacy
    for path in sorted(directory.glob("claim*.json")):
        payload = read_json(path)
        if not payload:
            continue
        role = payload.get("host_role")
        if role in ROLES and path.name == f"claim.{role}.json":
            claims[role] = payload
        else:
            legacy.append(str(path))
    return claims, legacy


def arbitrate(claims: Dict[str, Dict[str, Any]], home_role: str = DEFAULT_HOME_ROLE) -> Optional[str]:
    """Pure race rule. The home role wins whenever it holds a non-ABANDONED claim; among
    the other roles the earliest ``claimed_at_utc`` wins; an unparsable timestamp never
    wins."""

    home = claims.get(home_role)
    if home and home.get("state") != "ABANDONED":
        return home_role
    contenders: List[Tuple[_dt.datetime, str]] = []
    for role, claim in claims.items():
        if claim.get("state") == "ABANDONED":
            continue
        try:
            when = parse_utc(str(claim.get("claimed_at_utc")))
        except (TypeError, ValueError):
            continue
        contenders.append((when, role))
    if not contenders:
        return None
    contenders.sort()
    return contenders[0][1]


def new_claim(role: str, batch: str, feature: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "stage": "PS3-R",
        "batch_id": batch,
        "feature_id": feature,
        "host_role": role,
        "state": "CLAIMED",
        "claimed_at_utc": utc_now(),
        "pid": os.getpid(),
        "nonce": hashlib.sha256(f"{role}{batch}{feature}{os.getpid()}{socket.gethostname()}{utc_now()}".encode()).hexdigest()[:16],
        "seed": 0,
        "history": [],
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def decide(
    claims_root: Path,
    batch: str,
    feature: str,
    role: str,
    own_out_dir: Path,
    peer_terminal_dirs: Iterable[Path] = (),
    heartbeat: Optional[Path] = None,
    max_heartbeat_age: Optional[int] = None,
    home_role: str = DEFAULT_HOME_ROLE,
    extra: Optional[Dict[str, Any]] = None,
    stale_after_seconds: Optional[int] = None,
    now: Optional[_dt.datetime] = None,
) -> Dict[str, Any]:
    """Step 1-3. Writes a claim only when the cell is free; never blocks."""

    now = now or _dt.datetime.now(_dt.timezone.utc)
    terminal = terminal_state([own_out_dir, *peer_terminal_dirs])
    if terminal:
        return {"action": "SKIP_TERMINAL", "terminal": terminal}
    claims, legacy = load_claims(claims_root, batch, feature)
    stale_peers: List[str] = []
    for other, claim in claims.items():
        if other == role or claim.get("state") == "ABANDONED":
            continue
        if claim.get("state") in TERMINAL_STATES:
            return {"action": "SKIP_TERMINAL", "terminal": {"kind": claim["state"], "path": str(claim_path(claims_root, batch, feature, other)), "results_sha256": claim.get("results_sha256")}}
        if is_stale(claim, stale_after_seconds, now):
            stale_peers.append(other)
            continue
        return {"action": "SKIP_PEER_CLAIM", "peer": other, "peer_state": claim.get("state"), "peer_claimed_at_utc": claim.get("claimed_at_utc"), "legacy": legacy}
    own = claims.get(role)
    if own and own.get("state") in TERMINAL_STATES:
        return {"action": "SKIP_TERMINAL", "terminal": {"kind": own["state"], "path": str(claim_path(claims_root, batch, feature, role))}}
    beat = read_heartbeat(heartbeat)
    if max_heartbeat_age is not None:
        age = heartbeat_age_seconds(heartbeat, now)
        if age is None or age > max_heartbeat_age:
            return {"action": "SKIP_RELAY_STALE", "heartbeat_age_seconds": age, "max_heartbeat_age": max_heartbeat_age}
    payload = new_claim(role, batch, feature, **(extra or {}))
    payload["heartbeat_cycle_at_claim"] = beat.get("cycle") if beat else None
    if stale_peers:
        payload["reclaimed_from_stale"] = sorted(stale_peers)
    atomic_write_json(claim_path(claims_root, batch, feature, role), payload)
    return {"action": "CLAIMED", "claim": payload, "legacy": legacy, "heartbeat_cycle": payload["heartbeat_cycle_at_claim"], "reclaimed_from_stale": sorted(stale_peers)}


def is_stale(claim: Dict[str, Any], stale_after_seconds: Optional[int], now: _dt.datetime) -> bool:
    if stale_after_seconds is None or claim.get("state") not in LIVE_STATES:
        return False
    try:
        when = parse_utc(str(claim.get("claimed_at_utc")))
    except (TypeError, ValueError):
        return False
    return (now - when).total_seconds() > stale_after_seconds


def read_heartbeat(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    return read_json(path) if path is not None else None


def cycles_advanced(heartbeat: Optional[Path], claim_cycle: Optional[int]) -> Optional[int]:
    beat = read_heartbeat(heartbeat)
    if not beat or claim_cycle is None or not isinstance(beat.get("cycle"), int):
        return None
    return beat["cycle"] - claim_cycle


def settle(
    claims_root: Path,
    batch: str,
    feature: str,
    role: str,
    home_role: str = DEFAULT_HOME_ROLE,
    heartbeat: Optional[Path] = None,
    min_cycle_advance: int = 3,
    stale_after_seconds: Optional[int] = None,
    now: Optional[_dt.datetime] = None,
) -> Dict[str, Any]:
    """Step 4. A non-home role WAITs until the relay has run ``min_cycle_advance`` cycles
    past the cycle seen at claim time (when a heartbeat is configured), then arbitrates.
    The home role never waits. The loser abandons before training. A peer claim that is
    stale is not a contender."""

    now = now or _dt.datetime.now(_dt.timezone.utc)
    claims, _ = load_claims(claims_root, batch, feature)
    own = claims.get(role)
    if not own:
        return {"action": "LOSE", "reason": "OWN_CLAIM_MISSING"}
    if heartbeat is not None and role != home_role:
        advanced = cycles_advanced(heartbeat, own.get("heartbeat_cycle_at_claim"))
        if advanced is None or advanced < min_cycle_advance:
            return {"action": "WAIT", "cycles_advanced": advanced, "min_cycle_advance": min_cycle_advance}
    for other in [r for r in claims if r != role]:
        if is_stale(claims[other], stale_after_seconds, now):
            claims[other] = dict(claims[other], state="ABANDONED")
    winner = arbitrate(claims, home_role)
    if winner == role:
        return {"action": "WIN", "winner": winner, "competitors": sorted(r for r in claims if r != role)}
    mark(claims_root, batch, feature, role, "ABANDONED", lost_to=winner, peer_claimed_at_utc=(claims.get(winner) or {}).get("claimed_at_utc"))
    return {"action": "LOSE", "winner": winner, "peer_claimed_at_utc": (claims.get(winner) or {}).get("claimed_at_utc")}


def mark(claims_root: Path, batch: str, feature: str, role: str, state: str, **extra: Any) -> Dict[str, Any]:
    path = claim_path(claims_root, batch, feature, role)
    payload = read_json(path) or new_claim(role, batch, feature)
    history = payload.setdefault("history", [])
    history.append({"state": payload.get("state"), "until_utc": utc_now()})
    payload["state"] = state
    payload[f"{state.lower()}_at_utc"] = utc_now()
    payload.update({k: v for k, v in extra.items() if v is not None})
    atomic_write_json(path, payload)
    return payload


def heartbeat_age_seconds(path: Optional[Path], now: Optional[_dt.datetime] = None) -> Optional[int]:
    payload = read_heartbeat(path)
    if not payload:
        return None
    try:
        then = parse_utc(str(payload.get("relayed_at_utc")))
    except (TypeError, ValueError):
        return None
    return int(((now or _dt.datetime.now(_dt.timezone.utc)) - then).total_seconds())


def ledger(claims_roots: Dict[str, Path]) -> Dict[str, Any]:
    """Coordinator view: every claim from every relayed root, merged by (batch, feature)."""

    cells: Dict[str, Dict[str, Any]] = {}
    for source, root in claims_roots.items():
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*/claim*.json")):
            payload = read_json(path)
            if not payload:
                continue
            key = f"{path.parent.parent.name}::{path.parent.name}"
            cell = cells.setdefault(key, {"batch_id": path.parent.parent.name, "feature_id": path.parent.name, "claims": {}, "legacy": []})
            role = payload.get("host_role")
            if role in ROLES:
                prior = cell["claims"].get(role)
                # the same role's claim may be visible in several roots; keep the newest state
                if prior is None or json.dumps(payload, sort_keys=True) != json.dumps(prior, sort_keys=True):
                    if prior is None or len(payload.get("history", [])) >= len(prior.get("history", [])):
                        cell["claims"][role] = payload
            else:
                cell["legacy"].append(f"{source}:{path}")
    duplicates = []
    for key, cell in cells.items():
        live = [r for r, c in cell["claims"].items() if c.get("state") in LIVE_STATES]
        done = [r for r, c in cell["claims"].items() if c.get("state") == "COMPLETED"]
        cell["winner"] = arbitrate(cell["claims"])
        if len(live) > 1 or len(done) > 1 or (live and done):
            duplicates.append(key)
    return {
        "schema": "ps3r_claims_ledger.v1",
        "generated_at_utc": utc_now(),
        "cells": dict(sorted(cells.items())),
        "counts": {
            "cells": len(cells),
            "by_state": _count_states(cells),
            "contested": len(duplicates),
        },
        "contested_cells": duplicates,
    }


def _count_states(cells: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cell in cells.values():
        for role, claim in cell["claims"].items():
            key = f"{role}:{claim.get('state')}"
            counts[key] = counts.get(key, 0) + 1
    return counts


# --------------------------------------------------------------------------------- CLI


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--claims-root", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--home-role", default=DEFAULT_HOME_ROLE, choices=ROLES)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("decide", help="skip or write a claim (exit 0 claimed, 10 skip)")
    _common(p)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--peer-terminal-dir", action="append", default=[], type=Path)
    p.add_argument("--heartbeat", type=Path)
    p.add_argument("--max-heartbeat-age", type=int)
    p.add_argument("--extra", help="JSON object merged into the claim (code_commit, input_digest, families, cap)")
    p.add_argument("--stale-after-seconds", type=int, help="a live peer claim older than this with no terminal is reclaimable")

    p = sub.add_parser("settle", help="exit 0 win, 11 lose (claim ABANDONED), 12 wait for more relay cycles")
    _common(p)
    p.add_argument("--heartbeat", type=Path)
    p.add_argument("--min-cycle-advance", type=int, default=3)
    p.add_argument("--stale-after-seconds", type=int)

    p = sub.add_parser("mark", help="transition own claim state")
    _common(p)
    p.add_argument("--state", required=True, choices=["RUNNING", "COMPLETED", "FAILED", "ABANDONED"])
    p.add_argument("--results-dir", type=Path, help="read results_sha256 and wall_seconds from this terminal")
    p.add_argument("--extra", help="JSON object merged into the claim")

    p = sub.add_parser("ledger", help="merge claims from several roots into one JSON ledger")
    p.add_argument("--root", action="append", required=True, help="source=path")
    p.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "decide":
            extra = json.loads(args.extra) if args.extra else None
            result = decide(args.claims_root, args.batch, args.feature, args.role, args.out_dir, args.peer_terminal_dir, args.heartbeat, args.max_heartbeat_age, args.home_role, extra, args.stale_after_seconds)
            print(json.dumps(result, sort_keys=True))
            return EXIT_CLAIMED if result["action"] == "CLAIMED" else EXIT_SKIP
        if args.command == "settle":
            result = settle(args.claims_root, args.batch, args.feature, args.role, args.home_role, args.heartbeat, args.min_cycle_advance, args.stale_after_seconds)
            print(json.dumps(result, sort_keys=True))
            return {"WIN": EXIT_WIN, "WAIT": EXIT_WAIT}.get(result["action"], EXIT_LOSE)
        if args.command == "mark":
            extra = json.loads(args.extra) if args.extra else {}
            if args.results_dir is not None:
                digest = valid_terminal(args.results_dir)
                manifest = read_json(args.results_dir / "run_manifest.json") or {}
                extra.update({"results_sha256": digest, "wall_seconds": manifest.get("wall_seconds"), "cgroup_peak_bytes": manifest.get("cgroup_peak_bytes"), "results_dir": str(args.results_dir)})
            print(json.dumps(mark(args.claims_root, args.batch, args.feature, args.role, args.state, **extra), sort_keys=True))
            return 0
        if args.command == "ledger":
            roots = {}
            for item in args.root:
                source, _, path = item.partition("=")
                roots[source or path] = Path(path)
            result = ledger(roots)
            if args.out:
                atomic_write_json(args.out, result)
            else:
                print(json.dumps(result, indent=2, sort_keys=True))
            return 0
    except OSError as error:
        print(f"ps3r_claims: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
