"""Finite, resumable CLI for the I6-N C1 support table (candidate, not deployed).

Usage (from the repository root)::

    python -m tools.i6n_contracts.eligibility_cli --config CONFIG.json --output DIR
    python -m tools.i6n_contracts.eligibility_cli --status --output DIR

``DIR`` receives ``SUPPORT_TABLE.json`` and an atomically replaced
``STATUS.json`` (schema ``i6n.eligibility.cli_status.v1``).

Rules:

* ``--status`` reads ``STATUS.json`` once, checks the result digest read-only,
  prints it and exits. It never creates, starts or resumes anything.
* Identity = canonical config digest + code digest. An identical identity with
  an intact COMPLETED result is reused (exit 0, ``REUSED``). A FAILED or
  interrupted RUNNING record of the same identity is recomputed once, from
  scratch (the computation is pure and deterministic).
* A different identity never overwrites the directory: typed refusal, exit 2.
  A COMPLETED record whose result bytes no longer match is refused as well.
* Single writer per directory by contract; there are no leases or workers.

Exit codes: 0 completed/reused/status, 1 typed contract failure (FAILED
recorded), 2 refusal or usage error (nothing written).

Standard library only, and no database module: the canonical JSON digest
(``eligibility.canonical_digest``) and the atomic writer (:func:`atomic_json`:
unique temp file in the same directory, fsync, ``os.replace``) follow the
formulas of ``tools.i6m_queue.digest``/``atomic_json`` but are implemented
locally, because importing ``tools.i6m_queue`` would import ``sqlite3``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.i6n_contracts import eligibility as elig

digest = elig.canonical_digest

STATUS_SCHEMA = "i6n.eligibility.cli_status.v1"
STATUS_NAME = "STATUS.json"
RESULT_NAME = "SUPPORT_TABLE.json"
COMPONENT = "I6N_C1_ELIGIBILITY_SUPPORT_TABLE"
_HERE = Path(__file__).resolve().parent
_CODE_FILES = (
    ("tools/i6n_contracts/eligibility.py", _HERE / "eligibility.py"),
    ("tools/i6n_contracts/eligibility_cli.py", _HERE / "eligibility_cli.py"),
    ("tools/business_weekly_protocol.py", _HERE.parent / "business_weekly_protocol.py"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    """Replace ``path`` atomically: unique temp beside it, fsync, ``os.replace``.

    A reader sees either the previous complete file or the new complete file;
    a failure before the replace leaves the previous file and removes the temp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_sha256() -> str:
    """Digest of the exact code that computes the table (relative names only)."""
    h = hashlib.sha256()
    for name, path in _CODE_FILES:
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _emit(kind: str, payload: Dict[str, Any]) -> None:
    print(json.dumps({"event": kind, **payload}, sort_keys=True))


def _read_status(out: Path) -> Optional[Dict[str, Any]]:
    path = out / STATUS_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _result_intact(out: Path, status: Dict[str, Any]) -> Optional[bool]:
    result = status.get("result")
    if not result:
        return None
    path = out / result["path"]
    return path.is_file() and _sha256_bytes(path.read_bytes()) == result["sha256"]


def status_command(out: Path) -> int:
    """Print the persisted status once. Read-only; never starts work."""
    try:
        status = _read_status(out)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema_version": STATUS_SCHEMA, "state": "UNKNOWN",
                          "error": {"code": "STATUS_UNREADABLE", "message": str(exc)}}))
        return 2
    if status is None:
        print(json.dumps({"schema_version": STATUS_SCHEMA, "state": "NOT_STARTED",
                          "error": {"code": "STATUS_ABSENT",
                                    "message": "no STATUS.json in output directory"}}))
        return 0
    print(json.dumps(dict(status, result_intact=_result_intact(out, status)), sort_keys=True))
    return 0


class _Status:
    """Mutable run record persisted atomically after every completed week."""

    def __init__(self, out: Path, identity: Dict[str, Optional[str]]):
        self.out = out
        self.t0 = time.monotonic()
        self.record: Dict[str, Any] = {
            "schema_version": STATUS_SCHEMA, "component": COMPONENT, "state": "RUNNING",
            "units_done": 0, "units_total": None, "started_at": _now(),
            "last_progress_at": None, "duration_seconds": 0.0, "eta_seconds": None,
            "eta_unavailable_reason": "NO_COMPLETED_UNITS", **identity,
            "result": None, "error": None, "candidate_implementation": True,
            "deployed": False, "scientific_measurement": False}

    def write(self, **changes: Any) -> None:
        self.record.update(changes)
        elapsed = time.monotonic() - self.t0
        self.record["duration_seconds"] = round(elapsed, 6)
        state, done, total = (self.record["state"], self.record["units_done"],
                              self.record["units_total"])
        if state == "COMPLETED":
            self.record.update(eta_seconds=0, eta_unavailable_reason=None)
        elif state == "FAILED":
            self.record.update(eta_seconds=None, eta_unavailable_reason="TERMINAL_FAILED")
        elif not total:
            self.record.update(eta_seconds=None, eta_unavailable_reason="UNITS_NOT_YET_PLANNED")
        elif done == 0:
            self.record.update(eta_seconds=None, eta_unavailable_reason="NO_COMPLETED_UNITS")
        else:
            self.record.update(eta_seconds=round(elapsed / done * (total - done), 6),
                               eta_unavailable_reason=None)
        atomic_json(self.out / STATUS_NAME, self.record)


def run_command(config_path: Path, out: Path) -> int:
    """Compute (or reuse) the support table for ``config_path`` into ``out``."""
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        _emit("REFUSED", {"error": {"code": "CONFIG_UNREADABLE", "message": str(exc)}})
        return 2
    try:
        config = json.loads(raw)
        config_sha = digest(config)
    except ValueError:
        config, config_sha = None, None
    identity = {"input_sha256": _sha256_bytes(raw), "config_sha256": config_sha,
                "code_sha256": code_sha256()}
    identity["identity_sha256"] = digest(
        {"config": config_sha, "input": None if config_sha else identity["input_sha256"],
         "code": identity["code_sha256"]})
    try:
        previous = _read_status(out)
    except (OSError, ValueError) as exc:
        _emit("REFUSED", {"error": {"code": "STATUS_UNREADABLE", "message": str(exc)}})
        return 2
    if previous is not None:
        if previous.get("identity_sha256") != identity["identity_sha256"]:
            code = ("OUTPUT_HOLDS_DIFFERENT_CODE"
                    if previous.get("config_sha256") == config_sha and config_sha
                    else "OUTPUT_HOLDS_DIFFERENT_INPUT")
            _emit("REFUSED", {"error": {"code": code, "message":
                                        "output directory belongs to another identity"},
                              "existing_identity_sha256": previous.get("identity_sha256")})
            return 2
        if previous.get("state") == "COMPLETED":
            if _result_intact(out, previous):
                _emit("REUSED", {"status": previous})
                return 0
            _emit("REFUSED", {"error": {"code": "TERMINAL_OUTPUT_DAMAGED", "message":
                                        "completed result does not match its digest"}})
            return 2
    status = _Status(out, identity)
    status.write()
    try:
        if config is None:
            raise elig.EligibilityError(elig.ErrorCode.CONFIG_INVALID, "config is not valid JSON")
        plan = elig.prepare_support_plan(config)
        status.write(units_total=plan.units_total)
        rows: List[Dict[str, Any]] = []
        per_week = len(plan.roles) * len(plan.horizons)
        for row in elig.iter_support_rows(plan):
            rows.append(row)
            if len(rows) % per_week == 0:
                status.write(units_done=len(rows), last_progress_at=_now())
        table = elig.assemble_support_table(plan, rows)
        table["config_sha256"] = config_sha
        table["code_sha256"] = identity["code_sha256"]
    except elig.EligibilityError as exc:
        status.write(state="FAILED", error=exc.to_json())
        _emit("FAILED", {"status": status.record})
        return 1
    atomic_json(out / RESULT_NAME, table)
    result_sha = _sha256_bytes((out / RESULT_NAME).read_bytes())
    status.write(state="COMPLETED", last_progress_at=_now(),
                 result={"path": RESULT_NAME, "sha256": result_sha, "rows": len(rows)})
    _emit("COMPLETED", {"status": status.record})
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; see module docstring for the contract."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="support-table config JSON")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--status", action="store_true", help="print status once (read-only)")
    args = parser.parse_args(argv)
    if args.status:
        return status_command(args.output)
    if args.config is None:
        _emit("REFUSED", {"error": {"code": "USAGE", "message": "--config is required"}})
        return 2
    return run_command(args.config, args.output)


if __name__ == "__main__":
    sys.exit(main())
