"""Finite CLI for the I6-N C1 support table (candidate, not deployed).

Usage (from the repository root)::

    python -m tools.i6n_contracts.eligibility_cli --config CONFIG.json --output DIR [--recover]
    python -m tools.i6n_contracts.eligibility_cli --status --output DIR

``DIR`` holds two files, and each is verified ON ITS OWN:

* ``SUPPORT_TABLE.json`` is the result. It carries its own identity
  (``config_sha256``, ``code_sha256``, ``identity_sha256``) and a self digest
  ``table_sha256`` (canonical digest of the table without that field). Its
  identity and integrity can therefore be checked when ``STATUS.json`` is
  missing or damaged. The file is CREATED and never replaced: a complete,
  fsynced temp file is published with ``os.link``, which fails if the name
  already exists.
* ``STATUS.json`` (schema ``i6n.eligibility.cli_status.v2``) is the progress
  and terminal record. It is replaced atomically (temp, fsync, ``os.replace``).

Run rules. Identity = canonical config digest + code digest.

* A status or a result that belongs to ANOTHER identity, or a damaged result,
  is refused (exit 2). Nothing is written, with or without ``--recover``.
* A COMPLETED status plus an intact result of the same identity is ``REUSED``
  (exit 0).
* An intact result of the same identity that no matching COMPLETED status
  backs is never adopted just because it exists. This covers a status that is
  missing, damaged, FAILED, INTERRUPTED or stale RUNNING, and a status digest
  that does not match the bytes. Without ``--recover`` the run is refused
  (``RECOVERY_REQUIRED``, exit 2). With ``--recover`` the table is recomputed
  from scratch in memory. The existing file is accepted only if its bytes are
  identical (``RECOVERED``, exit 0; the result file is not touched). Otherwise
  the run is refused (``RECOVERY_MISMATCH``, exit 2) and nothing is written.
* With no result, a FAILED, INTERRUPTED or stale RUNNING record of the same
  identity is recomputed once from scratch. A damaged status, or a COMPLETED
  status whose result is missing, needs ``--recover``.
* A RUNNING record whose writer process is alive is refused
  (``RUN_IN_PROGRESS``). So is one whose liveness cannot be verified.

Failures:

* A contract error (``EligibilityError``) is recorded as a typed FAILED, exit 1.
* Any other exception, including one raised while the result is written, is
  recorded as FAILED with code ``UNEXPECTED_ERROR`` and re-raised by
  :func:`run_command`. ``main`` maps it to exit 1.
* ``KeyboardInterrupt`` and ``SystemExit`` (and SIGTERM, under ``main``) are
  recorded as INTERRUPTED and propagated.
* If the status itself cannot be written, nothing durable is claimed: the
  event ``STATUS_NOT_RECORDED`` (``durable: false``) is printed and the exit
  code is 3.

``--status`` is read-only and returns once. It verifies the status, the
result, their identities and digests, then prints a single ``disposition``
with its exit code::

    COMPLETE_INTACT 0   ABSENT 10   RUNNING 11   INCOMPLETE 12
    RUNNING_UNVERIFIED 13   FAILED 14   INTERRUPTED 15   DAMAGED 16
    ORPHAN_RESULT 17

Zero means an intact, complete terminal result and nothing else. A persisted
RUNNING record is a claim, not evidence. Liveness is checked exactly once,
using the pid plus the process start time from ``/proc``. A writer killed by
SIGKILL or a power loss therefore reads as INCOMPLETE, and a reused pid is not
mistaken for the writer. ALIVE means the writer PROCESS still exists (same
host and pid namespace only), not that it is making progress. There is no
polling, lease or worker; there is one writer per directory, by contract.

Run exit codes: 0 completed/reused/recovered, 1 FAILED recorded, 2 refusal or
usage error (nothing written), 3 the status could not be recorded.

Standard library only, and no database module. The canonical JSON digest is
``eligibility.canonical_digest``; the writers are local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.i6n_contracts import eligibility as elig

digest = elig.canonical_digest

STATUS_SCHEMA = "i6n.eligibility.cli_status.v2"
STATUS_NAME = "STATUS.json"
RESULT_NAME = "SUPPORT_TABLE.json"
COMPONENT = "I6N_C1_ELIGIBILITY_SUPPORT_TABLE"
RESULT_IDENTITY_KEYS = ("config_sha256", "code_sha256", "identity_sha256")
STATUS_IDENTITY_KEYS = ("input_sha256",) + RESULT_IDENTITY_KEYS
SELF_DIGEST_KEY = "table_sha256"
STATES = ("RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")

STATUS_EXIT_CODES = {
    "COMPLETE_INTACT": 0, "ABSENT": 10, "RUNNING": 11, "INCOMPLETE": 12,
    "RUNNING_UNVERIFIED": 13, "FAILED": 14, "INTERRUPTED": 15, "DAMAGED": 16,
    "ORPHAN_RESULT": 17,
}
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_STATUS_NOT_RECORDED = 0, 1, 2, 3

_HERE = Path(__file__).resolve().parent
_CODE_FILES = (
    ("tools/i6n_contracts/eligibility.py", _HERE / "eligibility.py"),
    ("tools/i6n_contracts/eligibility_cli.py", _HERE / "eligibility_cli.py"),
    ("tools/business_weekly_protocol.py", _HERE.parent / "business_weekly_protocol.py"),
)


class StatusNotRecorded(RuntimeError):
    """The status file could not be written; no durable state is claimed."""


class TerminationRequested(KeyboardInterrupt):
    """SIGTERM received under :func:`main`; handled like an interrupt."""


class _CliError(Exception):
    """Typed CLI-level contract failure (recorded as FAILED, exit 1)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.detail = code, message

    def to_json(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.detail}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _temp_beside(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def _stage(tmp: Path, data: bytes) -> None:
    """Write ``data`` to the new file ``tmp`` and fsync it (caller removes ``tmp``)."""
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, value: Any) -> None:
    """Replace ``path`` atomically: unique temp beside it, fsync, ``os.replace``.

    A reader sees either the previous complete file or the new one. Any
    failure (serialization, write, fsync, replace) leaves the previous file
    unchanged and removes the temp file.
    """
    path = Path(path)
    data = _serialize(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_beside(path)
    try:
        _stage(tmp, data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def create_json_exclusive(path: Path, value: Any) -> bytes:
    """Create ``path`` from a complete fsynced temp file; never replace it.

    ``os.link`` publishes the temp file under ``path`` and fails if that name
    already exists. Returns the bytes written. The temp file is always removed.
    """
    path = Path(path)
    data = _serialize(value)
    tmp = _temp_beside(path)
    try:
        _stage(tmp, data)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise _CliError("RESULT_APPEARED_DURING_RUN",
                            f"{path.name} was created by someone else during this run")
    finally:
        if tmp.exists():
            tmp.unlink()
    return data


def code_sha256() -> str:
    """Digest of the exact code that computes the table (relative names only)."""
    h = hashlib.sha256()
    for name, path in _CODE_FILES:
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _identity_sha(config_sha: Optional[str], input_sha: Optional[str], code_sha: str) -> str:
    return digest({"config": config_sha, "input": None if config_sha else input_sha,
                   "code": code_sha})


def _is_hex(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _emit(kind: str, payload: Dict[str, Any]) -> None:
    print(json.dumps({"event": kind, **payload}, sort_keys=True))


# ------------------------------------------------------------ liveness


def _proc_stat(pid: int) -> Optional[Tuple[str, int]]:
    """(state letter, start time in clock ticks) of ``pid`` from /proc, or None."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text.rsplit(")", 1)[1].split()
        return fields[0], int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _writer() -> Dict[str, Any]:
    pid = os.getpid()
    stat = _proc_stat(pid)
    return {"pid": pid, "process_start_ticks": stat[1] if stat else None}


def writer_liveness(writer: Any) -> str:
    """``ALIVE``, ``NOT_FOUND`` or ``UNVERIFIABLE`` from one look; never polls."""
    if not isinstance(writer, dict):
        return "UNVERIFIABLE"
    pid, ticks = writer.get("pid"), writer.get("process_start_ticks")
    if type(pid) is not int or pid <= 0:
        return "UNVERIFIABLE"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "NOT_FOUND"
    except PermissionError:
        pass                                   # exists, owned by another user
    except OSError:
        return "UNVERIFIABLE"
    if type(ticks) is not int:
        return "UNVERIFIABLE"
    stat = _proc_stat(pid)
    if stat is None:
        return "UNVERIFIABLE"
    state, current = stat
    if current != ticks or state in ("Z", "X"):
        return "NOT_FOUND"                     # pid reused, or writer is a zombie
    return "ALIVE"


# ------------------------------------------------------- verification


def _problem(presence: str, problem: str, **extra: Any) -> Dict[str, Any]:
    return {"presence": presence, "problem": problem, **extra}


def inspect_result(out: Path) -> Dict[str, Any]:
    """Verify ``SUPPORT_TABLE.json`` on its own: parse, identity, self digest, rows.

    ``presence`` is ``ABSENT``, ``DAMAGED`` (with ``problem``) or ``INTACT``
    (with ``identity`` and ``file_sha256``). Read-only.
    """
    path = Path(out) / RESULT_NAME
    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return {"presence": "ABSENT", "problem": None}
    except OSError as exc:
        return _problem("DAMAGED", "RESULT_UNREADABLE", message=str(exc))
    base = {"file_sha256": _sha256_bytes(data)}
    try:
        table = json.loads(data)
    except ValueError:
        return _problem("DAMAGED", "RESULT_NOT_JSON", **base)
    if not isinstance(table, dict):
        return _problem("DAMAGED", "RESULT_NOT_OBJECT", **base)
    identity = {k: table.get(k) for k in RESULT_IDENTITY_KEYS}
    if not (all(_is_hex(v) for v in identity.values())
            and identity["identity_sha256"] == _identity_sha(
                identity["config_sha256"], None, identity["code_sha256"])):
        return _problem("DAMAGED", "RESULT_IDENTITY_INVALID", **base)
    body = {k: v for k, v in table.items() if k != SELF_DIGEST_KEY}
    try:
        sealed = table.get(SELF_DIGEST_KEY) == digest(body)
    except ValueError:
        sealed = False
    if not sealed:
        return _problem("DAMAGED", "RESULT_SELF_DIGEST_MISMATCH", **base)
    if not _rows_complete(table):
        return _problem("DAMAGED", "RESULT_INCOMPLETE", **base)
    return {"presence": "INTACT", "problem": None, "identity": identity, **base}


def _rows_complete(table: Dict[str, Any]) -> bool:
    rows, roles, horizons = table.get("rows"), table.get("roles"), table.get("horizons_hours")
    if not (isinstance(rows, list) and isinstance(roles, list) and isinstance(horizons, list)):
        return False
    try:
        keys = [(r["week_id"], r["role"], r["horizon_hours"]) for r in rows]
    except (TypeError, KeyError):
        return False
    weeks = {k[0] for k in keys}
    expected = {(w, r, h) for w in weeks for r in roles for h in horizons}
    return (len(keys) == len(set(keys)) == table.get("units_total") == len(expected)
            and set(keys) == expected)


def inspect_status(out: Path) -> Dict[str, Any]:
    """Verify ``STATUS.json`` on its own: schema, state, identity, result pin."""
    path = Path(out) / STATUS_NAME
    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return {"presence": "ABSENT", "problem": None, "record": None}
    except OSError as exc:
        return _problem("DAMAGED", "STATUS_UNREADABLE", record=None, message=str(exc))
    try:
        record = json.loads(data)
    except ValueError:
        return _problem("DAMAGED", "STATUS_NOT_JSON", record=None)
    if not isinstance(record, dict) or record.get("schema_version") != STATUS_SCHEMA:
        return _problem("DAMAGED", "STATUS_SCHEMA_UNKNOWN", record=None)
    if record.get("state") not in STATES:
        return _problem("DAMAGED", "STATUS_STATE_INVALID", record=None)
    config_sha, input_sha, code_sha = (record.get("config_sha256"), record.get("input_sha256"),
                                       record.get("code_sha256"))
    if not ((config_sha is None or _is_hex(config_sha)) and _is_hex(input_sha)
            and _is_hex(code_sha)
            and record.get("identity_sha256") == _identity_sha(config_sha, input_sha,
                                                               code_sha)):
        return _problem("DAMAGED", "STATUS_IDENTITY_INVALID", record=None)
    if record["state"] == "COMPLETED":
        result = record.get("result")
        if not (isinstance(result, dict) and result.get("path") == RESULT_NAME
                and _is_hex(result.get("sha256"))):
            return _problem("DAMAGED", "STATUS_RESULT_PIN_INVALID", record=None)
    return {"presence": "VALID", "problem": None, "record": record}


def assess(out: Path) -> Dict[str, Any]:
    """Combine the two independent checks into one disposition. Read-only."""
    st, rs = inspect_status(out), inspect_result(out)
    record = st["record"]
    liveness = None
    problem = None
    if st["presence"] == "DAMAGED":
        disposition, problem = "DAMAGED", st["problem"]
    elif rs["presence"] == "DAMAGED":
        disposition, problem = "DAMAGED", rs["problem"]
    elif record is None:
        disposition = "ABSENT" if rs["presence"] == "ABSENT" else "ORPHAN_RESULT"
    elif (rs["presence"] == "INTACT"
          and rs["identity"]["identity_sha256"] != record["identity_sha256"]):
        disposition, problem = "DAMAGED", "STATUS_RESULT_IDENTITY_CONFLICT"
    elif record["state"] == "COMPLETED":
        if rs["presence"] == "ABSENT":
            disposition, problem = "DAMAGED", "TERMINAL_OUTPUT_MISSING"
        elif rs["file_sha256"] != record["result"]["sha256"]:
            disposition, problem = "DAMAGED", "TERMINAL_OUTPUT_DAMAGED"
        else:
            disposition = "COMPLETE_INTACT"
    elif record["state"] in ("FAILED", "INTERRUPTED"):
        disposition = record["state"]
    else:
        liveness = writer_liveness(record.get("writer"))
        disposition = {"ALIVE": "RUNNING", "NOT_FOUND": "INCOMPLETE"}.get(
            liveness, "RUNNING_UNVERIFIED")
    return {"disposition": disposition, "exit_code": STATUS_EXIT_CODES[disposition],
            "problem": problem, "status": st, "result": rs, "writer_liveness": liveness}


def status_command(out: Path) -> int:
    """Print the verified disposition once and return its exit code.

    Read-only: never creates, starts, resumes or rewrites anything. Exit 0
    only for ``COMPLETE_INTACT``; see :data:`STATUS_EXIT_CODES`.
    """
    a = assess(Path(out))
    record = a["status"]["record"]
    rs = a["result"]
    try:
        current = code_sha256()
    except OSError:
        current = None
    identity = rs.get("identity")
    print(json.dumps({
        "schema_version": STATUS_SCHEMA, "read_only": True,
        "disposition": a["disposition"], "exit_code": a["exit_code"], "problem": a["problem"],
        "state": record["state"] if record else None,
        "result_intact": a["disposition"] == "COMPLETE_INTACT",
        "writer_liveness": a["writer_liveness"],
        "status_presence": a["status"]["presence"],
        "result": {"presence": rs["presence"], "problem": rs["problem"],
                   "file_sha256": rs.get("file_sha256"), "identity": identity},
        "code_matches_current": (identity["code_sha256"] == current
                                 if identity and current else None),
        "status": record,
    }, sort_keys=True))
    return a["exit_code"]


# ---------------------------------------------------------------- run


class _Status:
    """Mutable run record persisted atomically after every completed week."""

    def __init__(self, out: Path, identity: Dict[str, Optional[str]],
                 previous_disposition: str):
        self.out = out
        self.t0 = time.monotonic()
        self.record: Dict[str, Any] = {
            "schema_version": STATUS_SCHEMA, "component": COMPONENT, "state": "RUNNING",
            "units_done": 0, "units_total": None, "started_at": _now(),
            "last_progress_at": None, "duration_seconds": 0.0, "eta_seconds": None,
            "eta_unavailable_reason": "NO_COMPLETED_UNITS", **identity,
            "writer": _writer(), "previous_disposition": previous_disposition,
            "recovered": False, "result": None, "error": None,
            "candidate_implementation": True, "deployed": False,
            "scientific_measurement": False}

    def write(self, **changes: Any) -> None:
        self.record.update(changes)
        elapsed = time.monotonic() - self.t0
        self.record["duration_seconds"] = round(elapsed, 6)
        state, done, total = (self.record["state"], self.record["units_done"],
                              self.record["units_total"])
        if state == "COMPLETED":
            self.record.update(eta_seconds=0, eta_unavailable_reason=None)
        elif state in ("FAILED", "INTERRUPTED"):
            self.record.update(eta_seconds=None, eta_unavailable_reason=f"TERMINAL_{state}")
        elif not total:
            self.record.update(eta_seconds=None, eta_unavailable_reason="UNITS_NOT_YET_PLANNED")
        elif done == 0:
            self.record.update(eta_seconds=None, eta_unavailable_reason="NO_COMPLETED_UNITS")
        else:
            self.record.update(eta_seconds=round(elapsed / done * (total - done), 6),
                               eta_unavailable_reason=None)
        atomic_json(self.out / STATUS_NAME, self.record)


def _refuse(code: str, message: str, **extra: Any) -> int:
    _emit("REFUSED", {"error": {"code": code, "message": message}, **extra})
    return EXIT_REFUSED


def _not_recorded(state: str, error: Optional[Dict[str, Any]], exc: BaseException) -> None:
    _emit("STATUS_NOT_RECORDED", {
        "attempted_state": state, "durable": False, "error": error,
        "write_error": {"exception_type": type(exc).__name__, "message": str(exc)}})


def _terminal(status: _Status, state: str, error: Dict[str, Any]) -> bool:
    """Record a terminal state; False (and nothing claimed) if it cannot be written."""
    try:
        status.write(state=state, error=error)
    except OSError as exc:
        _not_recorded(state, error, exc)
        return False
    _emit(state, {"status": status.record})
    return True


def _compute(config: Any, identity: Dict[str, Optional[str]],
             status: Optional[_Status]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if config is None:
        raise elig.EligibilityError(elig.ErrorCode.CONFIG_INVALID, "config is not valid JSON")
    plan = elig.prepare_support_plan(config)
    if status is not None:
        status.write(units_total=plan.units_total)
    rows: List[Dict[str, Any]] = []
    per_week = len(plan.roles) * len(plan.horizons)
    for row in elig.iter_support_rows(plan):
        rows.append(row)
        if status is not None and len(rows) % per_week == 0:
            status.write(units_done=len(rows), last_progress_at=_now())
    table = elig.assemble_support_table(plan, rows)
    table.update({k: identity[k] for k in RESULT_IDENTITY_KEYS})
    table[SELF_DIGEST_KEY] = digest(table)
    return table, rows


def _identity_refusal(holder: Dict[str, Any], identity: Dict[str, Optional[str]],
                      where: str) -> int:
    same_config = (holder.get("config_sha256") == identity["config_sha256"]
                   and identity["config_sha256"])
    code = "OUTPUT_HOLDS_DIFFERENT_CODE" if same_config else "OUTPUT_HOLDS_DIFFERENT_INPUT"
    return _refuse(code, f"output directory {where} belongs to another identity",
                   existing_identity_sha256=holder.get("identity_sha256"))


def _recover(out: Path, config: Any, identity: Dict[str, Optional[str]],
             a: Dict[str, Any]) -> int:
    """Accept an existing same-identity result only if recomputation is byte-identical."""
    try:
        table, rows = _compute(config, identity, None)
        expected = _serialize(table)
        actual = (out / RESULT_NAME).read_bytes()
    except (elig.EligibilityError, ValueError, OSError) as exc:
        return _refuse("RECOVERY_RECOMPUTATION_FAILED", str(exc))
    if actual != expected:
        return _refuse("RECOVERY_MISMATCH", "existing result bytes differ from the "
                       "recomputation; nothing was written",
                       existing_result_sha256=_sha256_bytes(actual),
                       recomputed_result_sha256=_sha256_bytes(expected))
    status = _Status(out, identity, a["disposition"])
    try:
        status.write(state="COMPLETED", recovered=True, units_total=len(rows),
                     units_done=len(rows), last_progress_at=_now(),
                     result={"path": RESULT_NAME, "sha256": _sha256_bytes(actual),
                             "rows": len(rows)})
    except OSError as exc:
        _not_recorded("COMPLETED", None, exc)
        return EXIT_STATUS_NOT_RECORDED
    _emit("RECOVERED", {"status": status.record})
    return EXIT_OK


def run_command(config_path: Path, out: Path, recover: bool = False) -> int:
    """Compute, reuse or explicitly recover the support table for ``config_path``.

    Returns 0/1/2/3 (see module docstring). An unexpected exception is
    recorded as FAILED and re-raised; :class:`StatusNotRecorded` is raised when
    that record cannot be written. Interrupts are recorded and propagated.
    """
    out = Path(out)
    try:
        raw = Path(config_path).read_bytes()
    except OSError as exc:
        return _refuse("CONFIG_UNREADABLE", str(exc))
    try:
        config = json.loads(raw)
        config_sha = digest(config)
    except ValueError:
        config, config_sha = None, None
    try:
        code_sha = code_sha256()
    except OSError as exc:
        return _refuse("CODE_UNREADABLE", str(exc))
    identity = {"input_sha256": _sha256_bytes(raw), "config_sha256": config_sha,
                "code_sha256": code_sha}
    identity["identity_sha256"] = _identity_sha(config_sha, identity["input_sha256"],
                                                identity["code_sha256"])

    a = assess(out)
    record, rs = a["status"]["record"], a["result"]
    if record is not None and record["identity_sha256"] != identity["identity_sha256"]:
        return _identity_refusal(record, identity, "status")
    if rs["presence"] == "DAMAGED":
        return _refuse("RESULT_DAMAGED", f"existing {RESULT_NAME} is damaged "
                       f"({rs['problem']}); it is never overwritten or adopted")
    if (rs["presence"] == "INTACT"
            and rs["identity"]["identity_sha256"] != identity["identity_sha256"]):
        return _identity_refusal(rs["identity"], identity, "result")
    if a["disposition"] == "RUNNING":
        return _refuse("RUN_IN_PROGRESS", "the recorded writer process is alive")
    if a["disposition"] == "RUNNING_UNVERIFIED":
        return _refuse("RUNNING_LIVENESS_UNVERIFIABLE",
                       "a RUNNING record exists and its writer cannot be checked")
    if a["disposition"] == "COMPLETE_INTACT":
        _emit("REUSED", {"status": record})
        return EXIT_OK
    if rs["presence"] == "INTACT":
        if not recover:
            return _refuse("RECOVERY_REQUIRED", "an intact result of this identity is not "
                           "backed by a matching COMPLETED status; rerun with --recover to "
                           "verify it by recomputation", disposition=a["disposition"],
                           reason=a["problem"])
        return _recover(out, config, identity, a)
    if a["disposition"] == "DAMAGED" and not recover:
        return _refuse("RECOVERY_REQUIRED", "status is damaged or its result is missing; "
                       "rerun with --recover to recompute", disposition=a["disposition"],
                       reason=a["problem"])

    status = _Status(out, identity, a["disposition"])
    try:
        status.write()
    except OSError as exc:
        _not_recorded("RUNNING", None, exc)
        return EXIT_STATUS_NOT_RECORDED
    try:
        table, rows = _compute(config, identity, status)
        data = create_json_exclusive(out / RESULT_NAME, table)
        status.write(state="COMPLETED", last_progress_at=_now(),
                     result={"path": RESULT_NAME, "sha256": _sha256_bytes(data),
                             "rows": len(rows)})
    except (elig.EligibilityError, _CliError) as exc:
        return EXIT_FAILED if _terminal(status, "FAILED", exc.to_json()) \
            else EXIT_STATUS_NOT_RECORDED
    except Exception as exc:
        error = {"code": "UNEXPECTED_ERROR", "exception_type": type(exc).__name__,
                 "message": str(exc)}
        if not _terminal(status, "FAILED", error):
            raise StatusNotRecorded("FAILED could not be recorded") from exc
        raise
    except BaseException as exc:
        _terminal(status, "INTERRUPTED", {"code": "INTERRUPTED",
                                          "exception_type": type(exc).__name__,
                                          "message": str(exc)})
        raise
    _emit("COMPLETED", {"status": status.record})
    return EXIT_OK


def _on_sigterm(signum: int, frame: Any) -> None:
    raise TerminationRequested(f"signal {signum}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; see module docstring for the contract."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="support-table config JSON")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--status", action="store_true", help="print status once (read-only)")
    parser.add_argument("--recover", action="store_true",
                        help="verify an unbacked same-identity result by recomputation")
    args = parser.parse_args(argv)
    if args.status:
        if args.config is not None or args.recover:
            return _refuse("USAGE", "--status takes only --output")
        return status_command(args.output)
    if args.config is None:
        return _refuse("USAGE", "--config is required")
    try:
        previous = signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:                     # not the main thread: leave signals alone
        previous = None
    try:
        return run_command(args.config, args.output, recover=args.recover)
    except StatusNotRecorded:
        traceback.print_exc()
        return EXIT_STATUS_NOT_RECORDED
    except Exception:                      # run_command recorded and emitted FAILED
        traceback.print_exc()
        return EXIT_FAILED
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
