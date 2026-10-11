"""Finite, resumable CLI for I6-N C2 paired metrics (candidate, not deployed).

Usage::

    python -m tools.i6n_contracts.paired_metrics_cli --config INPUT.json --output OUT_DIR
    python -m tools.i6n_contracts.paired_metrics_cli --status --output OUT_DIR

``--config`` is an input document of schema ``i6n.paired_metrics.input.v1``
(see ``tools/i6n_contracts/examples/paired_metrics_synthetic.json``).
``OUT_DIR`` receives ``STATUS.json`` and, on success, ``RESULT.json``; both are
replaced atomically (temporary file + ``os.replace`` through the reused
``tools.i6e_compact_run.atomic_json``). A failed write removes its temporary
file before the failure is reported.

* An output directory is bound to one identity (config file, input, contract
  and code digests). Identical input with an intact terminal ``RESULT.json`` is
  reused without writing; a different identity is refused with
  ``OUTPUT_IDENTITY_MISMATCH`` and nothing is overwritten.
  A reused or recovered ``RESULT.json`` is re-verified semantically
  (``paired_metrics.verify_result``), not only by its digests.
* ``--status`` reads once, prints and exits; it never creates, starts, resumes
  or rewrites anything. It reports success (0, or 4 for INCOMPLETE) ONLY after
  verifying that ``RESULT.json`` exists, parses, has an intact wrapper digest,
  carries the same identity as ``STATUS.json``, passes ``verify_result`` and
  matches the recorded file and result digests. Otherwise it reports
  ``RESULT_INVALID`` with a typed ``disposition`` (``RESULT_MISSING``,
  ``RESULT_UNREADABLE``, ``RESULT_SEAL_INVALID``, ``RESULT_IDENTITY_MISMATCH``,
  ``RESULT_INVALID``, ``RESULT_HASH_MISMATCH``) and exit 6.
* Every failure is typed: a validation refusal keeps its own code; a write
  failure (scoring progress, ``RESULT.json`` or the terminal status) is
  ``OUTPUT_WRITE_FAILED``; any other exception is ``UNEXPECTED_ERROR``; each
  carries the ``phase`` it happened in. An interrupt (``KeyboardInterrupt``,
  ``SystemExit``) is recorded as ``FAILED`` / ``INTERRUPTED`` when the status
  can be written, and is re-raised either way. When the failure itself cannot
  be recorded the run returns a non-success payload with ``durable: false``
  and never claims a durable ``FAILED``; the last status on disk stays what it
  was (``RUNNING`` or absent), which ``--status`` reports as non-success.
  Only a process killed without running Python code (SIGKILL, power loss) or
  such an unrecordable failure can leave ``RUNNING`` behind. There is no
  lease, so ``--status`` cannot tell a live run from a dead one: it reports
  ``RUNNING`` with exit 5 and the ``last_progress_at`` time. Rerunning the
  same input recomputes from scratch (scoring is finite and pure), or
  recovers the status from an intact verified result of the same identity.
* No network, database, lease, remote worker, pandas or TensorFlow. The
  reused atomic writer module imports numpy, but neither sqlite3 nor pandas.

Exit codes: 0 verified terminal COMPLETE result, 2 refused/failed (typed
error), 3 status absent, 4 verified terminal result whose scoring state is
``INCOMPLETE`` (never an approval), 5 status ``RUNNING`` (live or crashed),
6 damaged (status unreadable or unknown, or a COMPLETED status whose
``RESULT.json`` is missing, damaged or does not match it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tools.i6e_compact_run import atomic_json, file_hash
from tools.i6n_contracts import paired_metrics as pm

STATUS_SCHEMA_VERSION = "i6n.paired_metrics.status.v1"
RESULT_FILE_SCHEMA_VERSION = "i6n.paired_metrics.result_file.v1"
STATUS_NAME = "STATUS.json"
RESULT_NAME = "RESULT.json"
EXIT_COMPLETE = 0
EXIT_REFUSED = 2
EXIT_STATUS_ABSENT = 3
EXIT_INCOMPLETE = 4
EXIT_RUNNING = 5
EXIT_DAMAGED = 6

_CODE_FILES = (
    Path(pm.__file__),
    Path(__file__),
    Path(sys.modules[atomic_json.__module__].__file__),
)


def code_sha256() -> str:
    """Digest of this CLI, the pure module and the reused atomic-write helper."""
    return pm.canonical_sha256({p.name: file_hash(p) for p in _CODE_FILES})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else pm.iso_utc(value)


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def _write(path: Path, value: dict) -> None:
    """``atomic_json`` that removes its temporary file when the write fails."""
    try:
        atomic_json(path, value)
    except BaseException:
        try:
            path.with_suffix(".tmp").unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_object(path: Path) -> Optional[dict]:
    """Read an existing JSON object, ``None`` when absent; ValueError when not an object."""
    if not path.exists():
        return None
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def identity_for(raw: bytes) -> dict:
    """Identity bound to an output directory: config file, input, contract and code digests.

    ``input_sha256``/``config_sha256`` stay ``None`` when the bytes are not a
    JSON object; the config file digest still binds the directory.
    """
    identity = {"config_file_sha256": hashlib.sha256(raw).hexdigest(),
                "input_sha256": None, "config_sha256": None, "code_sha256": code_sha256()}
    try:
        document = json.loads(raw, parse_constant=_reject_constant)
        if isinstance(document, dict):
            identity.update(pm.input_digests(document))
    except (ValueError, RecursionError, pm.PairedMetricsError):
        pass
    return identity


class _Status:
    """Atomic STATUS.json writer; every write replaces one complete record."""

    def __init__(self, path: Path, identity: dict, started: datetime, started_clock: float) -> None:
        self.path, self.identity = path, identity
        self.started, self.started_clock = started, started_clock
        self.done: int = 0
        self.total: Optional[int] = None
        self.last_progress = started

    def write(self, state: str, result: Optional[dict] = None, error: Optional[dict] = None,
              reused: bool = False) -> dict:
        elapsed = time.monotonic() - self.started_clock
        terminal = state in ("COMPLETED", "FAILED")
        if terminal:
            eta, reason = None, "TERMINAL_STATE"
        elif self.total is None:
            eta, reason = None, "UNITS_UNKNOWN_BEFORE_VALIDATION"
        elif self.done == 0:
            eta, reason = None, "NO_COMPLETED_UNIT_YET"
        else:
            eta, reason = elapsed / self.done * (self.total - self.done), None
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "component": pm.COMPONENT,
            "state": state,
            "units_done": self.done,
            "units_total": self.total,
            "unit": "population_horizon_role_group",
            "started_at": _iso(self.started),
            "last_progress_at": _iso(self.last_progress),
            "finished_at": _iso(_now()) if terminal else None,
            "duration_seconds": elapsed,
            "eta_seconds": eta,
            "eta_unavailable_reason": reason,
            "identity": self.identity,
            "reused": reused,
            "result": result,
            "error": error,
            "claims": dict(pm.CLAIMS),
        }
        _write(self.path, status)
        return status

    def progress(self, done: int, total: int) -> None:
        self.done, self.total, self.last_progress = done, total, _now()
        self.write("RUNNING")


def _result_summary(result: dict, result_file_sha256: str) -> dict:
    groups = result["groups"]
    return {
        "result_state": result["state"],
        "result_file": RESULT_NAME,
        "result_sha256": result["sha256"],
        "result_file_sha256": result_file_sha256,
        "groups_total": len(groups),
        "groups_by_state": {s: sum(1 for g in groups if g["state"] == s)
                            for s in sorted({g["state"] for g in groups})},
    }


def _refusal(code: str, detail: str) -> tuple:
    return EXIT_REFUSED, {"state": "REFUSED", "error": {"code": code, "detail": detail}}


def _error(code: str, error: BaseException, phase: Optional[str] = None) -> dict:
    typed = {"code": code, "detail": f"{type(error).__name__}: {error}"[:500]}
    if phase is not None:
        typed["phase"] = phase
    return typed


def _verify_stored(stored: Any) -> dict:
    """Wrapper digest, schema and semantic verification of a stored result file.

    Raises ``PairedMetricsError``: ``RESULT_SEAL_INVALID`` for the wrapper, or
    any typed code of ``paired_metrics.verify_result`` for the inner result.
    """
    if (not isinstance(stored, dict) or stored.get("schema_version") != RESULT_FILE_SCHEMA_VERSION
            or stored.get("sha256") != pm.canonical_sha256({k: v for k, v in stored.items() if k != "sha256"})):
        raise pm.PairedMetricsError("RESULT_SEAL_INVALID", "result file digest or schema does not verify")
    pm.verify_result(stored.get("result"))
    return stored["result"]


def _fail(status: "_Status", typed: dict) -> tuple:
    """Record a typed FAILED status; if even that cannot be written, say so (not durable)."""
    try:
        status.write("FAILED", error=typed)
    except Exception as error:  # noqa: BLE001 - the failure is reported, never claimed durable
        return EXIT_REFUSED, {"state": "FAILED_UNRECORDED", "reused": False, "durable": False,
                              "error": typed, "status_write_error": _error("STATUS_WRITE_FAILED", error)}
    return EXIT_REFUSED, {"state": "FAILED", "reused": False, "durable": True, "error": typed}


def _exit_for(result: dict) -> int:
    return EXIT_COMPLETE if result["state"] == "COMPLETE" else EXIT_INCOMPLETE


def run(config_path: Path, output_dir: Path) -> tuple:
    """Score one input into ``output_dir``; returns (exit_code, printed payload)."""
    started, started_clock = _now(), time.monotonic()
    try:
        raw = Path(config_path).read_bytes()
    except OSError as error:
        # Nothing to bind an output directory to: refuse without creating anything.
        return _refusal("CONFIG_UNREADABLE", f"{type(error).__name__}: {error.strerror or error}")
    identity = identity_for(raw)
    parse_error = None
    try:
        document = json.loads(raw, parse_constant=_reject_constant)
        if not isinstance(document, dict):
            raise ValueError("the config must be a JSON object")
    except (ValueError, RecursionError) as error:
        document, parse_error = None, {"code": "CONFIG_INVALID", "detail": f"{type(error).__name__}: {error}"}
    if document is not None and identity["input_sha256"] is None:
        document, parse_error = None, {"code": "CONFIG_INVALID", "detail": "the input has no canonical digest"}

    status_path, result_path = output_dir / STATUS_NAME, output_dir / RESULT_NAME
    try:
        previous = _read_object(status_path)
        stored = _read_object(result_path)
    except (OSError, ValueError, RecursionError) as error:
        return _refusal("OUTPUT_CORRUPT", f"existing output cannot be read: {error}")
    if previous is not None and previous.get("identity") != identity:
        return _refusal("OUTPUT_IDENTITY_MISMATCH", "the output directory holds a different identity")
    if stored is not None:
        if stored.get("identity") != identity:
            return _refusal("OUTPUT_IDENTITY_MISMATCH", "RESULT.json belongs to a different identity")
        try:
            result = _verify_stored(stored)
            summary = _result_summary(result, file_hash(result_path))
        except pm.PairedMetricsError as error:
            return _refusal("OUTPUT_CORRUPT", str(error))
        except OSError as error:
            return _refusal("OUTPUT_CORRUPT", f"existing result cannot be hashed: {type(error).__name__}")
        if previous is None or previous.get("state") != "COMPLETED" or previous.get("result") != summary:
            # Explicit recovery of the status of an intact, verified result of THIS identity.
            status = _Status(status_path, identity, started, started_clock)
            status.done = status.total = len(result["groups"])
            try:
                status.write("COMPLETED", result=summary, reused=True)
            except Exception as error:  # noqa: BLE001 - verified result, status not recorded
                return EXIT_REFUSED, {"state": "REUSE_UNRECORDED", "reused": True, "durable": False,
                                      "result": summary, "error": _error("STATUS_WRITE_FAILED", error)}
        return _exit_for(result), {"state": "COMPLETED", "reused": True, "result": summary}

    status = _Status(status_path, identity, started, started_clock)
    if parse_error is not None:
        return _fail(status, parse_error)
    try:
        status.write("RUNNING")
    except Exception as error:  # noqa: BLE001 - nothing started, nothing claimed
        return EXIT_REFUSED, {"state": "NOT_STARTED", "reused": False, "durable": False,
                              "error": _error("STATUS_WRITE_FAILED", error, "STARTING")}
    phase = "SCORING"
    try:
        result = pm.score(document, progress=status.progress)
        phase = "WRITING_RESULT"
        wrapped = {"schema_version": RESULT_FILE_SCHEMA_VERSION, "identity": identity, "result": result}
        wrapped["sha256"] = pm.canonical_sha256(wrapped)
        _write(result_path, wrapped)
        summary = _result_summary(result, file_hash(result_path))
        phase = "WRITING_STATUS"
        status.write("COMPLETED", result=summary)
    except pm.PairedMetricsError as error:
        typed = {"code": error.code, "detail": error.detail, "phase": phase}
    except OSError as error:
        typed = _error("OUTPUT_WRITE_FAILED", error, phase)
    except Exception as error:  # noqa: BLE001 - every other failure becomes a typed FAILED status
        typed = _error("UNEXPECTED_ERROR", error, phase)
    except BaseException as error:
        try:
            status.write("FAILED", error={"code": "INTERRUPTED", "detail": type(error).__name__, "phase": phase})
        except Exception:  # noqa: BLE001 - the interrupt itself must propagate
            pass
        raise
    else:
        return _exit_for(result), {"state": "COMPLETED", "reused": False, "result": summary}
    return _fail(status, typed)


def _check_completed(output_dir: Path, recorded: dict) -> Optional[tuple]:
    """Verify the RESULT.json behind a COMPLETED status; ``None`` or (disposition, detail)."""
    path = output_dir / RESULT_NAME
    if not path.is_file():
        return "RESULT_MISSING", "STATUS.json says COMPLETED but RESULT.json does not exist"
    try:
        stored = _read_json(path)
    except (OSError, ValueError, RecursionError) as error:
        return "RESULT_UNREADABLE", f"{type(error).__name__}: {error}"[:300]
    if not isinstance(stored, dict):
        return "RESULT_UNREADABLE", "RESULT.json is not a JSON object"
    try:
        body = pm.canonical_sha256({k: v for k, v in stored.items() if k != "sha256"})
    except pm.PairedMetricsError as error:
        return "RESULT_SEAL_INVALID", str(error)[:300]
    if stored.get("schema_version") != RESULT_FILE_SCHEMA_VERSION or stored.get("sha256") != body:
        return "RESULT_SEAL_INVALID", "the result file digest or schema does not verify"
    if stored.get("identity") != recorded.get("identity"):
        return "RESULT_IDENTITY_MISMATCH", "RESULT.json belongs to a different identity than STATUS.json"
    try:
        result = _verify_stored(stored)
    except pm.PairedMetricsError as error:
        return "RESULT_INVALID", str(error)[:300]
    try:
        summary = _result_summary(result, file_hash(path))
    except OSError as error:
        return "RESULT_UNREADABLE", f"{type(error).__name__}: {error}"[:300]
    if summary != recorded.get("result"):
        return "RESULT_HASH_MISMATCH", "RESULT.json differs from the result recorded in STATUS.json"
    return None


def status(output_dir: Path) -> tuple:
    """Read STATUS.json once and verify its result; never creates or changes anything."""
    path = output_dir / STATUS_NAME
    if not path.is_file():
        return EXIT_STATUS_ABSENT, {"schema_version": STATUS_SCHEMA_VERSION, "state": "ABSENT",
                                    "disposition": "STATUS_ABSENT",
                                    "error": {"code": "STATUS_ABSENT", "detail": "no STATUS.json"}}
    try:
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError("STATUS.json is not a JSON object")
        state = value.get("state")
        result_state = (value.get("result") or {}).get("result_state") if state == "COMPLETED" else None
        if state not in ("RUNNING", "FAILED", "COMPLETED") or (
                state == "COMPLETED" and result_state not in ("COMPLETE", "INCOMPLETE")):
            raise ValueError(f"unknown status state {state!r}/{result_state!r}")
    except (OSError, ValueError, RecursionError, AttributeError) as error:
        return EXIT_DAMAGED, {"schema_version": STATUS_SCHEMA_VERSION, "state": "UNREADABLE",
                              "disposition": "STATUS_UNREADABLE",
                              "error": {"code": "STATUS_UNREADABLE", "detail": str(error)}}
    if state == "RUNNING":
        return EXIT_RUNNING, dict(value, disposition="RUNNING_OR_DEAD_NO_LEASE")
    if state == "FAILED":
        return EXIT_REFUSED, dict(value, disposition="FAILED")
    problem = _check_completed(output_dir, value)
    if problem is not None:
        disposition, detail = problem
        return EXIT_DAMAGED, {
            "schema_version": STATUS_SCHEMA_VERSION, "state": "RESULT_INVALID", "disposition": disposition,
            "recorded_state": state, "identity": value.get("identity"),
            "result": {"verified": False, "result_state": None, "recorded": value.get("result")},
            "error": {"code": disposition, "detail": detail}}
    verification = {"verified": True, "result_file": RESULT_NAME,
                    "result_file_sha256": value["result"]["result_file_sha256"],
                    "result_sha256": value["result"]["result_sha256"]}
    if result_state == "INCOMPLETE":
        return EXIT_INCOMPLETE, dict(value, disposition="INCOMPLETE_VERIFIED", result_verification=verification)
    return EXIT_COMPLETE, dict(value, disposition="COMPLETE_VERIFIED", result_verification=verification)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="input document (i6n.paired_metrics.input.v1)")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--status", action="store_true", help="print STATUS.json once; read-only")
    args = parser.parse_args(argv)
    if args.status:
        code, payload = status(args.output)
    elif args.config is None:
        parser.error("--config is required unless --status is given")
    else:
        code, payload = run(args.config, args.output)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
