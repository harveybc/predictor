"""Finite, resumable CLI for I6-N C2 paired metrics (candidate, not deployed).

Usage::

    python -m tools.i6n_contracts.paired_metrics_cli --config INPUT.json --output OUT_DIR
    python -m tools.i6n_contracts.paired_metrics_cli --status --output OUT_DIR

``--config`` is an input document of schema ``i6n.paired_metrics.input.v1``
(see ``tools/i6n_contracts/examples/paired_metrics_synthetic.json``).
``OUT_DIR`` receives ``STATUS.json`` and, on success, ``RESULT.json``; both are
replaced atomically (temporary file + ``os.replace`` through the reused
``tools.i6e_compact_run.atomic_json``).

* An output directory is bound to one identity (config file, input, contract
  and code digests). Identical input with an intact terminal ``RESULT.json`` is
  reused without writing; a different identity is refused with
  ``OUTPUT_IDENTITY_MISMATCH`` and nothing is overwritten.
* ``--status`` reads ``STATUS.json`` once, prints it and exits. It never
  creates, starts or resumes work.
* No network, database, lease, remote worker, pandas or TensorFlow.

Exit codes: 0 complete, 2 refused/failed (typed error), 3 status absent,
4 terminal result whose scoring state is ``INCOMPLETE`` (never an approval).
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
        atomic_json(self.path, status)
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


def _exit_for(result: dict) -> int:
    return EXIT_COMPLETE if result["state"] == "COMPLETE" else EXIT_INCOMPLETE


def run(config_path: Path, output_dir: Path) -> tuple:
    """Score one input into ``output_dir``; returns (exit_code, printed payload)."""
    started, started_clock = _now(), time.monotonic()
    raw = Path(config_path).read_bytes()
    identity = {"config_file_sha256": hashlib.sha256(raw).hexdigest(),
                "input_sha256": None, "config_sha256": None, "code_sha256": code_sha256()}
    parse_error = None
    try:
        document = json.loads(raw, parse_constant=_reject_constant)
        if not isinstance(document, dict):
            raise ValueError("the config must be a JSON object")
        identity.update(pm.input_digests(document))
    except (ValueError, pm.PairedMetricsError) as error:
        document, parse_error = None, {"code": "CONFIG_INVALID", "detail": str(error)}

    status_path, result_path = output_dir / STATUS_NAME, output_dir / RESULT_NAME
    try:
        previous = _read_json(status_path) if status_path.exists() else None
        stored = _read_json(result_path) if result_path.exists() else None
    except (OSError, ValueError) as error:
        return _refusal("OUTPUT_CORRUPT", f"existing output cannot be read: {error}")
    if previous is not None and previous.get("identity") != identity:
        return _refusal("OUTPUT_IDENTITY_MISMATCH", "the output directory holds a different identity")
    if stored is not None:
        if not isinstance(stored, dict) or stored.get("identity") != identity:
            return _refusal("OUTPUT_IDENTITY_MISMATCH", "RESULT.json belongs to a different identity")
        try:
            pm.verify_result(stored.get("result"))
            if stored.get("sha256") != pm.canonical_sha256({k: v for k, v in stored.items() if k != "sha256"}):
                raise pm.PairedMetricsError("RESULT_SEAL_INVALID", "result file digest does not verify")
        except pm.PairedMetricsError as error:
            return _refusal("OUTPUT_CORRUPT", str(error))
        result = stored["result"]
        summary = _result_summary(result, file_hash(result_path))
        if previous is None or previous.get("state") != "COMPLETED" or previous.get("result") != summary:
            status = _Status(status_path, identity, started, started_clock)
            status.done = status.total = len(result["groups"])
            status.write("COMPLETED", result=summary, reused=True)
        return _exit_for(result), {"state": "COMPLETED", "reused": True, "result": summary}

    status = _Status(status_path, identity, started, started_clock)
    if parse_error is not None:
        status.write("FAILED", error=parse_error)
        return EXIT_REFUSED, {"state": "FAILED", "reused": False, "error": parse_error}
    status.write("RUNNING")
    try:
        result = pm.score(document, progress=status.progress)
    except pm.PairedMetricsError as error:
        typed = {"code": error.code, "detail": error.detail}
        status.write("FAILED", error=typed)
        return EXIT_REFUSED, {"state": "FAILED", "reused": False, "error": typed}
    wrapped = {"schema_version": RESULT_FILE_SCHEMA_VERSION, "identity": identity, "result": result}
    wrapped["sha256"] = pm.canonical_sha256(wrapped)
    atomic_json(result_path, wrapped)
    summary = _result_summary(result, file_hash(result_path))
    status.write("COMPLETED", result=summary)
    return _exit_for(result), {"state": "COMPLETED", "reused": False, "result": summary}


def status(output_dir: Path) -> tuple:
    """Read STATUS.json once without creating or changing anything."""
    path = output_dir / STATUS_NAME
    if not path.is_file():
        return EXIT_STATUS_ABSENT, {"schema_version": STATUS_SCHEMA_VERSION, "state": "ABSENT",
                                    "error": {"code": "STATUS_ABSENT", "detail": "no STATUS.json"}}
    try:
        value = _read_json(path)
    except (OSError, ValueError) as error:
        return EXIT_REFUSED, {"schema_version": STATUS_SCHEMA_VERSION, "state": "UNREADABLE",
                              "error": {"code": "STATUS_UNREADABLE", "detail": str(error)}}
    if value.get("state") == "FAILED":
        return EXIT_REFUSED, value
    if value.get("state") == "COMPLETED" and (value.get("result") or {}).get("result_state") == "INCOMPLETE":
        return EXIT_INCOMPLETE, value
    return EXIT_COMPLETE, value


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
