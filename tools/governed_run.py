#!/usr/bin/env python3
"""Fail-safe governed predictor run (data-gov Flow v3).

The campaign and its units are registered before data is opened. Each role is
delivered and confirmed independently, predictor runs on CPU, and every
outcome is persisted to a durable local outbox before it is reported to the
remote terminal lake. Network calls never occur inside training.

    tools/governed_run.py --load_config <cfg> --experiment-key K
        [--experiment-set-key S] --gov-url http://127.0.0.1:5055
        --api-key-file <file> --lake predictor_examples
        --lake-root examples/data_downsampled --metrics-lake olap_cube
        --out-dir <dir> [--cache-dir <dir>] [--from YYYY-MM-DD --to YYYY-MM-DD]
        -- <extra --flags for app/main.py>

The HTTP client is kept in this file: importing data-gov's `app.client`
would shadow predictor's own `app` package.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = (
    "x_train_file", "y_train_file",
    "x_validation_file", "y_validation_file",
    "x_test_file", "y_test_file",
)
# Outputs always redirected under --out-dir, with the config's basename when
# it names one, else predictor's default basename, so a config that relies
# on the defaults cannot write into the repository root either.
OUTPUT_DEFAULTS = {
    "results_file": "results.csv",
    "output_file": "prediction.csv",
    "uncertainties_file": "test_uncertainties.csv",
    "save_model": "predictor_model.keras",
    "loss_plot_file": "loss_plot.png",
    "model_plot_file": "model_plot.png",
    "predictions_plot_file": "predictions_plot.png",
}
PRIVATE_DEFAULTS = {"save_config": "config_out.json", "save_log": "debug_out.json"}
DEFAULT_CACHE = "~/.cache/data-gov"
DEFAULT_OUTBOX = "~/.local/state/data-gov/terminal-outbox"
KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
METRIC_ROW = re.compile(r"^\s*(Train|Validation|Test)\s+(.+?)(?:\s+H(\d+))?\s*$", re.I)
CHUNK = 1 << 20


class GovernedRunError(Exception):
    pass


# ---------------------------------------------------------------- pure parts


def resolve_inputs(config: dict, base_dir) -> dict:
    """key -> resolved path for each of the six input keys the config sets."""
    out = {}
    for key in INPUT_KEYS:
        value = config.get(key)
        if value:
            path = Path(str(value)).expanduser()
            out[key] = (path if path.is_absolute() else Path(base_dir) / path).resolve()
    return out


def distinct_paths(inputs: dict) -> list:
    seen = []
    for path in inputs.values():
        if path not in seen:
            seen.append(path)
    return seen


def resource_for(path, lake_root) -> str:
    root = Path(lake_root).resolve()
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        raise GovernedRunError(f"input {path} is outside the lake root {root}") from None


def redirect_outputs(config: dict, out_dir) -> dict:
    out = dict(config)
    out_dir = Path(out_dir)
    for key, default in {**OUTPUT_DEFAULTS, **PRIVATE_DEFAULTS}.items():
        out[key] = str(out_dir / Path(str(config.get(key) or default)).name)
    for key, value in config.items():
        if key.endswith("_plot_file") and key not in OUTPUT_DEFAULTS and value:
            out[key] = str(out_dir / Path(str(value)).name)
    return out


def governed_config(config: dict, cached: dict, out_dir) -> dict:
    out = redirect_outputs(config, out_dir)
    for key, path in cached.items():
        out[key] = str(path)
    return out


def refuse_stale_outputs(config: dict, out_dir) -> None:
    """A governing unit gets a fresh output namespace; prior bytes are never replaced."""
    redirected = redirect_outputs(config, out_dir)
    paths = {Path(redirected[key]) for key in {**OUTPUT_DEFAULTS, **PRIVATE_DEFAULTS}}
    paths.update(
        Path(value) for key, value in redirected.items()
        if key.endswith("_plot_file") and value
    )
    paths.add(Path(out_dir) / "governed_config.json")
    stale = sorted(str(path) for path in paths if path.exists())
    if stale:
        raise GovernedRunError(
            "governing output namespace is not fresh: " + ", ".join(stale)
        )


def _num(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


METRIC_KEY_DISALLOWED = re.compile(r"[^A-Za-z0-9._:-]+")


def metric_key(label: str) -> str:
    """governed_terminal.v1 identifies a metric by a key over [A-Za-z0-9._:-].
    predictor labels carry spaces (`Naive MAE`); every run of other characters
    becomes one underscore, deterministically, so `Naive MAE` -> `Naive_MAE`."""
    key = METRIC_KEY_DISALLOWED.sub("_", label.strip()).strip("_")
    return key or "metric"


def parse_results_rows(rows) -> list:
    """Results rows (Metric, Average, Std Dev, Min, Max) -> metric rows.
    `Train MAE H24` -> metric MAE, split train, horizon 24; a label without
    the split or the horizon keeps them null. Metric names are terminal keys
    (see metric_key)."""
    metrics = []
    for row in rows:
        label = (row.get("Metric") or "").strip()
        if not label:
            continue
        match = METRIC_ROW.match(label)
        if match:
            split = match.group(1).lower()
            metric = metric_key(match.group(2))
            horizon = int(match.group(3)) if match.group(3) else None
        else:
            split, metric, horizon = None, metric_key(label), None
        metrics.append({
            "metric": metric,
            "value": _num(row.get("Average")),
            "split": split,
            "horizon": horizon,
            "std_dev": _num(row.get("Std Dev")),
            "min_value": _num(row.get("Min")),
            "max_value": _num(row.get("Max")),
            "unit": None,
        })
    return metrics


def parse_results_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return parse_results_rows(csv.DictReader(handle))


def canonical_config(effective: dict, identities: dict) -> str:
    """Section 8.4: the six input keys become gov:<lake>/<resource>@<sha256>,
    output paths their basenames, save_config/save_log are dropped, then
    compact sorted JSON. Invariant to the cache and output directories."""
    # load_config names the governed config under the output directory: dropped like save_config/save_log
    out = {k: v for k, v in effective.items() if k not in PRIVATE_DEFAULTS and k != "load_config"}
    for key in INPUT_KEYS:
        if key in identities:
            out[key] = identities[key]
    for key in list(out):
        if (key in OUTPUT_DEFAULTS or key.endswith("_plot_file")) and out[key]:
            out[key] = Path(str(out[key])).name
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def refuse_governed_overrides(extra) -> None:
    """Extra flags may tune the run (--epochs ...), never replace a governed input, an output or the config."""
    governed = set(INPUT_KEYS) | set(OUTPUT_DEFAULTS) | set(PRIVATE_DEFAULTS) | {"load_config"}
    for token in extra:
        if not str(token).startswith("--"):
            continue
        name = str(token)[2:].split("=", 1)[0]
        if name in governed or name.endswith("_plot_file"):
            raise GovernedRunError(f"--{name} would override a governed input or output; refused")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def strict_code_identity(repo_root) -> dict:
    """Return a governing commit only for an exact, clean checkout."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise GovernedRunError("governing run requires a 40-hex git commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        # Name what is uncommitted: the refusal is about uncommitted SOURCE, and an
        # engineer must not have to re-derive which path dirtied the checkout.
        offenders = dirty.splitlines()
        shown = "; ".join(offenders[:10])
        if len(offenders) > 10:
            shown += f"; ... and {len(offenders) - 10} more"
        raise GovernedRunError("governing run requires a clean checkout; uncommitted: " + shown)
    return {"kind": "git_commit", "value": head}


def code_commit(repo_root) -> str:
    """Compatibility helper; governing callers use strict_code_identity()."""
    identity = strict_code_identity(repo_root)
    return identity["value"]


def execution_spec(config: dict, datasets: list, extra: list) -> str:
    """Canonical pre-execution contract, independent of local paths."""
    body = {
        key: value for key, value in config.items()
        if key not in set(INPUT_KEYS) | set(PRIVATE_DEFAULTS) | {"load_config"}
    }
    for key in list(body):
        if key in OUTPUT_DEFAULTS or key.endswith("_plot_file"):
            body[key] = Path(str(body[key])).name if body[key] else body[key]
    body = {
        "schema": "predictor_execution_spec.v1",
        "config": body,
        "datasets": sorted(datasets, key=lambda item: (
            item["lake"], item["resource"], item["role"],
            item.get("from") or "", item.get("to") or "",
        )),
        "extra_arguments": list(extra),
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n").encode("utf-8")
    part = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, path)
        _fsync_dir(path.parent)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class OutboxItem:
    path: Path
    state: str
    payload: dict


FAILURE_CLASSES = ("TRANSIENT", "CONFIGURATION", "REFUSED_BY_SERVER", "UNRESOLVED")
DISPOSITIONS = ("INVALID_ENVELOPE", "SUPERSEDED")
_HTTP_IN_ERROR = re.compile(r"\bhttp (\d{3})\b")


def classify_failure(error: str) -> str:
    """What a send failure means for the pending envelope. A 4xx alone never
    decides that the envelope is invalid: it awaits an explicit disposition."""
    match = _HTTP_IN_ERROR.search(error or "")
    if match is None:
        text = (error or "").lower()
        if "diverge" in text or "reconciliation" in text or "missing after accepted" in text:
            return "UNRESOLVED"
        return "TRANSIENT"
    status = int(match.group(1))
    if status >= 500 or status == 429:
        return "TRANSIENT"
    if status in (401, 403, 404):
        return "CONFIGURATION"
    return "REFUSED_BY_SERVER"


class _SpoolLock:
    """Exclusive access to one outbox while it is flushed."""

    def __init__(self, root):
        self.path = Path(root) / ".flush.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        return False


class TerminalOutbox:
    """Write-once terminal queue; accepted sends move atomically to sent/.

    A refused send never deletes evidence: the envelope stays in pending/ with a
    failure sidecar (attempts, last error, class). An envelope that can never be
    accepted, or that a corrected successor replaces, is moved unchanged to
    adjudicated/ next to a write-once disposition record, so a permanent
    refusal is visible, traceable and no longer blocks other work."""

    def __init__(self, root):
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.sent = self.root / "sent"
        self.adjudicated = self.root / "adjudicated"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.sent.mkdir(parents=True, exist_ok=True)
        self.adjudicated.mkdir(parents=True, exist_ok=True)

    # failure sidecars -------------------------------------------------
    def _failure_path(self, path: Path) -> Path:
        # sidecars are not envelopes: a different suffix keeps every *.json glob honest
        return path.with_name(path.name[:-5] + ".failure")

    def _record_failure(self, path: Path, error: str):
        sidecar = self._failure_path(path)
        record = {"attempts": 0, "first_seen": _utc_now()}
        if sidecar.is_file():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        record.update(attempts=int(record.get("attempts", 0)) + 1, last_seen=_utc_now(),
                      last_error=error, **{"class": classify_failure(error)})
        _write_json_atomic(sidecar, record)
        return record

    def _failure(self, path: Path):
        sidecar = self._failure_path(path)
        return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None

    def _pending_files(self):
        return sorted(self.pending.glob("*.json"))

    # health -----------------------------------------------------------
    def status(self) -> dict:
        """Recoverable pendings, envelopes awaiting adjudication, unresolved
        failures and adjudicated cases are told apart; nothing is hidden."""
        pending = []
        for path in self._pending_files():
            envelope = json.loads(path.read_text(encoding="ascii"))
            failure = self._failure(path) or {}
            pending.append({
                "file": path.name, "campaign_sha256": envelope["campaign_sha256"],
                "unit_id": envelope["unit_id"], "generation": envelope["terminal"].get("generation"),
                "status": envelope["terminal"].get("status"),
                "class": failure.get("class", "NOT_YET_SENT"), "attempts": failure.get("attempts", 0),
                "last_error": failure.get("last_error"),
            })
        adjudicated = []
        for path in sorted(self.adjudicated.glob("*.disposition.json")):
            adjudicated.append(json.loads(path.read_text(encoding="utf-8")))
        counts = {"recoverable": sum(p["class"] in ("NOT_YET_SENT", "TRANSIENT", "CONFIGURATION") for p in pending),
                  "awaiting_adjudication": sum(p["class"] == "REFUSED_BY_SERVER" for p in pending),
                  "unresolved": sum(p["class"] == "UNRESOLVED" for p in pending)}
        return {"schema": "terminal_outbox_status.v1", "sent": len(list(self.sent.glob("*.json"))),
                "pending": pending, "adjudicated": adjudicated, **counts}

    # disposition ------------------------------------------------------
    def dispose(self, name: str, decision: str, reason: str, *, successor_terminal_sha256=None,
                successor_generation=None) -> dict:
        """Move a pending envelope, unchanged, to adjudicated/ with a write-once
        disposition. INVALID_ENVELOPE closes it; SUPERSEDED links the accepted
        successor terminal. Nothing is deleted or rewritten."""
        if decision not in DISPOSITIONS:
            raise GovernedRunError(f"unknown disposition {decision!r}")
        if not reason or not str(reason).strip():
            raise GovernedRunError("a disposition states its reason")
        path = self.pending / name
        if not path.is_file() or not name.endswith(".json"):
            raise GovernedRunError(f"{name} is not a pending envelope")
        if decision == "SUPERSEDED" and not (isinstance(successor_terminal_sha256, str)
                                             and re.fullmatch(r"[0-9a-f]{64}", successor_terminal_sha256)):
            raise GovernedRunError("SUPERSEDED needs the accepted successor terminal_sha256")
        raw = path.read_bytes()
        envelope = json.loads(raw)
        record = {
            "schema": "terminal_outbox_disposition.v1", "file": name, "decision": decision,
            "reason": str(reason), "envelope_sha256": hashlib.sha256(raw).hexdigest(),
            "campaign_sha256": envelope["campaign_sha256"], "unit_id": envelope["unit_id"],
            "generation": envelope["terminal"].get("generation"), "status": envelope["terminal"].get("status"),
            "failure": self._failure(path), "successor_terminal_sha256": successor_terminal_sha256,
            "successor_generation": successor_generation, "disposed_at": _utc_now(),
        }
        target = self.adjudicated / name
        if target.exists():
            raise GovernedRunError("adjudicated outbox identity conflict")
        disposition = self.adjudicated / (name[:-5] + ".disposition.json")
        fd = os.open(disposition, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(path, target)
        sidecar = self._failure_path(path)
        if sidecar.is_file():
            os.replace(sidecar, self.adjudicated / sidecar.name)
        _fsync_dir(self.pending)
        _fsync_dir(self.adjudicated)
        return record

    def supersede(self, name: str, corrected_terminal: dict, sender, reason: str) -> dict:
        """Send a corrected terminal as the next generation of the same campaign and
        unit, then dispose the original as SUPERSEDED. The successor keeps the
        original's outcome and deliveries: a FAILED run never becomes COMPLETED
        by correction, and no delivery is added or dropped."""
        path = self.pending / name
        if not path.is_file() or not name.endswith(".json"):
            raise GovernedRunError(f"{name} is not a pending envelope")
        original = json.loads(path.read_text(encoding="ascii"))
        base = original["terminal"]
        if not isinstance(corrected_terminal, dict) or corrected_terminal.get("schema") != "governed_terminal.v1":
            raise GovernedRunError("the successor must be a governed_terminal.v1")
        if corrected_terminal.get("status") != base.get("status"):
            raise GovernedRunError("a successor keeps the original outcome")
        if sorted(corrected_terminal.get("deliveries") or []) != sorted(base.get("deliveries") or []):
            raise GovernedRunError("a successor keeps the original deliveries")
        generation = int(base.get("generation", 1)) + 1
        successor = dict(corrected_terminal, generation=generation)
        successor["tags"] = {**(successor.get("tags") or {}),
                             "supersedes_envelope_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "supersedes_generation": str(base.get("generation", 1))}
        envelope = {"campaign_sha256": original["campaign_sha256"], "unit_id": original["unit_id"],
                    "terminal": successor}
        item = self.put(envelope)
        receipt = sender(envelope)
        if not isinstance(receipt, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("terminal_sha256"))):
            self._record_failure(item.path, "terminal receipt missing")
            raise GovernedRunError("successor terminal was not accepted")
        target = self.sent / item.path.name
        os.replace(item.path, target)
        _fsync_dir(self.pending)
        _fsync_dir(self.sent)
        return self.dispose(name, "SUPERSEDED", reason, successor_terminal_sha256=receipt["terminal_sha256"],
                            successor_generation=generation)

    def put(self, payload):
        raw = (json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n").encode("ascii")
        digest = hashlib.sha256(raw).hexdigest()
        path = self.pending / f"{digest}.json"
        if path.exists():
            if path.read_bytes() != raw:
                raise GovernedRunError("outbox identity conflict")
            return OutboxItem(path, "PENDING", payload)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        _fsync_dir(self.pending)
        return OutboxItem(path, "PENDING", payload)

    def flush(self, sender):
        """Send every pending envelope once. A refusal keeps the envelope pending
        and is reported under `failures` (file -> reason) so a permanent refusal
        is diagnosable from GOVERNED_RUN.json instead of a bare pending count.

        One flusher at a time per spool, and an envelope another flusher already
        delivered is not a failure: two processes sharing a spool used to race on
        the same file and die with FileNotFoundError mid-send."""
        with _SpoolLock(self.root):
            return self._flush_locked(sender)

    def _flush_locked(self, sender):
        sent = 0
        failures = {}
        delivered_elsewhere = []
        for path in self._pending_files():
            try:
                payload = json.loads(path.read_text(encoding="ascii"))
            except FileNotFoundError:
                delivered_elsewhere.append(path.name)      # another flusher moved it while we looked
                continue
            try:
                receipt = sender(payload)
                if not isinstance(receipt, dict) or not receipt.get("terminal_sha256"):
                    raise GovernedRunError("terminal receipt missing")
            except FileNotFoundError:
                delivered_elsewhere.append(path.name)
                continue
            except Exception as exc:
                failures[path.name] = f"{type(exc).__name__}: {exc}"
                self._record_failure(path, failures[path.name])
                continue
            target = self.sent / path.name
            try:
                if target.exists():
                    if target.read_bytes() != path.read_bytes():
                        raise GovernedRunError("sent outbox identity conflict")
                    path.unlink()
                else:
                    os.replace(path, target)
            except FileNotFoundError:
                delivered_elsewhere.append(path.name)
                continue
            self._failure_path(path).unlink(missing_ok=True)
            _fsync_dir(self.pending)
            _fsync_dir(self.sent)
            sent += 1
        return {
            "sent": sent,
            "pending": len(self._pending_files()),
            "failures": failures,
            **({"delivered_by_another_flusher": sorted(set(delivered_elsewhere))} if delivered_elsewhere else {}),
        }


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_report(metrics_lake, metrics, datasets, *, experiment_set_key=None,
                 config_sha256=None, code_commit=None, project=None, phase=None,
                 tags=None) -> dict:
    return {
        "lake": metrics_lake,
        "experiment_set_key": experiment_set_key,
        "config_sha256": config_sha256,
        "code_commit": code_commit,
        "project": project,
        "phase": phase,
        "tags": tags or {},
        "datasets": datasets,
        "metrics": metrics,
    }


# ------------------------------------------------------------------- HTTP


def _error_text(raw: bytes) -> str:
    try:
        return str(json.loads(raw.decode()).get("error") or "")
    except (ValueError, AttributeError):
        return raw[:200].decode(errors="replace").strip()


class GovHttp:
    """Minimal data-gov client over http.client: JSON POST and a streamed,
    hash-verified download with a connect timeout only."""

    def __init__(self, base_url: str, api_key: str, experiment_key: str):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise GovernedRunError(f"bad --gov-url {base_url!r}")
        self._parts = parts
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Experiment-Key": experiment_key,
        }

    def _request_headers(self, campaign_sha256=None, unit_id=None):
        headers = dict(self._headers)
        if campaign_sha256:
            headers["X-Campaign-SHA256"] = campaign_sha256
        if unit_id:
            headers["X-Unit-ID"] = unit_id
        return headers

    def _connection(self, connect_timeout=30):
        cls = http.client.HTTPSConnection if self._parts.scheme == "https" else http.client.HTTPConnection
        try:
            conn = cls(self._parts.hostname, self._parts.port, timeout=connect_timeout)
            conn.connect()
        except OSError as exc:
            raise GovernedRunError(f"data-gov unreachable at {self._parts.netloc}: {exc}") from None
        conn.sock.settimeout(None)
        return conn

    def _path(self, path: str, params=None) -> str:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        return self._parts.path.rstrip("/") + path + (f"?{query}" if query else "")

    def post_json(self, path: str, body: dict, *, campaign_sha256=None, unit_id=None):
        data = json.dumps(body, allow_nan=False).encode()
        conn = self._connection()
        try:
            conn.request("POST", self._path(path), body=data,
                         headers={**self._request_headers(campaign_sha256, unit_id),
                                  "Content-Type": "application/json"})
            response = conn.getresponse()
            status, raw = response.status, response.read()
        finally:
            conn.close()
        try:
            payload = json.loads(raw.decode() or "{}")
        except ValueError:
            payload = {"error": raw[:200].decode(errors="replace")}
        return status, payload

    def get_json(self, path: str, *, campaign_sha256=None, unit_id=None):
        conn = self._connection()
        try:
            conn.request(
                "GET", self._path(path),
                headers=self._request_headers(campaign_sha256, unit_id),
            )
            response = conn.getresponse()
            status, raw = response.status, response.read()
        finally:
            conn.close()
        try:
            payload = json.loads(raw.decode() or "{}")
        except ValueError:
            payload = {"error": raw[:200].decode(errors="replace")}
        return status, payload

    def submit_campaign(self, body):
        return self.post_json("/api/v2/campaigns", body)

    def governed_download(
        self, campaign_sha256, unit_id, lake, resource, role, cache_dir,
        start=None, end=None, attempts=20,
    ):
        """Receive and confirm one role only after both stream and cache verify."""
        params = {
            "lake": lake, "resource": resource, "role": role,
            "from": start, "to": end,
        }
        target_dir = Path(cache_dir) / lake
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(resource).suffix
        for _ in range(attempts):
            conn = self._connection()
            try:
                conn.request(
                    "GET", self._path("/api/v2/download", params),
                    headers=self._request_headers(campaign_sha256, unit_id),
                )
                response = conn.getresponse()
                if response.status == 503 and response.getheader("Retry-After"):
                    response.read()
                    wait = _num(response.getheader("Retry-After")) or 30
                    time.sleep(min(wait, 300))
                    continue
                if response.status != 200:
                    raise GovernedRunError(
                        f"download {lake}/{resource}: http {response.status} "
                        f"{_error_text(response.read())}"
                    )
                expected = (response.getheader("X-Content-SHA256") or "").lower()
                delivery_id = response.getheader("X-Delivery-ID") or ""
                contract_sha = (
                    response.getheader("X-Availability-Contract-SHA256") or ""
                ).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise GovernedRunError(f"download {lake}/{resource}: missing content digest")
                if not re.fullmatch(r"[0-9a-f]{32}", delivery_id):
                    raise GovernedRunError(f"download {lake}/{resource}: missing delivery identity")
                if not re.fullmatch(r"[0-9a-f]{64}", contract_sha):
                    raise GovernedRunError(
                        f"download {lake}/{resource}: missing availability contract digest"
                    )
                target = target_dir / f"{expected}{ext}"
                cached = target.is_file()
                if cached:
                    cache_digest, cache_size = _sha256_file(target)
                    if cache_digest != expected:
                        raise GovernedRunError(
                            f"cached {lake}/{resource} has a different digest; refused"
                        )
                part = target_dir / (
                    f"{expected}{ext}.{os.getpid()}.{uuid.uuid4().hex}.part"
                )
                digest = hashlib.sha256()
                size = 0
                fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        while True:
                            chunk = response.read(CHUNK)
                            if not chunk:
                                break
                            digest.update(chunk)
                            handle.write(chunk)
                            size += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    part.unlink(missing_ok=True)
                    raise
                actual = digest.hexdigest()
                if actual != expected:
                    part.unlink(missing_ok=True)
                    raise GovernedRunError(
                        f"download {lake}/{resource}: sha256 mismatch "
                        f"(lake said {expected[:12]}, got {actual[:12]})"
                    )
                if cached:
                    if cache_size != size:
                        part.unlink(missing_ok=True)
                        raise GovernedRunError("cache size differs from governed delivery")
                    part.unlink()
                else:
                    try:
                        os.link(part, target)
                    except FileExistsError:
                        winner_digest, winner_size = _sha256_file(target)
                        if winner_digest != expected or winner_size != size:
                            raise GovernedRunError("cache publication identity conflict")
                        cached = True
                    finally:
                        part.unlink(missing_ok=True)
                    _fsync_dir(target_dir)
                info = {
                    "path": str(target), "sha256": actual, "bytes": size,
                    "source_sha256": response.getheader("X-Source-SHA256") or None,
                    "delivery": response.getheader("X-Delivery") or None,
                    "time_column": response.getheader("X-Time-Column") or None,
                    "availability_contract_sha256": contract_sha,
                    # the contract's scope, published by the lake separately from its digest
                    "availability_use": response.getheader("X-Availability-Use") or "UNDECLARED",
                    "availability_label": response.getheader("X-Availability-Label") or "UNKNOWN",
                    "availability_completion_lag_max": response.getheader("X-Availability-Completion-Lag-Max") or None,
                    "timezone_evidence": response.getheader("X-Timezone-Evidence") or "UNKNOWN",
                    "delivery_id": delivery_id, "cached": cached,
                    "resource": resource, "role": role,
                }
            finally:
                conn.close()
            status, receipt = self.post_json(
                f"/api/v2/deliveries/{delivery_id}/confirm",
                {
                    "schema": "delivery_confirmation.v1",
                    "sha256": actual,
                    "bytes": size,
                    "cached": cached,
                },
                campaign_sha256=campaign_sha256,
            )
            if status != 200:
                raise GovernedRunError(
                    f"delivery confirmation refused: http {status} "
                    f"{receipt.get('error', '')}".strip()
                )
            info["verification_state"] = receipt.get("state")
            return 200, info
        raise GovernedRunError(f"download {lake}/{resource}: busy after {attempts} attempts")

    def report_terminal(self, campaign_sha256, unit_id, terminal):
        return self.post_json(
            f"/api/v2/campaigns/{campaign_sha256}/units/{unit_id}/terminal",
            terminal, campaign_sha256=campaign_sha256, unit_id=unit_id,
        )

    def reconcile_campaign(self, campaign_sha256):
        return self.get_json(
            f"/api/v2/campaigns/{campaign_sha256}/reconcile",
            campaign_sha256=campaign_sha256,
        )

    def download(self, lake: str, resource: str, cache_dir, start=None, end=None,
                 attempts=20) -> dict:
        params = {"lake": lake, "resource": resource, "from": start, "to": end}
        target_dir = Path(cache_dir) / lake
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(resource).suffix
        for _ in range(attempts):
            conn = self._connection()
            try:
                conn.request("GET", self._path("/api/v1/download", params), headers=self._headers)
                response = conn.getresponse()
                if response.status == 503 and response.getheader("Retry-After"):
                    response.read()
                    wait = _num(response.getheader("Retry-After")) or 30
                    time.sleep(min(wait, 300))
                    continue
                if response.status != 200:
                    raise GovernedRunError(
                        f"download {lake}/{resource}: http {response.status} "
                        f"{_error_text(response.read())}"
                    )
                expected = (response.getheader("X-Content-SHA256") or "").lower()
                if not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise GovernedRunError(f"download {lake}/{resource}: no X-Content-SHA256")
                info = {
                    "resource": resource,
                    "filename": response.getheader("Content-Disposition") or "",
                    "source_sha256": response.getheader("X-Source-SHA256"),
                    "delivery": response.getheader("X-Delivery"),
                    "time_column": response.getheader("X-Time-Column") or None,
                }
                # one writer, one part file: concurrent runs of the same bytes never share a partial file
                part = target_dir / f"{expected}{ext}.{os.getpid()}.{uuid.uuid4().hex}.part"
                digest = hashlib.sha256()
                size = 0
                with open(part, "wb") as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        digest.update(chunk)
                        handle.write(chunk)
                        size += len(chunk)
            finally:
                conn.close()
            actual = digest.hexdigest()
            if actual != expected:
                part.unlink(missing_ok=True)
                raise GovernedRunError(
                    f"download {lake}/{resource}: sha256 mismatch "
                    f"(lake said {expected[:12]}, got {actual[:12]})"
                )
            final = target_dir / f"{expected}{ext}"
            os.replace(part, final)
            info.update({"path": str(final), "sha256": actual, "bytes": size,
                         "cached": f"{lake}/{expected}{ext}"})
            return info
        raise GovernedRunError(f"download {lake}/{resource}: busy after {attempts} attempts")


# -------------------------------------------------------------------- run


def load_api_key(path) -> str:
    if path:
        key = Path(path).expanduser().read_text(encoding="utf-8").strip()
    else:
        key = (os.getenv("DATA_GOV_API_KEY") or "").strip()
    if not key:
        raise GovernedRunError("no API key: pass --api-key-file or set DATA_GOV_API_KEY")
    return key


def _parser():
    parser = argparse.ArgumentParser(
        description="Run predictor on data-gov governed inputs and report the metrics.",
        epilog="Arguments after `--` are passed to app/main.py (long flags only).",
    )
    parser.add_argument("--load_config", required=True, help="predictor config JSON")
    parser.add_argument("--experiment-key", required=True)
    parser.add_argument("--experiment-set-key")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", help="file with the service key (else DATA_GOV_API_KEY)")
    parser.add_argument("--lake", default="predictor_examples", help="data-gov lake of the inputs")
    parser.add_argument("--lake-root", default="examples/data_downsampled",
                        help="directory the lake serves; inputs map to paths relative to it")
    parser.add_argument("--metrics-lake", default="olap_cube")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--outbox-dir", default=DEFAULT_OUTBOX)
    parser.add_argument("--from", dest="range_from", metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="range_to", metavar="YYYY-MM-DD")
    parser.add_argument("--project", default="predictor")
    parser.add_argument("--phase", help="default: the config's parent directory name")
    parser.add_argument("--classification", choices=("GOVERNING", "NON_GOVERNING"),
                        default="GOVERNING",
                        help="NON_GOVERNING for a mechanical check: it records transport and "
                             "cost, and grants nothing scientific")
    return parser


def parse_args(argv):
    parser = _parser()
    return parser.parse_args(argv)


def _under_repo(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _artifact(role, path):
    path = Path(path)
    if not path.is_file():
        return None
    digest, size = _sha256_file(path)
    return {"role": role, "sha256": digest, "bytes": size}


def _terminal_artifacts(config):
    artifacts = []
    candidates = {
        "governed_config": config.get("_governed_config_path"),
        "effective_config": config.get("save_config"),
        "results": config.get("results_file"),
        "predictions": config.get("output_file"),
        "uncertainties": config.get("uncertainties_file"),
        "model": config.get("save_model"),
    }
    for role, path in candidates.items():
        if path:
            item = _artifact(role, path)
            if item:
                artifacts.append(item)
    return artifacts


def _send_pending(gov, outbox):
    def sender(envelope):
        status, receipt = gov.report_terminal(
            envelope["campaign_sha256"], envelope["unit_id"], envelope["terminal"]
        )
        if status not in (200, 201):
            raise GovernedRunError(
                f"terminal refused: http {status} {receipt.get('error', '')}".strip()
            )
        _require_reconciled(
            gov, envelope["campaign_sha256"], envelope["unit_id"], before_run=False
        )
        return receipt

    return outbox.flush(sender)


def _require_reconciled(gov, campaign_sha256, unit_id, *, before_run=False):
    status, body = gov.reconcile_campaign(campaign_sha256)
    if status != 200:
        raise GovernedRunError(
            f"reconciliation failed: http {status} {body.get('error', '')}".strip()
        )
    if body.get("accounting_only") or body.get("lake_only"):
        raise GovernedRunError("terminal accounting and terminal lake diverge")
    missing = body.get("missing_units")
    if not isinstance(missing, list):
        raise GovernedRunError("invalid reconciliation response")
    if before_run and unit_id not in missing:
        raise GovernedRunError("campaign unit already has a terminal")
    if not before_run and unit_id in missing:
        raise GovernedRunError("terminal is missing after accepted report")
    return body


def run(args, extra) -> dict:
    key = args.experiment_key
    if not KEY_RE.match(key):
        raise GovernedRunError(f"invalid --experiment-key {key!r}")
    if args.experiment_set_key and not KEY_RE.match(args.experiment_set_key):
        raise GovernedRunError(f"invalid --experiment-set-key {args.experiment_set_key!r}")
    refuse_governed_overrides(extra)
    api_key = load_api_key(args.api_key_file)
    config_path = _under_repo(args.load_config)
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    code_identity = strict_code_identity(REPO_ROOT)
    inputs = resolve_inputs(config, REPO_ROOT)
    if not inputs:
        raise GovernedRunError("the config names none of the six input keys")
    lake_root = _under_repo(args.lake_root)
    datasets = [{
        "lake": args.lake,
        "resource": resource_for(path, lake_root),
        "role": role,
        "from": args.range_from,
        "to": args.range_to,
    } for role, path in inputs.items()]
    spec = execution_spec(config, datasets, extra)
    config_sha256 = sha256_text(spec)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.path.expanduser(args.cache_dir)).resolve()
    outbox = TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    state = {
        "status": "RUNNING",
        "experiment_key": key,
        "experiment_set_key": args.experiment_set_key,
        "gov_url": args.gov_url,
        "lake": args.lake,
        "metrics_lake": args.metrics_lake,
        "config": str(config_path),
        "cache_dir": args.cache_dir,
        "out_dir": str(out_dir),
        "range": {"from": args.range_from, "to": args.range_to},
        "code_identity": code_identity,
        "execution_spec": json.loads(spec),
        "config_sha256": config_sha256,
        "classification": args.classification,
    }
    receipt_path = out_dir / "GOVERNED_RUN.json"

    def checkpoint():
        _write_json_atomic(receipt_path, state)

    gov = GovHttp(args.gov_url, api_key, key)
    campaign = {
        "schema": "governed_campaign.v1",
        "campaign_key": key,
        "classification": args.classification,
        "project": args.project,
        "code_identity": code_identity,
        "config_sha256": config_sha256,
        "input_mode": "DATASETS",
        "synthetic_spec_sha256": None,
        "units": [key],
        "datasets": datasets,
        "terminal_lake": args.metrics_lake,
    }
    status, campaign_receipt = gov.submit_campaign(campaign)
    if status not in (200, 201):
        raise GovernedRunError(
            f"campaign refused: http {status} {campaign_receipt.get('error', '')}".strip()
        )
    campaign_sha256 = campaign_receipt.get("campaign_sha256")
    if not isinstance(campaign_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", campaign_sha256
    ):
        raise GovernedRunError("campaign receipt has no valid identity")
    state["campaign_sha256"] = campaign_sha256
    state["campaign_receipt"] = campaign_receipt
    checkpoint()

    prior = _send_pending(gov, outbox)
    state["prior_outbox_flush"] = prior
    if prior["pending"]:
        # adjudicated envelopes no longer block; pending ones do, and their class says why
        health = outbox.status()
        state["status"] = "REFUSED"
        state["reason"] = "PRIOR_TERMINAL_PENDING"
        state["outbox_status"] = health
        checkpoint()
        raise GovernedRunError(
            "a prior terminal remains pending: " + ", ".join(
                f"{p['file'][:12]} {p['class']}" for p in health["pending"]))
    _require_reconciled(gov, campaign_sha256, key, before_run=True)

    started_at = _utc_now()
    wall_start = time.monotonic()
    downloads = {}
    delivery_ids = []
    metrics = []
    gcfg = {}
    failure = None
    terminal_status = "COMPLETED"
    terminal_reason = None
    try:
        refuse_stale_outputs(config, out_dir)
        for item in datasets:
            status, info = gov.governed_download(
                campaign_sha256, key, item["lake"], item["resource"], item["role"],
                cache_dir, item["from"], item["to"],
            )
            if status != 200:
                raise GovernedRunError(
                    f"download {item['lake']}/{item['resource']} refused: "
                    f"http {status} {info.get('error', '')}".strip()
                )
            downloads[item["role"]] = info
            delivery_ids.append(info["delivery_id"])
        state["inputs"] = [
            {"role": role, **{k: downloads[role].get(k) for k in (
                "resource", "sha256", "bytes", "cached", "source_sha256", "delivery",
                "time_column", "availability_contract_sha256", "availability_use", "availability_label",
                "availability_completion_lag_max", "timezone_evidence", "delivery_id",
                "verification_state",
            )}}
            for role in inputs
        ]
        cached = {role: downloads[role]["path"] for role in inputs}
        gcfg = governed_config(config, cached, out_dir)
        gcfg_path = out_dir / "governed_config.json"
        _write_json_atomic(gcfg_path, gcfg)
        gcfg["_governed_config_path"] = str(gcfg_path)
        state["governed_config"] = str(gcfg_path)

        cmd = [sys.executable, "app/main.py", "--load_config", str(gcfg_path), *extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO_ROOT))
        state["command"] = cmd
        checkpoint()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        state["exit_code"] = proc.returncode
        if proc.returncode != 0:
            terminal_status = "FAILED"
            terminal_reason = f"PREDICTOR_EXIT_{proc.returncode}"
            raise GovernedRunError(f"predictor exited {proc.returncode}")

        effective_path = Path(gcfg["save_config"])
        if not effective_path.is_file():
            raise GovernedRunError(f"predictor wrote no effective config at {effective_path}")
        with open(effective_path, encoding="utf-8") as handle:
            effective = json.load(handle)
        for role, cached_path in cached.items():
            used = effective.get(role)
            if used is None or Path(str(used)).resolve() != Path(cached_path).resolve():
                raise GovernedRunError(f"the run did not use the governed input for {role}: {used!r}")
        identities = {
            role: f"gov:{args.lake}/{downloads[role]['resource']}@{downloads[role]['sha256']}"
            for role in inputs
        }
        canonical = canonical_config(effective, identities)
        state["config_canonical"] = canonical
        state["effective_config_sha256"] = sha256_text(canonical)

        results_path = Path(gcfg["results_file"])
        if not results_path.is_file():
            raise GovernedRunError(f"no results CSV at {results_path}")
        metrics = parse_results_csv(results_path)
        if not metrics:
            raise GovernedRunError(f"no metric rows in {results_path}")
    except BaseException as exc:
        failure = exc
        if terminal_status == "COMPLETED":
            terminal_status = "REFUSED" if isinstance(exc, GovernedRunError) else "FAILED"
            terminal_reason = (
                f"GOVERNED_RUN_REFUSED:{exc}" if isinstance(exc, GovernedRunError)
                else f"UNEXPECTED_{type(exc).__name__.upper()}"
            )
    finished_at = _utc_now()
    plugin = config.get("predictor_plugin") or config.get("plugin")
    terminal = {
        "schema": "governed_terminal.v1",
        "generation": 1,
        "status": terminal_status,
        "reason": terminal_reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "costs": {"wall_seconds": max(0.0, time.monotonic() - wall_start)},
        "deliveries": delivery_ids,
        "artifacts": _terminal_artifacts(gcfg),
        "metrics": metrics if terminal_status == "COMPLETED" else [],
        "tags": {
            "phase": str(args.phase or config_path.parent.name),
            "experiment_set_key": str(args.experiment_set_key or ""),
            "plugin": str(plugin or ""),
            "exit_code": str(state.get("exit_code", "")),
            # the weakest availability scope among the inputs bounds what the result may claim
            "availability_use": ",".join(sorted({
                str(info.get("availability_use") or "UNDECLARED") for info in downloads.values()
            })) or "NONE",
        },
    }
    envelope = {
        "campaign_sha256": campaign_sha256,
        "unit_id": key,
        "terminal": terminal,
    }
    item = outbox.put(envelope)
    state["terminal_outbox"] = str(item.path)
    state["terminal"] = terminal
    state["status"] = terminal_status
    if terminal_reason:
        state["reason"] = terminal_reason
    checkpoint()

    flushed = _send_pending(gov, outbox)
    state["outbox_flush"] = flushed
    if flushed["pending"]:
        state["terminal_pending"] = True
        checkpoint()
        raise GovernedRunError("terminal remains pending; scientific result is not governing")
    state["reconciliation"] = _require_reconciled(
        gov, campaign_sha256, key, before_run=False
    )
    state["terminal_pending"] = False
    checkpoint()
    if failure is not None:
        if isinstance(failure, GovernedRunError):
            raise failure
        raise GovernedRunError(f"{type(failure).__name__}: {failure}") from failure
    return state


def build_parser():
    """The CLI surface as one object, so a caller or a test exercises it instead of
    rebuilding it (order P4: a reconstructed parser proves what the test wrote)."""
    return _parser()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = parse_args(argv)
    try:
        state = run(args, extra)
    except GovernedRunError as exc:
        print(f"governed_run: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"governed_run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"governed_run: {state['experiment_key']} status={state['status']} "
        f"campaign={state['campaign_sha256']} "
        f"receipt={Path(state['out_dir']) / 'GOVERNED_RUN.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
