#!/usr/bin/env python3
"""Atomically collect one coherent four-island terminal observation.

The collector is a bridge between live DOIN node APIs and
``i6n_island_successor``.  A timer tick fetches ``/chain/status`` and
``/status`` from exactly four private nodes concurrently, validates the
captured responses with the fleet verifier, and publishes all four sealed
terminal envelopes with one directory rename.  No per-node snapshot is ever
made visible.

Node URLs and runtime paths belong only in the private JSON spec.  ``--status``
does no network I/O and never creates or modifies a file.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any

from tools import i6n_doin_fleet_verify as fleet_verify


SPEC_SCHEMA = "i6n.island_terminal_collect.spec.v1"
STATUS_SCHEMA = "i6n.island_terminal_collect.status.v1"
TERMINAL_SCHEMA = "i6n.island.terminal.v1"
COMPONENTS = fleet_verify.COMPONENTS
REPO_ROOT = Path(__file__).resolve().parents[1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISLAND_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SpecError(ValueError):
    """The private collection specification is malformed or unsafe."""


class Refusal(RuntimeError):
    """Observed or retained evidence contradicts the collection contract."""


FetchJSON = Callable[[str, float], Mapping[str, Any]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _absolute_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SpecError(f"{field} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise SpecError(f"{field} must be an absolute path")
    return path


def _paths_overlap(first: Path, second: Path) -> bool:
    return _inside(first, second) or _inside(second, first)


def _atomic_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"INVALID_JSON: {field}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise Refusal(f"INVALID_JSON: {field} root must be an object")
    return value


def _validated_spec(raw: Mapping[str, Any], *, spec_path: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"schema must equal {SPEC_SCHEMA}")
    if _inside(spec_path, REPO_ROOT):
        raise SpecError("the private collector spec must be outside the repository")

    state_dir = _absolute_path(raw.get("state_dir"), "state_dir")
    output_dir = _absolute_path(raw.get("output_dir"), "output_dir")
    if _inside(state_dir, REPO_ROOT) or _inside(output_dir, REPO_ROOT):
        raise SpecError("state_dir and output_dir must be outside the repository")
    if _paths_overlap(state_dir, output_dir):
        raise SpecError("state_dir and output_dir must not overlap")

    expected = raw.get("expected")
    expected_keys = {
        "horizon", "seed", "scientific_config_sha256", "component_versions"
    }
    if not isinstance(expected, Mapping) or set(expected) != expected_keys:
        raise SpecError(f"expected must contain exactly {sorted(expected_keys)}")
    if expected["horizon"] != 4:
        raise SpecError("expected.horizon must equal 4")
    if type(expected["seed"]) is not int or not 0 <= expected["seed"] <= 0xFFFFFFFF:
        raise SpecError("expected.seed must be a uint32 integer")
    config_digest = expected["scientific_config_sha256"]
    if not isinstance(config_digest, str) or not _SHA256_RE.fullmatch(config_digest):
        raise SpecError(
            "expected.scientific_config_sha256 must be 64 lowercase hexadecimal characters"
        )

    fleet_spec = {
        "schema": fleet_verify.SPEC_SCHEMA,
        "expected": {
            "seed": expected["seed"],
            "config_digest": config_digest,
            "component_versions": expected["component_versions"],
        },
        "nodes": raw.get("nodes"),
    }
    try:
        clean_fleet = fleet_verify._validated_spec(fleet_spec)
    except fleet_verify.FleetSpecError as exc:
        raise SpecError(str(exc)) from exc

    timeout = raw.get("timeout_seconds", 5)
    if type(timeout) not in (int, float) or isinstance(timeout, bool) or timeout <= 0:
        raise SpecError("timeout_seconds must be a positive number")
    poll = raw.get("poll_interval_seconds", 60)
    if type(poll) is not int or poll <= 0:
        raise SpecError("poll_interval_seconds must be a positive integer")

    return {
        "schema": SPEC_SCHEMA,
        "spec_path": spec_path.resolve(),
        "state_dir": state_dir,
        "output_dir": output_dir,
        "timeout_seconds": float(timeout),
        "poll_interval_seconds": poll,
        "expected": {
            "horizon": 4,
            "seed": expected["seed"],
            "scientific_config_sha256": config_digest,
            "component_versions": dict(clean_fleet["expected"]["component_versions"]),
        },
        "fleet_spec": clean_fleet,
    }


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load a private collection spec without creating runtime state."""
    spec_path = Path(path)
    if _inside(spec_path, REPO_ROOT):
        raise SpecError("the private collector spec must be outside the repository")
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read private spec: {type(exc).__name__}") from exc
    return _validated_spec(raw, spec_path=spec_path)


def _terminal_paths(spec: Mapping[str, Any]) -> list[Path]:
    return [spec["output_dir"] / f"island-{index}.json" for index in range(4)]


def _envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    return {
        "schema": TERMINAL_SCHEMA,
        "payload": clean,
        "payload_sha256": _digest(clean),
    }


def _capture(
    spec: Mapping[str, Any], fetch_json: FetchJSON
) -> tuple[dict[str, Mapping[str, Any] | Exception], dict[str, Any]]:
    """Fetch every endpoint concurrently, then run the verifier over that cache."""
    requests = [
        (f"{node['base_url']}/{endpoint}", spec["timeout_seconds"])
        for node in spec["fleet_spec"]["nodes"]
        for endpoint in ("chain/status", "status")
    ]
    captured: dict[str, Mapping[str, Any] | Exception] = {}

    def fetch_one(url: str, timeout: float) -> Mapping[str, Any]:
        value = fetch_json(url, timeout)
        if not isinstance(value, Mapping):
            raise ValueError("response root is not an object")
        # Freeze the complete JSON tree returned by this request.  The verifier
        # and terminal adapter must consume the same captured bytes, not a
        # mutable object that an in-process transport could later change.
        return json.loads(_canonical(value))

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="i6n-terminal") as pool:
        futures = {pool.submit(fetch_one, url, timeout): url for url, timeout in requests}
        for future in as_completed(futures):
            url = futures[future]
            try:
                captured[url] = future.result()
            except Exception as exc:
                captured[url] = exc

    def from_capture(url: str, _timeout: float) -> Mapping[str, Any]:
        value = captured[url]
        if isinstance(value, Exception):
            raise value
        return value

    report = fleet_verify.verify_fleet(
        spec["fleet_spec"],
        fetch_json=from_capture,
        timeout=spec["timeout_seconds"],
    )
    return captured, report


def _fleet_refusal(report: Mapping[str, Any]) -> str | None:
    if report.get("ready") is True:
        return None
    codes = sorted({str(error.get("code", "UNKNOWN")) for error in report.get("errors", [])})
    if codes and set(codes) <= {"NODE_UNREACHABLE"}:
        return "WAIT_OBSERVATION"
    raise Refusal("FLEET_REFUSED: " + ",".join(codes or ["UNKNOWN"]))


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Refusal(f"INVALID_TERMINAL_FIELD: {field}")
    return value


def _required_digest(value: Any, field: str) -> str:
    value = _required_string(value, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Refusal(f"INVALID_TERMINAL_FIELD: {field}")
    return value


def _terminal_payloads(
    spec: Mapping[str, Any],
    captured: Mapping[str, Mapping[str, Any] | Exception],
    fleet_report: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    expected = spec["expected"]
    fleet = fleet_report["fleet"]
    finalized = fleet.get("finalized_checkpoint")
    if (not isinstance(finalized, Mapping) or not isinstance(finalized.get('hash'), str)
            or (finalized.get('height') == -1 and finalized['hash'] != '')
            or (finalized.get('height') != -1 and not finalized['hash'])):
        raise Refusal("MISSING_FINALIZED_HASH")

    terminal_rows: list[dict[str, Any]] = []
    waiting = False
    for node in spec["fleet_spec"]["nodes"]:
        base = node["base_url"]
        status = captured[f"{base}/status"]
        chain = captured[f"{base}/chain/status"]
        if not isinstance(status, Mapping) or not isinstance(chain, Mapping):
            raise Refusal("OBSERVATION_CACHE_INCOMPLETE")
        if status.get("chain_height") != chain.get("chain_height"):
            raise Refusal(
                f"ENDPOINT_OBSERVATION_MISMATCH: {node['node_id']}.chain_height"
            )
        optimization = status.get("optimization")
        terminal = optimization.get("terminal") if isinstance(optimization, Mapping) else None
        if terminal is None:
            waiting = True
            continue
        if not isinstance(terminal, Mapping):
            raise Refusal(f"INVALID_TERMINAL_FIELD: {node['node_id']}.terminal")
        terminal_status = terminal.get("status")
        if terminal_status == "FAILED":
            raise Refusal(f"TERMINAL_FAILED: {node['node_id']}")
        if terminal_status != "COMPLETE":
            waiting = True
            continue

        island_id = _required_string(terminal.get("island_id"), "island_id")
        if not _ISLAND_RE.fullmatch(island_id):
            raise Refusal(f"INVALID_TERMINAL_FIELD: {node['node_id']}.island_id")
        if terminal.get("horizon") != expected["horizon"]:
            raise Refusal(f"HORIZON_MISMATCH: {island_id}")
        if terminal.get("seed") != expected["seed"]:
            raise Refusal(f"SEED_MISMATCH: {island_id}.terminal")
        if terminal.get("scientific_config_sha256") != expected["scientific_config_sha256"]:
            raise Refusal(f"SCIENTIFIC_CONFIG_MISMATCH: {island_id}.terminal")
        generation = terminal.get("terminal_generation")
        if type(generation) is not int or generation < 1:
            raise Refusal(f"INVALID_TERMINAL_FIELD: {island_id}.terminal_generation")
        champion_fitness = terminal.get("champion_fitness")
        if (type(champion_fitness) not in (int, float)
                or not math.isfinite(champion_fitness)):
            raise Refusal(f"INVALID_TERMINAL_FIELD: {island_id}.champion_fitness")
        evaluated = terminal.get("evaluated_candidate_count")
        if type(evaluated) is not int or evaluated < 1:
            raise Refusal(
                f"INVALID_TERMINAL_FIELD: {island_id}.evaluated_candidate_count"
            )
        terminal_rows.append({
            "status": "COMPLETE",
            "island_id": island_id,
            "horizon": expected["horizon"],
            "seed": expected["seed"],
            "scientific_config_sha256": expected["scientific_config_sha256"],
            "stop_reason": _required_string(terminal.get("stop_reason"), "stop_reason"),
            "terminal_generation": generation,
            "champion_sha256": _required_digest(
                terminal.get("champion_sha256"), "champion_sha256"
            ),
            "champion_fitness": float(champion_fitness),
            "plan_sha256": _required_digest(terminal.get("plan_sha256"), "plan_sha256"),
            "contract_sha256": _required_digest(
                terminal.get("contract_sha256"), "contract_sha256"
            ),
            "checkpoint_identity_sha256": _required_digest(
                terminal.get("checkpoint_identity_sha256"),
                "checkpoint_identity_sha256",
            ),
            "final_population_sha256": _required_digest(
                terminal.get("final_population_sha256"), "final_population_sha256"
            ),
            "candidate_ledger_sha256": _required_digest(
                terminal.get("candidate_ledger_sha256"), "candidate_ledger_sha256"
            ),
            "evaluated_candidate_count": evaluated,
        })

    if waiting:
        return "WAIT_TERMINALS", []
    if len(terminal_rows) != 4:
        return "WAIT_TERMINALS", []
    island_ids = [row["island_id"] for row in terminal_rows]
    if len(set(island_ids)) != 4:
        raise Refusal("DUPLICATE_ISLAND")

    observation = {
        "seed": expected["seed"],
        "scientific_config_sha256": expected["scientific_config_sha256"],
        "network_id": fleet["network_id"],
        "genesis_hash": fleet["genesis_hash"],
        "chain_height": fleet["chain_height"],
        "tip_hash": fleet["tip_hash"],
        "finalized_height": finalized["height"],
        "finalized_hash": finalized["hash"],
        "component_versions": expected["component_versions"],
        "terminals": sorted(terminal_rows, key=lambda row: row["island_id"]),
    }
    observation_sha256 = _digest(observation)
    payloads = []
    for terminal in terminal_rows:
        payloads.append(dict(
            terminal,
            network_id=fleet["network_id"],
            genesis_hash=fleet["genesis_hash"],
            chain_height=fleet["chain_height"],
            tip_hash=fleet["tip_hash"],
            finalized_height=finalized["height"],
            finalized_hash=finalized["hash"],
            component_versions=dict(expected["component_versions"]),
            observation_sha256=observation_sha256,
        ))
    return "READY", payloads


def _validate_existing(spec: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    output_dir = spec["output_dir"]
    if not output_dir.exists():
        return None
    try:
        if not output_dir.is_dir():
            raise Refusal("output path is not a directory")
        expected_names = {f"island-{index}.json" for index in range(4)}
        observed_names = {path.name for path in output_dir.iterdir()}
        if observed_names != expected_names:
            raise Refusal("terminal file set is incomplete")
        documents = [_read_json(path, path.name) for path in _terminal_paths(spec)]
        payloads = []
        for document in documents:
            if set(document) != {"schema", "payload", "payload_sha256"}:
                raise Refusal("invalid envelope keys")
            if document.get("schema") != TERMINAL_SCHEMA:
                raise Refusal("invalid envelope schema")
            payload = document.get("payload")
            if not isinstance(payload, dict) or document.get("payload_sha256") != _digest(payload):
                raise Refusal("invalid envelope digest")
            if payload.get("status") != "COMPLETE":
                raise Refusal("terminal is not complete")
            if payload.get("horizon") != 4 or payload.get("seed") != spec["expected"]["seed"]:
                raise Refusal("terminal identity mismatch")
            if payload.get("scientific_config_sha256") != spec["expected"]["scientific_config_sha256"]:
                raise Refusal("terminal config mismatch")
            if payload.get("component_versions") != spec["expected"]["component_versions"]:
                raise Refusal("component versions mismatch")
            payloads.append(payload)
        if len({payload.get("island_id") for payload in payloads}) != 4:
            raise Refusal("duplicate island identity")
        if len({payload.get("observation_sha256") for payload in payloads}) != 1:
            raise Refusal("mixed observations")
        common_fields = (
            "network_id", "genesis_hash", "chain_height", "tip_hash",
            "finalized_height", "finalized_hash", "observation_sha256",
        )
        baseline = payloads[0]
        if any(payload.get(field) != baseline.get(field)
               for payload in payloads[1:] for field in common_fields):
            raise Refusal("mixed fleet checkpoints")
        terminal_fields = (
            "status", "island_id", "horizon", "seed",
            "scientific_config_sha256", "stop_reason", "terminal_generation",
            "champion_sha256", "champion_fitness", "plan_sha256",
            "contract_sha256", "checkpoint_identity_sha256",
            "final_population_sha256", "candidate_ledger_sha256",
            "evaluated_candidate_count",
        )
        observation = {
            "seed": baseline["seed"],
            "scientific_config_sha256": baseline["scientific_config_sha256"],
            "network_id": baseline.get("network_id"),
            "genesis_hash": baseline.get("genesis_hash"),
            "chain_height": baseline.get("chain_height"),
            "tip_hash": baseline.get("tip_hash"),
            "finalized_height": baseline.get("finalized_height"),
            "finalized_hash": baseline.get("finalized_hash"),
            "component_versions": baseline["component_versions"],
            "terminals": sorted(
                ({field: payload.get(field) for field in terminal_fields}
                 for payload in payloads),
                key=lambda row: str(row["island_id"]),
            ),
        }
        if baseline.get("observation_sha256") != _digest(observation):
            raise Refusal("observation digest mismatch")
        return documents
    except Refusal as exc:
        raise Refusal(f"EXISTING_OUTPUT_MISMATCH: {exc}") from exc


def _report(state: str, *, observed: int, published: int, detail: str | None = None,
            observed_at_unix_ns: int | None = None) -> dict[str, Any]:
    report = {
        "schema": STATUS_SCHEMA,
        "state": state,
        "progress": {"observed": observed, "required": 4, "published": published},
        "eta": ({"kind": "complete", "seconds": 0}
                if state == "COMPLETE" else {"kind": "unknown"}),
    }
    if detail is not None:
        report["detail"] = detail
    if observed_at_unix_ns is not None:
        report["observed_at_unix_ns"] = observed_at_unix_ns
    return report


@contextmanager
def _exclusive_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "LOCK"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _publish_directory(
    spec: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    output_dir = spec["output_dir"]
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    os.chmod(stage, 0o700)
    try:
        for index, payload in enumerate(payloads):
            _atomic_json(stage / f"island-{index}.json", _envelope(payload))
        if before_publish is not None:
            before_publish()
        try:
            os.rename(stage, output_dir)
        except OSError:
            existing = _validate_existing(spec)
            wanted = [_envelope(payload) for payload in payloads]
            if existing != wanted:
                raise Refusal("EXISTING_OUTPUT_MISMATCH: retained receipts differ")
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def tick(
    spec_path: str | Path,
    *,
    fetch_json: FetchJSON = fleet_verify._fetch_json,
    now_ns: Callable[[], int] = time.time_ns,
    before_publish: Callable[[], None] | None = None,
    after_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Perform one timer-safe collection attempt."""
    spec = load_spec(spec_path)
    with _exclusive_lock(spec["state_dir"]):
        existing = _validate_existing(spec)
        if existing is not None:
            report = _report("COMPLETE", observed=4, published=4)
            _atomic_json(spec["state_dir"] / "STATUS.json", report)
            return report

        observed_at = now_ns()
        captured, fleet_report = _capture(spec, fetch_json)
        fleet_state = _fleet_refusal(fleet_report)
        if fleet_state == "WAIT_OBSERVATION":
            responding = int(fleet_report.get("fleet", {}).get("responding_nodes", 0))
            report = _report(
                "WAIT_OBSERVATION", observed=responding, published=0,
                observed_at_unix_ns=observed_at,
            )
            _atomic_json(spec["state_dir"] / "STATUS.json", report)
            return report

        state, payloads = _terminal_payloads(spec, captured, fleet_report)
        if state == "WAIT_TERMINALS":
            report = _report(
                state, observed=4, published=0, observed_at_unix_ns=observed_at
            )
            _atomic_json(spec["state_dir"] / "STATUS.json", report)
            return report

        _publish_directory(spec, payloads, before_publish=before_publish)
        if after_publish is not None:
            after_publish()
        report = _report(
            "COMPLETE", observed=4, published=4, observed_at_unix_ns=observed_at
        )
        _atomic_json(spec["state_dir"] / "STATUS.json", report)
        return report


def status(spec_path: str | Path) -> dict[str, Any]:
    """Inspect retained state without network or filesystem writes."""
    spec = load_spec(spec_path)
    existing = _validate_existing(spec)
    if existing is not None:
        return _report("COMPLETE", observed=4, published=4)
    status_path = spec["state_dir"] / "STATUS.json"
    if status_path.is_file():
        return _read_json(status_path, "STATUS.json")
    return _report("NOT_COLLECTED", observed=0, published=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically collect one coherent four-island terminal observation."
    )
    parser.add_argument("--spec", required=True, type=Path, help="private JSON spec")
    parser.add_argument(
        "--status", action="store_true", help="read retained state only; do no network I/O"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = status(args.spec) if args.status else tick(args.spec)
    except (SpecError, Refusal) as exc:
        report = _report("REFUSED", observed=0, published=0, detail=str(exc))
        if not args.status:
            try:
                spec = load_spec(args.spec)
                _atomic_json(spec["state_dir"] / "STATUS.json", report)
            except (SpecError, Refusal, OSError):
                pass
        print(_canonical(report))
        return 2
    print(_canonical(report))
    return 0 if report["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
