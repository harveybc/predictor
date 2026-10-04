"""Durable follower for the TRAIN-only PS3-R -> PS4 evidence chain.

The follower discovers completed PS3-R terminals, binds each feature to its
declared PS2 batch, and asks the existing PS4 runner to profile the five inner
TRAIN folds.  It deliberately does not open target, validation, or test data.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


STATUS_SCHEMA = "feature_selection_follower_status.v1"
ACCEPTED_INDEX_SCHEMA = "ps4_accepted_units.v1"
DEFAULT_FOLDS = ("inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023")


class FollowerError(RuntimeError):
    """A configuration or evidence error that must be reported explicitly."""


@dataclass(frozen=True)
class Settings:
    result_roots: tuple[Path, ...]
    ps2_batch_dirs: tuple[Path, ...]
    folds_file: Path
    ps4_runner: Path
    ps4_output_dir: Path
    ps4_units_dir: Path
    accepted_units_path: Path
    status_path: Path
    lock_path: Path
    poll_interval_seconds: float
    canonical_bounds: bool
    dataset_id: str


@dataclass(frozen=True)
class Batch:
    batch_id: str
    manifest_path: Path
    manifest_sha256: str
    series_path: Path
    series_sha256: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class Terminal:
    feature_id: str
    manifest_path: Path
    manifest: dict[str, Any]
    results_path: Path
    results_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise FollowerError(f"regular file required: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FollowerError(f"invalid JSON at {path}: {error}") from error


def _atomic_json(path: Path, payload: Any) -> None:
    """Publish JSON only after its complete bytes are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path(value: Any, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FollowerError(f"{name} must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def load_settings(path: Path) -> Settings:
    config_path = Path(path).resolve()
    payload = _read_json(config_path)
    if not isinstance(payload, dict):
        raise FollowerError("config must be a JSON object")
    base = config_path.parent

    def paths(name: str) -> tuple[Path, ...]:
        values = payload.get(name)
        if not isinstance(values, list) or not values:
            raise FollowerError(f"{name} must be a non-empty list")
        return tuple(_path(value, f"{name}[]", base) for value in values)

    output = _path(payload.get("ps4_output_dir"), "ps4_output_dir", base)
    units = _path(payload.get("ps4_units_dir"), "ps4_units_dir", base)
    if units != output / "units":
        raise FollowerError("ps4_units_dir must equal ps4_output_dir/units for the PS4 runner")
    poll = payload.get("poll_interval_seconds", 60)
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or poll <= 0:
        raise FollowerError("poll_interval_seconds must be a positive number")
    accepted = _path(
        payload.get("accepted_units_path", str(output / "accepted_units.json")),
        "accepted_units_path",
        base,
    )
    status = _path(
        payload.get("status_path", str(output / "STATUS.json")), "status_path", base
    )
    lock = _path(payload.get("lock_path", str(output / ".follower.lock")), "lock_path", base)
    return Settings(
        result_roots=paths("result_roots"),
        ps2_batch_dirs=paths("ps2_batch_dirs"),
        folds_file=_path(payload.get("folds_file"), "folds_file", base),
        ps4_runner=_path(payload.get("ps4_runner"), "ps4_runner", base),
        ps4_output_dir=output,
        ps4_units_dir=units,
        accepted_units_path=accepted,
        status_path=status,
        lock_path=lock,
        poll_interval_seconds=float(poll),
        canonical_bounds=bool(payload.get("canonical_bounds", False)),
        dataset_id=str(payload.get("dataset_id", "eurusd_ps4_incremental_profile")),
    )


def _load_runner(path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise FollowerError(f"PS4 runner absent: {path}")
    name = f"feature_selection_ps4_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FollowerError(f"cannot import PS4 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for symbol in ("execute_unit", "classify_unit"):
        if not callable(getattr(module, symbol, None)):
            raise FollowerError(f"PS4 runner lacks callable {symbol}")
    return module


def _manifest_paths(batch_dir: Path) -> Iterable[Path]:
    direct = batch_dir / "batch_manifest.json"
    if direct.is_file() and not direct.is_symlink():
        yield direct
        return
    if batch_dir.is_dir() and not batch_dir.is_symlink():
        yield from sorted(batch_dir.rglob("batch_manifest.json"))


def load_batches(settings: Settings) -> tuple[dict[str, Batch], dict[str, str]]:
    by_feature: dict[str, Batch] = {}
    rejected: dict[str, str] = {}
    seen_manifests: set[Path] = set()
    for directory in settings.ps2_batch_dirs:
        for manifest_path in _manifest_paths(directory):
            resolved = manifest_path.resolve()
            if resolved in seen_manifests:
                continue
            seen_manifests.add(resolved)
            try:
                payload = _read_json(resolved)
                features = payload.get("features") if isinstance(payload, dict) else None
                series = payload.get("series") if isinstance(payload, dict) else None
                batch_id = payload.get("batch_id") if isinstance(payload, dict) else None
                if (
                    payload.get("schema") != "ps2_batch.v1"
                    or not isinstance(batch_id, str)
                    or not batch_id
                    or not isinstance(features, list)
                    or not features
                    or len(features) != len(set(features))
                    or not all(isinstance(item, str) and item for item in features)
                    or not isinstance(series, dict)
                    or series.get("file") != "series.npz"
                    or not _is_sha256(series.get("sha256"))
                ):
                    raise FollowerError("invalid PS2 batch manifest")
                series_path = resolved.parent / "series.npz"
                if series_path.is_symlink() or not series_path.is_file():
                    raise FollowerError("declared series.npz is absent")
                actual_series = _sha256_file(series_path)
                if actual_series != series["sha256"]:
                    raise FollowerError("series.npz digest mismatch")
                batch = Batch(
                    batch_id=batch_id,
                    manifest_path=resolved,
                    manifest_sha256=_sha256_file(resolved),
                    series_path=series_path,
                    series_sha256=actual_series,
                    features=tuple(features),
                )
                for feature in features:
                    if feature in by_feature and by_feature[feature] != batch:
                        rejected[feature] = "FEATURE_DECLARED_BY_MULTIPLE_PS2_BATCHES"
                        by_feature.pop(feature, None)
                    elif feature not in rejected:
                        by_feature[feature] = batch
            except (FollowerError, OSError) as error:
                rejected[f"batch:{resolved}"] = str(error)
    return by_feature, rejected


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_terminals(settings: Settings) -> tuple[list[Terminal], dict[str, str], int]:
    terminals: list[Terminal] = []
    rejected: dict[str, str] = {}
    discovered = 0
    seen: set[Path] = set()
    for root in settings.result_roots:
        if not root.is_dir() or root.is_symlink():
            rejected[f"root:{root}"] = "RESULT_ROOT_ABSENT"
            continue
        for manifest_path in sorted(root.rglob("run_manifest.json")):
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            resolved = manifest_path.resolve()
            if resolved in seen or not _inside(resolved, root):
                continue
            seen.add(resolved)
            discovered += 1
            key = f"manifest:{resolved}"
            try:
                payload = _read_json(resolved)
                features = payload.get("features") if isinstance(payload, dict) else None
                result_name = payload.get("results_file", "results.jsonl")
                if (
                    payload.get("schema") != "ut_pilot_run.v1"
                    or payload.get("status") != "COMPLETED"
                    or not isinstance(features, list)
                    or len(features) != 1
                    or not isinstance(features[0], str)
                    or not features[0]
                    or not isinstance(result_name, str)
                    or result_name != Path(result_name).name
                    or not _is_sha256(payload.get("results_sha256"))
                ):
                    raise FollowerError("terminal identity or COMPLETED status is invalid")
                results_path = resolved.parent / result_name
                if results_path.is_symlink() or not results_path.is_file():
                    raise FollowerError("terminal results are absent")
                actual = _sha256_file(results_path)
                if actual != payload["results_sha256"]:
                    raise FollowerError("terminal results digest mismatch")
                terminals.append(Terminal(features[0], resolved, payload, results_path, actual))
            except (FollowerError, OSError) as error:
                rejected[key] = str(error)
    return terminals, rejected, discovered


def _bind_terminals(
    terminals: list[Terminal], batches: dict[str, Batch]
) -> tuple[dict[str, tuple[Terminal, Batch]], dict[str, str]]:
    candidates: dict[str, list[tuple[Terminal, Batch]]] = {}
    rejected: dict[str, str] = {}
    for terminal in terminals:
        feature = terminal.feature_id
        batch = batches.get(feature)
        if batch is None:
            rejected[feature] = "FEATURE_ABSENT_FROM_PS2_BATCHES"
            continue
        manifest = terminal.manifest
        if manifest.get("batch_id") != batch.batch_id:
            rejected[feature] = "TERMINAL_BATCH_ID_MISMATCH"
            continue
        if manifest.get("batch_manifest_sha256") != batch.manifest_sha256:
            rejected[feature] = "TERMINAL_BATCH_MANIFEST_DIGEST_MISMATCH"
            continue
        if manifest.get("series_sha256") != batch.series_sha256:
            rejected[feature] = "TERMINAL_SERIES_DIGEST_MISMATCH"
            continue
        candidates.setdefault(feature, []).append((terminal, batch))
    accepted: dict[str, tuple[Terminal, Batch]] = {}
    for feature, rows in sorted(candidates.items()):
        identities = {
            (
                row[0].results_sha256,
                row[0].manifest.get("seed"),
                row[0].manifest.get("code_commit"),
                tuple(row[0].manifest.get("families") or []),
            )
            for row in rows
        }
        if len(identities) != 1:
            rejected[feature] = "CONFLICTING_COMPLETED_TERMINALS"
            continue
        accepted[feature] = min(rows, key=lambda row: str(row[0].manifest_path))
    return accepted, rejected


def _acceptance_from_unit(path: Path) -> dict[str, Any] | None:
    try:
        payload = _read_json(path)
    except FollowerError:
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "unit": path.name,
        "unit_sha256": _sha256_file(path),
        "feature_id": payload.get("feature_id"),
        "fold_id": payload.get("fold_id"),
        "source_sha256": payload.get("source_sha256"),
        "terminal_identity": payload.get("terminal_identity"),
    }


def _terminal_identity(
    settings: Settings, runner: ModuleType, terminal: Terminal, batch: Batch
) -> dict[str, Any]:
    loader = getattr(runner, "load_terminal_identity", None)
    if callable(loader):
        return loader(
            _runner_terminal_manifest(settings, terminal),
            terminal.feature_id,
            batch.series_sha256,
        )
    return {"results_sha256": terminal.results_sha256}


def _runner_terminal_manifest(settings: Settings, terminal: Terminal) -> Path:
    """Bind legacy terminal pairs without rewriting their retained evidence."""
    if terminal.manifest.get("results_file") == terminal.results_path.name:
        return terminal.manifest_path
    identity = _sha256_file(terminal.manifest_path)
    destination = settings.ps4_output_dir / "terminal_bindings" / identity
    results_path = destination / terminal.results_path.name
    manifest_path = destination / "run_manifest.json"
    if not results_path.is_file() or _sha256_file(results_path) != terminal.results_sha256:
        _atomic_bytes(results_path, terminal.results_path.read_bytes())
    payload = dict(terminal.manifest)
    payload["results_file"] = terminal.results_path.name
    payload["source_terminal_manifest_sha256"] = identity
    if not manifest_path.is_file() or _read_json(manifest_path) != payload:
        _atomic_json(manifest_path, payload)
    return manifest_path


def _valid_unit(
    runner: ModuleType,
    path: Path,
    feature: str,
    fold: str,
    *,
    expected_source: dict[str, str] | None = None,
    expected_terminal: dict[str, Any] | None = None,
) -> bool:
    acceptance = _acceptance_from_unit(path)
    if acceptance is None:
        return False
    try:
        return runner.classify_unit(
            path,
            expected_feature_id=feature,
            expected_fold_id=fold,
            expected_source_sha256=expected_source,
            expected_terminal_identity=expected_terminal,
            expected_acceptance=acceptance,
        ) == "MEASURED"
    except Exception:
        return False


def _unit_path(settings: Settings, runner: ModuleType, feature: str, fold: str) -> Path:
    helper = getattr(runner, "unit_path", None)
    if callable(helper):
        return Path(helper(settings.ps4_output_dir, feature, fold))
    safe = feature.replace("/", "_")
    return settings.ps4_units_dir / f"{safe}__{fold}.json"


def _profile_feature(
    settings: Settings,
    runner: ModuleType,
    feature: str,
    terminal: Terminal,
    batch: Batch,
    folds: tuple[str, ...],
) -> tuple[bool, str]:
    expected_source = {
        "series.npz": batch.series_sha256,
        "folds.json": _sha256_file(settings.folds_file),
    }
    try:
        expected_terminal = _terminal_identity(settings, runner, terminal, batch)
    except Exception as error:
        return False, f"TERMINAL_IDENTITY_ERROR: {type(error).__name__}: {error}"
    for fold in folds:
        destination = _unit_path(settings, runner, feature, fold)
        if _valid_unit(
            runner,
            destination,
            feature,
            fold,
            expected_source=expected_source,
            expected_terminal=expected_terminal,
        ):
            continue
        expected_acceptance = _acceptance_from_unit(destination)
        try:
            result = runner.execute_unit(
                batch.series_path,
                settings.folds_file,
                feature,
                fold,
                settings.ps4_output_dir,
                split="train",
                dataset_id=settings.dataset_id,
                canonical_bounds=settings.canonical_bounds,
                expected_digest=batch.series_sha256,
                terminal_manifest=_runner_terminal_manifest(settings, terminal),
                expected_acceptance=expected_acceptance,
            )
        except Exception as error:
            return False, f"PS4_UNIT_ERROR[{fold}]: {type(error).__name__}: {error}"
        if not isinstance(result, dict) or result.get("unit_status") != "MEASURED":
            return False, f"PS4_UNIT_NOT_MEASURED[{fold}]"
        if not _valid_unit(
            runner,
            destination,
            feature,
            fold,
            expected_source=expected_source,
            expected_terminal=expected_terminal,
        ):
            return False, f"PS4_UNIT_VERIFICATION_FAILED[{fold}]"
    return True, ""


def regenerate_accepted_index(
    settings: Settings,
    runner: ModuleType,
    folds: tuple[str, ...],
    accepted_terminals: dict[str, tuple[Terminal, Batch]],
) -> tuple[list[str], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    invalid: list[dict[str, Any]] = []
    settings.ps4_units_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(settings.ps4_units_dir.glob("*.json")):
        acceptance = _acceptance_from_unit(path)
        feature = acceptance.get("feature_id") if acceptance else None
        fold = acceptance.get("fold_id") if acceptance else None
        terminal_and_batch = accepted_terminals.get(feature) if isinstance(feature, str) else None
        expected_source = None
        expected_terminal = None
        if terminal_and_batch is not None:
            terminal, batch = terminal_and_batch
            expected_source = {
                "series.npz": batch.series_sha256,
                "folds.json": _sha256_file(settings.folds_file),
            }
            try:
                expected_terminal = _terminal_identity(settings, runner, terminal, batch)
            except Exception:
                terminal_and_batch = None
        if (
            not isinstance(feature, str)
            or fold not in folds
            or terminal_and_batch is None
            or not _valid_unit(
                runner,
                path,
                feature,
                fold,
                expected_source=expected_source,
                expected_terminal=expected_terminal,
            )
        ):
            invalid.append({"unit": path.name, "reason": "UNIT_NOT_VERIFIED"})
            continue
        grouped.setdefault(feature, {})[fold] = (path, acceptance)
    complete = sorted(feature for feature, rows in grouped.items() if set(rows) == set(folds))
    units: list[dict[str, Any]] = []
    for feature in complete:
        for fold in folds:
            units.append(grouped[feature][fold][1])
    _atomic_json(
        settings.accepted_units_path,
        {"schema": ACCEPTED_INDEX_SCHEMA, "units": units},
    )
    return complete, invalid


def run_cycle(settings: Settings, runner: ModuleType, *, started_at: str) -> dict[str, Any]:
    cycle_started = _utc_now()
    folds = tuple(getattr(runner, "INNER_FOLDS", DEFAULT_FOLDS))
    if folds != DEFAULT_FOLDS:
        raise FollowerError("PS4 runner must expose the five authenticated inner folds")
    if settings.folds_file.is_symlink() or not settings.folds_file.is_file():
        raise FollowerError(f"folds file absent: {settings.folds_file}")
    batches, batch_rejections = load_batches(settings)
    terminals, terminal_rejections, discovered_count = discover_terminals(settings)
    accepted, binding_rejections = _bind_terminals(terminals, batches)
    failures = {**batch_rejections, **terminal_rejections, **binding_rejections}

    for feature, (terminal, batch) in sorted(accepted.items()):
        ok, reason = _profile_feature(settings, runner, feature, terminal, batch, folds)
        if not ok:
            failures[feature] = reason

    profiled, invalid_units = regenerate_accepted_index(settings, runner, folds, accepted)
    profiled_set = set(profiled)
    accepted_features = sorted(accepted)
    pending = sorted(feature for feature in accepted_features if feature not in profiled_set)
    status = {
        "schema": STATUS_SCHEMA,
        "started_at": started_at,
        "cycle_started_at": cycle_started,
        "updated_at": _utc_now(),
        "counts": {
            "discovered": discovered_count,
            "accepted": len(accepted_features),
            "profiled": len(profiled),
            "rejected": len(failures) + len(invalid_units),
            "pending": len(pending),
        },
        "discovered": sorted(str(item.manifest_path) for item in terminals),
        "accepted": accepted_features,
        "profiled": profiled,
        "rejected": [
            {"subject": subject, "reason": reason}
            for subject, reason in sorted(failures.items())
        ] + invalid_units,
        "pending": pending,
        "safety": {
            "split": "train",
            "targets_read": False,
            "validation_read": False,
            "test_read": False,
        },
    }
    _atomic_json(settings.status_path, status)
    return status


def run(settings: Settings, *, once: bool) -> int:
    settings.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another follower holds {settings.lock_path}", file=sys.stderr)
            return 75
        runner = _load_runner(settings.ps4_runner)
        started_at = _utc_now()
        while True:
            try:
                status = run_cycle(settings, runner, started_at=started_at)
                print(json.dumps(status["counts"], sort_keys=True), flush=True)
            except Exception as error:
                failure = {
                    "schema": STATUS_SCHEMA,
                    "started_at": started_at,
                    "updated_at": _utc_now(),
                    "fatal_error": f"{type(error).__name__}: {error}",
                }
                _atomic_json(settings.status_path, failure)
                if once:
                    raise
            if once:
                return 0
            time.sleep(settings.poll_interval_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="run one discovery/profile cycle")
    args = parser.parse_args(argv)
    try:
        return run(load_settings(args.config), once=args.once)
    except FollowerError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
