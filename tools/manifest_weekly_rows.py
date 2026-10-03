"""Authenticated local-manifest rows for BUSINESS weekly training.

The provider is deliberately a storage adapter, not a data-preparation stage.
It authenticates and parses an immutable JSONL or CSV manifest, then returns
the event-time candidates for a requested :class:`WeekSpec`.  Availability,
support, purge, and fit/validation membership remain the responsibility of
``tools.business_asof_window``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Mapping, Tuple

from tools.business_asof_window import AsOfRow
from tools.business_weekly_protocol import WeekSpec


_FIELDS = (
    "record_id",
    "event_time",
    "available_time",
    "target_available_time",
    "row_digest",
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NONFINITE_TEXT = frozenset({"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
_READ_SIZE = 1024 * 1024


class ManifestContractError(ValueError):
    """Raised when manifest custody or schema authentication fails."""


@dataclass(frozen=True)
class _FileIdentity:
    """Stable regular-file identity captured at provider construction."""

    device: int
    inode: int


def _require_digest(name: str, value: object) -> str:
    """Return one canonical SHA-256 identity and reject weak aliases."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ManifestContractError(f"{name} must be a sha256 digest string")
    if not _DIGEST_RE.fullmatch(value):
        raise ManifestContractError(f"{name} must match sha256:<64 lowercase hex>")
    return value


def _require_text(name: str, value: object) -> str:
    """Return a non-empty finite text scalar."""

    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ManifestContractError(f"{name} must be a non-empty string, not bool")
    if value.strip().lower() in _NONFINITE_TEXT:
        raise ManifestContractError(f"{name} contains a non-finite value")
    return value


def _parse_timestamp(name: str, value: object) -> datetime:
    """Parse an explicit timezone-aware timestamp and normalize it to UTC."""

    text = _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ManifestContractError(f"{name} is a malformed timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestContractError(f"{name} must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _reject_json_constant(value: str) -> object:
    """Reject JSON's non-standard NaN and Infinity constants."""

    raise ManifestContractError(f"manifest contains non-finite JSON value: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build an object while rejecting duplicate JSON keys."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestContractError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_record(raw: Mapping[str, object], *, line_number: int) -> AsOfRow:
    """Parse one exact manifest record into the shared immutable row type."""

    keys = tuple(raw.keys())
    missing = [field for field in _FIELDS if field not in raw]
    if missing:
        availability = {"available_time", "target_available_time"}
        if availability.intersection(missing):
            raise ManifestContractError(
                f"line {line_number} has missing explicit availability: {missing}"
            )
        raise ManifestContractError(f"line {line_number} has missing fields: {missing}")
    extras = [field for field in keys if field not in _FIELDS]
    if extras:
        raise ManifestContractError(f"line {line_number} has unexpected fields: {extras}")
    for field in ("available_time", "target_available_time"):
        if raw[field] is None:
            raise ManifestContractError(
                f"line {line_number} has missing explicit availability: {field}"
            )

    record_id = _require_text("record_id", raw["record_id"])
    event_time = _parse_timestamp("event_time", raw["event_time"])
    available_time = _parse_timestamp("available_time", raw["available_time"])
    target_available_time = _parse_timestamp(
        "target_available_time", raw["target_available_time"]
    )
    row_digest = _require_digest("row_digest", raw["row_digest"])
    return AsOfRow(
        record_id=record_id,
        event_time=event_time,
        available_time=available_time,
        target_available_time=target_available_time,
        row_digest=row_digest,
    )


def _parse_jsonl(payload: bytes) -> Tuple[AsOfRow, ...]:
    """Parse strict UTF-8 JSON Lines without accepting blank records."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestContractError("manifest must be valid UTF-8") from exc
    if not text:
        raise ManifestContractError("manifest must contain at least one record")

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ManifestContractError(f"line {line_number} is blank")
        try:
            raw = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except ManifestContractError:
            raise
        except (json.JSONDecodeError, TypeError) as exc:
            raise ManifestContractError(f"line {line_number} is malformed JSON") from exc
        if not isinstance(raw, dict):
            raise ManifestContractError(f"line {line_number} must be a JSON object")
        rows.append(_parse_record(raw, line_number=line_number))
    return tuple(rows)


def _parse_csv(payload: bytes) -> Tuple[AsOfRow, ...]:
    """Parse strict UTF-8 CSV with one exact, non-duplicated header."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestContractError("manifest must be valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != _FIELDS:
        raise ManifestContractError(f"CSV header must be exactly {_FIELDS!r}")

    rows = []
    try:
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ManifestContractError(f"line {line_number} has extra CSV columns")
            rows.append(_parse_record(raw, line_number=line_number))
    except csv.Error as exc:
        raise ManifestContractError("manifest contains malformed CSV") from exc
    if not rows:
        raise ManifestContractError("manifest must contain at least one record")
    return tuple(rows)


_PARSERS: Mapping[str, Callable[[bytes], Tuple[AsOfRow, ...]]] = {
    ".csv": _parse_csv,
    ".jsonl": _parse_jsonl,
}


class ManifestWeeklyRows:
    """Callable, immutable manifest-backed ``RowProvider``.

    Args:
        path: Local ``.jsonl`` or ``.csv`` regular file.  Parent traversal and
            a symlink at the final path are rejected.
        source_digest: Exact ``sha256:<hex>`` digest of the complete file bytes.

    Calling the instance re-authenticates both file identity and bytes before
    returning an event-time slice.  It performs no fitting or learned data
    preparation and deliberately leaves all availability checks to the as-of
    resolver.
    """

    def __init__(self, path: os.PathLike[str] | str, source_digest: str) -> None:
        if isinstance(path, bool) or not isinstance(path, (str, os.PathLike)):
            raise ManifestContractError("path must be a local filesystem path")
        raw_path = Path(path)
        if ".." in raw_path.parts:
            raise ManifestContractError("parent path traversal is forbidden")
        suffix = raw_path.suffix.lower()
        if suffix not in _PARSERS:
            raise ManifestContractError("manifest extension must be .jsonl or .csv")

        self._path = raw_path.absolute()
        self._source_digest = _require_digest("source_digest", source_digest)
        payload, identity = self._read_authenticated(expected_identity=None)
        self._source_bytes = payload
        self._file_identity = identity
        parsed = _PARSERS[suffix](payload)
        self._rows = self._validate_unique_and_order(parsed)

    @property
    def source_digest(self) -> str:
        """Return the exact caller-supplied, verified whole-file digest."""

        return self._source_digest

    @property
    def path(self) -> Path:
        """Return the absolute path whose identity is pinned by this provider."""

        return self._path

    @staticmethod
    def _validate_unique_and_order(rows: Tuple[AsOfRow, ...]) -> Tuple[AsOfRow, ...]:
        """Reject duplicate identities and establish canonical row order."""

        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        for row in rows:
            if row.record_id in seen_ids:
                raise ManifestContractError(f"duplicate record_id: {row.record_id}")
            if row.row_digest in seen_digests:
                raise ManifestContractError(f"duplicate row_digest: {row.row_digest}")
            seen_ids.add(row.record_id)
            seen_digests.add(row.row_digest)
        return tuple(
            sorted(rows, key=lambda row: (row.event_time, row.record_id, row.row_digest))
        )

    def _read_authenticated(
        self,
        *,
        expected_identity: _FileIdentity | None,
    ) -> tuple[bytes, _FileIdentity]:
        """Read one regular file without following a final symlink."""

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ManifestContractError("manifest path is a symlink") from exc
            raise ManifestContractError(f"manifest cannot be opened: {exc.strerror}") from exc

        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ManifestContractError("manifest path must identify a regular file")
            identity = _FileIdentity(before.st_dev, before.st_ino)
            if expected_identity is not None and identity != expected_identity:
                raise ManifestContractError("manifest source changed after construction")

            chunks = []
            while True:
                chunk = os.read(descriptor, _READ_SIZE)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ManifestContractError("manifest source changed while being read")
        finally:
            os.close(descriptor)

        payload = b"".join(chunks)
        actual_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if actual_digest != self._source_digest:
            if expected_identity is None:
                raise ManifestContractError("manifest digest mismatch")
            raise ManifestContractError("manifest source changed after construction")
        if expected_identity is not None and payload != self._source_bytes:
            raise ManifestContractError("manifest source changed after construction")
        return payload, identity

    def __call__(self, week: WeekSpec) -> Tuple[AsOfRow, ...]:
        """Return deterministic candidates whose event lies in the week window."""

        if not isinstance(week, WeekSpec):
            raise ManifestContractError("week must be a WeekSpec")
        if week.fit_start is None:
            raise ManifestContractError("week.fit_start is required")
        self._read_authenticated(expected_identity=self._file_identity)
        return tuple(
            row
            for row in self._rows
            if week.fit_start <= row.event_time < week.cutoff
        )
