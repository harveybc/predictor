"""Strict parsing for evidence documents.

`json.loads` is a convenience, not an evidence boundary. It keeps
the last of a pair of duplicate keys, happily materialises `NaN`
and `Infinity`, and says nothing about whether a field that looks
like a digest is one. A manifest, a submission and a review
record are documents on which a scientific claim rests, so they
are read here instead: duplicate keys refuse, non-finite
constants refuse, digests must be canonical, timestamps must be
canonical UTC and may not be in the future, and the whole file is
read rather than a prefix.

Nothing here is a security control. It is the difference between
"the document said so" and "the document could only have said
so".
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# RFC3339 in UTC, with an explicit Z or +00:00 offset.
CANONICAL_TS = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?"
    r"(Z|\+00:00)$")

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024


class StrictParseRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _no_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise StrictParseRefusal(
                f"duplicate JSON key {key!r} — a document that "
                "says two things cannot be evidence for either")
        seen[key] = value
    return seen


def _no_constants(token):
    raise StrictParseRefusal(
        f"non-finite JSON constant {token!r} — NaN and Infinity "
        "are not measurements")


def strict_loads(raw: bytes, *, what: str) -> dict:
    """Parse bytes with every permissive behaviour disabled."""
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise StrictParseRefusal(
            f"{what} exceeds {MAX_DOCUMENT_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictParseRefusal(
            f"{what} is not valid UTF-8") from exc
    try:
        doc = json.loads(text,
                         object_pairs_hook=_no_duplicate_keys,
                         parse_constant=_no_constants)
    except json.JSONDecodeError as exc:
        raise StrictParseRefusal(
            f"{what} is not strict JSON: {exc.msg} "
            f"(line {exc.lineno})") from exc
    if not isinstance(doc, dict):
        raise StrictParseRefusal(
            f"{what} is a {type(doc).__name__}, not an object")
    _reject_non_finite(doc, what, "")
    return doc


def strict_load_file(path: str | Path, *, what: str) -> dict:
    """Read a WHOLE file through one descriptor and parse it."""
    p = Path(path)
    if not p.is_file():
        raise StrictParseRefusal(f"{what} is absent at {p.name}")
    fd = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        size = os.fstat(fd).st_size
        if size > MAX_DOCUMENT_BYTES:
            raise StrictParseRefusal(
                f"{what} is {size} bytes, over the limit")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise StrictParseRefusal(
            f"{what} changed size while being read")
    return strict_loads(raw, what=what)


def _reject_non_finite(node, what: str, path: str) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _reject_non_finite(v, what,
                               f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _reject_non_finite(v, what, f"{path}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        raise StrictParseRefusal(
            f"{what}: {path or 'value'} is {node!r} — a "
            "non-finite number is not a measurement")


# --------------------------------------------------------------
# typed field checks
# --------------------------------------------------------------

def require_keys(doc: dict, exact: set, *, what: str) -> None:
    got = set(doc)
    if got != exact:
        extra = sorted(got - exact)
        missing = sorted(exact - got)
        raise StrictParseRefusal(
            f"{what} keys are not the exact schema "
            f"(missing: {missing}, unexpected: {extra})")


def require_sha256(value, *, what: str) -> str:
    if not isinstance(value, str):
        raise StrictParseRefusal(
            f"{what} is a {type(value).__name__}, not a digest "
            "string")
    if not CANONICAL_SHA256.match(value):
        raise StrictParseRefusal(
            f"{what} is not a canonical SHA-256 (64 lowercase "
            f"hex): {value[:24]!r}")
    return value


def require_timestamp(value, *, what: str,
                      now: datetime | None = None) -> datetime:
    if not isinstance(value, str):
        raise StrictParseRefusal(
            f"{what} is a {type(value).__name__}, not a "
            "timestamp string")
    if not CANONICAL_TS.match(value):
        raise StrictParseRefusal(
            f"{what} is not a canonical RFC3339 UTC timestamp: "
            f"{value!r}")
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    now = now or datetime.now(timezone.utc)
    if ts > now:
        raise StrictParseRefusal(
            f"{what} is in the FUTURE ({value}) — a document "
            "cannot have been reviewed after now")
    return ts


def require_str(value, *, what: str,
                allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise StrictParseRefusal(
            f"{what} is a {type(value).__name__}, not a string")
    if not allow_empty and not value.strip():
        raise StrictParseRefusal(f"{what} is empty")
    return value


def require_number(value, *, what: str) -> float:
    # bool is a subclass of int in Python; a flag is not a
    # measurement and must not pass a numeric field
    if isinstance(value, bool) or not isinstance(
            value, (int, float)):
        raise StrictParseRefusal(
            f"{what} is a {type(value).__name__}, not a number")
    if not math.isfinite(float(value)):
        raise StrictParseRefusal(f"{what} is not finite")
    return float(value)


def require_unique(values, *, what: str) -> list:
    seen, dupes = set(), []
    for v in values:
        if v in seen:
            dupes.append(v)
        seen.add(v)
    if dupes:
        raise StrictParseRefusal(
            f"{what} contains duplicates: {sorted(set(dupes))}")
    return list(values)


def require_chronology(pairs, *, what: str) -> None:
    """pairs: [(label, datetime)] in the order they must occur."""
    for (an, a), (bn, b) in zip(pairs, pairs[1:]):
        if a > b:
            raise StrictParseRefusal(
                f"{what}: {an} ({a.isoformat()}) is after {bn} "
                f"({b.isoformat()}) — the chronology is "
                "impossible")
