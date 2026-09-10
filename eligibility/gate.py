"""The `PUBLICLY_ELIGIBLE` gate: one manifest, one consumer.

Every repository in the program asks the same question before it
acts — *may this variable or operator be used, for this purpose?*
— and until now each answered it its own way: by plugin name, by
directory membership, by a producer's own label, or by a fallback
that silently readmitted whatever the filter had just removed.

This module is the single consumer. A subject is usable only if a
reviewed manifest entry says so, for the scope being asked, with
every binding present and every digest matching the bytes
actually in hand. There is no default-allow path: an unknown
subject, a stale manifest, a renamed plugin or a swapped piece of
evidence all refuse.

The manifest is a REVIEWED artifact. This module never writes
one, never upgrades a decision, and never infers a missing
binding — a gate that can issue its own permissions is not a
gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SCHEMA = "crispdm.eligibility_manifest.v1"

SUBJECT_KINDS = ("variable", "operator")
DECISIONS = ("PUBLICLY_ELIGIBLE", "REJECTED", "INCONCLUSIVE",
             "LAB_CALIBRATED", "DOMAIN_REVALIDATED")
POSITIVE_DECISIONS = ("PUBLICLY_ELIGIBLE",)

# Every binding the order requires. A missing one is a refusal,
# never a default.
REQUIRED_ENTRY_FIELDS = (
    "subject_kind", "subject_id", "version",
    "io_schema", "unit", "temporal_availability",
    "fit_scope", "incremental_state_policy", "parameters",
    "digests", "measured_cost",
    "decision", "decision_scope", "decision_reason",
    "reviewer", "reviewed_at",
)
REQUIRED_DIGEST_KINDS = ("data", "code", "partitions", "evidence")
REQUIRED_AVAILABILITY_FIELDS = ("event_time", "available_time")

FIT_SCOPES = ("TRAIN_ONLY", "TRAIN_PLUS_CALIBRATION",
              "NO_FIT_REQUIRED")


class EligibilityRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _self_sha(doc: dict, key: str = "manifest_sha256") -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parse_ts(value: str, what: str) -> datetime:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z",
                                                       "+00:00"))
    except ValueError as exc:
        raise EligibilityRefusal(
            f"{what} is not an ISO-8601 timestamp: {value!r}"
        ) from exc
    if ts.tzinfo is None:
        raise EligibilityRefusal(
            f"{what} has no timezone: {value!r}")
    return ts


# --------------------------------------------------------------
# loading: absent, stale or altered manifests refuse
# --------------------------------------------------------------

def load_manifest(path: str | Path, *,
                  expected_sha256: str | None = None,
                  max_age_days: float | None = None,
                  now: datetime | None = None) -> dict:
    """Read a reviewed manifest, or refuse.

    `expected_sha256` is how a caller pins the exact manifest it
    was built against: a different manifest with a valid internal
    digest is still the wrong manifest.
    """
    p = Path(path)
    if not p.is_file():
        raise EligibilityRefusal(
            f"eligibility manifest is ABSENT at {p.name} — "
            "without a reviewed manifest nothing is eligible")
    # C4: duplicate keys, non-finite constants and partial
    # reads are refused by the strict reader, not tolerated by
    # json.loads.
    from eligibility.strict import strict_load_file
    doc = strict_load_file(p, what="eligibility manifest")
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise EligibilityRefusal(
            f"eligibility manifest schema is "
            f"{doc.get('schema')!r}, not {MANIFEST_SCHEMA!r}")
    declared = doc.get("manifest_sha256")
    if not declared:
        raise EligibilityRefusal(
            "eligibility manifest carries no self digest")
    if _self_sha(doc) != declared:
        raise EligibilityRefusal(
            "eligibility manifest self digest does not "
            "re-derive — the bytes were altered after review")
    if expected_sha256 and declared != expected_sha256:
        raise EligibilityRefusal(
            "eligibility manifest digest is not the pinned one — "
            "a different manifest is not the reviewed manifest")
    from eligibility.strict import (require_sha256,
                                    require_timestamp)
    require_sha256(declared, what="manifest_sha256")
    if expected_sha256 is not None:
        require_sha256(expected_sha256,
                       what="pinned manifest digest")
    issued = require_timestamp(doc.get("issued_at", ""),
                               what="issued_at", now=now)
    if max_age_days is not None:
        now = now or datetime.now(timezone.utc)
        age = (now - issued).total_seconds() / 86400.0
        if age > max_age_days:
            raise EligibilityRefusal(
                f"eligibility manifest is STALE: issued "
                f"{age:.1f} days ago, limit {max_age_days} — a "
                "stale review is not a review")
    for entry in doc.get("entries", []):
        _validate_entry(entry)
    ids = [e["subject_id"] for e in doc.get("entries", [])]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise EligibilityRefusal(
            f"eligibility manifest declares a subject twice: "
            f"{dup} — one subject, one decision")
    return doc


def _validate_entry(entry: dict) -> None:
    sid = entry.get("subject_id", "<no id>")
    missing = [f for f in REQUIRED_ENTRY_FIELDS
               if f not in entry]
    if missing:
        raise EligibilityRefusal(
            f"{sid}: manifest entry is missing required "
            f"bindings {missing} — an absent binding is never a "
            "default")
    if entry["subject_kind"] not in SUBJECT_KINDS:
        raise EligibilityRefusal(
            f"{sid}: unknown subject kind "
            f"{entry['subject_kind']!r}")
    if entry["decision"] not in DECISIONS:
        raise EligibilityRefusal(
            f"{sid}: unknown decision {entry['decision']!r}")
    if entry["fit_scope"] not in FIT_SCOPES:
        raise EligibilityRefusal(
            f"{sid}: unknown fit scope {entry['fit_scope']!r}")
    dig = entry["digests"]
    missing_d = [k for k in REQUIRED_DIGEST_KINDS if k not in dig]
    if missing_d:
        raise EligibilityRefusal(
            f"{sid}: entry declares no {missing_d} digest")
    # C4: a field named "digest" must BE one
    from eligibility.strict import (require_sha256,
                                    require_str,
                                    require_timestamp)
    for kind in REQUIRED_DIGEST_KINDS:
        require_sha256(dig[kind],
                       what=f"{sid}: digests.{kind}")
    require_str(entry["subject_id"], what=f"{sid}: subject_id")
    require_str(entry["version"], what=f"{sid}: version")
    avail = entry["temporal_availability"]
    missing_a = [k for k in REQUIRED_AVAILABILITY_FIELDS
                 if k not in avail]
    if missing_a:
        raise EligibilityRefusal(
            f"{sid}: temporal availability is missing "
            f"{missing_a} — event time and available time are "
            "never the same field by default")
    if not str(entry["decision_reason"]).strip():
        raise EligibilityRefusal(
            f"{sid}: the reviewer recorded no reason")
    if not str(entry["decision_scope"]).strip():
        raise EligibilityRefusal(
            f"{sid}: the reviewer recorded no scope")
    require_timestamp(entry["reviewed_at"],
                      what=f"{sid}: reviewed_at")


# --------------------------------------------------------------
# asking the gate
# --------------------------------------------------------------

def _entry_index(manifest: dict) -> dict:
    return {e["subject_id"]: e for e in manifest.get("entries",
                                                     [])}


def require_eligible(manifest: dict, subject_id: str, *,
                     scope: str,
                     subject_kind: str | None = None,
                     version: str | None = None,
                     evidence_digest: str | None = None,
                     code_digest: str | None = None,
                     data_digest: str | None = None,
                     partitions_digest: str | None = None
                     ) -> dict:
    """Return the reviewed entry, or refuse.

    There is deliberately no `default` parameter and no boolean
    return: a caller cannot accidentally treat "no decision" as
    permission.
    """
    entry = _entry_index(manifest).get(subject_id)
    if entry is None:
        raise EligibilityRefusal(
            f"{subject_id}: no reviewed manifest entry — an "
            "unlisted subject is not eligible by omission")
    if subject_kind and entry["subject_kind"] != subject_kind:
        raise EligibilityRefusal(
            f"{subject_id}: reviewed as "
            f"{entry['subject_kind']}, requested as "
            f"{subject_kind}")
    if version is not None and entry["version"] != version:
        raise EligibilityRefusal(
            f"{subject_id}: reviewed version "
            f"{entry['version']!r}, presented {version!r} — a "
            "version bump is a new review")
    if entry["decision"] not in POSITIVE_DECISIONS:
        raise EligibilityRefusal(
            f"{subject_id}: decision is {entry['decision']!r} "
            f"({entry['decision_reason']})")
    if scope not in _scopes(entry):
        raise EligibilityRefusal(
            f"{subject_id}: reviewed for scope "
            f"{entry['decision_scope']!r}, used for {scope!r} — "
            "eligibility is scoped, never global")
    if code_digest is not None and \
            entry["digests"]["code"] != code_digest:
        raise EligibilityRefusal(
            f"{subject_id}: the code in hand does not match the "
            "reviewed code digest — a positive label never "
            "transfers to different code")
    if evidence_digest is not None and \
            entry["digests"]["evidence"] != evidence_digest:
        raise EligibilityRefusal(
            f"{subject_id}: the evidence in hand does not match "
            "the reviewed evidence digest — replacing the "
            "evidence under a positive label is exactly what "
            "this gate exists to stop")
    # C2/C3: the data and partition bindings are compared at the
    # POINT OF USE. Without this, an id was the only thing that
    # travelled and the reviewed bytes were never checked.
    if data_digest is not None and \
            entry["digests"]["data"] != data_digest:
        raise EligibilityRefusal(
            f"{subject_id}: the data in hand does not match the "
            "reviewed data digest — a decision about other "
            "bytes is not a decision about this run")
    if partitions_digest is not None and \
            entry["digests"]["partitions"] != partitions_digest:
        raise EligibilityRefusal(
            f"{subject_id}: the partition layout in hand does "
            "not match the reviewed partitions digest — a "
            "re-cut split is a new review")
    return entry


def _scopes(entry: dict) -> tuple:
    scope = entry["decision_scope"]
    if isinstance(scope, str):
        return (scope,)
    return tuple(scope)


def is_eligible(manifest: dict, subject_id: str, *,
                scope: str, **kw) -> bool:
    """Boolean form for filtering. Callers that ACT on a subject
    must use `require_eligible`, so the refusal reason survives."""
    try:
        require_eligible(manifest, subject_id, scope=scope, **kw)
    except EligibilityRefusal:
        return False
    return True


def eligible_universe(manifest: dict, *, scope: str,
                      subject_kind: str = "variable",
                      group: list[str] | None = None) -> list[str]:
    """The ordered universe for a scope — the ONLY way to build
    one.

    `group` restricts the universe to an explicit membership list.
    An EMPTY group yields an EMPTY universe: a group that selects
    nothing must never fall back to selecting everything.
    """
    if group is not None:
        allowed = set(group)
        if not allowed:
            return []
    else:
        allowed = None
    out = []
    for e in manifest.get("entries", []):
        if e["subject_kind"] != subject_kind:
            continue
        if allowed is not None and e["subject_id"] not in allowed:
            continue
        if e["decision"] not in POSITIVE_DECISIONS:
            continue
        if scope not in _scopes(e):
            continue
        out.append(e["subject_id"])
    # deterministic: the same manifest yields the same ordered
    # universe in any process, on any host
    return sorted(out)


def filter_to_eligible(manifest: dict, candidates: list[str], *,
                       scope: str,
                       subject_kind: str = "variable") -> list[str]:
    """Intersect a proposed set with the reviewed universe.

    The result is always a SUBSET of `candidates` and of the
    universe — a filter can only ever remove.
    """
    universe = set(eligible_universe(
        manifest, scope=scope, subject_kind=subject_kind))
    kept = [c for c in candidates if c in universe]
    assert set(kept) <= set(candidates)
    return sorted(kept)


def require_operator(manifest: dict, *, operator_id: str,
                     version: str, code_digest: str,
                     scope: str,
                     plugin_name: str | None = None) -> dict:
    """Operators are admitted by reviewed id + version + code
    bytes.

    `plugin_name` is accepted only to be checked against the
    reviewed entry: a plugin that registers itself under an
    eligible operator's name does not thereby become that
    operator.
    """
    entry = require_eligible(manifest, operator_id, scope=scope,
                             subject_kind="operator",
                             version=version,
                             code_digest=code_digest)
    if plugin_name is not None:
        declared = entry["parameters"].get("plugin_name")
        if declared is not None and declared != plugin_name:
            raise EligibilityRefusal(
                f"{operator_id}: reviewed plugin name "
                f"{declared!r}, presented {plugin_name!r} — an "
                "operator never enters by borrowing a name")
    return entry


def manifest_fingerprint(manifest: dict) -> str:
    """What a producer records so a reader can prove which
    reviewed manifest governed a run."""
    return manifest["manifest_sha256"]
