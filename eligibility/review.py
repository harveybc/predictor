"""Separated authority: a submission is not a decision.

The gate previously took its manifest path and its expected
digest from the same configuration that asked for the permission.
A candidate could therefore write a manifest, name itself the
reviewer, pin its own digest, and be told it was gated. The
label was real; the review was not.

Authority is split here into two documents that cannot both come
from the candidate:

  * a **submission** — what this repository produces. It states
    what was consumed, with which digests, under which code, and
    asks for a decision. It grants nothing, and the gate refuses
    it if it is ever presented as a decision.
  * a **review record** — what an external reviewer produces. It
    binds, by digest, the exact submission, the physical census,
    the code, the partitions, the scope and the date. Its
    location is a repository constant (overridable only by an
    environment variable set outside the run's config), so the
    configuration that requests a permission cannot also choose
    the document that grants it.

This module authors submissions and templates. It never authors,
installs or simulates a review record.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from eligibility.strict import (StrictParseRefusal,  # noqa: F401
                                require_chronology, require_keys,
                                require_sha256, require_str,
                                require_timestamp,
                                strict_load_file)

SUBMISSION_SCHEMA = "crispdm.eligibility_submission.v1"
REVIEW_RECORD_SCHEMA = "crispdm.eligibility_review_record.v1"

# The review record lives OUTSIDE any run configuration. The
# environment variable exists so an operator can relocate the
# authority store; a run's config cannot reach it.
REVIEW_RECORD_ENV = "CRISPDM_ELIGIBILITY_REVIEW_RECORD"
DEFAULT_REVIEW_RECORD = (
    Path.home() / ".local/share/predictor/eligibility_authority"
    / "MUSASHI_ELIGIBILITY_REVIEW_RECORD.json")

REVIEW_DECISION = "ELIGIBILITY_MANIFEST_APPROVED"

_SUBMISSION_KEYS = {
    "schema", "submitted_at", "submitter", "scope",
    "manifest_sha256", "census_sha256", "code_digest",
    "partitions_digest", "data_digest", "schema_digest",
    "dataset_id", "subject_ids", "grants_nothing",
    "submission_sha256",
}
_RECORD_KEYS = {
    "schema", "reviewed_at", "reviewer", "scope",
    "reviewed_submission_sha256", "reviewed_manifest_sha256",
    "reviewed_census_sha256", "reviewed_code_digest",
    "reviewed_partitions_digest", "decision", "record_sha256",
}


class ReviewAuthorityRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _self_sha(doc: dict, key: str) -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------
# submissions — produced here, authorising nothing
# --------------------------------------------------------------

def build_submission(*, submitted_at: str, submitter: str,
                     scope: str, manifest_sha256: str,
                     census_sha256: str, consumed: dict) -> dict:
    d = consumed["digests"]
    doc = {
        "schema": SUBMISSION_SCHEMA,
        "submitted_at": submitted_at,
        "submitter": submitter,
        "scope": scope,
        "manifest_sha256": manifest_sha256,
        "census_sha256": census_sha256,
        "code_digest": d["code"],
        "partitions_digest": d["partitions"],
        "data_digest": d["data"],
        "schema_digest": d["schema"],
        "dataset_id": consumed["dataset_id"],
        "subject_ids": list(consumed["subject_ids"]),
        "grants_nothing":
            "a submission states what was consumed and asks for "
            "a decision; it is not a decision, and the gate "
            "refuses to treat it as one",
    }
    doc["submission_sha256"] = _self_sha(doc,
                                         "submission_sha256")
    verify_submission(doc)
    return doc


def verify_submission(doc: dict) -> dict:
    require_keys(doc, _SUBMISSION_KEYS, what="submission")
    if doc["schema"] != SUBMISSION_SCHEMA:
        raise ReviewAuthorityRefusal(
            f"submission schema is {doc['schema']!r}")
    for field in ("manifest_sha256", "census_sha256",
                  "code_digest", "partitions_digest",
                  "data_digest", "schema_digest"):
        require_sha256(doc[field], what=f"submission.{field}")
    require_timestamp(doc["submitted_at"],
                      what="submission.submitted_at")
    require_str(doc["submitter"], what="submission.submitter")
    require_str(doc["scope"], what="submission.scope")
    if not doc["subject_ids"]:
        raise ReviewAuthorityRefusal(
            "a submission with no subjects asks for nothing")
    if _self_sha(doc, "submission_sha256") != \
            doc["submission_sha256"]:
        raise ReviewAuthorityRefusal(
            "submission self digest does not re-derive")
    return doc


# --------------------------------------------------------------
# review records — consumed here, NEVER authored here
# --------------------------------------------------------------

def review_record_path() -> Path:
    """Where the decision lives. Deliberately not a config key."""
    env = os.environ.get(REVIEW_RECORD_ENV)
    return Path(env) if env else DEFAULT_REVIEW_RECORD


# C17: the submission is a PERSISTED artifact with its own
# identity, written once by the SUBMIT_ONLY phase and consumed
# unchanged by the EXECUTE_REVIEWED phase. It is content-addressed
# so the two phases cannot silently disagree about which bytes
# were reviewed.
SUBMISSION_ENV = "CRISPDM_ELIGIBILITY_SUBMISSION_DIR"
DEFAULT_SUBMISSION_DIR = (
    Path.home() / ".local/share/predictor/eligibility_submissions")


def submission_dir() -> Path:
    env = os.environ.get(SUBMISSION_ENV)
    return Path(env) if env else DEFAULT_SUBMISSION_DIR


def submission_path(submission_sha256: str) -> Path:
    return submission_dir() / f"submission-{submission_sha256}.json"


def persist_submission(doc: dict) -> Path:
    """Write a submission content-addressed, exactly once.

    Re-writing the same submission is a no-op; a DIFFERENT
    document under the same name is impossible, because the name
    IS the digest.
    """
    verify_submission(doc)
    d = submission_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = submission_path(doc["submission_sha256"])
    payload = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if p.exists():
        if p.read_text() != payload:
            raise ReviewAuthorityRefusal(
                f"a different submission already occupies "
                f"{p.name} — a content-addressed name is never "
                "overwritten")
        return p
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, payload.encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return p


def load_persisted_submission(submission_sha256: str) -> dict:
    """Consume the submission the reviewer actually reviewed."""
    p = submission_path(submission_sha256)
    if not p.is_file():
        raise ReviewAuthorityRefusal(
            f"no persisted submission {submission_sha256[:12]} — "
            "run the SUBMIT_ONLY phase first; the executing "
            "phase never invents a new submission")
    doc = strict_load_file(p, what="persisted submission")
    verify_submission(doc)
    if doc["submission_sha256"] != submission_sha256:
        raise ReviewAuthorityRefusal(
            "the persisted submission's digest does not match "
            "its own name")
    return doc


def assert_same_submission(persisted: dict, rederived: dict
                           ) -> None:
    """The executing phase re-derives every fact and must land on
    the SAME submission. One changed byte between the phases is a
    different run, and refuses."""
    ignore = {"submitted_at", "submission_sha256"}
    a = {k: v for k, v in persisted.items() if k not in ignore}
    b = {k: v for k, v in rederived.items() if k not in ignore}
    if a != b:
        differing = sorted(k for k in set(a) | set(b)
                           if a.get(k) != b.get(k))
        raise ReviewAuthorityRefusal(
            "the executing phase derived DIFFERENT facts than "
            f"the reviewed submission (differing: {differing}) — "
            "one changed byte between submission and execution "
            "is a different run")


def read_review_record(*, submission: dict, census_sha256: str,
                       scope: str,
                       now: datetime | None = None) -> dict:
    """Consume the external decision, or refuse.

    Every binding is compared against the submission that was
    actually produced by this run, so a record reviewing other
    bytes authorises nothing here.
    """
    path = review_record_path()
    if not path.is_file():
        raise ReviewAuthorityRefusal(
            "no external eligibility review record exists — a "
            "new experiment cannot be gated on a decision that "
            "has not been made (set "
            f"${REVIEW_RECORD_ENV} if the authority store has "
            "moved)")
    doc = strict_load_file(path, what="review record")
    require_keys(doc, _RECORD_KEYS, what="review record")
    if doc["schema"] != REVIEW_RECORD_SCHEMA:
        raise ReviewAuthorityRefusal(
            f"review record schema is {doc['schema']!r}")
    for field in ("reviewed_submission_sha256",
                  "reviewed_manifest_sha256",
                  "reviewed_census_sha256",
                  "reviewed_code_digest",
                  "reviewed_partitions_digest"):
        require_sha256(doc[field], what=f"review record.{field}")
    reviewed_at = require_timestamp(
        doc["reviewed_at"], what="review record.reviewed_at",
        now=now)
    submitted_at = require_timestamp(
        submission["submitted_at"], what="submission.submitted_at",
        now=now)
    require_chronology([("submitted_at", submitted_at),
                        ("reviewed_at", reviewed_at)],
                       what="review chronology")
    require_str(doc["reviewer"], what="review record.reviewer")
    if doc["decision"] != REVIEW_DECISION:
        raise ReviewAuthorityRefusal(
            f"review decision is {doc['decision']!r}, not "
            f"{REVIEW_DECISION!r}")
    if _self_sha(doc, "record_sha256") != doc["record_sha256"]:
        raise ReviewAuthorityRefusal(
            "review record self digest does not re-derive")
    if doc["reviewed_submission_sha256"] != \
            submission["submission_sha256"]:
        raise ReviewAuthorityRefusal(
            "the review record reviews a DIFFERENT submission — "
            "a decision about other bytes is not a decision "
            "about this run")
    if doc["reviewed_manifest_sha256"] != \
            submission["manifest_sha256"]:
        raise ReviewAuthorityRefusal(
            "the review record binds a different manifest")
    if doc["reviewed_census_sha256"] != census_sha256:
        raise ReviewAuthorityRefusal(
            "the review record binds a different physical census")
    if doc["reviewed_code_digest"] != submission["code_digest"]:
        raise ReviewAuthorityRefusal(
            "the review record binds different code than the "
            "code about to consume the data")
    if doc["reviewed_partitions_digest"] != \
            submission["partitions_digest"]:
        raise ReviewAuthorityRefusal(
            "the review record binds a different partition "
            "layout")
    if doc["scope"] != scope:
        raise ReviewAuthorityRefusal(
            f"the review record covers scope {doc['scope']!r}, "
            f"not {scope!r} — eligibility is never global")
    if "<" in json.dumps(doc):
        raise ReviewAuthorityRefusal(
            "the review record carries template placeholders — "
            "a template grants nothing")
    return doc


def review_record_template() -> dict:
    """A NON-AUTHORIZING template. Every value is a placeholder
    and the strict reader refuses it."""
    return {
        "_template_note":
            "NON-AUTHORIZING TEMPLATE. The external reviewer "
            "fills every <field> and installs the result at the "
            "authority path himself. This repository never "
            "authors, installs or simulates a real record; the "
            "extra key and the placeholders are both refused.",
        "schema": REVIEW_RECORD_SCHEMA,
        "reviewed_at": "<RFC3339 UTC, not in the future>",
        "reviewer": "<reviewer identity>",
        "scope": "<the exact scope this decision covers>",
        "reviewed_submission_sha256":
            "<64-hex submission_sha256 of the submitted "
            "document>",
        "reviewed_manifest_sha256": "<64-hex manifest_sha256>",
        "reviewed_census_sha256":
            "<64-hex census_sha256 of the physical census>",
        "reviewed_code_digest": "<64-hex code digest>",
        "reviewed_partitions_digest":
            "<64-hex partitions digest>",
        "decision": REVIEW_DECISION,
        "record_sha256":
            "<64-hex sha256 of the canonical body without this "
            "field>",
    }


def utc_now_stamp() -> str:
    return datetime.now(timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
