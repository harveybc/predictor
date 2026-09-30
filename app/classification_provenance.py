"""Provenance as a checked field: where the evidence came from, and what it may become.

Why this module exists
----------------------
Two live facts, both from 2026-09-29.

In the product, a declared non-model answering path published a real checkpoint's
macro-F1 and its `n` beside its own answers. The record was genuine and the
answers were genuine; what was false was that they belonged together. That was
withdrawn, with the record named as not belonging to the answering path.

In the warehouse, the producer contract had no field for any of this. A receipt
written by a path that had loaded no weights at all still carried a 64-hex
`checkpoint_sha256`, and a badge could be earned from it: measured at the base
tip of this branch, a constant answer table earned macro-F1 0.8889 as a provider
quality badge, and no field anywhere named what had answered.

What this module does NOT do
----------------------------
It does not refuse a word. The first repair in the product read
``backend != "fixture"``, and that is a spell check, not a contract: a declared
test may be named anything at all, and an honest measurement's own prose may
legitimately contain that word. ``test_the_gate_never_matches_a_word`` in
``tools/test_classification_provenance.py`` fails if such a match is
reintroduced here, and two counterexamples are kept: a declared test that
carries the word nowhere is still refused promotion, and a real measurement that
carries it in its limitations text is still admitted.

What is checked instead
-----------------------
Four facts, each its own field, each from the answering path itself:

* ``path_id``   - which code path produced the answers;
* ``kind``      - what that path is, from a closed vocabulary in the contract;
* ``weights_present`` - whether learned parameters were loaded, as a boolean the
  kind determines, so the two cannot drift apart;
* ``served_checkpoint`` / ``served_checkpoint_sha256`` - which checkpoint served,
  or an explicit sentinel saying none did, or that none was digested;
* ``attestation`` - how the above was established: observed from the path that
  answered, declared by a configuration, or not established at all.

And three rules over them:

* **the record must belong to the path.** A receipt's ``checkpoint`` and
  ``checkpoint_sha256`` must be the ones the answering path served. A digest that
  belongs to some other checkpoint is refused by name; that is this warehouse's
  form of the withdrawal the product made.
* **a promotion is refused by reason.** ``evidence_class`` ``MEASUREMENT``
  requires a path that had weights and an established attestation. A path with no
  weights is a declared test, and it stays one - it is kept, stored, and legible
  as a test through ``evidence_role``.
* **a badge may not rest on a declared test.** It additionally requires the
  answering path to have been observed, not merely configured.

Absence is not coincidence. An undeclared provenance block is refused rather
than read as a model path.
"""

from __future__ import annotations

import hashlib
import json
import re

from app.classification_receipt import CONTRACT as _METRIC_CONTRACT

SCHEMA = "classification_answering_path.v1"

#: the whole vocabulary lives in the contract document; this module reads it and
#: keeps no second copy that could drift from it.
SPEC = _METRIC_CONTRACT["answering_path"]
PATH_KINDS = SPEC["kinds"]
MODEL_KINDS = {kind for kind, spec in PATH_KINDS.items() if spec["weights_present"]}
ATTESTATIONS = SPEC["attestations"]
OBSERVED = SPEC["observed_attestation"]
NOT_ESTABLISHED = SPEC["unestablished_attestation"]
NO_CHECKPOINT = SPEC["no_checkpoint_sentinel"]
NOT_DIGESTED = SPEC["not_digested_sentinel"]
CHECKPOINT_SENTINELS = (NO_CHECKPOINT, NOT_DIGESTED)
NAMEABLE_WITHOUT_WEIGHTS = set(SPEC["kinds_that_may_name_a_foreign_checkpoint"])
PATH_FIELDS = tuple(SPEC["fields"])

PROMOTION = _METRIC_CONTRACT["promotion_rule"]
DECLARATION_CORPUS_CLASSES = set(PROMOTION["corpus_classes_that_are_declarations"])
MEASUREMENT = PROMOTION["measured_evidence_class"]

#: the two roles a stored classification row can have. One of them is not a result.
MODEL_RESULT = PROMOTION["roles"]["model"]
DECLARED_NON_MODEL_TEST = PROMOTION["roles"]["declared"]
#: what admission returns for a terminal of some other contract: the
#: general-purpose warehouse stays generic and is not judged here.
NOT_THIS_CONTRACT = "NOT_THIS_CONTRACT"

# --- refusal names, all declared in the contract document -------------------- #
KIND_NOT_DECLARED = "PROVENANCE_REFUSED_KIND_NOT_DECLARED"
KIND_CONTRADICTS_WEIGHTS = "PROVENANCE_REFUSED_KIND_CONTRADICTS_WEIGHTS"
ATTESTATION_NOT_DECLARED = "PROVENANCE_REFUSED_ATTESTATION_NOT_DECLARED"
FIELD_MISSING = "PROVENANCE_REFUSED_FIELD_MISSING"
DIGEST_ON_A_PATH_THAT_SERVED_NONE = \
    "PROVENANCE_REFUSED_CHECKPOINT_DIGEST_ON_A_PATH_THAT_SERVED_NONE"
SERVING_PATH_MUST_NAME_ITS_CHECKPOINT = \
    "PROVENANCE_REFUSED_A_SERVING_PATH_MUST_NAME_THE_CHECKPOINT_IT_SERVED"
MAY_NOT_NAME_A_CHECKPOINT = \
    "PROVENANCE_REFUSED_A_PATH_WITHOUT_WEIGHTS_MAY_NOT_NAME_A_CHECKPOINT"
RECORD_IS_NOT_OF_THE_ANSWERING_PATH = \
    "PROVENANCE_REFUSED_RECORD_IS_NOT_OF_THE_ANSWERING_PATH"

NO_WEIGHTS_IS_NOT_A_MODEL_RESULT = \
    "PROMOTION_REFUSED_A_PATH_WITH_NO_WEIGHTS_IS_NOT_A_MODEL_RESULT"
MEASUREMENT_WITHOUT_AN_ESTABLISHED_PATH = \
    "PROMOTION_REFUSED_MEASUREMENT_WITHOUT_AN_ESTABLISHED_ANSWERING_PATH"
DECLARED_CORPUS_IS_NOT_A_MEASUREMENT = \
    "PROMOTION_REFUSED_A_DECLARED_CORPUS_CLASS_IS_NOT_A_MEASUREMENT"

BADGE_PROVENANCE_NOT_DECLARED = "BADGE_REFUSED_PROVENANCE_NOT_DECLARED"
BADGE_NON_MODEL_ANSWERING_PATH = "BADGE_REFUSED_NON_MODEL_ANSWERING_PATH"
BADGE_ANSWERING_PATH_NOT_OBSERVED = "BADGE_REFUSED_ANSWERING_PATH_NOT_OBSERVED"
BADGE_RECORD_NOT_OF_THE_PATH = "BADGE_REFUSED_RECORD_IS_NOT_OF_THE_ANSWERING_PATH"

ADMISSION_PROVENANCE_TAGS_MISSING = "ADMISSION_REFUSED_PROVENANCE_TAGS_MISSING"
ADMISSION_PROVENANCE_DIGEST_MISMATCH = "ADMISSION_REFUSED_PROVENANCE_DIGEST_MISMATCH"
ADMISSION_ROLE_CONTRADICTS_THE_PATH = "ADMISSION_REFUSED_EVIDENCE_ROLE_CONTRADICTS_THE_PATH"
ADMISSION_NON_MODEL_MAY_NOT_GOVERN = \
    "ADMISSION_REFUSED_A_PATH_WITHOUT_WEIGHTS_MAY_NOT_BE_A_GOVERNING_TERMINAL"

_HEX64 = re.compile(r"[0-9a-f]{64}")

#: the tags the producer-to-store contract carries for provenance. Every one of
#: them is a string, like every other governed terminal tag.
PROVENANCE_TAGS = ("answering_path_id", "answering_path_kind", "weights_present",
                   "served_checkpoint", "served_checkpoint_sha256",
                   "provenance_attestation", "provenance_sha256", "evidence_role")


class ProvenanceError(Exception):
    """Base class. Every refusal here is named and carries its reason."""

    def __init__(self, refusal: str, detail: str):
        super().__init__(f"{refusal}: {detail}")
        self.refusal = refusal
        self.detail = detail


class ProvenanceRefused(ProvenanceError):
    """A provenance block that misstates what produced the answers."""


class PromotionRefused(ProvenanceError):
    """A declared non-model path was offered as a model result."""


class AdmissionRefused(ProvenanceError):
    """The producer-to-store boundary refused a terminal, by reason."""


def canonical_text(body) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def provenance_sha256(path: dict) -> str:
    """Digest over every field that decides where the evidence came from.

    Carried as a tag so a later editor cannot change one provenance tag and
    leave the rest agreeing with each other.
    """
    return hashlib.sha256(canonical_text({
        "schema": SCHEMA,
        **{name: path[name] for name in PATH_FIELDS},
    }).encode("ascii")).hexdigest()


def build_answering_path(block) -> dict:
    """Validate one provenance block and return it normalised, or refuse by name."""
    if not isinstance(block, dict):
        raise ProvenanceRefused(FIELD_MISSING, f"{SPEC['field']} must be an object")
    for name in PATH_FIELDS:
        if name not in block:
            raise ProvenanceRefused(
                FIELD_MISSING,
                f"{SPEC['field']}.{name} is required; " + SPEC["absence_rule"])

    path_id = block["path_id"]
    if not isinstance(path_id, str) or not path_id:
        raise ProvenanceRefused(FIELD_MISSING,
                                f"{SPEC['field']}.path_id must name the code path that answered")

    kind = block["kind"]
    if kind not in PATH_KINDS:
        raise ProvenanceRefused(
            KIND_NOT_DECLARED,
            f"{kind!r} is not a declared answering-path kind; the contract declares "
            + ", ".join(sorted(PATH_KINDS)))

    weights = block["weights_present"]
    expected = PATH_KINDS[kind]["weights_present"]
    if not isinstance(weights, bool) or weights is not expected:
        raise ProvenanceRefused(
            KIND_CONTRADICTS_WEIGHTS,
            f"kind {kind} declares weights_present={expected!r} and the block says "
            f"{weights!r}; the two may not disagree")

    attestation = block["attestation"]
    if attestation not in ATTESTATIONS:
        raise ProvenanceRefused(
            ATTESTATION_NOT_DECLARED,
            f"{attestation!r} is not a declared attestation; the contract declares "
            + ", ".join(sorted(ATTESTATIONS)))

    name, digest = block["served_checkpoint"], block["served_checkpoint_sha256"]
    for label, value in (("served_checkpoint", name), ("served_checkpoint_sha256", digest)):
        if not isinstance(value, str) or not value:
            raise ProvenanceRefused(FIELD_MISSING,
                                    f"{SPEC['field']}.{label} must be a string")
    if weights:
        if not (_HEX64.fullmatch(digest) or digest == NOT_DIGESTED):
            raise ProvenanceRefused(
                SERVING_PATH_MUST_NAME_ITS_CHECKPOINT,
                f"kind {kind} loaded weights, so served_checkpoint_sha256 is a sha256 digest "
                f"or the sentinel {NOT_DIGESTED}, not {digest!r}")
        if name == NO_CHECKPOINT:
            raise ProvenanceRefused(
                SERVING_PATH_MUST_NAME_ITS_CHECKPOINT,
                f"kind {kind} loaded weights and must name what it loaded")
    else:
        if digest != NO_CHECKPOINT:
            raise ProvenanceRefused(
                DIGEST_ON_A_PATH_THAT_SERVED_NONE,
                f"kind {kind} loaded no weights, so served_checkpoint_sha256 is the sentinel "
                f"{NO_CHECKPOINT}, not {digest!r}; a digest here would be shaped exactly like "
                f"one a model produced")
        if name != NO_CHECKPOINT and kind not in NAMEABLE_WITHOUT_WEIGHTS:
            raise ProvenanceRefused(
                MAY_NOT_NAME_A_CHECKPOINT,
                f"kind {kind} loaded no weights and may not name {name!r} as what served; "
                f"only " + ", ".join(sorted(NAMEABLE_WITHOUT_WEIGHTS)) + " may name a "
                "checkpoint it did not run")

    return {"schema": SCHEMA, **{key: block[key] for key in PATH_FIELDS}}


def evidence_role(path: dict) -> str:
    """`MODEL_RESULT` when weights were loaded; otherwise a declared test."""
    return MODEL_RESULT if path["kind"] in MODEL_KINDS else DECLARED_NON_MODEL_TEST


def quoted_record_check(*, path: dict, checkpoint, checkpoint_sha256) -> None:
    """The model identity a receipt quotes must be the one the path served."""
    if checkpoint != path["served_checkpoint"] or \
            checkpoint_sha256 != path["served_checkpoint_sha256"]:
        raise ProvenanceRefused(
            RECORD_IS_NOT_OF_THE_ANSWERING_PATH,
            f"the receipt quotes checkpoint {checkpoint!r} / {checkpoint_sha256} and the path "
            f"{path['path_id']} served {path['served_checkpoint']!r} / "
            f"{path['served_checkpoint_sha256']}; a number about one is not a number about "
            f"the other")


def promotion_check(*, path: dict, evidence_class: str, corpus_class: str) -> str:
    """The role this evidence may carry, or a refusal naming the reason.

    Only the declared path decides. Nothing here reads any free text.
    """
    role = evidence_role(path)
    if evidence_class != MEASUREMENT:
        return role
    if role != MODEL_RESULT:
        raise PromotionRefused(
            NO_WEIGHTS_IS_NOT_A_MODEL_RESULT,
            f"the answering path {path['path_id']} is {path['kind']} and loaded no weights, "
            f"so its answers are a declared test and not a {MEASUREMENT} of a model; "
            f"{PATH_KINDS[path['kind']]['what_it_is']}")
    if path["attestation"] == NOT_ESTABLISHED:
        raise PromotionRefused(
            MEASUREMENT_WITHOUT_AN_ESTABLISHED_PATH,
            f"the answering path {path['path_id']} was never established "
            f"({path['attestation']}); absence of a contradiction is not evidence that a "
            f"model answered")
    if corpus_class in DECLARATION_CORPUS_CLASSES:
        raise PromotionRefused(
            DECLARED_CORPUS_IS_NOT_A_MEASUREMENT,
            f"corpus_class {corpus_class} is declared author material, which the contract "
            f"names as a declaration rather than a population a {MEASUREMENT} is made over")
    return role


def provenance_tags(path: dict, role: str) -> dict:
    """The provenance the producer-to-store contract carries, as string tags."""
    return {"answering_path_id": path["path_id"],
            "answering_path_kind": path["kind"],
            "weights_present": "true" if path["weights_present"] else "false",
            "served_checkpoint": path["served_checkpoint"],
            "served_checkpoint_sha256": path["served_checkpoint_sha256"],
            "provenance_attestation": path["attestation"],
            "provenance_sha256": provenance_sha256(path),
            "evidence_role": role}


def badge_provenance_refusal(record) -> tuple | None:
    """`(refusal, detail)` when a quality claim would rest on a declared test.

    Returned rather than raised so the badge keeps one exception type for all of
    its refusals. A badge asks for more than a receipt does: the answering path
    must have been observed, not merely configured.
    """
    block = record.get(SPEC["field"]) if isinstance(record, dict) else None
    if not isinstance(block, dict):
        return (BADGE_PROVENANCE_NOT_DECLARED,
                f"{record.get('receipt_id')} declares no answering path; "
                + SPEC["absence_rule"])
    try:
        path = build_answering_path(block)
    except ProvenanceRefused as error:
        return (BADGE_PROVENANCE_NOT_DECLARED,
                f"{record.get('receipt_id')} carries an answering path that misstates "
                f"itself: {error}")
    if evidence_role(path) != MODEL_RESULT:
        return (BADGE_NON_MODEL_ANSWERING_PATH,
                f"{record.get('receipt_id')} was answered by {path['path_id']}, which is "
                f"{path['kind']} and loaded no weights; a quality claim about a provider "
                f"cannot rest on it, whatever the record is labelled")
    if path["attestation"] != OBSERVED:
        return (BADGE_ANSWERING_PATH_NOT_OBSERVED,
                f"{record.get('receipt_id')} declares its answering path as "
                f"{path['attestation']}; a badge requires {OBSERVED}")
    try:
        quoted_record_check(path=path, checkpoint=record.get("checkpoint"),
                            checkpoint_sha256=record.get("checkpoint_sha256"))
    except ProvenanceRefused as error:
        return BADGE_RECORD_NOT_OF_THE_PATH, str(error.detail)
    return None


def admit_classification_terminal(terminal) -> dict:
    """The producer-to-store gate, reading only what the store would read.

    It takes the terminal body, not a receipt object, and re-derives every rule
    from the tags alone - so a terminal hand-written by any producer, under any
    actor name, is held to the same contract as one this repository built. A
    terminal of another contract is returned unjudged: the general-purpose
    warehouse stays generic.
    """
    if not isinstance(terminal, dict):
        raise AdmissionRefused(ADMISSION_PROVENANCE_TAGS_MISSING,
                               "a terminal must be an object")
    tags = terminal.get("tags")
    if not isinstance(tags, dict) or tags.get("metric_contract") != _METRIC_CONTRACT["schema"]:
        return {"admitted": True, "evidence_role": NOT_THIS_CONTRACT,
                "contract": tags.get("metric_contract") if isinstance(tags, dict) else None,
                "checked": []}

    missing = [name for name in PROVENANCE_TAGS if not isinstance(tags.get(name), str)]
    if missing:
        raise AdmissionRefused(
            ADMISSION_PROVENANCE_TAGS_MISSING,
            f"a {_METRIC_CONTRACT['schema']} terminal must carry its provenance; missing "
            + ", ".join(missing))
    if tags["weights_present"] not in ("true", "false"):
        raise AdmissionRefused(ADMISSION_PROVENANCE_TAGS_MISSING,
                               f"weights_present is {tags['weights_present']!r}")

    block = {"path_id": tags["answering_path_id"], "kind": tags["answering_path_kind"],
             "weights_present": tags["weights_present"] == "true",
             "served_checkpoint": tags["served_checkpoint"],
             "served_checkpoint_sha256": tags["served_checkpoint_sha256"],
             "attestation": tags["provenance_attestation"]}
    if provenance_sha256(block) != tags["provenance_sha256"]:
        raise AdmissionRefused(
            ADMISSION_PROVENANCE_DIGEST_MISMATCH,
            "the provenance tags do not hash to the provenance_sha256 they carry; one of "
            "them was changed after the receipt was sealed")
    try:
        path = build_answering_path(block)
        quoted_record_check(path=path, checkpoint=tags.get("checkpoint"),
                            checkpoint_sha256=tags.get("checkpoint_sha256"))
        role = promotion_check(path=path, evidence_class=tags.get("evidence_class"),
                               corpus_class=tags.get("corpus_class"))
    except ProvenanceError as error:
        raise AdmissionRefused(error.refusal, error.detail) from error

    if tags["evidence_role"] != role:
        raise AdmissionRefused(
            ADMISSION_ROLE_CONTRADICTS_THE_PATH,
            f"the terminal declares evidence_role {tags['evidence_role']} and its answering "
            f"path {path['path_id']} ({path['kind']}) gives {role}")
    if terminal.get("classification") == "GOVERNING" and role != MODEL_RESULT:
        raise AdmissionRefused(
            ADMISSION_NON_MODEL_MAY_NOT_GOVERN,
            f"the answering path {path['path_id']} is {path['kind']}; a declared test is kept "
            f"and stored, and it does not govern anything")
    return {"admitted": True, "evidence_role": role, "answering_path": path,
            "contract": _METRIC_CONTRACT["schema"],
            "checked": ["provenance_sha256", "answering_path", "quoted_record",
                        "promotion", "evidence_role", "governing"]}
