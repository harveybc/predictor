"""Provenance of learned weights, and the versioned migration of historical artifacts (b327b771 item 6).

Donor sidecars move from schema 1 to schema 2, and bundles from ``predictor.modular.bundle.v1`` to
``v2``. Both carry a provenance block (``predictor.modular.provenance.v1``):

``conditioning_contract``  OPERATIONAL (only information available at t) | SYNTHETIC_OFFLINE
                           (may condition on future targets; never served) | UNKNOWN
``learned_corpus``         kind TRAIN_ONLY (dataset_id, data_sha256, support; no pretrained source)
                           | FOREIGN_PRETRAINED / MIXED (pretrained_weights_source named) | UNKNOWN
``reconstruction``         state MEASURED (mae_z, mse_z) | NOT_APPLICABLE (no decoder: a state, not a
                           failure) | FAILED | NOT_EVALUATED | UNKNOWN

A producer that declares nothing gets UNKNOWN written EXPLICITLY. A historical artifact migrated
from schema 1 / bundle.v1 is UNKNOWN in every field and records the migration; no safe default is
ever invented for it. Consumers that need OPERATIONAL weights ask for it
(``load_donor(..., require_contract="OPERATIONAL")``) and UNKNOWN is refused.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from pathlib import Path


def _copy(value):        # standard library only: this module is loadable without TensorFlow
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

PROVENANCE_SCHEMA = "predictor.modular.provenance.v1"
DONOR_SCHEMA = 2
BUNDLE_V1, BUNDLE_V2 = "predictor.modular.bundle.v1", "predictor.modular.bundle.v2"
CONTRACTS = ("OPERATIONAL", "SYNTHETIC_OFFLINE", "UNKNOWN")
CORPUS_KINDS = ("TRAIN_ONLY", "FOREIGN_PRETRAINED", "MIXED", "UNKNOWN")
RECONSTRUCTION_STATES = ("MEASURED", "NOT_APPLICABLE", "FAILED", "NOT_EVALUATED", "UNKNOWN")
DECLARABLE = ("conditioning_contract", "learned_corpus", "reconstruction")


def _refuse(code, detail):
    raise ValueError(f"{code}: {detail}")


def unknown():
    return {"provenance_schema": PROVENANCE_SCHEMA, "conditioning_contract": "UNKNOWN",
            "learned_corpus": {"kind": "UNKNOWN"}, "reconstruction": {"state": "UNKNOWN"}}


def complete(declared=None):
    """Validated provenance block; undeclared fields are UNKNOWN, written explicitly."""
    declared = _copy(declared or {})
    extra = set(declared) - set(DECLARABLE)
    if extra:
        _refuse("PROVENANCE_FIELD_UNKNOWN", f"undeclarable provenance fields {sorted(extra)}")
    p = unknown()
    p.update(declared)
    if p["conditioning_contract"] not in CONTRACTS:
        _refuse("CONDITIONING_CONTRACT_INVALID", f"{p['conditioning_contract']!r} not in {CONTRACTS}")
    corpus = p["learned_corpus"]
    if not isinstance(corpus, dict) or corpus.get("kind") not in CORPUS_KINDS:
        _refuse("CORPUS_KIND_INVALID", f"learned_corpus.kind must be one of {CORPUS_KINDS}")
    source = corpus.get("pretrained_weights_source")
    if corpus["kind"] == "TRAIN_ONLY":
        if source:
            _refuse("FOREIGN_CORPUS_WEIGHTS_NOT_TRAIN_ONLY", f"weights from {source!r} are not TRAIN_ONLY")
        if not (corpus.get("dataset_id") and re.fullmatch(r"[0-9a-f]{64}", str(corpus.get("data_sha256", "")))
                and corpus.get("support")):
            _refuse("TRAIN_CORPUS_IDENTITY_MISSING", "TRAIN_ONLY needs dataset_id, data_sha256 and support")
    elif corpus["kind"] in ("FOREIGN_PRETRAINED", "MIXED") and not source:
        _refuse("PRETRAINED_SOURCE_MISSING", f"a {corpus['kind']} corpus names its pretrained weights source")
    recon = p["reconstruction"]
    if not isinstance(recon, dict) or recon.get("state") not in RECONSTRUCTION_STATES:
        _refuse("RECONSTRUCTION_STATE_INVALID", f"reconstruction.state must be one of {RECONSTRUCTION_STATES}")
    if recon["state"] == "MEASURED" and not all(isinstance(recon.get(k), (int, float)) for k in ("mae_z", "mse_z")):
        _refuse("RECONSTRUCTION_MEASURE_MISSING", "MEASURED reconstruction carries mae_z and mse_z")
    return p


def _atomic(path, document):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _migration(from_schema, to_schema):
    return {"from_schema": from_schema, "to_schema": to_schema,
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "rule": "historical provenance is UNKNOWN; nothing was inferred"}


ALONGSIDE_SUFFIX = ".manifest.v2.json"
_BINDING = ("manifest_sha256", "model_sha256", "weights_sha256")


def _alongside(path):
    return Path(str(Path(path).with_suffix("")) + ALONGSIDE_SUFFIX)


def write_alongside(path, declared, derivation):
    """Schema-2 provenance sidecar written NEXT TO an untouched schema-1 sidecar (versioned migration
    mode ALONGSIDE). ``declared`` must be derived by a stated rule from the producer's own records
    (``derivation`` names the rule and the source files with their sha256); it is validated like any
    declaration. The schema-1 sidecar, the archive and the producer's records are never modified. An
    existing alongside file is reused only if identical; a different one is refused."""
    sidecar = Path(path).with_suffix(".manifest.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if document.get("schema") != 1:
        _refuse("ALONGSIDE_NEEDS_SCHEMA_1", "only a schema-1 sidecar gets an alongside declaration")
    if not isinstance(derivation, dict) or not derivation.get("rule") or not derivation.get("sources"):
        _refuse("DERIVATION_MISSING", "an alongside declaration names its rule and its source records")
    keep = {k: v for k, v in (document.get("provenance") or {}).items() if k == "keras_version"}
    alongside = {"schema": DONOR_SCHEMA, "manifest": document["manifest"],
                 **{k: document[k] for k in _BINDING},
                 "provenance": {**complete(declared), **keep, "derivation": _copy(derivation),
                                "migration": {"from_schema": 1, "to_schema": DONOR_SCHEMA, "mode": "ALONGSIDE",
                                              "schema1_sidecar": sidecar.name}}}
    target = _alongside(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        strip = lambda d: {**d, "provenance": {k: v for k, v in d["provenance"].items() if k != "written_utc"}}
        if strip(existing) == strip(alongside):
            return {"status": "ALREADY_CURRENT", "path": str(target)}
        _refuse("ALONGSIDE_CONFLICT", f"{target.name} exists with a different declaration")
    alongside["provenance"]["written_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    _atomic(target, alongside)
    return {"status": "WRITTEN", "path": str(target)}


def donor_provenance(path):
    """Provenance of a donor. A schema-1 sidecar reports UNKNOWN with migrated=False, unless an
    ALONGSIDE schema-2 sidecar binds the very same manifest, archive and weights hashes."""
    sidecar = Path(path).with_suffix(".manifest.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    target = _alongside(path)
    if document.get("schema") == 1 and target.is_file():
        alongside = json.loads(target.read_text(encoding="utf-8"))
        if all(alongside.get(k) == document.get(k) for k in _BINDING) and alongside.get("schema") == DONOR_SCHEMA:
            return _copy(alongside["provenance"])
    if document.get("schema") == 1:
        return {**unknown(), **(document.get("provenance") or {}), "migrated": False,
                "conditioning_contract": "UNKNOWN", "learned_corpus": {"kind": "UNKNOWN"},
                "reconstruction": {"state": "UNKNOWN"}}
    return _copy(document["provenance"])


def migrate_donor_sidecar(path):
    """Schema 1 -> 2 in place; the schema-1 file is kept as <stem>.manifest.schema1.json."""
    sidecar = Path(path).with_suffix(".manifest.json")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if document.get("schema") == DONOR_SCHEMA:
        return {"status": "ALREADY_CURRENT", "schema": DONOR_SCHEMA}
    if document.get("schema") != 1:
        _refuse("SIDECAR_SCHEMA_UNKNOWN", f"cannot migrate schema {document.get('schema')!r}")
    backup = Path(str(Path(path).with_suffix("")) + ".manifest.schema1.json")
    if not backup.exists():
        shutil.copy2(sidecar, backup)
    migrated = _copy(document)
    kept = {k: v for k, v in (document.get("provenance") or {}).items()
            if k in ("keras_version", "declared_params", "objective", "remanifested_from_manifest_sha256",
                     "keras_version_source")}
    migrated["provenance"] = {**unknown(), **kept, "migration": _migration(1, DONOR_SCHEMA)}
    migrated["schema"] = DONOR_SCHEMA
    _atomic(sidecar, migrated)
    return {"status": "MIGRATED", "from_schema": 1, "to_schema": DONOR_SCHEMA}


def bundle_provenance(document):
    if document.get("schema") == BUNDLE_V1:
        return {**unknown(), "migrated": False}
    return _copy(document["provenance"])


def migrate_bundle(directory):
    """bundle.v1 -> v2 in place; the v1 document is kept as bundle.v1.json; archive bytes untouched."""
    path = Path(directory) / "bundle.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") == BUNDLE_V2:
        return {"status": "ALREADY_CURRENT", "schema": BUNDLE_V2}
    if document.get("schema") != BUNDLE_V1:
        _refuse("BUNDLE_SCHEMA_UNKNOWN", f"cannot migrate {document.get('schema')!r}")
    backup = Path(directory) / "bundle.v1.json"
    if not backup.exists():
        shutil.copy2(path, backup)
    migrated = _copy(document)
    migrated["schema"] = BUNDLE_V2
    migrated["provenance"] = {**unknown(), "migration": _migration(BUNDLE_V1, BUNDLE_V2)}
    _atomic(path, migrated)
    return {"status": "MIGRATED", "from_schema": BUNDLE_V1, "to_schema": BUNDLE_V2}
