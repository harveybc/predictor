"""I6-DMO feature provenance: manifest identity, availability gate, producer status.

Reuses the sealed I6N availability contract (``tools.i6n_contracts.
feature_availability``, contract 9668898c...) and the 19-feature manifest
without STLFSI4 (manifest 2ff34fa2...). Nothing here shifts a feature by an
assumed lag, substitutes a feature or reads a TEST row.

``producer_status`` records the recovery attempt for the seven features whose
producer code or publication/consumption clock was unresolved by the causal
trace audit. A feature is RECOVERED only with a producer path at a commit and
a reproduction match on retained TRAIN rows; otherwise it is STILL_UNRESOLVED
and names the missing object. While any is unresolved, ``confirmatory_gate``
refuses (``PROVENANCE_UNRESOLVED``): development VALIDATION may proceed, a
confirmatory TEST may not.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.i6n_contracts import feature_availability as fa

EXPECTED_MANIFEST_SHA256 = "2ff34fa21d5841da1befdd4d898db911128d959a5e460d04cc45b54955416470"
EXPECTED_CONTRACT_SHA256 = "9668898c28cafeef2bc48e7d3a42e82a2655d699ff8f0d8d59e921bf120b1bc9"
MANIFEST = tuple(fa.MANIFEST_NO_STLFSI4)
EXCLUDED = tuple(fa.EXCLUDED_FEATURES)
_ALIAS = re.compile(r"stlfsi", re.IGNORECASE)

#: Typed clock states. PARTIAL carries a measurement: declared-rule coverage is
#: refuted by a retained artifact without that being a claim of look-ahead.
CLOCK_PROVEN = "PROVEN"
CLOCK_PARTIAL = "PARTIAL_COVERAGE_REFUTED_BY_ALFRED_NOT_PROVEN_LOOKAHEAD"
CLOCK_UNRESOLVED = "UNRESOLVED"
CLOCK_STATES = (CLOCK_PROVEN, CLOCK_PARTIAL, CLOCK_UNRESOLVED)
CLOCK_REFUTATION_NOTE = (
    "ALFRED realtime_start is the date a value entered the FRED/ALFRED database and is an UPPER "
    "bound on public availability, so a realtime_start later than the declared instant REFUTES "
    "coverage of the declared clock by this artifact; it does not by itself prove look-ahead. "
    "Whether the documented H.15 4:15pm ET posting may be accepted as the availability instant "
    "is a ruling for Musashi/the owner, not for this record.")
#: Incomplete provenance blocks the confirmatory TEST only.
BLOCKS_TEST_ONLY = ("confirmatory TEST only; labelled development execution (the MG2/MG12 "
                    "development fits) is NOT blocked")

UNRESOLVED_SEVEN = (
    "fred.rates.dgs3mo.diff_5d",
    "fred.rates.dgs7.logret_5d",
    "fx.gbpusd.logret_1h",
    "px.close_loc",
    "px.log_co",
    "px.rv5",
    "rg.atr_ratio",
)

#: Recovery record (see docs/i6n/direct_multi_output_static/PRODUCER_RECOVERY.json
#: for the evidence). Filled from the data-lane search of 2026-10-14.
PRODUCERS_PATH = Path(__file__).with_name("producer_recovery.v1.json")


class ProvenanceError(ValueError):
    """Typed refusal; ``code`` is stable."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Content digest of named arrays (dtype, shape, C-order bytes); file-format free."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        if a.dtype.kind in "OU":
            payload = json.dumps([str(v) for v in a.reshape(-1).tolist()]).encode()
        else:
            payload = a.tobytes()
        h.update(json.dumps([key, a.dtype.str, list(a.shape)]).encode())
        h.update(hashlib.sha256(payload).digest())
    return h.hexdigest()


def check_manifest(features: Sequence[str]) -> dict:
    """Refuse STLFSI4 or any alias, duplicates and a manifest other than the 19 sealed names."""
    names = list(features)
    aliases = [f for f in names if _ALIAS.search(str(f))]
    if aliases:
        raise ProvenanceError("EXCLUDED_FEATURE_ALIAS", f"STLFSI4 or alias present: {aliases}")
    if len(set(names)) != len(names):
        raise ProvenanceError("MANIFEST_INVALID", "duplicate feature names")
    digest = fa.manifest_sha256(names)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ProvenanceError("MANIFEST_MISMATCH",
                              f"manifest {digest} != sealed {EXPECTED_MANIFEST_SHA256}")
    return {"features": names, "manifest_sha256": digest}


def availability_gate(features: Sequence[str], contract: Mapping[str, Any] | None = None) -> dict:
    """Run the sealed availability gate (development mode) and report blocking state."""
    check_manifest(features)
    contract = fa.load_contract() if contract is None else contract
    if contract.get("contract_sha256") != EXPECTED_CONTRACT_SHA256:
        raise ProvenanceError("CONTRACT_MISMATCH", str(contract.get("contract_sha256")))
    try:
        report = fa.gate(list(features), contract, require_verified=False)
    except fa.AvailabilityGateError as exc:
        raise ProvenanceError(exc.code, str(exc)) from None
    status = producer_status()
    unresolved = sorted(f for f, row in status.items() if row["status"] != "RECOVERED")
    report = dict(report)
    report["producer_unresolved"] = unresolved
    report["confirmatory_test_blocked"] = bool(report["unverified_features"] or unresolved)
    report["mode"] = "DEVELOPMENT_VALIDATION_ONLY"
    report["blocks"] = BLOCKS_TEST_ONLY
    report["clock_states"] = {f: (status[f].get("clock") or {}).get("state")
                              for f in unresolved}
    return report


def producer_status() -> dict:
    """Per-feature recovery record for the seven unresolved producers/clocks."""
    record = json.loads(PRODUCERS_PATH.read_text(encoding="utf-8"))
    rows = record["features"]
    if set(rows) != set(UNRESOLVED_SEVEN):
        raise ProvenanceError("PRODUCER_RECORD_INVALID", "record must name exactly the seven")
    for name, row in rows.items():
        if row["status"] == "RECOVERED":
            if not row.get("producer") or not row.get("reproduction"):
                raise ProvenanceError("PRODUCER_RECORD_INVALID", f"{name}: RECOVERED needs evidence")
        elif row["status"] == "STILL_UNRESOLVED":
            if not row.get("missing"):
                raise ProvenanceError("PRODUCER_RECORD_INVALID", f"{name}: missing object required")
            clock = row.get("clock") or {}
            if clock.get("state") not in CLOCK_STATES:
                raise ProvenanceError("PRODUCER_RECORD_INVALID",
                                      f"{name}: clock.state must be one of {CLOCK_STATES}")
            if clock["state"] == CLOCK_PARTIAL:
                m = clock.get("measurement") or {}
                for field in ("observations_compared",
                              "observations_not_covered_by_declared_clock",
                              "fraction_not_covered", "pattern", "interpretation",
                              "not_a_claim_of", "ruling_owner"):
                    if field not in m:
                        raise ProvenanceError("PRODUCER_RECORD_INVALID",
                                              f"{name}: clock.measurement.{field} required")
                if m["interpretation"] != CLOCK_REFUTATION_NOTE:
                    raise ProvenanceError("PRODUCER_RECORD_INVALID",
                                          f"{name}: clock interpretation was altered")
                for artifact in (clock.get("evidence") or {}).get("artifacts", ()):
                    if len(artifact.get("sha256", "")) != 64 or not artifact.get("path"):
                        raise ProvenanceError("PRODUCER_RECORD_INVALID",
                                              f"{name}: cited artifact needs path + sha256")
            if row.get("blocks") != BLOCKS_TEST_ONLY:
                raise ProvenanceError("PRODUCER_RECORD_INVALID",
                                      f"{name}: blocks must state TEST-only scope")
        else:
            raise ProvenanceError("PRODUCER_RECORD_INVALID", f"{name}: status {row['status']}")
    return deepcopy(rows)


def confirmatory_gate(features: Sequence[str] = MANIFEST) -> dict:
    """Promotion gate for a confirmatory TEST: every clock verified, every producer recovered."""
    check_manifest(features)
    unresolved = [f for f, r in producer_status().items() if r["status"] != "RECOVERED"]
    try:
        fa.gate(list(features), require_verified=True)
    except fa.AvailabilityGateError as exc:
        raise ProvenanceError("PROVENANCE_UNRESOLVED",
                              f"{exc}; unresolved producers {unresolved}") from None
    if unresolved:
        raise ProvenanceError("PROVENANCE_UNRESOLVED", f"unresolved producers {unresolved}")
    return {"status": "PASS"}


#: Successor availability contract PREPARED (Satoshi ruling 2026-10-14) and NOT
#: adopted by the short design: it promotes the five recovered hourly features
#: and keeps the two FRED clocks UNVERIFIED. It is a TEST gate input only.
SUCCESSOR_CONTRACT_PATH = Path(__file__).with_name("feature_availability.successor.v2.json")


def load_successor_contract() -> dict:
    return fa.validate_contract(json.loads(SUCCESSOR_CONTRACT_PATH.read_text(encoding="utf-8")))
