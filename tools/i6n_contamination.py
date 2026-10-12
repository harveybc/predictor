"""Refuse point-in-time-contaminated I6N results as valid skill (owner order 2026-10-12).

Every H4 result (and any H72 result) produced under the 20-feature manifest
consumed STLFSI4 about five days before publication.  The machine-readable label
``docs/audits/evidence/I6N_STLFSI4_EXCLUSION_20261012/POINT_IN_TIME_CONTAMINATION.json``
lists the identities concerned and the label
``DEVELOPMENT_TUNING_POINT_IN_TIME_CONTAMINATED``.  Historical artifacts are not
edited; successor tooling consults this module and refuses to rank, compare,
adopt or resume anything that carries a labelled identity.

Pure standard library.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LABEL = "DEVELOPMENT_TUNING_POINT_IN_TIME_CONTAMINATED"
SCHEMA = "i6n.point_in_time_contamination.label.v1"
LABEL_PATH = (Path(__file__).resolve().parents[1] / "docs" / "audits" / "evidence"
              / "I6N_STLFSI4_EXCLUSION_20261012" / "POINT_IN_TIME_CONTAMINATION.json")
#: record field -> label identity class.  Aliases cover the names used across
#: terminals, manifests, node status, fleet reports and receipts.
FIELD_CLASSES = {
    "plan_sha256": "plan_sha256",
    "scientific_config_sha256": "scientific_config_sha256",
    "optimization_config_sha256": "scientific_config_sha256",
    "config_digest": "scientific_config_sha256",
    "h4_scientific_config_sha256": "scientific_config_sha256",
    "h72_scientific_config_sha256": "scientific_config_sha256",
    "experiment": "experiment",
    "feature_manifest_sha256": "feature_manifest_sha256",
    "network_id": "network_id",
    "h72_network_id": "network_id",
}
#: Provenance sub-trees that legitimately NAME a superseded identity without carrying it.
SKIP_KEYS = frozenset(("supersedes", "point_in_time", "contaminated_identities"))
IDENTITY_CLASSES = ("plan_sha256", "scientific_config_sha256", "experiment",
                    "feature_manifest_sha256", "network_id")


class ContaminatedResult(ValueError):
    """A result carries a point-in-time-contaminated identity."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = list(reasons)
        super().__init__(f"POINT_IN_TIME_CONTAMINATED: {LABEL}: " + "; ".join(self.reasons))


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def label_sha256(label: Mapping[str, Any]) -> str:
    return _sha({k: v for k, v in label.items() if k != "label_sha256"})


def load_label(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else LABEL_PATH
    label = json.loads(path.read_text(encoding="utf-8"))
    if label.get("schema") != SCHEMA or label.get("label") != LABEL:
        raise ValueError("contamination label schema mismatch")
    identities = label.get("identities")
    if (not isinstance(identities, Mapping) or set(identities) != set(IDENTITY_CLASSES)
            or any(not isinstance(v, list) or any(not isinstance(i, str) or not i for i in v)
                   for v in identities.values())):
        raise ValueError("contamination label identities malformed")
    if label.get("label_sha256") != label_sha256(label):
        raise ValueError("contamination label seal mismatch")
    return label


def _walk(record: Any, prefix: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(record, Mapping):
        for key, value in record.items():
            if key in SKIP_KEYS:
                continue
            yield prefix + str(key), str(key), value
            if isinstance(value, (Mapping, list)):
                yield from _walk(value, prefix + str(key) + ".")
    elif isinstance(record, list):
        for index, value in enumerate(record):
            yield from _walk(value, f"{prefix}{index}.")


def is_contaminated(record: Any, label: Mapping[str, Any] | None = None) -> list[str]:
    """Return the reasons a record (searched recursively) carries a labelled identity."""
    label = load_label() if label is None else label
    sets = {k: set(v) for k, v in label["identities"].items()}
    reasons = []
    for path, key, value in _walk(record):
        klass = FIELD_CLASSES.get(key)
        if klass and isinstance(value, str) and value in sets[klass]:
            reasons.append(f"{path}={value[:16]} ({klass})")
    return reasons


def assert_valid_skill(record: Any, label: Mapping[str, Any] | None = None) -> None:
    reasons = is_contaminated(record, label)
    if reasons:
        raise ContaminatedResult(reasons)


def rank_valid_skill(records: Sequence[Mapping[str, Any]], key: str = "fitness") -> list:
    """Rank by ``key`` (lower is better) only when no record is contaminated."""
    label = load_label()
    for record in records:
        assert_valid_skill(record, label)
    return sorted(records, key=lambda r: r[key])


def compare_skill(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    label = load_label()
    assert_valid_skill(a, label)
    assert_valid_skill(b, label)
