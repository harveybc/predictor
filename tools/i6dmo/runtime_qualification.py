"""Versioned runtime qualifications of CLOSED results (MUSASHI_AUDIT_20261015 finding 1).

A qualification is an ADDED, machine-readable statement about already-closed evidence. It never
edits, reseals, reruns or re-grades the closed receipts, errors or percentages it qualifies.

``citation_problems`` is the documentation gate: any text that cites the qualified results
(campaign ids, fit identities, headline numbers or the closure return) must also cite the
qualification record by file name, unless the record lists it as an exemption with a reason.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "i6dmo.runtime_qualification.v1"
RECORD_NAME = "RUNTIME_QUALIFICATION_MG2_MG12.json"
RECORD_PATH = REPO / "docs/i6n/direct_multi_output_static" / RECORD_NAME


class QualificationError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_sha256(record: Mapping) -> str:
    """Self digest over the record body without its own ``record_sha256`` field."""
    return hashlib.sha256(_canonical({k: v for k, v in record.items()
                                      if k != "record_sha256"}).encode()).hexdigest()


def load_record(path: Path = RECORD_PATH) -> Dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise QualificationError("QUALIFICATION_SCHEMA", str(record.get("schema")))
    if record.get("record_sha256") != record_sha256(record):
        raise QualificationError("QUALIFICATION_SEAL", "the qualification record was modified")
    return record


def citation_pattern(record: Mapping) -> re.Pattern:
    tokens = record["citation_tokens"]
    return re.compile("|".join(re.escape(t) for t in tokens))


def citation_problems(files: Iterable[Path], record: Mapping, *, root: Path = REPO) -> List[str]:
    """Files that cite the qualified results but neither cite the record nor are exempt."""
    rx = citation_pattern(record)
    exempt = set(record["exemptions"])
    out = []
    for f in files:
        f = Path(f)
        rel = f.resolve().relative_to(root.resolve()).as_posix() if f.is_absolute() else f.as_posix()
        if rel in exempt or rel.endswith(RECORD_NAME) or \
                any(e.endswith("/") and rel.startswith(e) for e in exempt):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8") if not f.is_absolute() else f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        if rx.search(text) and RECORD_NAME not in text:
            out.append(rel)
    return sorted(out)
