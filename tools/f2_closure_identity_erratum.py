#!/usr/bin/env python3
"""Create an additive identity erratum for an immutable F2 closure.

The original closure and metrics are never rewritten.  The erratum binds their
bytes by SHA-256 and derives one scientific identity from every retained
``EVIDENCE_*.json`` file.  Mixed evidence is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tools.eth_forecast_closure import _canonical_identity, _literature, _receipt_identity


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_erratum(closure_path, evidence_paths):
    closure_path = Path(closure_path)
    closure = json.loads(closure_path.read_text())
    evidence_paths = sorted(Path(path) for path in evidence_paths)
    if not evidence_paths:
        raise ValueError("at least one evidence file is required")

    identities = {_canonical_identity(_receipt_identity(json.loads(path.read_text())))
                  for path in evidence_paths}
    if len(identities) != 1:
        raise ValueError("mixed campaign identity across retained evidence")
    identity = json.loads(next(iter(identities)))
    old_identity = closure.get("campaign_identity")
    old_literature = closure.get("literature")
    corrected_literature = _literature(identity)
    if old_identity == identity and old_literature == corrected_literature:
        raise ValueError("closure identity already matches retained evidence")

    return {
        "schema": "f2.closure_identity_erratum.v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "identity_and_literature_metadata_only",
        "metrics_changed": False,
        "original": {
            "path": closure_path.name,
            "sha256": _sha256(closure_path),
            "campaign_identity": old_identity,
            "literature": old_literature,
        },
        "retained_evidence": {
            "count": len(evidence_paths),
            "files": [{"path": path.name, "sha256": _sha256(path)} for path in evidence_paths],
        },
        "corrected": {"campaign_identity": identity, "literature": corrected_literature},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closure", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evidence = sorted(Path(args.evidence_dir).glob("EVIDENCE_*.json"))
    erratum = build_erratum(args.closure, evidence)
    Path(args.out).write_text(json.dumps(erratum, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "evidence": len(evidence),
                      "campaign_id": erratum["corrected"]["campaign_identity"]["campaign_id"]}))


if __name__ == "__main__":
    main()
