#!/usr/bin/env python3
"""The four archive cases, run against whatever code is present. Frozen before any fix.

V1.1 of `docs/handoffs/MUSASHI_U1_U4_REVIEW_AND_V1_V4_2026_09_15.md`:

    "Freeze the four-case reproducer before edits. Keep UNKNOWN as the positive control; the
     other three archive cases must fail for semantic reasons."

The finding it reproduces is mine to own: I made the reader verify that the retained bytes
hash to the digest they are filed under, and then treated that as enough. It is not. Integrity
of bytes is not validity of a contract. An `ARCHIVE_RETROSPECTIVE` whose completion lag says
`0s`, or `-1`, or `not-a-duration` hashes perfectly well, and the reader called every one of
them VERIFIED and handed back the lag as though it meant something.

Each case writes through the production writer, records a delivery against that exact digest,
and reads back through the production reader. Nothing is stubbed; the point is what the
deployed path does.

usage:
  temporal_contract_reproducer.py --out RESULT.json [--label BEFORE|AFTER]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))

#: UNKNOWN is the positive control: it is what a retrospective archive must say, and it must
#: keep working. The other three are the counterexamples and must be refused for SEMANTIC
#: reasons — not because their bytes are wrong, which they are not.
CASES = (
    ("UNKNOWN", "the positive control: an archive that admits it does not know"),
    ("0s", "a known lag on an archive whose publication was never observed"),
    ("not-a-duration", "a string that is not a duration at all"),
    (-1, "a negative duration"),
)


def contract_with(lag):
    return {
        "resource_id": "ethusdt_4h.parquet",
        "available_time_column": "open_time",
        "time_unit": "ms",
        "timezone": "utc",
        "availability": {
            "label": "WINDOW_START",
            "completion_lag_max": lag,
            "timezone_evidence": "UNKNOWN",
            "use_class": "ARCHIVE_RETROSPECTIVE",
        },
    }


def run_case(lag, note, work: Path) -> dict:
    from sqlalchemy import text

    from predictor_olap_store.query import Plugin

    store = Plugin()
    store.set_params(sqlite_path=str(work / f"case_{abs(hash(str(lag)))}.sqlite"))
    store.engine()

    body = json.dumps(contract_with(lag), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    outcome = {"declared_lag": lag, "note": note, "contract_sha256": digest,
               "bytes_hash_correctly": True}
    try:
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])
        outcome["write"] = "ACCEPTED"
    except ValueError as exc:
        outcome["write"] = "REFUSED"
        outcome["write_reason"] = str(exc)
        outcome["read"] = None
        return outcome

    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
            " resource_id, role, sha256, bytes, availability_contract_sha256,"
            " verification_state) VALUES ('t', 'd-1', 'l', 'ethusdt_4h.parquet', 'archive',"
            " 'b', 1, :c, 'VERIFIED_TRANSFER')"), {"c": digest})

    answer = store.resolve_delivery_availability("d-1")
    outcome["read"] = answer.get("contract_resolution")
    outcome["read_lag"] = answer.get("completion_lag_max")
    outcome["read_use_class"] = answer.get("use_class")
    outcome["read_reason"] = answer.get("reason")
    return outcome


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="UNLABELLED",
                        help="BEFORE for the frozen reproduction, AFTER once fixed")
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="temporal-repro-"))
    cases = [run_case(lag, note, work) for lag, note in CASES]

    # The whole point, stated as a machine-checkable verdict rather than left to a reader.
    control = cases[0]
    counterexamples = cases[1:]
    verdict = {
        "control_survives": control["write"] == "ACCEPTED"
        and control["read"] == "VERIFIED" and control["read_lag"] == "UNKNOWN",
        "counterexamples_refused": all(
            case["write"] == "REFUSED" or case["read"] != "VERIFIED"
            for case in counterexamples),
        "counterexamples_wrongly_verified": [
            case["declared_lag"] for case in counterexamples
            if case["write"] == "ACCEPTED" and case["read"] == "VERIFIED"],
    }
    body = {"schema": "temporal_contract_reproducer.v1", "label": args.label,
            "generated_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "cases": cases, "verdict": verdict}
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=1))
    return 0 if verdict["control_survives"] and verdict["counterexamples_refused"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
