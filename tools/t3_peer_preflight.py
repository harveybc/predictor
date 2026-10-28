#!/usr/bin/env python3
"""Read-only peer artifact audit; never writes producer receipts or ranks models."""
import argparse
from pathlib import Path

import expert_package_worker as w
from expert_package_t3 import jobs
from prepare_expert_package_t3 import PINS


ASSIGNMENT = "bfa89f30f81f81476e1c79fe08f3108f7e43e09395d163550a2bf09c52c20a9f"


def verify_inputs(root, reference):
    root = Path(root)
    for name, expected in {"T3_MANIFEST.json": PINS["T3_MANIFEST.json"],
                           "T3_UNITS_satoshi.json": ASSIGNMENT}.items():
        if w.digest(root / name) != expected:
            raise ValueError("PEER_MANIFEST_OR_ASSIGNMENT_CHANGED:" + name)
    checked, expected_code = 0, set()
    for filename, expected in reference["pins"].items():
        relative = Path(filename).relative_to(reference["root"])
        if relative.parts[0] not in ("code", "data", "in", "t3"):
            continue
        if w.digest(root / relative) != expected:
            raise ValueError("PEER_INPUT_CHANGED:" + str(relative))
        checked += 1
        if relative.parts[0] == "code":
            expected_code.add(str(relative.relative_to("code")))
    actual_code = {str(p.relative_to(root / "code")) for area in ("app", "tools")
                   for p in (root / "code" / area).rglob("*.py")}
    if not expected_code or expected_code != actual_code:
        raise ValueError("PEER_CODE_FILE_SET_CHANGED")
    return checked


def check(root, reference_file):
    root = Path(root).resolve()
    reference = w.read(reference_file)
    input_count = verify_inputs(root, reference)
    selected = jobs(w.read(root / "T3_MANIFEST.json"), w.read(root / "T3_UNITS_satoshi.json"), worker="satoshi")
    cfg = {"root": str(root), "campaign": "v4_t3", "seeds": reference["seeds"]}
    rows, support, columns = {}, None, None
    for job in selected:
        receipt = w.validate_result(cfg, job)
        if receipt != w.validate_result(cfg, job):
            raise ValueError("PEER_ARTIFACTS_CHANGED_DURING_READ:" + job["name"])
        if support is not None and (receipt["support_hash"] != support or receipt["consumed_features"] != columns):
            raise ValueError("PEER_SUPPORT_OR_COLUMN_ORDER_CHANGED:" + job["name"])
        support, columns = receipt["support_hash"], receipt["consumed_features"]
        rows[job["name"]] = {"unit_identity": job["peer_unit_identity"],
                             "command_identity": job["peer_command_identity"],
                             "hashes": receipt["hashes"], "wall_seconds": receipt["wall_seconds"],
                             "metrics": receipt["metrics"]}
    if verify_inputs(root, reference) != input_count:
        raise ValueError("PEER_INPUT_SET_CHANGED_DURING_READ")
    return {"schema": "t3_peer_preflight.v1", "state": "ARTIFACTS_VERIFIED", "units": rows,
            "count": len(rows), "manifest_sha256": PINS["T3_MANIFEST.json"],
            "assignment_sha256": ASSIGNMENT, "reference_config_file_sha256": w.digest(reference_file),
            "input_files_checked": input_count, "consumed_features": columns, "support_hash": support,
            "historical_execution_attested": False, "checkpoint_forward_replayed": False,
            "metrics_recomputed": False, "ranks_or_promotes": False,
            "limit": "Checks retained artifacts against the shared contract; not a signed execution receipt or replay."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--reference-config", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    destination = Path(a.out).resolve()
    if destination.is_relative_to(Path(a.root).resolve()):
        p.error("audit output must be outside the peer producer root")
    result = check(a.root, a.reference_config)
    w.atomic(destination, result)
    print({"state": result["state"], "count": result["count"], "input_files_checked": result["input_files_checked"]})
