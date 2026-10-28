#!/usr/bin/env python3
"""Deliver artifact readiness after both T3 shards finish; no ranking or training."""
import argparse
from pathlib import Path
import shlex
import subprocess

from collect_expert_package import validate_mirror
import expert_package_worker as w


def assemble(coverage, peer):
    if (coverage.get("total") != 21 or coverage.get("done") != 21 or coverage.get("failed") != 0
            or coverage.get("pending") != 0):
        raise ValueError("OWN_SHARD_NOT_COMPLETE")
    if (peer.get("schema") != "t3_peer_preflight.v1" or peer.get("state") != "ARTIFACTS_VERIFIED"
            or peer.get("count") != 22):
        raise ValueError("PEER_SHARD_NOT_VERIFIED")
    expected = {f"t3_unit_{i:02d}" for i in range(43)}
    own = coverage["units"]
    if (set(own) != {f"t3_unit_{i:02d}" for i in range(1, 43, 2)}
            or set(peer["units"]) != expected - set(own)):
        raise ValueError("T3_JOIN_MISSING_OR_DUPLICATE_UNITS")
    rows, mirrors = {}, set()
    for name, row in own.items():
        receipt_file = Path(row["receipt"])
        root = receipt_file.parent.parent
        if root not in mirrors:
            validate_mirror(root)
            if w.digest(root / "T3_MANIFEST.json") != peer["manifest_sha256"]:
                raise ValueError("T3_JOIN_MANIFEST_MISMATCH")
            mirrors.add(root)
        if w.digest(receipt_file) != row["receipt_sha256"]:
            raise ValueError("T3_JOIN_RECEIPT_CHANGED")
        receipt = w.read(receipt_file)
        if (receipt["job"]["name"] != name or receipt["config_sha256"] != row["config_sha256"]
                or receipt["support_hash"] != peer["support_hash"]
                or receipt["consumed_features"] != peer["consumed_features"]
                or set(receipt["metrics"]) != {"20261019", "20261020", "20261021"}):
            raise ValueError("T3_JOIN_PAIRING_MISMATCH")
        rows[name] = {"origin": row["origin"], "receipt_sha256": row["receipt_sha256"],
                      "result_sha256": receipt["hashes"]["E3M.json"]}
    for name, row in peer["units"].items():
        rows[name] = {"origin": "peer", "result_sha256": row["hashes"]["E3M.json"],
                      "unit_identity": row["unit_identity"], "command_identity": row["command_identity"]}
    return {"schema": "t3_joined_artifact_readiness.v1", "state": "READY_FOR_PER_GROUP_DECISION_AUDIT",
            "count": len(rows), "units": rows, "support_hash": peer["support_hash"],
            "manifest_sha256": peer["manifest_sha256"], "columns": len(peer["consumed_features"]),
            "seeds": [20261019, 20261020, 20261021], "checkpoint_forward_replayed": False,
            "scientific_release": False, "decision_executed": False, "combined_fit_started": False,
            "next_owner_action": "Apply frozen T3 rule, then the combined confirmation; retain replay and final audit."}


def tick(cfg):
    coverage = w.read(cfg["coverage"])
    if coverage.get("failed"):
        return {"state": "RECOVERY_REQUIRED", "failed": coverage["failed"]}
    if coverage.get("done") != 21:
        return {"state": "WAITING_FOR_OWN_SHARD", "done": coverage.get("done")}
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", cfg["peer_host"],
                    shlex.join(cfg["peer_command"])], check=True, timeout=120)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    cfg["peer_host"] + ":" + cfg["peer_report"], cfg["local_peer_report"]], check=True, timeout=60)
    result = assemble(coverage, w.read(cfg["local_peer_report"]))
    result.update(coverage_sha256=w.digest(cfg["coverage"]), peer_report_sha256=w.digest(cfg["local_peer_report"]))
    w.atomic(cfg["output"], result)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    cfg["output"], cfg["destination"]], check=True, timeout=60)
    subprocess.run(["systemctl", "--user", "disable", "--now", cfg["timer"]], check=True)
    return {"state": result["state"], "count": result["count"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    print(tick(w.read(a.config)))
