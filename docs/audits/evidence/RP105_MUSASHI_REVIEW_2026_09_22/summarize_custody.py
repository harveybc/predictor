"""Publish non-secret maintenance and retained-metadata audit evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backup", type=Path, required=True)
    p.add_argument("--private-receipt", type=Path, required=True)
    p.add_argument("--pre-report", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    a = p.parse_args()
    maintenance = read(a.private_receipt)
    public = {k: maintenance[k] for k in ("state", "provider_sha256", "before", "after", "rehearsal")}
    public["snapshot"] = {k: maintenance["snapshot"][k] for k in ("kind", "verified", "source_counts", "source_digests", "relations_differing_from_source")}
    public["operation_seconds"] = maintenance["finished_unix"] - maintenance["started_unix"]
    public["restart_scope"] = "one intentional warehouse stop/start; no other service restarted"
    (a.evidence / "WAREHOUSE_ADOPTION.json").write_text(json.dumps(public, indent=2) + "\n")
    live = {x["unit"]: x for x in read(a.evidence / "LIVE_RECEIPTS.json")["rows"]}
    report = read(a.pre_report)
    report_sha = digest(a.pre_report)
    rows = {x["unit"]: x for x in report["verification"]["rows"]}
    output = {"pre_deletion_report_sha256": report_sha, "rows": [],
              "scope": "retained bytes versus preserved pre-deletion report and independently queried accepted artifacts; NOT re-estimation from deleted arrays"}
    for unit in sorted(live):
        folder = a.backup / "attempts" / unit
        marker_path = folder / "PREDICTIONS_DELETED.json"
        if not marker_path.exists():
            continue
        marker, record = read(marker_path), read(folder / "cell.json")
        vault_sha = digest(folder / "METRICS_VAULT.json")
        row = {"unit": unit, "record_sha256": digest(folder / "cell.json"), "vault_sha256": vault_sha,
               "record_matches_live_accepted": digest(folder / "cell.json") == live[unit]["record_digest"] and live[unit]["record_bound"] and live[unit]["accepted"],
               "checkpoint_matches_live_accepted": live[unit]["checkpoint_bound"],
               "vault_matches_pre_deletion_report": vault_sha == rows[unit]["recomputed"]["metrics_vault_sha256"],
               "vault_matches_worker_current": vault_sha == live[unit]["vault_digest"],
               "report_resolves": marker["closure_report_sha256"] == report_sha,
               "arrays_identity_matches_record": marker["arrays_sha256"] == record["arrays_sha256"],
               "pre_report_verified": rows[unit]["verified"]}
        output["rows"].append(row)
    output["pass"] = len(output["rows"]) == 6 and all(
        all(v is True for k, v in row.items() if k not in ("unit", "record_sha256", "vault_sha256"))
        for row in output["rows"])
    output["backup_files"] = sum(1 for p in a.backup.rglob("*") if p.is_file())
    output["backup_bytes"] = sum(p.stat().st_size for p in a.backup.rglob("*") if p.is_file())
    (a.evidence / "RETAINED_CUSTODY.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
