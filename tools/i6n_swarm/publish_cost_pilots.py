"""Publish verified TRAIN pilot C2 metrics (all heads) with exact OLAP readback.

Uses the existing private warehouse configuration; never logs its credentials.
Local completion, OLAP publication and blockchain finalization are distinct.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import urllib.parse

from tools import i6n_publish as base
from tools.i6n_search_publish import verified_search
from tools.i6n_swarm import common


def publish(root, warehouse):
    root = Path(root)
    statuses = {}
    for mirror in sorted((root / "mirror").glob("*")):
        if not (mirror / "RESULT.json").exists():
            continue
        cost = json.loads((mirror / "RESULT.json").read_text())
        if cost.get("state") != "COMPLETED":
            continue
        canonical = root / mirror.name / "run"
        if not (canonical / "RESULT.json").exists():
            shutil.copytree(mirror, canonical, dirs_exist_ok=True)
        if common.file_sha256(canonical / "RESULT.json") != common.file_sha256(mirror / "RESULT.json"):
            raise ValueError("canonical pilot differs from verified mirror")
        for candidate in sorted((canonical / "binding/candidates").glob("*")):
            if not candidate.is_dir():
                continue
            marker = candidate / "olap_cost/STATUS.json"
            terminal_sha = common.file_sha256(candidate / "RESULT.json")
            if marker.exists():
                done = json.loads(marker.read_text())
                if done.get("terminal_sha256") != terminal_sha:
                    raise ValueError("published pilot changed")
                statuses[mirror.name] = done
                continue
            result = verified_search(candidate)
            reports = []
            for split in ("train", "inner"):
                for path in sorted(candidate.glob(split+"*.c2.json")):
                    projected = dict(scope="TRAIN_COST_PILOT", state="COMPLETED",
                                     identity=result["identity"]["cell"], metrics=json.loads(path.read_text()))
                    for report in base.reports(projected, terminal_sha, cost["code_commit"]):
                        report["experiment_key"] += ":cost:"+split
                        report["phase"] = "I6N_TRAIN_COST_PILOT"
                        report["tags"].update(scope="TRAIN_INNER_COST_PILOT_NOT_ANNUAL",
                            scoring_split=split, supervision=result["identity"]["effective_config"]["output_mode"],
                            observed_peak_bytes=str(cost["observed_peak_bytes"]),
                            cost_receipt_sha256=common.file_sha256(canonical/"RESULT.json"),
                            selection_role="COST_ONLY_NOT_RANKED", chain_finalized="false")
                        for metric in report["metrics"]:
                            metric["split"] = split
                        report["report_sha256"] = base.report_sha256(report)
                        reports.append(report)
            token = base.token_from_file(Path(warehouse["token_file"]))
            url = warehouse["url"].rstrip("/")
            rows_written = 0
            for report in reports:
                response = base.request_json(url+"/api/v1/metrics", token, "POST", report)
                sha = report["report_sha256"]
                if response.get("report_sha256") != sha:
                    raise ValueError("warehouse receipt mismatch")
                sql = ("SELECT report_sha256,metric,value,unit,horizon,split FROM gov_metric "
                       "WHERE report_sha256='"+sha+"' LIMIT 1000")
                rows = base.request_json(url+"/api/v1/query?"+urllib.parse.urlencode({"sql":sql}), token, "GET")["rows"]
                base.verify_readback([report], rows)
                rows_written += len(rows)
            done = dict(state="PUBLISHED_READBACK_VERIFIED", terminal_sha256=terminal_sha,
                        reports=len(reports), metric_rows=rows_written, chain_finalized=False)
            common.atomic_json(marker.parent/"REPORTS.json", reports, mode=0o600)
            common.atomic_json(marker, done, mode=0o600)
            statuses[mirror.name] = done
    common.atomic_json(root/"OLAP_STATUS.json", statuses, mode=0o600)
    return statuses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warehouse-config", required=True)
    p.add_argument("--root", action="append", required=True)
    args = p.parse_args()
    warehouse = json.loads(Path(args.warehouse_config).read_text())
    for root in args.root:
        with (Path(root)/"OLAP.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(publish(root, warehouse)))


if __name__ == "__main__":
    main()
