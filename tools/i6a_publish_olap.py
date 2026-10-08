"""Publish a complete I6-A validation campaign to the governed metrics API.

Reports are explicitly UNVERIFIED lineage: this campaign used retained local
parquets rather than a data-gov delivery. Tokens are read from a mode-0600
environment file and are never included in reports or printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tools import fs4_weekly_wrapper as W
from tools.i6a_architectures import ARMS
from tools.i6a_campaign import verify_result
from tools.i6a_close import close
from tools.i6a_weekly_arch_pilot import make_task, target_horizon_hours

STORE_SRC = Path(__file__).resolve().parent.parent / "olap" / "store" / "src"
sys.path.insert(0, str(STORE_SRC))
from predictor_olap_store.query import report_sha256  # noqa: E402


def token_from_file(path: Path) -> str:
    if path.stat().st_mode & 0o077:
        raise ValueError("TOKEN_FILE_NOT_PRIVATE")
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip().strip('"').strip("'")
    token = values.get("WAREHOUSE_TOKEN")
    if not token:
        raise ValueError("WAREHOUSE_TOKEN_MISSING")
    return token


def experiment_set_key(target):
    base = f"i6a:2024:EURUSD:{target}"
    return base if target == "Y_s_1h" else base + ":metric-horizon-v2"


def metric(name, value, unit, horizon):
    return {"metric": name, "value": value, "split": "validation", "horizon": horizon, "unit": unit}


def make_report(arm, week, record, code_commit, target="Y_s_1h"):
    result = record["result"]
    if result.get("target_id") != target or result.get("horizon_hours") != target_horizon_hours(target):
        raise ValueError("REPORT_TARGET_MISMATCH")
    values = result["metrics"]
    cost = result["cost"]
    horizon = target_horizon_hours(target)
    experiment_key = (f"i6a:2024:w{week:03d}:{arm}" if target == "Y_s_1h"
                      else f"i6a:2024:{target}:metric-horizon-v2:w{week:03d}:{arm}")
    report = {"experiment_key": experiment_key,
              "experiment_set_key": experiment_set_key(target), "actor": "predictor",
              "lake": "LOCAL_RETAINED", "lineage": "UNVERIFIED",
              "config_sha256": result["plan_sha256"], "code_commit": code_commit,
              "project": "predictor", "phase": "I6A",
              "tags": {"arm": arm, "week": str(week), "population": "EURUSD", "target": target,
                       "selected_set_conditioned_on_validation": "true", "test_read": "false",
                       "cell_sha256": record["sha256"], "rows_sha256": result["rows_sha256"]},
              "metrics": [metric("MAE", values["mae"], "log_return", horizon),
                          metric("MSE", values["mse"], "log_return_squared", horizon),
                          metric("Naive_MAE", values["naive_mae"], "log_return", horizon),
                          metric("Naive_MSE", values["naive_mse"], "log_return_squared", horizon),
                          metric("Skill_MAE", result["skill_mae"], "ratio", horizon),
                          metric("Fit_Seconds", cost["fit_seconds"], "seconds", horizon),
                          metric("Parameters", cost["n_params"], "count", horizon),
                          metric("Scored_Rows", result["n_scored"], "count", horizon)]}
    if target != "Y_s_1h":
        report["tags"]["metric_horizon_binding"] = "v2"
    report["report_sha256"] = report_sha256(report)
    return report


def request_json(url, token, method, body=None):
    data = None if body is None else json.dumps(body, sort_keys=True).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"WAREHOUSE_HTTP_{exc.code}: {exc.read(300).decode(errors='replace')}") from exc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--target", default="Y_s_1h")
    ap.add_argument("--result-dir", action="append", required=True, type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument("--token-file", required=True, type=Path)
    ap.add_argument("--code-commit", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    freeze = json.loads(args.freeze.read_text())
    closure = close(freeze, args.result_dir, args.target)
    if closure["state"] != "COMPLETE":
        raise ValueError(f"INCOMPLETE_EVIDENCE: {closure['verified_cells']}/{closure['expected_cells']}")
    reports = []
    for week in range(closure["expected_weeks"]):
        task = make_task(freeze, args.target, week, 2024)
        for arm in ARMS:
            paths = [root / f"{arm}_val2024_week{week}.json" for root in args.result_dir]
            path = next(p for p in paths if p.is_file())
            reports.append(make_report(arm, week, verify_result(path, arm, task), args.code_commit, args.target))
    if args.dry_run:
        print(json.dumps({"state": "READY", "reports": len(reports), "closure_sha256": closure["sha256"]}))
        return 0
    token = token_from_file(args.token_file)
    endpoint = args.url.rstrip("/")
    stored = already = 0
    for report in reports:
        receipt = request_json(endpoint + "/api/v1/metrics", token, "POST", report)
        if receipt.get("report_sha256") != report["report_sha256"]:
            raise ValueError("WAREHOUSE_RECEIPT_IDENTITY_MISMATCH")
        stored += int(receipt.get("stored") is True)
        already += int(receipt.get("already_stored") is True)
    set_key = experiment_set_key(args.target)
    sql = ("SELECT COUNT(DISTINCT r.report_sha256) AS n, COUNT(m.report_sha256) AS metrics "
           "FROM gov_report r LEFT JOIN gov_metric m ON m.report_sha256 = r.report_sha256 "
           f"WHERE r.experiment_set_key = '{set_key}' AND m.horizon = {target_horizon_hours(args.target)} LIMIT 1")
    readback = request_json(endpoint + "/api/v1/query?" + urllib.parse.urlencode({"sql": sql}), token, "GET")
    rows = readback.get("rows") if isinstance(readback, dict) else None
    if (not isinstance(rows, list) or len(rows) != 1
            or int(rows[0].get("n", -1)) != len(reports)
            or int(rows[0].get("metrics", -1)) != sum(len(r["metrics"]) for r in reports)):
        raise ValueError("WAREHOUSE_READBACK_COUNT_MISMATCH")
    print(json.dumps({"state": "PUBLISHED", "reports": len(reports), "stored": stored,
                      "already_stored": already, "readback": readback, "closure_sha256": closure["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
