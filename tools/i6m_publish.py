"""Retryable OLAP outbox, independent of fitting, with exact value readback."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.parse

from tools.i6a_publish_olap import request_json, token_from_file
from tools.i6m_queue import atomic_json, seal, verify

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"olap/store/src"))
from predictor_olap_store.query import report_sha256


def report(cell, design, pin):
    """Price units and user-signed skill; never borrow a baseline from another h."""
    h = cell["arm"]["horizon_hours"]
    values = cell.get("metrics")
    items = []
    if values is not None:
        if values["horizon_hours"] != h or values["naive_horizon_hours"] != h:
            raise ValueError("OLAP horizon/naive mismatch")
        for name, key, unit in (("Price_MAE", "mae", "USD_per_EUR"),
                                ("Price_MSE", "mse", "USD_per_EUR_squared"),
                                ("Naive_MAE", "naive_mae", "USD_per_EUR"),
                                ("Naive_MSE", "naive_mse", "USD_per_EUR_squared"),
                                ("Skill_Error_Relative_To_Naive", "skill_mae", "ratio"),
                                ("Scored_Rows", "n", "count")):
            if values[key] is not None:
                items.append(dict(metric=name, value=values[key], horizon=h, split="validation", unit=unit))
    for name, value, unit in (("Cell_Seconds", cell["elapsed_seconds"], "seconds"),
                              ("Completed", int(cell["status"] == "COMPLETED"), "count"),
                              ("Cgroup_Peak", cell.get("cgroup_peak_bytes"), "bytes"),
                              ("Parameters", cell.get("parameters"), "count"),
                              ("Optimizer_Updates", cell.get("training", {}).get("observed_updates"), "count")):
        if value is not None:
            items.append(dict(metric=name, value=value, horizon=h, split="validation", unit=unit))
    for naive in cell.get("diagnostic_naives", []):
        items.append(dict(metric=f"Diagnostic_Naive_MAE_H{naive['horizon_hours']}", value=naive["naive_mae"],
                          horizon=naive["horizon_hours"], split="validation", unit="USD_per_EUR"))
    result = dict(experiment_key="i6m:"+design["sha256"][:12]+":2024:EURUSD:"+cell["job_id"],
                  experiment_set_key="i6m:2024:EURUSD:"+design["sha256"][:12],
                  actor="predictor", lake="LOCAL_RETAINED", lineage="UNVERIFIED",
                  config_sha256=design["sha256"], code_commit=pin, project="predictor", phase="I6E_MULTI_HORIZON",
                  tags=dict(status=cell["status"], model=cell["model"], seed=str(cell["seed"]),
                            horizon_hours=str(h), spacing_hours=str(cell["arm"]["spacing_hours"]),
                            samples=str(cell["arm"]["samples"]), forecast_steps=str(cell["arm"]["forecast_steps"]),
                            week=str(cell["week"]["ordinal"]), cell_sha256=cell["sha256"],
                            skill_formula=design["skill_formula"], skill_negative_is_better="true", test_read="false"),
                  metrics=items)
    result["report_sha256"] = report_sha256(result)
    return result


def publish(root, url, token_file, pin, limit=20):
    from tools.i6e_publish_olap import verify_readback
    design = verify(json.loads((root/"DESIGN.json").read_text()))
    token = token_from_file(token_file)
    with sqlite3.connect(root/"QUEUE.db") as conn:
        rows = conn.execute("SELECT id,terminal FROM jobs WHERE terminal IS NOT NULL AND published=0 LIMIT ?", (limit,)).fetchall()
    for job_id, text in rows:
        cell = verify(json.loads(text))
        body = report(cell, design, pin)
        out = root/"olap"/job_id
        atomic_json(out/"REPORT.json", body)
        response = request_json(url+"/api/v1/metrics", token, "POST", body)
        if response.get("report_sha256") != body["report_sha256"]:
            raise ValueError("OLAP receipt digest mismatch")
        sql = "SELECT report_sha256,metric,value,unit,horizon,split FROM gov_metric WHERE report_sha256='"+body["report_sha256"]+"' LIMIT 100"
        rows = request_json(url+"/api/v1/query?"+urllib.parse.urlencode(dict(sql=sql)), token, "GET")["rows"]
        verify_readback([body], rows)
        atomic_json(out/"RECEIPT.json", seal(dict(status="PUBLISHED_READBACK_VERIFIED", report_sha256=body["report_sha256"], cell_sha256=cell["sha256"])))
        with sqlite3.connect(root/"QUEUE.db") as conn:
            conn.execute("UPDATE jobs SET published=1 WHERE id=? AND terminal=?", (job_id, text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    while True:
        try:
            publish(a.root, a.url.rstrip("/"), a.token_file, a.pin)
            atomic_json(a.root/"OLAP_STATUS.json", dict(status="FOLLOWING", heartbeat=time.time()))
        except Exception as exc:
            atomic_json(a.root/"OLAP_STATUS.json", dict(status="RETRY_PENDING_NO_REFIT", heartbeat=time.time(), reason=str(exc)))
            if not a.follow:
                raise
        if not a.follow:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
