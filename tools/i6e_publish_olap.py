"""Idempotent I6E neural metric publication with exact live-value readback.

Retained local input lineage stays UNVERIFIED; publishing does not retroactively
turn a local parquet into a governed data-gov delivery. Credentials are never
printed or placed in an evidence file. The report bundle is a physical backup.
"""

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

from tools.i6a_publish_olap import request_json, token_from_file
from tools.i6e_campaign import close, validate
from tools.i6e_compact_run import atomic_json, file_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "olap/store/src"))
from predictor_olap_store.query import report_sha256

SET_KEY = "i6e:compact72:2024:EURUSD"


def make_report(cell, prices, design_sha, code_commit):
    """Keep price and log-return errors separate, with paired baselines for both."""
    metrics = []
    for prefix, values, unit in (("", cell["metrics"], "log_return"),
                                 ("Price_", prices, "USD_per_EUR")):
        for name, key, metric_unit in (("MAE", "mae", unit), ("MSE", "mse", unit+"_squared"),
                                      ("Naive_MAE", "naive_mae", unit),
                                      ("Naive_MSE", "naive_mse", unit+"_squared"),
                                      ("Skill_MAE", "skill_mae", "ratio")):
            metrics.append(dict(metric=prefix+name, value=values[key], split="validation",
                                horizon=72, unit=metric_unit))
    for name, value, unit in (("Scored_Rows", cell["metrics"]["n"], "count"),
                              ("Parameters", cell["parameters"], "count"),
                              ("Optimizer_Updates", cell["training"]["observed_updates"], "count"),
                              ("Fit_Seconds", cell["training"]["elapsed_seconds"], "seconds")):
        metrics.append(dict(metric=name, value=value, split="validation", horizon=72, unit=unit))
    if prices["n"] != cell["metrics"]["n"]:
        raise ValueError("price/log metric population mismatch")
    report = dict(experiment_key=f"{SET_KEY}:w{cell['week_ordinal']:03d}:n{cell['samples']}",
                  experiment_set_key=SET_KEY, actor="predictor", lake="LOCAL_RETAINED",
                  lineage="UNVERIFIED", config_sha256=design_sha, code_commit=code_commit,
                  project="predictor", phase="I6E",
                  tags={"samples": str(cell["samples"]), "week": str(cell["week_ordinal"]),
                        "seed": str(cell["seed"]), "regime": "R0", "test_read": "false",
                        "spacing_hours": str(cell["spacing_hours"]),
                        "cell_sha256": cell["cell_sha256"], "rows_sha256": cell["metrics"]["rows_sha256"]},
                  metrics=metrics)
    report["report_sha256"] = report_sha256(report)
    return report


def verify_readback(reports, rows):
    """Counts alone cannot detect a swapped metric; compare every identity/value."""
    expected = {(r["report_sha256"], m["metric"]): (m["value"], m["unit"], m["horizon"], m["split"])
                for r in reports for m in r["metrics"]}
    actual = {}
    for row in rows:
        key = (row["report_sha256"], row["metric"])
        if key in actual:
            raise ValueError("duplicate warehouse metric")
        actual[key] = (row["value"], row["unit"], row["horizon"], row["split"])
    if actual != expected:
        raise ValueError("warehouse metric identity/value mismatch")


def statistical_reports(root, code_commit):
    """Publish only complete independently reconciled statistical terminals."""
    from tools.i6e_statistical_campaign import close as stat_close, read_design, validate as stat_validate
    stat_close(root)
    design = read_design(root)
    reports = []
    for model in ("ARIMA","ARIMAX"):
        for week in range(52):
            for spacing in (72,36,18):
                cell = stat_validate(root,model,week,spacing)
                folder=root/"cells"/f"{model}_w{week:03d}_d{spacing}"
                phases=[json.loads((folder/f"PHASE_{p:02d}.json").read_text()) for p in range(spacing)]
                values=cell["metrics"]
                metrics=[dict(metric="Price_"+name,value=values[key],split="validation",horizon=72,unit=unit)
                    for name,key,unit in (("MAE","mae","USD_per_EUR"),("MSE","mse","USD_per_EUR_squared"),
                        ("Naive_MAE","naive_mae","USD_per_EUR"),("Naive_MSE","naive_mse","USD_per_EUR_squared"),
                        ("Skill_MAE","skill_mae","ratio"))]
                metrics.extend(dict(metric=name,value=value,split="validation",horizon=72,unit=unit)
                    for name,value,unit in (("Scored_Rows",values["n"],"count"),
                        ("Phase_Fits",spacing,"count"),("Fit_Seconds",cell["elapsed_seconds"],"seconds"),
                        ("Cgroup_Peak",cell["cgroup_peak_bytes"],"bytes")))
                report=dict(experiment_key=f"i6e:stat72:2024:EURUSD:{model}:w{week:03d}:d{spacing}",
                    experiment_set_key="i6e:stat72:2024:EURUSD",actor="predictor",lake="LOCAL_RETAINED",
                    lineage="UNVERIFIED",config_sha256=design["design_sha256"],code_commit=code_commit,
                    project="predictor",phase="I6E_STATISTICAL",tags=dict(model=model,week=str(week),
                        spacing_hours=str(spacing),seed=str(design["seed"]),test_read="false",
                        cell_sha256=file_hash(folder/"CELL.json"),converged=str(all(p["converged"] for p in phases)).lower()),
                    metrics=metrics)
                report["report_sha256"]=report_sha256(report)
                reports.append(report)
    return reports


def publish(reports,args):
    """Idempotent submission plus bounded exact readback, with a physical backup."""
    atomic_json(args.output / "REPORTS.json", reports)
    if args.dry_run:
        print(json.dumps(dict(status="READY", reports=len(reports))))
        return
    if not args.url or not args.token_file:
        raise ValueError("live publication requires url and token file")
    token = token_from_file(args.token_file)
    url = args.url.rstrip("/")
    for report in reports:
        receipt = request_json(url+"/api/v1/metrics", token, "POST", report)
        if receipt.get("report_sha256") != report["report_sha256"]:
            raise ValueError("warehouse receipt digest mismatch")
    rows = []
    for report in reports:
        sql = ("SELECT report_sha256,metric,value,unit,horizon,split FROM gov_metric "
               "WHERE report_sha256='"+report["report_sha256"]+"' LIMIT 100")
        response = request_json(url+"/api/v1/query?"+urllib.parse.urlencode({"sql":sql}), token, "GET")
        rows.extend(response["rows"])
    verify_readback(reports, rows)
    receipt = dict(status="PUBLISHED_READBACK_VERIFIED", reports=len(reports),
                   metrics=sum(len(r["metrics"]) for r in reports),
                   reports_sha256=file_hash(args.output / "REPORTS.json"), test_read=False)
    atomic_json(args.output / "PUBLICATION.json", receipt)
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--price-report", type=Path)
    parser.add_argument("--statistical-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.statistical_root:
        publish(statistical_reports(args.statistical_root,args.code_commit),args)
        return
    if args.campaign is None or args.price_report is None:
        parser.error("neural publication requires campaign and price report")
    close(args.campaign)
    price = json.loads(args.price_report.read_text())
    if (price["status"] != "COMPLETE" or price["exact_target_reconciled"] is not True
            or price["neural_closure_sha256"] != file_hash(args.campaign / "NEURAL_CLOSURE.json")):
        raise ValueError("price closure is not bound to current neural closure")
    prices = {(r["week"], r["samples"]): r for r in price["weekly"]}
    if len(prices) != 156 or len(price["weekly"]) != 156:
        raise ValueError("incomplete or duplicate price population")
    design = json.loads((args.campaign / "DESIGN.json").read_text())
    reports = []
    for week in range(52):
        for samples in (3,5,9):
            cell = validate(args.campaign, week, samples)
            p = prices[(week,samples)]
            if p["cell_sha256"] != cell["cell_sha256"]:
                raise ValueError("price metric cell binding mismatch")
            reports.append(make_report(cell, p["metrics"], design["design_sha256"], args.code_commit))
    publish(reports,args)


if __name__ == "__main__":
    main()
