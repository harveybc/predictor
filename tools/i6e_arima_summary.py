"""Close and publish complete ARIMA arms independently of unfinished ARIMAX.

Each arm requires all 52 successful weeks and exact registered common origins.
This does not mark the combined statistical experiment complete or open TEST.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.i6e_statistical_campaign import read_input, validate, cell_path, lookup
from tools.i6e_price_metrics import price_errors
from tools.i6e_compact_run import atomic_json, file_hash
from tools.i6e_publish_olap import publish, report_sha256


def summarize(root):
    """Reject missing/failed weeks and re-derive pooled metrics from predictions."""
    design,data=read_input(root)
    arms={}
    for spacing in (72,36,18):
        parts=[]; digests={}
        for week in range(52):
            r=validate(root,"ARIMA",week,spacing)
            if r["status"]!="COMPLETED":
                raise ValueError("ARIMA week is not complete; no arm summary")
            directory=cell_path(root,"ARIMA",week,spacing)
            digests[str(week)]=file_hash(directory/"CELL.json")
            with np.load(directory/"PREDICTIONS.npz",allow_pickle=False) as d:
                selected=data["weeks"]==week
                current=lookup(data["ts"],data["price"],data["origins"][selected])
                if not (np.array_equal(d["timestamps"],data["origins"][selected])
                        and np.array_equal(d["future"],data["future"][selected])
                        and np.array_equal(d["current"],current)):
                    raise ValueError("ARIMA arm population changed")
                parts.append((current,d["future"].copy(),d["prediction"].copy()))
        current,future,predicted=(np.concatenate([p[i] for p in parts]) for i in range(3))
        metrics=price_errors(current,future,np.log(future/current),np.log(predicted/current))
        if metrics["n"]!=design["scored_origins"]:
            raise ValueError("ARIMA scored denominator differs from registered population")
        arms[str(spacing)]=dict(metrics=metrics,cell_digests=digests,weeks=52)
    result=dict(status="COMPLETE_ARIMA_ONLY",combined_statistical_complete=False,arms=arms,
        test_read=False,design_sha256=design["design_sha256"])
    atomic_json(root/"ARIMA_CLOSURE.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--url")
    p.add_argument("--token-file",type=Path)
    p.add_argument("--code-commit",required=True)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    summary=summarize(args.root)
    reports=[]
    for spacing,arm in summary["arms"].items():
        values=arm["metrics"]
        metrics=[dict(metric="Price_"+name,value=values[key],split="validation",horizon=72,unit=unit)
            for name,key,unit in (("MAE","mae","USD_per_EUR"),("MSE","mse","USD_per_EUR_squared"),
                ("Naive_MAE","naive_mae","USD_per_EUR"),("Naive_MSE","naive_mse","USD_per_EUR_squared"),
                ("Skill_MAE","skill_mae","ratio"))]
        metrics.append(dict(metric="Scored_Rows",value=values["n"],split="validation",horizon=72,unit="count"))
        report=dict(experiment_key=f"i6e:arima72:2024:pooled:d{spacing}",
            experiment_set_key="i6e:arima72:2024:pooled",actor="predictor",lake="LOCAL_RETAINED",
            lineage="UNVERIFIED",config_sha256=summary["design_sha256"],code_commit=args.code_commit,
            project="predictor",phase="I6E_ARIMA_POOLED",tags=dict(model="ARIMA",spacing_hours=spacing,
                seed="6102026",weeks="52",test_read="false",closure_sha256=file_hash(args.root/"ARIMA_CLOSURE.json")),metrics=metrics)
        report["report_sha256"]=report_sha256(report)
        reports.append(report)
    publish(reports,args)
    print(json.dumps({s:a["metrics"] for s,a in summary["arms"].items()}))


if __name__=="__main__":
    main()
