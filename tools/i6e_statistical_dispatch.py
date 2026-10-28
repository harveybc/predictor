"""Continue independent statistical cells after a typed numerical failure.

Scientific engine and its sealed design stay unchanged. Successful and failed
terminals are never repeated. Admission/runtime failures still stop dispatch;
optimizer nonconvergence is a recorded disposition, not a fleet-wide barrier.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.i6e_statistical_campaign import MODELS, SPACINGS, code_digest, cell_path, validate, status, read_input, lookup
from tools.i6e_compact_run import atomic_json, file_hash


def next_cells(root, model, week_parity=None):
    """Only absent cells are runnable; numerical failures remain in the denominator."""
    for week in range(52):
        if week_parity is not None and week%2!=week_parity:
            continue
        for spacing in SPACINGS:
            directory=cell_path(root,model,week,spacing)
            if (directory/"CELL.json").exists():
                record=validate(root,model,week,spacing)
                if record["status"] not in ("COMPLETED","FAILED_CONVERGENCE"):
                    raise ValueError("runtime failure retained; diagnose before continuing this model")
                continue
            if (directory/"CLAIM").exists():
                raise ValueError("interrupted phase fits retained; do not repeat under a new claim")
            yield week,spacing


def run(root,model,week_parity=None):
    pilot=json.loads((root/f"cost_{model}"/"PILOT.json").read_text())
    if pilot["status"]!="COMPLETED" or pilot["code_sha256"]!=code_digest():
        raise ValueError("verified TRAIN cost required")
    cap=pilot["production_memory_bytes"]
    if cap>2147483648:
        raise ValueError("measured envelope exceeds declared reservation")
    for week,spacing in next_cells(root,model,week_parity):
        command=[str(Path.home()/".local/bin/crispdm-run"),"-m",str(cap),"-t","3600",
            "-n",f"i6e-stat-{model.lower()}-w{week:03d}-d{spacing}","--",sys.executable,
            "-m","tools.i6e_statistical_campaign","cell","--output",str(root),"--model",model,
            "--week",str(week),"--spacing",str(spacing)]
        if subprocess.run(command,check=False).returncode:
            raise ValueError("admission/runtime failure; no automatic repeat")
        record=validate(root,model,week,spacing)
        if record["status"] not in ("COMPLETED","FAILED_CONVERGENCE"):
            raise ValueError("unrecognized disposition")
        status(root)
    print(json.dumps(status(root)))


def close_dispositions(root):
    """Close all declared dispositions; only fully successful arms get summaries."""
    import numpy as np
    from tools.i6e_price_metrics import price_errors
    design,data=read_input(root)
    arms={}; failed=0
    for model in MODELS:
        for spacing in SPACINGS:
            cells=[validate(root,model,w,spacing) for w in range(52)]
            bad=[r["week"] for r in cells if r["status"]!="COMPLETED"]
            for week,cell in enumerate(cells):
                if cell["status"] not in ("COMPLETED","FAILED_CONVERGENCE"):
                    raise ValueError("nonterminal/runtime failure: no disposition closure")
                if cell["phase_fits"]!=spacing or len(cell["phases"])!=spacing:
                    raise ValueError("missing phase disposition")
                for name,digest in cell["phases"].items():
                    if file_hash(cell_path(root,model,week,spacing)/name)!=digest:
                        raise ValueError("phase disposition digest mismatch")
            failed+=len(bad)
            if bad:
                arms[f"{model}_{spacing}h"]=dict(status="INCOMPLETE_NUMERICAL_EVIDENCE",failed_weeks=bad,
                    completed=52-len(bad),expected=52,metrics=None)
                continue
            parts=[]
            for week in range(52):
                with np.load(cell_path(root,model,week,spacing)/"PREDICTIONS.npz") as d:
                    selected=data["weeks"]==week
                    ts=data["origins"][selected]
                    current=lookup(data["ts"],data["price"],ts)
                    future=data["future"][selected]
                    if not (np.array_equal(d["timestamps"],ts) and np.array_equal(d["current"],current)
                            and np.array_equal(d["future"],future)):
                        raise ValueError("statistical arm scored rows differ from neural population")
                    parts.append((current,future,d["prediction"].copy()))
            current,future,predicted=(np.concatenate([p[i] for p in parts]) for i in range(3))
            metrics=price_errors(current,future,np.log(future/current),np.log(predicted/current))
            if metrics["n"]!=design["scored_origins"]:
                raise ValueError("statistical arm population incomplete")
            arms[f"{model}_{spacing}h"]=dict(status="COMPLETE",metrics=metrics,completed=52,expected=52)
    report=dict(status="COMPLETE_DISPOSITIONS",scientific_status="COMPLETE" if failed==0 else "INCOMPLETE_EVIDENCE",
        expected=312,completed=312-failed,failed_numerical=failed,arms=arms,test_read=False,
        design_sha256=design["design_sha256"])
    atomic_json(root/"NUMERICAL_CLOSURE.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=MODELS,required=True)
    parser.add_argument("--week-parity",type=int,choices=(0,1))
    args=parser.parse_args()
    run(args.output,args.model,args.week_parity)


if __name__=="__main__":
    main()
