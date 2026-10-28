"""Generate a compact progress diagram and cost-based ETA from retained evidence.

No model fitting, remote polling or database access occurs here. The durable
coordinator collects terminals; this renderer is callable at any moment.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

from tools.i6e_compact_run import atomic_json
from tools.i6e_statistical_campaign import status, validate, cell_path


def progress(root):
    """Estimate remaining work separately by model and resolution, never by GPU load."""
    state=status(root)
    remaining={}
    for model in ("ARIMA","ARIMAX"):
        cost=root/f"cost_{model}"/"PILOT.json"
        if not cost.exists():
            remaining[model]=None
            continue
        pilot=json.loads(cost.read_text())
        projected=0
        runtime_failure=False
        for spacing in (72,36,18):
            records=[]
            for week in range(52):
                if (cell_path(root,model,week,spacing)/"CELL.json").exists():
                    records.append(validate(root,model,week,spacing))
            good=[r for r in records if r["status"]=="COMPLETED"]
            runtime_failure |= any(r["status"] not in ("COMPLETED","FAILED_CONVERGENCE") for r in records)
            # Numerical failures consume real CPU too; do not estimate only from winners.
            unit=(sum(r["elapsed_seconds"] for r in records)/len(records) if records else
                  next(r["elapsed_seconds"]*spacing for r in pilot["records"] if r["spacing"]==spacing))
            projected+=(52-len(records))*unit
        remaining[model]=None if runtime_failure else projected
    placement=root/"PLACEMENT.json"
    slots=json.loads(placement.read_text())["model_parallel_slots"] if placement.exists() else {}
    wall={model:None if value is None else value/slots.get(model,1) for model,value in remaining.items()}
    measured=[v for v in wall.values() if v is not None]
    eta=max(measured) if len(measured)==2 else None
    state.update(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        estimated_remaining_seconds_by_model=remaining,estimated_parallel_remaining_seconds=eta,
        model_parallel_slots=slots,
        eta_note="Projection, not deadline. Assumes declared slots remain healthy. TRAIN phase cost then same-resolution cell means; delivery excluded.")
    atomic_json(root/"PROGRESS_STATUS.json",state)
    return state


def render(root, output):
    """Actual completed/expected counters, with no invented master-plan percentages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    state=progress(root)
    neural=root.parent/"weekly_compact72_v1"
    closure=json.loads((neural/"CAMPAIGN_STATUS.json").read_text())
    prices=json.loads((neural/"PRICE_CLOSURE.json").read_text())
    publication=json.loads((neural/"olap/PUBLICATION.json").read_text())
    rows=[("Neural: 52 weekly refits x 3 resolutions",closure["completed"],156),
          ("Original-price reconciliation",len(prices["weekly"]) if prices["status"]=="COMPLETE" else 0,156),
          ("OLAP: reports read back exactly",publication["reports"] if publication["status"]=="PUBLISHED_READBACK_VERIFIED" else 0,156),
          ("ARIMA: cell dispositions",sum(state["by_model"]["ARIMA"][k] for k in ("completed","failed")),156),
          ("ARIMAX: cell dispositions",sum(state["by_model"]["ARIMAX"][k] for k in ("completed","failed")),156)]
    fig,ax=plt.subplots(figsize=(12,5.4),facecolor="white")
    for i,(label,done,total) in enumerate(rows):
        ax.barh(i,1,color="#e8ecef",height=.48)
        ax.barh(i,done/total,color="#14856b" if done==total else "#227dad",height=.48)
        ax.text(1.025,i,f"{done}/{total} ({100*done/total:.1f}%)",va="center",fontsize=11)
        if i>=3:
            failed=state["by_model"]["ARIMA" if i==3 else "ARIMAX"]["failed"]
            if failed:
                ax.barh(i,failed/total,left=(done-failed)/total,color="#bc6c25",height=.48)
    ax.set_yticks(range(len(rows)),[r[0] for r in rows],fontsize=11)
    ax.invert_yaxis(); ax.set_xlim(0,1.25); ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y",length=0)
    fig.suptitle("I6-E | 72-hour origin-anchored forecast",fontsize=18,fontweight="bold",x=.05,ha="left")
    eta=state["estimated_parallel_remaining_seconds"]
    note="ETA not yet measurable" if eta is None else f"Estimated remaining statistical compute: {eta/3600:.1f} h"
    fig.text(.05,.04,note+f" | {state['failed']} numerical failures retained | TEST closed",fontsize=10,color="#414a50")
    fig.subplots_adjust(left=.37,right=.96,top=.83,bottom=.13)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor="white")
    plt.close(fig)
    print(json.dumps(state))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--png",type=Path,required=True)
    args=parser.parse_args()
    render(args.root,args.png)


if __name__=="__main__":
    main()
