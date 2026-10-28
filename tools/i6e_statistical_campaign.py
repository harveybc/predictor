"""Cost, run, resume and inspect I6E statistical controls without agent polling.

The two workers own different models, never overlapping cells. A single bounded
child fits one week/resolution. Completed terminals are checked and skipped;
failed or interrupted fits are retained and never silently repeated.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.i6e_compact_run import atomic_json, file_hash
from tools.i6e_price_metrics import load_prices, exact_price_values
from tools.i6e_statistical import phase_grid, delayed_exog, masked_exog, fit_phase, forecast_origins

SPACINGS = (72,36,18)
MODELS = ("ARIMA", "ARIMAX")


def design_digest(design):
    return hashlib.sha256(json.dumps({k:v for k,v in design.items() if k!="design_sha256"},
                                     sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read_design(root):
    design=json.loads((root/"DESIGN.json").read_text())
    if (design.get("design_sha256")!=design_digest(design) or design["code_sha256"]!=code_digest()
            or design["expected_cells"]!=312 or len(design["weeks"])!=52
            or design["spacings"]!=list(SPACINGS) or design["models"]!=list(MODELS)
            or design["test_read"] is not False):
        raise ValueError("statistical design identity/population mismatch")
    return design


def code_digest():
    """Bind fitting, clock and price implementation independently of neural code."""
    paths = [Path(__file__),Path(__file__).with_name("i6e_statistical.py"),
             Path(__file__).with_name("i6e_price_metrics.py")]
    return hashlib.sha256(json.dumps({p.name:file_hash(p) for p in paths},sort_keys=True).encode()).hexdigest()


def epoch(text):
    return int(dt.datetime.fromisoformat(text.replace("Z","+00:00")).timestamp())


def prepare(args):
    """Export only declared members and verified common origins; never open TEST."""
    from tools.fs4_weekly_wrapper import DataStore
    design = json.loads((args.campaign/"DESIGN.json").read_text())
    parent = design["parent"]
    root = args.root
    store = DataStore.from_paths("EURUSD",
        [root/f"ps1/batch_{b}/features_train.parquet" for b in ("001","002","003")],
        root/"fs_causal/inputs/ps1/batch_001/targets_train.parquet",
        [root/f"i6a/validation_2024/batch_{b}/features_train.parquet" for b in ("001","002","003")],
        root/"i6a/validation_2024/batch_001/targets_train.parquet",bar_hours=1)
    prices = load_prices(args.source)
    price = exact_price_values(prices,store.ts)
    x = store.X[:,[store.col[n] for n in parent["members"]]]
    origins,ordinal,truth = [],[],[]
    for week in range(52):
        arrays = []
        for n in (3,5,9):
            with np.load(args.campaign/"cells"/f"week_{week:03d}_samples_{n}"/"PREDICTIONS.npz") as d:
                arrays.append((d["timestamps"].copy(),d["actual"].copy()))
        if not all(np.array_equal(arrays[0][0],a[0]) and np.array_equal(arrays[0][1],a[1]) for a in arrays):
            raise ValueError("neural common population mismatch")
        t,y = arrays[0]
        p0,p1 = exact_price_values(prices,t),exact_price_values(prices,t+72*3600)
        if not np.allclose(np.log(p1/p0),y,rtol=0,atol=1e-12):
            raise ValueError("statistical input target mismatch")
        origins.extend(t); ordinal.extend([week]*len(t)); truth.extend(p1)
    args.output.mkdir(parents=True,exist_ok=True)
    common = dict(ts=store.ts,x=x,price=price)
    np.savez_compressed(args.output/"INPUT.npz",**common,origins=origins,weeks=ordinal,future=truth)
    train = store.ts<epoch("2024-01-01T00:00:00Z")
    np.savez_compressed(args.output/"TRAIN.npz",**{k:v[train] for k,v in common.items()})
    manifest = dict(schema="predictor.i6e.statistical.v1",code_sha256=code_digest(),
        expected_cells=312,expected_phase_fits_per_model=6552,models=list(MODELS),spacings=list(SPACINGS),
        weeks=parent["weeks"],members=parent["members"],order=[1,1,0],trend="n",maxiter=100,
        seed=6102026,statsmodels_version="0.14.5",exogenous_delay_hours=72,weekly_parameter_refit=True,hourly_state_update=True,
        train_years=4,test_read=False,neural_design_sha256=design["design_sha256"],
        neural_closure_sha256=file_hash(args.campaign/"NEURAL_CLOSURE.json"),
        input_sha256=file_hash(args.output/"INPUT.npz"),train_sha256=file_hash(args.output/"TRAIN.npz"),
        original_price_sha256=file_hash(args.source),source_digests=store.digests,scored_origins=len(origins))
    manifest["design_sha256"]=design_digest(manifest)
    atomic_json(args.output/"DESIGN.json",manifest)
    print(json.dumps(dict(status="PREPARED",origins=len(origins),input_sha256=manifest["input_sha256"])))


def read_input(root,train_only=False):
    design = read_design(root)
    name = "TRAIN.npz" if train_only else "INPUT.npz"
    key = "train_sha256" if train_only else "input_sha256"
    if design["code_sha256"]!=code_digest() or file_hash(root/name)!=design[key]:
        raise ValueError("statistical code/input identity mismatch")
    with np.load(root/name,allow_pickle=False) as d:
        data = {k:d[k].copy() for k in d.files}
    if train_only and (data["ts"].max()>=epoch("2024-01-01T00:00:00Z") or "origins" in data):
        raise ValueError("TRAIN-only cost input contains outer observations")
    return design,data


def lookup(ts,values,grid):
    indices = np.searchsorted(ts,grid)
    safe = np.minimum(indices,len(ts)-1)
    hit = (indices<len(ts)) & (ts[safe]==grid)
    output = np.full(len(grid),np.nan)
    output[hit] = values[safe[hit]]
    return output


def phase_fit(data,minimum,cutoff,end,spacing,phase,model,origins):
    """Fit strictly before cutoff, then condition each forecast only through origin."""
    import statsmodels
    if statsmodels.__version__!="0.14.5":
        raise ValueError("statistical production requires the measured worker statsmodels 0.14.5")
    grid = phase_grid(minimum,end,spacing,phase)
    price = lookup(data["ts"],data["price"],grid)
    nfit = int(np.searchsorted(grid,cutoff))
    exog = None
    if model=="ARIMAX":
        raw,observed = delayed_exog(data["ts"],data["x"],grid)
        exog = masked_exog(raw,observed)
    began = time.monotonic()
    fitted = fit_phase(price[:nfit],None if exog is None else exog[:nfit])
    center,scale,mean,sd,columns = fitted._i6e_scaling
    convergence = bool(fitted.mle_retvals.get("converged",False))
    record = dict(phase=phase,spacing=spacing,statsmodels_version=statsmodels.__version__,
        fit_slots=nfit,observed_fit=int(np.isfinite(price[:nfit]).sum()),
        converged=convergence,parameters=fitted.params.tolist(),parameter_names=fitted.param_names,
        optimizer=fitted.mle_retvals,price_center=center,price_scale=scale,
        exog_mean=None if mean is None else mean.tolist(),exog_scale=None if sd is None else sd.tolist(),
        retained_exog_columns=None if columns is None else columns.tolist(),elapsed_seconds=time.monotonic()-began)
    # NumPy scalars/arrays in statsmodels optimizer diagnostics are not JSON primitives.
    record["optimizer"] = {k:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v
                           for k,v in record["optimizer"].items()}
    if len(origins):
        indices = np.searchsorted(grid,origins)
        if not np.array_equal(grid[indices],origins):
            raise ValueError("origin routed to wrong physical phase")
        predictions = forecast_origins(fitted,price,exog,indices,72//spacing) if convergence else None
    else:
        predictions = np.empty(0)
    return record,predictions


def peak():
    """Read this admitted child's own cgroup peak, not host RSS."""
    rel = Path("/proc/self/cgroup").read_text().strip().split("::",1)[1]
    return int((Path("/sys/fs/cgroup")/rel.lstrip("/")/"memory.peak").read_text())


def pilot(args):
    """Measure three distinct TRAIN phase fits; not stochastic repetitions."""
    design,data = read_input(args.output,True)
    folder = args.output/f"cost_{args.model}"
    folder.mkdir(exist_ok=True)
    with (folder/"CLAIM").open("x") as stream:
        stream.write(code_digest())
    records=[]
    began=time.monotonic()
    for spacing in SPACINGS:
        record,_=phase_fit(data,epoch("2020-01-01T00:00:00Z"),epoch("2024-01-01T00:00:00Z"),
            epoch("2024-01-01T00:00:00Z"),spacing,0,args.model,[])
        records.append(record)
        atomic_json(folder/"STATUS.json",dict(status="RUNNING",phase_fits=len(records),expected=3))
    measured=peak()
    result=dict(status="COMPLETED" if all(r["converged"] for r in records) else "FAILED_CONVERGENCE",
        model=args.model,records=records,cgroup_peak_bytes=measured,production_memory_bytes=math.ceil(measured*1.25),
        elapsed_seconds=time.monotonic()-began,code_sha256=code_digest(),train_sha256=design["train_sha256"],
        outer_validation_read=False,test_read=False,seed=design["seed"],
        projected_serial_fit_seconds=52*sum(r["elapsed_seconds"]*r["spacing"] for r in records))
    atomic_json(folder/"PILOT.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"}))


def cell_path(root,model,week,spacing):
    return root/"cells"/f"{model}_w{week:03d}_d{spacing}"


def run_cell(args):
    design,data=read_input(args.output)
    folder=cell_path(args.output,args.model,args.week,args.spacing)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"CLAIM").open("x") as stream:
        stream.write(code_digest())
    began=time.monotonic()
    week=design["weeks"][args.week]
    selected=data["weeks"]==args.week
    origins,future=data["origins"][selected],data["future"][selected]
    prediction=np.full(len(origins),np.nan)
    records=[]
    try:
        for phase in range(args.spacing):
            which=np.flatnonzero((origins//3600)%args.spacing==phase)
            record,pred=phase_fit(data,epoch(week["fit_start"]),epoch(week["cutoff"]),
                epoch(week["end"])+73*3600,args.spacing,phase,args.model,origins[which])
            atomic_json(folder/f"PHASE_{phase:02d}.json",record)
            records.append(record)
            if pred is not None:
                prediction[which]=pred
            atomic_json(folder/"STATUS.json",dict(status="RUNNING",phase_fits=len(records),
                expected=args.spacing,elapsed_seconds=time.monotonic()-began,week=args.week,
                model=args.model,spacing=args.spacing,heartbeat_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        converged=all(r["converged"] for r in records)
        current=lookup(data["ts"],data["price"],origins)
        metrics=None
        if converged and np.isfinite(prediction).all():
            error=prediction-future; naive=current-future
            mae=float(np.abs(error).mean()); nm=float(np.abs(naive).mean())
            metrics=dict(n=len(origins),mae=mae,mse=float(np.square(error).mean()),naive_mae=nm,
                naive_mse=float(np.square(naive).mean()),skill_mae=1-mae/nm,unit="USD_per_EUR")
            np.savez_compressed(folder/"PREDICTIONS.npz",timestamps=origins,current=current,future=future,prediction=prediction)
        result=dict(status="COMPLETED" if metrics is not None else "FAILED_CONVERGENCE",model=args.model,
            week=args.week,spacing=args.spacing,seed=design["seed"],metrics=metrics,
            phase_fits=len(records),elapsed_seconds=time.monotonic()-began,cgroup_peak_bytes=peak(),
            code_sha256=code_digest(),input_sha256=design["input_sha256"],test_read=False,
            phases={f"PHASE_{p:02d}.json":file_hash(folder/f"PHASE_{p:02d}.json") for p in range(args.spacing)},
            predictions_sha256=file_hash(folder/"PREDICTIONS.npz") if metrics else None)
        atomic_json(folder/"CELL.json",result)
    except Exception as exc:
        atomic_json(folder/"CELL.json",dict(status="FAILED",model=args.model,week=args.week,spacing=args.spacing,
            error=f"{type(exc).__name__}: {exc}",elapsed_seconds=time.monotonic()-began,
            code_sha256=code_digest(),input_sha256=design["input_sha256"],test_read=False))
        raise


def validate(root,model,week,spacing):
    folder=cell_path(root,model,week,spacing)
    r=json.loads((folder/"CELL.json").read_text())
    design=read_design(root)
    if (r["model"]!=model or r["week"]!=week or r["spacing"]!=spacing or r["code_sha256"]!=code_digest()
            or r["input_sha256"]!=design["input_sha256"] or r["test_read"] is not False):
        raise ValueError("statistical terminal identity mismatch")
    if r["status"]=="COMPLETED":
        if r["phase_fits"]!=spacing or set(r["phases"])!={f"PHASE_{p:02d}.json" for p in range(spacing)}:
            raise ValueError("incomplete statistical phase population")
        for name,digest in {**r["phases"],"PREDICTIONS.npz":r["predictions_sha256"]}.items():
            if file_hash(folder/name)!=digest:
                raise ValueError("statistical artifact changed")
    return r


def status(root):
    completed=[]; failed=[]
    for model in MODELS:
        for week in range(52):
            for spacing in SPACINGS:
                if (cell_path(root,model,week,spacing)/"CELL.json").exists():
                    r=validate(root,model,week,spacing)
                    (completed if r["status"]=="COMPLETED" else failed).append(r)
    result=dict(status="COMPLETE" if len(completed)==312 else "IN_PROGRESS" if not failed else "INCOMPLETE_EVIDENCE",
        completed=len(completed),failed=len(failed),expected=312,test_read=False,
        by_model={m:dict(completed=sum(r["model"]==m for r in completed),expected=156,
            failed=sum(r["model"]==m for r in failed)) for m in MODELS},
        eta_basis="TRAIN pilot fit projection excludes filtering, delivery and variable optimizer convergence")
    atomic_json(root/"CAMPAIGN_STATUS.json",result)
    return result


def close(root):
    """Require all 312 cells and exactly the registered common 3,712 origins."""
    current=status(root)
    if current["status"]!="COMPLETE":
        raise ValueError("incomplete evidence; no statistical aggregate")
    design,data=read_input(root)
    arms={}
    for model in MODELS:
        for spacing in SPACINGS:
            predictions=[];actual=[];prices=[];stamps=[]
            for week in range(52):
                validate(root,model,week,spacing)
                with np.load(cell_path(root,model,week,spacing)/"PREDICTIONS.npz") as d:
                    selected=data["weeks"]==week
                    if not (np.array_equal(d["timestamps"],data["origins"][selected])
                            and np.array_equal(d["future"],data["future"][selected])
                            and np.array_equal(d["current"],lookup(data["ts"],data["price"],data["origins"][selected]))):
                        raise ValueError("statistical scored population changed")
                    predictions.extend(d["prediction"]);actual.extend(d["future"])
                    prices.extend(d["current"]);stamps.extend(d["timestamps"])
            if len(stamps)!=design["scored_origins"]:
                raise ValueError("statistical pooled population incomplete")
            from tools.i6e_price_metrics import price_errors
            actual,predictions,prices=map(np.asarray,(actual,predictions,prices))
            metrics=price_errors(prices,actual,np.log(actual/prices),np.log(predictions/prices))
            error=np.log(predictions/prices)-np.log(actual/prices)
            baseline=np.log(actual/prices)
            metrics["log_return"]=dict(mae=float(np.abs(error).mean()),mse=float(np.square(error).mean()),
                naive_mae=float(np.abs(baseline).mean()),naive_mse=float(np.square(baseline).mean()))
            arms[f"{model}_{spacing}h"]=metrics
    result=dict(status="COMPLETE",metrics_by_arm=arms,test_read=False,design_sha256=design["design_sha256"])
    atomic_json(root/"STATISTICAL_CLOSURE.json",result)
    return result


def worker(args):
    """Cost once, admit one child at a time, resume only checked missing terminals."""
    cost=args.output/f"cost_{args.model}"/"PILOT.json"
    admission=str(Path.home()/".local/bin/crispdm-run")
    def launch(command,cap,name,extra=()):
        return subprocess.run([admission,"-m",str(cap),"-t","3600","-n",name,"--",sys.executable,
            "-m","tools.i6e_statistical_campaign",command,"--output",str(args.output),
            "--model",args.model,*extra],check=False).returncode
    if not cost.exists():
        if launch("pilot",2147483648,f"i6e-stat-cost-{args.model.lower()}"):
            raise ValueError("TRAIN cost pilot failed; no full dispatch")
    pilot_record=json.loads(cost.read_text())
    if pilot_record["status"]!="COMPLETED" or pilot_record["code_sha256"]!=code_digest():
        raise ValueError("TRAIN cost pilot not admissible")
    cap=pilot_record["production_memory_bytes"]
    if cap>2147483648:
        raise ValueError("measured production reservation exceeds declared 2GiB envelope")
    for week in range(52):
        for spacing in SPACINGS:
            folder=cell_path(args.output,args.model,week,spacing)
            if (folder/"CELL.json").exists():
                if validate(args.output,args.model,week,spacing)["status"]!="COMPLETED":
                    raise ValueError("failed terminal retained; no repeated fit")
                continue
            if launch("cell",cap,f"i6e-stat-{args.model.lower()}-w{week:03d}-d{spacing}",
                ("--week",str(week),"--spacing",str(spacing))):
                raise ValueError("statistical child stopped; partial evidence retained")
            if validate(args.output,args.model,week,spacing)["status"]!="COMPLETED":
                raise ValueError("nonconverged cell retained; no silent naive substitution")
            status(args.output)
    print(json.dumps(status(args.output)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","pilot","cell","worker","status","close"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--week",type=int,choices=range(52))
    p.add_argument("--spacing",type=int,choices=SPACINGS)
    p.add_argument("--root",type=Path)
    p.add_argument("--campaign",type=Path)
    p.add_argument("--source",type=Path)
    args=p.parse_args()
    if args.command in ("status","close"):
        print(json.dumps(status(args.output) if args.command=="status" else close(args.output)))
    else:
        if args.command!="prepare" and args.model is None:
            p.error("model required")
        {"prepare":prepare,"pilot":pilot,"cell":run_cell,"worker":worker}[args.command](args)


if __name__=="__main__":
    main()
