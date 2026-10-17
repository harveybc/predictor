"""Finite, parent-pinned MG2m +18h/conditional single18/window study."""
import argparse
import ast
import copy
import fcntl
import json
import math
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np

from tools.i6n_swarm.common import atomic_json, file_sha256, sha256_json

REPO=Path(__file__).resolve().parents[2]
BASE="796ec8f48b0b1621fb9fa9e60e12c1c37079e51e"
SHORT=[2,3,4,5,6,12]
CELLS=["plus18","single18","window7","window13","window25"]
NAME="i6dmo-mg2m-extension-20261017"
PREFIT_HEAD_ERROR="AttributeError:'DmoBundle' object has no attribute 'heads'"


def campaign_name(spec):
    return NAME+"-single18-repair" if spec.get("study_mode")=="single18_repair" else NAME


def active_cells(spec):
    if not spec.get("study_mode"):return CELLS
    r=spec.get("repair_authority",{})
    if spec["study_mode"]!="single18_repair" or r.get("state")!="FAILED" or \
            r.get("reason")!=PREFIT_HEAD_ERROR or r.get("files")!={}:
        raise ValueError("REPAIR_AUTHORITY_INVALID")
    return ["single18"]


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    atomic_json(Path(path),value,mode=0o600)


def command(args, timeout=600):
    result=subprocess.run([str(a) for a in args],capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError("COMMAND_FAILED:"+str(args[0])+":"+result.stderr[-800:])
    return result.stdout


def authenticate(root, files):
    root=Path(root)
    for rel,expected in files.items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts or file_sha256(root/rel)!=expected:
            raise ValueError("ARTIFACT_DIGEST_CHANGED:"+rel)


def same_arrays(old, new):
    if set(old)!=set(new):
        raise ValueError("PARENT_ARRAY_CHANGED:keys")
    for key in old:
        equal_nan=np.asarray(old[key]).dtype.kind in "fc"
        if not np.array_equal(old[key],new[key],equal_nan=equal_nan):
            raise ValueError("PARENT_ARRAY_CHANGED:"+key)


def model_config(parent, cell):
    result=copy.deepcopy(parent)
    result["horizons_hours"]=SHORT+[18] if cell=="plus18" else [18] if cell=="single18" else SHORT
    if cell.startswith("window"):
        n=int(cell[6:]);result["slots"]=n
        result["market_time"].update(observations=n,span_open_market_hours=2*(n-1))
    return result


def metric_rows(arrays, prediction, horizons):
    rows=[]
    for j,h in enumerate(horizons):
        m=arrays["mask"][:,j];n=int(m.sum())
        if not n or not np.isfinite(prediction[m,j]).all():
            raise ValueError("EMPTY_OR_NONFINITE_HEAD:"+str(h))
        y=arrays["target_price"][m,j].astype("float64")
        origin=arrays["origin_price"][m].astype("float64")
        naive=math.fsum(np.abs(y-origin).tolist())/n
        if naive<=0 or not math.isfinite(naive):
            raise ValueError("ZERO_OR_NONFINITE_NAIVE")
        mae=math.fsum(np.abs(prediction[m,j].astype("float64")-y).tolist())/n
        rows.append(dict(horizon=h,n=n,mae=mae,naive_mae=naive,relative_error=(mae-naive)/naive,
            origins_sha256=sha256_json(arrays["row_id"][m].tolist())))
    return rows


def check_cache(actual, cached):
    for a,b in zip(actual,cached,strict=True):
        for key in ("horizon","n","naive_mae","origins_sha256"):
            if a[key]!=b[key]:
                raise ValueError("NAIVE_CACHE_CHANGED:"+key)


def need_single(plus):
    if plus.get("state")!="COMPLETE":
        return False
    rows=plus["metrics"]["validation"]
    short=math.fsum(r["relative_error"] for r in rows if r["horizon"] in SHORT)/6
    return next(r for r in rows if r["horizon"]==18)["relative_error"]>=0 or short>plus["parent_short_mean"]


def next_cell(results, reuse_control, ordered=None):
    return next((c for c in (CELLS if ordered is None else ordered) if c not in results),None)


def source_proof(spec):
    source=Path(spec["source_fit"]);receipt=read(source/"RECEIPT.json")
    if receipt["status"]!="COMPLETED" or receipt["experiment_id"]!="I6-DMO-SHORT-MG2-MATCHED-R0":
        raise ValueError("WRONG_MG2M_REFERENCE")
    authenticate(source,receipt["artifacts"])
    for rel,expected in receipt["identity"]["code_files_sha256"].items():
        if rel!="tools/i6dmo/model.py" and file_sha256(REPO/rel)!=expected:
            raise ValueError("PARENT_SOURCE_CHANGED:"+rel)
    old=command(["git","show",BASE+":tools/i6dmo/model.py"])
    new=(REPO/"tools/i6dmo/model.py").read_text()
    def definitions(source):
        tree=ast.parse(source)
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!="_normalize_market_config"}
    if definitions(old)!=definitions(new):
        raise ValueError("PARENT_LAYER_OR_BUILDER_CHANGED")
    return receipt


def subset(a, mask):
    return {k:v[mask] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==len(mask) else v for k,v in a.items()}


def window_arrays(a, bundle, grid, slots):
    origins=a["row_id"];p=np.searchsorted(grid,origins)
    if np.any(p>=len(grid)) or not np.array_equal(grid[p],origins):
        raise ValueError("ORIGIN_NOT_IN_MARKET_GRID")
    indices=p[:,None]-2*np.arange(slots-1,-1,-1)
    valid=(indices>=0).all(axis=1)
    stamps=grid[np.clip(indices,0,len(grid)-1)]
    ix=np.searchsorted(bundle.ts,stamps)
    bounded=np.minimum(ix,len(bundle.ts)-1)
    valid&=((ix<len(bundle.ts))&(bundle.ts[bounded]==stamps)).all(axis=1)
    valid&=np.isfinite(bundle.x[bounded]).all(axis=(1,2))
    closure=np.zeros_like(stamps,dtype=bool);closure[:,1:]=np.diff(stamps)>7200
    out={**a,"x":bundle.x[bounded].copy(),"offsets_hours":(stamps-origins[:,None])/3600,
         "closure_crossed":closure.astype(a["closure_crossed"].dtype),"slot_instants":stamps}
    return out,valid


def prepare(spec):
    from tools.i6dmo import train as T, data as D, market_grid as G
    root=Path(spec["data_root"])
    if (root/"PREPARED.json").exists():
        document=read(root/"PREPARED.json");authenticate(root,document["files"]);return document
    root.mkdir(parents=True,exist_ok=True)
    receipt=source_proof(spec);source=Path(spec["source_fit"])
    effective=read(spec["source_effective"])["adapter_effective"]
    data,scales,_=T._prepare_matched(dict(path=spec["population"],
        expected_digest=receipt["identity"]["common_population_sha256"],input_dir=spec["input_dir"]),effective)
    if scales!=read(source/"SCALES.json"):
        raise ValueError("PARENT_MATCHED_SCALER_CHANGED")
    parent_model=read(source/"model/MODEL.json")["config"]
    training=read(source/"TRAINING.json")["settings"]
    if effective["training"]!=training:
        raise ValueError("PARENT_TRAINING_CHANGED")
    view=T._elapsed_view(effective["design"]);view["horizons_hours"]=SHORT+[18]
    for key in D.SEAL_KEYS:view.pop(key,None)
    bundle=D.load_input(spec["input_dir"],view);table=G.build_market_table(bundle,view)
    grid=G.expected_market_hours(bundle);prepared={c:{} for c in CELLS}
    excluded={};window_reuse=True
    for split in T.SPLITS:
        old=data[split];origins=old["row_id"];pos=np.searchsorted(table.origin_ts,origins)
        if np.any(pos>=len(table.origin_ts)) or not np.array_equal(table.origin_ts[pos],origins):
            raise ValueError("ORIGINAL_ORIGIN_MISSING")
        if not table.input_ok[pos].all() or np.any(table.split[pos]!=split.upper()):
            raise ValueError("ORIGINAL_INPUT_OR_SPLIT_CHANGED")
        if (old["mask"] & ~table.label_ok[pos,:6]).any():
            raise ValueError("PARENT_MASK_OUTSIDE_BUSINESS")
        check,valid=window_arrays(old,bundle,grid,13)
        if not valid.all():raise ValueError("PARENT_WINDOW_MISSING")
        same_arrays(old,check)
        mask18=table.label_ok[pos,6]
        target18=np.full(len(origins),np.nan)
        ts=origins[mask18]+18*3600;ix=np.searchsorted(bundle.ts,ts)
        if np.any(ix>=len(bundle.ts)) or not np.array_equal(bundle.ts[ix],ts):
            raise ValueError("EXACT_18H_TARGET_MISSING")
        target18[mask18]=bundle.price[ix]
        extended={**old,"target_price":np.column_stack([old["target_price"],target18]),
                  "mask":np.column_stack([old["mask"],mask18])}
        if "entry_mask" in old:
            extended["entry_mask"]=np.column_stack([old["entry_mask"],table.entry_ok[pos,6]])
        projected={k:v[:,:6] if k in ("target_price","mask","entry_mask") else v for k,v in extended.items()}
        same_arrays(old,projected);prepared["plus18"][split]=extended
        single={k:v[:,6:] if k in ("target_price","mask","entry_mask") else v for k,v in extended.items()}
        prepared["single18"][split]=subset(single,mask18)
        windows={};common=np.ones(len(origins),bool)
        for n in (7,13,25):
            windows[n],ok=window_arrays(old,bundle,grid,n);common&=ok
        excluded[split]=int((~common).sum());window_reuse&=bool(common.all())
        for n in windows:prepared["window"+str(n)][split]=subset(windows[n],common)
    files={};summaries={};original_prediction={}
    for split in T.SPLITS:
        with np.load(source/"predictions"/(split+".npz"),allow_pickle=False) as z:
            original_prediction[split]=z["predicted_price"].copy()
            for key in ("row_id","origin_price","target_price","mask"):
                if not np.array_equal(z[key],data[split][key],equal_nan=True):
                    raise ValueError("PARENT_REPORTED_DATA_CHANGED:"+key)
        np.savez_compressed(root/("REFERENCE."+split+".npz"),prediction=original_prediction[split],
                            **{k:data[split][k] for k in ("row_id","origin_price","target_price","mask")})
    for cell,arrays in prepared.items():
        folder=root/cell;folder.mkdir(exist_ok=True)
        config=model_config(parent_model,cell);hs=config["horizons_hours"];s=copy.deepcopy(scales)
        fit=arrays["fit"];cached=metric_rows(fit,fit["target_price"],hs)
        s.update(naive_mae=[r["naive_mae"] for r in cached],counts=[r["n"] for r in cached],rows=len(fit["row_id"]))
        if cell=="plus18" and s["naive_mae"][:6]!=scales["naive_mae"]:
            raise ValueError("PARENT_SHORT_NAIVE_CHANGED")
        if cell=="window13" and window_reuse:
            s=copy.deepcopy(scales)
        else:
            s.pop("scales_sha256",None);s["scales_sha256"]=sha256_json(s)
        cfg=dict(model=config,original_model=parent_model,training=training,seed=parent_model["seed"],scales=s,
            population=dict(parent_common_population_sha256=receipt["identity"]["common_population_sha256"],
                source_input_sha256=bundle.files["INPUT.npz"],source_receipt_sha256=file_sha256(source/"RECEIPT.json"),
                business=effective["design"]["business"],splits=effective["splits"],test_read=False),
            parent_initial_weights_sha256=read(source/"TRAINING.json")["initial_weights_sha256"],
            parent_native_metrics={sp:metric_rows(data[sp],original_prediction[sp],SHORT) for sp in T.SPLITS},
            cell=cell,reuse_reference=cell=="window13" and window_reuse,
            intentional_difference="append_18h_head_only" if cell=="plus18" else cell,
            monitoring_checkpoint="epoch2_diagnostic_plus_original_selected_best",test_read=False)
        put(folder/"CONFIG.json",cfg);summaries[cell]={}
        for split,a in arrays.items():
            np.savez_compressed(folder/(split+".npz"),**a)
            summaries[cell][split]=[{k:r[k] for k in ("horizon","n","naive_mae","origins_sha256")}
                                    for r in metric_rows(a,a["target_price"],hs)]
        put(folder/"NAIVE_CACHE.json",summaries[cell])
    put(root/"PARENT_TRAINING.json",read(source/"TRAINING.json"))
    shutil.copytree(source/"model",root/"parent_model",dirs_exist_ok=True)
    for p in sorted(root.rglob("*")):
        if p.is_file():files[str(p.relative_to(root))]=file_sha256(p)
    doc=dict(files=files,scope="DEVELOPMENT_2024_TEST_CLOSED",window_control_reused=window_reuse,
        window_origins_excluded=excluded,support=summaries,parent_receipt_sha256=file_sha256(source/"RECEIPT.json"))
    doc["sha256"]=sha256_json(doc);put(root/"PREPARED.json",doc)
    return doc


def load_arrays(folder, split):
    with np.load(Path(folder)/(split+".npz"),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def canonical_layer_config(config):
    config=json.loads(json.dumps(config))
    aliases={}
    for layer in config.get("layers",[]):
        name=layer.get("name","")
        kind=layer.get("class_name")
        if (kind=="InputLayer" and re.fullmatch(r"input_layer(?:_[0-9]+)?",name)) or \
                (kind=="Concatenate" and re.fullmatch(r"concatenate(?:_[0-9]+)?",name)):
            aliases[name]="GENERATED_NODE_"+str(len(aliases))
    def visit(value):
        if isinstance(value,dict):return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list):return [visit(v) for v in value]
        return aliases.get(value,value) if isinstance(value,str) else value
    return visit(config)


def model_proof(config, original, target):
    from predictor_plugins.modular_temporal import weights_hash
    if weights_hash(original.forecast_model)!=config["parent_initial_weights_sha256"]:
        raise ValueError("PARENT_INITIAL_WEIGHT_DIGEST_CHANGED")
    if config["cell"]=="plus18":
        if target.forecast_model.count_params()!=5315:
            raise ValueError("PLUS18_CAPACITY_CHANGED")
        for layer in original.forecast_model.layers:
            other=target.forecast_model.get_layer(layer.name)
            if canonical_layer_config(layer.get_config())!=canonical_layer_config(other.get_config()):
                raise ValueError("PARENT_LAYER_CONFIG_CHANGED:"+layer.name)
            for a,b in zip(layer.get_weights(),other.get_weights(),strict=True):
                if not np.array_equal(a,b):raise ValueError("PARENT_INITIAL_WEIGHT_CHANGED:"+layer.name)
        x=np.random.default_rng(17).normal(size=(3,13,19)).astype("float32")
        inputs=[x,np.tile(np.arange(-24,1,2,dtype="float32"),(3,1)),np.zeros((3,13),"float32")]
        a=original.forecast_model(inputs,training=False)
        b=target.forecast_model(inputs,training=False)
        if not all(np.array_equal(x,y) for x,y in zip(a,b[:6],strict=True)):
            raise ValueError("PARENT_SHORT_INITIAL_FORWARD_CHANGED")
    return dict(parent_initial_weights="EXACT",structural_name_normalization="generated_InputLayer_and_Concatenate_UIDs_only",
                shared_short_layers="EXACT" if config["cell"]=="plus18"
                else "WINDOW_OR_SINGLE_HEAD_CONTRAST",parameters=int(target.forecast_model.count_params()))


def preflight(spec):
    from tools.i6dmo import train as T,model as M
    import tensorflow as tf
    import importlib.metadata
    if tf.__version__!="2.21.0" or importlib.metadata.version("keras")!="3.13.2" \
            or not tf.config.list_physical_devices("GPU"):
        raise ValueError("PREFLIGHT_GPU_RUNTIME_REQUIRED")
    config=read(Path(spec["gpu_data"])/"plus18/CONFIG.json")
    old=M.build_dmo(config["original_model"]);new=M.build_dmo(config["model"])
    proof=model_proof(config,old,new)
    for cell in active_cells(spec):
        folder=Path(spec["gpu_data"])/cell;c=read(folder/"CONFIG.json")
        bundle=M.build_dmo(c["model"])
        if cell=="single18":
            align_single_head(bundle,new)
            probe=[np.ones((1,13,19),"float32"),np.arange(-24,1,2,dtype="float32")[None,:],np.zeros((1,13),"float32")]
            if M.predict_matrix(bundle.forecast_model,probe).shape!=(1,1):
                raise ValueError("SINGLE_HEAD_FORWARD_SHAPE")
            if not np.array_equal(M.predict_matrix(bundle.forecast_model,probe)[:,0],
                                  M.predict_matrix(new.forecast_model,probe)[:,6]):
                raise ValueError("SINGLE_HEAD_INITIAL_FORWARD_MISMATCH")
        a=load_arrays(folder,"fit")
        if not np.array_equal(a["slot_instants"][:,-1],a["row_id"]):
            raise ValueError("PREFLIGHT_WINDOW_EXCLUDES_ORIGIN")
        if a["slot_instants"].shape!=a["offsets_hours"].shape or \
                not np.array_equal((a["slot_instants"]-a["row_id"][:,None])/3600,a["offsets_hours"]):
            raise ValueError("PREFLIGHT_WINDOW_METADATA")
        x,y,mask=T.materialize_monitor_inputs(a,c["scales"])
        if not np.isfinite(x[0]).all() or not np.isfinite(y[mask]).all():
            raise ValueError("PREFLIGHT_NONFINITE_INPUT")
        proof[cell]=dict(parameters=int(bundle.forecast_model.count_params()),supported=mask.sum(axis=0).tolist())
    proof.update(state="PASS",fit_executed=False,test_read=False)
    put(Path(spec["gpu_root"])/"PREFLIGHT.json",proof);return proof


def align_single_head(bundle, initial):
    bundle.head_models["h18"].set_weights(initial.head_models["h18"].get_weights())
    for a,b in zip(bundle.head_models["h18"].get_weights(),initial.head_models["h18"].get_weights(),strict=True):
        if not np.array_equal(a,b):raise ValueError("SINGLE_HEAD_INITIAL_WEIGHT_MISMATCH")


def fit_cell(spec, cell, folder):
    from tools.i6dmo import train as T,model as M
    import tensorflow as tf
    import importlib.metadata
    data_root=Path(spec["gpu_data"]);source=data_root/cell
    config=read(source/"CONFIG.json");hs=config["model"]["horizons_hours"]
    cache=read(source/"NAIVE_CACHE.json");arrays={s:load_arrays(source,s) for s in T.SPLITS}
    folder.mkdir(parents=True,exist_ok=True);put(folder/"CONFIG.json",config)
    def predictions(bundle, tag=""):
        reports={}
        for split,a in arrays.items():
            x,_,_=T.materialize_monitor_inputs(a,config["scales"])
            price=T.to_price(M.predict_matrix(bundle.forecast_model,x),a,config["scales"])
            rows=metric_rows(a,price,hs);check_cache(rows,cache[split]);reports[split]=rows
            np.savez_compressed(folder/(tag+split.upper()+".npz"),prediction=price,row_id=a["row_id"],
                origin_price=a["origin_price"],target_price=a["target_price"],mask=a["mask"])
        return reports
    if config["reuse_reference"]:
        training=read(data_root/"PARENT_TRAINING.json");reports={}
        shutil.copytree(data_root/"parent_model",folder/"model",dirs_exist_ok=True)
        for split in T.SPLITS:
            with np.load(data_root/("REFERENCE."+split+".npz"),allow_pickle=False) as z:
                a={k:z[k] for k in z.files}
            reports[split]=metric_rows(arrays[split],a["prediction"],hs);check_cache(reports[split],cache[split])
            shutil.copy2(data_root/("REFERENCE."+split+".npz"),folder/(split.upper()+".npz"))
        proof=dict(adopted_parent=True,parameters=4882,fit_executed=False)
    else:
        if tf.__version__!="2.21.0" or importlib.metadata.version("keras")!="3.13.2":
            raise ValueError("RUNTIME_PIN_CHANGED")
        if not tf.config.list_physical_devices("GPU"):raise ValueError("NO_GPU_NO_CPU_FALLBACK")
        original=M.build_dmo(config["original_model"]);bundle=M.build_dmo(config["model"])
        proof=model_proof(config,original,bundle)
        if cell=="single18":
            initial=M.build_dmo(model_config(config["original_model"],"plus18"))
            align_single_head(bundle,initial)
            proof["single18_initial_head"]="COPIED_FROM_UNTRAINED_PLUS18_SEED"
            del initial
        del original
        put(folder/"MODEL_PROOF.json",proof)
        def progress(event):
            put(folder/"PROGRESS.json",dict(state="RUNNING",cell=cell,at=time.time(),**event))
            if event["epoch"]==2:
                bundle.forecast_model.save_weights(folder/"epoch2.weights.h5")
                put(folder/"EPOCH2_METRICS.json",predictions(bundle,"EPOCH2."))
        fit=T.materialize_monitor_inputs(arrays["fit"],config["scales"])
        inner=T.materialize_monitor_inputs(arrays["inner"],config["scales"])
        result=T.fit_dmo(bundle,fit,inner,config["training"],config["seed"],progress,
                        ids=(arrays["fit"]["row_id"],arrays["inner"]["row_id"]),record_device="gpu")
        training=dict(result,settings=config["training"],split_roles={"fit":"FIT","inner":"INNER"})
        reports=predictions(bundle)
        M.save_dmo(bundle,folder/"model",extra={"scales_sha256":config["scales"]["scales_sha256"]})
        # Existing original trainer's monitor replay and restore checks govern this fit.
    put(folder/"TRAINING.json",training);put(folder/"MODEL_PROOF.json",proof)
    put(folder/"SCALES.json",config["scales"])
    parent=math.fsum(r["relative_error"] for r in config["parent_native_metrics"]["validation"])/6
    result=dict(state="COMPLETE",metrics=reports,parameters=proof["parameters"],seed=config["seed"],
        parent_short_mean=parent,adopted=config["reuse_reference"],fit_seconds=0 if config["reuse_reference"]
        else training["fit_seconds"],epochs=training["epochs_completed"],best_epoch=training["best_epoch"],
        device_class="rtx4090",test_read=False,
        files={str(p.relative_to(folder)):file_sha256(p) for p in folder.rglob("*")
               if p.is_file() and p.name not in ("STARTED.json","PROGRESS.json","RESULT.json")})
    return result


def terminal(root, cell, spec):
    folder=Path(root)/cell;path=folder/"RESULT.json"
    if not path.exists():return None
    d=read(path)
    if d.get("prepared_sha256")!=spec["prepared_sha256"] or d.get("cell",{}).get("id")!=cell or \
            d.get("sha256")!=sha256_json({k:v for k,v in d.items() if k!="sha256"}):
        raise ValueError("RESULT_IDENTITY_CHANGED")
    authenticate(folder,d.get("files",{}));return d


def run(spec):
    root=Path(spec["gpu_root"]);root.mkdir(parents=True,exist_ok=True)
    with (root/"gpu.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done={c:r for c in active_cells(spec) if (r:=terminal(root,c,spec))}
        cell=next_cell(done,False,active_cells(spec))
        if cell is None:return
        if command(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]).strip():
            put(root/"PROGRESS.json",dict(state="WAIT_GPU",next_cell=cell,at=time.time()));return
        actual=command(["nvidia-smi","-i",spec["gpu_uuid"],"--query-gpu=name","--format=csv,noheader"]).strip()
        if actual!=spec["expected_gpu_name"] or os.environ.get("CUDA_VISIBLE_DEVICES")!=spec["gpu_uuid"]:
            raise ValueError("GPU_IDENTITY_CHANGED")
        for rel,sha in spec["code_files"].items():
            if file_sha256(REPO/rel)!=sha:raise ValueError("CODE_PIN_CHANGED:"+rel)
        prepared=read(Path(spec["gpu_data"])/"PREPARED.json")
        if prepared["sha256"]!=spec["prepared_sha256"]:raise ValueError("PREPARED_CHANGED")
        authenticate(Path(spec["gpu_data"]),prepared["files"])
        folder=root/cell
        if spec.get("study_mode") and not (root/"PREFLIGHT.json").exists():
            preflight(spec)
        if cell=="single18" and not spec.get("study_mode") and not need_single(done["plus18"]):
            result=dict(state="SKIPPED",reason="18h beats naive without short-mean deterioration",files={})
        elif (folder/"STARTED.json").exists():
            result=dict(state="FAILED_INTERRUPTED",reason="no automatic fit repetition",files={})
        else:
            folder.mkdir(parents=True,exist_ok=True);put(folder/"STARTED.json",dict(at=time.time()))
            try:
                result=fit_cell(spec,cell,folder)
            except Exception as exc:
                result=dict(state="FAILED",reason=type(exc).__name__+":"+str(exc)[:1500],files={})
        result.update(cell=dict(id=cell),batch_id=campaign_name(spec),prepared_sha256=spec["prepared_sha256"])
        result["sha256"]=sha256_json(result);put(folder/"RESULT.json",result)
        put(root/"PROGRESS.json",dict(state=result["state"],cell=cell,at=time.time()))


def report(spec, cell, result):
    name=campaign_name(spec)
    rows=[]
    for split,values in result.get("metrics",{}).items():
        for r in values:
            for name,key,unit in (("MAE","mae","USD_per_EUR"),("Naive_MAE","naive_mae","USD_per_EUR"),
                ("Relative_Error_Vs_Naive","relative_error","ratio"),("N_Paired","n","count")):
                rows.append(dict(metric=name,value=r[key],unit=unit,horizon=r["horizon"],split=split))
    rows.append(dict(metric="Fit_Complete",value=int(result["state"]=="COMPLETE"),unit="bool",
                     horizon=None,split="validation"))
    tags=dict(parent="I6-DMO-SHORT-MG2-MATCHED-R0",parent_code_commit=BASE,
              cell=cell,disposition=result["state"],test_read="false",evaluation="DEVELOPMENT_TUNING",
              evidence_sha256=result["sha256"],seed="6102026")
    config=Path(spec["artifacts"])/cell/"CONFIG.json"
    if config.exists():tags["config_json"]=json.dumps(read(config),sort_keys=True)
    return dict(experiment_key="i6dmo:"+name+":"+cell,experiment_set_key="i6dmo:"+name,actor="predictor",
        lake="LOCAL_RETAINED",lineage="UNVERIFIED",config_sha256=spec["prepared_sha256"],
        code_commit=spec["code_commit"],project="predictor",phase="I6DMO_MG2M_EXTENSION",metrics=rows,tags=tags)


def tick(spec):
    from tools.i6dmo.outbox import Outbox,HttpWarehouse
    from tools.i6dmo.mirror import _sync_local,_sync_remote
    root=Path(spec["root"]);artifacts=Path(spec["artifacts"]);artifacts.mkdir(parents=True,exist_ok=True)
    name=campaign_name(spec);ordered=active_cells(spec)
    with (root/"tick.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/"CLOSE.json").exists():return read(root/"CLOSE.json")
        command(["rsync","-a","--checksum","--exclude=*.lock",spec["ssh_alias"]+":"+spec["gpu_root"]+"/",
                 str(artifacts)+"/"])
        results={c:r for c in ordered if (r:=terminal(artifacts,c,spec))}
        state=dict(state="RUNNING",complete=sum(r["state"]=="COMPLETE" for r in results.values()),
            terminal=len(results),total=len(ordered),cells={c:r["state"] for c,r in results.items()},test_read=False,at=time.time())
        active=next_cell(results,False,ordered)
        if active and (artifacts/active/"PROGRESS.json").exists():
            state["progress"]=read(artifacts/active/"PROGRESS.json")
            p=state["progress"];epoch=p.get("epoch",0)
            state["eta"]=dict(kind="CURRENT_FIT_CEILING_ONLY",seconds=(200-epoch)*p.get("elapsed_s",0)/epoch
                              if epoch else None,note="early stopping may finish sooner; not whole-study ETA")
        box=Outbox(root/"outbox.sqlite3")
        try:
            for c,r in results.items():box.enqueue(report(spec,c,r),source_unit=c)
            pub=box.pump(HttpWarehouse(spec["warehouse"]["url"],spec["warehouse"]["token_env"]))
            pub.pop("delivery_seconds",None);state["publication"]=pub
        finally:box.close()
        lines=["# MG2m Extension And Window Study","","DEVELOPMENT 2024; TEST closed.","",
               "| Cell | Horizon | MAE | Naive for horizon | Relative error | N |","|---|---:|---:|---:|---:|---:|"]
        for c,r in results.items():
            for m in r.get("metrics",{}).get("validation",[]):
                lines.append(f"| {c} | {m['horizon']}h | {m['mae']:.12f} | {m['naive_mae']:.12f} | {100*m['relative_error']:+.6f}% | {m['n']} |")
        tmp=artifacts/"RESULTS.md.tmp";tmp.write_text("\n".join(lines)+"\n");tmp.replace(artifacts/"RESULTS.md")
        files={"RESULTS.md":file_sha256(artifacts/"RESULTS.md")}
        for c,r in results.items():
            files.update({c+"/"+rel:sha for rel,sha in r.get("files",{}).items()})
            files[c+"/RESULT.json"]=file_sha256(artifacts/c/"RESULT.json")
        local_bad=_sync_local(artifacts,spec["local_mirror"],files)
        (artifacts/"backup").mkdir(exist_ok=True)
        remote_bad=_sync_remote(artifacts,spec["ssh_alias"],spec["remote_mirror"],files,subprocess.run)
        state["mirrors"]=dict(state="VERIFIED" if not local_bad and not remote_bad else "MISMATCH",files=len(files),
                              local_bad=local_bad[:5],remote_bad=remote_bad[:5],manifest_sha256=sha256_json(files))
        if len(results)==len(ordered) and pub["expected"]==len(ordered) and not pub["pending"] and not pub["mismatch"] \
                and state["mirrors"]["state"]=="VERIFIED":
            state["state"]="CLOSED" if all(r["state"] in ("COMPLETE","SKIPPED") for r in results.values()) else "CLOSED_WITH_FAILURE"
            put(root/"CLOSE.json",state)
            command(["ssh",spec["ssh_alias"],"systemctl","--user","disable","--now",name+"-gpu.timer"])
            command(["systemctl","--user","disable","--now",name+"-tick.timer"])
        put(root/"STATUS.json",state);return state


def launch(spec):
    name=campaign_name(spec);active_cells(spec)
    if command(["git","status","--porcelain"]).strip():raise ValueError("SOURCE_NOT_COMMITTED")
    prepared=prepare(spec);root=Path(spec["root"]);root.mkdir(parents=True,exist_ok=True)
    deploy=read(spec["source_deployment"]);role=deploy["roles"][spec["gpu_role"]]
    if role["ssh_alias"]!=spec["ssh_alias"]:raise ValueError("GPU_ROLE_CHANGED")
    spec=dict(spec,prepared_sha256=prepared["sha256"],code_commit=command(["git","rev-parse","HEAD"]).strip())
    spec["code_files"]={str(p.relative_to(REPO)):file_sha256(p) for directory in
        ("tools/i6dmo","tools/i6n_contracts","predictor_plugins/modular_temporal")
        for p in (REPO/directory).rglob("*.py")}
    runtime="/var/tmp/i6dmo-mg2m-runtime-"+spec["code_commit"][:8]
    def remote(args):return command(["ssh",spec["ssh_alias"],shlex.join([str(x) for x in args])])
    branch=command(["git","branch","--show-current"]).strip();archive=root/"code.bundle"
    command(["git","bundle","create",str(archive),BASE+".."+branch])
    remote(["mkdir","-p",str(root),spec["gpu_root"],spec["gpu_data"],spec["remote_mirror"]])
    command(["rsync","-a",str(archive),spec["ssh_alias"]+":"+str(archive)])
    remote(["git","-C",spec["remote_repository"],"fetch",str(archive),"refs/heads/"+branch])
    probe=subprocess.run(["ssh",spec["ssh_alias"],shlex.join(["test","-d",runtime])],capture_output=True)
    if probe.returncode:remote(["git","-C",spec["remote_repository"],"worktree","add","--detach",runtime,spec["code_commit"]])
    if remote(["git","-C",runtime,"rev-parse","HEAD"]).strip()!=spec["code_commit"]:
        raise ValueError("REMOTE_CODE_CHANGED")
    command(["rsync","-a","--checksum",spec["data_root"]+"/",spec["ssh_alias"]+":"+spec["gpu_data"]+"/"])
    destination=root/"RUN.private.json";spec["runtime"]=runtime
    if destination.exists() and read(destination)!=spec:raise ValueError("DEPLOYMENT_IDENTITY_CHANGED")
    put(destination,spec);command(["rsync","-a",str(destination),spec["ssh_alias"]+":"+str(destination)])
    environment={**role.get("environment",{}),"CUDA_VISIBLE_DEVICES":spec["gpu_uuid"],"PYTHONPATH":runtime,
                 "PYTHONDONTWRITEBYTECODE":"1","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
    if not spec.get("study_mode"):
        remote(["env",*[k+"="+v for k,v in environment.items()],deploy["launcher"]["crispdm_run"],"-m","4G",
                "-t","180s","-n",name+"-preflight","--",role["python"],"-B","-m","tools.i6dmo.mg2m_study",
                "preflight","--spec",str(destination)])
    for target,action in (("tick","tick"),("gpu","run")):
        wd=str(REPO) if target=="tick" else runtime
        python=deploy["roles"]["coordinator"]["python"] if target=="tick" else role["python"]
        args=[deploy["launcher"]["crispdm_run"],"-m","768M" if target=="tick" else "4G",
              "-t","600s" if target=="tick" else "1800s","-n",name+"-"+target,"--",python,
              "-B","-m","tools.i6dmo.mg2m_study",action,"--spec",str(destination)]
        env="EnvironmentFile="+spec["warehouse"]["environment_file"]+"\n" if target=="tick" else "".join(
            "Environment="+json.dumps(k+"="+v)+"\n" for k,v in role.get("environment",{}).items())
        env+="Environment=CUDA_VISIBLE_DEVICES="+("" if target=="tick" else spec["gpu_uuid"])+"\n"
        body="[Unit]\nDescription=Parent-pinned MG2m finite study\nX-I6DMO-Batch="+name+"\n[Service]\nType=oneshot\nWorkingDirectory="+wd+"\n"+env
        body+="Environment=PYTHONPATH="+wd+"\nEnvironment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\n"
        body+="ExecStart="+shlex.join(args)+"\nMemoryMax="+("1G" if target=="tick" else "5G")+"\nMemorySwapMax=0\nTimeoutStartSec=1900\nNice=10\n"
        timer="[Unit]\nDescription=Automatic MG2m sequence\n[Timer]\nOnBootSec=10s\nOnUnitInactiveSec=15s\nAccuracySec=1s\nPersistent=true\n[Install]\nWantedBy=timers.target\n"
        code="from pathlib import Path; d=Path.home()/'.config/systemd/user'; d.mkdir(parents=True,exist_ok=True); "
        for suffix,text in (("service",body),("timer",timer)):
            code+="p=d/"+repr(name+"-"+target+"."+suffix)+"; p.write_text("+repr(text)+"); "
        if target=="gpu":
            remote(["python3","-c",code]);remote(["systemctl","--user","daemon-reload"])
            remote(["systemctl","--user","enable","--now",name+"-gpu.timer"])
        else:
            command(["python3","-c",code]);command(["systemctl","--user","daemon-reload"])
            command(["systemctl","--user","enable","--now",name+"-tick.timer"])
    put(root/"LAUNCH.json",dict(state="ARMED",code_commit=spec["code_commit"],prepared_sha256=prepared["sha256"],test_read=False))
    return dict(state="ARMED",batch_id=name,code_commit=spec["code_commit"])


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","preflight","launch","run","tick","status"]);p.add_argument("--spec",required=True)
    a=p.parse_args();spec=read(a.spec)
    if a.action=="status":
        path=Path(spec["root"])/"STATUS.json";result=read(path) if path.exists() else {"state":"NOT_STARTED"}
    else:result=globals()[a.action](spec)
    if result:print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
