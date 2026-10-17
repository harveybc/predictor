"""Frozen weekly continuation plus one head; unattended finite experiment."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from unittest.mock import patch

import numpy as np

from tools.i6dmo import resolution_batch as B
from tools.i6n_swarm.common import file_sha256

REPO = Path(__file__).resolve().parents[2]
BASE = "e0626cd9b2ac491a4508487883d04f40692d798c"
NAME = "i6dmo-weekly-plus18-20261017"
SHORT = [2, 3, 4, 5, 6, 12]
HEADS = SHORT + [18]
CATALOGUE = "HORIZONS = (*R.HORIZONS, (4,), (2, 3, 4, 5, 6, 12, 18), (2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72))"
MODULE = "tools.i6dmo.weekly18"
PHASE = "I6DMO_WEEKLY_PLUS18"
TITLE = "# Weekly Short Plus 18 Hours"
GPU_WALL = 1800
GPU_ROLE = "preferred"
GPU_HINT = "5090"
FROZEN = ("tools/i6dmo/train.py", "tools/i6dmo/model.py", "tools/i6dmo/continuation.py",
          "tools/i6dmo/continuation_data.py", "tools/i6dmo/market_grid.py",
          "tools/i6n_contracts/eligibility.py")
QUALIFICATION = "RETROSPECTIVE_DEVELOPMENT_CHECKPOINT_TUNING_NOT_INDEPENDENT_OOS"


def read(path):
    return B.read(path)


def run_cmd(args, timeout=600):
    return B.command(args, timeout)


def authenticate(root, files):
    for rel, expected in files.items():
        p = Path(root) / rel
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not p.is_file() or file_sha256(p) != expected:
            raise ValueError("ARTIFACT_DIGEST:" + rel)


def source_identity(parent):
    for rel, expected in parent["code_files"].items():
        if rel != "tools/i6dmo/continuation_models.py" and file_sha256(REPO / rel) != expected:
            raise ValueError("PARENT_SOURCE_CHANGED:" + rel)
    old = run_cmd(["git", "-C", REPO, "show", BASE + ":tools/i6dmo/continuation_models.py"])
    if (REPO / "tools/i6dmo/continuation_models.py").read_text() != old.replace(
            "HORIZONS = (*R.HORIZONS, (4,))", CATALOGUE):
        raise ValueError("MODEL_CHANGED_BEYOND_HORIZON_CATALOGUE")
    if parent["code_commit"] != BASE or parent["training"] != B.TRAINING or parent["seed"] != B.SEED:
        raise ValueError("PARENT_TRAINING_IDENTITY")


def assert_projection(parent, extended):
    if set(parent) != set(extended):
        raise ValueError("PARENT_ARRAY_CHANGED:keys")
    for key, old in parent.items():
        new = extended[key][:, :6] if key in ("mask", "target_price") else extended[key]
        if not np.array_equal(old, new, equal_nan=np.asarray(old).dtype.kind in "fc"):
            raise ValueError("PARENT_ARRAY_CHANGED:" + key)


def append_column(a, scales, target, mask, *, fit_naive):
    target, mask = np.asarray(target).copy(), np.asarray(mask)
    if target.shape != (len(a["row_ids"]),) or mask.dtype != np.bool_ or mask.shape != target.shape \
            or not np.isfinite(target[mask]).all() or not math.isfinite(fit_naive) or fit_naive <= 0:
        raise ValueError("EXTENDED_HEAD_INVALID")
    target[~mask] = np.nan
    out = {**a, "target_price": np.column_stack([a["target_price"], target]),
           "mask": np.column_stack([a["mask"], mask])}
    assert_projection(a, out)
    s = copy.deepcopy(scales)
    s.update(naive_mae=list(scales["naive_mae"]) + [fit_naive], horizons=HEADS)
    s.pop("scales_sha256", None)
    s["scales_sha256"] = B.digest(s)
    return out, s


def extend_model(parent):
    if parent["horizons_hours"] != SHORT or parent["seed"] != B.SEED or parent["slots"] != 13 \
            or parent["spacing_open_market_hours"] != 2 or parent["core"] != [2] \
            or parent["head"] != "lstm" or parent["kind"] != "conv":
        raise ValueError("WRONG_WEEKLY_WINNER")
    return dict(copy.deepcopy(parent), horizons_hours=HEADS)


def label18(a, source, calendar, period, split, base):
    """Use the original C1 contract, exact observed target and original split cutoffs."""
    from tools.i6n_contracts import eligibility as E
    from tools.i6dmo import data
    _, weeks = E.load_calendar(calendar)
    by_id = {w.week_id: w for w in weeks}
    instant = lambda t: datetime.fromtimestamp(int(t), timezone.utc)
    observed = E.ObservedTargets({instant(v): instant(v) for v in source["ts"]})
    if split == "FIT":
        start, end = (data._instant(period[k]) for k in ("train_start", "train_end"))
    else:
        start, end = (data._instant(base["splits"]["validation"][k]) for k in ("start", "end"))
    bounds = E.SplitBounds(split, start, end, label_available_by=end)
    target = np.full(len(a["row_ids"]), np.nan)
    mask = np.zeros(len(target), bool)
    for i, (t, wid) in enumerate(zip(a["row_ids"], a["week_ids"], strict=True)):
        decision = E.evaluate_heads(instant(t), [18], E.Role.MANAGEMENT, by_id[str(wid)],
            input_available=True, observed_targets=observed, split=bounds)[0]
        if decision.label_eligible:
            at = int(t) + 18 * 3600
            ix = np.searchsorted(source["ts"], at)
            if ix == len(source["ts"]) or source["ts"][ix] != at or not np.isfinite(source["price"][ix]):
                raise ValueError("EXACT_18H_TARGET_MISSING")
            target[i], mask[i] = source["price"][ix], True
    return target, mask


def pool_metrics(rows, h):
    pieces = [r for r in rows if r["horizon"] == h]
    n = sum(r["n"] for r in pieces)
    se = math.fsum(r["sum_abs_error"] for r in pieces)
    sn = math.fsum(r["sum_abs_naive_error"] for r in pieces)
    return dict(horizon=h, n=n, mae=se/n if n else None, naive_mae=sn/n if n else None,
                relative_error=(se-sn)/sn if sn else None, sum_abs_error=se, sum_abs_naive_error=sn)


def prepare(spec):
    from tools.i6dmo import continuation_data as D
    from tools.i6dmo import continuation as C
    root = Path(spec["data_root"])
    if (root / "PREPARED.json").exists():
        p = read(root / "PREPARED.json"); authenticate(root, p["files"])
        if p["parent_spec_sha256"] != file_sha256(spec["parent_spec"]):
            raise ValueError("PARENT_SPEC_CHANGED")
        return p
    root.mkdir(parents=True, exist_ok=True)
    parent = read(spec["parent_spec"]); source_identity(parent)
    manifest = D._verify(spec["parent_data"], spec["input_dir"])
    source = D._read(Path(spec["parent_data"]) / "SOURCE.npz")
    cells = [c for c in parent["cells"] if c["stage"] == "cadence" and c["group"] == "short"
             and c["cadence"] == "weekly"]
    if len(cells) != len(parent["periods"]["weekly"]) or len(cells) != 53:
        raise ValueError("WEEKLY_PERIODS_CHANGED")
    transfer = B.terminal(parent["artifacts"], next(c for c in parent["cells"] if c["id"] == "transfer_short"), parent)
    if not transfer or transfer["state"] != "COMPLETE":
        raise ValueError("PARENT_TRANSFER_MISSING")
    resolved = transfer["resolved"]
    files, prepared = {}, []
    for c in cells:
        parent_result = B.terminal(parent["artifacts"], c, parent)
        if not parent_result or parent_result["state"] not in ("COMPLETE", "SKIPPED_NO_EVAL_ROWS"):
            raise ValueError("PARENT_WEEK_NOT_COMPLETE:" + c["id"])
        folder = root / c["id"]; folder.mkdir(exist_ok=True)
        new_cell = dict(copy.deepcopy(c), horizons=HEADS)
        record = dict(cell=new_cell, parent_result_sha256=parent_result["sha256"],
                      parent_state=parent_result["state"], parent_metrics=parent_result.get("metrics", {}),
                      resolved=resolved)
        if parent_result["state"] == "COMPLETE":
            arrays, scales, meta = D.load(spec["parent_data"], "short", 2, 13, horizons=SHORT,
                                        input_dir=spec["input_dir"], period=c["period"])
            parent_config = read(Path(parent["artifacts"]) / "results" / c["id"] / "CONFIG.json")
            if meta != parent_config["population"] or scales["scales_sha256"] != meta["scales_sha256"]:
                raise ValueError("PARENT_PERIOD_DATA_CHANGED:" + c["id"])
            original_model = parent_config["model"]
            target, mask = label18(arrays["FIT"], source, manifest["calendar"], c["period"], "FIT", manifest["base_design"])
            naive18 = math.fsum(np.abs(target[mask] - arrays["FIT"]["origin_price"][mask]).tolist()) / int(mask.sum())
            extended = {}
            for split, a in arrays.items():
                if split != "FIT":
                    target, mask = label18(a, source, manifest["calendar"], c["period"], split, manifest["base_design"])
                extended[split], new_scales = append_column(a, scales, target, mask, fit_naive=naive18)
                np.savez_compressed(folder / (split + ".npz"), **extended[split])
                files[c["id"] + "/" + split + ".npz"] = file_sha256(folder / (split + ".npz"))
                with np.load(Path(parent["artifacts"]) / "results" / c["id"] / (split + ".npz"), allow_pickle=False) as z:
                    for key in ("origin_ts", "origin_price", "target_price", "mask"):
                        if not np.array_equal(a[key], z[key], equal_nan=a[key].dtype.kind in "fc"):
                            raise ValueError("PARENT_SCORED_SUPPORT_CHANGED:" + key)
            projection = {k: D.R._digest(v) for k, v in arrays["FIT"].items()}
            new_meta = dict(meta, parent_manifest_sha256=meta["manifest_sha256"], horizons=HEADS,
                full_horizons=HEADS, scales_sha256=new_scales["scales_sha256"],
                counts={s: a["mask"].sum(axis=0).tolist() for s,a in extended.items()},
                validation_counts=extended["VALIDATION"]["mask"].sum(axis=0).tolist(),
                qualification=QUALIFICATION, parent_period_result_sha256=parent_result["sha256"],
                exact_short_projection_sha256=B.digest(projection))
            record.update(scales=new_scales, population=new_meta, parent_model=original_model,
                model=extend_model(original_model), training=parent_config["training"],
                parent_initial_weights_sha256=parent_result["initial_weights_sha256"],
                projection=dict(state="EXACT", arrays=projection),
                naive_cache={s: C.measured_metrics(
                    a, np.where(a["mask"], a["target_price"], 0), HEADS) for s,a in extended.items()})
            for split in arrays:
                for actual, original in zip(record['naive_cache'][split][:6], parent_result['metrics'][split], strict=True):
                    if any(actual[k] != original[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error','pairs_sha256')):
                        raise ValueError('PARENT_NAIVE_CHANGED:'+c['id'])
        B.put(folder / "PERIOD.json", record)
        files[c["id"] + "/PERIOD.json"] = file_sha256(folder / "PERIOD.json")
        prepared.append(new_cell)
        B.put(root / "PREPARE_PROGRESS.json", dict(terminal=len(prepared), total=len(cells), last=c["id"], at=time.time()))
    p = dict(schema="i6dmo.weekly_plus18.prepared.v1", parent_spec_sha256=file_sha256(spec["parent_spec"]),
        parent_population_digest=parent["population_digest"], cells=prepared, files=files,
        original_training=parent["training"], seed=B.SEED, qualification=QUALIFICATION, test_read=False)
    p["sha256"] = B.digest(p); B.put(root / "PREPARED.json", p)
    return p


def prepared_document(spec, root):
    p = read(Path(root) / "PREPARED.json")
    if p["sha256"] != spec["population_digest"] or B.digest({k:v for k,v in p.items() if k!="sha256"}) != p["sha256"]:
        raise ValueError("PREPARED_IDENTITY_CHANGED")
    return p


def initialization_proof(record, inputs=None):
    from tools.i6dmo import continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    old = M.build(record["parent_model"])
    if weights_hash(old.forecast_model) != record["parent_initial_weights_sha256"]:
        raise ValueError("PARENT_INITIAL_WEIGHTS_CHANGED")
    new = M.build(record["model"])
    lookup = {v.path: v.numpy() for v in new.forecast_model.weights}
    for v in old.forecast_model.weights:
        if v.path not in lookup or not np.array_equal(v.numpy(), lookup[v.path]):
            raise ValueError("SHORT_INITIAL_WEIGHT_CHANGED:" + v.path)
    if inputs is not None:
        before = old.forecast_model(inputs, training=False)
        after = new.forecast_model(inputs, training=False)
        if len(before) != 6 or len(after) != 7 or any(
                not np.array_equal(a.numpy(), b.numpy()) for a,b in zip(before,after[:6],strict=True)):
            raise ValueError('SHORT_INITIAL_FORWARD_CHANGED')
    return dict(state="EXACT", parent_initial_weights_sha256=weights_hash(old.forecast_model),
                new_initial_weights_sha256=weights_hash(new.forecast_model), compared_tensors=len(old.forecast_model.weights),
                short_forward='EXACT' if inputs is not None else 'NOT_REQUESTED')


def preflight(spec):
    verify(spec)
    p = prepared_document(spec, spec["gpu_data"])
    authenticate(spec["gpu_data"], p["files"])
    for cell in p['cells']:
        r=read(Path(spec['gpu_data'])/cell['id']/'PERIOD.json')
        if r['parent_state']!='COMPLETE':continue
        if r['training']!=B.TRAINING or extend_model(r['parent_model'])!=r['model']:
            raise ValueError('PERIOD_TRAINING_OR_MODEL_CHANGED')
        for split in ('FIT','VALIDATION'):
            for current,old in zip(r['naive_cache'][split][:6],r['parent_metrics'][split],strict=True):
                if any(current[k]!=old[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error','pairs_sha256')):
                    raise ValueError('PARENT_NAIVE_CHANGED:'+cell['id'])
    from tools.i6dmo import train as T
    import tensorflow as tf
    import importlib.metadata
    if tf.__version__ != "2.21.0" or importlib.metadata.version("keras") != "3.13.2" or not tf.config.list_physical_devices("GPU"):
        raise ValueError("ORIGINAL_GPU_RUNTIME_REQUIRED")
    record = next(read(Path(spec["gpu_data"]) / c["id"] / "PERIOD.json") for c in p["cells"]
                  if read(Path(spec["gpu_data"]) / c["id"] / "PERIOD.json")["parent_state"] == "COMPLETE")
    from tools.i6dmo import continuation_data as D
    a = D._read(Path(spec['gpu_data'])/record['cell']['id']/'FIT.npz')
    x, _, _ = T.transform(a,record['scales'])
    proof = initialization_proof(record,[x[:2],a['offsets_hours'][:2].astype('float32'),a['closure_crossed'][:2].astype('float32')])
    if T.MeanMonitorStopper(10, 1e-8, 200).identity()["rule"] != "i6dmo.mean_monitor_stop.v1":
        raise ValueError("ORIGINAL_PATIENCE_REQUIRED")
    B.put(Path(spec["gpu_root"]) / "PREFLIGHT.json", dict(state="PASS", initialization=proof,
        periods=len(p["cells"]), fits_run=0, test_read=False))
    return dict(state="PASS", periods=len(p["cells"]), initialization=proof, fits_run=0)


def verify(spec):
    authenticate(REPO, spec["code_files"])
    if spec["training"] != B.TRAINING or spec["seed"] != B.SEED or spec["batch_id"] != NAME:
        raise ValueError("EXPERIMENT_CHANGED")
    if any(c["horizons"] != HEADS or c["cadence"] != "weekly" or c["group"] != "short" for c in spec["cells"]):
        raise ValueError("UNAUTHORIZED_CELL")


def terminal(root, cell, spec):
    path = Path(root) / "results" / cell["id"] / "RESULT.json"
    if not path.exists():
        return None
    r = read(path)
    if r.get("sha256") != B.digest({k:v for k,v in r.items() if k!="sha256"}) or r.get("cell") != cell \
            or r.get("batch_id") != spec["batch_id"]:
        raise ValueError("TERMINAL_IDENTITY")
    authenticate(path.parent, r.get("files", {}))
    return r


def run(spec):
    verify(spec)
    root = Path(spec["gpu_root"]); root.mkdir(parents=True, exist_ok=True)
    with (root / "gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {c["id"]: r for c in spec["cells"] if (r:=terminal(root,c,spec))}
        cell = next((c for c in spec["cells"] if c["id"] not in results), None)
        if cell is None:
            B.progress(root, state="GPU_DONE"); return
        p = prepared_document(spec, spec["gpu_data"])
        prefix = cell["id"] + "/"
        authenticate(spec["gpu_data"], {n:h for n,h in p["files"].items() if n.startswith(prefix)})
        record = read(Path(spec["gpu_data"]) / cell["id"] / "PERIOD.json")
        folder = root / "results" / cell["id"]; folder.mkdir(parents=True, exist_ok=True)
        if record["parent_state"] == "SKIPPED_NO_EVAL_ROWS":
            result = dict(batch_id=NAME, cell=cell, state="SKIPPED_NO_EVAL_ROWS", reason="same empty parent period", files={})
        else:
            processes = run_cmd(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            if processes.strip():
                B.progress(root, state="WAIT_GPU", cell=cell["id"]); return
            gpu = run_cmd(["nvidia-smi", "-i", spec["gpu_uuid"], "--query-gpu=name", "--format=csv,noheader"])
            available = int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))) * 1024
            if GPU_HINT not in gpu or available < 7*1024**3:
                B.progress(root, state="WAIT_RESOURCES", cell=cell["id"], memory_available=available); return
            if (folder / "STARTED.json").exists():
                result = dict(batch_id=NAME, cell=cell, state="FAILED_INTERRUPTED", files={}, reason="no automatic refit")
            else:
                B.put(folder / "STARTED.json", dict(cell=cell,at=time.time()))
                try:
                    from tools.i6dmo import continuation as C, continuation_data as D
                    arrays = {s:D._read(Path(spec["gpu_data"]) / cell["id"] / (s+".npz")) for s in ("FIT","VALIDATION")}
                    for split,a in arrays.items():
                        cache=C.measured_metrics(a, np.where(a["mask"],a["target_price"],0),HEADS)
                        if cache != record["naive_cache"][split]:
                            raise ValueError("NAIVE_CACHE_CHANGED")
                    meta=dict(record["population"], manifest_sha256=spec["population_digest"])
                    B.put(folder / "CONTRAST_PROOF.json", dict(parent_result_sha256=record["parent_result_sha256"],
                        projection=record["projection"], initialization=initialization_proof(record),
                        parent_model=record["parent_model"], only_model_delta=record.get("model_delta",dict(horizons_hours=HEADS))))
                    B.put(folder / "SCALES.json", record["scales"])
                    def loader(*args, **kwargs):
                        if args[:4] != (spec["gpu_data"], "short", record["model"]["spacing_open_market_hours"], record["model"]["slots"]) or kwargs != dict(
                                horizons=HEADS, input_dir=spec["gpu_input"], period=cell["period"]):
                            raise ValueError("UNEXPECTED_DATA_ACCESS")
                        return arrays, record["scales"], meta
                    with patch.object(D,"load",loader), patch.object(C,"resolve",lambda *a:record["resolved"]):
                        result=C.fit_cell(spec,cell,{},folder)
                    expected=read(folder / "CONFIG.json")["model"]
                    if expected!=record["model"]:
                        raise ValueError("MODEL_CONFIG_CHANGED")
                except Exception as exc:
                    result=dict(batch_id=NAME,cell=cell,state="FAILED",files={},reason=type(exc).__name__+":"+str(exc)[:500])
        result["sha256"]=B.digest(result);B.put(folder/"RESULT.json",result)
        B.progress(root,state=result["state"],cell=cell["id"])


def annual(spec, artifacts, results):
    if len(results)!=len(spec["cells"]) or any(r["state"] not in ("COMPLETE","SKIPPED_NO_EVAL_ROWS") for r in results.values()):
        return dict(state="INCOMPLETE",test_read=False)
    rows=[]; seen={h:set() for h in HEADS}
    for c in spec["cells"]:
        r=results[c["id"]]
        if r["state"]!="COMPLETE":continue
        with np.load(Path(artifacts)/"results"/c["id"]/"VALIDATION.npz",allow_pickle=False) as z:
            for j,h in enumerate(HEADS):
                origins=z['origin_ts'][z['mask'][:,j]].tolist()
                if seen[h].intersection(origins) or len(set(origins))!=len(origins):raise ValueError("WEEK_OVERLAP")
                seen[h].update(origins)
        rows.extend(r["metrics"]["VALIDATION"])
    pooled=[pool_metrics(rows,h) for h in HEADS]
    if any(r['n']!=len(seen[r['horizon']]) for r in pooled):raise ValueError("ANNUAL_COUNT_MISMATCH")
    return dict(state="COMPLETE",metrics=pooled,qualification=QUALIFICATION,test_read=False)


def annual_report(spec, result):
    metrics=[dict(metric="Weekly_Annual_"+key,value=r[key],unit="ratio" if key=="relative_error" else
        "count" if key=="n" else "USD_per_EUR",horizon=r['horizon'],split="validation")
        for r in result.get('metrics',[]) for key in ('mae','naive_mae','relative_error','n') if r[key] is not None]
    metrics.append(dict(metric="Annual_Complete",value=int(result['state']=="COMPLETE"),unit="bool",horizon=None,split="validation"))
    return dict(experiment_key="i6dmo:"+NAME+":annual",experiment_set_key="i6dmo:"+NAME,
        actor="predictor",lake="LOCAL_RETAINED",lineage="UNVERIFIED",config_sha256=B.digest(result),
        code_commit=spec['code_commit'],project="predictor",phase=PHASE,
        tags=dict(evaluation=QUALIFICATION,test_read="false",seed=str(B.SEED)),metrics=metrics)


def mirror(spec, artifacts, files):
    from tools.i6dmo.mirror import _sync_local
    bad=_sync_local(artifacts,spec['local_mirror'],files)
    alias=spec['mirror_alias']; destination=spec['remote_mirror']
    remote=lambda args:run_cmd(['ssh','-o','BatchMode=yes',alias,shlex.join([str(x) for x in args])])
    remote(['mkdir','-p',destination])
    backup=Path(artifacts)/'backup';backup.mkdir(exist_ok=True)
    listing=backup/'.mirror-files.txt';listing.write_text('\n'.join(sorted(files))+'\n')
    run_cmd(['rsync','-a','--checksum','--files-from='+str(listing),str(artifacts)+'/',alias+':'+destination+'/'])
    remote_bad=[]; names=sorted(files)
    for i in range(0,len(names),64):
        chunk=names[i:i+64]
        output=remote(['sh','-c','cd '+shlex.quote(destination)+' && sha256sum -- '+' '.join(shlex.quote(n) for n in chunk)])
        got={line.partition('  ')[2]:line.partition('  ')[0] for line in output.splitlines()}
        remote_bad.extend(n for n in chunk if got.get(n)!=files[n])
    return dict(state="VERIFIED" if not bad and not remote_bad else "MISMATCH",files=len(files),
                local_bad=bad[:5],remote_bad=remote_bad[:5],manifest_sha256=B.digest(files))


def tick(spec):
    from tools.i6dmo.outbox import Outbox,HttpWarehouse
    verify(spec);root=Path(spec['root']);artifacts=Path(spec['artifacts']);artifacts.mkdir(parents=True,exist_ok=True)
    with (root/'tick.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'CLOSE.json').exists():return read(root/'CLOSE.json')
        run_cmd(['rsync','-a','--checksum','--exclude=*.lock',spec['ssh_alias']+':'+spec['gpu_root']+'/',str(artifacts)+'/'])
        results={c['id']:r for c in spec['cells'] if (r:=terminal(artifacts,c,spec))}
        completed=[r for r in results.values() if r['state']=='COMPLETE']
        progress=read(artifacts/'PROGRESS.json') if (artifacts/'PROGRESS.json').exists() else {}
        remaining=len(spec['cells'])-len(results)
        state=dict(state='RUNNING' if progress.get('state')=='RUNNING' else progress.get('state','PENDING'),
            at=time.time(),batch_id=NAME,total=len(spec['cells']),terminal=len(results),complete=len(completed),
            failed=[k for k,r in results.items() if r['state'].startswith('FAILED')],progress=progress,
            cells={k:dict(state=r['state'],development=r.get('metrics',{}).get('VALIDATION')) for k,r in results.items()},
            eta=dict(kind='MEASURED_FIT_MEAN' if completed else 'PARENT_PROJECTION',
                seconds=remaining*(math.fsum(r['training_seconds'] for r in completed)/len(completed) if completed else spec['parent_mean_fit_seconds']),
                note='fit-only estimate; excludes queue, delivery and mirrors'),test_read=False)
        box=Outbox(root/'outbox.sqlite3')
        try:
            for c in spec['cells']:
                if c['id'] in results:
                    report=B.report(spec,results[c['id']]);report['phase']=PHASE
                    report['tags'].update(evaluation=QUALIFICATION)
                    box.enqueue(report,source_unit=c['id'])
            yearly=annual(spec,artifacts,results)
            if not remaining:
                B.put(artifacts/'ANNUAL.json',yearly);box.enqueue(annual_report(spec,yearly),source_unit='annual')
            pub=box.pump(HttpWarehouse(spec['warehouse']['url'],spec['warehouse']['token_env']))
            pub.pop('delivery_seconds',None);state['publication']=pub
            if not remaining:state['state']='WAIT_PUBLICATION' if pub['pending'] or pub['mismatch'] else 'VERIFYING_MIRRORS'
            if not remaining and not pub['pending'] and not pub['mismatch']:
                B.put(artifacts/'PUBLICATION.json',pub)
                B.put(artifacts/'READBACK.json',[dict(report_sha256=x['report_sha256'],state=x['state'],readback_sha256=x['readback_sha256']) for x in box.items()])
                lines=[TITLE,'','DEVELOPMENT 2024, weekly checkpoint tuning; TEST closed.','',
                    '| Horizon | MAE | Naive for horizon | Relative error | N |','|---|---:|---:|---:|---:|']
                for r in yearly.get('metrics',[]):
                    lines.append(f"| {r['horizon']} h | {r['mae']:.12f} | {r['naive_mae']:.12f} | {100*r['relative_error']:+.6f}% | {r['n']} |")
                (artifacts/'RESULTS.md').write_text('\n'.join(lines)+'\n')
                B.put(artifacts/'PUBLIC_CONFIG.json',{k:spec[k] for k in ('batch_id','cells','code_commit','code_files','training','seed','population_digest')})
                files={str(p.relative_to(artifacts)):file_sha256(p) for p in artifacts.rglob('*') if p.is_file()
                       and p.name not in ('RUN.private.json','STARTED.json','PROGRESS.json','PREFLIGHT.json')
                       and 'backup' not in p.parts and not p.name.startswith('.')}
                state['mirrors']=mirror(spec,artifacts,files)
                if state['mirrors']['state']=='VERIFIED':
                    state['state']='CLOSED_WITH_FAILURE' if state['failed'] else 'CLOSED'
                    B.put(root/'CLOSE.json',state)
                    run_cmd(['ssh',spec['ssh_alias'],'systemctl','--user','disable','--now',NAME+'-gpu.timer'])
                    run_cmd(['systemctl','--user','disable','--now',NAME+'-tick.timer'])
            B.put(root/'STATUS.json',state);return state
        finally:box.close()


def launch(spec):
    if run_cmd(['git','-C',REPO,'status','--porcelain']).strip():raise ValueError('SOURCE_NOT_FROZEN')
    source_identity(read(spec['parent_spec']))
    p=prepare(spec);root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    deploy=read(spec['source_deployment']);role=deploy['roles'][GPU_ROLE]
    remote=lambda args:run_cmd(['ssh','-o','BatchMode=yes',role['ssh_alias'],shlex.join([str(x) for x in args])])
    commit=run_cmd(['git','-C',REPO,'rev-parse','HEAD']).strip()
    runtime='/var/tmp/i6dmo-weekly18-runtime-'+commit[:8]
    bundle=root/'code.bundle';branch=run_cmd(['git','-C',REPO,'branch','--show-current']).strip()
    run_cmd(['git','-C',REPO,'bundle','create',str(bundle),BASE+'..'+branch])
    remote(['mkdir','-p',str(root),spec['gpu_data'],spec['gpu_root']])
    run_cmd(['rsync','-a',str(bundle),role['ssh_alias']+':'+str(bundle)])
    remote(['git','-C',spec['remote_repository'],'fetch',str(bundle),'refs/heads/'+branch])
    if subprocess.run(['ssh',role['ssh_alias'],'test','-d',runtime],capture_output=True).returncode:
        remote(['git','-C',spec['remote_repository'],'worktree','add','--detach',runtime,commit])
    run_cmd(['rsync','-a','--checksum',spec['data_root']+'/',role['ssh_alias']+':'+spec['gpu_data']+'/'])
    code_files={str(p.relative_to(REPO)):file_sha256(p) for directory in
        ('tools/i6dmo','tools/i6n_contracts','predictor_plugins/modular_temporal') for p in (REPO/directory).rglob('*.py')}
    spec=dict(spec,batch_id=NAME,code_commit=commit,code_files=code_files,cells=p['cells'],training=B.TRAINING,
        seed=B.SEED,population_digest=p['sha256'],runtime=runtime,ssh_alias=role['ssh_alias'])
    path=root/'RUN.private.json'
    if path.exists() and read(path)!=spec:raise ValueError('DEPLOYMENT_IDENTITY_CHANGED')
    B.put(path,spec);run_cmd(['rsync','-a',str(path),role['ssh_alias']+':'+str(path)])
    env={**role.get('environment',{}),'PYTHONPATH':runtime,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],
         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    proof=remote(['env',*[k+'='+v for k,v in env.items()],deploy['launcher']['crispdm_run'],'-m','4G','-t','180s','-n',NAME+'-preflight',
        '--',role['python'],'-B','-m',MODULE,'preflight','--spec',str(path)])
    B.put(root/'LAUNCH_PREFLIGHT.json',dict(stdout=proof))
    for target,action in (('tick','tick'),('gpu','run')):
        wd=str(REPO) if target=='tick' else runtime
        python=deploy['roles']['coordinator']['python'] if target=='tick' else role['python']
        args=[deploy['launcher']['crispdm_run'],'-m','768M' if target=='tick' else '4352M','-t',
            '600s' if target=='tick' else str(GPU_WALL)+'s','-n',NAME+'-'+target,'--',python,'-B','-m',MODULE,action,'--spec',str(path)]
        envtext='EnvironmentFile='+spec['warehouse']['environment_file']+'\n' if target=='tick' else ''.join(
            'Environment='+json.dumps(k+'='+v)+'\n' for k,v in role.get('environment',{}).items())
        body='[Unit]\nDescription=Exact weekly short plus18\nX-I6DMO-Batch='+NAME+'\n[Service]\nType=oneshot\nWorkingDirectory='+wd+'\n'+envtext
        body+='Environment=PYTHONPATH='+wd+'\nEnvironment=CUDA_VISIBLE_DEVICES='+('' if target=='tick' else spec['gpu_uuid'])+'\n'
        body+='Environment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n'
        body+='ExecStart='+shlex.join(args)+'\nMemoryMax='+('1G' if target=='tick' else '6G')+'\nMemorySwapMax=0\nTimeoutStartSec='+str(max(1900,GPU_WALL+100))+'\nNice=10\n'
        timer='[Unit]\nDescription=Unattended weekly plus18\n[Timer]\nOnBootSec=10s\nOnUnitInactiveSec=15s\nAccuracySec=1s\nPersistent=true\n[Install]\nWantedBy=timers.target\n'
        code="from pathlib import Path; d=Path.home()/'.config/systemd/user'; d.mkdir(parents=True,exist_ok=True); "
        for suffix,text in (('service',body),('timer',timer)):
            code+='p=d/'+repr(NAME+'-'+target+'.'+suffix)+'; assert not p.exists() or '+repr(NAME)+' in p.read_text(); p.write_text('+repr(text)+'); '
        call=remote if target=='gpu' else run_cmd
        call(['python3','-c',code]);call(['systemctl','--user','daemon-reload'])
        call(['systemctl','--user','enable','--now',NAME+'-'+target+'.timer'])
    B.put(root/'LAUNCH.json',dict(state='ARMED',code_commit=commit,total=len(p['cells']),test_read=False))
    return dict(state='ARMED',batch_id=NAME,total=len(p['cells']),code_commit=commit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','launch','run','tick','status'))
    parser.add_argument('--spec',required=True);parser.add_argument('--cell')
    args=parser.parse_args();spec=read(args.spec)
    try:
        if args.action=='status':
            p=Path(spec['root'])/'STATUS.json';result=read(p) if p.exists() else dict(state='NOT_STARTED')
            if args.cell:result=dict(status=result.get('cells',{}).get(args.cell,{'state':'PENDING'}),progress=result.get('progress'))
            else:result.pop('cells',None)
        else:result=globals()[args.action](spec)
        if result:
            if args.action=='prepare':result={k:v for k,v in result.items() if k not in ('files','cells')}
            print(json.dumps(result,sort_keys=True))
    except BlockingIOError:
        return
    except Exception as exc:
        if args.action=='tick':B.put(Path(spec['root'])/'STATUS.json',dict(state='WAIT_OPERATIONAL_RETRY',error=type(exc).__name__+':'+str(exc)[:600],at=time.time()))
        raise


if __name__=='__main__':main()
