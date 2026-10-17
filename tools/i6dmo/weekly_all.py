"""Append five long heads to the frozen weekly seven-head experiment."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np

from tools.i6dmo import weekly18 as W

BASE = "bd21fc2fc22a6e8607df14d68412223dc97a64d5"
NAME = "i6dmo-weekly-all-20261017"
PARENT_HEADS = [2, 3, 4, 5, 6, 12, 18]
EXTRA = [24, 36, 48, 60, 72]
HEADS = PARENT_HEADS + EXTRA
OLD_CATALOGUE = "HORIZONS = (*R.HORIZONS, (4,), (2, 3, 4, 5, 6, 12, 18))"
NEW_CATALOGUE = "HORIZONS = (*R.HORIZONS, (4,), (2, 3, 4, 5, 6, 12, 18), (2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72))"


def extend_model(parent):
    if parent['horizons_hours'] != PARENT_HEADS or parent['slots'] != 13 or parent['core'] != [2] \
            or parent['head'] != 'lstm' or parent['kind'] != 'conv' or parent['seed'] != W.B.SEED \
            or parent['spacing_open_market_hours'] != 2:
        raise ValueError('WRONG_SEVEN_HEAD_PARENT')
    return dict(copy.deepcopy(parent), horizons_hours=HEADS)


def assert_projection(old, new):
    if set(old) != set(new): raise ValueError('PARENT_ARRAY_CHANGED:keys')
    for key, before in old.items():
        after = new[key][:,:7] if key in ('target_price','mask') else new[key]
        if not np.array_equal(before,after,equal_nan=before.dtype.kind in 'fc'):
            raise ValueError('PARENT_ARRAY_CHANGED:'+key)


def label_horizon(a, source, calendar, period, split, base, h):
    from tools.i6n_contracts import eligibility as E
    from tools.i6dmo import data
    if h not in EXTRA: raise ValueError('UNDECLARED_EXTRA_HORIZON')
    _,weeks=E.load_calendar(calendar); by_id={w.week_id:w for w in weeks}
    instant=lambda t:datetime.fromtimestamp(int(t),timezone.utc)
    observed=E.ObservedTargets({instant(v):instant(v) for v in source['ts']})
    if split=='FIT':start,end=(data._instant(period[k]) for k in ('train_start','train_end'))
    else:start,end=(data._instant(base['splits']['validation'][k]) for k in ('start','end'))
    bounds=E.SplitBounds(split,start,end,label_available_by=end)
    target=np.full(len(a['row_ids']),np.nan);mask=np.zeros(len(target),bool)
    for i,(t,wid) in enumerate(zip(a['row_ids'],a['week_ids'],strict=True)):
        d=E.evaluate_heads(instant(t),[h],E.Role.MANAGEMENT,by_id[str(wid)],
            input_available=True,observed_targets=observed,split=bounds)[0]
        if d.label_eligible:
            q=int(t)+h*3600; ix=np.searchsorted(source['ts'],q)
            if ix==len(source['ts']) or source['ts'][ix]!=q or not np.isfinite(source['price'][ix]):
                raise ValueError('EXACT_TARGET_MISSING:'+str(h))
            target[i],mask[i]=source['price'][ix],True
    return target,mask


def source_identity(parent):
    if parent['code_commit']!=BASE or parent['training']!=W.B.TRAINING or parent['seed']!=W.B.SEED:
        raise ValueError('PARENT_IDENTITY_CHANGED')
    for rel,sha in parent['code_files'].items():
        if rel not in ('tools/i6dmo/weekly18.py','tools/i6dmo/continuation_models.py') and W.file_sha256(W.REPO/rel)!=sha:
            raise ValueError('PARENT_SOURCE_CHANGED:'+rel)
    old=W.run_cmd(['git','-C',W.REPO,'show',BASE+':tools/i6dmo/continuation_models.py'])
    if (W.REPO/'tools/i6dmo/continuation_models.py').read_text()!=old.replace(OLD_CATALOGUE,NEW_CATALOGUE):
        raise ValueError('MODEL_CHANGED_BEYOND_CATALOGUE')
    old=W.run_cmd(['git','-C',W.REPO,'show',BASE+':tools/i6dmo/weekly18.py'])
    old=old.replace('CATALOGUE = '+repr(OLD_CATALOGUE).replace("'",'"'), 'CATALOGUE = '+repr(NEW_CATALOGUE).replace("'",'"')+
        '\nMODULE = "tools.i6dmo.weekly18"\nPHASE = "I6DMO_WEEKLY_PLUS18"\nTITLE = "# Weekly Short Plus 18 Hours"')
    old=old.replace('phase="I6DMO_WEEKLY_PLUS18"','phase=PHASE')
    old=old.replace("report['phase']='I6DMO_WEEKLY_PLUS18'","report['phase']=PHASE")
    old=old.replace("lines=['# Weekly Short Plus 18 Hours',","lines=[TITLE,")
    old=old.replace("'-m','tools.i6dmo.weekly18'","'-m',MODULE")
    if old!=(W.REPO/'tools/i6dmo/weekly18.py').read_text():
        raise ValueError('SHARED_ADAPTER_CHANGED_BEYOND_MODULE_AND_PRESENTATION')


def extend_period(parent_arrays, parent_record, parent_result, source, manifest):
    from tools.i6dmo import continuation as C, resolution_data as R
    arrays={}; naive_extra=[]; additions={}
    for split,a in parent_arrays.items():
        additions[split]=[label_horizon(a,source,manifest['calendar'],parent_record['cell']['period'],split,manifest['base_design'],h) for h in EXTRA]
    for target,mask in additions['FIT']:
        if not mask.any():raise ValueError('EXTRA_FIT_UNSUPPORTED')
        naive_extra.append(math.fsum(np.abs(target[mask]-parent_arrays['FIT']['origin_price'][mask]).tolist())/int(mask.sum()))
    scales=copy.deepcopy(parent_record['scales'])
    scales.update(naive_mae=list(scales['naive_mae'])+naive_extra,horizons=HEADS)
    scales.pop('scales_sha256',None);scales['scales_sha256']=W.B.digest(scales)
    for split,a in parent_arrays.items():
        arrays[split]={**a,'target_price':np.column_stack([a['target_price']]+[v[0] for v in additions[split]]),
            'mask':np.column_stack([a['mask']]+[v[1] for v in additions[split]])}
        assert_projection(a,arrays[split])
    model=extend_model(parent_record['model'])
    meta=dict(parent_record['population'],horizons=HEADS,full_horizons=HEADS,
        parent_prepared_period_sha256=W.B.digest(parent_record),
        scales_sha256=scales['scales_sha256'],counts={s:a['mask'].sum(axis=0).tolist() for s,a in arrays.items()},
        validation_counts=arrays['VALIDATION']['mask'].sum(axis=0).tolist())
    record=dict(cell=dict(parent_record['cell'],horizons=HEADS),parent_state='COMPLETE',
        parent_result_sha256=parent_result['sha256'],parent_metrics=parent_result['metrics'],
        parent_model=parent_record['model'],model=model,resolved=parent_record['resolved'],
        parent_initial_weights_sha256=parent_result['initial_weights_sha256'],scales=scales,population=meta,
        training=parent_record['training'],projection=dict(state='EXACT',preserved_heads=PARENT_HEADS,
            parent_array_digests={s:{k:R._digest(v) for k,v in a.items()} for s,a in parent_arrays.items()}),
        naive_cache={s:C.measured_metrics(a,np.where(a['mask'],a['target_price'],0),HEADS) for s,a in arrays.items()})
    for split in arrays:
        for actual,old in zip(record['naive_cache'][split][:7],parent_result['metrics'][split],strict=True):
            if any(actual[k]!=old[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error','pairs_sha256')):
                raise ValueError('PARENT_SEVEN_NAIVES_CHANGED')
    return arrays,record


def prepare(spec):
    from tools.i6dmo import continuation_data as D
    parent=W.read(spec['parent_spec']);source_identity(parent)
    source_root=Path(parent['data_root']);prior=W.prepared_document(parent,source_root)
    W.authenticate(source_root,prior['files'])
    out=Path(spec['data_root']);out.mkdir(parents=True,exist_ok=True)
    if (out/'PREPARED.json').exists():
        p=W.read(out/'PREPARED.json');W.authenticate(out,p['files'])
        if p['parent_spec_sha256']!=W.file_sha256(spec['parent_spec']):raise ValueError('PARENT_SPEC_CHANGED')
        return p
    manifest=D._verify(spec['parent_data'],spec['input_dir'])
    source=D._read(Path(spec['parent_data'])/'SOURCE.npz')
    files={};cells=[]
    for c in prior['cells']:
        terminal=W.terminal(parent['artifacts'],c,parent)
        if not terminal or terminal['state'] not in ('COMPLETE','SKIPPED_NO_EVAL_ROWS'):
            raise ValueError('PARENT_WEEK_NOT_COMPLETE:'+c['id'])
        record=W.read(source_root/c['id']/'PERIOD.json');folder=out/c['id'];folder.mkdir(exist_ok=True)
        if terminal['state']=='COMPLETE':
            old={s:D._read(source_root/c['id']/(s+'.npz')) for s in ('FIT','VALIDATION')}
            arrays,record=extend_period(old,record,terminal,source,manifest)
            for split,a in arrays.items():
                np.savez_compressed(folder/(split+'.npz'),**a)
                files[c['id']+'/'+split+'.npz']=W.file_sha256(folder/(split+'.npz'))
        else:record=dict(record,cell=dict(c,horizons=HEADS),parent_state=terminal['state'],parent_result_sha256=terminal['sha256'])
        W.B.put(folder/'PERIOD.json',record);files[c['id']+'/PERIOD.json']=W.file_sha256(folder/'PERIOD.json')
        cells.append(record['cell']);W.B.put(out/'PREPARE_PROGRESS.json',dict(terminal=len(cells),total=len(prior['cells']),at=time.time()))
    p=dict(schema='i6dmo.weekly_all.prepared.v1',cells=cells,files=files,seed=W.B.SEED,
        original_training=W.B.TRAINING,parent_spec_sha256=W.file_sha256(spec['parent_spec']),
        parent_prepared_sha256=prior['sha256'],test_read=False,qualification=W.QUALIFICATION)
    p['sha256']=W.B.digest(p);W.B.put(out/'PREPARED.json',p);return p


def initialization_proof(record,inputs=None):
    from tools.i6dmo import continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    old=M.build(record['parent_model']);new=M.build(record['model'])
    if weights_hash(old.forecast_model)!=record['parent_initial_weights_sha256']:raise ValueError('PARENT_INITIAL_WEIGHTS_CHANGED')
    lookup={v.path:v.numpy() for v in new.forecast_model.weights}
    for v in old.forecast_model.weights:
        if v.path not in lookup or not np.array_equal(v.numpy(),lookup[v.path]):raise ValueError('PARENT_SEVEN_INITIAL_WEIGHT_CHANGED:'+v.path)
    if inputs is not None:
        before=old.forecast_model(inputs,training=False);after=new.forecast_model(inputs,training=False)
        if len(before)!=7 or len(after)!=12 or any(not np.array_equal(a.numpy(),b.numpy()) for a,b in zip(before,after[:7],strict=True)):
            raise ValueError('PARENT_SEVEN_FORWARD_CHANGED')
    return dict(state='EXACT',compared_tensors=len(old.forecast_model.weights),preserved_heads=PARENT_HEADS,
        parent_initial_weights_sha256=weights_hash(old.forecast_model),new_initial_weights_sha256=weights_hash(new.forecast_model),
        short_forward='EXACT' if inputs is not None else 'NOT_REQUESTED')


def preflight(spec):
    W.verify(spec);p=W.prepared_document(spec,spec['gpu_data']);W.authenticate(spec['gpu_data'],p['files'])
    from tools.i6dmo import train as T,continuation_data as D
    import tensorflow as tf
    if tf.__version__!='2.21.0' or importlib.metadata.version('keras')!='3.13.2' or not tf.config.list_physical_devices('GPU'):
        raise ValueError('ORIGINAL_GPU_RUNTIME_REQUIRED')
    for cell in p['cells']:
        r=W.read(Path(spec['gpu_data'])/cell['id']/'PERIOD.json')
        if r['parent_state']!='COMPLETE':continue
        if r['training']!=W.B.TRAINING or extend_model(r['parent_model'])!=r['model']:raise ValueError('PERIOD_CONFIG_CHANGED')
        for split in ('FIT','VALIDATION'):
            for current,old in zip(r['naive_cache'][split][:7],r['parent_metrics'][split],strict=True):
                if any(current[k]!=old[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error','pairs_sha256')):
                    raise ValueError('PARENT_NAIVE_CHANGED')
    record=next(W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json') for c in p['cells'] if W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json')['parent_state']=='COMPLETE')
    a=D._read(Path(spec['gpu_data'])/record['cell']['id']/'FIT.npz');x,_,_=T.transform(a,record['scales'])
    proof=initialization_proof(record,[x[:2],a['offsets_hours'][:2].astype('float32'),a['closure_crossed'][:2].astype('float32')])
    result=dict(state='PASS',periods=len(p['cells']),initialization=proof,fits_run=0,test_read=False)
    W.B.put(Path(spec['gpu_root'])/'PREFLIGHT.json',result);return result


def configure():
    W.NAME=NAME;W.BASE=BASE;W.SHORT=PARENT_HEADS;W.HEADS=HEADS
    W.MODULE='tools.i6dmo.weekly_all';W.PHASE='I6DMO_WEEKLY_ALL';W.TITLE='# Weekly Twelve-Horizon Contrast'
    W.prepare=prepare;W.source_identity=source_identity;W.initialization_proof=initialization_proof
    W.extend_model=extend_model;W.preflight=preflight


def stop_successor():
    W.run_cmd(['systemctl','--user','disable','--now',NAME+'-successor.timer'])


def dispatch(spec):
    # Recovery after a completed install must never rerun a GPU preflight next to a fit.
    root=Path(spec['root'])
    if (root/'LAUNCH.json').exists():return W.read(root/'LAUNCH.json')
    if (root/'RUN.private.json').exists():
        run=W.read(root/'RUN.private.json')
        active=subprocess.run(['ssh',run['ssh_alias'],'systemctl','--user','is-active',NAME+'-gpu.timer',NAME+'-gpu.service'],capture_output=True,text=True,timeout=30)
        if active.returncode==0:return dict(state='ARMED',code_commit=run['code_commit'],recovered_existing_timers=True)
    deploy=W.read(spec['source_deployment']);alias=deploy['roles']['preferred']['ssh_alias']
    remote=lambda args:W.run_cmd(['ssh','-o','BatchMode=yes',alias,shlex.join([str(x) for x in args])])
    if remote(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip():
        return dict(state='WAIT_GPU',reason='existing compute process; not interrupted')
    gpu=remote(['nvidia-smi','-i',spec['gpu_uuid'],'--query-gpu=name','--format=csv,noheader'])
    if '5090' not in gpu:return dict(state='WAIT_GPU_HARDWARE')
    memory=int(remote(['python3','-c',"from pathlib import Path; print(next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))"]).strip())
    if memory<7*1024**3:return dict(state='WAIT_RESOURCES',memory_available=memory)
    storage=Path(deploy['roles']['coordinator']['experiment_storage_root'])
    actual_fs=W.run_cmd(['findmnt','-n','-o','FSTYPE','-T',storage]).strip()
    if actual_fs!=spec['required_storage_fstype'] or shutil.disk_usage(storage).free<5*1024**3:
        return dict(state='WAIT_STORAGE',filesystem=actual_fs)
    return W.launch(spec)


def successor(spec):
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'successor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'SUCCESSOR_DONE.json').exists():stop_successor();return W.read(root/'SUCCESSOR_DONE.json')
        close=Path(spec['parent_root'])/'CLOSE.json'
        if not close.exists():result=dict(state='WAIT_PREDECESSOR',at=time.time(),test_read=False)
        else:
            c=W.read(close);pub=c.get('publication',{})
            if c.get('state')!='CLOSED' or pub.get('pending')!=0 or pub.get('mismatch')!=0 or c.get('mirrors',{}).get('state')!='VERIFIED':
                result=dict(state='BLOCKED_PREDECESSOR',reason='parent closure/publication/mirrors not complete',at=time.time())
            else:
                result=dispatch(spec);result['parent_close_sha256']=W.file_sha256(close)
                if result.get('state')=='ARMED':
                    W.B.put(root/'SUCCESSOR_DONE.json',result);stop_successor()
        W.B.put(root/'SUCCESSOR_STATUS.json',result);return result


def install_successor(spec_path):
    spec_path=Path(spec_path).resolve();spec=W.read(spec_path)
    if W.run_cmd(['git','-C',W.REPO,'status','--porcelain']).strip():raise ValueError('SOURCE_NOT_FROZEN')
    deploy=W.read(spec['source_deployment']);python=deploy['roles']['coordinator']['python']
    args=[deploy['launcher']['crispdm_run'],'-m','768M','-t','600s','-n',NAME+'-successor','--',python,
        '-B','-m','tools.i6dmo.weekly_all','successor','--spec',str(spec_path)]
    folder=Path.home()/'.config/systemd/user';folder.mkdir(parents=True,exist_ok=True)
    service='[Unit]\nDescription=Automatic twelve-head successor\nX-I6DMO-Batch='+NAME+'\n[Service]\nType=oneshot\nWorkingDirectory='+str(W.REPO)+'\n'
    service+='Environment=PYTHONPATH='+str(W.REPO)+'\nEnvironment=CUDA_VISIBLE_DEVICES=\nEnvironment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n'
    service+='ExecStart='+shlex.join(args)+'\nMemoryMax=1G\nMemorySwapMax=0\nTimeoutStartSec=650\nNice=10\n'
    timer='[Unit]\nDescription=Unattended twelve-head handoff\n[Timer]\nOnBootSec=10s\nOnUnitInactiveSec=30s\nAccuracySec=1s\nPersistent=true\n[Install]\nWantedBy=timers.target\n'
    for suffix,text in (('service',service),('timer',timer)):
        p=folder/(NAME+'-successor.'+suffix)
        if p.exists() and NAME not in p.read_text():raise ValueError('FOREIGN_UNIT')
        p.write_text(text)
    W.run_cmd(['systemctl','--user','daemon-reload']);W.run_cmd(['systemctl','--user','enable','--now',NAME+'-successor.timer'])
    return dict(state='SUCCESSOR_ARMED',parent_untouched=True,test_read=False)


def main():
    configure()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','preflight','launch','run','tick','status','successor','install-successor'])
    p.add_argument('--spec',required=True);p.add_argument('--cell');a=p.parse_args();spec=W.read(a.spec)
    try:
        if a.action=='status':
            root=Path(spec['root']);path=root/'STATUS.json'
            result=W.read(path) if path.exists() else W.read(root/'SUCCESSOR_STATUS.json') if (root/'SUCCESSOR_STATUS.json').exists() else dict(state='NOT_STARTED')
            if a.cell:result=dict(cell=result.get('cells',{}).get(a.cell,{'state':'PENDING'}),progress=result.get('progress'))
            else:result.pop('cells',None)
        elif a.action=='install-successor':result=install_successor(a.spec)
        elif a.action=='successor':result=successor(spec)
        else:result=getattr(W,a.action)(spec)
        if result:print(json.dumps({k:v for k,v in result.items() if a.action!='prepare' or k not in ('cells','files')},sort_keys=True))
    except BlockingIOError:return
    except Exception as exc:
        if a.action in ('successor','tick'):
            W.B.put(Path(spec['root'])/('SUCCESSOR_STATUS.json' if a.action=='successor' else 'STATUS.json'),
                dict(state='WAIT_OPERATIONAL_RETRY',error=type(exc).__name__+':'+str(exc)[:600],at=time.time()))
        raise


if __name__=='__main__':main()
