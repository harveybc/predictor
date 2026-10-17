"""Finite paired window sweep, followed by conditional joint core contrasts."""
import argparse
import copy
import fcntl
from functools import cmp_to_key
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from tools.i6dmo import weekly18 as W

CODE_BASE = "469eef418c2519959f718d40bdc06e6ee66e256e"
PARENT_BASE = "bd21fc2fc22a6e8607df14d68412223dc97a64d5"
BUNDLE_BASE = "796ec8f48b0b1621fb9fa9e60e12c1c37079e51e"
NAME = "i6dmo-weekly-window-core-20261017"
HEADS = [2,3,4,5,6,12,18]
SLOT_CATALOGUE = "SLOTS = (6, 7, 9, 13, 18, 24, 25, 49)"
_run, _tick, _annual_report = W.run, W.tick, W.annual_report


def arms():
    return [dict(id='window'+str(n),slots=n,core=[2],stage='windows') for n in (13,7,25,49)]+[
        dict(id='window'+str(n)+'_core'+'_'.join(map(str,c)),slots=n,core=c,stage='cores')
        for n in (7,13,25,49) for c in ([4],[4,2])]


def subset(a, valid):
    return {k:v[valid] for k,v in a.items()}


def window_arrays(old, source, grid, slots):
    origins=old['row_ids']; at=np.searchsorted(grid,origins)
    safe_at=np.minimum(at,len(grid)-1)
    idx=at[:,None]-2*np.arange(slots-1,-1,-1)[None,:]
    valid=(at<len(grid)) & (grid[safe_at]==origins) & (idx[:,0]>=0)
    stamps=grid[np.clip(idx,0,len(grid)-1)]
    rows=np.searchsorted(source['ts'],stamps); safe=np.minimum(rows,len(source['ts'])-1)
    values=source['x'][safe]
    valid &= ((rows<len(source['ts'])) & (source['ts'][safe]==stamps)).all(axis=1)
    valid &= np.isfinite(values).all(axis=(1,2))
    closure=np.zeros(stamps.shape,bool);closure[:,1:]=np.diff(stamps,axis=1)>7200
    return dict(old,x=values,offsets_hours=(stamps-origins[:,None])/3600.,closure_crossed=closure),valid


def best_windows(summaries):
    rows=[summaries.get(a['id'],{}) for a in arms()[:4]]
    if any(r.get('state')!='COMPLETE' for r in rows):return None
    reference=rows[0]['metrics']
    for r in rows:
        if [m['horizon'] for m in r['metrics']]!=HEADS:raise ValueError('UNPAIRED_ANNUAL:horizons')
        for left,right in zip(reference,r['metrics'],strict=True):
            if left['n']!=right['n'] or left['naive_mae']!=right['naive_mae']:
                raise ValueError('UNPAIRED_ANNUAL:naive_or_count')
            if left.get('support_sha256')!=right.get('support_sha256'):
                raise ValueError('UNPAIRED_ANNUAL:support')
    scores={r['slots']:math.fsum(m['relative_error'] for m in r['metrics'])/len(HEADS) for r in rows}
    def compare(a,b):
        d=scores[a]-scores[b]
        return (a>b)-(a<b) if abs(d)<=1e-8 else (d>0)-(d<0)
    return sorted(scores,key=cmp_to_key(compare))[:2]


def arm_disposition(arm, selected):
    if arm['stage']=='windows':return 'RUN'
    if selected is None:return 'BLOCKED_DEPENDENCY'
    return 'RUN' if arm['slots'] in selected else 'SKIPPED_NOT_SELECTED'


def source_identity(parent):
    if parent['code_commit']!=PARENT_BASE or parent['training']!=W.B.TRAINING or parent['seed']!=W.B.SEED:
        raise ValueError('PARENT_IDENTITY_CHANGED')
    allowed={'tools/i6dmo/weekly18.py','tools/i6dmo/continuation_models.py','tools/i6dmo/resolution_models.py'}
    for rel,sha in parent['code_files'].items():
        if rel not in allowed and W.file_sha256(W.REPO/rel)!=sha:raise ValueError('PARENT_SOURCE_CHANGED:'+rel)
    for rel in W.FROZEN+('tools/i6dmo/continuation_models.py',):
        old=W.run_cmd(['git','-C',W.REPO,'show',CODE_BASE+':'+rel])
        if (W.REPO/rel).read_text()!=old:raise ValueError('SCIENTIFIC_SOURCE_CHANGED:'+rel)
    old=W.run_cmd(['git','-C',W.REPO,'show',CODE_BASE+':tools/i6dmo/resolution_models.py'])
    if (W.REPO/'tools/i6dmo/resolution_models.py').read_text()!=old.replace('SLOTS = (6, 9, 13, 18, 24)',SLOT_CATALOGUE):
        raise ValueError('MODEL_CHANGED_BEYOND_SLOT_CATALOGUE')


def prepare(spec):
    from tools.i6dmo import continuation_data as D, continuation as C
    from tools.i6n_contracts import eligibility as E
    parent=W.read(spec['parent_spec']);source_identity(parent)
    prior=W.prepared_document(parent,parent['data_root']);W.authenticate(parent['data_root'],prior['files'])
    out=Path(spec['data_root']);out.mkdir(parents=True,exist_ok=True)
    if (out/'PREPARED.json').exists():
        p=W.read(out/'PREPARED.json');W.authenticate(out,p['files'])
        if p['parent_spec_sha256']!=W.file_sha256(spec['parent_spec']):raise ValueError('PARENT_SPEC_CHANGED')
        return p
    manifest=D._verify(spec['parent_data'],spec['input_dir'])
    source=D._read(Path(spec['parent_data'])/'SOURCE.npz')
    _,weeks=E.load_calendar(manifest['calendar'])
    grid=np.array([t for w in weeks for t in range(int(w.market_open.timestamp())+3600,int(w.market_close.timestamp())+1,3600)])
    if np.any(np.diff(grid)<=0):raise ValueError('CALENDAR_NOT_ORDERED')
    files={};records={a['id']:[] for a in arms()};exclusions=[]
    for pc in prior['cells']:
        terminal=W.terminal(parent['artifacts'],pc,parent)
        if not terminal or terminal['state'] not in ('COMPLETE','SKIPPED_NO_EVAL_ROWS'):raise ValueError('PARENT_NOT_COMPLETE')
        original=W.read(Path(parent['data_root'])/pc['id']/'PERIOD.json')
        generated={};supports={};scales=None
        if terminal['state']=='COMPLETE':
            for split in ('FIT','VALIDATION'):
                old=D._read(Path(parent['data_root'])/pc['id']/(split+'.npz'))
                windows={n:window_arrays(old,source,grid,n) for n in (7,13,25,49)}
                same,valid13=windows[13]
                if not valid13.all() or any(not np.array_equal(old[k],same[k],equal_nan=True)
                        for k in ('x','offsets_hours','closure_crossed')):raise ValueError('ORIGINAL_13_WINDOW_CHANGED')
                common=np.logical_and.reduce([v for _,v in windows.values()])
                if split=='FIT' and not common.any():raise ValueError('COMMON_FIT_SUPPORT_EMPTY')
                supports[split]=dict(original=len(common),retained=int(common.sum()),excluded=int((~common).sum()),
                    support_sha256=D.R._digest(old['row_ids'][common]))
                for n,(a,_) in windows.items():generated[n,split]=subset(a,common)
            scales=copy.deepcopy(original['scales']);fit=generated[13,'FIT']
            ns=[]
            for j in range(len(HEADS)):
                mask=fit['mask'][:,j]
                if not mask.any():raise ValueError('FIT_HEAD_EMPTY')
                ns.append(math.fsum(np.abs(fit['target_price'][mask,j]-fit['origin_price'][mask]).tolist())/int(mask.sum()))
            scales.update(naive_mae=ns);scales.pop('scales_sha256',None);scales['scales_sha256']=W.B.digest(scales)
        disposition=terminal['state']
        if generated and not generated[13,'VALIDATION']['mask'].any():disposition='SKIPPED_NO_EVAL_ROWS'
        for arm in arms():
            cid=arm['id']+'__'+pc['id'];folder=out/cid;folder.mkdir(exist_ok=True)
            cell=dict(pc,id=cid,arm=arm['id'])
            record=dict(original,cell=cell,parent_state=disposition,arm=arm,
                parent_result_sha256=terminal['sha256'],model_delta=dict(slots=arm['slots'],core=arm['core']))
            if disposition=='SKIPPED_NO_EVAL_ROWS':record['skip_reason']='no common observed validation inputs across declared windows' if generated else 'same empty parent period'
            if disposition=='COMPLETE':
                model=dict(original['model'],slots=arm['slots'],core=arm['core'])
                if model['horizons_hours']!=HEADS:raise ValueError('WRONG_PARENT_HEADS')
                record.update(parent_model=original['model'],model=model,scales=scales,
                    parent_initial_weights_sha256=terminal['initial_weights_sha256'],
                    resolved=dict(original['resolved'],slots=arm['slots'],core=arm['core']),
                    projection=dict(state='PAIRED_COMMON_SUPPORT',support=supports),
                    population=dict(original['population'],slots=arm['slots'],common_support=supports,
                        scales_sha256=scales['scales_sha256'],
                        counts={s:generated[arm['slots'],s]['mask'].sum(axis=0).tolist() for s in ('FIT','VALIDATION')},
                        validation_counts=generated[arm['slots'],'VALIDATION']['mask'].sum(axis=0).tolist()),
                    naive_cache={s:C.measured_metrics(generated[arm['slots'],s],
                        np.where(generated[arm['slots'],s]['mask'],generated[arm['slots'],s]['target_price'],0),HEADS)
                        for s in ('FIT','VALIDATION')})
                for split in ('FIT','VALIDATION'):
                    target=folder/(split+'.npz')
                    if arm['stage']=='windows':np.savez_compressed(target,**generated[arm['slots'],split])
                    else:shutil.copyfile(out/('window'+str(arm['slots'])+'__'+pc['id'])/(split+'.npz'),target)
                    files[cid+'/'+split+'.npz']=W.file_sha256(target)
            W.B.put(folder/'PERIOD.json',record);files[cid+'/PERIOD.json']=W.file_sha256(folder/'PERIOD.json')
            records[arm['id']].append(cell)
        exclusions.append(dict(period=pc['id'],splits=supports))
        W.B.put(out/'PREPARE_PROGRESS.json',dict(periods=len(exclusions),total=len(prior['cells']),at=time.time()))
    W.B.put(out/'SUPPORT.json',exclusions);files['SUPPORT.json']=W.file_sha256(out/'SUPPORT.json')
    nonempty=sum(W.read(out/c['id']/'PERIOD.json')['parent_state']=='COMPLETE' for c in records['window13'])
    p=dict(schema='i6dmo.weekly_window_core.prepared.v1',cells=[c for a in arms() for c in records[a['id']]],
        arms=arms(),files=files,parent_spec_sha256=W.file_sha256(spec['parent_spec']),parent_prepared_sha256=prior['sha256'],
        nonempty_periods=nonempty,planned_actual_fits=8*nonempty,
        original_training=W.B.TRAINING,seed=W.B.SEED,test_read=False,qualification=W.QUALIFICATION)
    p['sha256']=W.B.digest(p);W.B.put(out/'PREPARED.json',p);return p


def initialization_proof(record, inputs=None):
    from tools.i6dmo import continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    old=M.build(record['parent_model']);new=M.build(record['model'])
    parent_hash=weights_hash(old.forecast_model)
    if parent_hash!=record['parent_initial_weights_sha256']:raise ValueError('PARENT_INITIAL_WEIGHTS_CHANGED')
    if record['arm']['core']==[2]:
        if old.forecast_model.count_params()!=1901 or new.forecast_model.count_params()!=1901:
            raise ValueError('WINDOW_CHANGED_CAPACITY')
        if weights_hash(new.forecast_model)!=parent_hash:raise ValueError('WINDOW_CHANGED_INITIAL_WEIGHTS')
        if inputs is not None and record['arm']['slots']==13:
            if any(not np.array_equal(a.numpy(),b.numpy()) for a,b in zip(
                    old.forecast_model(inputs,training=False),new.forecast_model(inputs,training=False),strict=True)):
                raise ValueError('CONTROL_FORWARD_CHANGED')
    return dict(state='EXACT_WINDOW_WEIGHTS' if record['arm']['core']==[2] else 'DECLARED_CORE_CHANGE',
        parameters=int(new.forecast_model.count_params()),parent_initial_weights_sha256=parent_hash,
        initial_weights_sha256=weights_hash(new.forecast_model),model_delta=record['model_delta'])


def preflight(spec):
    W.verify(spec);p=W.prepared_document(spec,spec['gpu_data']);W.authenticate(spec['gpu_data'],p['files'])
    import tensorflow as tf
    if tf.__version__!='2.21.0' or importlib.metadata.version('keras')!='3.13.2' or not tf.config.list_physical_devices('GPU'):
        raise ValueError('ORIGINAL_GPU_RUNTIME_REQUIRED')
    from tools.i6dmo import continuation_data as D, train as T
    proofs={}
    supports={}
    for cell in p['cells']:
        record=W.read(Path(spec['gpu_data'])/cell['id']/'PERIOD.json')
        if record['parent_state']!='COMPLETE':continue
        arm=next(a for a in arms() if a['id']==cell['arm'])
        if record['arm']!=arm or record['training']!=W.B.TRAINING or record['model']!=dict(
                record['parent_model'],slots=arm['slots'],core=arm['core']):raise ValueError('UNDECLARED_MODEL_CHANGE')
        key=cell['period']['id']
        signature=W.B.digest(dict(naive=record['naive_cache'],scales=record['scales'],support=record['projection']['support']))
        if key in supports and supports[key]!=signature:raise ValueError('UNPAIRED_PERIOD_DATA')
        supports[key]=signature
    for arm in arms():
        record=next(W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json') for c in p['cells']
            if c['arm']==arm['id'] and W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json')['parent_state']=='COMPLETE')
        a=D._read(Path(spec['gpu_data'])/record['cell']['id']/'FIT.npz');x,_,_=T.transform(a,record['scales'])
        proofs[arm['id']]=initialization_proof(record,[x[:2],a['offsets_hours'][:2].astype('float32'),a['closure_crossed'][:2].astype('float32')])
        tf.keras.backend.clear_session()
    result=dict(state='PASS',arms=proofs,logical_cells=len(p['cells']),planned_fits=p['planned_actual_fits'],fits_run=0,test_read=False)
    W.B.put(Path(spec['gpu_root'])/'PREFLIGHT.json',result);return result


def summaries(spec, artifacts, results):
    out={}
    for arm in arms():
        cells=[c for c in spec['cells'] if c['arm']==arm['id']]
        rs=[results[c['id']] for c in cells if c['id'] in results]
        entry=dict(arm,terminal=len(rs),total=len(cells),state='INCOMPLETE',metrics=[])
        if len(rs)==len(cells):
            states={r['state'] for r in rs}
            if states<= {'COMPLETE','SKIPPED_NO_EVAL_ROWS'}:
                rows=[];seen={h:[] for h in HEADS}
                for c,r in zip(cells,rs,strict=True):
                    if r['state']!='COMPLETE':continue
                    with np.load(Path(artifacts)/'results'/c['id']/'VALIDATION.npz',allow_pickle=False) as z:
                        for j,h in enumerate(HEADS):
                            mask=z['mask'][:,j]
                            seen[h].extend(zip(z['origin_ts'][mask].tolist(),z['origin_price'][mask].tolist(),z['target_price'][mask,j].tolist()))
                    rows.extend(r['metrics']['VALIDATION'])
                for h in HEADS:
                    pairs=seen[h]
                    if len({p[0] for p in pairs})!=len(pairs):raise ValueError('WEEK_OVERLAP')
                    metric=W.pool_metrics(rows,h)
                    if metric['n']!=len(pairs) or not metric['n']:raise ValueError('ANNUAL_COUNT_MISMATCH')
                    metric['support_sha256']=W.B.digest(sorted(pairs));entry['metrics'].append(metric)
                entry['state']='COMPLETE'
            elif states<= {'SKIPPED_NOT_SELECTED'}:entry['state']='SKIPPED_NOT_SELECTED'
            else:entry['state']='FAILED'
        out[arm['id']]=entry
    return out


def run(spec):
    W.verify(spec);root=Path(spec['gpu_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={c['id']:r for c in spec['cells'] if (r:=W.terminal(root,c,spec))}
        if all(c['id'] in results for c in spec['cells'] if c['arm'].startswith('window') and '_core' not in c['arm']):
            selected=best_windows(summaries(spec,root,results))
            W.B.put(root/'SELECTION.json',dict(selected_windows=selected,test_read=False))
            for c in spec['cells']:
                if c['id'] in results:continue
                arm=next(a for a in arms() if a['id']==c['arm'])
                disposition=arm_disposition(arm,selected)
                if disposition!='RUN':
                    r=dict(batch_id=NAME,cell=c,state=disposition,files={},reason='complete annual coarse selection')
                    r['sha256']=W.B.digest(r);W.B.put(root/'results'/c['id']/'RESULT.json',r)
    return _run(spec)


def annual(spec, artifacts, results):
    s=summaries(spec,artifacts,results);selected=best_windows(s)
    complete=[r for r in s.values() if r['state']=='COMPLETE']
    winner=min(complete,key=lambda r:(math.fsum(m['relative_error'] for m in r['metrics'])/len(HEADS),r['slots'],len(r['core']))) if complete else None
    done=len(results)==len(spec['cells'])
    state='COMPLETE' if done and selected and not any(r['state']=='FAILED' for r in s.values()) else 'INCOMPLETE'
    result=dict(state=state,arms=s,selected_windows=selected,winner=winner['id'] if winner else None,
        metrics=winner['metrics'] if done and winner else [],qualification=W.QUALIFICATION,test_read=False)
    W.B.put(Path(artifacts)/'ARMS.json',result)
    return result


def annual_report(spec,result):
    report=_annual_report(spec,result)
    for name,arm in result['arms'].items():
        for r in arm['metrics']:
            for k in ('mae','naive_mae','relative_error','n'):
                report['metrics'].append(dict(metric=name+'_'+k,value=r[k],unit='ratio' if k=='relative_error' else 'count' if k=='n' else 'USD_per_EUR',
                    horizon=r['horizon'],split='validation'))
    return report


def tick(spec):
    result=_tick(spec)
    if result and result.get('state') not in ('CLOSED','CLOSED_WITH_FAILURE'):
        planned=W.read(Path(spec['data_root'])/'PREPARED.json')['planned_actual_fits']
        completed=result['complete'];remaining=max(0,planned-completed-len(result.get('failed',[])))
        eta=result['eta'];denom=max(1,result['total']-result['terminal'])
        eta.update(seconds=eta['seconds']*remaining/denom,planned_actual_fits=planned,remaining_actual_fits=remaining)
        W.B.put(Path(spec['root'])/'STATUS.json',result)
    return result


def configure():
    W.NAME=NAME;W.BASE=BUNDLE_BASE;W.HEADS=HEADS;W.SHORT=HEADS
    W.GPU_ROLE='secondary';W.GPU_HINT='4090';W.MODULE='tools.i6dmo.weekly_window_core'
    W.PHASE='I6DMO_WEEKLY_WINDOW_CORE';W.TITLE='# Weekly Paired Window And Core Sweep';W.GPU_WALL=7200
    W.prepare=prepare;W.source_identity=source_identity;W.initialization_proof=initialization_proof
    W.preflight=preflight;W.run=run;W.annual=annual;W.annual_report=annual_report;W.tick=tick


def main():
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','launch','run','tick','status'))
    parser.add_argument('--spec',required=True)
    args=parser.parse_args();spec=W.read(args.spec)
    try:
        if args.action=='status':
            path=Path(spec['root'])/'STATUS.json';result=W.read(path) if path.exists() else dict(state='NOT_STARTED')
            result.pop('cells',None)
        else:result=getattr(W,args.action)(spec)
        if result:print(json.dumps({k:v for k,v in result.items() if k not in ('files','cells')},sort_keys=True))
    except BlockingIOError:pass
    except Exception as exc:
        if args.action in ('tick','launch'):W.B.put(Path(spec['root'])/'STATUS.json',dict(state='WAIT_OPERATIONAL_RETRY',error=type(exc).__name__+':'+str(exc)[:600],at=time.time()))
        raise


if __name__=='__main__':main()
