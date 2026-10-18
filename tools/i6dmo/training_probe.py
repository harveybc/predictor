"""Small matched stopping-budget probes, scheduled after the parent GPU campaign."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import numpy as np

from tools.i6dmo import monitor_campaign as C, monitor_contrast as MC, weekly18 as W
from tools.i6dmo import prequential_test as S, continuation_data as CD, continuation as Metrics

NAME = 'i6dmo-stopping-probe-20261018'


def budget(old):
    return dict(old, max_epochs=2000, patience=50)


def select_cases(rows):
    rows=sorted(rows,key=lambda r:(r['unit']['period']['train_end'],r['key']))
    result=[]
    for arm in ('legacy4','relative1'):
        first=next((r for r in rows if r['arm']==arm and r['epochs']==11),None)
        if first:result.append((arm+'_first11',first['key']))
    legacy=[r for r in rows if r['arm']=='legacy4']
    if legacy:result.append(('legacy4_longest',max(legacy,key=lambda r:r['epochs'])['key']))
    return result


def prefix(old,new):
    fields=('epoch','fit_loss','inner_loss','monitored','updates','weights_sha256')
    errors=[]
    for i,a in enumerate(old):
        if i>=len(new):
            errors.append(dict(epoch=i+1,field='missing_epoch'));continue
        for key in fields:
            if a[key]!=new[i][key]:errors.append(dict(epoch=i+1,field=key,old=a[key],new=new[i][key]))
    return dict(exact=not errors,compared_epochs=len(old),differences=errors[:20])


def prepare(parent_spec,output):
    spec=W.read(parent_spec); packet=Path(spec['packet']); target=Path(output)
    doc=C.checked_doc(packet/'PREPARED.json'); identity=C.identity(doc,spec['code'])
    if (target/'PREPARED.json').exists():
        result=C.checked_doc(target/'PREPARED.json')
        if result['source_identity']!=identity:raise ValueError('SOURCE_IDENTITY_CHANGED')
        W.authenticate(target,result['files']);return result
    files={};units=[]
    for family in ('short','long'):
        source=Path(spec['artifacts'])/family
        family_units=[u for u in doc['units'] if u['family']==family]
        by_id={u['id']:i for i,u in enumerate(family_units)}
        rows=[]
        donors={d['id']:(d,packet) for d in doc['initial'].values()}
        for arm in MC.ARMS:
            for u in family_units:
                r=C.terminal(source,arm+'__'+u['id'],identity)
                if r is not None and r['state']=='COMPLETE':
                    donors[r['next_checkpoint']['id']]=(r['next_checkpoint'],source)
                    if r['arm'] in ('legacy4','relative1') and by_id[r['unit']['id']]<len(family_units)-1:
                        rows.append(r)
        lookup={r['key']:r for r in rows}
        for label,key in select_cases(rows):
            r=lookup[key];u=r['unit'];cid=family+'__'+label;folder=target/cid;folder.mkdir(parents=True,exist_ok=True)
            donor,base=donors[r['inference_donor']];W.authenticate(base,donor['files'])
            shutil.copy2(base/donor['archive'],folder/'DONOR.keras')
            new=dict(donor,root='data',archive=cid+'/DONOR.keras',files={cid+'/DONOR.keras':W.file_sha256(folder/'DONOR.keras')})
            new.pop('id',None);new['id']=W.B.digest(new)
            record=W.read(packet/u['id']/'PERIOD.json');record['cell']=dict(record['cell'],id=cid)
            W.B.put(folder/'PERIOD.json',record)
            for name in ('FIT.npz','OBSERVED.npz','PREDICT.npz'):
                shutil.copy2(packet/u['id']/name,folder/name)
            shutil.copy2(source/'results'/key/'update/TRAINING.json',folder/'SOURCE_TRAINING.json')
            following=family_units[by_id[u['id']]+1]
            ref=C.terminal(source,r['arm']+'__'+following['id'],identity)
            if ref is None or ref['inference_donor']!=r['next_checkpoint']['id']:
                raise ValueError('NEXT_WEEK_REFERENCE_DONOR')
            shutil.copy2(packet/following['id']/'PREDICT.npz',folder/'NEXT_INPUT.npz')
            shutil.copy2(packet/following['id']/'OBSERVED.npz',folder/'NEXT_OBSERVED.npz')
            shutil.copy2(source/'results'/ref['key']/'VALIDATION.npz',folder/'REFERENCE_NEXT.npz')
            for p in folder.iterdir():files[str(p.relative_to(target))]=W.file_sha256(p)
            units.append(dict(id=cid,family=family,arm=r['arm'],donor=new,source_key=key,
                source_result_sha256=r['sha256'],next_period=following['period']))
    return C.seal(target/'PREPARED.json',dict(schema='i6dmo.stopping_probe.v1',units=units,files=files,
        arms=['2000_epochs_patience50'],training=budget(doc['training']),source_identity=identity,
        qualification='SIX_CASE_STOPPING_BUDGET_DIAGNOSTIC_NOT_ANNUAL',test_read=False))


def terminal(root,unit,digest):
    path=Path(root)/'results'/unit['id']/'RESULT.json'
    if not path.exists():return None
    r=C.checked_doc(path)
    if r['identity']!=digest:raise ValueError('PROBE_IDENTITY_CHANGED')
    W.authenticate(path.parent,r['files']);return r


def worker(spec,family):
    root=Path(spec['output']);packet=Path(spec['packet']);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        doc=C.checked_doc(packet/'PREPARED.json');digest=C.identity(doc,spec['code'])
        units=[u for u in doc['units'] if u['family']==family]
        done={u['id']:r for u in units if (r:=terminal(root,u,digest))}
        state=dict(complete=len(done),total=len(units),failed=[k for k,r in done.items() if r['state']=='FAILED'],test_read=False)
        if len(done)==len(units):
            state['state']='COMPLETE' if not state['failed'] else 'COMPLETED_WITH_FAILURES'
            W.B.put(root/'STATUS.json',state);return state
        parent=W.read(Path(spec['parent_output'])/'STATUS.json')
        if parent.get('complete')!=parent.get('total') or parent.get('total')!=300:
            state.update(state='WAIT_PARENT',parent_complete=parent.get('complete'),eta_seconds=None)
            W.B.put(root/'STATUS.json',state);return state
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        mem=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
        if apps.strip() or mem<7*1024**3:
            state.update(state='WAIT_RESOURCES',eta_seconds=None);W.B.put(root/'STATUS.json',state);return state
        unit=next(u for u in units if u['id'] not in done);cid=unit['id'];folder=root/'results'/cid
        W.authenticate(W.REPO,spec['code']);W.authenticate(packet,{k:v for k,v in doc['files'].items() if k.startswith(cid+'/')})
        state.update(state='RUNNING',current=cid);W.B.put(root/'STATUS.json',state)
        folder.mkdir(parents=True,exist_ok=True)
        ledger=folder/'ATTEMPTS.json';attempts=W.read(ledger) if ledger.exists() else []
        try:
            if len(attempts)>=3 and not (folder/'update/UPDATE.json').exists():
                raise RuntimeError('PROCESS_RETRY_EXHAUSTED')
            attempts.append(dict(number=len(attempts)+1,at=time.time()))
            W.B.put(ledger,attempts)
            record=W.read(packet/cid/'PERIOD.json');reader=lambda name:CD._read(packet/cid/(name+'.npz'))
            backend=C.Backend(packet,root,unit['arm'])
            with patch.object(S,'TRAINING',doc['training']):
                r=S.execute_week(record,unit['donor'],folder,reader,backend,input_sha256=doc['files'][cid+'/PREDICT.npz'],
                    evaluation_split='VALIDATION',qualification=doc['qualification'])
                old=W.read(packet/cid/'SOURCE_TRAINING.json');new=W.read(folder/'update/TRAINING.json')
                proof=prefix(old['history'],new['history']);W.B.put(folder/'PREFIX.json',proof)
                x=CD._read(packet/cid/'NEXT_INPUT.npz');forecast=folder/'NEXT_PREDICTED.npz'
                if (folder/'NEXT_SEAL.json').exists():
                    seal=W.read(folder/'NEXT_SEAL.json')
                    if seal['donor']!=r['next_checkpoint']['id'] or seal['sha256']!=W.file_sha256(forecast):
                        raise ValueError('NEXT_FORECAST_CHANGED')
                    pred=CD._read(forecast)['prediction']
                else:
                    S.available(r['next_checkpoint'],x['origin_ts']);pred=backend.predict(r['next_checkpoint'],x)
                    temp=folder/'NEXT.partial.npz';np.savez_compressed(temp,prediction=pred)
                    with temp.open('rb') as stream:__import__('os').fsync(stream.fileno())
                    temp.replace(forecast)
                    W.B.put(folder/'NEXT_SEAL.json',dict(donor=r['next_checkpoint']['id'],sha256=W.file_sha256(forecast),at=time.time()))
                observed=CD._read(packet/cid/'NEXT_OBSERVED.npz');ref=CD._read(packet/cid/'REFERENCE_NEXT.npz')
                for k in ('origin_ts','origin_price','target_price','mask'):
                    if not np.array_equal(observed[k],ref[k],equal_nan=True):raise ValueError('NEXT_POPULATION_CHANGED')
                for k in S.PREDICT_KEYS:
                    if not np.array_equal(x[k],observed[k]):raise ValueError('NEXT_INPUT_CHANGED')
                heads=record['cell']['horizons']
                r['metrics']['NEXT_VALIDATION']=Metrics.measured_metrics(observed,pred,heads)
                r['metrics']['REFERENCE_NEXT_VALIDATION']=Metrics.measured_metrics(observed,ref['prediction'],heads)
                r.update(prefix=proof,best_epoch_after_source_stop=new['best_epoch']>old['epochs_completed'],state='COMPLETE')
        except Exception as exc:
            r=dict(state='FAILED',reason=type(exc).__name__+':'+str(exc)[:600],test_read=False)
        r.update(identity=digest,key=cid,arm=unit['arm'],family=family,source_key=unit['source_key'],
            files={str(p.relative_to(folder)):W.file_sha256(p) for p in folder.rglob('*') if p.is_file() and p.name!='RESULT.json'})
        C.seal(folder/'RESULT.json',r)
        state.update(state='UNIT_TERMINAL',last_state=r['state'],complete=len(done)+1)
        W.B.put(root/'STATUS.json',state);return state


def collect(spec):
    root=Path(spec['root']);doc=C.checked_doc(Path(spec['packet'])/'PREPARED.json');digest=C.identity(doc,spec['code'])
    with (root/'collect.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states={}
        for family,role in spec['roles'].items():
            local=Path(spec['artifacts'])/family;local.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--checksum','--exclude=*.lock',role['host']+':'+role['output']+'/',str(local)+'/'],check=True,timeout=120)
            units=[u for u in doc['units'] if u['family']==family]
            results={u['id']:r for u in units if (r:=terminal(local,u,digest))}
            delivered=None
            for key,r in results.items():
                qualification=doc['qualification']+('_EXACT_PREFIX' if r.get('prefix',{}).get('exact') else '_PREFIX_NOT_EXACT')
                with patch.object(C,'NAME',NAME),patch.object(C,'QUALIFICATION',qualification):
                    delivered=C.publish(dict(doc,code_commit=spec['code_commit']),local,dict(identity=digest),
                        {key:r},spec['outbox'],spec['warehouse']['url'],spec['warehouse']['token_env'])
            states[family]=dict(terminal=len(results),total=len(units),failed=[k for k,r in results.items() if r['state']=='FAILED'],publication=delivered)
            W.B.put(local/'REPORT.json',dict(states[family],results=results))
            if len(results)==len(units) and (not delivered or delivered['pending']==delivered['mismatch']==0):
                subprocess.run(['ssh',role['host'],'systemctl','--user','disable','--now',NAME+'-'+family+'.timer'],check=True)
        closed=all(s['terminal']==s['total'] and (not s['publication'] or s['publication']['pending']==s['publication']['mismatch']==0) for s in states.values())
        result=dict(state='CLOSED' if closed else 'RUNNING',families=states,test_read=False,at=time.time())
        W.B.put(root/'STATUS.json',result)
        if closed:subprocess.run(['systemctl','--user','disable','--now',NAME+'-collect.timer'],check=True)
        return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('prepare','worker','collect','deploy','status'))
    p.add_argument('--spec');p.add_argument('--output');p.add_argument('--family');a=p.parse_args()
    try:
        if a.action=='prepare':
            doc=prepare(a.spec,a.output);r=dict(state='PREPARED',cases=len(doc['units']),sha256=doc['sha256'])
        elif a.action=='deploy':
            from tools.i6dmo import monitor_ops as O
            with patch.object(C,'NAME',NAME):r=O.deploy(W.read(a.spec))
        else:
            spec=W.read(a.spec)
            if a.action=='worker':r=worker(spec,a.family)
            elif a.action=='collect':r=collect(spec)
            else:r=W.read(Path(spec['root'])/'STATUS.json')
        print(json.dumps(r,sort_keys=True))
    except BlockingIOError:print(json.dumps(dict(state='BUSY')))


if __name__=='__main__':main()
