"""Finite weekly TEST replication; one bounded GPU fit per timer invocation."""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
import shutil
from unittest.mock import patch

import numpy as np

from tools.i6dmo import weekly18 as W, weekly_test_data as D

NAME='i6dmo-weekly-test-20261017'
MODULE='tools.i6dmo.weekly_test'
PHASE='I6DMO_WEEKLY_TEST_2025'


def storage_guard(path):
    from tools.i6dmo.storage import probe_mount,_check_target
    target=Path(path)
    while not target.exists():target=target.parent
    return _check_target(target,probe_mount(target),2*1024**3,5*1024**3)


def verify(spec):
    if spec['training']!=W.B.TRAINING or spec['seed']!=W.B.SEED or spec['evaluation_split']!='TEST':
        raise ValueError('REPLICATION_RECIPE_CHANGED')
    W.authenticate(W.REPO,spec['code_files'])
    if spec['qualification']!=D.QUALIFICATION:raise ValueError('TEST_QUALIFICATION_CHANGED')


def prepared(spec,root):
    doc=W.read(Path(root)/'PREPARED.json')
    if doc['sha256']!=spec['population_digest'] or doc['sha256']!=W.B.digest({k:v for k,v in doc.items() if k!='sha256'}):
        raise ValueError('PREPARED_CHANGED')
    if doc['cells']!=spec['cells']:raise ValueError('CELLS_CHANGED')
    return doc


def run(spec):
    verify(spec);root=Path(spec['gpu_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={c['id']:r for c in spec['cells'] if (r:=W.terminal(root,c,spec))}
        cell=next((c for c in spec['cells'] if c['id'] not in results),None)
        if not cell:W.B.progress(root,state='GPU_DONE');return
        doc=prepared(spec,spec['gpu_data']);cid=cell['id']
        W.authenticate(spec['gpu_data'],{n:h for n,h in doc['files'].items() if n.startswith(cid+'/')})
        record=W.read(Path(spec['gpu_data'])/cid/'PERIOD.json')
        folder=root/'results'/cid;folder.mkdir(parents=True,exist_ok=True)
        if record['parent_state']=='SKIPPED_NO_EVAL_ROWS':
            result=dict(batch_id=NAME,cell=cell,state='SKIPPED_NO_EVAL_ROWS',files={},test_read=True)
        else:
            apps=W.run_cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
            available=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines()
                if v.startswith('MemAvailable:')))*1024
            gpu=W.run_cmd(['nvidia-smi','-i',spec['gpu_uuid'],'--query-gpu=name','--format=csv,noheader'])
            if apps.strip() or available<7*1024**3 or '5090' not in gpu:
                W.B.progress(root,state='WAIT_RESOURCES',cell=cid,memory_available=available);return
            if (folder/'STARTED.json').exists():
                result=dict(batch_id=NAME,cell=cell,state='FAILED_INTERRUPTED',files={},test_read=True)
            else:
                W.B.put(folder/'STARTED.json',dict(at=time.time(),cell=cell))
                try:
                    from tools.i6dmo import continuation as C,continuation_data as CD,continuation_models as M
                    from tools.i6dmo import weekly_long as L
                    from predictor_plugins.modular_temporal import weights_hash
                    arrays={s:CD._read(Path(spec['gpu_data'])/cid/(s+'.npz')) for s in ('FIT','TEST')}
                    for split,a in arrays.items():
                        if C.measured_metrics(a,np.where(a['mask'],a['target_price'],0),cell['horizons'])!=record['naive_cache'][split]:
                            raise ValueError('NAIVE_CACHE_CHANGED')
                    meta=dict(record['population'],manifest_sha256=spec['population_digest'])
                    W.B.put(folder/'SCALES.json',record['scales'])
                    def loader(*args,**kwargs):
                        if args!=(spec['gpu_data'],'short',2,13) or kwargs!=dict(
                                horizons=cell['horizons'],input_dir=spec['gpu_input'],period=cell['period']):
                            raise ValueError('UNEXPECTED_DATA_ACCESS')
                        return dict(FIT=arrays['FIT'],VALIDATION=arrays['TEST']),record['scales'],meta
                    original=M.build
                    def builder(config):
                        bundle=L.project_model(config,original) if cell['family']=='long' else original(config)
                        if weights_hash(bundle.forecast_model)!=record['reference_initial_weights_sha256']:
                            raise ValueError('WINNER_INITIAL_TENSORS_CHANGED')
                        return bundle
                    with patch.object(CD,'load',loader),patch.object(C,'resolve',lambda *a:record['resolved']),patch.object(M,'build',builder):
                        result=C.fit_cell(spec,cell,{},folder)
                    if W.read(folder/'CONFIG.json')['model']!=record['model']:raise ValueError('WINNER_MODEL_CHANGED')
                    result.update(qualification=D.QUALIFICATION,reference_profile=record['reference_profile'])
                except Exception as exc:
                    result=dict(batch_id=NAME,cell=cell,state='FAILED',files={},test_read=True,
                        reason=type(exc).__name__+':'+str(exc)[:700])
        result['sha256']=W.B.digest(result);W.B.put(folder/'RESULT.json',result)
        W.B.progress(root,state=result['state'],cell=cid)


def annual(spec,artifacts,results):
    output={}
    for family,heads in (('short',D.SHORT),('long',D.LONG)):
        cells=[c for c in spec['cells'] if c['family']==family]
        terminal=[results[c['id']] for c in cells if c['id'] in results]
        if len(terminal)!=len(cells):output[family]=dict(state='INCOMPLETE');continue
        if any(r['state'] not in ('COMPLETE','SKIPPED_NO_EVAL_ROWS') for r in terminal):
            output[family]=dict(state='FAILED');continue
        rows=[];seen={h:set() for h in heads}
        for c,r in zip(cells,terminal,strict=True):
            if r['state']!='COMPLETE':continue
            with np.load(Path(artifacts)/'results'/c['id']/'TEST.npz',allow_pickle=False) as z:
                for j,h in enumerate(heads):
                    stamps=z['origin_ts'][z['mask'][:,j]].tolist()
                    if seen[h].intersection(stamps) or len(set(stamps))!=len(stamps):raise ValueError('WEEK_OVERLAP')
                    seen[h].update(stamps)
            rows.extend(r['metrics']['TEST'])
        metrics=[W.pool_metrics(rows,h) for h in heads]
        if any(m['n']!=len(seen[m['horizon']]) for m in metrics):raise ValueError('ANNUAL_COUNT_MISMATCH')
        output[family]=dict(state='COMPLETE',metrics=metrics,all_heads_beat_naive=all(
            m['relative_error'] is not None and m['relative_error']<0 for m in metrics))
    return dict(families=output,test_read=True,qualification=D.QUALIFICATION)


def enrich_status(state,spec,artifacts,results):
    """Status seam; the prequential campaign patches a typed recovery view in (prequential_test.context)."""
    return state


def closure_coverage(spec,results):
    from tools.i6dmo import prequential_closure as PC
    return PC.coverage(spec,results,dict(short=D.SHORT,long=D.LONG))


def report(spec,result):
    report=W.B.report(spec,result)
    report['phase']=PHASE;report['tags'].update(evaluation=D.QUALIFICATION,test_read='true',family=result['cell']['family'])
    if result['state']!='COMPLETE':
        for row in report['metrics']:row['split']='test'
    return report


def tick(spec):
    from tools.i6dmo.outbox import Outbox,HttpWarehouse
    verify(spec);root=Path(spec['root']);artifacts=Path(spec['artifacts']);artifacts.mkdir(parents=True,exist_ok=True)
    with (root/'tick.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'CLOSE.json').exists():return W.read(root/'CLOSE.json')
        storage_guard(artifacts)
        W.run_cmd(['rsync','-a','--checksum','--exclude=*.lock',spec['ssh_alias']+':'+spec['gpu_root']+'/',str(artifacts)+'/'])
        results={c['id']:r for c in spec['cells'] if (r:=W.terminal(artifacts,c,spec))}
        done=[r for r in results.values() if r['state']=='COMPLETE']
        progress=W.read(artifacts/'PROGRESS.json') if (artifacts/'PROGRESS.json').exists() else {}
        remaining=len(spec['cells'])-len(results)
        elapsed=[r['training_seconds'] for r in done]
        state=dict(state='RUNNING' if remaining else 'COMPUTE_COMPLETE',at=time.time(),batch_id=NAME,
            total=len(spec['cells']),terminal=len(results),complete=len(done),
            failed={k:r.get('reason',r['state']) for k,r in results.items() if r['state'].startswith('FAILED')},
            progress=progress,test_read=True,qualification=D.QUALIFICATION,
            families={f:dict(complete=sum(r['cell']['family']==f for r in done),
                total=sum(c['family']==f for c in spec['cells'])) for f in ('short','long')},
            cells={k:dict(state=r['state'],test=r.get('metrics',{}).get('TEST')) for k,r in results.items()},
            eta=dict(kind='MEASURED_FIT_MEAN' if elapsed else 'NOT_MEASURED',
                seconds=remaining*(math.fsum(elapsed)/len(elapsed)+30) if elapsed else None,
                note='Includes 30s timer/start overhead per cell; excludes resource waits/publication'))
        state=enrich_status(state,spec,artifacts,results)
        cov=closure_coverage(spec,results);W.B.put(artifacts/'COVERAGE.json',cov)
        state['coverage']={f:{k:v[k] for k in ('declared_weeks','scored_weeks','no_pair_weeks','failed_weeks','pending_weeks')}
            for f,v in cov['families'].items()}
        W.B.put(root/'STATUS.json',state)
        box=Outbox(root/'outbox.sqlite3')
        try:
            for r in results.values():box.enqueue(report(spec,r),source_unit=r['cell']['id'])
            if not remaining:
                pooled=dict(annual(spec,artifacts,results),coverage=cov);W.B.put(artifacts/'ANNUAL_TEST.json',pooled)
                metrics=[dict(metric=f+'_'+key,value=m[key],horizon=m['horizon'],split='test',
                    unit='ratio' if key=='relative_error' else 'count' if key=='n' else 'USD_per_EUR')
                    for f,v in pooled['families'].items() for m in v.get('metrics',[])
                    for key in ('mae','naive_mae','relative_error','n') if m[key] is not None]
                box.enqueue(dict(experiment_key='i6dmo:'+NAME+':annual',experiment_set_key='i6dmo:'+NAME,
                    actor='predictor',lake='LOCAL_RETAINED',lineage='UNVERIFIED',config_sha256=W.B.digest(pooled),
                    code_commit=spec['code_commit'],project='predictor',phase=PHASE,
                    tags=dict(evaluation=D.QUALIFICATION,test_read='true'),metrics=metrics),source_unit='annual')
            pub=box.pump(HttpWarehouse(spec['warehouse']['url'],spec['warehouse']['token_env']))
            pub.pop('delivery_seconds',None);state['publication']=pub
            if not remaining and not pub['pending'] and not pub['mismatch']:
                W.B.put(artifacts/'PUBLICATION.json',pub)
                W.B.put(artifacts/'READBACK.json',[dict(report_sha256=x['report_sha256'],state=x['state'],
                    readback_sha256=x['readback_sha256']) for x in box.items()])
                W.B.put(artifacts/'PUBLIC_CONFIG.json',{k:spec[k] for k in ('cells','code_commit','code_files',
                    'training','seed','population_digest','qualification','evaluation_split')})
                lines=['# Weekly TEST 2025','',D.QUALIFICATION,'',
                    '| Family | Horizon | MAE | Naive for horizon | Relative error | N |',
                    '|---|---:|---:|---:|---:|---:|']
                from tools.i6dmo import prequential_closure as PC
                for family,v in pooled['families'].items():
                    for m in v.get('metrics',[]):lines.append(PC.metric_row(family,m))
                (artifacts/'RESULTS.md').write_text('\n'.join(lines)+'\n\n'+PC.markdown(cov))
                files={str(p.relative_to(artifacts)):W.file_sha256(p) for p in artifacts.rglob('*') if p.is_file()
                    and p.name not in ('RUN.private.json','STARTED.json','PROGRESS.json','PREFLIGHT.json')
                    and 'backup' not in p.parts and not p.name.startswith('.')}
                state['mirrors']=W.mirror(spec,artifacts,files)
                if state['mirrors']['state']=='VERIFIED':
                    state['state']='CLOSED_WITH_FAILURE' if state['failed'] else 'CLOSED'
                    W.B.put(root/'CLOSE.json',state)
                    W.run_cmd(['ssh',spec['ssh_alias'],'systemctl','--user','disable','--now',NAME+'-gpu.timer'])
                    W.run_cmd(['systemctl','--user','disable','--now',NAME+'-tick.timer'])
            W.B.put(root/'STATUS.json',state)
            return state
        finally:box.close()


def launch(spec):
    if W.run_cmd(['git','-C',W.REPO,'status','--porcelain']).strip():raise ValueError('SOURCE_NOT_FROZEN')
    storage_guard(spec['data_root'])
    p=D.prepare(spec);root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    deploy=W.read(spec['source_deployment']);role=deploy['roles']['preferred']
    remote=lambda args:W.run_cmd(['ssh','-o','BatchMode=yes',role['ssh_alias'],shlex.join([str(v) for v in args])])
    code=W.run_cmd(['git','-C',W.REPO,'rev-parse','HEAD']).strip()
    runtime='/var/tmp/i6dmo-weekly-test-runtime-'+code[:8]
    remote(['mkdir','-p',str(root),runtime,spec['gpu_data'],spec['gpu_root']])
    for directory in ('tools','predictor_plugins','app','olap/store/src'):
        remote(['mkdir','-p',runtime+'/'+directory])
        W.run_cmd(['rsync','-a','--exclude=__pycache__',str(W.REPO/directory)+'/',role['ssh_alias']+':'+runtime+'/'+directory+'/'])
    W.run_cmd(['rsync','-a','--checksum',spec['data_root']+'/',role['ssh_alias']+':'+spec['gpu_data']+'/'])
    evidence=Path(spec['artifacts'])/'source';evidence.mkdir(parents=True,exist_ok=True)
    for name in ('SOURCE.json','SOURCE.npz','CALENDAR.json','PREFIX_COMPARISON.json'):
        shutil.copyfile(Path(spec['source'])/name,evidence/name)
    code_files={str(p.relative_to(W.REPO)):W.file_sha256(p) for d in
        ('tools','predictor_plugins/modular_temporal','olap/store/src') for p in (W.REPO/d).rglob('*.py')}
    gpu=next(g for g in role['gpus'] if g.get('heavy') and g['class']=='rtx5090')
    spec=dict(spec,batch_id=NAME,code_commit=code,code_files=code_files,runtime=runtime,
        cells=p['cells'],population_digest=p['sha256'],gpu_uuid=gpu['uuid'],ssh_alias=role['ssh_alias'],
        evaluation_split='TEST',qualification=D.QUALIFICATION,training=W.B.TRAINING,seed=W.B.SEED)
    path=root/'RUN.private.json'
    if path.exists() and W.read(path)!=spec:
        old=W.read(path)
        running=subprocess.run(['ssh',role['ssh_alias'],'systemctl','--user','is-enabled',NAME+'-gpu.timer'],
            capture_output=True,text=True).returncode==0
        if (root/'LAUNCH.json').exists() or running or old['population_digest']!=spec['population_digest']:
            raise ValueError('DEPLOYMENT_IDENTITY_CHANGED')
        path.rename(root/('UNDEPLOYED_'+old['code_commit'][:8]+'.private.json'))
    W.B.put(path,spec);W.run_cmd(['rsync','-a',str(path),role['ssh_alias']+':'+str(path)])
    for target,action in (('tick','tick'),('gpu','run')):
        wd=str(W.REPO) if target=='tick' else runtime
        python=deploy['roles']['coordinator']['python'] if target=='tick' else role['python']
        args=[deploy['launcher']['crispdm_run'],'-m','768M' if target=='tick' else '4352M',
            '-t','600s' if target=='tick' else '3600s','-n',NAME+'-'+target,'--',python,'-B','-m',MODULE,action,'--spec',str(path)]
        environment=('EnvironmentFile='+spec['warehouse']['environment_file']+'\n') if target=='tick' else ''.join(
            'Environment='+json.dumps(k+'='+v)+'\n' for k,v in role.get('environment',{}).items())
        body='[Unit]\nDescription=Owner-authorized weekly TEST replication\nX-I6DMO-Batch='+NAME+'\n[Service]\nType=oneshot\nWorkingDirectory='+wd+'\n'+environment
        body+='Environment=PYTHONPATH='+wd+'\nEnvironment=CUDA_VISIBLE_DEVICES='+('' if target=='tick' else gpu['uuid'])+'\n'
        body+='Environment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n'
        body+='ExecStart='+shlex.join(args)+'\nMemoryMax='+('1G' if target=='tick' else '6G')+'\nMemorySwapMax=0\nTimeoutStartSec=3700\nNice=10\n'
        timer='[Unit]\nDescription=Autonomous weekly TEST sequence\n[Timer]\nOnBootSec=10s\nOnUnitInactiveSec=15s\nAccuracySec=1s\nPersistent=true\n[Install]\nWantedBy=timers.target\n'
        script="from pathlib import Path; d=Path.home()/'.config/systemd/user'; d.mkdir(parents=True,exist_ok=True); "
        for suffix,text in (('service',body),('timer',timer)):
            script+='p=d/'+repr(NAME+'-'+target+'.'+suffix)+'; assert not p.exists() or '+repr(NAME)+' in p.read_text(); p.write_text('+repr(text)+'); '
        call=remote if target=='gpu' else W.run_cmd
        call(['python3','-c',script]);call(['systemctl','--user','daemon-reload'])
        call(['systemctl','--user','enable','--now',NAME+'-'+target+'.timer'])
    state=dict(state='ARMED',total=len(p['cells']),code_commit=code,test_read=True,qualification=D.QUALIFICATION)
    W.B.put(root/'LAUNCH.json',state);return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','run','tick','status'))
    parser.add_argument('--spec',required=True);parser.add_argument('--cell')
    args=parser.parse_args();spec=W.read(args.spec)
    try:
        if args.action=='status':
            p=Path(spec['root'])/'STATUS.json';result=W.read(p) if p.exists() else dict(state='NOT_STARTED')
            if args.cell:result=result.get('cells',{}).get(args.cell,dict(state='PENDING'))
            else:result.pop('cells',None)
        elif args.action=='prepare':
            p=D.prepare(spec);result=dict(sha256=p['sha256'],cells=len(p['cells']))
        else:result=globals()[args.action](spec)
        if result:print(json.dumps(result,sort_keys=True))
    except BlockingIOError:return
    except Exception as exc:
        if args.action=='tick':W.B.put(Path(spec['root'])/'STATUS.json',dict(state='WAIT_OPERATIONAL_RETRY',
            error=type(exc).__name__+':'+str(exc)[:500],at=time.time()))
        raise


if __name__=='__main__':main()
