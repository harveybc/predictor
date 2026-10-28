"""Fit-free retention of all-head eligible annual recipes and business CSVs."""
import argparse
import bisect
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from tools.i6dmo import best_registry as B
from tools.i6dmo.storage import _copy_verified
from tools.i6n_contracts import eligibility as E

SCHEMA='i6dmo.separate_pair_release.v1'


def forecast_rows(data,horizons,family,checkpoint,weeks):
    """Calendar/input availability only: masks and target prices are NOT consulted."""
    wanted=B.FAMILIES[family];columns=[horizons.index(h) for h in wanted]
    opens=[w.market_open for w in weeks];rows=[]
    origins=data['origin_ts'];pred=data['prediction'];prices=data['origin_price']
    if pred.shape!=(len(origins),len(horizons)) or len(prices)!=len(origins):
        raise ValueError('PREDICTION_SHAPE')
    for i,ts in enumerate(origins):
        t=datetime.fromtimestamp(int(ts),tz=timezone.utc)
        if t.year!=2024:raise ValueError('DEVELOPMENT_ORIGIN_REQUIRED')
        k=bisect.bisect_right(opens,t)-1
        if k<0 or t>=weeks[k].market_close:raise ValueError('ORIGIN_OUTSIDE_CALENDAR')
        ds=E.evaluate_heads(t,wanted,E.Role.MANAGEMENT,weeks[k])
        if not math.isfinite(float(prices[i])) or prices[i]<=0:raise ValueError('ORIGIN_PRICE')
        row=dict(DATE_TIME=t.isoformat(),origin_price=float(prices[i]),model_id=checkpoint)
        for h,j,d in zip(wanted,columns,ds):
            value=float(pred[i,j]);available=d.operational_eligible and math.isfinite(value) and value>0
            row.update({f'prediction_h{h}':value if available else '',f'available_h{h}':int(available)})
        rows.append(row)
    return rows


def test_gate(spec):
    # A chat authorization never substitutes for as-of evidence or a sealed
    # evaluation procedure. This module has no code path that opens TEST.
    return dict(state='BLOCKED',requested_by_owner=True,test_read=False,missing=[
        'POINT_IN_TIME_SOURCE_RECEIPT_FOR_ALL_SELECTED_INPUTS',
        'FROZEN_PROSPECTIVE_TEST_RETRAINING_RECIPE_WITHOUT_TEST_CHECKPOINT_SELECTION'])


def prepare(spec):
    registry=Path(spec['registry']);root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'release.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        board=B.read(registry/'LEADERBOARD.json');sources={s['label']:s for s in B.read(spec['registry_spec'])['sources']}
        selected={};profiles={}
        for family in B.FAMILIES:
            winner=board.get('eligible_families',{}).get(family,{}).get('winner')
            if not winner:raise ValueError('NO_ALL_OUTPUT_ELIGIBLE_FAMILY:'+family)
            p=B.read(registry/'profiles'/(winner+'.json'));hs=p['model']['horizons_hours']
            native={r['horizon']:r for r in p['metrics']}
            if p['scope']!='ANNUAL' or not p.get('member_profiles') or any(
                    h not in native or not math.isfinite(native[h]['relative_error']) or native[h]['relative_error']>=0
                    for h in hs):raise ValueError('ALL_OUTPUT_NATIVE_GATE:'+family)
            selected[family]=p;profiles[family]=winner
        calendar=B.read(spec['calendar']);_,weeks=E.load_calendar(calendar)
        if (calendar['entry_delay_after_open_hours'],calendar['force_flat_before_close_hours'])!=(12,12):
            raise ValueError('BUSINESS_OFFSETS_CHANGED')
        identity=B.sha256_json(dict(profiles=profiles,calendar_sha256=B.file_sha256(spec['calendar']),schema=SCHEMA))
        target=Path(spec['artifacts'])/identity;target.mkdir(parents=True,exist_ok=True)
        families={};dispositions=[]
        B.put(root/'STATUS.json',dict(state='PREPARING',pair_id=identity,at=time.time(),test_read=False,fit_executed=False))
        for family,p in selected.items():
            rows=[];members=[];source=sources[p['source_label']]
            for member_id in p['member_profiles']:
                member=B.read(registry/'profiles'/(member_id+'.json'))
                original=Path(source['root'])/member['artifact_ref'];receipt=B.read(original)
                if B.file_sha256(original)!=member['result_file_sha256']:raise ValueError('MEMBER_RESULT_CHANGED')
                B.authenticated(original.parent,receipt['files'])
                folder=target/family/member_id;folder.mkdir(parents=True,exist_ok=True)
                files={}
                for name in ('CONFIG.json','TRAINING.json','SCALES.json','model.weights.h5','VALIDATION.npz'):
                    src=original.parent/name;digest=receipt['files'][name];dst=folder/name
                    if dst.exists():
                        if B.file_sha256(dst)!=digest:raise ValueError('RETAINED_COPY_CONFLICT')
                    else:_copy_verified(src,dst,digest)
                    files[name]=digest
                with np.load(folder/'VALIDATION.npz',allow_pickle=False) as z:
                    rows.extend(forecast_rows(z,member['model']['horizons_hours'],family,member_id,weeks))
                members.append(dict(id=member_id,folder=str(folder.relative_to(target)),files=files,
                    source_result_sha256=member['result_file_sha256'],source_code_commit=member['code_commit'],
                    weights_sha256=member['weights_sha256'],period=member['data'].get('period')))
            if source.get('certify_empty_periods'):
                plan=B.read(source['weekly_annual'])
                for cell in plan['cells']:
                    path=Path(source['root'])/'results'/cell['id']/'RESULT.json'
                    proof=B.certify_empty_period(plan,path)
                    if proof:dispositions.append(dict(cell=cell['id'],**proof))
            rows.sort(key=lambda r:r['DATE_TIME'])
            if len({r['DATE_TIME'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_CSV_ORIGIN')
            csv_path=target/(family+'.csv');temp=target/(family+'.tmp.csv')
            if not rows:raise ValueError('EMPTY_FAMILY_ROWS')
            with temp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                f.flush();os.fsync(f.fileno())
            os.replace(temp,csv_path)
            families[family]=dict(profile_id=p['id'],label=p['label'],model=p['model'],
                metrics=p['metrics'],members=members,csv=csv_path.name,csv_sha256=B.file_sha256(csv_path),
                rows=len(rows),all_actual_outputs_beat_naive=True)
        manifest=dict(schema=SCHEMA,id=identity,families=families,
            calendar=calendar,calendar_sha256=B.file_sha256(spec['calendar']),
            supplemental_empty_periods=dispositions,test_gate=test_gate(spec),test_read=False,
            qualification='DEVELOPMENT_2024_CHECKPOINT_TUNING_NOT_INDEPENDENT_OOS',fit_executed=False)
        B.put(target/'PAIR.json',manifest)
        state=dict(state='RETAINED_CSV_READY_KERAS_EXPORT_PENDING',pair_id=identity,
            artifacts=str(target),members={f:len(v['members']) for f,v in families.items()},
            test_gate=manifest['test_gate'],test_read=False,fit_executed=False,at=time.time())
        B.put(root/'CURRENT.json',state);B.put(root/'STATUS.json',state)
        return state


def export_archives(root):
    """CPU-only archive conversion; never optimizes or touches immutable source results."""
    from tools.i6dmo import continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    root=Path(root);manifest=B.read(root/'PAIR.json');count=0
    progress=root/'ARCHIVE_STATUS.json';started=time.time()
    for family in manifest['families'].values():
        for member in family['members']:
            folder=root/member['folder'];B.authenticated(folder,member['files'])
            proof=folder/'ARCHIVE.json';archive=folder/'forecast_model.keras'
            if proof.exists():
                old=B.read(proof)
                if old['weights_sha256']!=member['weights_sha256'] or \
                        B.file_sha256(archive)!=old['archive_sha256']:raise ValueError('ARCHIVE_CONFLICT')
            else:
                M.keras.backend.clear_session()
                config=B.read(folder/'CONFIG.json')['model'];bundle=M.build(config)
                model=bundle.forecast_model;model.load_weights(folder/'model.weights.h5')
                if weights_hash(model)!=member['weights_sha256']:raise ValueError('REBUILT_WEIGHTS_MISMATCH')
                temp=folder/'forecast_model.tmp.keras';model.save(temp)
                loaded=M.keras.models.load_model(temp,compile=False,safe_mode=True)
                if weights_hash(loaded)!=member['weights_sha256']:raise ValueError('ARCHIVE_WEIGHTS_MISMATCH')
                n=config['slots'];x=np.zeros((1,n,len(config['features'])),dtype='float32')
                off=(np.arange(n,dtype='float32')-(n-1))[None,:]*config['spacing_open_market_hours']
                inputs=[x,off,np.zeros_like(off)]
                a=model(inputs,training=False);b=loaded(inputs,training=False)
                for left,right in zip(a,b):np.testing.assert_array_equal(left.numpy(),right.numpy())
                os.replace(temp,archive)
                B.put(proof,dict(state='VERIFIED',archive_sha256=B.file_sha256(archive),
                    weights_sha256=member['weights_sha256'],source_files=member['files'],
                    source_code_commit=member['source_code_commit'],
                    tensorflow=M.tf.__version__,keras=M.keras.__version__,
                    tensor_parity='EXACT',cpu_synthetic_forward_parity='EXACT',fit_executed=False,test_read=False))
                del loaded,model,bundle
            count+=1
            B.put(progress,dict(state='RUNNING',completed=count,at=time.time(),started=started,
                total=sum(len(f['members']) for f in manifest['families'].values()),fit_executed=False,test_read=False))
    result=dict(state='COMPLETE',completed=count,at=time.time(),elapsed_s=time.time()-started,fit_executed=False,test_read=False)
    B.put(progress,result);return result


def follow(spec):
    """One bounded transition; timers own retries, never an agent log-polling loop."""
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'follow.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        board=B.read(Path(spec['registry'])/'LEADERBOARD.json')
        expected={f:board['eligible_families'][f]['winner'] for f in B.FAMILIES}
        current=B.read(root/'CURRENT.json') if (root/'CURRENT.json').exists() else None
        if current:
            manifest=B.read(Path(current['artifacts'])/'PAIR.json')
            if {f:v['profile_id'] for f,v in manifest['families'].items()}!=expected:current=None
        if current is None:current=prepare(spec)
        target=Path(current['artifacts']);manifest=B.read(target/'PAIR.json')
        remote=spec['archive_worker'];remote_root=remote['root'].rstrip('/')+'/'+manifest['id']
        def ssh(args,check=True):
            return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',remote['alias'],
                shlex.join([str(a) for a in args])],check=check,capture_output=True,text=True,timeout=120)
        def save(state,**extra):
            doc=dict(current,state=state,at=time.time(),**extra);B.put(root/'STATUS.json',doc);return doc
        result=ssh(['cat',remote_root+'/ARCHIVE_STATUS.json'],check=False)
        status=json.loads(result.stdout) if result.returncode==0 else None
        if status and status['state'].startswith('FAILED'):
            return save('ARCHIVE_FAILED_ORIGINAL_WEIGHTS_RETAINED',archive=status)
        if not status or status['state']!='COMPLETE':
            unit='i6dmo-keras-export-'+manifest['id'][:12]
            # Adopt the first operator-launched conversion rather than duplicate it.
            for name in (remote.get('initial_unit',''),unit):
                if name and ssh(['systemctl','--user','is-active',name+'.service'],check=False).returncode==0:
                    return save('WAIT_ARCHIVES',archive=status)
            ssh(['mkdir','-p',remote_root])
            subprocess.run(['rsync','-a',str(target)+'/',remote['alias']+':'+remote_root+'/'],check=True,timeout=120)
            command=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=4G',
                '--property=MemorySwapMax=0','--property=CPUQuota=100%','--property=TasksMax=64',
                '--property=Nice=10','--property=RuntimeMaxSec=3600',
                '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=PYTHONPATH='+remote['runtime'],
                '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1',
                '--setenv=TF_NUM_INTRAOP_THREADS=1','--setenv=TF_NUM_INTEROP_THREADS=1',
                '--setenv=TF_ENABLE_ONEDNN_OPTS=0','--setenv=PYTHONDONTWRITEBYTECODE=1',
                '--working-directory='+remote['runtime'],remote['python'],'-B','-m',
                'tools.i6dmo.pair_release','archive','--root',remote_root]
            ssh(command)
            return save('WAIT_ARCHIVES',archive=status)
        subprocess.run(['rsync','-a','--include=*/','--include=forecast_model.keras',
            '--include=ARCHIVE.json','--include=ARCHIVE_STATUS.json','--exclude=*',
            remote['alias']+':'+remote_root+'/',str(target)+'/'],check=True,timeout=120)
        count=0
        for family in manifest['families'].values():
            for member in family['members']:
                folder=target/member['folder'];B.authenticated(folder,member['files'])
                proof=B.read(folder/'ARCHIVE.json')
                if proof['weights_sha256']!=member['weights_sha256'] or \
                        proof['archive_sha256']!=B.file_sha256(folder/'forecast_model.keras'):
                    raise ValueError('ARCHIVE_FINAL_READBACK_MISMATCH')
                count+=1
        if status['completed']!=count:raise ValueError('ARCHIVE_MEMBER_COUNT')
        csv_proof=verify_csvs(target)
        from tools.i6dmo.outbox import Outbox,HttpWarehouse
        warehouse=B.read(spec['registry_spec'])['warehouse'];box=Outbox(root/'outbox.sqlite3')
        try:
            for family,p in manifest['families'].items():
                metrics=[]
                for row in p['metrics']:
                    for key,name,unit in (('mae','MAE','USD_per_EUR'),('naive_mae','Naive_MAE','USD_per_EUR'),
                        ('relative_error','Relative_Error_Vs_Naive','ratio'),('n','N','count')):
                        metrics.append(dict(metric=name,value=row[key],unit=unit,horizon=row['horizon'],split='validation'))
                report=dict(experiment_key='i6dmo:pair-retention:'+manifest['id']+':'+family,
                    experiment_set_key='i6dmo:pair-retention',project='predictor',phase='I6DMO_PAIR_RETENTION',
                    actor='predictor',lake='LOCAL_RETAINED',lineage='UNVERIFIED',
                    config_sha256=p['profile_id'],code_commit=p['members'][0]['source_code_commit'],metrics=metrics,
                    tags=dict(pair_id=manifest['id'],family=family,profile_id=p['profile_id'],
                        all_outputs_beat_naive='true',checkpoints=str(len(p['members'])),
                        csv_sha256=p['csv_sha256'],pair_manifest_sha256=B.file_sha256(target/'PAIR.json'),
                        test_read='false',fit_executed='false',model_json=json.dumps(p['model'],sort_keys=True)))
                box.enqueue(report,source_unit='pair-retention:'+family)
            publication=box.pump(HttpWarehouse(warehouse['url'],warehouse['token_env']))
        finally:box.close()
        state='READY_FOR_DEVELOPMENT_SIMULATION_TEST_BLOCKED' if not publication['pending'] and not publication['mismatch'] else 'WAIT_PUBLICATION'
        proof=dict(state='VERIFIED',copies=2,checkpoints=count,
            manifest_sha256=B.file_sha256(target/'PAIR.json'),archive_status=status,
            note='Local artifact volume and distinct archive worker; source weights authenticated on both.')
        B.put(target/'RETENTION_READBACK.json',proof)
        return save(state,archive=status,retention=proof,publication=publication,csv_readback=csv_proof)


def verify_csvs(target):
    import types
    from tools.i6dmo import strategy_adapter as S
    target=Path(target);manifest=B.read(target/'PAIR.json');observed={}
    for family in manifest['families'].values():
        if B.file_sha256(target/family['csv'])!=family['csv_sha256']:raise ValueError('CSV_DIGEST_CHANGED')
        for member in family['members']:
            with np.load(target/member['folder']/'VALIDATION.npz',allow_pickle=False) as z:
                for ts,price in zip(z['origin_ts'],z['origin_price']):
                    t=datetime.fromtimestamp(int(ts),tz=timezone.utc)
                    if t in observed and observed[t]!=float(price):raise ValueError('CSV_SOURCE_PRICE_CHANGED')
                    observed[t]=float(price)
    data,_=S._read_pinned(S.BUSINESS_CSV_PIN['path'],S.BUSINESS_CSV_PIN['commit'])
    if hashlib.sha256(data).hexdigest()!=S.BUSINESS_CSV_PIN['sha256']:raise ValueError('CSV_PIN_CHANGED')
    module=types.ModuleType('business_csv');exec(compile(data,'<pinned-business-csv>','exec'),module.__dict__)
    pair=module.load_pair(target/'short.csv',target/'long.csv',observed)
    result=dict(state='VERIFIED',origin_rows={k:len(v) for k,v in pair.items()},
        price_alignment='EXACT',timestamp_semantics='ORIGIN_UTC',
        calendar_sha256=manifest['calendar_sha256'],csv_pin=S.BUSINESS_CSV_PIN,test_read=False)
    B.put(target/'CSV_READBACK.json',result);return result


def install(spec_path):
    spec=B.read(spec_path);warehouse=B.read(spec['registry_spec'])['warehouse']
    folder=Path.home()/'.config/systemd/user';folder.mkdir(parents=True,exist_ok=True)
    values=[str(Path(__file__).resolve().parents[2]),str(Path(spec_path).resolve()),sys.executable,
        warehouse['environment_file']]
    if any(not Path(v).is_absolute() or any(c.isspace() for c in v) for v in values):raise ValueError('SYSTEMD_PATH')
    wd,path,python,env=values
    service=f'''[Unit]
Description=Retain strictly eligible forecast pair without refits
[Service]
Type=oneshot
WorkingDirectory={wd}
EnvironmentFile={env}
Environment=CUDA_VISIBLE_DEVICES=
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
Environment=PYTHONDONTWRITEBYTECODE=1
MemoryMax=768M
MemorySwapMax=0
CPUQuota=50%
TasksMax=32
Nice=10
TimeoutStartSec=600
ExecStart={python} -B -m tools.i6dmo.pair_release follow --spec {path}
'''
    timer='''[Timer]
OnBootSec=90s
OnUnitInactiveSec=120s
Persistent=true
[Install]
WantedBy=timers.target
'''
    for suffix,body in (('service',service),('timer',timer)):
        p=folder/('i6dmo-pair-retention.'+suffix);tmp=p.with_suffix('.tmp');tmp.write_text(body);tmp.replace(p)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=30)
    subprocess.run(['systemctl','--user','enable','--now','i6dmo-pair-retention.timer'],check=True,timeout=30)
    return dict(state='TIMER_INSTALLED',fit_executed=False,test_read=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','archive','status','follow','install'))
    parser.add_argument('--spec');parser.add_argument('--root');args=parser.parse_args()
    try:
        if args.action=='prepare':result=prepare(B.read(args.spec))
        elif args.action=='archive':result=export_archives(args.root)
        elif args.action=='follow':result=follow(B.read(args.spec))
        elif args.action=='install':result=install(args.spec)
        else:
            root=Path(args.root)
            result=B.read(root/('ARCHIVE_STATUS.json' if (root/'PAIR.json').exists() else 'STATUS.json'))
    except Exception as exc:
        if args.action=='archive':
            B.put(Path(args.root)/'ARCHIVE_STATUS.json',dict(state='FAILED_ORIGINAL_WEIGHTS_RETAINED',
                error=type(exc).__name__+':'+str(exc)[:500],at=time.time(),fit_executed=False,test_read=False))
        elif args.action in ('follow','prepare'):
            spec=B.read(args.spec)
            B.put(Path(spec['root'])/'STATUS.json',dict(state='FAILED_EXISTING_ARTIFACTS_RETAINED',
                error=type(exc).__name__+':'+str(exc)[:500],at=time.time(),fit_executed=False,test_read=False))
        raise
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
