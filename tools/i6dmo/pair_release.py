"""Fit-free retention of all-head eligible annual recipes and business CSVs."""
import argparse
import bisect
import csv
import fcntl
import json
import math
import os
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','archive','status'))
    parser.add_argument('--spec');parser.add_argument('--root');args=parser.parse_args()
    try:
        if args.action=='prepare':result=prepare(B.read(args.spec))
        elif args.action=='archive':result=export_archives(args.root)
        else:
            root=Path(args.root)
            result=B.read(root/('ARCHIVE_STATUS.json' if (root/'PAIR.json').exists() else 'STATUS.json'))
    except Exception as exc:
        if args.action=='archive':
            B.put(Path(args.root)/'ARCHIVE_STATUS.json',dict(state='FAILED_ORIGINAL_WEIGHTS_RETAINED',
                error=type(exc).__name__+':'+str(exc)[:500],at=time.time(),fit_executed=False,test_read=False))
        raise
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
