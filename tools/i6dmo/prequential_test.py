"""Load VALIDATION weights; predict/seal week n BEFORE updating for week n+1."""
import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from tools.i6dmo import weekly18 as W, weekly_test as P, continuation as C
from tools.i6dmo import continuation_data as CD

NAME='i6dmo-prequential-test-20261017'
MODULE='tools.i6dmo.prequential_test'
PHASE='I6DMO_PREQUENTIAL_TEST_2025'
QUALIFICATION='PREQUENTIAL_TEST_PREDICT_THEN_UPDATE'
TRAINING=W.B.TRAINING
PREDICT_KEYS={'x','offsets_hours','closure_crossed','origin_ts','row_ids','origin_price'}


def inference_inputs(a,scales):
    if set(a)!=PREDICT_KEYS:raise ValueError('PREDICT_INPUT_KEYS')
    x=((np.asarray(a['x'],dtype='float64')-np.asarray(scales['input_mean']))/
        np.asarray(scales['input_std'])).astype('float32')
    if not np.isfinite(x).all() or np.any(a['offsets_hours']>0) or np.any(a['offsets_hours'][:,-1]!=0):
        raise ValueError('PREDICT_INPUT_NONFINITE_OR_FUTURE')
    return [x,a['offsets_hours'].astype('float32'),a['closure_crossed'].astype('float32')]


def available(donor,origins):
    if len(origins) and int(CD.data._instant(donor['available_by']).timestamp())>int(np.min(origins)):
        raise ValueError('DONOR_NOT_YET_AVAILABLE')


def execute_week(record,donor,folder,reader,backend,*,input_sha256):
    """The label reader is invoked only AFTER durable forecast creation/verification."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    a=reader('PREDICT');available(donor,a['origin_ts'])
    seal=folder/'PREDICTION.json';forecast=folder/'PREDICTED.npz'
    identity=dict(donor_id=donor['id'],input_sha256=input_sha256,cell=record['cell'])
    if seal.exists():
        proof=W.read(seal)
        if proof['identity']!=identity or W.file_sha256(forecast)!=proof['prediction_sha256']:
            raise ValueError('SEALED_FORECAST_IDENTITY')
        predicted=CD._read(forecast)
        if not np.array_equal(predicted['origin_ts'],a['origin_ts']):raise ValueError('SEALED_ORIGINS_CHANGED')
        price=predicted['prediction']
    else:
        price=np.asarray(backend.predict(donor,a),dtype='float64')
        if price.shape!=(len(a['origin_ts']),len(record['cell']['horizons'])) or not np.isfinite(price).all():
            raise ValueError('FORECAST_SHAPE_OR_NONFINITE')
        temp=folder/'PREDICTED.partial.npz'
        np.savez_compressed(temp,prediction=price,origin_ts=a['origin_ts'],origin_price=a['origin_price'])
        with temp.open('rb') as stream:__import__('os').fsync(stream.fileno())
        temp.replace(forecast)
        proof=dict(identity=identity,prediction_sha256=W.file_sha256(forecast),
            stage='PREDICTION_SEALED_BEFORE_OBSERVATION',created_at=time.time())
        W.B.put(seal,proof)
    observed=reader('OBSERVED')
    for key in PREDICT_KEYS:
        if not np.array_equal(observed[key],a[key]):raise ValueError('OBSERVED_INPUT_CHANGED:'+key)
    scores=C.measured_metrics(observed,price,record['cell']['horizons'])
    W.B.put(folder/'SCORE.json',dict(metrics=scores,forecast_sha256=proof['prediction_sha256'],
        stage='OBSERVED_WEEK_SCORED_BEFORE_UPDATE',created_at=time.time()))
    np.savez_compressed(folder/'TEST.npz',prediction=price,origin_ts=a['origin_ts'],
        origin_price=a['origin_price'],target_price=observed['target_price'],mask=observed['mask'])
    fit=reader('FIT')
    if record.get('naive_cache'):
        for split,values in (('FIT',fit),('TEST',observed)):
            cache=C.measured_metrics(values,np.where(values['mask'],values['target_price'],0),record['cell']['horizons'])
            if cache!=record['naive_cache'][split]:raise ValueError('NAIVE_CACHE_CHANGED')
    updated=backend.update(donor,fit,observed,record,folder/'update')
    if W.file_sha256(forecast)!=proof['prediction_sha256']:raise ValueError('UPDATE_CHANGED_PREDICTION')
    return dict(updated,metrics=dict(updated.get('metrics',{}),TEST=scores),
        inference_donor=donor['id'],prediction_sha256=proof['prediction_sha256'],
        qualification=QUALIFICATION,test_read=True)


def checkpoint_files(root,folder,model,scales,tensor_hash,available_by):
    files={str(p.relative_to(root)):W.file_sha256(p) for p in folder.iterdir() if p.is_file()}
    body=dict(root='worker',model=model,scales=scales,weights_sha256=tensor_hash,
        archive=str((folder/'forecast_model.keras').relative_to(root)),files=files,available_by=available_by)
    return dict(body,id=W.B.digest(body))


def validate_update_span(period,fit,observed,heads):
    cutoff=int(CD.data._instant(period['train_end']).timestamp())
    start=CD.data._instant(period['train_start']);end=CD.data._instant(period['train_end'])
    if start!=end.replace(year=end.year-4):raise ValueError('NOT_FOUR_CALENDAR_YEARS')
    hs=np.array(heads)*3600
    if np.any(fit['origin_ts']<int(start.timestamp())) or np.any(fit['origin_ts']>=cutoff) or np.any(
            (fit['origin_ts'][:,None]+hs)[fit['mask']]>=cutoff):raise ValueError('TRAIN_CUTOFF_OR_LABEL_MATURITY')
    if not observed['mask'].any():raise ValueError('NO_OBSERVED_MONITOR_PAIRS')
    if np.any(observed['origin_ts']<cutoff) or np.any((observed['origin_ts'][:,None]+hs)[observed['mask']]>=
            int(CD.data._instant(period['validation_end']).timestamp())):raise ValueError('OBSERVED_WEEK_NOT_FINISHED')


class GpuBackend:
    def __init__(self,spec,root):self.spec=spec;self.root=Path(root)

    def load(self,donor):
        from tools.i6dmo import continuation_models as M,model as Base
        from predictor_plugins.modular_temporal import weights_hash
        if M.tf.__version__!='2.21.0' or importlib.metadata.version('keras')!='3.13.2' or not M.tf.config.list_physical_devices('GPU'):
            raise ValueError('RETAINED_GPU_RUNTIME_REQUIRED')
        base=Path(self.spec['gpu_data']) if donor['root']=='data' else self.root
        W.authenticate(base,donor['files'])
        if Path(donor['archive']).is_absolute() or '..' in Path(donor['archive']).parts:raise ValueError('DONOR_PATH')
        Base.enable_determinism();M.keras.backend.clear_session();M.keras.utils.set_random_seed(W.B.SEED)
        model=M.keras.models.load_model(base/donor['archive'],compile=False,safe_mode=True)
        if weights_hash(model)!=donor['weights_sha256']:raise ValueError('DONOR_WEIGHTS_CHANGED')
        return M.ContinuationBundle(copy.deepcopy(donor['model']),model)

    def predict(self,donor,a):
        from tools.i6dmo import train as T
        bundle=self.load(donor)
        inputs=inference_inputs(a,donor['scales'])
        output=T.monitor_forward(bundle.forecast_model,inputs,TRAINING['batch_size'])
        return T.to_price(output,a,donor['scales'])

    def update(self,donor,fit,observed,record,folder):
        from tools.i6dmo import train as T,continuation_models as M
        from predictor_plugins.modular_temporal import weights_hash
        if donor['model']!=record['model']:raise ValueError('WINNING_MODEL_CHANGED')
        period=record['cell']['period']
        validate_update_span(period,fit,observed,record['cell']['horizons'])
        folder=Path(folder)
        receipt=folder/'UPDATE.json'
        if receipt.exists():
            r=W.read(receipt);W.authenticate(self.root,r['next_checkpoint']['files'])
            if r['donor_id']!=donor['id']:raise ValueError('UPDATE_DONOR_CHANGED')
            return r
        folder.mkdir(parents=True,exist_ok=True)
        W.B.put(folder/'STARTED.json',dict(donor_id=donor['id'],at=time.time()))
        bundle=self.load(donor)
        scales=record['scales'];prepared={}
        for name,a in (('FIT',fit),('OBSERVED',observed)):
            x,y,mask=T.transform(a,scales)
            prepared[name]=([x,a['offsets_hours'].astype('float32'),a['closure_crossed'].astype('float32')],y,mask)
        def progress(p):
            q=dict(p);q['train_loss']=q.pop('fit_loss');q['observed_week_loss']=q.pop('inner_loss')
            W.B.progress(self.root,state='RUNNING',stage='UPDATE_FOR_NEXT_WEEK',cell=record['cell']['id'],**q)
        training=T.fit_dmo(bundle,prepared['FIT'],prepared['OBSERVED'],TRAINING,W.B.SEED,
            ids=(fit['row_ids'],observed['row_ids']),record_device='gpu',progress=progress)
        if training['initial_weights_sha256']!=donor['weights_sha256']:raise ValueError('WARM_START_WEIGHTS_CHANGED')
        training['split_roles']=dict(fit='FOUR_YEARS_BEFORE_OBSERVED_WEEK',inner='ALREADY_OBSERVED_WEEK',
            use='NEXT_WEEK_ONLY',test_read=True)
        W.B.put(folder/'TRAINING.json',training);W.B.put(folder/'SCALES.json',scales)
        W.B.put(folder/'CONFIG.json',dict(model=record['model'],training=TRAINING,seed=W.B.SEED,
            period=period,donor_id=donor['id'],monitor='0.5*(TRAIN+ALREADY_OBSERVED_WEEK)',qualification=QUALIFICATION))
        bundle.forecast_model.save_weights(folder/'model.weights.h5')
        bundle.forecast_model.save(folder/'forecast_model.keras')
        fresh=M.keras.models.load_model(folder/'forecast_model.keras',compile=False,safe_mode=True)
        if weights_hash(fresh)!=training['restored_weights_sha256']:raise ValueError('SAVED_CHECKPOINT_RELOAD')
        rows={}
        for name,a in (('FIT',fit),('OBSERVED',observed)):
            pred=T.to_price(T.monitor_forward(fresh,prepared[name][0],TRAINING['batch_size']),a,scales)
            rows[name]=C.measured_metrics(a,pred,record['cell']['horizons'])
        next_checkpoint=checkpoint_files(self.root,folder,record['model'],scales,training['restored_weights_sha256'],period['validation_end'])
        result=dict(next_checkpoint=next_checkpoint,donor_id=donor['id'],parameters=int(fresh.count_params()),
            epochs=training['epochs_completed'],best_epoch=training['best_epoch'],stop_reason=training['stop_reason'],
            training_seconds=training['fit_seconds'],initial_weights_sha256=training['initial_weights_sha256'],
            restored_weights_sha256=training['restored_weights_sha256'],metrics=rows,
            cgroup_peak_bytes=T._cgroup_peak(),tensorflow=M.tf.__version__,keras=importlib.metadata.version('keras'))
        W.B.put(receipt,result)
        return result


def prepare(spec):
    old_root=Path(spec['prepared_source']);prior=W.read(old_root/'PREPARED.json')
    if prior['sha256']!=spec['prepared_source_sha256'] or prior['sha256']!=W.B.digest({k:v for k,v in prior.items() if k!='sha256'}):
        raise ValueError('DATA_RECIPE_IDENTITY')
    W.authenticate(old_root,prior['files'])
    root=Path(spec['data_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'PREPARED.json').exists():
        p=W.read(root/'PREPARED.json');W.authenticate(root,p['files'])
        if p['parent_prepared_sha256']!=prior['sha256'] or p['reference_pair_sha256']!=W.file_sha256(spec['pair']):
            raise ValueError('PREPARED_IDENTITY_CHANGED')
        return p
    pair=W.read(spec['pair']);files={};initial={}
    for family in ('short','long'):
        member=max(pair['families'][family]['members'],key=lambda m:m['period']['train_end'])
        source=Path(spec['pair']).parent/member['folder'];W.authenticate(source,member['files'])
        proof=W.read(source/'ARCHIVE.json')
        if proof['state']!='VERIFIED' or W.file_sha256(source/'forecast_model.keras')!=proof['archive_sha256']:
            raise ValueError('VALIDATION_ARCHIVE_NOT_VERIFIED')
        folder=root/'initial'/family;folder.mkdir(parents=True,exist_ok=True)
        for name in ('forecast_model.keras','model.weights.h5','CONFIG.json','SCALES.json','TRAINING.json','ARCHIVE.json'):
            shutil.copyfile(source/name,folder/name);files[str((folder/name).relative_to(root))]=W.file_sha256(folder/name)
        body=dict(root='data',model=W.read(folder/'CONFIG.json')['model'],scales=W.read(folder/'SCALES.json'),
            weights_sha256=proof['weights_sha256'],available_by=member['period']['validation_end'],
            archive='initial/'+family+'/forecast_model.keras',source_member=member['id'],
            files={k:v for k,v in files.items() if k.startswith('initial/'+family+'/')})
        if CD.data._instant(body['available_by'])>datetime(2025,1,1,tzinfo=timezone.utc):raise ValueError('INITIAL_DONOR_FROM_TEST')
        initial[family]=dict(body,id=W.B.digest(body))
    cells=[];previous={}
    for old_cell in prior['cells']:
        c=dict(old_cell,stage='prequential_test',predecessor=previous.get(old_cell['family']))
        previous[c['family']]=c['id'];cells.append(c)
        source=old_root/c['id'];record=W.read(source/'PERIOD.json');record=dict(record,cell=c,
            qualification=QUALIFICATION,procedure='PREDICT_SEAL_SCORE_UPDATE_NEXT_WEEK')
        folder=root/c['id'];folder.mkdir(exist_ok=True)
        if record['parent_state']=='COMPLETE':
            if record['model']!=initial[c['family']]['model']:raise ValueError('WINNING_CONFIG_CHANGED')
            for dst,src in (('FIT.npz','FIT.npz'),('OBSERVED.npz','TEST.npz')):
                shutil.copyfile(source/src,folder/dst);files[c['id']+'/'+dst]=W.file_sha256(folder/dst)
            observed=CD._read(source/'TEST.npz')
            np.savez_compressed(folder/'PREDICT.npz',**{k:observed[k] for k in PREDICT_KEYS})
            files[c['id']+'/PREDICT.npz']=W.file_sha256(folder/'PREDICT.npz')
        W.B.put(folder/'PERIOD.json',record);files[c['id']+'/PERIOD.json']=W.file_sha256(folder/'PERIOD.json')
        W.B.put(root/'PREPARE_PROGRESS.json',dict(completed=len(cells),total=len(prior['cells']),at=time.time()))
    p=dict(schema='i6dmo.prequential_test.v1',cells=cells,files=files,initial=initial,
        parent_prepared_sha256=prior['sha256'],reference_pair_sha256=W.file_sha256(spec['pair']),
        qualification=QUALIFICATION,training=TRAINING,seed=W.B.SEED,test_read=True)
    p['sha256']=W.B.digest(p);W.B.put(root/'PREPARED.json',p);return p


def run(spec):
    P.verify(spec);root=Path(spec['gpu_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'gpu.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={c['id']:r for c in spec['cells'] if (r:=W.terminal(root,c,spec))}
        pending=[c for c in spec['cells'] if c['id'] not in results]
        if not pending:W.B.progress(root,state='GPU_DONE');return
        cell=next((c for c in pending if not c['predecessor'] or
            (c['predecessor'] in results and results[c['predecessor']]['state'] in ('COMPLETE','SKIPPED_NO_EVAL_ROWS'))),None)
        if cell is None:W.B.progress(root,state='BLOCKED_DEPENDENCY');return
        doc=P.prepared(spec,spec['gpu_data']);cid=cell['id']
        W.authenticate(spec['gpu_data'],{k:v for k,v in doc['files'].items() if k.startswith(cid+'/')})
        record=W.read(Path(spec['gpu_data'])/cid/'PERIOD.json')
        donor=results[cell['predecessor']]['next_checkpoint'] if cell['predecessor'] else doc['initial'][cell['family']]
        folder=root/'results'/cid;folder.mkdir(parents=True,exist_ok=True)
        if record['parent_state']=='SKIPPED_NO_EVAL_ROWS':
            result=dict(state='SKIPPED_NO_EVAL_ROWS',next_checkpoint=donor,files={},test_read=True)
        else:
            apps=W.run_cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
            available_bytes=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines()
                if v.startswith('MemAvailable:')))*1024
            if apps.strip() or available_bytes<7*1024**3:
                W.B.progress(root,state='WAIT_RESOURCES',cell=cid,memory_available=available_bytes);return
            try:
                W.B.progress(root,state='RUNNING',stage='PREDICT_BEFORE_UPDATE',cell=cid)
                reader=lambda name:CD._read(Path(spec['gpu_data'])/cid/(name+'.npz'))
                result=execute_week(record,donor,folder,reader,GpuBackend(spec,root),
                    input_sha256=doc['files'][cid+'/PREDICT.npz'])
                result.update(state='COMPLETE',actual_spacing=2,seed=W.B.SEED,
                    population=dict(record['population'],manifest_sha256=spec['population_digest'],
                        inference_donor=donor['id'],inference_scaler_sha256=donor['scales']['scales_sha256'],
                        qualification=QUALIFICATION))
                result['files']={str(p.relative_to(folder)):W.file_sha256(p) for p in folder.rglob('*')
                    if p.is_file() and p.name!='RESULT.json'}
            except Exception as exc:
                result=dict(state='FAILED',files={},reason=type(exc).__name__+':'+str(exc)[:800],test_read=True)
        result.update(batch_id=NAME,cell=cell);result['sha256']=W.B.digest(result)
        W.B.put(folder/'RESULT.json',result);W.B.progress(root,state=result['state'],cell=cid)


def context():
    stack=ExitStack()
    for name,value in (('NAME',NAME),('MODULE',MODULE),('PHASE',PHASE)):
        stack.enter_context(patch.object(P,name,value))
    stack.enter_context(patch.object(P.D,'QUALIFICATION',QUALIFICATION))
    return stack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','launch','run','tick','status'))
    parser.add_argument('--spec',required=True);parser.add_argument('--cell')
    args=parser.parse_args();spec=W.read(args.spec)
    try:
        with context():
            if args.action=='prepare':
                p=prepare(spec);result=dict(cells=len(p['cells']),sha256=p['sha256'])
            elif args.action=='launch':
                with patch.object(P.D,'prepare',prepare):result=P.launch(spec)
            elif args.action=='run':result=run(spec)
            elif args.action=='tick':result=P.tick(spec)
            else:
                path=Path(spec['root'])/'STATUS.json';result=W.read(path) if path.exists() else dict(state='NOT_STARTED')
                if args.cell:result=result.get('cells',{}).get(args.cell,dict(state='PENDING'))
                else:result.pop('cells',None)
            if result:print(json.dumps(result,sort_keys=True))
    except BlockingIOError:return
    except Exception as exc:
        if args.action=='tick':W.B.put(Path(spec['root'])/'STATUS.json',dict(state='WAIT_OPERATIONAL_RETRY',
            error=type(exc).__name__+':'+str(exc)[:600],at=time.time()))
        raise


if __name__=='__main__':main()
