"""Remove only short outputs from the exact weekly twelve-head parent."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from tools.i6dmo import weekly18 as W

BASE = "5e1bb01b2b2b3a41da9191442e4399c4b2b7d95f"
PARENT = "469eef418c2519959f718d40bdc06e6ee66e256e"
NAME = "i6dmo-weekly-long-only-20261017"
PARENT_HEADS = [2,3,4,5,6,12,18,24,36,48,60,72]
HEADS = PARENT_HEADS[6:]
INITIALIZATION = "FRESH_SEEDED_PARENT_GRAPH_LONG_OUTPUT_SUBMODEL_NOT_TRAINED_WEIGHTS"
_run, _annual = W.run, W.annual


def long_model(parent):
    if parent['horizons_hours']!=PARENT_HEADS or parent['slots']!=13 or parent['core']!=[2] \
            or parent['head']!='lstm' or parent['kind']!='conv' or parent['seed']!=W.B.SEED \
            or parent['spacing_open_market_hours']!=2:
        raise ValueError('WRONG_TWELVE_HEAD_PARENT')
    return dict(copy.deepcopy(parent),horizons_hours=HEADS)


def project_arrays(old):
    if old['mask'].ndim!=2 or old['mask'].shape[1]!=12 or old['target_price'].shape!=old['mask'].shape:
        raise ValueError('PARENT_HEAD_SHAPE')
    return {k:v[:,6:].copy() if k in ('target_price','mask') else v.copy() for k,v in old.items()}


def assert_projection(old,new):
    if set(old)!=set(new):raise ValueError('PARENT_ARRAY_CHANGED:keys')
    for k,v in old.items():
        expected=v[:,6:] if k in ('target_price','mask') else v
        if not np.array_equal(expected,new[k],equal_nan=v.dtype.kind in 'fc'):
            raise ValueError('PARENT_ARRAY_CHANGED:'+k)


def project_scales(parent):
    if parent['horizons']!=PARENT_HEADS or len(parent['naive_mae'])!=12:raise ValueError('PARENT_SCALE_SHAPE')
    s=copy.deepcopy(parent);s.update(naive_mae=list(parent['naive_mae'][6:]),horizons=HEADS)
    s.pop('scales_sha256',None);s['scales_sha256']=W.B.digest(s)
    return s


def source_identity(parent):
    if parent['code_commit']!=PARENT or parent['training']!=W.B.TRAINING or parent['seed']!=W.B.SEED:
        raise ValueError('PARENT_IDENTITY_CHANGED')
    operational={'tools/i6dmo/weekly18.py','tools/i6dmo/resolution_models.py','tools/i6dmo/best_registry.py'}
    for rel,sha in parent['code_files'].items():
        if rel not in operational and W.file_sha256(W.REPO/rel)!=sha:raise ValueError('PARENT_SOURCE_CHANGED:'+rel)
    for rel in W.FROZEN+('tools/i6dmo/continuation_models.py',)+tuple(sorted(operational)):
        old=W.run_cmd(['git','-C',W.REPO,'show',BASE+':'+rel])
        if (W.REPO/rel).read_text()!=old:raise ValueError('SOURCE_CHANGED:'+rel)


def prepare(spec):
    from tools.i6dmo import continuation_data as D, continuation as C
    parent=W.read(spec['parent_spec']);source_identity(parent)
    close=W.read(Path(parent['root'])/'CLOSE.json')
    if close['state']!='CLOSED' or close['publication']['pending'] or close['publication']['mismatch'] \
            or close['mirrors']['state']!='VERIFIED':raise ValueError('PARENT_NOT_CLOSED')
    src=Path(parent['data_root']);prior=W.prepared_document(parent,src);W.authenticate(src,prior['files'])
    out=Path(spec['data_root']);out.mkdir(parents=True,exist_ok=True)
    if (out/'PREPARED.json').exists():
        p=W.read(out/'PREPARED.json');W.authenticate(out,p['files'])
        if p['parent_spec_sha256']!=W.file_sha256(spec['parent_spec']):raise ValueError('PARENT_SPEC_CHANGED')
        return p
    files={};cells=[]
    for pc in prior['cells']:
        terminal=W.terminal(parent['artifacts'],pc,parent)
        if not terminal or terminal['state'] not in ('COMPLETE','SKIPPED_NO_EVAL_ROWS'):raise ValueError('PARENT_WEEK_NOT_COMPLETE')
        old_record=W.read(src/pc['id']/'PERIOD.json');cell=dict(pc,horizons=HEADS)
        folder=out/pc['id'];folder.mkdir(exist_ok=True)
        record=dict(cell=cell,parent_state=terminal['state'],parent_result_sha256=terminal['sha256'])
        if terminal['state']=='COMPLETE':
            arrays={};digests={}
            for split in ('FIT','VALIDATION'):
                before=D._read(src/pc['id']/(split+'.npz'));arrays[split]=project_arrays(before)
                assert_projection(before,arrays[split])
                digests[split]={k:D.R._digest(v) for k,v in arrays[split].items()}
                np.savez_compressed(folder/(split+'.npz'),**arrays[split])
                files[pc['id']+'/'+split+'.npz']=W.file_sha256(folder/(split+'.npz'))
            scales=project_scales(old_record['scales']);model=long_model(old_record['model'])
            caches={s:C.measured_metrics(a,np.where(a['mask'],a['target_price'],0),HEADS) for s,a in arrays.items()}
            for split in arrays:
                for actual,expected in zip(caches[split],terminal['metrics'][split][6:],strict=True):
                    if any(actual[k]!=expected[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error','pairs_sha256')):
                        raise ValueError('PARENT_LONG_NAIVE_CHANGED')
            meta=dict(old_record['population'],horizons=HEADS,full_horizons=HEADS,scales_sha256=scales['scales_sha256'],
                counts={s:a['mask'].sum(axis=0).tolist() for s,a in arrays.items()},
                validation_counts=arrays['VALIDATION']['mask'].sum(axis=0).tolist(),
                initialization_policy=INITIALIZATION,parent_period_result_sha256=terminal['sha256'],
                parent_prepared_period_sha256=W.B.digest(old_record))
            record.update(model=model,parent_model=old_record['model'],resolved=old_record['resolved'],scales=scales,
                population=meta,training=old_record['training'],parent_metrics=terminal['metrics'],naive_cache=caches,
                parent_initial_weights_sha256=terminal['initial_weights_sha256'],
                model_delta=dict(horizons_hours=HEADS),projection=dict(state='EXACT_LONG_COLUMNS',arrays=digests))
        W.B.put(folder/'PERIOD.json',record);files[pc['id']+'/PERIOD.json']=W.file_sha256(folder/'PERIOD.json');cells.append(cell)
        W.B.put(out/'PREPARE_PROGRESS.json',dict(terminal=len(cells),total=len(prior['cells']),at=time.time()))
    p=dict(schema='i6dmo.weekly_long_only.prepared.v1',cells=cells,files=files,
        parent_spec_sha256=W.file_sha256(spec['parent_spec']),parent_prepared_sha256=prior['sha256'],
        original_training=W.B.TRAINING,seed=W.B.SEED,initialization=INITIALIZATION,test_read=False,qualification=W.QUALIFICATION)
    p['sha256']=W.B.digest(p);W.B.put(out/'PREPARED.json',p);return p


def project_model(config,builder):
    from tools.i6dmo import continuation_models as M
    parent_config=dict(copy.deepcopy(config),horizons_hours=PARENT_HEADS)
    if long_model(parent_config)!=config:raise ValueError('LONG_MODEL_CONFIG_CHANGED')
    parent=builder(parent_config)
    forecast=M.R.HeadListModel(parent.forecast_model.inputs,parent.forecast_model.outputs[6:],name=parent.forecast_model.name)
    return M.ContinuationBundle(copy.deepcopy(config),forecast)


def initialization_proof(record,inputs=None):
    from tools.i6dmo import continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    parent=M.build(record['parent_model']);new=project_model(record['model'],M.build)
    if weights_hash(parent.forecast_model)!=record['parent_initial_weights_sha256']:raise ValueError('PARENT_INITIAL_WEIGHTS_CHANGED')
    lookup={v.path:v.numpy() for v in parent.forecast_model.weights}
    for v in new.forecast_model.weights:
        if v.path not in lookup or not np.array_equal(v.numpy(),lookup[v.path]):raise ValueError('LONG_INITIAL_WEIGHT_CHANGED:'+v.path)
    if new.forecast_model.count_params()!=1740 or len(new.forecast_model.weights)!=82:raise ValueError('LONG_CAPACITY_CHANGED')
    if inputs is not None:
        a=parent.forecast_model(inputs,training=False);b=new.forecast_model(inputs,training=False)
        if len(a)!=12 or len(b)!=6 or any(not np.array_equal(x.numpy(),y.numpy()) for x,y in zip(a[6:],b,strict=True)):
            raise ValueError('LONG_INITIAL_FORWARD_CHANGED')
    return dict(state='EXACT',parameters=1740,compared_tensors=82,
        parent_initial_weights_sha256=weights_hash(parent.forecast_model),new_initial_weights_sha256=weights_hash(new.forecast_model),
        long_forward='EXACT' if inputs is not None else 'NOT_REQUESTED',initialization=INITIALIZATION)


def fit_scoped(*args,original_fit=None,**kwargs):
    from tools.i6dmo import continuation as C, continuation_models as M
    original=M.build
    with patch.object(M,'build',lambda c:project_model(c,original)):
        return (original_fit or C.fit_cell)(*args,**kwargs)


def run(spec):
    # Importing this lightweight controller does not initialize CUDA before admission.
    from tools.i6dmo import continuation as C
    original=C.fit_cell
    def invoke(*a,**k):
        return fit_scoped(*a,original_fit=original,**k)
    with patch.object(C,'fit_cell',invoke):return _run(spec)


def preflight(spec):
    from tools.i6dmo import continuation_data as D, train as T
    import tensorflow as tf
    W.verify(spec);p=W.prepared_document(spec,spec['gpu_data']);W.authenticate(spec['gpu_data'],p['files'])
    if tf.__version__!='2.21.0' or importlib.metadata.version('keras')!='3.13.2' or not tf.config.list_physical_devices('GPU'):
        raise ValueError('ORIGINAL_GPU_RUNTIME_REQUIRED')
    for c in p['cells']:
        r=W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json')
        if r['parent_state']!='COMPLETE':continue
        if r['training']!=W.B.TRAINING or long_model(r['parent_model'])!=r['model']:raise ValueError('UNDECLARED_CONFIG_CHANGE')
        for split in ('FIT','VALIDATION'):
            for a,b in zip(r['naive_cache'][split],r['parent_metrics'][split][6:],strict=True):
                if any(a[k]!=b[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error','pairs_sha256')):
                    raise ValueError('UNPAIRED_LONG_NAIVE')
    record=next(W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json') for c in p['cells']
        if W.read(Path(spec['gpu_data'])/c['id']/'PERIOD.json')['parent_state']=='COMPLETE')
    arrays=D._read(Path(spec['gpu_data'])/record['cell']['id']/'FIT.npz');x,_,_=T.transform(arrays,record['scales'])
    proof=initialization_proof(record,[x[:2],arrays['offsets_hours'][:2].astype('float32'),arrays['closure_crossed'][:2].astype('float32')])
    result=dict(state='PASS',periods=len(p['cells']),fits_run=0,test_read=False,initialization=proof)
    W.B.put(Path(spec['gpu_root'])/'PREFLIGHT.json',result);return result


def annual(spec,artifacts,results):
    result=_annual(spec,artifacts,results)
    if result['state']=='COMPLETE':
        parent=W.read(spec['parent_spec']);old=W.read(Path(parent['artifacts'])/'ANNUAL.json')
        rows=old['metrics'][6:]
        for actual,expected in zip(result['metrics'],rows,strict=True):
            if any(actual[k]!=expected[k] for k in ('horizon','n','naive_mae','sum_abs_naive_error')):
                raise ValueError('ANNUAL_PARENT_LONG_SUPPORT_CHANGED')
        result['parent_comparison']=[dict(horizon=a['horizon'],parent_relative_error=b['relative_error'],
            long_only_relative_error=a['relative_error'],delta_relative_error=a['relative_error']-b['relative_error'])
            for a,b in zip(result['metrics'],rows,strict=True)]
    return result


def configure():
    W.NAME=NAME;W.BASE=PARENT;W.HEADS=HEADS;W.SHORT=HEADS
    W.MODULE='tools.i6dmo.weekly_long';W.PHASE='I6DMO_WEEKLY_LONG_ONLY'
    W.TITLE='# Weekly Long Only Versus Twelve Heads';W.GPU_WALL=3600
    W.prepare=prepare;W.source_identity=source_identity;W.initialization_proof=initialization_proof
    W.preflight=preflight;W.run=run;W.annual=annual


def main():
    configure();W.main()


if __name__=='__main__':main()
