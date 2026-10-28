"""Year-shifted replication of the two retained weekly data recipes, fit-free."""
import copy
from datetime import datetime, timezone
import math
import inspect
from pathlib import Path

import numpy as np

from tools.i6dmo import weekly18 as W, continuation_data as D, market_grid as G
from tools.i6dmo import data, resolution_data as R, weekly_window_core as WC
from tools.i6dmo import continuation as C
from tools.i6n_contracts import eligibility as E

SHORT = [2,3,4,5,6,12,18]
LONG = [18,24,36,48,60,72]
ALL = [2,3,4,5,6,12,18,24,36,48,60,72]
UTC = timezone.utc
END = datetime(2026,1,1,tzinfo=UTC)
QUALIFICATION = 'TEST_2025_CHECKPOINT_TUNING_NOT_INDEPENDENT_OOS'


def bounds(year):
    return datetime(year,1,1,tzinfo=UTC),datetime(year+1,1,1,tzinfo=UTC)


def label(frame, source, calendar, heads, end):
    """Existing business eligibility, exact observed t+h, including target maturity."""
    _,weeks=E.load_calendar(calendar);by_id={w.week_id:w for w in weeks}
    stamp=lambda t:datetime.fromtimestamp(int(t),UTC)
    observed=E.ObservedTargets({stamp(t):stamp(t) for t in source['ts']})
    split=E.SplitBounds('FIT',stamp(source['ts'][0]),end,label_available_by=end)
    mask=np.zeros((len(frame['row_ids']),len(heads)),bool)
    target=np.full(mask.shape,np.nan)
    for i,(t,wid) in enumerate(zip(frame['row_ids'],frame['week_ids'],strict=True)):
        decisions=E.evaluate_heads(stamp(t),heads,E.Role.MANAGEMENT,by_id[str(wid)],
            input_available=True,observed_targets=observed,split=split)
        for j,d in enumerate(decisions):
            if d.label_eligible:
                ix=np.searchsorted(source['ts'],int(t)+heads[j]*3600)
                if ix>=len(source['ts']) or source['ts'][ix]!=int(t)+heads[j]*3600:
                    raise ValueError('EXACT_TARGET_MISSING')
                target[i,j]=source['price'][ix];mask[i,j]=True
    return dict(frame,target_price=target,mask=mask)


def master(source, calendar, features, base):
    """Original SHORT six-head common population, max24 slots at spacings2/4."""
    bundle=data.InputBundle(**source,features=tuple(features),calendar_config=calendar,files={},
        clock={},design=base,window=data.WindowSpec.from_design(base),
        horizons=tuple(SHORT[:6]),input_dir=Path('.'))
    broad=E.SplitBounds('FIT',datetime.fromtimestamp(int(source['ts'][0]),UTC),END,label_available_by=END)
    common=None;indices={};origins=weeks=None
    for spacing in (2,4):
        design=R._design(base,spacing,24,SHORT[:6])
        table=G.build_market_table(bundle,design,evaluation_splits=(broad,),authorized_evaluation_end=END)
        if common is None:
            origins=table.origin_ts;weeks=table.week_id;common=np.ones_like(table.label_ok)
        if not np.array_equal(origins,table.origin_ts) or not np.array_equal(weeks,table.week_id):
            raise ValueError('MASTER_CANDIDATES_DIFFER')
        common &= table.label_ok & table.input_ok[:,None]
        indices[spacing]=table.slot_index
    keep=common.any(axis=1)
    frame=R._common_arrays(bundle,origins[keep],weeks[keep],common[keep],SHORT[:6])
    extended=label(frame,source,calendar,ALL,END)
    extended['mask'][:,:6]=frame['mask'];extended['target_price'][:,:6]=frame['target_price']
    return extended,{s:ix[keep] for s,ix in indices.items()},G.expected_market_hours(bundle)


def naive(a):
    result=[]
    for j in range(a['mask'].shape[1]):
        use=a['mask'][:,j]
        if not use.any():raise ValueError('FIT_HEAD_UNSUPPORTED')
        result.append(math.fsum(np.abs(a['target_price'][use,j]-a['origin_price'][use]).tolist())/int(use.sum()))
    if not all(math.isfinite(n) and n>0 for n in result):raise ValueError('NAIVE_SCALE_INVALID')
    return result


def fold(frame, indices, source, grid, period, year, family, calendar):
    hs=SHORT if family=='short' else LONG
    start,end=bounds(year)
    train0,train1=(int(data._instant(period[k]).timestamp()) for k in ('train_start','train_end'))
    if data._instant(period['train_start']) != data._instant(period['train_end']).replace(
            year=data._instant(period['train_end']).year-4):raise ValueError('NOT_FOUR_YEARS')
    base_frame={k:v[:,:6] if k in ('mask','target_price') else v for k,v in frame.items()}
    selected={
        'FIT':D._selection(base_frame,train0,train1,SHORT[:6],label_end=train1),
        'TEST':D._selection(base_frame,int(start.timestamp()),int(end.timestamp()),SHORT[:6],
            label_end=int(end.timestamp()),week_ids=period['week_ids'])}
    if not len(selected['TEST'][0]):return None
    union=np.zeros(len(source['ts']),bool)
    for s in (2,4):union[indices[s][selected['FIT'][0]].ravel()]=True
    if not union.any() or np.any(source['ts'][union]>=train1):raise ValueError('SCALER_NOT_TRAIN_ONLY')
    mean=source['x'][union].mean(axis=0).tolist();std=source['x'][union].std(axis=0).tolist()
    arrays={}
    for split,(ix,mask) in selected.items():
        a={k:v[ix].copy() for k,v in base_frame.items()};a['mask']=mask.copy()
        a['target_price'][~mask]=np.nan
        columns=[ALL.index(h) for h in hs]
        output_mask=frame['mask'][ix][:,columns].copy()
        limit=train1 if split=='FIT' else int(end.timestamp())
        output_mask &= frame['row_ids'][ix,None]+np.array(hs)[None,:]*3600<limit
        target=frame['target_price'][ix][:,columns].copy();target[~output_mask]=np.nan
        a.update(mask=output_mask,target_price=target)
        slots=indices[2][ix,-13:];instants=source['ts'][slots];origins=a['row_ids']
        if not np.array_equal(instants[:,-1],origins) or np.any(instants>origins[:,None]):
            raise ValueError('ORIGIN_ALIGNMENT')
        closure=np.zeros(instants.shape,bool);closure[:,1:]=np.diff(instants,axis=1)>7200
        a.update(x=source['x'][slots],offsets_hours=(instants-origins[:,None])/3600.,closure_crossed=closure)
        if family=='short':
            windows={n:WC.window_arrays(a,source,grid,n) for n in (7,13,25,49)}
            common=np.logical_and.reduce([valid for _,valid in windows.values()])
            a=WC.subset(windows[13][0],common)
        arrays[split]=a
    if not arrays['TEST']['mask'].any():return None
    scales=R._seal_scales(dict(input_mean=mean,input_std=std,std=std,
        naive_mae=naive(arrays['FIT']),horizons=hs))
    for split,a in arrays.items():
        limit=train1 if split=='FIT' else int(end.timestamp())
        if np.any((a['origin_ts'][:,None]+np.array(hs)[None,:]*3600)[a['mask']]>=limit):
            raise ValueError('LABEL_NOT_MATURE')
    return arrays,scales


def periods(calendar,year):
    base=W.read(W.REPO/'docs/i6n/direct_multi_output_static/EXPERIMENT.short.json')
    base=copy.deepcopy(base)
    base['splits']['validation']=dict(zip(('start','end'),(x.isoformat() for x in bounds(year))))
    return D._calendar_periods(calendar,base)['WEEKLY']


def prepare(spec):
    from tools.i6dmo import continuation_models as M
    source_root=Path(spec['source']);receipt=W.read(source_root/'SOURCE.json')
    W.authenticate(source_root,receipt['files']);source=D._read(source_root/'SOURCE.npz')
    calendar=W.read(source_root/'CALENDAR.json');out=Path(spec['data_root']);out.mkdir(parents=True,exist_ok=True)
    if (out/'PREPARED.json').exists():
        result=W.read(out/'PREPARED.json');W.authenticate(out,result['files'])
        if result['source_sha256']!=receipt['sha256']:raise ValueError('SOURCE_CHANGED')
        return result
    base=W.read(W.REPO/'docs/i6n/direct_multi_output_static/EXPERIMENT.short.json')
    W.B.put(out/'PREPARE_PROGRESS.json',dict(state='BUILDING_MASTER'))
    cache_key=W.B.digest(dict(source=receipt['sha256'],base=base,
        master=inspect.getsource(master),label=inspect.getsource(label),grid_code=W.file_sha256(Path(G.__file__))))
    cached=out/'MASTER_CACHE.json'
    if cached.exists() and W.read(cached)['key']==cache_key:
        cache=W.read(cached);W.authenticate(out,cache['files'])
        frame=D._read(out/'MASTER.npz');indices={s:D._read(out/('INDEX'+str(s)+'.npz'))['slot_index'] for s in (2,4)}
        grid=D._read(out/'GRID.npz')['grid']
    else:
        frame,indices,grid=master(source,calendar,receipt['features'],base)
        np.savez_compressed(out/'MASTER.npz',**frame)
        for s,v in indices.items():np.savez_compressed(out/('INDEX'+str(s)+'.npz'),slot_index=v)
        np.savez_compressed(out/'GRID.npz',grid=grid)
        W.B.put(cached,dict(key=cache_key,files={n:W.file_sha256(out/n) for n in ('MASTER.npz','INDEX2.npz','INDEX4.npz','GRID.npz')}))
    references=W.read(spec['pair'])['families'];files={};cells=[]
    replay=[]
    for family in ('short','long'):
        original_spec=W.read(spec['reference_specs'][family])
        candidates=[c for c in original_spec['cells'] if family=='long' or c.get('arm')=='window13']
        candidates=[c for c in candidates if W.read(Path(original_spec['data_root'])/c['id']/'PERIOD.json')['parent_state']=='COMPLETE']
        for index in sorted({0,len(candidates)//2,len(candidates)-1}):
            cell=candidates[index];old_root=Path(original_spec['data_root'])/cell['id']
            generated=fold(frame,indices,source,grid,cell['period'],2024,family,calendar)
            if generated is None:
                expected=D._read(old_root/'VALIDATION.npz')
                if expected['mask'].any():raise ValueError('REFERENCE_SUPPORTED_WEEK_LOST')
                replay.append(dict(family=family,period=cell['period']['id'],arrays='EXACT_EMPTY_EVALUATION'))
                continue
            actual,scales=generated
            old=W.read(old_root/'PERIOD.json')
            if scales!=old['scales']:raise ValueError('REFERENCE_2024_SCALES_DIFFER:'+cell['id'])
            for split,prior in (('FIT','FIT'),('TEST','VALIDATION')):
                expected=D._read(old_root/(prior+'.npz'))
                if set(actual[split])!=set(expected):raise ValueError('REFERENCE_ARRAY_KEYS_DIFFER')
                for key,value in expected.items():
                    if not np.array_equal(actual[split][key],value,equal_nan=value.dtype.kind in 'fc'):
                        raise ValueError('REFERENCE_2024_ARRAY_DIFFER:'+family+':'+cell['id']+':'+split+':'+key)
            replay.append(dict(family=family,period=cell['period']['id'],arrays='EXACT',scales='EXACT'))
    W.B.put(out/'RECIPE_REPLAY_2024.json',replay);files['RECIPE_REPLAY_2024.json']=W.file_sha256(out/'RECIPE_REPLAY_2024.json')
    for period in periods(calendar,2025):
        for family in ('short','long'):
            hs=SHORT if family=='short' else LONG
            cid=family+'__'+period['id'];folder=out/cid;folder.mkdir(exist_ok=True)
            cell=dict(id=cid,stage='weekly_test',group='short',family=family,period=period,
                horizons=hs,kind='conv',head='lstm',slots=13,core=[2],spacing=2)
            generated=fold(frame,indices,source,grid,period,2025,family,calendar)
            record=dict(cell=cell,parent_state='COMPLETE' if generated else 'SKIPPED_NO_EVAL_ROWS',
                test_read=True,qualification=QUALIFICATION)
            if generated:
                arrays,scales=generated;model=copy.deepcopy(references[family]['model'])
                expected=M.build_config(receipt['features'],13,2,hs,kind='conv',seed=W.B.SEED,head='lstm',core=[2])
                if model!=expected:raise ValueError('WINNING_MODEL_CHANGED:'+family)
                resolved=dict(kind='conv',head='lstm',slots=13,spacing=2,core=[2])
                record.update(model=model,resolved=resolved,scales=scales,training=W.B.TRAINING,
                    reference_profile=references[family]['profile_id'],
                    reference_initial_weights_sha256=W.read(Path(spec['pair']).parent/
                        references[family]['members'][0]['folder']/'TRAINING.json')['initial_weights_sha256'],
                    population=dict(features=receipt['features'],period=period,test_read=True,
                        qualification=QUALIFICATION,scales_sha256=scales['scales_sha256'],
                        counts={s:a['mask'].sum(axis=0).tolist() for s,a in arrays.items()}),
                    naive_cache={s:C.measured_metrics(a,np.where(a['mask'],a['target_price'],0),hs) for s,a in arrays.items()})
                for split,a in arrays.items():
                    np.savez_compressed(folder/(split+'.npz'),**a)
                    files[cid+'/'+split+'.npz']=W.file_sha256(folder/(split+'.npz'))
            W.B.put(folder/'PERIOD.json',record);files[cid+'/PERIOD.json']=W.file_sha256(folder/'PERIOD.json')
            cells.append(cell)
            W.B.put(out/'PREPARE_PROGRESS.json',dict(state='PREPARING',completed=len(cells),at=__import__('time').time()))
    result=dict(schema='i6dmo.weekly_test.prepared.v1',cells=cells,files=files,
        source_sha256=receipt['sha256'],reference_pair_sha256=W.file_sha256(spec['pair']),
        qualification=QUALIFICATION,test_read=True,training=W.B.TRAINING,seed=W.B.SEED)
    result['sha256']=W.B.digest(result);W.B.put(out/'PREPARED.json',result)
    return result
