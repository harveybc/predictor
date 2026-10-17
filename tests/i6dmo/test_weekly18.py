import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.i6dmo import weekly18 as W


def fixture_arrays():
    origin = np.arange(10, 14, dtype=np.int64) * 3600
    a = dict(x=np.arange(4*13*19).reshape(4,13,19).astype(float),
        target_price=np.ones((4,6)), mask=np.ones((4,6),bool),
        origin_price=np.ones(4), origin_ts=origin, row_ids=origin.copy(),
        offsets_hours=np.tile(np.arange(-24,1,2),(4,1)).astype(float),
        closure_crossed=np.zeros((4,13),bool), week_ids=np.array(['w']*4))
    scales=dict(input_mean=[0.]*19, input_std=[1.]*19, std=[1.]*19,
        naive_mae=[.1]*6,horizons=W.SHORT)
    return a, scales


def test_append_changes_only_target_and_mask_and_new_naive():
    a,s=fixture_arrays(); original=copy.deepcopy(a); ss=copy.deepcopy(s)
    out,sc=W.append_column(a,s,np.array([2.,np.nan,3.,4.]),np.array([1,0,1,1],bool),
                           fit_naive=2.)
    W.assert_projection(original,out)
    assert sc['naive_mae']==s['naive_mae']+[2.]
    assert sc['input_mean']==s['input_mean'] and s==ss
    assert out['mask'].shape==(4,7) and out['mask'].sum(axis=0).tolist()==[4]*6+[3]


@pytest.mark.parametrize('key',['x','mask','target_price','origin_ts','offsets_hours','closure_crossed'])
def test_projection_detects_mutation(key):
    a,s=fixture_arrays();out,_=W.append_column(a,s,np.full(4,2.),np.ones(4,bool),fit_naive=1.)
    out[key]=out[key].copy();out[key].flat[0]=not out[key].flat[0] if key in ('mask','closure_crossed') else out[key].flat[0]+1
    with pytest.raises(ValueError,match='PARENT_ARRAY_CHANGED'):
        W.assert_projection(a,out)


def test_model_delta_only_horizons():
    parent=dict(schema='i6dmo.continuation_model.v1',seed=6102026,features=['a'],
        core=[2],head='lstm',kind='conv',slots=13,spacing_open_market_hours=2,horizons_hours=W.SHORT)
    new=W.extend_model(parent)
    assert new['horizons_hours']==W.HEADS
    new['horizons_hours']=W.SHORT
    assert new==parent


def test_original_monitor_is_not_any_head():
    from tools.i6dmo.train import MeanMonitorStopper
    st=MeanMonitorStopper(10,1e-8,200)
    assert st.identity()['monitor']=='0.5*(fit_loss+inner_loss)'
    st.update(1.,1.,[np.array([1.])])
    st.update(.9,1.2,[np.array([2.])])
    assert st.stale==1 and st.best_epoch==1


def test_scientific_sources_unchanged_except_catalogue():
    for rel in W.FROZEN:
        original=subprocess.check_output(['git','show',W.BASE+':'+rel],cwd=W.REPO)
        assert (W.REPO/rel).read_bytes()==original
    old=subprocess.check_output(['git','show',W.BASE+':tools/i6dmo/continuation_models.py'],cwd=W.REPO).decode()
    current=(W.REPO/'tools/i6dmo/continuation_models.py').read_text()
    assert current==old.replace('HORIZONS = (*R.HORIZONS, (4,))',W.CATALOGUE)


def test_naive_per_head_masks_and_scale():
    a,_=fixture_arrays();a['target_price'][:]=2.;a['mask'][1,0]=False
    a['target_price'][:,0]=[2.,999.,4.,5.]
    pred=a['target_price'].copy();pred[:,0]=a['origin_price']
    from tools.i6dmo.continuation import measured_metrics
    m=measured_metrics(a,pred,W.SHORT)
    assert m[0]['n']==3 and m[0]['naive_mae']==pytest.approx(8/3)
    assert m[0]['relative_error']==0


def test_pool_uses_sums_not_mean_weekly_percent():
    rows=[dict(horizon=18,n=1,sum_abs_error=2.,sum_abs_naive_error=1.),
          dict(horizon=18,n=9,sum_abs_error=8.,sum_abs_naive_error=9.)]
    r=W.pool_metrics(rows,18)
    assert r['mae']==1. and r['naive_mae']==1. and r['relative_error']==0


def test_new_head_uses_exact_target_and_business_deadline():
    from datetime import datetime,timedelta,timezone
    from tools.i6n_contracts import eligibility as E
    import inspect
    source=inspect.getsource(W.label18)
    assert 'evaluate_heads' in source and 'Role.MANAGEMENT' in source
    assert '18' in source and 'searchsorted' in source
    assert 'interpolate' not in source and 'ffill' not in source


def test_h18_force_flat_missing_target_and_train_maturity():
    from datetime import datetime,timedelta,timezone
    from tests.i6dmo import _synth
    from tools.i6n_contracts import eligibility as E
    weeks=_synth.calendar_weeks(datetime(2024,1,7,tzinfo=timezone.utc),datetime(2024,1,7,tzinfo=timezone.utc))
    calendar=_synth.calendar_config(weeks)
    _,ws=E.load_calendar(calendar);w=ws[0]
    ts=_synth.bar_ends(weeks);price=np.arange(len(ts),dtype=float)+1
    good=int(w.operational_start.timestamp());at_flat=int(w.force_flat.timestamp())-18*3600
    origins=np.array([good,at_flat],dtype=np.int64)
    a=dict(row_ids=origins,week_ids=np.array([w.week_id]*2))
    period=dict(train_start=E.iso(w.market_open),train_end=E.iso(w.market_close))
    base=dict(splits={'validation':{'start':E.iso(w.market_open),'end':E.iso(w.market_close)}})
    target,mask=W.label18(a,dict(ts=ts,price=price),calendar,period,'VALIDATION',base)
    assert mask.tolist()==[True,False]
    assert target[0]==price[np.searchsorted(ts,good+18*3600)]
    missing=ts!=good+18*3600
    _,mask=W.label18(a,dict(ts=ts[missing],price=price[missing]),calendar,period,'VALIDATION',base)
    assert not mask.any()
    period['train_end']=E.iso(w.operational_start+timedelta(hours=18))
    _,mask=W.label18(a,dict(ts=ts,price=price),calendar,period,'FIT',base)
    assert not mask.any()


def test_terminal_rejects_changed_artifact(tmp_path):
    folder=tmp_path/'results'/'a';folder.mkdir(parents=True)
    (folder/'v.txt').write_text('x')
    r=dict(cell={'id':'a'},batch_id='c',state='COMPLETE',files={'v.txt':W.file_sha256(folder/'v.txt')})
    r['sha256']=W.B.digest(r);W.B.put(folder/'RESULT.json',r)
    assert W.terminal(tmp_path,{'id':'a'},{'batch_id':'c'})['state']=='COMPLETE'
    (folder/'v.txt').write_text('y')
    with pytest.raises(ValueError,match='ARTIFACT'):
        W.terminal(tmp_path,{'id':'a'},{'batch_id':'c'})
