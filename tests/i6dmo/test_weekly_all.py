import ast
import copy
from datetime import datetime,timezone,timedelta
import subprocess

import numpy as np
import pytest

from tools.i6dmo import weekly_all as A, weekly18 as W


def parent_model():
    return dict(schema='i6dmo.continuation_model.v1',features=['a']*19,
        slots=13,spacing_open_market_hours=2,core=[2],head='lstm',kind='conv',
        seed=6102026,horizons_hours=A.PARENT_HEADS)


def test_only_config_delta_is_five_new_heads():
    p=parent_model(); c=A.extend_model(p)
    assert c['horizons_hours']==[2,3,4,5,6,12,18,24,36,48,60,72]
    c['horizons_hours']=A.PARENT_HEADS
    assert c==p


def test_preserve_seven_columns_without_global_intersection():
    old=dict(x=np.arange(12).reshape(3,4),mask=np.ones((3,7),bool),
        target_price=np.arange(21).reshape(3,7).astype(float),row_ids=np.arange(3))
    new={**old,'mask':np.column_stack([old['mask'],np.zeros((3,5),bool)]),
         'target_price':np.column_stack([old['target_price'],np.full((3,5),np.nan)])}
    A.assert_projection(old,new)
    assert len(new['row_ids'])==3
    new['target_price'][1,6]+=1
    with pytest.raises(ValueError,match='PARENT_ARRAY_CHANGED'):
        A.assert_projection(old,new)


@pytest.mark.parametrize('h',[24,36,48,60,72])
def test_exact_new_target_force_flat_and_maturity(h):
    from tests.i6dmo import _synth
    from tools.i6n_contracts import eligibility as E
    weeks=_synth.calendar_weeks(datetime(2024,1,7,tzinfo=timezone.utc),datetime(2024,1,7,tzinfo=timezone.utc))
    cal=_synth.calendar_config(weeks);_,ws=E.load_calendar(cal);w=ws[0]
    ts=_synth.bar_ends(weeks);prices=np.arange(len(ts),dtype=float)+1
    good=int(w.operational_start.timestamp());deadline=int(w.force_flat.timestamp())-h*3600
    a=dict(row_ids=np.array([good,deadline]),week_ids=np.array([w.week_id]*2))
    period=dict(train_start=E.iso(w.market_open),train_end=E.iso(w.market_close))
    base=dict(splits={'validation':dict(start=E.iso(w.market_open),end=E.iso(w.market_close))})
    y,m=A.label_horizon(a,dict(ts=ts,price=prices),cal,period,'VALIDATION',base,h)
    assert m.tolist()==[True,False] and y[0]==prices[np.searchsorted(ts,good+h*3600)]
    keep=ts!=good+h*3600
    _,m=A.label_horizon(a,dict(ts=ts[keep],price=prices[keep]),cal,period,'VALIDATION',base,h)
    assert not m.any()
    period['train_end']=E.iso(w.operational_start+timedelta(hours=h))
    _,m=A.label_horizon(a,dict(ts=ts,price=prices),cal,period,'FIT',base,h)
    assert not m.any()


def test_trainer_fit_and_layers_are_original_bytes():
    for rel in W.FROZEN:
        assert (W.REPO/rel).read_bytes()==subprocess.check_output(['git','show',A.BASE+':'+rel],cwd=W.REPO)
    old=subprocess.check_output(['git','show',A.BASE+':tools/i6dmo/continuation_models.py'],cwd=W.REPO).decode()
    assert (W.REPO/'tools/i6dmo/continuation_models.py').read_text()==old.replace(A.OLD_CATALOGUE,A.NEW_CATALOGUE)


def test_shared_worker_science_unchanged():
    old=ast.parse(subprocess.check_output(['git','show',A.BASE+':tools/i6dmo/weekly18.py'],cwd=W.REPO).decode())
    new=ast.parse((W.REPO/'tools/i6dmo/weekly18.py').read_text())
    for name in ('run','annual','terminal','verify','mirror','prepared_document','pool_metrics'):
        get=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        assert get(old)==get(new)


def test_successor_waits_without_launch_or_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(A,'dispatch',lambda s:pytest.fail('premature dispatch'))
    spec=dict(root=str(tmp_path/'new'),parent_root=str(tmp_path/'old'))
    assert A.successor(spec)['state']=='WAIT_PREDECESSOR'


@pytest.mark.parametrize('change',[{'state':'CLOSED_WITH_FAILURE'},{'publication':{'pending':1,'mismatch':0}},{'mirrors':{'state':'MISMATCH'}}])
def test_successor_refuses_unfinished_publication_or_failed_parent(tmp_path,monkeypatch,change):
    parent=tmp_path/'old';parent.mkdir()
    r=dict(state='CLOSED',publication=dict(pending=0,mismatch=0),mirrors=dict(state='VERIFIED'))
    r.update(change);W.B.put(parent/'CLOSE.json',r)
    monkeypatch.setattr(A,'dispatch',lambda s:pytest.fail('unsafe dispatch'))
    assert A.successor(dict(root=str(tmp_path/'new'),parent_root=str(parent)))['state']=='BLOCKED_PREDECESSOR'


def test_successor_dispatches_once_then_stops_itself(tmp_path,monkeypatch):
    parent=tmp_path/'old';parent.mkdir()
    W.B.put(parent/'CLOSE.json',dict(state='CLOSED',publication=dict(pending=0,mismatch=0),mirrors=dict(state='VERIFIED')))
    calls=[];monkeypatch.setattr(A,'dispatch',lambda s:calls.append('launch') or {'state':'ARMED'})
    monkeypatch.setattr(A,'stop_successor',lambda:calls.append('stop'))
    spec=dict(root=str(tmp_path/'new'),parent_root=str(parent))
    assert A.successor(spec)['state']=='ARMED'
    assert A.successor(spec)['state']=='ARMED'
    assert calls.count('launch')==1


def test_wait_resource_never_completes_successor(tmp_path,monkeypatch):
    parent=tmp_path/'old';parent.mkdir()
    W.B.put(parent/'CLOSE.json',dict(state='CLOSED',publication=dict(pending=0,mismatch=0),mirrors=dict(state='VERIFIED')))
    monkeypatch.setattr(A,'dispatch',lambda s:{'state':'WAIT_GPU'})
    monkeypatch.setattr(A,'stop_successor',lambda:pytest.fail('premature stop'))
    spec=dict(root=str(tmp_path/'new'),parent_root=str(parent))
    assert A.successor(spec)['state']=='WAIT_GPU'
    assert not (tmp_path/'new/SUCCESSOR_DONE.json').exists()
