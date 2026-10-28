import copy
import subprocess

import numpy as np
import pytest

from tools.i6dmo import weekly_long as L, weekly18 as W


def model():
    return dict(schema='i6dmo.continuation_model.v1',features=['f'+str(i) for i in range(19)],
        slots=13,spacing_open_market_hours=2,core=[2],head='lstm',kind='conv',seed=6102026,
        horizons_hours=L.PARENT_HEADS)


def test_only_config_delta_is_delete_short_heads():
    before=model();after=L.long_model(before)
    assert after['horizons_hours']==[18,24,36,48,60,72]
    after['horizons_hours']=L.PARENT_HEADS
    assert after==before


@pytest.mark.parametrize('key,value',[('slots',25),('core',[4]),('seed',0),('head','dense'),('spacing_open_market_hours',4)])
def test_wrong_parent_fails_closed(key,value):
    p=model();p[key]=value
    with pytest.raises(ValueError,match='WRONG_TWELVE_HEAD_PARENT'):L.long_model(p)


def test_projection_keeps_all_origins_and_individual_masks():
    mask=np.ones((3,12),bool);mask[0,6:]=False;mask[1,8]=False
    old=dict(x=np.arange(78).reshape(3,13,2),row_ids=np.arange(3),target_price=np.arange(36.).reshape(3,12),mask=mask)
    new=L.project_arrays(old)
    assert len(new['row_ids'])==3 and not new['mask'][0].any()
    assert new['mask'].tolist()==mask[:,6:].tolist()
    L.assert_projection(old,new)
    new['row_ids']=np.array([1,2,3])
    with pytest.raises(ValueError,match='PARENT_ARRAY_CHANGED'):L.assert_projection(old,new)


def test_scalers_and_naive_are_exact_suffix_not_refit():
    s=dict(mean=[1.,2.],std=[3.,4.],naive_mae=list(range(1,13)),horizons=L.PARENT_HEADS,scales_sha256='old')
    before=copy.deepcopy(s);after=L.project_scales(s)
    assert s==before and after['mean']==s['mean'] and after['std']==s['std']
    assert after['naive_mae']==[7,8,9,10,11,12] and after['horizons']==L.HEADS


def test_scientific_and_worker_files_are_original_bytes():
    for rel in W.FROZEN+('tools/i6dmo/continuation_models.py','tools/i6dmo/weekly18.py'):
        assert (W.REPO/rel).read_bytes()==subprocess.check_output(['git','show',L.BASE+':'+rel],cwd=W.REPO)


def test_shared_fit_adapter_is_scoped_and_restores_factory(monkeypatch):
    from tools.i6dmo import continuation as C, continuation_models as M
    original=M.build;calls=[]
    def fit(*args,**kwargs):
        assert M.build is not original
        calls.append('fit');return dict(state='COMPLETE')
    monkeypatch.setattr(C,'fit_cell',fit)
    assert L.fit_scoped({}, {}, {}, None)['state']=='COMPLETE'
    assert calls==['fit'] and M.build is original


def test_retained_initial_weights_and_forward_and_native_negative_control():
    from tools.i6dmo import continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    parent=M.build(model());new=L.project_model(L.long_model(model()),M.build)
    lookup={w.path:w.numpy() for w in parent.forecast_model.weights}
    assert len(new.forecast_model.weights)==82 and new.forecast_model.count_params()==1740
    assert len(new.forecast_model.outputs)==6
    for w in new.forecast_model.weights:assert np.array_equal(w.numpy(),lookup[w.path])
    x=np.arange(2*13*19,dtype='float32').reshape(2,13,19)/1000
    offsets=np.tile(np.arange(-24,1,2,dtype='float32'),(2,1));inputs=[x,offsets,np.zeros((2,13),dtype='float32')]
    a=parent.forecast_model(inputs,training=False);b=new.forecast_model(inputs,training=False)
    assert all(np.array_equal(u.numpy(),v.numpy()) for u,v in zip(a[6:],b,strict=True))
    native=M.build(L.long_model(model()))
    assert weights_hash(native.forecast_model)!=weights_hash(new.forecast_model)


def test_monitor_loader_uses_the_same_scoped_factory(tmp_path):
    from unittest.mock import patch
    from tools.i6dmo import continuation_models as M, train as T
    from predictor_plugins.modular_temporal import weights_hash
    original=M.build;c=L.long_model(model());bundle=L.project_model(c,original)
    with patch.object(M,'build',lambda cfg:L.project_model(cfg,original)):
        monitor=T._monitor_model(bundle,tmp_path)
        assert weights_hash(monitor)==weights_hash(bundle.forecast_model)
        assert len(monitor.outputs)==6 and monitor.count_params()==1740
    assert M.build is original
