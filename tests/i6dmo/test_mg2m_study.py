import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    p=Path(__file__).parents[2]/'tools/i6dmo/mg2m_study.py'
    s=importlib.util.spec_from_file_location('study',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def arrays():
    return dict(row_id=np.array([100,200]),origin_price=np.array([1.,2.]),
        target_price=np.array([[2.,3.],[4.,6.]]),mask=np.ones((2,2),bool))


def test_each_horizon_own_naive_and_scale():
    r=module().metric_rows(arrays(),np.array([[1.,3.],[2.,5.]]),[2,18])
    assert [x['naive_mae'] for x in r]==[1.5,3.]
    assert r[0]['relative_error']==0 and r[1]['relative_error']==pytest.approx(-5/6)


def test_invalid_target_excluded_but_prediction_failure_is_not_hidden():
    m=module();a=arrays();a['mask'][0,1]=False;a['target_price'][0,1]=np.nan
    r=m.metric_rows(a,np.array([[1.,np.nan],[2.,5.]]),[2,18]);assert r[1]['n']==1
    with pytest.raises(ValueError,match='NONFINITE'):
        m.metric_rows(a,np.array([[np.nan,0.],[2.,5.]]),[2,18])


def test_cached_naive_cannot_be_reused_on_other_population():
    m=module();a=arrays();cached=m.metric_rows(a,a['target_price'],[2,18]);a['mask'][0,0]=False
    with pytest.raises(ValueError,match='NAIVE_CACHE_CHANGED'):
        m.check_cache(m.metric_rows(a,a['target_price'],[2,18]),cached)


def test_joint_mask_must_not_replace_per_head_masks():
    m=module();a=arrays();a['mask'][0,1]=False;b=copy.deepcopy(a)
    m.same_arrays(a,b)
    b['mask'][0,0]=False
    with pytest.raises(ValueError,match='PARENT_ARRAY_CHANGED'):m.same_arrays(a,b)


def test_model_change_is_only_output_list_for_plus18():
    m=module();c={'horizons_hours':m.SHORT,'slots':13,'market_time':{'observations':13,'span_open_market_hours':24}}
    d=m.model_config(c,'plus18');assert d['horizons_hours']==m.SHORT+[18]
    d['horizons_hours']=m.SHORT;assert d==c


def test_window_coverage_includes_current_tick():
    m=module();c={'horizons_hours':m.SHORT,'slots':13,'market_time':{'observations':13,'span_open_market_hours':24}}
    for n,span in ((7,12),(13,24),(25,48)):
        r=m.model_config(c,'window'+str(n));assert r['slots']==n
        assert r['market_time']['span_open_market_hours']==span


def test_finite_queue_and_conditional_single18():
    m=module();assert m.next_cell({},False)=='plus18'
    plus={'state':'COMPLETE','metrics':{'validation':[{'horizon':h,'relative_error':-.1} for h in m.SHORT+[18]]},'parent_short_mean':-.05}
    assert m.need_single(plus) is False
    plus['metrics']['validation'][-1]['relative_error']=.1
    assert m.need_single(plus) is True
    done={c:{'state':'COMPLETE'} for c in m.CELLS};assert m.next_cell(done,False) is None


def test_file_corruption_rejected(tmp_path):
    m=module();(tmp_path/'a').write_text('a');manifest={'a':m.file_sha256(tmp_path/'a')}
    m.authenticate(tmp_path,manifest);(tmp_path/'a').write_text('b')
    with pytest.raises(ValueError,match='DIGEST'):m.authenticate(tmp_path,manifest)


def test_subset_preserves_scalar_grid_metadata_and_string_row_ids():
    m=module();a=arrays();a["row_ids"]=np.array(["a","b"]);a["spacing_hours"]=2.
    b=m.subset(a,np.array([True,False]))
    assert b["spacing_hours"]==2. and b["row_ids"].tolist()==["a"]
    m.same_arrays(a,copy.deepcopy(a))


def test_window_metadata_matches_new_shape_and_includes_origin():
    from types import SimpleNamespace
    m=module();grid=np.arange(100,dtype="int64")*3600
    a=arrays();a["row_id"]=grid[[30,31]];a["closure_crossed"]=np.zeros((2,13),dtype="float64")
    b,ok=m.window_arrays(a,SimpleNamespace(ts=grid,x=np.ones((100,19))),grid,7)
    assert ok.all() and b["slot_instants"].shape==(2,7)
    assert np.array_equal(b["slot_instants"][:,-1],a["row_id"])
    assert (b["offsets_hours"][:,-1]==0).all() and b["closure_crossed"].dtype==a["closure_crossed"].dtype


def test_legacy_trainer_and_layer_builders_are_unchanged():
    import ast
    import subprocess
    root=Path(__file__).parents[2]
    old=subprocess.check_output(["git","show","796ec8f4:tools/i6dmo/train.py"],cwd=root)
    assert (root/"tools/i6dmo/train.py").read_bytes()==old
    path="tools/i6dmo/model.py";old=subprocess.check_output(["git","show","796ec8f4:"+path],cwd=root,text=True)
    def definitions(s):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!="_normalize_market_config"}
    assert definitions(old)==definitions((root/path).read_text())
