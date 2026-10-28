import copy
import sys

import numpy as np
import pytest

from tools.i6dmo import weekly_test as T,weekly18 as W


def test_operations_does_not_import_tensorflow():
    import subprocess
    result=subprocess.run([sys.executable,'-c',
        "import sys;import tools.i6dmo.weekly_test;assert 'tensorflow' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_test_report_never_claims_validation_or_unread():
    spec=dict(batch_id=T.NAME,code_commit='a'*40)
    result=dict(state='FAILED',cell=dict(id='short__week',family='short'),sha256='b'*64)
    report=T.report(spec,result)
    assert report['tags']['test_read']=='true'
    assert report['tags']['evaluation']==T.D.QUALIFICATION
    assert {r['split'] for r in report['metrics']}=={'test'}


def test_annual_pools_own_horizons_and_rejects_overlap(tmp_path):
    cells=[];results={}
    for family,heads in [('short',T.D.SHORT),('long',T.D.LONG)]:
        for week in range(2):
            cid=family+str(week);cell=dict(id=cid,family=family);cells.append(cell)
            folder=tmp_path/'results'/cid;folder.mkdir(parents=True)
            np.savez(folder/'TEST.npz',origin_ts=np.array([week+1]),mask=np.ones((1,len(heads)),bool))
            results[cid]=dict(state='COMPLETE',metrics=dict(TEST=[dict(horizon=h,n=1,
                sum_abs_error=h*.9,sum_abs_naive_error=float(h)) for h in heads]))
    yearly=T.annual(dict(cells=cells),tmp_path,results)
    for family in ('short','long'):
        assert yearly['families'][family]['all_heads_beat_naive']
        for r in yearly['families'][family]['metrics']:
            assert r['relative_error']==pytest.approx(-.1)
            assert r['naive_mae']==r['horizon']
    np.savez(tmp_path/'results'/'short1'/'TEST.npz',origin_ts=np.array([1]),mask=np.ones((1,7),bool))
    with pytest.raises(ValueError,match='WEEK_OVERLAP'):T.annual(dict(cells=cells),tmp_path,results)


def test_recipe_changes_fail_closed():
    spec=dict(training=W.B.TRAINING,seed=W.B.SEED,evaluation_split='TEST',code_files={},qualification=T.D.QUALIFICATION)
    T.verify(spec)
    for key,value in [('seed',1),('training',{}),('evaluation_split','VALIDATION'),('qualification','OOS')]:
        bad=copy.deepcopy(spec);bad[key]=value
        with pytest.raises(ValueError):T.verify(bad)


def test_training_call_and_trainer_bytes_match_retained_recipe():
    import ast
    import subprocess
    base='0a2f26e9'
    def call(source):
        tree=ast.parse(source)
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
            and n.func.value.id=='T' and n.func.attr=='fit_dmo']
    path='tools/i6dmo/continuation.py'
    before=subprocess.check_output(['git','show',base+':'+path],cwd=W.REPO).decode()
    assert call((W.REPO/path).read_text())==call(before)
    for path in ('tools/i6dmo/train.py','tools/i6dmo/model.py','tools/i6dmo/continuation_models.py'):
        assert (W.REPO/path).read_bytes()==subprocess.check_output(['git','show',base+':'+path],cwd=W.REPO)
