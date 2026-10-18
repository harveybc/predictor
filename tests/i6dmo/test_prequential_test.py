import copy
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import prequential_test as S,weekly18 as W


def arrays():
    return dict(x=np.ones((2,13,1)),offsets_hours=np.tile(np.arange(-24,1,2),(2,1)),
        closure_crossed=np.zeros((2,13),bool),origin_ts=np.array([100,200]),
        row_ids=np.array([100,200]),origin_price=np.array([1.,2.]))


class Backend:
    def __init__(self,events):self.events=events
    def predict(self,donor,a):
        assert set(a)==S.PREDICT_KEYS
        self.events.append('predict')
        return a['origin_price'][:,None]+.1
    def update(self,donor,fit,observed,record,folder):
        assert (folder.parent/'PREDICTION.json').is_file()
        self.events.append('update')
        assert donor['id']=='validation-donor'
        return dict(next_checkpoint=dict(id='updated',root='worker',files={}),
            epochs=12,best_epoch=2,stop_reason='patience',training_seconds=1.,parameters=1)


def fixture(tmp_path,target_shift=0.):
    events=[];a=arrays();o=dict(a,mask=np.ones((2,1),bool),target_price=a['origin_price'][:,None]+.2+target_shift)
    record=dict(cell=dict(id='week',horizons=[2],period=dict(train_start='2021-01-03T22:00:00+00:00',
        train_end='2025-01-03T22:00:00+00:00',validation_end='2025-01-10T22:00:00+00:00')),
        model=dict(horizons_hours=[2]),scales=dict(input_mean=[9.],input_std=[2.],naive_mae=[.2]),
        naive_cache={},population={})
    donor=dict(id='validation-donor',available_by='1970-01-01T00:00:00+00:00',
        scales=dict(input_mean=[1.],input_std=[1.],naive_mae=[.1]))
    def reader(name):
        if name=='OBSERVED':assert (tmp_path/'PREDICTION.json').is_file()
        events.append('read:'+name)
        return copy.deepcopy(a if name=='PREDICT' else o)
    return events,record,donor,reader


def test_predict_seal_score_then_update(tmp_path):
    e,r,d,reader=fixture(tmp_path)
    result=S.execute_week(r,d,tmp_path,reader,Backend(e),input_sha256='input')
    assert e==['read:PREDICT','predict','read:OBSERVED','read:FIT','update']
    assert result['metrics']['TEST'][0]['mae']==pytest.approx(.1)
    assert result['next_checkpoint']['id']=='updated'


def test_future_targets_do_not_change_predictions(tmp_path):
    forecasts=[]
    for i,shift in enumerate((0.,100.)):
        root=tmp_path/str(i);root.mkdir()
        e,r,d,reader=fixture(root,target_shift=shift)
        S.execute_week(r,d,root,reader,Backend(e),input_sha256='input')
        forecasts.append(np.load(root/'PREDICTED.npz')['prediction'])
    np.testing.assert_array_equal(*forecasts)


def test_resume_never_recomputes_or_replaces_sealed_prediction(tmp_path):
    e,r,d,reader=fixture(tmp_path)
    S.execute_week(r,d,tmp_path,reader,Backend(e),input_sha256='input')
    old=W.file_sha256(tmp_path/'PREDICTED.npz');e.clear()
    S.execute_week(r,d,tmp_path,reader,Backend(e),input_sha256='input')
    assert 'predict' not in e
    assert W.file_sha256(tmp_path/'PREDICTED.npz')==old
    d['id']='different'
    with pytest.raises(ValueError,match='SEALED_FORECAST_IDENTITY'):
        S.execute_week(r,d,tmp_path,reader,Backend(e),input_sha256='input')


def test_inference_uses_donor_scaler_and_no_target():
    a=arrays();scales=dict(input_mean=[1.],input_std=[2.],naive_mae=[.1])
    result=S.inference_inputs(a,scales)
    np.testing.assert_array_equal(result[0],np.zeros_like(a['x']))
    a['target_price']=np.zeros((2,1))
    with pytest.raises(ValueError,match='PREDICT_INPUT_KEYS'):S.inference_inputs(a,scales)


def test_future_donor_is_rejected_before_predict(tmp_path):
    e,r,d,reader=fixture(tmp_path);d['available_by']='2025-01-01T00:00:00+00:00'
    with pytest.raises(ValueError,match='DONOR_NOT_YET_AVAILABLE'):
        S.execute_week(r,d,tmp_path,reader,Backend(e),input_sha256='input')
    assert 'predict' not in e


def test_original_training_and_callback_order_are_preserved():
    assert S.TRAINING==dict(batch_size=256,learning_rate=.001,weight_decay=.0001,
        patience=10,min_delta=1e-8,max_epochs=200)
    assert S.QUALIFICATION=='PREQUENTIAL_TEST_PREDICT_THEN_UPDATE'


def test_operations_is_cpu_only():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-c',
        "import sys;import tools.i6dmo.prequential_test;assert 'tensorflow' not in sys.modules"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_refit_is_four_years_before_observed_week_with_mature_labels():
    stamp=lambda v:int(S.CD.data._instant(v).timestamp())
    p=dict(train_start='2021-01-03T22:00:00+00:00',train_end='2025-01-03T22:00:00+00:00',
        validation_end='2025-01-10T22:00:00+00:00')
    fit=dict(origin_ts=np.array([stamp('2024-01-01T00:00:00+00:00')]),mask=np.ones((1,1),bool))
    observed=dict(origin_ts=np.array([stamp('2025-01-06T00:00:00+00:00')]),mask=np.ones((1,1),bool))
    S.validate_update_span(p,fit,observed,[2])
    bad=dict(p,train_start='2020-01-03T22:00:00+00:00')
    with pytest.raises(ValueError,match='NOT_FOUR_CALENDAR_YEARS'):S.validate_update_span(bad,fit,observed,[2])
    future=dict(origin_ts=np.array([stamp(p['train_end'])]),mask=np.ones((1,1),bool))
    with pytest.raises(ValueError,match='TRAIN_CUTOFF'):S.validate_update_span(p,future,observed,[2])
    with pytest.raises(ValueError,match='OBSERVED_WEEK_NOT_FINISHED'):S.validate_update_span(p,fit,observed,[120])


def test_scheduling_context_restores_previous_campaign_globals():
    name=S.P.NAME;qualification=S.P.D.QUALIFICATION
    with S.context():
        assert S.P.NAME==S.NAME
        assert S.P.MODULE=='tools.i6dmo.prequential_test'
        assert S.P.D.QUALIFICATION==S.QUALIFICATION
    assert S.P.NAME==name and S.P.D.QUALIFICATION==qualification


def test_trainer_and_models_have_not_changed():
    import subprocess
    for rel in ('tools/i6dmo/train.py','tools/i6dmo/continuation_models.py','tools/i6dmo/model.py'):
        assert (W.REPO/rel).read_bytes()==subprocess.check_output(['git','show','658a7733:'+rel],cwd=W.REPO)
