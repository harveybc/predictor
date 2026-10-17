import copy
from datetime import datetime,timezone

import numpy as np
import pytest

from tools.i6dmo import best_registry as B


def test_zero_label_disposition_requires_pinned_arrays_and_keeps_failure(tmp_path):
    data=tmp_path/'data';folder=tmp_path/'results'/'cell';folder.mkdir(parents=True);(data/'cell').mkdir(parents=True)
    cell=dict(id='cell',horizons=[18,24]);result=dict(cell=cell,state='FAILED',
        reason='TrainDisposition:NO_SUPPORTED_HEAD: split has no valid label')
    result['sha256']=B.sha256_json(result);B.put(folder/'RESULT.json',result)
    B.put(data/'cell/PERIOD.json',dict(cell=cell,population={'counts':{'VALIDATION':[0,0]}}))
    np.savez(data/'cell/VALIDATION.npz',mask=np.zeros((2,2),dtype=bool))
    files={f'cell/{n}':B.file_sha256(data/'cell'/n) for n in ('PERIOD.json','VALIDATION.npz')}
    prepared=dict(files=files);prepared['sha256']=B.sha256_json(prepared);B.put(data/'PREPARED.json',prepared)
    proof=B.certify_empty_period({'data_root':str(data)},folder/'RESULT.json')
    assert proof['state']=='SKIPPED_NO_EVAL_ROWS' and B.read(folder/'RESULT.json')['state']=='FAILED'
    np.savez(data/'cell/VALIDATION.npz',mask=np.ones((2,2),dtype=bool))
    with pytest.raises(ValueError,match='ARTIFACT_DIGEST_MISMATCH'):
        B.certify_empty_period({'data_root':str(data)},folder/'RESULT.json')
    prepared['files']['cell/VALIDATION.npz']=B.file_sha256(data/'cell/VALIDATION.npz')
    prepared['sha256']=B.sha256_json({k:v for k,v in prepared.items() if k!='sha256'});B.put(data/'PREPARED.json',prepared)
    with pytest.raises(ValueError,match='HAS_LABELS'):
        B.certify_empty_period({'data_root':str(data)},folder/'RESULT.json')


def test_csv_availability_never_consults_future_labels():
    from tools.i6dmo.pair_release import forecast_rows
    from tools.i6n_contracts.eligibility import load_calendar
    _,weeks=load_calendar(dict(entry_delay_after_open_hours=12,force_flat_before_close_hours=12,
        market_calendar=dict(calendar_id='test',version='v1',weeks=[dict(week_id='w',
            market_open='2024-01-07T22:00:00+00:00',market_close='2024-01-12T22:00:00+00:00',
            entry_cutoff='2024-01-11T05:00:00+00:00')])))
    ts=int(datetime(2024,1,8,12,tzinfo=timezone.utc).timestamp())
    data=dict(origin_ts=np.array([ts]),origin_price=np.array([1.1]),prediction=np.ones((1,6))*1.101,
        mask=np.zeros((1,6),dtype=bool),target_price=np.full((1,6),np.nan))
    rows=forecast_rows(data,[2,3,4,5,6,12],'short','checkpoint',weeks)
    assert rows[0]['available_h2']==1
    data['target_price'][:]=100.;data['mask'][:]=True
    assert rows==forecast_rows(data,[2,3,4,5,6,12],'short','checkpoint',weeks)


def test_test_gate_cannot_be_opened_by_boolean_or_stale_weights():
    from tools.i6dmo.pair_release import test_gate
    assert test_gate({'test_authorized':True})['state']=='BLOCKED'
    assert 'POINT_IN_TIME' in str(test_gate({})['missing'])
