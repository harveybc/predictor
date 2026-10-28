from datetime import datetime,timezone
import numpy as np
import pytest

from tools.i6dmo import weekly_test_data as T,weekly_test_source as S,market_grid as G,data
from tools.i6n_contracts import eligibility as E
from tests.i6dmo import _synth


def test_weekly_periods_cover_2025_and_four_years():
    periods=T.periods(S.calendar(),2025)
    assert len(periods)==53
    assert periods[0]['week_ids']==['2024-12-29']
    assert periods[-1]['week_ids']==['2025-12-28']
    for p in periods:
        a,b=(data._instant(p[k]) for k in ('train_start','train_end'))
        assert a==b.replace(year=b.year-4)


def test_naive_is_own_horizon_mask_and_original_scale():
    a=dict(mask=np.array([[True,False],[True,True]]),target_price=np.array([[1.1,np.nan],[1.3,1.9]]),origin_price=np.array([1.,1.2]))
    assert T.naive(a)==pytest.approx([.1,.7])


def test_naive_does_not_score_missing_head():
    with pytest.raises(ValueError,match='FIT_HEAD_UNSUPPORTED'):
        T.naive(dict(mask=np.zeros((2,1),bool),target_price=np.zeros((2,1)),origin_price=np.ones(2)))


def test_default_market_grid_still_refuses_test_bounds(tmp_path):
    path=_synth.make_input(tmp_path/'input')
    design=_synth.load_design();bundle=data.load_input(path,design)
    split=E.SplitBounds('FIT',datetime(2024,1,1,tzinfo=timezone.utc),T.END,label_available_by=T.END)
    with pytest.raises(data.DataContractError,match='EVALUATION_SPLITS_INVALID'):
        G.build_market_table(bundle,design,evaluation_splits=(split,))
    with pytest.raises(data.DataContractError,match='AUTHORIZED_EVALUATION_END_INVALID'):
        G.build_market_table(bundle,design,evaluation_splits=(split,),authorized_evaluation_end=datetime(2027,1,1,tzinfo=timezone.utc))
    table=G.build_market_table(bundle,design,evaluation_splits=(split,),authorized_evaluation_end=T.END)
    assert table.identity['owner_authorized_test_end']==T.END.isoformat()
