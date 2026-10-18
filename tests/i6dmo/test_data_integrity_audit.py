import numpy as np
import pandas as pd
import pytest

from tools.i6dmo.data_integrity_audit import thresholds,describe,expected_hours,audit_arrays


def calendar():
    weeks=[]
    for s in pd.date_range('2019-12-29T22:00Z','2025-12-28T22:00Z',freq='7D'):
        weeks.append(dict(market_open=s.isoformat(),market_close=(s+pd.Timedelta(days=5)).isoformat()))
    return dict(market_calendar=dict(weeks=weeks))


def fixture():
    c=calendar();grid=expected_hours(c)
    grid=grid[(grid.year>=2020)&(grid.year<2026)]
    ts=grid.as_unit('s').asi8
    x=np.sin(np.arange(len(ts))[:,None]/17)
    return ts,x,np.ones(len(ts)),['feature'],c


def test_weekends_not_misreported_as_holes_and_no_full_certification():
    report=audit_arrays(*fixture())
    assert report['blockers']==[]
    assert report['training_authorized'] is False
    assert len(report['feature_monthly'])==72


def test_missing_month_refused_even_if_remaining_values_all_finite():
    ts,x,p,n,c=fixture();idx=pd.to_datetime(ts,unit='s',utc=True)
    keep=~((idx.year==2023)&(idx.month==4))
    report=audit_arrays(ts[keep],x[keep],p[keep],n,c)
    rows=[r for r in report['monthly'] if r['month']=='2023-04']
    assert rows[0]['observed_source_rows']==0 and rows[0]['missing_open_hours']>400
    assert report['state']=='DATA_INTEGRITY_HOLD'


def test_one_missing_market_hour_and_nonfinite_input_are_distinct():
    ts,x,p,n,c=fixture();x=x.copy();x[1]=np.nan
    report=audit_arrays(ts[1:],x[1:],p[1:],n,c)
    assert {b['code'] for b in report['blockers']}=={'UNRESOLVED_OPEN_SESSION_GAPS','NONFINITE_SELECTED_INPUT'}


def test_outliers_remain_unmodified_and_zero_mad_not_reported_as_zero_outliers():
    a=np.array([-1.,0.,1.,2.,1e9]);copy=a.copy()
    fit=thresholds([-1.,0.,1.,2.])
    assert describe(a,fit)['outliers_modified_z']==1
    np.testing.assert_array_equal(a,copy)
    assert describe([1.,100.],thresholds([1.,1.,1.]))['outliers_modified_z'] is None


def test_validation_perturbation_does_not_change_outlier_threshold():
    ts,x,p,n,c=fixture();first=audit_arrays(ts,x,p,n,c)
    x=x.copy();x[ts>=pd.Timestamp('2024-01-01T00:00Z').timestamp()]*=1e9
    second=audit_arrays(ts,x,p,n,c)
    assert first['features'][0]['outlier_fit']==second['features'][0]['outlier_fit']


@pytest.mark.parametrize('change',['duplicate','reorder','wrong_shape'])
def test_invalid_alignment_refused(change):
    ts,x,p,n,c=fixture();ts=ts.copy()
    if change=='duplicate':ts[1]=ts[0]
    elif change=='reorder':ts[[0,1]]=ts[[1,0]]
    else:x=x[:1]
    with pytest.raises(ValueError):audit_arrays(ts,x,p,n,c)


def test_calendar_truncation_cannot_hide_missing_months():
    ts,x,p,n,c=fixture();c['market_calendar']['weeks']=c['market_calendar']['weeks'][:1]
    with pytest.raises(ValueError,match='CALENDAR_DOES_NOT_COVER'):audit_arrays(ts,x,p,n,c)
