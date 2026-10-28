"""Read-only calendar coverage and TRAIN-fitted outlier audit of selected inputs.

This is a rejection gate, not a source-availability certificate. Outliers are
diagnostic flags only: no row, feature, price or target is changed or removed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.i6dmo.recover_fx_quotes import put, sha


def expected_hours(calendar):
    frames = []
    for week in calendar['market_calendar']['weeks']:
        a, b = pd.Timestamp(week['market_open']), pd.Timestamp(week['market_close'])
        if a.tzinfo is None or b.tzinfo is None or a >= b:
            raise ValueError('CALENDAR_INTERVAL')
        frames.append(pd.date_range(a.tz_convert('UTC')+pd.Timedelta(hours=1),
                                    b.tz_convert('UTC'),freq='1h'))
    if not frames:
        raise ValueError('EMPTY_CALENDAR')
    grid = frames[0].append(frames[1:])
    if grid.has_duplicates or not grid.is_monotonic_increasing:
        raise ValueError('CALENDAR_ORDER')
    return grid


def thresholds(train):
    finite = np.asarray(train,dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return dict(state='NO_FINITE_TRAIN',median=None,mad=None,iqr=None)
    q1, median, q3 = np.quantile(finite,[.25,.5,.75])
    mad = float(np.median(np.abs(finite-median)))
    return dict(state='READY' if mad>0 else 'ZERO_MAD_NOT_ESTIMABLE',
                median=float(median),mad=mad,iqr=float(q3-q1),q1=float(q1),q3=float(q3),
                modified_z_threshold=3.5,reference='TRAIN_2020_2023_ONLY')


def describe(values, fit):
    a = np.asarray(values,dtype=float)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return dict(rows=len(a),finite=0,nonfinite=len(a),constant=None,
                    outliers_modified_z=None,outliers_iqr=None,quantiles=None)
    z = (int((.6744897501960817*np.abs(finite-fit['median'])/fit['mad']>3.5).sum())
         if fit['state']=='READY' else None)
    iqr = fit.get('iqr')
    fences = (int(((finite<fit['q1']-1.5*iqr)|(finite>fit['q3']+1.5*iqr)).sum())
              if iqr and iqr>0 else None)
    unique, counts = np.unique(finite,return_counts=True)
    return dict(rows=len(a),finite=len(finite),nonfinite=int((~np.isfinite(a)).sum()),
                constant=len(unique)==1,unique=len(unique),top_value_share=float(counts.max()/len(finite)),
                outliers_modified_z=z,outliers_iqr=fences,
                outlier_state=fit['state'],quantiles=[float(v) for v in np.quantile(finite,[0,.001,.01,.25,.5,.75,.99,.999,1])])


def audit_arrays(ts,x,price,names,calendar,start_year=2020,end_year=2026):
    ts,x,price = np.asarray(ts),np.asarray(x),np.asarray(price)
    if (ts.ndim!=1 or ts.dtype.kind not in 'iu' or not len(ts)
            or np.any(np.diff(ts)<=0) or np.any(ts%3600)):
        raise ValueError('SOURCE_CLOCK_NOT_ORDERED_UNIQUE_HOURLY')
    if (x.shape!=(len(ts),len(names)) or price.shape!=(len(ts),)
            or len(names)!=len(set(names))):
        raise ValueError('SOURCE_SCHEMA_OR_FEATURE_ORDER')
    index = pd.to_datetime(ts,unit='s',utc=True)
    grid = expected_hours(calendar)
    start,end = pd.Timestamp(f'{start_year}-01-01',tz='UTC'),pd.Timestamp(f'{end_year}-01-01',tz='UTC')
    grid = grid[(grid>=start)&(grid<end)]
    if not len(grid):
        raise ValueError('CALENDAR_DOES_NOT_COVER_REQUESTED_SPLITS')
    # A truncated calendar cannot silently redefine required months away.
    for year in range(start_year,end_year):
        if set(grid[grid.year==year].month)!=set(range(1,13)):
            raise ValueError('CALENDAR_DOES_NOT_COVER_REQUESTED_SPLITS')
    train = (index>=pd.Timestamp('2020-01-01',tz='UTC'))&(index<pd.Timestamp('2024-01-01',tz='UTC'))
    thresholds_by_feature = {name:thresholds(x[train,j]) for j,name in enumerate(names)}
    monthly, columns, blockers = [],[],[]
    for year in range(start_year,end_year):
        for month in range(1,13):
            expected = grid[(grid.year==year)&(grid.month==month)]
            selected = (index.year==year)&(index.month==month)
            observed = index[selected]
            missing = expected.difference(observed)
            off_grid = observed.difference(expected)
            split = 'TRAIN' if year<2024 else 'VALIDATION' if year==2024 else 'TEST'
            body = dict(month=f'{year}-{month:02d}',split=split,expected_open_hours=len(expected),
                        observed_source_rows=int(selected.sum()),missing_open_hours=len(missing),
                        observed_outside_assumed_calendar=len(off_grid),
                        first_missing=missing[0].isoformat() if len(missing) else None,
                        last_missing=missing[-1].isoformat() if len(missing) else None,
                        all_features_finite_rows=int(np.isfinite(x[selected]).all(axis=1).sum()),
                        invalid_prices=int((~np.isfinite(price[selected])|(price[selected]<=0)).sum()))
            monthly.append(body)
            if len(missing):blockers.append(dict(code='UNRESOLVED_OPEN_SESSION_GAPS',month=body['month'],count=len(missing)))
            if body['invalid_prices']:blockers.append(dict(code='INVALID_PRICE',month=body['month'],count=body['invalid_prices']))
            for j,name in enumerate(names):
                stats=describe(x[selected,j],thresholds_by_feature[name])
                columns.append(dict(feature_id=name,month=body['month'],split=split,**stats))
                if stats['nonfinite']:blockers.append(dict(code='NONFINITE_SELECTED_INPUT',month=body['month'],feature_id=name,count=stats['nonfinite']))
    summaries=[]
    for j,name in enumerate(names):
        split_stats={}
        for split,years in [('TRAIN',range(2020,2024)),('VALIDATION',[2024]),('TEST',[2025])]:
            split_stats[split]=describe(x[np.isin(index.year,list(years)),j],thresholds_by_feature[name])
        summaries.append(dict(feature_id=name,outlier_fit=thresholds_by_feature[name],splits=split_stats))
    return dict(schema='i6dmo.data_integrity_audit.v1',state='DATA_INTEGRITY_HOLD' if blockers else 'STRUCTURAL_CHECKS_PASS_NOT_FULL_ADMISSION',
                training_authorized=False,selected_features=len(names),rows=len(ts),
                splits=dict(TRAIN=[2020,2023],VALIDATION=[2024,2024],TEST=[2025,2025]),
                monthly=monthly,feature_monthly=columns,features=summaries,blockers=blockers,
                outlier_action='FLAG_ONLY_NO_DELETION_NO_CLIPPING_NO_IMPUTATION',
                unresolved=['Source event/availability clocks require independent evidence',
                            'Assumed calendar is not a broker holiday certificate',
                            'Source lineage, raw-source checks and prepared-window coverage are separate gates'],
                source_values_modified=False,test_access='DATA_QUALITY_ONLY_NO_MODEL_OR_HYPERPARAMETER_SELECTION')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--names-json',required=True)
    p.add_argument('--calendar',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    inputs={name:sha(getattr(args,name)) for name in ('source','calendar')}
    inputs['names']=sha(args.names_json)
    names=json.loads(Path(args.names_json).read_text())['features']
    with np.load(args.source,allow_pickle=False) as a:
        report=audit_arrays(a['ts'],a['x'],a['price'],names,json.loads(Path(args.calendar).read_text()))
    report['inputs_sha256']=inputs
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and json.loads(out.read_text())!=report:
        raise ValueError('IMMUTABLE_AUDIT_CHANGED_USE_NEW_OUTPUT')
    put(out,report)
    print(json.dumps(dict(state=report['state'],features=report['selected_features'],
                         monthly_rows=len(report['monthly']),feature_monthly_rows=len(report['feature_monthly']),
                         blockers=len(report['blockers']))))
    raise SystemExit(2 if report['blockers'] else 0)


if __name__=='__main__':main()
