"""Acquire independent observed bid/ask quotes; never splice into a live lake."""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from tools.i6dmo.recover_fx_quotes import put, sha

BASE = 'https://candledata.fxcorporate.com'
SPEC = 'https://github.com/fxcm/MarketData/blob/master/CandleData/README.md'


def parse(raw):
    if len(raw) > 8*1024**2:
        raise ValueError('DOWNLOAD_TOO_LARGE')
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
        decoded = f.read(32*1024**2+1)
    if len(decoded) > 32*1024**2:
        raise ValueError('DECODED_TOO_LARGE')
    df = pd.read_csv(io.BytesIO(decoded))
    columns = [side+k for side in ('Bid','Ask') for k in ('Open','High','Low','Close')]
    if set(df.columns) != {'DateTime', *columns} or df.empty:
        raise ValueError('QUOTE_SCHEMA')
    ts = pd.to_datetime(df.pop('DateTime'), format='%m/%d/%Y %H:%M:%S.%f', utc=True)
    df.index = pd.DatetimeIndex(ts, name='timestamp')
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('NONFINITE_OR_NONPOSITIVE_QUOTE')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('QUOTE_ORDER')
    for side in ('Bid','Ask'):
        if ((df[side+'High'] < df[[side+'Open',side+'Low',side+'Close']].max(axis=1)).any()
                or (df[side+'Low'] > df[[side+'Open',side+'High',side+'Close']].min(axis=1)).any()):
            raise ValueError('INVALID_OHLC')
    if (df.AskClose < df.BidClose).any() or (df.AskOpen < df.BidOpen).any():
        raise ValueError('CROSSED_QUOTE')
    return df


def acquire(root, pairs, year, weeks, periodicity='m1'):
    if periodicity not in ('m1','H1'):
        raise ValueError('UNSUPPORTED_NATIVE_PERIODICITY')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    scope=dict(pairs=list(pairs),year=year,weeks=list(weeks),periodicity=periodicity)
    scope_path=root/'ACQUISITION_IDENTITY.json'
    if scope_path.exists():
        if json.loads(scope_path.read_text())!=scope:raise ValueError('RECOVERY_SCOPE_CHANGED')
    elif (root/'STATUS.json').exists():raise ValueError('LEGACY_RECOVERY_SCOPE_NOT_SEALED')
    else:put(scope_path,scope)
    status = dict(schema='i6dmo.fxcm_recovery.v2', state='ACQUIRING',periodicity=periodicity,
                  qualification='INDEPENDENT_SOURCE_CANDIDATES_NOT_ADMITTED',
                  source_spec=SPEC, total=len(pairs)*len(weeks), complete=0,
                  results=[], failures=[], gaps_filled=0)
    with requests.Session() as session:
        for pair in pairs:
            frames = []
            for week in weeks:
                key = f'{pair}-{year}-{week:02d}'
                status['current'] = key
                put(root/'STATUS.json', status)
                folder = root/pair
                folder.mkdir(exist_ok=True)
                suffix='' if periodicity=='m1' else '.H1'
                path = folder/f'{year}-{week:02d}{suffix}.csv.gz'
                url = f'{BASE}/{periodicity}/{pair.upper()}/{year}/{week}.csv.gz'
                try:
                    if not path.exists():
                        r = session.get(url, timeout=(15,45))
                        if r.status_code == 429:
                            value=r.headers.get('Retry-After','3600')
                            try:delay=float(value)
                            except ValueError:
                                try:delay=(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
                                except (ValueError,TypeError):delay=3600
                            status['retry_not_before_epoch']=time.time()+max(60,delay)
                            raise ValueError('PROVIDER_RATE_LIMIT_STOP')
                        r.raise_for_status()
                        parse(r.content)
                        tmp = path.with_suffix('.partial')
                        with tmp.open('wb') as f:
                            f.write(r.content)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp, path)
                    frame = parse(path.read_bytes())
                    if periodicity=='H1' and ((frame.index.minute!=0).any()
                            or (frame.index.second!=0).any() or (frame.index.microsecond!=0).any()):
                        raise ValueError('HOURLY_TIMESTAMP_ALIGNMENT')
                    # FXCM file weeks use Sunday boundaries, not ISO week IDs.
                    if frame.index.min().year < year-1 or frame.index.max().year > year+1:
                        raise ValueError('WRONG_SOURCE_YEAR')
                    frames.append(frame)
                    status['results'].append(dict(pair=pair,year=year,week=week,
                                                  minutes=len(frame),sha256=sha(path)))
                    status['complete'] += 1
                except Exception as exc:
                    status['failures'].append(dict(job=key,error_type=type(exc).__name__,reason=str(exc)))
                    if str(exc) == 'PROVIDER_RATE_LIMIT_STOP':
                        status.update(state='RATE_LIMITED',current=key)
                        put(root/'STATUS.json', status)
                        return status
                put(root/'STATUS.json', status)
                time.sleep(1 if periodicity=='H1' else .15)
            if frames:
                all_rows = pd.concat(frames).sort_index()
                if all_rows.index.has_duplicates:
                    raise ValueError('CROSS_FILE_DUPLICATE_QUOTES')
                periods=['1h'] if periodicity=='H1' else ['1min','5min','1h']
                for period in periods:
                    if periodicity=='H1' or period == '1min':
                        df = all_rows.copy()
                    else:
                        agg = {side+k:v for side in ('Bid','Ask')
                               for k,v in [('Open','first'),('High','max'),('Low','min'),('Close','last')]}
                        df = all_rows.resample(period).agg(agg).dropna()
                    # Store observed provider bar starts, plus explicit availability.
                    df = df.reset_index()
                    df['available_at'] = df.timestamp + pd.Timedelta(period)
                    target = folder/f'{year}-{period}.parquet'
                    if target.exists():
                        previous=pd.read_parquet(target).set_index('timestamp')
                        observed=df.set_index('timestamp')
                        if not previous.index.isin(observed.index).all():raise ValueError('RECOVERED_FRAME_CHANGED')
                        try:pd.testing.assert_frame_equal(previous,observed.loc[previous.index])
                        except AssertionError as exc:raise ValueError('RECOVERED_FRAME_CHANGED') from exc
                        if len(previous)<len(observed):
                            tmp = target.with_suffix('.partial');df.to_parquet(tmp,index=False);os.replace(tmp,target)
                    else:
                        tmp = target.with_suffix('.partial')
                        df.to_parquet(tmp,index=False)
                        os.replace(tmp,target)
                h = pd.read_parquet(folder/f'{year}-1h.parquet')
                report = dict(pair=pair,year=year,minutes=len(all_rows) if periodicity=='m1' else None,hours=len(h),
                              native_rows=len(all_rows),periodicity=periodicity,
                              native_hourly_ohlc=periodicity=='H1',
                              acquisition_scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest(),
                              monthly_hours={str(k):int(v) for k,v in h.groupby(h.timestamp.dt.strftime('%Y-%m')).size().items()},
                              files={p.name:sha(p) for p in folder.glob('*.parquet')},
                              source=SPEC,clock='UTC bar START; available_at = bar end',
                              gaps_filled=0,qualification=status['qualification'])
                put(folder/'MANIFEST.json',report)
                print(json.dumps(report),flush=True)
    status.update(state='ACQUIRED' if not status['failures'] else 'INCOMPLETE',current=None)
    put(root/'STATUS.json',status)
    return status


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--pairs',nargs='+',choices=['eurusd','gbpusd'],default=['eurusd','gbpusd'])
    p.add_argument('--year',type=int,default=2023)
    p.add_argument('--periodicity',choices=['m1','H1'],default='m1')
    p.add_argument('--weeks',nargs='+',type=int,default=list(range(1,53)))
    p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.status:
        print((Path(a.output)/'STATUS.json').read_text())
        return
    r=acquire(a.output,a.pairs,a.year,a.weeks,a.periodicity)
    raise SystemExit(0 if r['state']=='ACQUIRED' else 2)


if __name__=='__main__':
    main()
