"""Acquire independent observed bid/ask quotes; never splice into a live lake."""
import argparse
import gzip
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


def acquire(root, pairs, year, weeks):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    status = dict(schema='i6dmo.fxcm_recovery.v1', state='ACQUIRING',
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
                path = folder/f'{year}-{week:02d}.csv.gz'
                url = f'{BASE}/m1/{pair.upper()}/{year}/{week}.csv.gz'
                try:
                    if not path.exists():
                        r = session.get(url, timeout=(15,45))
                        if r.status_code == 429:
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
                    # FXCM file weeks use Sunday boundaries, not ISO week IDs.
                    if frame.index.min().year < year-1 or frame.index.max().year > year+1:
                        raise ValueError('WRONG_SOURCE_YEAR')
                    frames.append(frame)
                    status['results'].append(dict(pair=pair,year=year,week=week,
                                                  minutes=len(frame),sha256=sha(path)))
                    status['complete'] += 1
                except Exception as exc:
                    status['failures'].append(dict(job=key,error_type=type(exc).__name__))
                    if str(exc) == 'PROVIDER_RATE_LIMIT_STOP':
                        status.update(state='RATE_LIMITED',current=key)
                        put(root/'STATUS.json', status)
                        return status
                put(root/'STATUS.json', status)
                time.sleep(.15)
            if frames:
                all_rows = pd.concat(frames).sort_index()
                if all_rows.index.has_duplicates:
                    raise ValueError('CROSS_FILE_DUPLICATE_QUOTES')
                for period, hours in [('1min',False),('5min',False),('1h',True)]:
                    if period == '1min':
                        df = all_rows.copy()
                    else:
                        agg = {side+k:v for side in ('Bid','Ask')
                               for k,v in [('Open','first'),('High','max'),('Low','min'),('Close','last')]}
                        df = all_rows.resample(period).agg(agg).dropna()
                    # Store observed provider bar starts, plus explicit availability.
                    df = df.reset_index()
                    df['available_at'] = df.timestamp + pd.Timedelta(period)
                    target = folder/f'{year}-{period}.parquet'
                    tmp = target.with_suffix('.partial')
                    df.to_parquet(tmp,index=False)
                    os.replace(tmp,target)
                h = pd.read_parquet(folder/f'{year}-1h.parquet')
                report = dict(pair=pair,year=year,minutes=len(all_rows),hours=len(h),
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
    p.add_argument('--weeks',nargs='+',type=int,default=list(range(1,53)))
    p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.status:
        print((Path(a.output)/'STATUS.json').read_text())
        return
    r=acquire(a.output,a.pairs,a.year,a.weeks)
    raise SystemExit(0 if r['state']=='ACQUIRED' else 2)


if __name__=='__main__':
    main()
