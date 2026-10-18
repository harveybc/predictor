"""Recover observed HistData quotes without modifying a lake or filling gaps.

Acquisition only: candidates are NOT admitted training datasets. The provider
documents its ASCII clock as fixed EST (UTC-5), without daylight saving time.
Minute timestamps denote bar starts; derived hour timestamps denote bar ends.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = 'https://www.histdata.com'
MAX_ZIP = 64 * 1024**2
MAX_CSV = 160 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def minute_frame(archive, pair, year, month=None):
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > MAX_CSV:
            raise ValueError('ARCHIVE_TOO_LARGE')
        names = [n for n in z.namelist() if n.upper().endswith('.CSV')
                 and pair.upper() in Path(n).name.upper()]
        if len(names) != 1 or z.testzip() is not None:
            raise ValueError('ARCHIVE_IDENTITY_OR_CRC')
        with z.open(names[0]) as stream:
            df = pd.read_csv(stream, sep=';', header=None,
                             names=['time', 'open', 'high', 'low', 'close', 'volume'])
    local = pd.to_datetime(df.pop('time'), format='%Y%m%d %H%M%S', errors='raise')
    if not (local.dt.year == year).all() or (month and not (local.dt.month == month).all()):
        raise ValueError('ARCHIVE_PERIOD_MISMATCH')
    if df.empty or df.isna().any().any():
        raise ValueError('MISSING_QUOTE_VALUES')
    import numpy as np
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('NONFINITE_QUOTE_VALUES')
    if ((df[['open','high','low','close']] <= 0).any().any()
            or (df.high < df[['open','close','low']].max(axis=1)).any()
            or (df.low > df[['open','close','high']].min(axis=1)).any()):
        raise ValueError('INVALID_OHLC')
    df.index = (pd.DatetimeIndex(local).tz_localize(timezone.utc)
                + pd.Timedelta(hours=5))
    df.index.name = 'timestamp'
    duplicated = df.index.duplicated(keep=False)
    if duplicated.any():
        for _, rows in df[duplicated].groupby(level=0):
            if len(rows.drop_duplicates()) != 1:
                raise ValueError('CONFLICTING_DUPLICATE_QUOTES')
    duplicate_rows = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated()].sort_index()
    return df, duplicate_rows


def download(session, pair, year, month, target):
    if target.exists():
        return
    period = str(year) + (f'/{month}' if month else '')
    page = BASE + f'/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{period}'
    r = session.get(page, timeout=(15, 45))
    r.raise_for_status()
    form = BeautifulSoup(r.text, 'html.parser').find('form', action='/get.php')
    if form is None:
        raise ValueError('DOWNLOAD_FORM_ABSENT')
    fields = {i['name']: i.get('value', '') for i in form.find_all('input', attrs={'name': True})}
    if (fields.get('fxpair','').lower() != pair or fields.get('date') != str(year)
            or fields.get('timeframe') != 'M1' or fields.get('platform') != 'ASCII'):
        raise ValueError('DOWNLOAD_FORM_IDENTITY')
    tmp = target.with_suffix('.partial')
    try:
        with session.post(BASE + '/get.php', data=fields, headers={'Referer': page},
                          timeout=(15, 60), stream=True) as r:
            r.raise_for_status()
            size = 0
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024**2):
                    size += len(chunk)
                    if size > MAX_ZIP:
                        raise ValueError('DOWNLOAD_TOO_LARGE')
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
        minute_frame(tmp, pair, year, month)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def recover(root, pairs, years, months=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(pair, year, month) for pair in pairs for year in years for month in (months or [None])]
    status = dict(schema='i6dmo.quote_recovery.v1', state='ACQUIRING',
                  qualification='SOURCE_CANDIDATES_NOT_ADMITTED_FOR_TRAINING',
                  total=len(jobs), complete=0, results=[], failures=[],
                  clock='HistData fixed EST +5h => UTC; hourly bar-END +1h',
                  timestamp_spec=BASE + '/f-a-q/data-files-detailed-specification/')
    with requests.Session() as session:
        for pair, year, month in jobs:
            key = f'{pair}-{year}' + (f'-{month:02d}' if month else '')
            status.update(current=key, at=datetime.now(timezone.utc).isoformat())
            put(root/'STATUS.json', status)
            try:
                folder = root/key
                folder.mkdir(exist_ok=True)
                raw = folder/'source.zip'
                download(session, pair, year, month, raw)
                df, duplicates = minute_frame(raw, pair, year, month)
                hourly = df.resample('1h', closed='left', label='left').agg(
                    dict(open='first', high='max', low='min', close='last', volume='sum'))
                hourly = hourly.dropna(subset=['open','high','low','close'])
                hourly.index += pd.Timedelta(hours=1)
                minute_counts = df.resample('1h').size()
                hourly['observed_minutes'] = minute_counts.reindex(hourly.index-pd.Timedelta(hours=1)).to_numpy()
                for name, frame in [('minute.parquet', df), ('hourly.parquet', hourly)]:
                    path = folder/name
                    if not path.exists():
                        tmp = path.with_suffix('.partial')
                        frame.reset_index().to_parquet(tmp, index=False)
                        os.replace(tmp, path)
                    else:
                        pd.testing.assert_frame_equal(pd.read_parquet(path), frame.reset_index())
                report = dict(pair=pair, year=year, month=month, minutes=len(df), hours=len(hourly),
                              duplicates_identical_removed=duplicates,
                              monthly_hours={str(k):int(v) for k,v in hourly.groupby(hourly.index.strftime('%Y-%m')).size().items()},
                              incomplete_hours=int((hourly.observed_minutes < 60).sum()),
                              files={p.name:sha(p) for p in (raw,folder/'minute.parquet',folder/'hourly.parquet')},
                              clock=status['clock'], source='HistData ASCII M1', gaps_filled=0)
                manifest = folder/'MANIFEST.json'
                if manifest.exists() and json.loads(manifest.read_text()) != report:
                    raise ValueError('RECOVERED_MANIFEST_CHANGED')
                put(manifest, report)
                status['results'].append(report)
                status['complete'] += 1
                print(json.dumps({k:report[k] for k in ('pair','year','month','minutes','hours','monthly_hours')}), flush=True)
            except Exception as exc:
                status['failures'].append(dict(job=key, error_type=type(exc).__name__))
            put(root/'STATUS.json', status)
    status.update(state='ACQUIRED' if not status['failures'] else 'INCOMPLETE', current=None)
    put(root/'STATUS.json', status)
    return status


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--pairs', nargs='+', choices=['eurusd','gbpusd'], default=['eurusd','gbpusd'])
    p.add_argument('--years', nargs='+', type=int, default=[2023])
    p.add_argument('--months', nargs='+', type=int, choices=range(1,13))
    p.add_argument('--status', action='store_true')
    a = p.parse_args()
    if a.status:
        print((Path(a.output)/'STATUS.json').read_text())
        return
    result = recover(a.output, a.pairs, a.years, a.months)
    raise SystemExit(0 if result['state']=='ACQUIRED' else 2)


if __name__ == '__main__':
    main()
