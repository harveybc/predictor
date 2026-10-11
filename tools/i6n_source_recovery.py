"""Bounded public HistData recovery; never reads original parquet or TEST data."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import re
import signal
import time
from urllib.parse import urljoin
import zipfile


ROOT = Path('/var/tmp/i6n-source-clock-recovery')
BASE = 'https://www.histdata.com'
MONTHS = ('202403', '201803', '201903')
MAX_BYTES = 30_000_000


def period_page(period):
    """Closed years use annual archives; explicitly reject TEST downloads."""
    if not re.fullmatch(r'\d{4}(?:\d{2})?', period) or not 2000 <= int(period[:4]) < 2025:
        raise ValueError('only pre-2025 source periods permitted')
    if len(period) == 6 and not 1 <= int(period[4:]) <= 12:
        raise ValueError('invalid source month')
    suffix = period[:4] + ('/' + str(int(period[4:])) if len(period) == 6 else '')
    return BASE + '/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/' + suffix


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    with path.open('x', encoding='ascii') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def public_form(html, month):
    period_page(month)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    if soup.select('[class*="captcha"], [id*="captcha"], input[type="password"]'):
        raise ValueError('authentication/captcha encountered; stop')
    form = soup.select_one('form#file_down')
    if form is None or form.get('method', '').upper() != 'POST':
        raise ValueError('public POST download form absent; stop')
    action = urljoin(BASE, form.get('action', ''))
    if action != BASE + '/get.php':
        raise ValueError('unexpected public download endpoint; stop')
    fields = {item['name']: item.get('value', '') for item in form.select('input[name]')}
    expected = dict(date=month[:4], datemonth=month, platform='ASCII',
                    timeframe='M1', fxpair='EURUSD')
    if any(fields.get(k) != v for k, v in expected.items()) or set(fields) != set(expected) | {'tk'}:
        raise ValueError('unexpected download fields; stop')
    return action, fields


def bounded_body(response, budget, deadline):
    if int(response.headers.get('Content-Length', '0')) > budget:
        raise ValueError('response exceeds remaining byte budget')
    data = bytearray()
    for block in response.iter_content(chunk_size=65536):
        if time.monotonic() > deadline:
            raise TimeoutError('request wall deadline exceeded')
        if len(data) + len(block) > budget:
            raise ValueError('response exceeds remaining byte budget')
        data.extend(block)
    return bytes(data)


def download(periods=None):
    import requests
    os.umask(0o077)
    ROOT.mkdir(mode=0o700, exist_ok=True)
    if ROOT.stat().st_mode & 0o077:
        raise ValueError('private directory permissions required')
    # An exclusive marker prevents accidental retries across invocations.
    periods = tuple(periods or MONTHS)
    if len(set(periods)) != len(periods):
        raise ValueError('duplicate periods')
    for period in periods:
        period_page(period)
    save_json(ROOT / 'ATTEMPT.json', {'months': periods, 'max_zip_bytes': MAX_BYTES})
    manifest = dict(status='STARTED', attempts=[], zip_bytes=0,
                    policy='one GET and one normal form POST per month; no retries or redirects')
    with requests.Session() as session:
        for month in periods:
            item = dict(month=month, post_attempted=False)
            manifest['attempts'].append(item)
            try:
                page = period_page(month)
                item['page_url'] = page
                with session.get(page, timeout=(10, 20), stream=True, allow_redirects=False) as response:
                    item['page_status'] = response.status_code
                    if response.status_code != 200:
                        raise ValueError('public page did not return 200; stop')
                    html = bounded_body(response, 2_000_000, time.monotonic() + 30)
                (ROOT / (month + '.html')).write_bytes(html)
                item['page_sha256'] = digest(html)
                action, fields = public_form(html, month)
                item['form_fields'] = {k: v for k, v in fields.items() if k != 'tk'}
                item['token_empty'] = not bool(fields['tk'])
                item['post_attempted'] = True
                with session.post(action, data=fields, headers={'Referer': page},
                                  timeout=(10, 20), stream=True, allow_redirects=False) as response:
                    item['post_status'] = response.status_code
                    item['content_type'] = response.headers.get('Content-Type')
                    data = bounded_body(response, MAX_BYTES - manifest['zip_bytes'], time.monotonic() + 30)
                item['response_bytes'] = len(data)
                item['response_sha256'] = digest(data)
                if item['post_status'] != 200 or not zipfile.is_zipfile(io.BytesIO(data)):
                    (ROOT / (month + '.blocked-response')).write_bytes(data)
                    raise ValueError('normal public form POST did not deliver a ZIP; no retry')
                name = 'HISTDATA_COM_ASCII_EURUSD_M1_' + month + '.zip'
                (ROOT / name).write_bytes(data)
                item.update(status='DOWNLOADED', file=name, zip_sha256=digest(data))
                manifest['zip_bytes'] += len(data)
            except (requests.RequestException, ValueError, TimeoutError) as exc:
                item.update(status='BLOCKED', reason=str(exc))
                manifest['status'] = 'BLOCKED'
                break
        else:
            manifest['status'] = 'DOWNLOADED'
    save_json(ROOT / 'MANIFEST.json', manifest)
    print(json.dumps(manifest, indent=2))


def hourly_closes(data, month):
    import numpy as np
    import pandas as pd
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith('.csv')]
        if len(members) != 1 or members[0].file_size > 100_000_000:
            raise ValueError('expected one bounded monthly CSV')
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream, sep=';', header=None, usecols=[0, 4], names=['wall', 'close'])
    wall = pd.DatetimeIndex(pd.to_datetime(frame.wall, format='%Y%m%d %H%M%S', errors='raise'))
    period_page(month)
    if not len(wall) or not (wall.strftime('%Y' if len(month) == 4 else '%Y%m') == month).all():
        raise ValueError('unexpected source period')
    price = frame.close.to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError('invalid source prices')
    series = pd.Series(price, index=wall)
    duplicates = int(series.index.duplicated().sum())
    if duplicates and (series.groupby(level=0).nunique() > 1).any():
        raise ValueError('conflicting duplicate minute closes')
    series = series[~series.index.duplicated(keep='first')].sort_index()
    result = series.groupby(series.index.floor('h')).last()
    return result, dict(member=members[0].filename, minute_rows=len(series), hourly_rows=len(result),
                       duplicate_minutes_removed=duplicates)


def clock_series(hourly, policy):
    import pandas as pd
    zone = {'fixed_est': 'Etc/GMT+5', 'legacy_ny': 'America/New_York', 'literal_utc': 'UTC'}[policy]
    index = hourly.index.tz_localize(zone, ambiguous='NaT', nonexistent='NaT').tz_convert('UTC')
    valid = ~index.isna()
    # Both the original resampler and i6e use bar starts; retained ts are bar ends.
    return pd.Series(hourly.to_numpy()[valid], index=index[valid] + pd.Timedelta(hours=1))


def score(series, retained):
    import numpy as np
    common = series.index.intersection(retained.index)
    error = np.abs(series.loc[common].to_numpy() - retained.loc[common].to_numpy())
    return dict(source_hours=len(series), overlap=len(common),
                matched_1e12=int((error <= 1e-12).sum()),
                mismatched_1e12=int((error > 1e-12).sum()),
                max_abs_error=float(error.max()) if len(error) else None,
                mean_abs_error=float(error.mean()) if len(error) else None)


def retained_prices(path):
    import numpy as np
    import pandas as pd
    with np.load(path, allow_pickle=False) as archive:
        ts, price = archive['ts'], archive['price']
    if ts.ndim != 1 or price.shape != ts.shape or ts.dtype.kind not in 'iu':
        raise ValueError('unexpected retained ts/price schema')
    if not len(ts) or (ts >= 1735689600).any():
        raise ValueError('retained timestamps must be strictly pre-2025')
    if not (np.diff(ts) > 0).all() or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError('invalid retained ts/price')
    scope = dict(rows=len(ts), start_epoch_s=int(ts[0]), end_epoch_s=int(ts[-1]),
                 ts_int64_le_sha256=digest(ts.astype('<i8').tobytes()),
                 price_float64_le_sha256=digest(price.astype('<f8').tobytes()),
                 digest_scope='decoded retained ts and price arrays only, not source files or all history')
    return pd.Series(price, index=pd.to_datetime(ts, unit='s', utc=True)), scope


def compare(path):
    # Limit the comparison, including imports, to 2 GiB and 120 CPU/wall seconds.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.alarm(120)
    import pandas as pd
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    retained, scope = retained_prices(path)
    report = dict(status='NO_RECOVERED_SAMPLES', retained=scope, months=[],
                  test_read=False, parquet_read=False, all_history_proven=False)
    for item in manifest['attempts']:
        if item.get('status') != 'DOWNLOADED':
            continue
        month = item['month']
        period_page(month)
        if item['file'] != 'HISTDATA_COM_ASCII_EURUSD_M1_' + month + '.zip':
            raise ValueError('unexpected recovery manifest scope')
        data = (ROOT / item['file']).read_bytes()
        if digest(data) != item['zip_sha256']:
            raise ValueError('recovered ZIP digest mismatch')
        hourly, info = hourly_closes(data, month)
        ny_index = hourly.index.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
        is_dst = [False if pd.isna(t) else bool(t.dst()) for t in ny_index]
        policies = {}
        for policy in ('fixed_est', 'legacy_ny', 'literal_utc'):
            policies[policy] = {}
            for split, sample in [('all', hourly), ('ny_dst', hourly[is_dst]),
                                  ('ny_standard', hourly[[not x for x in is_dst]])]:
                policies[policy][split] = score(clock_series(sample, policy), retained)
        report['months'].append(dict(month=month, zip_sha256=item['zip_sha256'],
                                     source=info, policies=policies))
    if report['months']:
        report['status'] = 'COMPARED_BOUNDED_SAMPLES'
    save_json(ROOT / 'COMPARISON.json', report)
    print(json.dumps(report, indent=2))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'compare'])
    parser.add_argument('--retained', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--periods', nargs='+', help='Pre-2025 years or YYYYMM periods')
    args = parser.parse_args()
    ROOT = args.root
    if args.action == 'download':
        download(args.periods)
    elif args.retained is None:
        parser.error('--retained is required for comparison')
    else:
        os.umask(0o077)
        compare(args.retained)


if __name__ == '__main__':
    main()
