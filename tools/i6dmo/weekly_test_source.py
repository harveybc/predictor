"""Extend the retained feature recipe, without redoing feature selection."""
import argparse
from datetime import datetime, timedelta
import importlib.util
import importlib
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tools.i6dmo import weekly18 as W
from tools.i6n_contracts import eligibility as E

PRODUCER = '42272ad7f288d39d5b9f84c63bb219ef0b9a7b19'
END = pd.Timestamp('2026-01-01T00:00:00Z')


def clock(ts):
    ts = np.asarray(ts)
    if ts.ndim != 1 or ts.dtype.kind not in 'iu' or not len(ts) or np.any(np.diff(ts) <= 0):
        raise ValueError('ORDERED_SOURCE_CLOCK_REQUIRED')
    old = pd.to_datetime(ts - 3600, unit='s', utc=True)
    wall = old.tz_convert('America/New_York').tz_localize(None)
    out = ((wall + pd.Timedelta(hours=5)).tz_localize('Europe/London', ambiguous='raise',
        nonexistent='raise').tz_convert('UTC') + pd.Timedelta(hours=1)).as_unit('s').asi8
    if np.any(out < ts) or np.any(np.diff(out) <= 0):
        raise ValueError('SOURCE_CLOCK_MOVED_EARLIER_OR_COLLIDED')
    return out


def calendar():
    sunday = datetime(2018, 12, 30, 17, tzinfo=ZoneInfo('America/New_York'))
    weeks = []
    while sunday.year < 2026:
        weeks.append(dict(week_id=sunday.date().isoformat(), market_open=E.iso(sunday),
            market_close=E.iso(sunday + timedelta(days=5)),
            entry_cutoff=E.iso((sunday + timedelta(days=4)).replace(hour=0))))
        sunday += timedelta(days=7)
    return dict(market_calendar=dict(calendar_id='FX_NY_17_ASSUMED_NOT_BROKER',
        version='i6n-v1', weeks=weeks), entry_delay_after_open_hours=12,
        force_flat_before_close_hours=12)


def original_package(root):
    actual = W.run_cmd(['git', '-C', root, 'rev-parse', 'HEAD']).strip()
    if actual != PRODUCER:
        raise ValueError('FEATURE_PRODUCER_PIN')
    path = Path(root) / 'tools/eurusd_ps'
    spec = importlib.util.spec_from_file_location('weekly_test_original_features', path / '__init__.py',
        submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return lambda name: importlib.import_module(spec.name + '.' + name)


def generate(spec):
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=True)
    def progress(stage):
        W.B.put(out / 'STATUS.json', dict(state='PREPARING_SOURCE', stage=stage))
    progress('LOAD_PINNED_PRODUCER')
    original = original_package(spec['producer'])
    contract = original('contract'); contract.READ_END = END
    sources, features, cov = (original(n) for n in ('sources', 'features', 'covariates'))
    lake = Path(spec['lake']); reference = Path(spec['reference'])
    names = W.read(reference / 'CLOCK.json')['features']
    if len(names) != 19 or any('stlfsi4' in n for n in names):
        raise ValueError('SELECTED_FEATURE_IDENTITY')
    metadata = {}
    for b in ('batch_001', 'batch_002', 'batch_003'):
        for row in W.read(Path(spec['metadata']) / b / 'admissible_features.json')['features']:
            metadata[row['feature_id']] = row
    progress('ORIGINAL_PRICE_FEATURES')
    raw_path = lake / 'features/trading_asset_data/eurusd/5m.parquet'
    bars, measured_clock = sources.load_lake_5m(str(raw_path))
    hourly = sources.hourly_from_5m(bars)
    decision = hourly.index[(hourly.index >= pd.Timestamp('2019-01-01T00:00:00Z')) & (hourly.index < END)]
    x, _ = features.price_features(hourly, decision)
    x = x[['px.close_loc', 'px.log_co', 'px.rv5']].copy()
    progress('ORIGINAL_REGIME_FEATURE')
    path = Path(spec['producer']) / 'app/regime_detector.py'
    module_spec = importlib.util.spec_from_file_location('weekly_test_original_regime', path)
    regime = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(regime)
    x['rg.atr_ratio'] = regime.compute_regime_features(hourly[['open','high','low','close']])['atr_ratio'].reindex(decision)
    source_files = {str(raw_path.relative_to(lake)): W.file_sha256(raw_path)}
    for fid in names:
        if fid in x:
            continue
        progress(fid)
        source = metadata[fid]['source']
        if not source.startswith('lake:'):
            raise ValueError('UNEXPECTED_SELECTED_SOURCE:' + fid)
        path = lake / source[5:]; source_files[source[5:]] = W.file_sha256(path)
        if fid.startswith('fx.'):
            data, _, _ = cov.fx_pair_features('gbpusd', str(path), decision)
        else:
            yahoo = fid.startswith('yh.')
            series = cov.yahoo_series(str(path)) if yahoo else cov.fred_series(str(path))
            data, _ = cov.daily_features(fid.rsplit('.', 1)[0], series, 'close' if yahoo else 'value',
                decision, 'yahoo' if yahoo else 'fred', '', source, levels=fid.endswith('.level'))
        x[fid] = data[fid]
    progress('VERIFY_RETAINED_PREFIX')
    ts = clock(decision.as_unit('s').asi8)
    values = x[names].to_numpy(dtype=float); price = hourly['close'].reindex(decision).to_numpy(float)
    np.savez_compressed(out/'REPLAY.npz',ts=ts,x=values,price=price)
    old = dict(np.load(reference / 'INPUT.npz', allow_pickle=False))
    ix = np.searchsorted(ts, old['ts'])
    if np.any(ix >= len(ts)) or not np.array_equal(ts[ix], old['ts']):
        raise ValueError('RETAINED_PREFIX_TIMESTAMPS_DIFFER')
    # Transcendental ufuncs can differ by a few float64 ULPs across CPU builds.
    # Never replace retained bytes; require the replay also agree at model input precision.
    comparisons = {n: dict(exact=bool(np.array_equal(values[ix,j], old['x'][:,j], equal_nan=True)),
        float32_exact=bool(np.array_equal(values[ix,j].astype('float32'),old['x'][:,j].astype('float32'),equal_nan=True)),
        max_abs=float(np.nanmax(np.abs(values[ix,j]-old['x'][:,j])))) for j,n in enumerate(names)}
    price_exact = np.array_equal(price[ix], old['price'])
    W.B.put(out / 'PREFIX_COMPARISON.json', dict(features=comparisons, price_exact=bool(price_exact), rows=len(ix)))
    if not price_exact or not all(v['float32_exact'] and v['max_abs'] <= 8*np.finfo('float64').eps for v in comparisons.values()):
        raise ValueError('ORIGINAL_FEATURE_REPLAY_DIFFERS:see PREFIX_COMPARISON.json')
    new = (ts >= int(pd.Timestamp('2025-01-01T00:00:00Z').timestamp())) & (ts < int(END.timestamp()))
    if not new.any() or old['ts'][-1] >= ts[new][0]:
        raise ValueError('TEST_SOURCE_EXTENSION_EMPTY_OR_OVERLAP')
    np.savez_compressed(out / 'SOURCE.npz', ts=np.concatenate([old['ts'],ts[new]]),
        x=np.concatenate([old['x'],values[new]]), price=np.concatenate([old['price'],price[new]]))
    W.B.put(out / 'CALENDAR.json', calendar())
    report = dict(schema='i6dmo.weekly_test.source.v1', state='COMPLETE', features=names,
        prefix_rows=len(old['ts']), test_rows=int(new.sum()), producer_commit=PRODUCER,
        producer_files={str(p.relative_to(Path(spec['producer']))):W.file_sha256(p)
            for p in sorted((Path(spec['producer'])/'tools/eurusd_ps').glob('*.py'))},
        regime_sha256=W.file_sha256(Path(spec['producer'])/'app/regime_detector.py'),
        source_files=source_files, retained_input_sha256=W.file_sha256(reference/'INPUT.npz'),
        prefix_policy='RETAINED_BYTES_UNCHANGED; replay requires float32 equality and <=8 float64 epsilon absolute',
        files={n:W.file_sha256(out/n) for n in ('SOURCE.npz','CALENDAR.json','PREFIX_COMPARISON.json')},
        test_read=True, clock=measured_clock,
        limitations=['Current-vintage FRED; publication assumptions inherited',
            'Assumed FX calendar, not broker-certified', 'Source clock extended to 2025, not independently quote-certified'])
    report['sha256'] = W.B.digest(report); W.B.put(out/'SOURCE.json',report)
    W.B.put(out/'STATUS.json',dict(state='COMPLETE',test_rows=int(new.sum()),test_read=True))
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--spec',required=True)
    args=parser.parse_args(); print(generate(W.read(args.spec)))


if __name__ == '__main__':
    main()
