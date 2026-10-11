"""Build a separate, pre-TEST I6N input after original-source clock reconciliation.

This preserves the selected feature population and values. It does not certify
vintages, rerun selection, infer a broker calendar, or overwrite old evidence.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tools.i6m_data import file_hash, load_input
from tools.i6n_contracts import eligibility as c1
from tools.i6n_model import DEFAULT_CONFIG
from tools.i6n_source_recovery import hourly_closes, clock_series
from tools import i6n_external_clock_audit as clock_v1
from tools import i6n_external_clock_audit_v2 as clock_v2


def corrected_timestamps(ts):
    """Invert the old NY clock; apply the separately tested European DST rule."""
    ts = np.asarray(ts)
    if (ts.ndim != 1 or ts.dtype.kind not in 'iu' or not len(ts)
            or (np.diff(ts) <= 0).any() or (ts >= 1735689600).any()):
        raise ValueError('ordered pre-TEST timestamps required')
    old_start = pd.to_datetime(ts-3600, unit='s', utc=True)
    wall = old_start.tz_convert('America/New_York').tz_localize(None)
    result = ((wall+pd.Timedelta(hours=5)).tz_localize('Europe/London', ambiguous='raise',
              nonexistent='raise').tz_convert('UTC')+pd.Timedelta(hours=1)).as_unit('s').asi8
    if ((result < ts).any() or (result >= 1735689600).any()
            or (np.diff(result) <= 0).any()):
        raise ValueError('correction would move availability earlier, collide, or enter TEST')
    return result


def business_calendar():
    """Explicit nominal FX calendar, not a claim of observed broker sessions."""
    ny = ZoneInfo('America/New_York')
    sunday = datetime(2018, 12, 30, 17, tzinfo=ny)
    weeks = []
    while sunday.year < 2025:
        close = sunday + timedelta(days=5)
        cutoff = (sunday + timedelta(days=4)).replace(hour=0)
        weeks.append(dict(week_id=sunday.date().isoformat(), market_open=c1.iso(sunday),
                          market_close=c1.iso(close), entry_cutoff=c1.iso(cutoff)))
        sunday += timedelta(days=7)
    return dict(market_calendar=dict(calendar_id='FX_NY_17_ASSUMED_NOT_BROKER',
                version='i6n-v1', weeks=weeks), entry_delay_after_open_hours=12,
                force_flat_before_close_hours=12)


def write(path, value):
    """Exclusive artifact creation: a successor never replaces an old receipt."""
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


SCOPE_START = '2019-12-01T00:00:00Z'
SCOPE_END = '2025-01-01T00:00:00Z'


def validate_clock_audit(path, manifest_path):
    """Verify the consumed scope without upgrading the wider audit's verdict."""
    audit = json.loads(path.read_text())
    if path.with_suffix('.sha256').read_text().strip() != file_hash(path):
        raise ValueError('clock report checksum mismatch')
    expected = [d.isoformat() for d in clock_v1.planned_dates() if d.isoformat() != '2019-01-21']
    expected += [d.isoformat() for d in clock_v2.holdout_dates()]
    if (audit.get('status') not in ('VERIFIED', 'INCONCLUSIVE') or audit.get('clock_policy') != clock_v2.CLOCK_POLICY
            or audit.get('input_manifest_sha256') != file_hash(manifest_path)
            or [c.get('date') for c in audit.get('cases', [])] != expected):
        raise ValueError('independent clock audit incomplete or wrong source manifest')
    relevant = [c for c in audit['cases'] if c['date'] >= '2020-01-01']
    if len(relevant) != 20:
        raise ValueError('twenty predetermined in-scope clock cases required')
    for case in relevant:
        n = case.get('available_endpoints', 0)
        if (case.get('status') != 'PASS' or n < 12 or case.get('matched_exact') != n
                or case.get('absent_endpoint_count', -1)+n != 24
                or file_hash(path.parent/(case['date']+'.bi5')) != case.get('dukascopy_sha256')):
            raise ValueError('clock case failed, incomplete, or altered')
    return audit


def build(source, source_digest, design, archives, clock_audit, output):
    """Verify all original closes in the retained six-year span before emitting."""
    manifest_path = archives/'MANIFEST.json'
    audit = validate_clock_audit(clock_audit, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'DOWNLOADED':
        raise ValueError('source downloads incomplete')
    original, x, price = load_input(source, source_digest)
    keep = original >= int(pd.Timestamp(SCOPE_START).timestamp())
    ts, x, price = original[keep], x[keep], price[keep]
    matched, source_rows, records = set(), 0, []
    for item in manifest['attempts']:
        year = item['month']
        if len(year) != 4 or not 2019 <= int(year) <= 2024:
            raise ValueError('only complete 2019-2024 annual archives admitted')
        path = archives/item['file']
        if file_hash(path) != item['zip_sha256']:
            raise ValueError('source archive digest mismatch')
        hourly, info = hourly_closes(path.read_bytes(), year)
        candidate, _ = clock_v2.eu_clock_series(hourly)
        for case in [c for c in audit['cases'] if c['date'].startswith(year+'-') and int(year) >= 2020]:
            day = datetime.fromisoformat(case['date']).date()
            reference = clock_v1.decode((clock_audit.parent/(case['date']+'.bi5')).read_bytes(), day)
            rescored = clock_v2.compare_case(day, reference, candidate)
            if rescored['status'] != 'PASS' or any(rescored[k] != case[k] for k in rescored):
                raise ValueError('independent clock case replay mismatch')
        old = clock_series(hourly, 'legacy_ny')
        old = old[old.index >= pd.Timestamp(SCOPE_START)]
        seconds = old.index.as_unit('s').asi8
        indices = np.searchsorted(ts, seconds)
        valid = (indices < len(ts)) & (ts[np.minimum(indices, len(ts)-1)] == seconds)
        if not valid.all() or not np.allclose(price[indices], old.to_numpy(), atol=1e-12, rtol=0):
            raise ValueError('annual original source and retained prices differ')
        matched.update(indices.tolist())
        source_rows += len(old)
        records.append(dict(year=year, sha256=item['zip_sha256'], reconciled_retained_rows=len(old), **info))
    if matched != set(range(len(ts))) or source_rows != len(ts):
        raise ValueError('annual archives do not cover every retained successor row exactly once')
    corrected = corrected_timestamps(ts)
    features = json.loads(design.read_text())['members']
    if len(features) != x.shape[1] or len(set(features)) != 20:
        raise ValueError('selected feature order mismatch')
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'INPUT.npz', ts=corrected, x=x, price=price)
    cal = business_calendar()
    write(output/'CALENDAR.json', cal)
    clock_scope = dict(status='SCOPED_ACCEPTED', data_start=SCOPE_START, data_end=SCOPE_END,
                       maximum_planned_context_hours=432, evaluation_year=2024,
                       full_audit_status=audit['status'], full_audit_sha256=file_hash(clock_audit),
                       independently_checked_cases=20, independently_checked_years=[2020, 2021, 2022, 2023, 2024],
                       warmup_2019_december='Declared standard-time agreement of the candidate clock policies; '
                       'not independently quote-verified by this certificate',
                       excluded_out_of_scope_case='2019-10-28; before the earliest 2019-12-11 dependency; '
                       'original inconclusive audit retained unchanged',
                       scope_basis='Four preceding calendar years for every 2024 weekly refit, plus at most 432h context')
    write(output/'CLOCK_SCOPE.json', clock_scope)
    report = dict(status='PREPARED', rows=len(ts), shifted_rows=int((corrected != ts).sum()),
                  source_input_sha256=source_digest, design_sha256=file_hash(design),
                  independent_clock_audit_sha256=file_hash(clock_audit), archives=records,
                  input_sha256=file_hash(output/'INPUT.npz'), test_read=False,
                  clock_scope_sha256=file_hash(output/'CLOCK_SCOPE.json'),
                  inherited_feature_limits=['Selection uses DEVELOPMENT 2024, not prospective validation',
                      'FRED current vintage; historical revisions not certified',
                      'Daily publication lags are inherited declared assumptions',
                      'Nominal NY weekly calendar is not a verified broker schedule'],
                  transform='Invert legacy NY start clock; localize wall+5h in Europe/London, UTC+1h bar end; '
                            'move complete rows later or unchanged; preserve all selected feature values')
    write(output/'PREPARATION.json', report)
    clock = dict(schema='i6n.source_clock.v1', status='RESOLVED', coverage_mode='complete',
                 data_kind='historical', input_sha256=report['input_sha256'], features=features,
                 calendar_sha256=c1.canonical_digest(cal), timestamp_semantics='UTC_BAR_END',
                 availability_policy='AT_TIMESTAMP', clock_policy='EU_DST_OFFSET_MINUS5_BASE_BAR_START; '
                 'retained December 2019-2024 prices reconciled to annual source; 20/20 independent 2020-2024 '
                 'UTC cases; December 2019 standard-offset assumption; full wider audit remains '
                 +audit['status']+'; scoped certificate '+file_hash(output/'CLOCK_SCOPE.json')+
                 '; inherited publication/vintage assumptions, retrospective research only; '
                 'see PREPARATION.json '+file_hash(output/'PREPARATION.json'))
    write(output/'CLOCK.json', clock)
    # This is a full-data admission pilot, not a reduced-data accuracy substitute.
    cfg = dict(schema='i6n.historical_cell.v1', input=dict(path=str(output/'INPUT.npz'), sha256=report['input_sha256']),
               source_clock=dict(path=str(output/'CLOCK.json'), sha256=file_hash(output/'CLOCK.json')),
               calendar=cal, evaluation_week_id='2024-01-07', evaluation_split=dict(split_id='DEVELOPMENT',
               start='2024-01-01T00:00:00Z', end='2025-01-01T00:00:00Z', label_available_by=None),
               coverage_mode='complete', role='MANAGEMENT', horizon_hours=2, samples=13, spacing_hours=2,
               features=features, model=dict(DEFAULT_CONFIG), training=dict(max_epochs=50, patience=5,
               min_delta=0.000001, batch_size=256, learning_rate=.001, weight_decay=.0001,
               huber_delta=1., seed=6102026, head_weights=[1.]), device='gpu', asset='EURUSD', unit='USD_per_EUR',
               resources=dict(max_input_bytes=100_000_000, max_rows=100_000, max_samples=145,
               max_materialized_bytes=400_000_000, max_batch_bytes=20_000_000,
               max_updates=10000, max_parameters=1_000_000, max_seconds=3600))
    write(output/'PILOT.json', cfg)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'design', 'archives', 'clock-audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-digest', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.source_digest, args.design, args.archives,
                           args.clock_audit, args.output), indent=2))


if __name__ == '__main__':
    main()
