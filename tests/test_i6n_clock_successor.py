"""Clock conversion and successor guards; no historical fitting."""
import numpy as np
import pandas as pd
import pytest

from tools.i6n_clock_successor import corrected_timestamps, business_calendar, validate_clock_audit


def test_current_bar_end_moves_only_later_in_us_eu_mismatch():
    old = pd.to_datetime(['2024-01-15T18:00Z', '2024-03-18T17:00Z', '2024-07-15T17:00Z', '2024-10-28T17:00Z'])
    actual = corrected_timestamps(old.as_unit('s').asi8)
    expected = pd.to_datetime(['2024-01-15T18:00Z', '2024-03-18T18:00Z', '2024-07-15T17:00Z', '2024-10-28T18:00Z']).as_unit('s').asi8
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual >= old.as_unit('s').asi8)


def test_target_test_boundary_and_disordered_source_rejected():
    with pytest.raises(ValueError):
        corrected_timestamps(np.array([1735689600]))
    with pytest.raises(ValueError):
        corrected_timestamps(np.array([1700006400, 1700002800]))


def test_business_calendar_is_explicit_assumption_not_broker_evidence():
    from tools.i6n_contracts import eligibility as c1
    cal = business_calendar()
    assert 'ASSUMED' in cal['market_calendar']['calendar_id']
    _, weeks = c1.load_calendar(cal)
    assert len(weeks) > 300
    w = next(w for w in weeks if w.week_id == '2024-03-10')
    assert w.market_open.hour == 21
    assert (w.market_close-w.market_open).total_seconds() == 120*3600


def test_verified_word_without_cases_is_rejected(tmp_path):
    import json
    from tools.i6m_data import file_hash
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    path = tmp_path/'REPORT.json'
    path.write_text(json.dumps(dict(status='VERIFIED', clock_policy='EU_DST_OFFSET_MINUS5_BASE_BAR_START',
        input_manifest_sha256=file_hash(manifest), cases=[])))
    path.with_suffix('.sha256').write_text(file_hash(path))
    with pytest.raises(ValueError, match='incomplete'):
        validate_clock_audit(path, manifest)


def test_scoped_gate_keeps_wider_failure_but_rejects_in_scope_failure(tmp_path):
    import json
    from tools.i6m_data import file_hash
    from tools import i6n_external_clock_audit as v1, i6n_external_clock_audit_v2 as v2
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    dates = [d.isoformat() for d in v1.planned_dates() if d.isoformat() != '2019-01-21']
    dates += [d.isoformat() for d in v2.holdout_dates()]
    cases = []
    for day in dates:
        artifact = tmp_path/(day+'.bi5')
        artifact.write_bytes(b'fixture-only-not-decoded-by-metadata-gate')
        cases.append(dict(date=day, status='PASS', available_endpoints=24,
            matched_exact=24, absent_endpoint_count=0, dukascopy_sha256=file_hash(artifact)))
    next(c for c in cases if c['date'] == '2019-10-28')['status'] = 'INCONCLUSIVE'
    report = dict(status='INCONCLUSIVE', clock_policy=v2.CLOCK_POLICY,
        input_manifest_sha256=file_hash(manifest), cases=cases)
    path = tmp_path/'REPORT.json'
    def seal():
        path.write_text(json.dumps(report))
        path.with_suffix('.sha256').write_text(file_hash(path))
    seal()
    assert validate_clock_audit(path, manifest)['status'] == 'INCONCLUSIVE'
    next(c for c in cases if c['date'] == '2024-10-28')['status'] = 'INCONCLUSIVE'
    seal()
    with pytest.raises(ValueError, match='case failed'):
        validate_clock_audit(path, manifest)
