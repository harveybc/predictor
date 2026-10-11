"""Publication projection tests; fake C2 rows do not certify model quality."""
import pytest
from tools import i6n_publish as p


def test_synthetic_and_failed_rejected():
    for scope, state in [('SYNTHETIC_WEEKLY', 'COMPLETED'), ('HISTORICAL_SINGLE_WEEK_DEVELOPMENT', 'FAILED')]:
        with pytest.raises(ValueError):
            p.reports(dict(scope=scope, state=state), 'a'*64, 'b'*40)


def test_projection_preserves_horizon_sign_and_populations(monkeypatch):
    rows = []
    for kind in ['S', 'U']:
        rows.append(dict(record_type='I6N_PAIRED_METRIC', population_kind=kind,
            role='management_origin', horizon_hours=72, unit='USD_per_EUR',
            mae=.001, naive_mae=.002, skill_mae=-.5, n=12, n_declared=12,
            config_id='c'*64, calendar_id='calendar', protocol_id='protocol', row_keys_sha256='d'*64))
    calls = []
    def verified(metrics):
        calls.append(metrics)
        return rows
    monkeypatch.setattr(p.c2, 'olap_records', verified)
    cfg = dict(evaluation_week_id='2024-01-07', training=dict(seed=6102026),
               source_clock=dict(sha256='e'*64))
    source = dict(scope='HISTORICAL_SINGLE_WEEK_DEVELOPMENT', state='COMPLETED',
                  metrics={'fixture': True}, identity=dict(config=cfg))
    reports = p.reports(source, 'a'*64, 'b'*40)
    assert calls == [source['metrics']]
    assert len({r['experiment_key'] for r in reports}) == 2
    for r in reports:
        assert all(m['horizon'] == 72 for m in r['metrics'])
        assert next(m['value'] for m in r['metrics'] if m['metric'] == 'skill_mae') == -.5
        assert r['tags']['production_eligible'] == 'false'
        assert r['tags']['scope'] == 'ONE_WEEK_NOT_ANNUAL'
        assert r['report_sha256'] == p.report_sha256(r)


def test_corrupt_c2_cannot_be_exported():
    with pytest.raises(Exception):
        p.reports(dict(scope='HISTORICAL_SINGLE_WEEK_DEVELOPMENT', state='COMPLETED',
                       metrics={}, identity={}), 'a'*64, 'b'*40)
