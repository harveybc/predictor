"""Search publication is separate from outer validation and never trains."""
import json

import pytest

from tools import i6n_search_publish as p


def test_projection_keeps_train_and_inner_distinct(monkeypatch, tmp_path):
    root = tmp_path
    result = {'identity': {'cell': {'config': {
        'evaluation_week_id': '2024-01-07', 'training': {'seed': 6102026},
        'source_clock': {'sha256': 'a'*64}}}}, 'receipt': {'fitness': 1.02}}
    for name in ('train', 'inner'):
        (root/f'{name}.c2.json').write_text(json.dumps({'split': name}))
    monkeypatch.setattr(p, 'verified_search', lambda path: result)
    (root/'RESULT.json').write_text(json.dumps(result))
    def reports(source, sha, pin):
        return [dict(experiment_key='same', actor='predictor', lake='LOCAL_RETAINED', phase='old', metrics=[dict(
            metric='mae', value=.1, split='validation', horizon=2, unit='USD_per_EUR')],
            tags={'scope': 'old'}, report_sha256='old')]
    monkeypatch.setattr(p.base, 'reports', reports)
    rows = p.search_reports(root, 'b'*40)
    assert len({r['experiment_key'] for r in rows}) == 2
    assert [r['metrics'][0]['split'] for r in rows] == ['train', 'inner']
    assert all(r['tags']['scope'] == 'SEARCH_TRAIN_INNER_NOT_OUTER_VALIDATION' for r in rows)
    assert all(r['report_sha256'] == p.base.report_sha256(r) for r in rows)


def test_failed_candidate_cannot_publish(tmp_path):
    (tmp_path/'STATUS.json').write_text(json.dumps({'state': 'FAILED'}))
    with pytest.raises(ValueError, match='completed'):
        p.verified_search(tmp_path)


def test_verifier_uses_trajectory_preparation_for_v2(monkeypatch, tmp_path):
    from tools import i6n_historical_evaluator as evaluator
    cfg = {"model": {"n_horizons": 4}}
    identity = dict(effective_config={}, context={"schema": evaluator.SCHEMA},
                    support={}, cell={"config": cfg})
    (tmp_path/'STATUS.json').write_text(json.dumps({"state": "COMPLETED"}))
    (tmp_path/'RESULT.json').write_text(json.dumps({"identity": identity}))
    monkeypatch.setattr(evaluator, 'load_request', lambda _: (cfg, {"support": {}}))
    seen = []
    monkeypatch.setattr(evaluator.cell, 'prepare', lambda c, **kw: seen.append(kw) or {})
    monkeypatch.setattr(evaluator, 'support', lambda *a: {})
    monkeypatch.setattr(evaluator, 'retained', lambda *a, **kw: None)
    p.verified_search(tmp_path)
    assert seen == [{"trajectory": True}]


def test_publish_failure_does_not_write_success(monkeypatch, tmp_path):
    monkeypatch.setattr(p, 'search_reports', lambda *a: [{'report_sha256': 'a'*64}])
    monkeypatch.setattr(p.base, 'token_from_file', lambda *a: 'not-a-real-token')
    def fail(*a):
        raise ConnectionError('offline')
    monkeypatch.setattr(p.base, 'request_json', fail)
    with pytest.raises(ConnectionError):
        p.publish(tmp_path, 'b'*40, 'http://localhost', tmp_path/'secret')
    assert not (tmp_path/'olap'/'STATUS.json').exists()
