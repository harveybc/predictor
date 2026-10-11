"""Finite publication follower tests; no SSH, warehouse, or training."""
import json
import pytest

from tools import i6n_search_follower as f


def test_only_completed_candidates_publish_and_retained_receipt_skips(tmp_path, monkeypatch):
    root = tmp_path/'search'
    root.mkdir()
    for name, state in [('a', 'COMPLETED'), ('b', 'RUNNING'), ('c', 'FAILED')]:
        p = root/'candidates'/name
        p.mkdir(parents=True)
        (p/'STATUS.json').write_text(json.dumps({'state': state}))
        (p/'RESULT.json').write_text('{}')
    calls = []
    monkeypatch.setattr(f, 'sync', lambda entry: None)
    def publish(p, pin, url, token):
        calls.append(p.name)
        (p/'olap').mkdir(exist_ok=True)
        receipt = {'state': 'PUBLISHED_READBACK_VERIFIED', 'terminal_sha256': f.file_hash(p/'RESULT.json'),
                   'destination': url.rstrip('/'), 'code_commit': pin}
        (p/'olap'/'STATUS.json').write_text(json.dumps(receipt))
        return receipt
    monkeypatch.setattr(f, 'publish', publish)
    cfg = {'entries': [{'root': str(root), 'host': 'fixture'}], 'pin': 'a'*40,
           'url': 'http://fixture', 'token_file': str(tmp_path/'secret')}
    first = f.once(cfg)
    assert calls == ['a'] and first['published'] == 1
    assert f.once(cfg)['already_published'] == 1 and calls == ['a']
    cfg['url'] = 'http://other-warehouse'
    assert f.once(cfg)['published'] == 1 and calls == ['a', 'a']
    cfg['pin'] = 'b'*40
    assert f.once(cfg)['published'] == 1 and calls == ['a', 'a', 'a']


def test_transport_failure_is_pending_not_success(tmp_path, monkeypatch):
    def fail(entry):
        raise TimeoutError('test')
    monkeypatch.setattr(f, 'sync', fail)
    result = f.once({'entries': [{'root': str(tmp_path), 'host': 'fixture'}]})
    assert result['state'] == 'PENDING_DELIVERY'
    assert result['published'] == 0 and len(result['errors']) == 1


def test_historical_pilot_uses_its_own_publisher(tmp_path, monkeypatch):
    (tmp_path/'STATUS.json').write_text(json.dumps({'state': 'COMPLETED'}))
    (tmp_path/'RESULT.json').write_text('{}')
    calls = []
    monkeypatch.setattr(f, 'sync', lambda *a: None)
    monkeypatch.setattr(f, 'publish', lambda *a: pytest.fail('search publisher used for pilot'))
    monkeypatch.setattr(f, 'publish_pilot', lambda *a: calls.append(a[0]))
    cfg = {'entries': [{'root': str(tmp_path), 'host': 'fixture', 'kind': 'pilot', 'pin': 'b'*40}],
           'pin': 'a'*40, 'url': 'http://fixture', 'token_file': str(tmp_path/'secret')}
    assert f.once(cfg)['published'] == 1
    assert calls == [tmp_path]
