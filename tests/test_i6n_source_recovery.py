import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from tools.i6n_source_recovery import bounded_body, clock_series, hourly_closes, public_form, retained_prices, score
from tools import i6n_source_recovery as recovery


def form(extra=''):
    fields = dict(tk='', date='2024', datemonth='202403', platform='ASCII', timeframe='M1', fxpair='EURUSD')
    return '<form id="file_down" method="POST" action="/get.php">' + ''.join(
        f'<input name="{k}" value="{v}">' for k, v in fields.items()) + '</form>' + extra


def test_empty_public_token_is_preserved():
    action, fields = public_form(form(), '202403')
    assert action == 'https://www.histdata.com/get.php'
    assert fields['tk'] == ''


def test_annual_public_form_and_closes():
    html = form().replace('202403', '2024')
    assert public_form(html, '2024')[1]['datemonth'] == '2024'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('year.csv', '20240301 125900;1;1;1;1.2;0\n20240401 125900;1;1;1;1.3;0\n')
    result, info = hourly_closes(buffer.getvalue(), '2024')
    assert len(result) == info['hourly_rows'] == 2
    with pytest.raises(ValueError):
        hourly_closes(buffer.getvalue(), '2023')


@pytest.mark.parametrize('second,accepted', [('1.2', True), ('1.3', False)])
def test_duplicate_minute_close_must_agree(second, accepted):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('year.csv', f'20240301 125900;1;1;1;1.2;0\n20240301 125900;1;1;1;{second};0\n')
    if accepted:
        result, info = hourly_closes(buffer.getvalue(), '2024')
        assert len(result) == 1 and info['duplicate_minutes_removed'] == 1
    else:
        with pytest.raises(ValueError, match='conflicting'):
            hourly_closes(buffer.getvalue(), '2024')


@pytest.mark.parametrize('period', ['2025', '202501', '202413', '20240', '../2024'])
def test_bad_or_test_download_period_rejected(period):
    with pytest.raises(ValueError):
        recovery.period_page(period)


@pytest.mark.parametrize('html', [form('<input type="password">'), form('<div class="g-recaptcha"></div>'),
                                       form().replace('/get.php', 'https://example.com/get.php')])
def test_access_controls_and_changed_action_stop(html):
    with pytest.raises(ValueError):
        public_form(html, '202403')


def test_march_dst_distinguishes_policies():
    wall = pd.DatetimeIndex(['2024-03-08 12:00', '2024-03-11 12:00'])
    hourly = pd.Series([1.1, 1.2], index=wall)
    fixed = clock_series(hourly, 'fixed_est')
    old = clock_series(hourly, 'legacy_ny')
    assert fixed.index[0] == old.index[0]
    assert fixed.index[1] - old.index[1] == pd.Timedelta(hours=1)
    assert old.index[1].hour == 17
    assert score(old, old)['matched_1e12'] == 2
    assert score(fixed, old)['overlap'] == 1


def test_hourly_last_close_and_month_guard():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('DAT_ASCII_EURUSD_M1_202403.csv',
                         '20240301 120000;1;1;1;1.1;0\n20240301 125900;1;1;1;1.2;0\n')
    result, info = hourly_closes(buffer.getvalue(), '202403')
    assert result.iloc[0] == 1.2
    assert info['minute_rows'] == 2
    with pytest.raises(ValueError):
        hourly_closes(buffer.getvalue(), '201803')


def test_retained_only_ts_price_and_test_guard(tmp_path):
    path = tmp_path / 'input.npz'
    # An object member would fail if decoded with allow_pickle=False.
    np.savez(path, ts=np.array([1710000000]), price=np.array([1.2]), x=np.array([object()]))
    retained, scope = retained_prices(path)
    assert len(retained) == scope['rows'] == 1
    np.savez(path, ts=np.array([1735689600]), price=np.array([1.2]))
    with pytest.raises(ValueError, match='pre-2025'):
        retained_prices(path)


def test_stream_budget_is_enforced():
    class Response:
        headers = {}

        def iter_content(self, chunk_size):
            yield b'123'
            yield b'456'

    with pytest.raises(ValueError, match='budget'):
        bounded_body(Response(), 5, float('inf'))


def test_empty_post_stops_all_downloads_and_prevents_replay(tmp_path, monkeypatch):
    import requests
    calls = []

    class Response:
        status_code = 200
        headers = {'Content-Type': 'text/html'}

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            yield self.body

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            calls.append('GET')
            assert kwargs['allow_redirects'] is False
            return Response(form().encode())

        def post(self, url, **kwargs):
            calls.append('POST')
            assert kwargs['data']['tk'] == ''
            assert kwargs['allow_redirects'] is False
            return Response(b'')

    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(recovery, 'ROOT', tmp_path / 'private')
    old_umask = os.umask(0o077)
    try:
        recovery.download()
        with pytest.raises(FileExistsError):
            recovery.download()
    finally:
        os.umask(old_umask)
    manifest = json.loads((recovery.ROOT / 'MANIFEST.json').read_text())
    assert manifest['status'] == 'BLOCKED'
    assert manifest['zip_bytes'] == 0
    assert calls == ['GET', 'POST']
    assert len(manifest['attempts']) == 1
