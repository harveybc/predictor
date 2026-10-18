import gzip

import pandas as pd
import pytest

from tools.i6dmo.recover_fxcm_quotes import acquire, parse


HEADER='DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose\n'
ROW='03/06/2023 00:00:00.000,1.06,1.07,1.05,1.065,1.061,1.071,1.051,1.066\n'


def test_provider_utc_and_real_bid_ask_preserved():
    a=parse(gzip.compress((HEADER+ROW).encode()))
    assert a.index[0] == pd.Timestamp('2023-03-06T00:00Z')
    assert a.BidClose.iloc[0] == 1.065 and a.AskClose.iloc[0] == 1.066


@pytest.mark.parametrize('row,reason',[(ROW+ROW,'QUOTE_ORDER'),
    (ROW.replace('1.066','1.064'),'CROSSED_QUOTE'),
    (ROW.replace('1.065','nan'),'NONFINITE'),
    (ROW.replace('1.07,','1.04,'),'INVALID_OHLC')])
def test_bad_source_quotes_refused(row,reason):
    with pytest.raises(ValueError,match=reason):
        parse(gzip.compress((HEADER+row).encode()))


def fake_session(monkeypatch, row):
    class Response:
        content=gzip.compress((HEADER+row).encode())
        status_code=200
        def raise_for_status(self):pass
    class Session:
        urls=[]
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,**kwargs):
            self.urls.append(url)
            return Response()
    monkeypatch.setattr('tools.i6dmo.recover_fxcm_quotes.requests.Session',Session)
    monkeypatch.setattr('tools.i6dmo.recover_fxcm_quotes.time.sleep',lambda _:None)
    return Session


def test_native_hourly_preserves_gap_and_quotes_without_synthetic_minutes(tmp_path,monkeypatch):
    session=fake_session(monkeypatch,ROW+ROW.replace('00:00:00','02:00:00'))
    result=acquire(tmp_path,['eurusd'],2023,[10],periodicity='H1')
    assert result['state']=='ACQUIRED' and result['periodicity']=='H1'
    a=pd.read_parquet(tmp_path/'eurusd/2023-1h.parquet')
    assert len(a)==2 and a.timestamp.diff().iloc[1]==pd.Timedelta(hours=2)
    assert (a.available_at-a.timestamp).eq(pd.Timedelta(hours=1)).all()
    assert a.BidClose.tolist()==[1.065,1.065]
    assert not (tmp_path/'eurusd/2023-1min.parquet').exists()
    assert not (tmp_path/'eurusd/2023-5min.parquet').exists()
    assert all('/H1/' in url for url in session.urls)


def test_native_hourly_rejects_minute_clock_and_empty_source(tmp_path,monkeypatch):
    fake_session(monkeypatch,ROW.replace('00:00:00','00:01:00'))
    result=acquire(tmp_path,['eurusd'],2023,[10],periodicity='H1')
    assert result['state']=='INCOMPLETE' and result['complete']==0
    assert result['failures'][0]['reason']=='HOURLY_TIMESTAMP_ALIGNMENT'
    assert not (tmp_path/'eurusd/2023-1h.parquet').exists()


def test_native_hourly_resume_rejects_changed_materialized_quotes(tmp_path,monkeypatch):
    fake_session(monkeypatch,ROW)
    acquire(tmp_path,['eurusd'],2023,[10],periodicity='H1')
    path=tmp_path/'eurusd/2023-1h.parquet';a=pd.read_parquet(path)
    a.loc[0,'BidClose']=1.064;a.to_parquet(path,index=False)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='RECOVERED_FRAME_CHANGED'):
        acquire(tmp_path,['eurusd'],2023,[10],periodicity='H1')
    assert path.read_bytes()==before


def test_changed_week_scope_cannot_reuse_outputs(tmp_path,monkeypatch):
    fake_session(monkeypatch,ROW)
    acquire(tmp_path,['eurusd'],2023,[10],periodicity='H1')
    with pytest.raises(ValueError,match='RECOVERY_SCOPE_CHANGED'):
        acquire(tmp_path,['eurusd'],2023,[10,11],periodicity='H1')


def test_retry_can_add_real_new_rows_but_never_change_existing_quotes(tmp_path,monkeypatch):
    class Response:
        status_code=200
        def raise_for_status(self):pass
    class Session:
        available=False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,**kwargs):
            row=ROW if '/10.csv.gz' in url else ROW.replace('03/06','03/13') if self.available else ''
            response=Response();response.content=gzip.compress((HEADER+row).encode());return response
    monkeypatch.setattr('tools.i6dmo.recover_fxcm_quotes.requests.Session',Session)
    monkeypatch.setattr('tools.i6dmo.recover_fxcm_quotes.time.sleep',lambda _:None)
    a=acquire(tmp_path,['eurusd'],2023,[10,11],periodicity='H1')
    assert a['state']=='INCOMPLETE'
    old=pd.read_parquet(tmp_path/'eurusd/2023-1h.parquet')
    Session.available=True
    b=acquire(tmp_path,['eurusd'],2023,[10,11],periodicity='H1')
    assert b['state']=='ACQUIRED'
    new=pd.read_parquet(tmp_path/'eurusd/2023-1h.parquet')
    assert len(new)==2
    pd.testing.assert_frame_equal(new.iloc[:1],old)


def test_provider_rate_limit_stops_before_next_pair_and_retains_cooldown(tmp_path,monkeypatch):
    session=fake_session(monkeypatch,ROW)
    class Refused:
        status_code=429
        headers={'Retry-After':'3600'}
    calls=[]
    def get(self,url,**kwargs):calls.append(url);return Refused()
    monkeypatch.setattr(session,'get',get)
    monkeypatch.setattr('tools.i6dmo.recover_fxcm_quotes.time.time',lambda:1000)
    result=acquire(tmp_path,['eurusd','gbpusd'],2023,[10],periodicity='H1')
    assert result['state']=='RATE_LIMITED' and len(calls)==1
    assert result['retry_not_before_epoch']==4600


def test_bid_candle_view_retains_side_discrepancy_without_claiming_executable_spread(tmp_path,monkeypatch):
    import json
    fake_session(monkeypatch,ROW.replace('1.066','1.064'))
    result=acquire(tmp_path,['eurusd'],2023,[10],periodicity='H1',quote_view='BID')
    assert result['state']=='ACQUIRED'
    a=pd.read_parquet(tmp_path/'eurusd/2023-1h.parquet')
    assert a.BidClose.iloc[0]==1.065 and a.AskClose.iloc[0]==1.064
    report=json.loads((tmp_path/'eurusd/MANIFEST.json').read_text())
    assert report['crossed_side_bar_endpoints']==1
    assert report['executable_spread_eligible'] is False
    assert report['quote_view']=='BID'
