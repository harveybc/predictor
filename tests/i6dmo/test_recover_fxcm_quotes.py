import gzip

import pandas as pd
import pytest

from tools.i6dmo.recover_fxcm_quotes import parse


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
