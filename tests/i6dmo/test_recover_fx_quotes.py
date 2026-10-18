import io
import zipfile

import pandas as pd
import pytest

from tools.i6dmo.recover_fx_quotes import minute_frame


def archive(rows):
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w') as z:
        z.writestr('DAT_ASCII_EURUSD_M1_2023.csv', '\n'.join(rows))
    result.seek(0)
    return result


GOOD = '20230301 000000;1.05;1.07;1.04;1.06;0'


def test_clock_is_fixed_est_including_summer():
    df, n = minute_frame(archive([GOOD, GOOD.replace('20230301','20230701')]), 'eurusd', 2023)
    assert list(df.index) == [pd.Timestamp('2023-03-01T05:00Z'), pd.Timestamp('2023-07-01T05:00Z')]
    assert n == 0


def test_identical_duplicates_removed_but_conflicts_rejected():
    df, n = minute_frame(archive([GOOD, GOOD]), 'eurusd', 2023)
    assert len(df) == 1 and n == 1
    with pytest.raises(ValueError, match='CONFLICTING_DUPLICATE'):
        minute_frame(archive([GOOD, GOOD.replace(';1.06;', ';1.065;')]), 'eurusd', 2023)


@pytest.mark.parametrize('row,reason', [
    (GOOD.replace('20230301','20220301'), 'PERIOD_MISMATCH'),
    (GOOD.replace(';1.06;', ';nan;'), 'MISSING_QUOTE'),
    (GOOD.replace(';1.06;', ';inf;'), 'NONFINITE'),
    (GOOD.replace(';1.07;', ';1.02;'), 'INVALID_OHLC'),
])
def test_bad_quotes_rejected(row, reason):
    with pytest.raises(ValueError, match=reason):
        minute_frame(archive([row]), 'eurusd', 2023)


def test_wrong_pair_not_adopted():
    with pytest.raises(ValueError, match='ARCHIVE_IDENTITY'):
        minute_frame(archive([GOOD]), 'gbpusd', 2023)
