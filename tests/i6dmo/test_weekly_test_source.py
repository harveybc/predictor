from datetime import datetime, timezone

import numpy as np
import pytest

from tools.i6dmo import weekly_test_source as S
from tools.i6n_clock_successor import corrected_timestamps, business_calendar


def test_clock_matches_original_on_retained_years():
    ts=np.array([int(datetime(y,month,15,12,tzinfo=timezone.utc).timestamp())
        for y in range(2020,2025) for month in (1,3,7,11)],dtype=np.int64)
    assert np.array_equal(S.clock(ts),corrected_timestamps(ts))


def test_calendar_preserves_old_weeks_and_extends_only():
    old=business_calendar();new=S.calendar()
    assert new['market_calendar']['weeks'][:len(old['market_calendar']['weeks'])]==old['market_calendar']['weeks']
    assert new['entry_delay_after_open_hours']==new['force_flat_before_close_hours']==12
    assert new['market_calendar']['weeks'][-1]['week_id']=='2025-12-28'


@pytest.mark.parametrize('values',[[2,1],[1,1],[]])
def test_clock_rejects_unordered(values):
    with pytest.raises(ValueError):S.clock(np.array(values,dtype=np.int64))
