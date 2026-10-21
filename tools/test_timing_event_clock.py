import pandas as pd

from tools import timing_event_clock as ec


def test_fix_clock_follows_each_zone_dst():
    idx = pd.DatetimeIndex(["2024-07-15 13:00", "2024-01-15 13:00", "2024-03-12 13:00"], tz="UTC")   # decisions 14:00 UTC
    f = ec.event_clock(idx)
    assert list(f["ev.london_fix.hours_to"]) == [1.0, 2.0, 2.0]          # BST / GMT / GMT (US-only DST week)
    assert list(f["ev.ny_cut.hours_since"]) == [0.0, 23.0, 0.0]          # EDT / EST / EDT
    assert round(f["ev.tokyo_fix.hours_since"].iloc[0], 2) == 13.08      # Tokyo has no DST
    assert ((f.filter(like="hours_to") > 0) & (f.filter(like="hours_to") <= 24)).all().all()
