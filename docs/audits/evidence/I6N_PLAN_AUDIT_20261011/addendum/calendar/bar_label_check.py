"""AUD-08 addendum: bar-label convention and the DST-start anomaly, timestamps only.

Part 1 reads ONLY the 'ts' member of the retained I6-E INPUT archive (rows end
2024-12-31 22:00 UTC; refuses TEST rows). Weeks are split at Saturday 00:00
UTC. For every week it prints the first and last bar label in
America/New_York and classifies the week as a US/EU DST-gap week when New
York and London disagree on daylight time at Wednesday 12:00 UTC.

Part 2 is a CONTROL with known inputs: it feeds synthetic 5-minute source
stamps (no market data) through the audited converter
tools/i6e_price_metrics.py:hourly_close (HEAD 269c2d13) to show which UTC
end label a given source stamp receives, for (i) a source clock that follows
US DST (the declared NEW_YORK_LOCAL clock) and (ii) a hypothetical source
clock that follows EU DST dates. It demonstrates the mechanism, not that the
real source used it.
Usage: python bar_label_check.py <INPUT.npz> <AUDITED_WORKTREE>
"""
import collections
import io
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")
LON = ZoneInfo("Europe/London")
UTC = timezone.utc
TEST_START = 1735689600
SAT0 = int(datetime(2012, 4, 28, tzinfo=UTC).timestamp())


def ny(x):
    return datetime.fromtimestamp(int(x), UTC).astimezone(NY)


def u(x):
    return datetime.fromtimestamp(int(x), UTC).strftime("%a %Y-%m-%d %H:%M UTC")


def part1(npz):
    with zipfile.ZipFile(npz) as z:
        ts = np.load(io.BytesIO(z.read("ts.npy")), allow_pickle=False).astype(np.int64)
    assert ts[-1] < TEST_START
    key = (ts - SAT0) // (7 * 86400)
    tab = collections.Counter()
    gap_rows = []
    for k in np.unique(key):
        w = ts[key == k]
        mid = datetime.fromtimestamp(SAT0 + int(k) * 7 * 86400 + 4 * 86400 + 12 * 3600, UTC)
        gap = bool(mid.astimezone(NY).dst()) != bool(mid.astimezone(LON).dst())
        f, l = ny(w[0]), ny(w[-1])
        era = "2012-2018" if mid.year <= 2018 else "2019-2024"
        cls = f"first={f.strftime('%a')} {f.hour:02d}:00 last={l.strftime('%a')} {l.hour:02d}:00"
        tab[(era, "US/EU-DST-gap" if gap else "aligned", cls)] += 1
        if gap:
            gap_rows.append((mid.date().isoformat(), f.strftime("%a %Y-%m-%d %H:%M %Z"), l.strftime("%a %Y-%m-%d %H:%M %Z"), int(len(w))))
    print("## 1a. Weekly first/last bar label (America/New_York), counts by era and DST alignment (only classes with >=1 week)")
    for (era, g, cls), n in sorted(tab.items()):
        print(f"{era}\t{g}\t{cls}\t{n}")
    print("\n## 1b. Every US/EU DST-gap week: Wednesday date, first label NY, last label NY, bars")
    for r in gap_rows:
        print("\t".join(map(str, r)))
    print("\n## 1c. Exact rows (timestamps only) around four 2024 weekend reopenings")
    for sunday in ("2024-03-03", "2024-03-10", "2024-03-24", "2024-03-31"):
        s = int(datetime.fromisoformat(sunday).replace(tzinfo=UTC).timestamp())
        i = int(np.searchsorted(ts, s - 2 * 86400))
        j = int(np.searchsorted(ts, s + 86400))
        seg = ts[i:j]
        d = np.diff(seg) // 3600
        g = int(np.argmax(d))
        rows = list(range(max(0, g - 1), g + 1)) + list(range(g + 1, min(len(seg), g + 4)))
        print(f"week of Sunday {sunday}:")
        for r in rows:
            lab = seg[r]
            print(f"  ts={int(lab)}  {u(lab)}  = {ny(lab).strftime('%a %H:%M %Z')} NY; END-label interval NY [{ny(lab - 3600).strftime('%H:%M')}, {ny(lab).strftime('%H:%M')})"
                  + (f"   <- {int(d[g])} h gap follows" if r == g else ""))


def part2(worktree):
    import pandas as pd
    sys.path.insert(0, worktree)
    from tools.i6e_price_metrics import hourly_close  # audited converter, read-only
    print("\n## 2. CONTROL: synthetic 5m source stamps through tools/i6e_price_metrics.py:hourly_close (no market data)")
    cases = [
        ("normal summer week, source opens 17:00 NY", "2024-06-09 17:00"),
        ("DST-start Sunday, US-DST source opens 17:00 EDT", "2024-03-10 17:00"),
        ("DST-start Sunday, EU-DST-dated source stamps the same instant as 16:00", "2024-03-10 16:00"),
        ("normal winter week, source opens 17:00 NY", "2024-01-07 17:00"),
    ]
    for name, first in cases:
        stamps = pd.date_range(first, periods=24, freq="5min")  # two hours of naive source stamps
        raw = pd.DataFrame({"timestamp": stamps, "close": np.linspace(1.08, 1.09, len(stamps))})
        out = hourly_close(raw)
        labels = [x.strftime("%a %Y-%m-%d %H:%M UTC") for x in out.index]
        print(f"{name}: first source stamp {first} (naive, read as America/New_York BAR_START) -> hourly END labels {labels}")


if __name__ == "__main__":
    part1(sys.argv[1])
    part2(sys.argv[2])
