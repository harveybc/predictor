"""Read only timestamps of the retained I6-E INPUT.npz (no TEST rows by contract)."""
import sys, collections
import numpy as np
from datetime import datetime, timezone
p = sys.argv[1]
with np.load(p, allow_pickle=False) as a:
    ts = a["ts"].copy(); price = a["price"].copy()
print("rows", len(ts), "first", datetime.fromtimestamp(int(ts[0]), timezone.utc), "last", datetime.fromtimestamp(int(ts[-1]), timezone.utc))
assert ts[-1] < 1735689600
dt = np.diff(ts)//3600
print("gap histogram (hours):", collections.Counter(dt.tolist()).most_common(12))
# weekend gaps: show first bar after each gap >= 24h and last bar before, sample in 2023/2024
big = np.flatnonzero(dt >= 24)
def fmt(x): d = datetime.fromtimestamp(int(x), timezone.utc); return d.strftime("%a %Y-%m-%d %H:%M")
sel = [i for i in big if 1672531200 <= ts[i] < 1735689600]
for i in sel[:6] + sel[-6:]:
    print("last before gap", fmt(ts[i]), "-> first after", fmt(ts[i+1]), "gap_h", int(dt[i]))
# weekday/hour of last bar before weekend and first after, distribution for 2020-2024
lb = collections.Counter(); fa = collections.Counter()
for i in big:
    d0 = datetime.fromtimestamp(int(ts[i]), timezone.utc); d1 = datetime.fromtimestamp(int(ts[i+1]), timezone.utc)
    lb[(d0.strftime("%a"), d0.hour)] += 1; fa[(d1.strftime("%a"), d1.hour)] += 1
print("last-bar-before-gap (weekday,hourUTC):", lb.most_common(8))
print("first-bar-after-gap (weekday,hourUTC):", fa.most_common(8))
# holiday gaps: gaps >=24 that are not Fri->Sun
odd = [ (fmt(ts[i]), fmt(ts[i+1]), int(dt[i])) for i in big if datetime.fromtimestamp(int(ts[i]), timezone.utc).strftime("%a") not in ("Fri",) ]
print("non-Friday long gaps:", len(odd), odd[:15])
# short intraweek gaps 2..23h
mid = [ (fmt(ts[i]), int(dt[i])) for i in np.flatnonzero((dt>1)&(dt<24)) if ts[i] >= 1672531200]
print("intraweek gaps 2..23h since 2023:", len(mid), mid[:20])
