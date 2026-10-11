import sys, numpy as np
from datetime import datetime, timezone
with np.load(sys.argv[1], allow_pickle=False) as a:
    ts = a["ts"].astype(np.int64).copy()
f = lambda x: datetime.fromtimestamp(int(x), timezone.utc).strftime("%a %Y-%m-%d %H:%M")
d = np.diff(ts) // 3600
lo = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
for i in np.flatnonzero(d > 3):
    if ts[i] >= lo and not (f(ts[i]).startswith("Fri") and d[i] in (48, 49, 50)):
        print(f(ts[i]), "->", f(ts[i + 1]), int(d[i]), "h")
m = (ts >= int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()))
dd = d[m[:-1]]
print("2024 gap histogram:", {int(k): int(v) for k, v in zip(*np.unique(dd, return_counts=True))})
