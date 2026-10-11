"""Identify which retained I6-E INPUT.npz the original sub-auditor read.

Reads ONLY the 'ts' member of each candidate archive (never x, price, origins,
weeks or future). Prints the archive sha256, the sha256 of the raw ts.npy
member bytes, row count, first/last timestamp, and whether the signature
printed by the retained probes/a/bar_clock.out (rows 77977, first
2012-05-01 00:00 UTC, last 2024-12-31 22:00 UTC) matches. Refuses any archive
whose ts reaches 2025-01-01 (TEST) before printing anything else about it.
Usage: python identify_input.py <STATE_ROOT> <relpath> [<relpath> ...]
"""
import hashlib, io, sys, zipfile
from datetime import datetime, timezone
import numpy as np

TEST_START = 1735689600  # 2025-01-01T00:00:00Z
SIG = (77977, "2012-05-01 00:00", "2024-12-31 22:00")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def fmt(x):
    return datetime.fromtimestamp(int(x), timezone.utc).strftime("%Y-%m-%d %H:%M")

root = sys.argv[1]
for rel in sys.argv[2:]:
    path = f"{root}/{rel}"
    with open(path, "rb") as f:
        raw = f.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = sorted(z.namelist())
        tsb = z.read("ts.npy")
    ts = np.load(io.BytesIO(tsb), allow_pickle=False).astype(np.int64)
    if ts[-1] >= TEST_START:
        print(f"{rel}: REFUSED (ts reaches TEST)")
        continue
    sig = (len(ts), fmt(ts[0]), fmt(ts[-1]))
    print(f"input=$STATE/{rel}")
    print(f"  archive_sha256={sha(raw)} members={members}")
    print(f"  ts.npy_member_sha256={sha(tsb)} dtype={ts.dtype} rows={len(ts)} first={sig[1]} UTC last={sig[2]} UTC")
    print(f"  matches_bar_clock_out_signature={sig == SIG}")
