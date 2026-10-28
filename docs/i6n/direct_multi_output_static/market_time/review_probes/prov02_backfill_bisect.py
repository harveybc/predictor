#!/usr/bin/env python3
"""PROV02 - find the vintage that back-filled the one revised observation.

The recovered ALFRED evidence (PROV01 control C5) shows exactly one of 1,587
observations per series whose first publication differs from the late anchor
vintage: 2020-08-19, published EMPTY at realtime_start 2020-08-21 and present
(DGS3MO 0.11, DGS7 0.48) at vintage 2025-01-10.

A value that appears later than its own realtime_start is precisely the case
the plan's section 5 warns about ("release time alone does not establish which
historical value was available"), so the back-fill vintage is a required
object: it is the instant from which a replay may use that value.

Bisection over the public ALFRED vintage list, one observation wide. Public
unauthenticated endpoint, no credential. argv[1] = output dir.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse as up
import zipfile

OUT = sys.argv[1]
OBS = "2020-08-19"
SERIES = ["DGS3MO", "DGS7"]
BASE = "https://alfred.stlouisfed.org/series/downloaddata?seid=%s"


def curl(url, data=None):
    cmd = ["curl", "-sS", "--fail", "--http1.1", "--max-time", "120",
           "--retry", "2", "--retry-delay", "3", "-o", "-", url]
    if data is not None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            p = f.name
        cmd += ["-X", "POST", "--data", "@" + p]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def vintages(seid):
    h = curl(BASE % seid).decode("utf8", "replace")
    m = re.search(
        r'<select[^>]*name="form\[selected_vintage_dates\]\[\]".*?</select>', h, re.S)
    return re.findall(r'value="(\d{4}-\d\d-\d\d)"', m.group(0))


def value_at(seid, vintage):
    body = up.urlencode([
        ("form[obs_start_date]", OBS), ("form[obs_end_date]", OBS),
        ("form[units]", "lin"), ("form[entered_vintage_dates]", ""),
        ("form[file_type]", "2"), ("form[file_format]", "csv"),
        ("form[download_data]", "Download Data"),
        ("form[selected_vintage_dates][]", vintage),
    ]).encode()
    raw = curl(BASE % seid, body)
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    txt = z.read(name).decode("utf8", "replace")
    val = None
    for line in txt.splitlines()[1:]:
        p = line.split(",")
        if p and p[0].strip() == OBS:
            val = p[1].strip() if len(p) > 1 else ""
    return val, hashlib.sha256(raw).hexdigest(), raw


out = {"probe": "PROV02_backfill_bisect", "observation": OBS,
       "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "authentication": "NONE - public ALFRED Download Data form",
       "series": {}}

for seid in SERIES:
    allv = [d for d in vintages(seid) if "2020-08-21" <= d <= "2025-01-10"]
    probed = []
    lo, hi = 0, len(allv) - 1
    vlo, dlo, _ = value_at(seid, allv[lo])
    probed.append({"vintage": allv[lo], "value": vlo, "sha256": dlo})
    vhi, dhi, _ = value_at(seid, allv[hi])
    probed.append({"vintage": allv[hi], "value": vhi, "sha256": dhi})
    time.sleep(1.5)
    if (vlo or "") == "" and (vhi or "") != "":
        while hi - lo > 1:
            mid = (lo + hi) // 2
            v, d, _ = value_at(seid, allv[mid])
            probed.append({"vintage": allv[mid], "value": v, "sha256": d})
            if (v or "") == "":
                lo = mid
            else:
                hi = mid
            time.sleep(1.5)
        first = allv[hi]
        prev_empty = allv[lo]
    else:
        first, prev_empty = None, None
    out["series"][seid] = {
        "vintages_searched": [allv[0], allv[-1]],
        "n_candidate_vintages": len(allv),
        "value_at_first_candidate_vintage": vlo,
        "value_at_last_candidate_vintage": vhi,
        "first_vintage_carrying_a_value": first,
        "last_vintage_still_empty": prev_empty,
        "requests": len(probed),
        "probe_trail": probed,
    }
    print(seid, "backfilled at", first, "after", prev_empty,
          "in", len(probed), "requests", flush=True)

with open(os.path.join(OUT, "PROV02_BACKFILL.json"), "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print("DONE")
