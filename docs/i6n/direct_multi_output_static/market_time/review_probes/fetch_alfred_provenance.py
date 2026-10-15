#!/usr/bin/env python3
"""Bounded unauthenticated ALFRED point-in-time fetch for DGS3MO / DGS7.

Public ALFRED "Download Data" form only: no API key, no credential, no secret.
Output format 4 = "Observations, Initial Release Only" yields, per observation,
the value as first published and its realtime_start_date.

Chunking rule (defeats left-censoring): chunk k requests observations in a
calendar quarter [q_start, q_end] with the vintage set covering
[q_start - PAD days, q_end + PAD days]; publication lag is <= a few business
days, so the true first-publication vintage of every observation in the
quarter lies inside the chunk's vintage set.

Also fetches output format 2 ("All Observations") at one late anchor vintage,
for the revision test against the initial releases.
"""
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse as up
import urllib.request as ur
import zipfile
import io

DL = sys.argv[1]
SERIES = ["DGS3MO", "DGS7"]
OBS_LO, OBS_HI = "2018-12-01", "2024-12-31"
ANCHOR_HI = "2025-01-10"
PAD = 20
UA = "predictor-i6dmo-provenance-review/1.0"


def get(url, data=None, timeout=180):
    import subprocess, tempfile
    cmd = ["curl", "-sS", "--fail", "--max-time", str(timeout),
           "--retry", "2", "--retry-delay", "3",
           "--http1.1", "-o", "-", url]
    if data is not None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            p = f.name
        cmd += ["-X", "POST", "--data", "@" + p]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return r.stdout


def vintage_dates(seid):
    h = get("https://alfred.stlouisfed.org/series/downloaddata?seid=%s" % seid).decode(
        "utf8", "replace")
    m = re.search(
        r'<select[^>]*name="form\[selected_vintage_dates\]\[\]".*?</select>', h, re.S)
    return re.findall(r'value="(\d{4}-\d\d-\d\d)"', m.group(0))


def post(seid, obs_lo, obs_hi, vints, file_type):
    body = [
        ("form[obs_start_date]", obs_lo),
        ("form[obs_end_date]", obs_hi),
        ("form[units]", "lin"),
        ("form[entered_vintage_dates]", ""),
        ("form[file_type]", str(file_type)),
        ("form[file_format]", "csv"),
        ("form[download_data]", "Download Data"),
    ] + [("form[selected_vintage_dates][]", d) for d in vints]
    return get("https://alfred.stlouisfed.org/series/downloaddata?seid=%s" % seid,
               up.urlencode(body).encode())


def quarters(lo, hi):
    a = dt.date.fromisoformat(lo)
    b = dt.date.fromisoformat(hi)
    out = []
    y, q = a.year, (a.month - 1) // 3
    while True:
        s = dt.date(y, q * 3 + 1, 1)
        e = (dt.date(y + (q == 3), (q * 3 + 4) if q < 3 else 1, 1)
             - dt.timedelta(days=1))
        if s > b:
            break
        out.append((max(s, a).isoformat(), min(e, b).isoformat()))
        q += 1
        if q == 4:
            q, y = 0, y + 1
    return out


manifest = {
    "artifact": "ALFRED point-in-time evidence for DGS3MO and DGS7",
    "source_endpoint": "https://alfred.stlouisfed.org/series/downloaddata?seid=<SERIES>",
    "method": "HTTP POST of the public ALFRED Download Data form",
    "authentication": "NONE - public unauthenticated endpoint, no API key used",
    "output_format": {"4": "Observations, Initial Release Only",
                      "2": "Observations by Vintage Date, All Observations"},
    "observation_window": [OBS_LO, OBS_HI],
    "chunk_rule": "one calendar quarter of observations; vintage set = quarter +/- %d days" % PAD,
    "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "series": {},
}

for seid in SERIES:
    allv = vintage_dates(seid)
    ent = {"alfred_vintage_dates_total": len(allv),
           "alfred_vintage_first": allv[0], "alfread_vintage_last": allv[-1],
           "chunks": [], "initial_release_rows": {}}
    rows = {}
    for (qs, qe) in quarters(OBS_LO, OBS_HI):
        vlo = (dt.date.fromisoformat(qs) - dt.timedelta(days=PAD)).isoformat()
        vhi = (dt.date.fromisoformat(qe) + dt.timedelta(days=PAD)).isoformat()
        vs = [d for d in allv if vlo <= d <= vhi]
        raw = post(seid, qs, qe, vs, 4)
        name = "%s_initial_release_%s_%s.zip" % (seid, qs, qe)
        with open(os.path.join(DL, name), "wb") as f:
            f.write(raw)
        rec = {"file": name, "obs_window": [qs, qe],
               "vintage_window": [vs[0], vs[-1]], "n_vintages": len(vs),
               "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            csvname = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            txt = z.read(csvname).decode("utf8", "replace")
            rec["csv_member"] = csvname
            rec["csv_sha256"] = hashlib.sha256(z.read(csvname)).hexdigest()
            lines = [l for l in txt.splitlines() if l.strip()]
            hdr = lines[0].split(",")
            rec["header"] = hdr
            n = 0
            for l in lines[1:]:
                p = l.split(",")
                if len(p) < 3:
                    continue
                obs, val, rts = p[0], p[1], p[2]
                if not (qs <= obs <= qe):
                    continue
                rows[obs] = (val, rts)
                n += 1
            rec["rows_kept"] = n
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
        ent["chunks"].append(rec)
        print(seid, qs, rec.get("rows_kept"), rec["sha256"][:12], flush=True)
        time.sleep(1.5)

    # revision test: all observations at one late anchor vintage
    vs = [d for d in allv if d <= ANCHOR_HI][-1:]
    raw = post(seid, OBS_LO, OBS_HI, vs, 2)
    name = "%s_all_obs_vintage_%s.zip" % (seid, vs[0])
    with open(os.path.join(DL, name), "wb") as f:
        f.write(raw)
    anchor = {"file": name, "vintage": vs[0], "bytes": len(raw),
              "sha256": hashlib.sha256(raw).hexdigest()}
    z = zipfile.ZipFile(io.BytesIO(raw))
    csvname = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    txt = z.read(csvname).decode("utf8", "replace")
    anchor["csv_member"] = csvname
    anchor["csv_sha256"] = hashlib.sha256(z.read(csvname)).hexdigest()
    cur = {}
    for l in txt.splitlines()[1:]:
        p = l.split(",")
        if len(p) >= 2 and re.match(r"\d{4}-\d\d-\d\d$", p[0]):
            cur[p[0]] = p[1]
    revised, missing_cur, compared = [], 0, 0
    for obs, (val, rts) in sorted(rows.items()):
        if obs not in cur:
            missing_cur += 1
            continue
        compared += 1
        if (val or "") != (cur[obs] or ""):
            revised.append({"obs": obs, "initial": val, "anchor": cur[obs],
                            "realtime_start": rts})
    anchor["compared"] = compared
    anchor["observations_revised"] = len(revised)
    anchor["revised_sample"] = revised[:25]
    anchor["absent_from_anchor"] = missing_cur
    ent["revision_test"] = anchor

    # lag statistics
    lags = {}
    bad = []
    for obs, (val, rts) in rows.items():
        try:
            d = (dt.date.fromisoformat(rts) - dt.date.fromisoformat(obs)).days
        except Exception:  # noqa: BLE001
            bad.append(obs)
            continue
        lags[d] = lags.get(d, 0) + 1
        if d <= 0:
            bad.append(obs)
    ent["initial_release_rows"] = {
        "n": len(rows),
        "first": min(rows) if rows else None,
        "last": max(rows) if rows else None,
        "calendar_day_lag_histogram": {str(k): v for k, v in sorted(lags.items())},
        "nonpositive_or_unparsable_lag": bad,
    }
    # retain the merged per-observation table
    outcsv = os.path.join(DL, "%s_realtime_start_2019_2024.csv" % seid)
    with open(outcsv, "w") as f:
        f.write("observation_date,initial_value,realtime_start_date\n")
        for obs in sorted(rows):
            f.write("%s,%s,%s\n" % (obs, rows[obs][0], rows[obs][1]))
    with open(outcsv, "rb") as f:
        ent["merged_table"] = {
            "file": os.path.basename(outcsv),
            "sha256": hashlib.sha256(f.read()).hexdigest(),
            "rows": len(rows),
        }
    manifest["series"][seid] = ent

with open(os.path.join(DL, "ALFRED_PROVENANCE.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
print("DONE")
