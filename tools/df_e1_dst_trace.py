#!/usr/bin/env python3
"""RP28: trace the DST discrepancy of the electricity family from the PRESERVED raw archive through the
parser to the canonical panel, over the change-day ranges only (no download, no re-census). The producer
states 96 rows per day, one hour of zeros on the March change day and two hours aggregated into one in
October; the earlier note wrote '23/25 records', which was wrong: they are hours of 23/25-hour days.
The script reads the raw rows of 00:00-06:00 on the last Sunday of March and the last Sunday of October of
every year, the same rows of the panel, and reports per row how many clients are zero, whether the raw and
the panel agree, the digests of both files, and the label semantics that can be inferred (label offsets,
interval end vs start) — and what cannot (UTC, publication/reception, revisions).

    python tools/df_e1_dst_trace.py --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path.home() / ".local/share/crispdm-data-foundation/public_raw_c126_20260912/data/uci_321_electricityloaddiagrams20112014__electricityloaddiagrams20112014.zip"
PANEL = Path.home() / ".local/state/crispdm-data-foundation/public_panels_c126_v2/uci_321_electricityloaddiagrams20112014"
RAW_SHA = "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1"


def last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1) if month < 12 else pd.Timestamp(year=year, month=12, day=31)
    while d.weekday() != 6:
        d -= pd.Timedelta(days=1)
    return d.normalize()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
    zf = zipfile.ZipFile(RAW)
    raw = zf.read("LD2011_2014.txt")
    member_sha = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8").splitlines()
    header = lines[0]
    labels = [ln.split(";", 1)[0].strip('"') for ln in lines[1:]]
    idx = {lab: i for i, lab in enumerate(labels)}
    panel = pd.read_parquet(PANEL / "panel.parquet")
    receipt = json.loads((PANEL / "PARSE_RECEIPT.json").read_text())
    pl = {lab: i for i, lab in enumerate(panel["timestamp_label"].tolist())}
    num = panel.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    out = {"schema": "e1_dst_trace.v1", "raw_archive": str(RAW), "raw_sha256": raw_sha, "raw_sha256_matches_contract": raw_sha == RAW_SHA,
           "member_sha256": member_sha, "member_sha256_matches_receipt": receipt.get("member_sha256", {}).get("LD2011_2014.txt") == member_sha,
           "rows_raw": len(labels), "rows_panel": int(len(panel)), "header_first_columns": header[:80], "days": {}, "per_day_counts": {}}
    for year in (2011, 2012, 2013, 2014):
        for month, tag in ((3, "march"), (10, "october")):
            day = last_sunday(year, month)
            rows = []
            for hh in range(0, 7):
                for mm in (15, 30, 45, 0):
                    t = day + pd.Timedelta(hours=hh, minutes=mm) if mm else day + pd.Timedelta(hours=hh + 1)
                    lab = t.strftime("%Y-%m-%d %H:%M:%S")
                    if lab not in idx:
                        rows.append({"label": lab, "raw": "ABSENT", "panel": "ABSENT" if lab not in pl else "PRESENT"})
                        continue
                    fields = lines[1 + idx[lab]].split(";")[1:]
                    vals = np.array([float(f.replace(",", ".")) if f not in ("", "NA") else np.nan for f in fields])
                    pv = num[pl[lab]] if lab in pl else None
                    rows.append({"label": lab, "raw_zero_clients": int((vals == 0).sum()), "raw_nonzero_clients": int((vals != 0).sum()),
                                 "panel_zero_clients": int((pv == 0).sum()) if pv is not None else None,
                                 "raw_equals_panel": bool(pv is not None and np.allclose(np.nan_to_num(vals), np.nan_to_num(pv)))})
            counts = int(sum(1 for lab in labels if lab.startswith(day.strftime("%Y-%m-%d"))))
            out["days"][f"{year}_{tag}"] = {"day": str(day.date()), "rows_00_to_07": rows, "raw_rows_that_day": counts}
            out["per_day_counts"][f"{year}_{tag}"] = counts
    # what the labels can and cannot say
    out["semantics"] = {"label_meaning": "the file labels each row with the END of its 15-minute interval (first row 2011-01-01 00:15:00, last 2015-01-01 00:00:00); the producer calls the values kW averages of the interval",
                        "rows_per_day": "96 labels per calendar day in the raw file and in the panel (see per_day_counts): the change days are NOT 92/100 rows; the producer's 'one hour of zeros' / 'two hours aggregated' are VALUE semantics inside a 96-row day",
                        "utc": "UNKNOWN: labels are Portuguese wall clock by the producer's statement; no UTC offset is stored; the March 02:xx labels denote a wall-clock hour that did not exist",
                        "reception_publication_revisions": "UNKNOWN: a static archive carries no reception, publication or revision history; nothing is inferred",
                        "disposition": "rows of the March change day between 01:00 and 03:00 wall clock and of the October change day between 01:00 and 02:00 are AMBIGUOUS_SUPPORT in the task contract: excluded from window support, with the count of excluded windows reported; no DST flag replaces this"}
    args.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("raw_sha256_matches_contract", "member_sha256_matches_receipt", "rows_raw", "rows_panel", "per_day_counts")}, indent=1))
    for k, d in out["days"].items():
        zs = [(r["label"][11:16], r.get("raw_zero_clients"), r.get("raw_equals_panel")) for r in d["rows_00_to_07"] if isinstance(r.get("raw_zero_clients"), int)]
        print(k, zs[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
