#!/usr/bin/env python3
"""Two bounded provenance probes.

(1) H.15 release-clock evidence from the public Federal Reserve pages:
    the release schedule / about page and the current-release RSS data feed
    (documented endpoints, no credential).
(2) Re-verification that no MT5 bridge lab database exists on this host, so
    the retained-live-spread claim in instrument config f2443141 can be ruled
    on from a fresh observation rather than a 2026-09-26 note.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

DL = sys.argv[1]
URLS = {
    "h15_index": "https://www.federalreserve.gov/releases/h15/",
    "h15_about": "https://www.federalreserve.gov/releases/h15/about.htm",
    "h15_current": "https://www.federalreserve.gov/releases/h15/current/default.htm",
    "h15_rss_data_current": "https://www.federalreserve.gov/feeds/Data/H15_H15.XML",
    "h15_archive_probe_2019": "https://www.federalreserve.gov/releases/h15/20190102/default.htm",
}

out = {"artifact": "H.15 release-clock evidence (public, unauthenticated)",
       "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "pages": {}}

for name, url in URLS.items():
    fn = os.path.join(DL, "h15_%s.html" % name)
    p = subprocess.run(["curl", "-sS", "--http1.1", "--max-time", "40",
                        "-o", fn, "-w", "%{http_code}", url],
                       capture_output=True, text=True)
    code = p.stdout.strip()
    raw = open(fn, "rb").read() if os.path.exists(fn) else b""
    rec = {"url": url, "http_status": code, "bytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest()}
    txt = raw.decode("utf8", "replace")
    rec["release_date_strings"] = sorted(set(
        re.findall(r"Release [Dd]ate:?\s*([A-Z][a-z]+ \d{1,2},? \d{4})", txt)))[:6]
    rec["clock_strings"] = sorted(set(
        re.findall(r"(\d{1,2}:\d{2}\s*[ap]\.?m\.?[^<\n]{0,20})", txt)))[:12]
    rec["schedule_sentences"] = [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))[:300]
        for s in re.findall(r"[^.<>]{0,200}(?:released|publish\w*|updated)[^.<>]{0,200}\.",
                            txt, re.I)[:8]]
    out["pages"][name] = rec
    print(name, code, len(raw), flush=True)
    time.sleep(1.5)

# (2) MT5 lab database re-verification
roots = [os.path.expanduser("~"), "/var/tmp", "/srv", "/opt", "/mnt", "/media", "/tmp"]
hits = []
for r in roots:
    if not os.path.isdir(r):
        continue
    p = subprocess.run(["find", r, "-xdev", "-maxdepth", "8", "-name",
                        "mt5-bridge.sqlite*", "-o", "-xdev", "-maxdepth", "8",
                        "-name", "*mt5*bridge*.sqlite*"],
                       capture_output=True, text=True)
    hits += [l for l in p.stdout.splitlines() if l.strip()]
out["mt5_lab_database_search"] = {
    "roots_searched": roots,
    "pattern": "mt5-bridge.sqlite* / *mt5*bridge*.sqlite*",
    "hits": hits,
    "verdict": "ABSENT" if not hits else "PRESENT",
}
print("mt5 hits:", hits, flush=True)

with open(os.path.join(DL, "H15_AND_MT5_PROBE.json"), "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print("DONE")
