# Bounded Source-Clock Recovery

Date: 2026-10-11. Outcome: **BLOCKED; no source-price comparison obtained.**

## Public Download Attempt

Allowed scope was EURUSD ASCII M1 for March 2018, March 2019 and March 2024,
at most three ZIPs and 30,000,000 total ZIP bytes. March 2024 was attempted
first. Its official [monthly page](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2024/3)
returned HTTP 200. The returned page's `#a_file` click handler simply submits
`#file_down` to `/get.php`; the hidden `tk` field was empty. No token was
invented, harvested elsewhere, or substituted.

One normal same-session POST was made with the page as Referer and these fields:

```text
tk=  date=2024  datemonth=202403  platform=ASCII  timeframe=M1  fxpair=EURUSD
```

Result: HTTP 200, `Content-Type: text/html; charset=UTF-8`, **zero response
bytes**, no ZIP. Cause of the empty response is unknown; this is not evidence
that authentication or a CAPTCHA is required. No such challenge was observed.
The attempt stopped immediately, with no retries, alternate endpoints,
credentials, redirects, access-control workarounds, or waits. March 2018 and
March 2019 were **not attempted** after this failure. Two earlier read-only GETs
inspected the March 2024 form/scripts; neither attempted a download.

Total recovered: **0 ZIPs, 0 ZIP bytes**. This result does not establish whether
either unattempted month is publicly downloadable.

Private evidence is in `/var/tmp/i6n-source-clock-recovery/` (directory mode
0700; files 0600): `ATTEMPT.json`, `MANIFEST.json`, `202403.html`,
`202403.blocked-response`, and `COMPARISON.json`. The exclusive attempt marker
prevents accidental replay. Download content and manifests are not in git.

## Clock Evidence

The official [HistData FAQ](https://www.histdata.com/f-a-q/), inspected on the
date above, specifies fixed EST without daylight-saving adjustments. Thus
source wall time maps to UTC by adding five hours year-round.

Read-only inspection of the sibling financial-data worker
`_scripts/workers/stage13_omega_light_worker.py` found:

- `parse_histdata_zip`, lines 61-87, parses the naive source strings with
  `utc=True` (lines 74, 78, 82), attaching UTC without EST conversion.
- `process_histdata`, lines 122-125, sorts/deduplicates and resamples OHLC,
  selecting the last close in each left-labelled hourly bucket.
- Predictor `tools/i6e_price_metrics.py`, lines 17-26, removes that timezone,
  localizes to `America/New_York`, converts to UTC, and adds one hour to label
  the hourly bar end.

Consequently, **if the original files follow the documented HistData clock**,
the legacy New York policy is one hour early in UTC during New York DST and
agrees with fixed EST during standard time. This is a code/documentation
inference, not a recovered-data match. The absent original download directories
on three hosts are supplied context; this attempt did not inspect remote hosts.

## Retained Scope And Limits

Only `ts` and `price` were decoded from the supplied retained
`canonical_20261003/i6e/statistical72_v1/INPUT.npz`; `x`, `origins`, `weeks`,
and `future` were not decoded. All 77,977 timestamps were checked to be strictly
before 2025-01-01 UTC; epoch-second endpoints are 1335830400 and 1735682400.
No financial parquet was opened, filtered or otherwise. No training, TEST data
access, original-data mutation, sibling edits, deployment or commit occurred.

Retained-array SHA-256 scope is the contiguous canonical little-endian array
bytes, not the NPZ container, original market files, or full-history provenance:

```text
ts, int64:    5516566d4b9ac992dad7c640e836cfda3b8212d4b17b098b3c5eb40e3255dc8b
price, f64:   601c0d187e5fb4b7a9a35555159208141a87507a42eb5694954c05cd2a606a17
```

`COMPARISON.json` explicitly records `NO_RECOVERED_SAMPLES`, empty `months`,
and `all_history_proven=false`. **Neither matching source prices nor the
retained time policy was empirically established by this recovery.** Do not
promote these digests to source-provenance claims or relabel retained history
as fixed-EST-validated based on this attempt.

## Helper Verification

The isolated helper is `tools/i6n_source_recovery.py`; its focused tests are
`tests/test_i6n_source_recovery.py`. The comparison code supports recovered
monthly last-minute hourly closes under fixed EST, legacy New York, and literal
UTC, with UTC bar-end labels, exact timestamp intersections, a 1e-12 absolute
price-match tolerance, and separate New York DST/standard-time counts. Actual
downloaded-data comparisons were not run because no ZIP was recovered.
Synthetic tests exercise aggregation, DST offsets, scope guards, byte limits,
public-form handling, blocking, and no-retry behavior; they do not substitute
for source evidence.

The retained-only comparison command enforced 2 GiB address space plus 120
CPU/wall seconds, with BLAS/OpenMP limited to one thread. Observed: 0.50 seconds
elapsed, 123,124 KiB peak RSS. Even successful matching of the three requested
months would support only those bounded samples, never all-history provenance.

Focused test command (no training or network calls):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider tests/test_i6n_source_recovery.py
```

This is an independent return; no changes to the parent's weekly-data work.
