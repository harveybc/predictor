"""Bounded public quote audit of the proposed fixed-EST source clock.

Only eighteen predetermined pre-2025 UTC days are downloaded, once each.
HTTP transport is unauthenticated; hashes bind artifacts, not the provider.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
import time
from zoneinfo import ZoneInfo

from tools import i6n_source_recovery as source


MAX_BYTES = 1_000_000
CLOCK_POLICY = "FIXED_EST_BAR_START"
BASE = "http://datafeed.dukascopy.com/datafeed/EURUSD"


def planned_dates():
    result = []
    for year in range(2019, 2025):
        for month in (1, 3, 7):
            day = date(year, month, 15)
            result.append(day + timedelta(days=(-day.weekday()) % 7))
    return result


def _date(day):
    if type(day) is not date or not 2019 <= day.year <= 2024:
        raise ValueError("only 2019-2024 dates permitted; TEST never downloaded")


def fetch(session, day, deadline):
    _date(day)
    if time.monotonic() + 20 >= deadline:
        raise TimeoutError("insufficient bounded request time; no attempt")
    url = f"{BASE}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    with session.get(url, timeout=(5, 15), stream=True, allow_redirects=False) as response:
        if response.status_code != 200:
            raise ValueError(f"expected HTTP 200, got {response.status_code}; no retry")
        if int(response.headers.get("Content-Length", "0")) > MAX_BYTES:
            raise ValueError("response exceeds byte limit")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            if time.monotonic() >= deadline:
                raise TimeoutError("download deadline exceeded")
            if len(body) + len(chunk) > MAX_BYTES:
                raise ValueError("response exceeds byte limit")
            body.extend(chunk)
    return bytes(body), url


def decode(data, day):
    import pandas as pd
    _date(day)
    if len(data) > MAX_BYTES:
        raise ValueError("compressed byte limit exceeded")
    decoder = lzma.LZMADecompressor(memlimit=64 * 1024**2)
    raw = decoder.decompress(data, max_length=1440 * 24 + 1)
    if len(raw) != 1440 * 24 or not decoder.eof or decoder.unused_data:
        raise ValueError("expected one complete bounded 1440-minute LZMA stream")
    records = list(struct.iter_unpack(">5if", raw))
    for minute, (seconds, opened, closed, low, high, volume) in enumerate(records):
        if seconds != minute * 60:
            raise ValueError("UTC minute timestamps must cover the ordered full day")
        if not (0 < low <= min(opened, closed) <= max(opened, closed) <= high):
            raise ValueError("invalid OHLC price ordering")
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("invalid volume")
    midnight = pd.Timestamp(day, tz="UTC")
    # Both sources use minute bar starts; comparison uses hourly bar ends.
    return pd.Series([records[h * 60 + 59][2] / 1e5 for h in range(24)],
                     index=[midnight + pd.Timedelta(hours=h + 1) for h in range(24)])


def _metrics(reference, candidate):
    paired = candidate.reindex(reference.index)
    valid = paired.notna()
    error = (paired[valid] - reference[valid]).abs()
    return dict(overlap=int(valid.sum()), matched_1e12=int((error <= 1e-12).sum()),
                mean_abs_error=float(error.mean()) if len(error) else None,
                max_abs_error=float(error.max()) if len(error) else None)


def compare_case(day, reference, fixed_est, legacy_ny):
    _date(day)
    fixed, ny = _metrics(reference, fixed_est), _metrics(reference, legacy_ny)
    winter = not bool(datetime.combine(day, datetime.min.time(),
                                      ZoneInfo("America/New_York")).dst())
    exact_fixed = fixed["overlap"] == fixed["matched_1e12"] == 24
    exact_ny = ny["overlap"] == ny["matched_1e12"] == 24
    passed = exact_fixed and ny["overlap"] == 24 and (
        exact_ny if winter else fixed["mean_abs_error"] < ny["mean_abs_error"])
    observed = ("indistinguishable" if exact_fixed and exact_ny else
                "fixed_est" if exact_fixed else "legacy_ny" if exact_ny else "neither_exact")
    return dict(date=day.isoformat(), status="PASS" if passed else "INCONCLUSIVE",
                season="ny_standard" if winter else "ny_dst",
                observed_exact_policy=observed, fixed_est=fixed, legacy_ny=ny,
                reason=None if passed else "frozen exact-match/season rule not satisfied")


def terminal_status(cases):
    expected = [day.isoformat() for day in planned_dates()]
    return "VERIFIED" if ([case.get("date") for case in cases] == expected
                          and all(case.get("status") == "PASS" for case in cases)) else "INCONCLUSIVE"


def _write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def _save(path, value):
    _write(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def seal_report(output, report):
    _save(output / "REPORT.json", report)
    _write(output / "REPORT.sha256",
           (source.digest((output / "REPORT.json").read_bytes()) + "\n").encode())


def _read(path, limit):
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds byte limit: {path.name}")
    return data


def run(annual_root, output):
    import requests
    start = time.monotonic()
    deadline = start + 260
    manifest_bytes = _read(annual_root / "MANIFEST.json", MAX_BYTES)
    manifest = json.loads(manifest_bytes)
    annual = {}
    for item in manifest["attempts"]:
        if item.get("status") == "DOWNLOADED" and item.get("month") in {str(y) for y in range(2019, 2025)}:
            if item["month"] in annual:
                raise ValueError("duplicate annual manifest period")
            annual[item["month"]] = item
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(schema="i6n.external_clock_audit.v1", status="INCONCLUSIVE",
                  clock_policy=CLOCK_POLICY, input_manifest_sha256=source.digest(manifest_bytes),
                  planned_dates=[d.isoformat() for d in planned_dates()], cases=[],
                  acceptance="all 18 dates: 24/24 closes within 1e-12; winter equal; DST fixed MAE < NY MAE",
                  transport="public HTTP; unauthenticated; hashes are not cryptographic provider attestation",
                  independent_coverage="18 sampled UTC days only, not all historical minutes",
                  test_read=False, test_downloaded=False, all_history_proven=False,
                  max_bytes_per_download=MAX_BYTES, timeout_connect=5, timeout_read=15,
                  retries=0, started_at=datetime.now(timezone.utc).isoformat(),
                  script_sha256=source.digest(Path(__file__).read_bytes()),
                  source_helper_sha256=source.digest(Path(source.__file__).read_bytes()))
    _save(output / "ATTEMPT.json", report)
    with requests.Session() as session:
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        for year in range(2019, 2025):
            fixed = ny = None
            archive_hash = None
            source_error = None
            try:
                item = annual[str(year)]
                filename = f"HISTDATA_COM_ASCII_EURUSD_M1_{year}.zip"
                if item["file"] != filename:
                    raise ValueError("unexpected annual filename")
                data = _read(annual_root / filename, 30_000_000)
                archive_hash = source.digest(data)
                if archive_hash != item["zip_sha256"]:
                    raise ValueError("annual ZIP does not match manifest pin")
                hourly, _ = source.hourly_closes(data, str(year))
                fixed = source.clock_series(hourly, "fixed_est")
                ny = source.clock_series(hourly, "legacy_ny")
            except Exception as exc:
                source_error = f"{type(exc).__name__}: {exc}"
            for day in [d for d in planned_dates() if d.year == year]:
                case = dict(date=day.isoformat(), status="INCONCLUSIVE",
                            source_zip_sha256=archive_hash, dukascopy_sha256=None)
                try:
                    if source_error:
                        raise ValueError(source_error)
                    body, url = fetch(session, day, deadline)
                    case.update(url=url, dukascopy_sha256=source.digest(body), downloaded_bytes=len(body))
                    _write(output / f"{day.isoformat()}.bi5", body)
                    reference = decode(body, day)
                    case.update(compare_case(day, reference, fixed, ny))
                except Exception as exc:
                    case.update(reason=f"{type(exc).__name__}: {exc}")
                _save(output / f"{day.isoformat()}.json", case)
                report["cases"].append(case)
                print(json.dumps(case, sort_keys=True), flush=True)
    report.update(status=terminal_status(report["cases"]),
                  elapsed_seconds=time.monotonic() - start,
                  completed_at=datetime.now(timezone.utc).isoformat())
    seal_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.annual_root, args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output / "REPORT.json")}, sort_keys=True))
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
