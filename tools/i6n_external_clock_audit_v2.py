"""Separate EU-DST clock hypothesis: 17 frozen replays and six October holdouts."""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time

from tools import i6n_external_clock_audit as v1
from tools import i6n_source_recovery as source


CLOCK_POLICY = "EU_DST_OFFSET_MINUS5_BASE_BAR_START"


def holdout_dates():
    dates = []
    for year in range(2019, 2025):
        last = date(year, 10, 31)
        sunday = last - timedelta(days=(last.weekday() + 1) % 7)
        dates.append(sunday + timedelta(days=1))
    return dates


def eu_clock_series(hourly):
    import numpy as np
    import pandas as pd
    index = (hourly.index + pd.Timedelta(hours=5)).tz_localize(
        "Europe/London", ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    result = pd.Series(hourly.to_numpy(), index=index + pd.Timedelta(hours=1))
    legacy = source.clock_series(hourly, "legacy_ny")
    if (not len(result) or len(result) != len(legacy) or not result.index.is_unique
            or not result.index.is_monotonic_increasing):
        raise ValueError("annual clock mapping must be complete, unique and ordered")
    shifts = (result.index.asi8 - legacy.index.asi8) // 1_000_000_000
    if not np.isin(shifts, [0, 3600]).all():
        raise ValueError("EU mapping must never move earlier or more than one hour later")
    return result, dict(rows=len(result), never_moves_earlier=True,
                       minimum_shift_vs_legacy_seconds=int(shifts.min()),
                       maximum_shift_vs_legacy_seconds=int(shifts.max()),
                       moved_later_rows=int((shifts == 3600).sum()),
                       unique=True, ordered=True)


def compare_case(day, reference, candidate):
    import numpy as np
    import pandas as pd
    expected = pd.date_range(pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=1),
                             periods=24, freq="h")
    if not reference.index.equals(expected) or not candidate.index.is_unique:
        raise ValueError("invalid reference endpoints or duplicate source endpoints")
    if not np.isfinite(reference.to_numpy()).all() or not np.isfinite(candidate.to_numpy()).all():
        raise ValueError("prices must be finite; invalid prices are not missing endpoints")
    available = expected.intersection(candidate.index)
    missing = expected.difference(candidate.index)
    errors = (reference.loc[available] - candidate.loc[available]).abs()
    exact = int((errors == 0).sum())
    passed = len(available) >= 12 and exact == len(available)
    return dict(date=day.isoformat(), status="PASS" if passed else "INCONCLUSIVE",
                expected_endpoints=24, available_endpoints=len(available),
                absent_endpoint_count=len(missing),
                absent_endpoints_utc=[t.isoformat() for t in missing], matched_exact=exact,
                mean_abs_error=float(errors.mean()) if len(errors) else None,
                max_abs_error=float(errors.max()) if len(errors) else None,
                reason=None if passed else "requires >=12 available endpoints and exact equality on all")


def terminal_status(cases, planned, annual_ok):
    return "VERIFIED" if (annual_ok and [c.get("date") for c in cases] == planned
                          and all(c.get("status") == "PASS" for c in cases)
                          and set(d.isoformat() for d in holdout_dates()) <= set(planned)) else "INCONCLUSIVE"


def load_replay(root, manifest_hash):
    raw = v1._read(root / "REPORT.json", 2_000_000)
    digest = source.digest(raw)
    if (root / "REPORT.sha256").read_text().strip() != digest:
        raise ValueError("V1 report hash mismatch")
    report = json.loads(raw)
    if report["input_manifest_sha256"] != manifest_hash:
        raise ValueError("V1 report annual manifest mismatch")
    if [c["date"] for c in report["cases"]] != [d.isoformat() for d in v1.planned_dates()]:
        raise ValueError("unexpected V1 plan")
    replay = [c for c in report["cases"] if c.get("dukascopy_sha256")]
    excluded = [c for c in report["cases"] if not c.get("dukascopy_sha256")]
    if len(replay) != 17 or [c["date"] for c in excluded] != ["2019-01-21"]:
        raise ValueError("expected the frozen 17 downloaded V1 cases and January 2019 timeout")
    return replay, excluded, digest


def run(annual_root, replay_root, output):
    import requests
    start = time.monotonic()
    deadline = start + 260
    manifest_bytes = v1._read(annual_root / "MANIFEST.json", v1.MAX_BYTES)
    manifest_hash = source.digest(manifest_bytes)
    replay, excluded, replay_hash = load_replay(replay_root, manifest_hash)
    planned = [c["date"] for c in replay] + [d.isoformat() for d in holdout_dates()]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(schema="i6n.external_clock_audit.v2", status="INCONCLUSIVE",
        clock_policy=CLOCK_POLICY, input_manifest_sha256=manifest_hash,
        v1_report_sha256=replay_hash, replay_cases=17, new_holdout_cases=6,
        planned_dates=planned, excluded_v1_cases=excluded, cases=[], annual_mapping=[],
        acceptance="each of 23 cases: >=12 available endpoints; ALL observed closes exactly equal; missing endpoints explicit; all annual mappings ordered/unique and never earlier than NY",
        transport="public HTTP; unauthenticated; hashes are integrity pins, not provider attestation",
        independent_coverage="6 new October holdout days; 17 hypothesis-development replay days; not all historical minutes",
        test_read=False, test_downloaded=False, all_history_proven=False,
        max_bytes_per_download=v1.MAX_BYTES, timeout_connect=5, timeout_read=15, retries=0,
        started_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=source.digest(Path(__file__).read_bytes()),
        v1_helper_sha256=source.digest(Path(v1.__file__).read_bytes()),
        source_helper_sha256=source.digest(Path(source.__file__).read_bytes()))
    # This immutable plan precedes every new holdout request and comparison.
    v1._save(output / "ATTEMPT.json", report)
    annual = {}
    for item in json.loads(manifest_bytes)["attempts"]:
        if item.get("status") == "DOWNLOADED" and item.get("month") in {str(y) for y in range(2019, 2025)}:
            if item["month"] in annual:
                raise ValueError("duplicate annual manifest period")
            annual[item["month"]] = item
    sources, source_errors = {}, {}
    for year in range(2019, 2025):
        entry = dict(year=year, status="INCONCLUSIVE", source_zip_sha256=None)
        try:
            item = annual[str(year)]
            name = f"HISTDATA_COM_ASCII_EURUSD_M1_{year}.zip"
            if item["file"] != name:
                raise ValueError("unexpected annual filename")
            raw = v1._read(annual_root / name, 30_000_000)
            entry["source_zip_sha256"] = source.digest(raw)
            if entry["source_zip_sha256"] != item["zip_sha256"]:
                raise ValueError("annual archive hash mismatch")
            hourly, _ = source.hourly_closes(raw, str(year))
            sources[year], mapping = eu_clock_series(hourly)
            entry.update(mapping, status="PASS")
        except Exception as exc:
            source_errors[year] = f"{type(exc).__name__}: {exc}"
            entry["reason"] = source_errors[year]
        report["annual_mapping"].append(entry)
    saved = {c["date"]: c for c in replay}
    with requests.Session() as session:
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        for stamp in planned:
            day = date.fromisoformat(stamp)
            case = dict(date=stamp, status="INCONCLUSIVE",
                        evidence_role="replay" if stamp in saved else "holdout",
                        source_zip_sha256=report["annual_mapping"][day.year - 2019]["source_zip_sha256"],
                        dukascopy_sha256=None)
            try:
                if day.year in source_errors:
                    raise ValueError(source_errors[day.year])
                if stamp in saved:
                    raw = v1._read(replay_root / f"{stamp}.bi5", v1.MAX_BYTES)
                    if source.digest(raw) != saved[stamp]["dukascopy_sha256"]:
                        raise ValueError("replay artifact hash mismatch")
                    case["url"] = saved[stamp].get("url")
                else:
                    raw, case["url"] = v1.fetch(session, day, deadline)
                case.update(dukascopy_sha256=source.digest(raw), artifact_bytes=len(raw))
                v1._write(output / f"{stamp}.bi5", raw)
                case.update(compare_case(day, v1.decode(raw, day), sources[day.year]))
            except Exception as exc:
                case["reason"] = f"{type(exc).__name__}: {exc}"
            v1._save(output / f"{stamp}.json", case)
            report["cases"].append(case)
            print(json.dumps(case, sort_keys=True), flush=True)
    report.update(status=terminal_status(report["cases"], planned, not source_errors),
                  elapsed_seconds=time.monotonic() - start,
                  completed_at=datetime.now(timezone.utc).isoformat())
    v1.seal_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.annual_root, args.replay_root, args.output)
    print(json.dumps({"status": report["status"], "clock_policy": CLOCK_POLICY,
                      "report": str(args.output / "REPORT.json")}, sort_keys=True))
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
