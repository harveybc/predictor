"""Explicitly admitted transport-only successors, retaining every successful case."""

import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlsplit

from tools import i6n_external_clock_audit_v2 as v2

v1 = v2.v1
source = v2.source


def redirect_target(original, location):
    target = urljoin(original, location)
    parsed, start = urlsplit(target), urlsplit(original)
    if (not location or parsed.scheme != "https" or parsed.hostname != "datafeed.dukascopy.com"
            or parsed.hostname != start.hostname or parsed.port not in (None, 443)
            or parsed.username is not None or parsed.password is not None
            or parsed.path != start.path or parsed.query or parsed.fragment):
        raise ValueError("redirect must preserve the exact path on canonical HTTPS datafeed host")
    return target


def fetch_holdout(session, day, deadline, trace):
    if day not in v2.holdout_dates():
        raise ValueError("only the six frozen October holdouts may be requested")
    if time.monotonic() + 40 >= deadline:
        raise TimeoutError("insufficient bounded time for one HTTP/HTTPS redirect chain")
    url = f"{v1.BASE}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    for hop in range(2):
        trace.append(dict(url=url))
        with session.get(url, timeout=(5, 15), stream=True, allow_redirects=False) as response:
            trace[-1]["status"] = response.status_code
            if response.status_code == 200:
                return source.bounded_body(response, v1.MAX_BYTES, deadline)
            if hop == 0 and response.status_code in (301, 302, 307, 308):
                location = response.headers.get("Location", "")
                trace[-1]["location"] = location
                url = redirect_target(url, location)
            else:
                raise ValueError(f"HTTP {response.status_code}; no retry, alternate host or further redirect")
    raise ValueError("redirect chain exhausted")


# HTTP 429 (rate limited) carries no quote bytes: a transport refusal, not a
# numeric result. It is retried only as an explicitly named single case.
RATE_LIMITED = "ValueError: HTTP 429; no retry, alternate host or further redirect"


#: Dates whose 429 may be retried, and how many single-case retries after the first 429.
#: 2019-10-28: one owner-authorized retry (also 429). It is used, the span is a declared
#: assumption (DCA_2019_EU_DST_POLICY_2019-03-11_2019-12-01), so a further attempt is refused.
RATE_LIMITED_RETRIES = {"2019-10-28": 1}


def rate_limited(case):
    return case.get("reason") == RATE_LIMITED


def count_rate_limited(reports):
    """Number of HTTP 429 outcomes per date across a report chain (prior + ancestors)."""
    counts = {}
    for report in reports:
        for case in report.get("cases", []):
            if rate_limited(case):
                counts[case["date"]] = counts.get(case["date"], 0) + 1
    return counts


def rate_limited_retry_admitted(day, prior_rate_limited):
    """True only for a named date whose retry budget is not yet used (first 429 is not a retry)."""
    return day in RATE_LIMITED_RETRIES and prior_rate_limited - 1 < RATE_LIMITED_RETRIES[day]


def chain_reports(prior_root):
    """The prior report and every ancestor bundled in its report_chain (read-only)."""
    reports = [json.loads(v1._read(prior_root / "REPORT.json", 2_000_000))]
    chain = prior_root / "report_chain"
    if chain.is_dir():
        for path in sorted(chain.glob("*.json")):
            raw = v1._read(path, 2_000_000)
            if source.digest(raw) != path.stem:
                raise ValueError("ancestor report SHA-256 mismatch")
            reports.append(json.loads(raw))
    return reports


def transport_failure(case):
    if (case.get("status") != "INCONCLUSIVE" or case.get("evidence_role") != "holdout"
            or case.get("dukascopy_sha256") is not None
            or any(k in case for k in ("available_endpoints", "matched_exact", "mean_abs_error", "max_abs_error"))):
        return False
    reason = case.get("reason", "")
    return reason.startswith(("ReadTimeout:", "ConnectTimeout:", "TimeoutError:")) or (
        reason == "ValueError: expected HTTP 200, got 301; no retry") or reason == RATE_LIMITED


def pending_cases(prior, only_dates=None, rate_limited_history=None):
    expected = [d.isoformat() for d in v1.planned_dates() if d.isoformat() != "2019-01-21"]
    expected += [d.isoformat() for d in v2.holdout_dates()]
    cases = prior["cases"]
    if prior["planned_dates"] != expected or [c["date"] for c in cases] != expected:
        raise ValueError("incomplete or reordered V2 case plan")
    if any(c["evidence_role"] != ("replay" if i < 17 else "holdout") for i, c in enumerate(cases)):
        raise ValueError("incorrect replay/holdout groups")
    pending = [c for c in cases if c["status"] != "PASS"]
    if not pending or any(not transport_failure(c) for c in pending):
        raise ValueError("only transport failures may be retried; numeric mismatches are final")
    if only_dates is not None and [c["date"] for c in pending] != list(only_dates):
        raise ValueError("pending dates differ from the explicit network authorization")
    if any(rate_limited(c) for c in pending) and (only_dates is None or len(pending) != 1):
        raise ValueError("a rate-limited case needs an explicit single-case --only-date authorization")
    history = rate_limited_history if rate_limited_history is not None else count_rate_limited([prior])
    for case in pending:
        if not rate_limited(case):
            continue
        if case["date"] not in RATE_LIMITED_RETRIES:
            raise ValueError(f"RATE_LIMIT_DATE_NOT_ADMITTED: {case['date']}")
        if not rate_limited_retry_admitted(case["date"], history.get(case["date"], 0)):
            raise ValueError(f"RATE_LIMIT_RETRY_EXHAUSTED: {case['date']} already has "
                             f"{history.get(case['date'], 0)} HTTP 429 outcomes in the chain")
    return pending


def run(annual_root, prior_root, output, *, only_dates=None, wall_seconds=260, chain_roots=()):
    import requests
    start = time.monotonic()
    deadline = start + wall_seconds
    prior_bytes = v1._read(prior_root / "REPORT.json", 2_000_000)
    prior_hash = source.digest(prior_bytes)
    if (prior_root / "REPORT.sha256").read_text().strip() != prior_hash:
        raise ValueError("prior report hash mismatch")
    prior = json.loads(prior_bytes)
    manifest_bytes = v1._read(annual_root / "MANIFEST.json", v1.MAX_BYTES)
    if source.digest(manifest_bytes) != prior["input_manifest_sha256"]:
        raise ValueError("annual manifest pin changed")
    if prior["clock_policy"] != v2.CLOCK_POLICY or prior["status"] != "INCONCLUSIVE":
        raise ValueError("expected the inconclusive EU-clock V2 attempt")
    pending = pending_cases(prior, only_dates, rate_limited_history=count_rate_limited(chain_reports(prior_root)))
    if any(m["status"] != "PASS" for m in prior["annual_mapping"]):
        raise ValueError("annual mapping failed")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = deepcopy(prior)
    report.update(schema="i6n.external_clock_audit.v2.transport_resume", cases=[],
                  prior_v2_report_sha256=prior_hash, started_at=datetime.now(timezone.utc).isoformat(),
                  download_plan=[c["date"] for c in pending],
                  transport="HTTP with at most one exact-path HTTPS redirect on datafeed.dukascopy.com; no www, retries, auth or alternate hosts",
                  script_sha256=source.digest(Path(__file__).read_bytes()),
                  v2_helper_sha256=source.digest(Path(v2.__file__).read_bytes()))
    report.pop("completed_at", None)
    report.pop("elapsed_seconds", None)
    from tools import i6n_external_clock_audit_v2_verify as gate
    gate.bundle_chain(output, (prior_root, *chain_roots))
    v1._save(output / "ATTEMPT.json", report)
    # Carry successes of either role byte-for-byte, without requesting them.
    successful = [c for c in prior["cases"] if c["status"] == "PASS"]
    for case in successful:
        raw = v1._read(prior_root / f"{case['date']}.bi5", v1.MAX_BYTES)
        if source.digest(raw) != case["dukascopy_sha256"]:
            raise ValueError("prior replay byte hash mismatch")
        v1._write(output / f"{case['date']}.bi5", raw)
        v1._save(output / f"{case['date']}.json", case)
        report["cases"].append(deepcopy(case))
    annual = {a["month"]: a for a in json.loads(manifest_bytes)["attempts"] if a.get("status") == "DOWNLOADED"}
    with requests.Session() as session:
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        for old in pending:
            day = date.fromisoformat(old["date"])
            case = dict(date=day.isoformat(), evidence_role="holdout", status="INCONCLUSIVE",
                        source_zip_sha256=old["source_zip_sha256"], dukascopy_sha256=None,
                        transport_trace=[])
            try:
                name = f"HISTDATA_COM_ASCII_EURUSD_M1_{day.year}.zip"
                if annual[str(day.year)]["file"] != name:
                    raise ValueError("unexpected annual filename")
                archive = v1._read(annual_root / name, 30_000_000)
                digest = source.digest(archive)
                if digest != case["source_zip_sha256"] or digest != annual[str(day.year)]["zip_sha256"]:
                    raise ValueError("annual archive hash changed")
                hourly, _ = source.hourly_closes(archive, str(day.year))
                candidate, _ = v2.eu_clock_series(hourly)
                raw = fetch_holdout(session, day, deadline, case["transport_trace"])
                case.update(dukascopy_sha256=source.digest(raw), artifact_bytes=len(raw))
                v1._write(output / f"{day.isoformat()}.bi5", raw)
                case.update(v2.compare_case(day, v1.decode(raw, day), candidate))
            except Exception as exc:
                case["reason"] = f"{type(exc).__name__}: {exc}"
            v1._save(output / f"{day.isoformat()}.json", case)
            report["cases"].append(case)
            print(json.dumps(case, sort_keys=True), flush=True)
    by_date = {c["date"]: c for c in report["cases"]}
    report["cases"] = [by_date[stamp] for stamp in report["planned_dates"]]
    report.update(status=v2.terminal_status(report["cases"], report["planned_dates"], True),
                  elapsed_seconds=time.monotonic() - start,
                  completed_at=datetime.now(timezone.utc).isoformat())
    gate.add_evidence_pins(report)
    v1.seal_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-date", action="append", dest="only_dates")
    parser.add_argument("--wall-seconds", type=int, default=260, choices=range(45, 261))
    parser.add_argument("--chain-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = run(args.annual_root, args.prior_root, args.output,
                 only_dates=args.only_dates, wall_seconds=args.wall_seconds,
                 chain_roots=args.chain_root)
    print(json.dumps({"status": report["status"], "clock_policy": v2.CLOCK_POLICY,
                      "report": str(args.output / "REPORT.json")}, sort_keys=True))
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
