"""I6-N lane C: can the clocked input be extended before the Dec-2019 scope? (no download).

The 2023 Q1-Q3 inner folds need input context from March 2019, before the
existing SCOPED_ACCEPTED clock certificate (2019-12-01 .. 2025-01-01). This
tool checks, with existing local sources and the existing successor pipeline
pieces only, what an extension would require:

1. Data: retained legacy-clock rows of the source input in [start, end) must
   reconcile exactly, row for row, with the already downloaded and
   digest-verified annual HistData archive (same rule as
   ``tools.i6n_clock_successor.build``), and the successor clock correction
   must apply without collision or earlier availability.
2. Clock: every PLANNED independent clock case dated inside [start, end) in the
   retained external audit must be PASS (same standard as the existing
   certificate, which required 20/20 in its scope).

It never downloads, never writes an input NPZ and never issues a certificate.
It writes one exclusive REPORT.json with a typed state and the exact missing
objects. Price/feature values of pre-2025 development rows are read only to
reconcile prices and to report per-feature availability fractions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

UTC = timezone.utc
SCHEMA = "i6n.weekly_history_extension.v1"


class ExtensionError(ValueError):
    """Typed refusal with a stable ``code``."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat()


def clock_decision(cases: Sequence[Mapping[str, Any]], start: datetime, end: datetime) -> Dict[str, Any]:
    """Typed clock state of [start, end) from the planned independent cases."""
    lo, hi = start.date().isoformat(), end.date().isoformat()
    in_span = [{"date": c["date"], "status": c.get("status"), "reason": c.get("reason")}
               for c in cases if lo <= c["date"] < hi]
    missing: List[Dict[str, Any]] = [
        {"object": "INDEPENDENT_CLOCK_CASE", "date": c["date"], "status": c["status"], "reason": c["reason"]}
        for c in in_span if c["status"] != "PASS"]
    if not in_span:
        missing.append({"object": "INDEPENDENT_CLOCK_CASE", "date": None, "status": "ABSENT",
                        "reason": "no planned independent case inside the span"})
    missing.append({"object": "CLOCK_SCOPE_CERTIFICATE", "start": _iso(start), "end": _iso(end),
                    "detail": "no scope certificate covers this span; this tool never issues one"})
    passed = bool(in_span) and len(missing) == 1
    return {"state": "CLOCK_CASES_PASS_CERTIFICATE_NOT_ISSUED" if passed else "WAIT_DEPENDENCY",
            "in_span": in_span, "missing": missing}


def reconcile_rows(ts, price, legacy) -> Dict[str, Any]:
    """Exact row-for-row match of retained (legacy clock) rows with the source series."""
    import numpy as np
    seconds = legacy.index.as_unit("s").asi8 if hasattr(legacy.index, "as_unit") else legacy.index.asi8 // 10 ** 9
    if len(seconds) != len(ts) or not np.array_equal(np.asarray(seconds, dtype=np.int64), np.asarray(ts)):
        raise ExtensionError("ROW_SET_MISMATCH", f"{len(ts)} retained vs {len(seconds)} source rows")
    diff = np.abs(np.asarray(price, dtype=float) - legacy.to_numpy(dtype=float))
    if not np.all(diff <= 1e-12):
        raise ExtensionError("PRICE_MISMATCH", f"max {float(diff.max())}")
    return {"retained_rows": int(len(ts)), "source_rows": int(len(seconds)), "matched": int(len(ts)),
            "max_abs_price_difference": float(diff.max()) if len(diff) else 0.0}


def feature_availability(x, names: Sequence[str]) -> Dict[str, float]:
    """Fraction of finite values per feature (availability only)."""
    import numpy as np
    finite = np.isfinite(np.asarray(x, dtype=float))
    return {n: float(finite[:, j].mean()) for j, n in enumerate(names)}


def successor_suffix_index(full_ts, full_x, full_price, succ_ts, succ_x, succ_price) -> int:
    """Index k where the existing successor equals full[k:] exactly (NaN-aware)."""
    import numpy as np
    k = len(full_ts) - len(succ_ts)
    if (k < 0 or not np.array_equal(full_ts[k:], succ_ts)
            or not np.array_equal(full_price[k:], succ_price)
            or not np.array_equal(full_x[k:], succ_x, equal_nan=True)):
        raise ExtensionError("SUCCESSOR_SUFFIX_MISMATCH", "extension is not a strict prefix of the successor")
    return int(k)


def declared_assumption(cases, replay, start: datetime, end: datetime, *, attempts) -> Dict[str, Any]:
    """Declare (never certify) the unverified in-span clock case, as December 2019 was declared."""
    decision = clock_decision(cases, start, end)
    passing = [c for c in decision["in_span"] if c["status"] == "PASS"]
    unverified = [c for c in decision["in_span"] if c["status"] != "PASS"]
    if not unverified:
        raise ExtensionError("NO_ASSUMPTION_NEEDED", "all planned in-span cases pass")
    if len(unverified) != 1 or not attempts or not passing:
        raise ExtensionError("ASSUMPTION_NOT_ADMISSIBLE", "exactly one unverified case, its attempts "
                                                          "and at least one passing case are required")
    replayed = {r["date"] for r in replay if r.get("status") == "PASS" and r.get("matches_retained_case")}
    if {c["date"] for c in passing} - replayed:
        raise ExtensionError("REPLAY_MISSING", "every passing in-span case must replay exactly")
    u = unverified[0]
    october = [c["date"] for c in cases if c["date"][5:7] in ("10", "11") and c["date"] >= "2020"
               and c.get("status") == "PASS"]
    body = {
        "status": "DECLARED_ASSUMPTION",
        "assumption_id": "DCA_2019_EU_DST_POLICY_" + start.date().isoformat() + "_" + end.date().isoformat(),
        "span": {"start": _iso(start), "end": _iso(end)},
        "unverified_case": u, "unverified_case_attempt_report_sha256": list(attempts),
        "passing_cases": passing, "passing_case_replay": [r for r in replay if r["date"] in replayed],
        "analogous_october_holdouts_pass": october,
        "limitation": (f"NOT independently quote-verified: the EU-DST clock policy over {_iso(start)}.."
                       f"{_iso(end)}, in particular the EU/US DST-gap week covered by planned case "
                       f"{u['date']} ({u['status']}: {u['reason']}; {len(attempts)} attempt reports). "
                       "Declared, as the December-2019 warmup was declared, on the basis of the passing "
                       "in-span cases replayed exactly and the passing October holdouts of later years."),
    }
    return dict(body, assumption_sha256=_digest(body))


def _digest(value: Any) -> str:
    import hashlib
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def run(*, source: Path, source_sha256: str, archives: Path, clock_audit: Path, design: Path,
        start: datetime, end: datetime, successor_input: Path, successor_sha256: str,
        output: Path) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    from tools.i6m_data import file_hash, load_input
    from tools.i6n_clock_successor import corrected_timestamps, validate_clock_audit
    from tools.i6n_external_clock_audit import decode
    from tools.i6n_external_clock_audit_v2 import compare_case, eu_clock_series
    from tools.i6n_source_recovery import clock_series, hourly_closes

    if end > datetime(2019, 12, 1, tzinfo=UTC) or start >= end or start.year != 2019:
        raise ExtensionError("SPAN_INVALID", "only a 2019 span before the existing scope")
    audit = validate_clock_audit(clock_audit, archives / "MANIFEST.json")
    manifest = json.loads((archives / "MANIFEST.json").read_text())
    item = next(a for a in manifest["attempts"] if a["month"] == "2019")
    archive = archives / item["file"]
    if file_hash(archive) != item["zip_sha256"]:
        raise ExtensionError("ARCHIVE_DIGEST_MISMATCH")
    hourly, info = hourly_closes(archive.read_bytes(), "2019")
    legacy = clock_series(hourly, "legacy_ny")
    legacy = legacy[(legacy.index >= pd.Timestamp(start)) & (legacy.index < pd.Timestamp(end))]
    ts, x, price = load_input(source, source_sha256)
    keep = (ts >= int(start.timestamp())) & (ts < int(end.timestamp()))
    rows = reconcile_rows(ts[keep], price[keep], legacy)
    corrected = corrected_timestamps(ts[keep])
    with np.load(successor_input, allow_pickle=False) as z:
        if file_hash(successor_input) != successor_sha256:
            raise ExtensionError("SUCCESSOR_DIGEST_MISMATCH")
        successor_first = int(z["ts"][0])
    if corrected.max() >= successor_first:
        raise ExtensionError("COLLIDES_WITH_SUCCESSOR")
    replay = []
    candidate, _ = eu_clock_series(hourly)
    for c in audit["cases"]:
        if c["status"] == "PASS" and start.date().isoformat() <= c["date"] < end.date().isoformat():
            day = datetime.fromisoformat(c["date"]).date()
            ref = decode((clock_audit.parent / (c["date"] + ".bi5")).read_bytes(), day)
            again = compare_case(day, ref, candidate)
            replay.append({"date": c["date"], "status": again["status"],
                           "matches_retained_case": all(again[k] == c[k] for k in again)})
    decision = clock_decision(audit["cases"], start, end)
    features = json.loads(design.read_text())["members"]
    report = {
        "schema": SCHEMA, "span": {"start": _iso(start), "end": _iso(end)},
        "state": decision["state"], "missing": decision["missing"], "clock_cases_in_span": decision["in_span"],
        "pass_case_replay": replay, "full_audit_status": audit["status"],
        "data": {**rows, "shifted_rows": int((corrected != ts[keep]).sum()),
                 "corrected_first": _iso(datetime.fromtimestamp(int(corrected.min()), UTC)),
                 "corrected_last": _iso(datetime.fromtimestamp(int(corrected.max()), UTC)),
                 "successor_first": _iso(datetime.fromtimestamp(successor_first, UTC)),
                 "archive": {"year": "2019", "sha256": item["zip_sha256"], **info},
                 "feature_finite_fraction": feature_availability(x[keep], features),
                 "state": "RECONCILED_NOT_MATERIALIZED"},
        "pins": {"source_sha256": source_sha256, "archive_manifest_sha256": file_hash(archives / "MANIFEST.json"),
                 "clock_audit_sha256": file_hash(clock_audit), "design_sha256": file_hash(design),
                 "successor_sha256": successor_sha256},
        "claims": {"downloaded": False, "npz_written": False, "certificate_issued": False,
                   "test_read": False},
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "REPORT.json").open("x") as stream:
        json.dump(report, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def materialize(*, source: Path, source_sha256: str, archives: Path, clock_audit: Path,
                attempt_reports: Sequence[Path], design: Path, successor: Path, start: datetime,
                output: Path) -> Dict[str, Any]:
    """Materialize the declared-assumption extension as a NEW successor directory.

    Every retained row with legacy ts >= start is reconciled with its annual
    archive (as ``i6n_clock_successor.build``); the existing successor must be
    an exact suffix of the result; the 2019 span carries a declared assumption.
    Writes INPUT.npz, CALENDAR.json (same bytes), CLOCK_SCOPE.json,
    PREPARATION.json and CLOCK.json exclusively in ``output``.
    """
    import numpy as np
    import pandas as pd
    from tools.i6m_data import file_hash, load_input
    from tools.i6n_clock_successor import corrected_timestamps, validate_clock_audit
    from tools.i6n_external_clock_audit import decode
    from tools.i6n_external_clock_audit_v2 import compare_case, eu_clock_series
    from tools.i6n_source_recovery import clock_series, hourly_closes

    end = datetime(2019, 12, 1, tzinfo=UTC)
    if start.year != 2019 or start >= end:
        raise ExtensionError("SPAN_INVALID")
    audit = validate_clock_audit(clock_audit, archives / "MANIFEST.json")
    manifest = json.loads((archives / "MANIFEST.json").read_text())
    ts, x, price = load_input(source, source_sha256)
    keep = ts >= int(start.timestamp())
    ts, x, price = ts[keep], x[keep], price[keep]
    matched, source_rows, records, hourly_2019 = set(), 0, [], None
    for item in manifest["attempts"]:
        year = item["month"]
        if len(year) != 4 or not 2019 <= int(year) <= 2024:
            raise ExtensionError("ARCHIVE_SET_INVALID")
        path = archives / item["file"]
        if file_hash(path) != item["zip_sha256"]:
            raise ExtensionError("ARCHIVE_DIGEST_MISMATCH", year)
        hourly, info = hourly_closes(path.read_bytes(), year)
        if year == "2019":
            hourly_2019 = hourly
        old = clock_series(hourly, "legacy_ny")
        old = old[old.index >= pd.Timestamp(start)]
        seconds = old.index.as_unit("s").asi8
        idx = np.searchsorted(ts, seconds)
        valid = (idx < len(ts)) & (ts[np.minimum(idx, len(ts) - 1)] == seconds)
        if not valid.all() or not np.allclose(price[idx], old.to_numpy(), atol=1e-12, rtol=0):
            raise ExtensionError("PRICE_MISMATCH", year)
        matched.update(idx.tolist())
        source_rows += len(old)
        records.append(dict(year=year, sha256=item["zip_sha256"], reconciled_retained_rows=len(old), **info))
    if matched != set(range(len(ts))) or source_rows != len(ts):
        raise ExtensionError("ROW_SET_MISMATCH", "archives do not cover every retained row exactly once")
    corrected = corrected_timestamps(ts)
    for name in ("INPUT.npz", "CALENDAR.json", "CLOCK.json", "CLOCK_SCOPE.json", "PREPARATION.json"):
        if not (successor / name).exists():
            raise ExtensionError("SUCCESSOR_INCOMPLETE", name)
    s_clock = json.loads((successor / "CLOCK.json").read_text())
    s_ts, s_x, s_price = load_input(successor / "INPUT.npz", s_clock["input_sha256"])
    k = successor_suffix_index(corrected, x, price, s_ts, s_x, s_price)
    candidate, _ = eu_clock_series(hourly_2019)
    replay = []
    for c in audit["cases"]:
        if c["status"] == "PASS" and start.date().isoformat() <= c["date"] < end.date().isoformat():
            day = datetime.fromisoformat(c["date"]).date()
            again = compare_case(day, decode((clock_audit.parent / (c["date"] + ".bi5")).read_bytes(), day),
                                 candidate)
            replay.append({"date": c["date"], "status": again["status"],
                           "matches_retained_case": all(again[key] == c[key] for key in again)})
    attempts = [file_hash(p) for p in attempt_reports]
    assumption = declared_assumption(audit["cases"], replay, start, end, attempts=attempts)
    features = json.loads(design.read_text())["members"]
    if features != s_clock["features"]:
        raise ExtensionError("FEATURE_ORDER_MISMATCH")
    calendar_bytes = (successor / "CALENDAR.json").read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "INPUT.npz", ts=corrected, x=x, price=price)
    (output / "CALENDAR.json").write_bytes(calendar_bytes)

    def write(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    base_scope = json.loads((successor / "CLOCK_SCOPE.json").read_text())
    scope = dict(status="SCOPED_ACCEPTED_WITH_DECLARED_ASSUMPTION", data_start=_iso(start),
                 data_end=base_scope["data_end"], assumption_id=assumption["assumption_id"],
                 assumption_sha256=assumption["assumption_sha256"], declared_assumption=assumption,
                 base_scope_sha256=file_hash(successor / "CLOCK_SCOPE.json"), base_scope=base_scope,
                 full_audit_status=audit["status"], full_audit_sha256=file_hash(clock_audit),
                 scope_basis="Context of the 2023 Q1-Q3 inner folds (four calendar years plus at most "
                             "432 h) before the base December-2019 scope")
    write("CLOCK_SCOPE.json", scope)
    report = dict(status="PREPARED", rows=int(len(corrected)), extension_rows=int(k),
                  shifted_rows=int((corrected != ts).sum()), source_input_sha256=source_sha256,
                  design_sha256=file_hash(design), base_successor_input_sha256=s_clock["input_sha256"],
                  base_successor_is_exact_suffix=True, archives=records,
                  input_sha256=file_hash(output / "INPUT.npz"), test_read=False,
                  clock_scope_sha256=file_hash(output / "CLOCK_SCOPE.json"),
                  inherited_feature_limits=json.loads((successor / "PREPARATION.json").read_text())
                  ["inherited_feature_limits"],
                  transform="Same as the base successor: invert legacy NY start clock; Europe/London "
                            "wall+5h; UTC+1h bar end; values preserved")
    write("PREPARATION.json", report)
    clock = dict(s_clock, input_sha256=report["input_sha256"],
                 clock_policy=("EU_DST_OFFSET_MINUS5_BASE_BAR_START; retained " + start.date().isoformat()
                               + "..2024 prices reconciled to annual source; base scope from 2019-12-01 as "
                               "before; DECLARED CLOCK ASSUMPTION " + assumption["assumption_id"] + " "
                               + assumption["assumption_sha256"] + ": " + assumption["limitation"]
                               + " Scope " + report["clock_scope_sha256"] + "; preparation "
                               + file_hash(output / "PREPARATION.json")
                               + "; inherited publication/vintage assumptions; retrospective research only"))
    write("CLOCK.json", clock)
    return {"state": "MATERIALIZED_WITH_DECLARED_ASSUMPTION", "assumption_id": assumption["assumption_id"],
            "assumption_sha256": assumption["assumption_sha256"], "rows": report["rows"],
            "extension_rows": report["extension_rows"], "input_sha256": report["input_sha256"],
            "clock_sha256": file_hash(output / "CLOCK.json"), "clock_scope_sha256": report["clock_scope_sha256"]}


def main(argv=None) -> int:
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if argv and argv[0] == "materialize":
        p = argparse.ArgumentParser(description="materialize the declared-assumption extension")
        for name in ("source", "archives", "clock-audit", "design", "successor", "output"):
            p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--attempt-report", type=Path, action="append", required=True)
        p.add_argument("--source-sha256", required=True)
        p.add_argument("--start", required=True)
        a = p.parse_args(argv[1:])
        print(json.dumps(materialize(source=a.source, source_sha256=a.source_sha256, archives=a.archives,
                                     clock_audit=a.clock_audit, attempt_reports=a.attempt_report,
                                     design=a.design, successor=a.successor,
                                     start=datetime.fromisoformat(a.start), output=a.output), indent=1))
        return 0
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("source", "archives", "clock-audit", "design", "successor-input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--successor-sha256", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default="2019-12-01T00:00:00+00:00")
    a = p.parse_args(argv)
    report = run(source=a.source, source_sha256=a.source_sha256, archives=a.archives,
                 clock_audit=a.clock_audit, design=a.design, start=datetime.fromisoformat(a.start),
                 end=datetime.fromisoformat(a.end), successor_input=a.successor_input,
                 successor_sha256=a.successor_sha256, output=a.output)
    print(json.dumps({k: report[k] for k in ("state", "missing", "span")}, indent=1))
    return 0 if report["state"] != "WAIT_DEPENDENCY" else 4


if __name__ == "__main__":
    raise SystemExit(main())
