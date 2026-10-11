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


def main(argv=None) -> int:
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
