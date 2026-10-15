"""Prepare private 2024 strategy OHLC from pinned HistData M1, without fitting.

Observed minutes only; exact approved EU clock mapping; every retained 2024
close must reconcile. This preserves clock uncertainty and is not broker proof.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd

from tools import i6n_external_clock_audit_v2 as clock


APPROVED_2024_SHA256 = "58f074ba6b835eb2fbeaeb6678502a23b26768eefdfa509eb75ff5296bb2904d"
YEAR_START = 1704067200
TEST_START = 1735689600
MAX_ARCHIVE_BYTES = 32 * 1024 ** 2
MAX_CSV_BYTES = 64 * 1024 ** 2
MAX_MINUTE_ROWS = 600000
OUTPUT_FILES = {"STRATEGY_BARS.npz", "CALENDAR.json", "RECONCILIATION.json"}
TOLERANCE = 1e-12


class StrategyInputError(ValueError):
    """Fail-closed preparation error; details contain bounded reconciliation facts."""

    def __init__(self, code, details=None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


def _require(ok, code, details=None):
    if not ok:
        raise StrategyInputError(code, details)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _hourly(archive):
    hours, bars = [], []
    previous, previous_values = None, None
    source_rows, duplicates = 0, 0
    with zipfile.ZipFile(archive) as zipped:
        members = [m for m in zipped.infolist() if m.filename.lower().endswith(".csv")]
        _require(len(members) == 1 and members[0].filename == "DAT_ASCII_EURUSD_M1_2024.csv",
                 "SOURCE_ZIP_MEMBER_INVALID")
        member = members[0]
        _require(0 < member.file_size <= MAX_CSV_BYTES, "SOURCE_CSV_SIZE_INVALID")
        with zipped.open(member) as raw, io.TextIOWrapper(raw, encoding="ascii", newline="") as stream:
            for row in csv.reader(stream, delimiter=";", strict=True):
                source_rows += 1
                _require(source_rows <= MAX_MINUTE_ROWS, "SOURCE_ROW_LIMIT")
                _require(len(row) == 6, "SOURCE_CSV_INVALID", dict(line=source_rows))
                text = row[0]
                _require(re.fullmatch(r"[0-9]{8} [0-9]{6}", text), "SOURCE_MINUTE_INVALID")
                try:
                    stamp = datetime.strptime(text, "%Y%m%d %H%M%S")
                except ValueError:
                    raise StrategyInputError("SOURCE_MINUTE_INVALID") from None
                _require(stamp.year == 2024, "SOURCE_YEAR_NOT_2024")
                _require(stamp.second == 0, "SOURCE_MINUTE_INVALID")
                try:
                    o, h, lo, c = (float(v) for v in row[1:5])
                except ValueError:
                    raise StrategyInputError("OHLC_INVALID") from None
                _require(all(math.isfinite(v) and v > 0 for v in (o, h, lo, c)) and
                         lo <= min(o, c) <= max(o, c) <= h, "OHLC_INVALID", dict(line=source_rows))
                values = (o, h, lo, c, row[5])
                if stamp == previous:
                    _require(values == previous_values, "CONFLICTING_DUPLICATE_MINUTE")
                    duplicates += 1
                    continue
                _require(previous is None or stamp > previous, "SOURCE_MINUTES_NOT_ORDERED")
                hour = stamp.replace(minute=0)
                if not hours or hours[-1] != hour:
                    hours.append(hour)
                    bars.append([o, h, lo, c])
                else:
                    bars[-1][1] = max(bars[-1][1], h)
                    bars[-1][2] = min(bars[-1][2], lo)
                    bars[-1][3] = c
                previous, previous_values = stamp, values
    _require(bool(hours), "SOURCE_EMPTY")
    values = np.asarray(bars, dtype=np.float64)
    # Mapping the close series also fixes the endpoint clock of its observed O/H/L.
    hourly_close = pd.Series(values[:, 3], index=pd.DatetimeIndex(hours))
    try:
        mapped, mapping = clock.eu_clock_series(hourly_close)
    except (ValueError, TypeError) as exc:
        raise StrategyInputError("SOURCE_CLOCK_INVALID", dict(reason=type(exc).__name__)) from None
    endpoints = mapped.index.as_unit("s").asi8.astype(np.int64)
    _require(len(endpoints) == len(values) and np.all(endpoints[1:] > endpoints[:-1])
             and np.all(endpoints % 3600 == 0), "SOURCE_CLOCK_NOT_UNIQUE_ORDERED")
    _require(np.all((endpoints >= YEAR_START) & (endpoints < TEST_START)), "TEST_OR_NON2024_BAR")
    return endpoints, values, dict(source_minute_rows=source_rows,
        observed_unique_minutes=source_rows - duplicates,
        identical_duplicate_minutes_removed=duplicates, observed_hours=len(endpoints)), mapping


def _retained(input_dir, expected_sha256):
    input_path, calendar_path = input_dir / "INPUT.npz", input_dir / "CALENDAR.json"
    _require(input_path.is_file() and calendar_path.is_file(), "RETAINED_FILES_MISSING")
    input_hash, calendar_bytes = _sha(input_path), calendar_path.read_bytes()
    with np.load(input_path, allow_pickle=False) as z:
        ts = z["ts"]
        _require(ts.ndim == 1 and len(ts) and ts.dtype.kind in "iu" and
                 np.all(ts[1:] > ts[:-1]), "INPUT_TIMESTAMPS_INVALID")
        # Refuse the TEST boundary before loading any retained price values.
        _require(np.all(ts < TEST_START), "INPUT_TEST_FORBIDDEN")
        price = z["price"]
        _require(price.shape == ts.shape and price.dtype.kind in "fiu" and
                 np.all(np.isfinite(price)) and np.all(price > 0), "INPUT_PRICES_INVALID")
        keep = ts >= YEAR_START
        _require(np.any(keep), "INPUT_2024_EMPTY")
        retained_ts, retained_price = ts[keep].astype(np.int64), price[keep].astype(np.float64)
        ignored = int((~keep).sum())
    preparation_path = input_dir / "PREPARATION.json"
    preparation_hash, preparation_pin = None, False
    if preparation_path.is_file():
        preparation = json.loads(preparation_path.read_text())
        preparation_hash = _sha(preparation_path)
        records = [r for r in preparation.get("archives", []) if str(r.get("year")) == "2024"]
        _require(len(records) == 1 and records[0].get("sha256") == expected_sha256,
                 "PREPARATION_ARCHIVE_MISMATCH")
        preparation_pin = True
    clock_path = input_dir / "CLOCK.json"
    retained_clock = json.loads(clock_path.read_text()) if clock_path.is_file() else None
    clock_hash = _sha(clock_path) if clock_path.is_file() else None
    return retained_ts, retained_price, calendar_bytes, dict(
        input_sha256=input_hash, calendar_sha256=_digest(calendar_bytes),
        preparation_sha256=preparation_hash, preparation_2024_pin_confirmed=preparation_pin,
        clock_sha256=clock_hash, clock_document=retained_clock), ignored


def _publish(out_dir, contents):
    if out_dir.exists() or out_dir.is_symlink():
        _require(out_dir.is_dir() and not out_dir.is_symlink() and
                 {p.name for p in out_dir.iterdir()} == OUTPUT_FILES, "OUTPUT_CONFLICT")
        for name, content in contents.items():
            path = out_dir / name
            _require(path.is_file() and not path.is_symlink() and _sha(path) == _digest(content),
                     "OUTPUT_CONFLICT", dict(file=name))
        return
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(mode=0o700)
    except FileExistsError:
        raise StrategyInputError("OUTPUT_CONFLICT") from None
    # Exclusive files in a newly reserved private directory. A crash can leave a
    # partial directory, which a later run refuses; no retained evidence is replaced.
    for name, content in contents.items():
        with (out_dir / name).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def prepare_strategy_bars(archive, input_dir, out_dir, expected_sha256):
    """Return the reconciliation receipt after exclusive, deterministic preparation.

    Inputs are read-only. Existing outputs are accepted only when all three file
    hashes equal a fresh deterministic regeneration; any discrepancy fails closed.
    Caller must pin the returned NPZ/calendar file digests into the RAW pair config.
    """
    archive, input_dir, out_dir = Path(archive), Path(input_dir), Path(out_dir)
    _require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
             "ARCHIVE_DIGEST_INVALID")
    _require(archive.is_file() and 0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES, "ARCHIVE_SIZE_INVALID")
    archive_sha = _sha(archive)
    _require(archive_sha == expected_sha256, "ARCHIVE_DIGEST_MISMATCH")
    _require(not out_dir.resolve().is_relative_to(input_dir.resolve()), "OUTPUT_INSIDE_RETAINED_INPUT")
    try:
        endpoints, values, counts, mapping = _hourly(archive)
        ts, price, calendar, retained, ignored = _retained(input_dir, expected_sha256)
    except StrategyInputError:
        raise
    except (ValueError, KeyError, TypeError, csv.Error, UnicodeError, zipfile.BadZipFile, RuntimeError) as exc:
        raise StrategyInputError("SOURCE_OR_INPUT_INVALID", dict(reason=type(exc).__name__)) from None
    positions = np.searchsorted(endpoints, ts)
    present = (positions < len(endpoints)) & (endpoints[np.minimum(positions, len(endpoints) - 1)] == ts)
    missing = ts[~present].tolist()
    _require(not missing, "RETAINED_ROWS_MISSING", dict(missing_retained_timestamps=missing))
    errors = np.abs(values[positions, 3] - price)
    bad = errors > TOLERANCE
    _require(not np.any(bad), "RETAINED_CLOSE_MISMATCH", dict(
        mismatched_timestamps=ts[bad].tolist(), mismatched_count=int(bad.sum()),
        max_abs_error=float(errors.max())))
    gaps = int(np.sum(np.diff(endpoints) // 3600 - 1))
    counts["unobserved_hours_between_first_last"] = gaps
    buffer = io.BytesIO()
    np.savez(buffer, close_time=endpoints, open=values[:, 0], high=values[:, 1],
             low=values[:, 2], close=values[:, 3])
    bars = buffer.getvalue()
    report = dict(schema="i6dmo.strategy_inputs.v1", status="PREPARED", source_year=2024,
        archive_sha256=archive_sha, expected_archive_sha256=expected_sha256,
        retained=retained, counts=counts,
        reconciliation=dict(retained_2024_rows=len(ts), matched_retained_rows=len(ts),
            ignored_pre2024_rows=ignored, missing_retained_timestamps=[], mismatched_timestamps=[],
            max_abs_error=float(errors.max()), absolute_tolerance=TOLERANCE, relative_tolerance=0,
            source_hours_not_in_retained=int(len(endpoints) - len(ts))),
        outputs={"STRATEGY_BARS.npz": _digest(bars), "CALENDAR.json": _digest(calendar)},
        coverage=dict(first_close_time=int(endpoints[0]), last_close_time=int(endpoints[-1]),
                      no_fill_or_invented_bars=True, observed_minute_ohlc_only=True),
        clock=dict(policy=clock.CLOCK_POLICY, mapping=mapping,
            transform="hour floor; (wall + 5h) localized Europe/London -> UTC + 1h bar end",
            aggregation="observed minute first open / max high / min low / last close",
            timestamp_semantics="UTC_BAR_END", availability="AT_BAR_END",
            assumption="Inherited EU-DST clock hypothesis; not a certified broker schedule",
            helper_sha256=_sha(Path(clock.__file__))),
        calendar_semantics="Byte-identical inherited calendar; not broker certification",
        code_sha256=_sha(Path(__file__)), test_read=False, broker_certified=False,
        all_2024_history_proven=False,
        limitations=["Only retained 2024 closes reconciled; no assertion of full-year source completeness",
                     "No independent verification of minute open/high/low or broker execution prices",
                     "Clock and calendar assumptions are inherited, not promoted to certification"])
    receipt = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    # Recheck read-only input identities before publishing, including the optional receipt.
    _require(_sha(archive) == archive_sha and _sha(input_dir / "INPUT.npz") == retained["input_sha256"]
             and _sha(input_dir / "CALENDAR.json") == retained["calendar_sha256"] and
             (retained["preparation_sha256"] is None or
              _sha(input_dir / "PREPARATION.json") == retained["preparation_sha256"]) and
             (retained["clock_sha256"] is None or
              _sha(input_dir / "CLOCK.json") == retained["clock_sha256"]), "INPUT_CHANGED_DURING_PREPARATION")
    _publish(out_dir, {"STRATEGY_BARS.npz": bars, "CALENDAR.json": calendar, "RECONCILIATION.json": receipt})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare_strategy_bars(args.archive, args.input_dir, args.out, args.expected_sha256)
    except (StrategyInputError, OSError) as exc:
        print(json.dumps(dict(status="FAILED", code=getattr(exc, "code", type(exc).__name__),
                              details=getattr(exc, "details", {}))), file=sys.stderr)
        return 2
    print(json.dumps(dict(status=report["status"], outputs=report["outputs"],
                         reconciliation_sha256=_sha(args.out / "RECONCILIATION.json"),
                         matched_retained_rows=report["reconciliation"]["matched_retained_rows"],
                         observed_hours=report["counts"]["observed_hours"], test_read=False), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
