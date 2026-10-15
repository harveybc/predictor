"""Bounded source-only OHLC preparation; run with --noconftest (no TensorFlow)."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import strategy_inputs as SI


ROWS = [
    "20240102 000000;1.1000;1.1010;1.0990;1.1005;0",
    "20240102 000100;1.1005;1.1030;1.1000;1.1020;0",
    "20240102 005900;1.1020;1.1025;1.0980;1.1015;0",
    # Hour 01 is absent, not a flat fabricated bar.
    "20240102 020000;1.1015;1.1050;1.1010;1.1040;0",
]
TS = np.array([1704175200, 1704182400], dtype=np.int64)  # 06:00, 08:00 UTC


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(tmp_path, rows=None, ts=None, price=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "HISTDATA_COM_ASCII_EURUSD_M1_2024.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("DAT_ASCII_EURUSD_M1_2024.csv", "\n".join(ROWS if rows is None else rows) + "\n")
        z.writestr("DAT_ASCII_EURUSD_M1_2024.txt", "retained source metadata")
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    np.savez(inp / "INPUT.npz", ts=TS if ts is None else ts,
             price=np.array([1.1015, 1.1040]) if price is None else price)
    (inp / "CALENDAR.json").write_bytes(b'{ "market_calendar": { "calendar_id": "ASSUMED_NOT_BROKER" } }\n')
    (inp / "CLOCK.json").write_text(json.dumps(dict(status="RESOLVED", uncertainty="ASSUMED_NOT_BROKER")))
    return archive, inp, tmp_path / "out", sha(archive)


def prepare(args):
    return SI.prepare_strategy_bars(*args)


def test_observed_ohlc_clock_calendar_and_receipt(tmp_path):
    args = make(tmp_path)
    report = prepare(args)
    with np.load(args[2] / "STRATEGY_BARS.npz", allow_pickle=False) as z:
        assert set(z.files) == {"close_time", "open", "high", "low", "close"}
        np.testing.assert_array_equal(z["close_time"], TS)
        assert z["close_time"].dtype == np.int64
        np.testing.assert_array_equal(z["open"], [1.1000, 1.1015])
        np.testing.assert_array_equal(z["high"], [1.1030, 1.1050])
        np.testing.assert_array_equal(z["low"], [1.0980, 1.1010])
        np.testing.assert_array_equal(z["close"], [1.1015, 1.1040])
    assert (args[2] / "CALENDAR.json").read_bytes() == (args[1] / "CALENDAR.json").read_bytes()
    assert report["status"] == "PREPARED"
    assert report["test_read"] is False and report["broker_certified"] is False
    assert report["reconciliation"]["matched_retained_rows"] == 2
    assert report["reconciliation"]["missing_retained_timestamps"] == []
    assert report["counts"]["unobserved_hours_between_first_last"] == 1
    assert report["retained"]["clock_sha256"] == sha(args[1] / "CLOCK.json")
    assert report["retained"]["clock_document"]["uncertainty"] == "ASSUMED_NOT_BROKER"
    assert report["outputs"]["STRATEGY_BARS.npz"] == sha(args[2] / "STRATEGY_BARS.npz")
    assert report == json.loads((args[2] / "RECONCILIATION.json").read_text())


def test_missing_retained_origin_fails_without_output(tmp_path):
    args = make(tmp_path, ts=np.array([TS[0], TS[0] + 3600, TS[1]]), price=[1.1015, 1.1015, 1.104])
    with pytest.raises(SI.StrategyInputError, match="RETAINED_ROWS_MISSING") as caught:
        prepare(args)
    assert caught.value.details["missing_retained_timestamps"] == [int(TS[0] + 3600)]
    assert not args[2].exists()


def test_wrong_retained_close_fails_without_output(tmp_path):
    args = make(tmp_path, price=[1.1016, 1.1040])
    with pytest.raises(SI.StrategyInputError, match="RETAINED_CLOSE_MISMATCH"):
        prepare(args)
    assert not args[2].exists()


def test_digest_refused_before_zip_open(tmp_path, monkeypatch):
    archive, inp, out, _ = make(tmp_path)
    monkeypatch.setattr(SI.zipfile, "ZipFile", lambda *a, **k: pytest.fail("unverified ZIP opened"))
    with pytest.raises(SI.StrategyInputError, match="ARCHIVE_DIGEST_MISMATCH"):
        SI.prepare_strategy_bars(archive, inp, out, "0" * 64)


@pytest.mark.parametrize("row,code", [
    ("20250102 000000;1;1;1;1;0", "SOURCE_YEAR_NOT_2024"),
    ("20231231 170000;1;1;1;1;0", "SOURCE_YEAR_NOT_2024"),
    ("20241231 230000;1;1;1;1;0", "TEST_OR_NON2024_BAR"),
    ("20240102 000000;0;1;0;1;0", "OHLC_INVALID"),
    ("20240102 000000;1;0.9;0.8;1;0", "OHLC_INVALID"),
    ("20240102 000000;1;1.1;1.01;1.05;0", "OHLC_INVALID"),
    ("20240102 000000;nan;1;1;1;0", "OHLC_INVALID"),
    ("20240102 000000;1;inf;1;1;0", "OHLC_INVALID"),
    ("20240102 000001;1;1;1;1;0", "SOURCE_MINUTE_INVALID"),
    ("not-a-time;1;1;1;1;0", "SOURCE_MINUTE_INVALID"),
    ("20240102 000000;1;1;1;1", "SOURCE_CSV_INVALID"),
])
def test_source_rejects_bad_rows_before_emission(tmp_path, row, code):
    args = make(tmp_path, rows=[row])
    with pytest.raises(SI.StrategyInputError, match=code):
        prepare(args)
    assert not args[2].exists()


def test_identical_duplicate_is_counted_but_conflict_rejected(tmp_path):
    args = make(tmp_path / "same", rows=[ROWS[0], *ROWS])
    assert prepare(args)["counts"]["identical_duplicate_minutes_removed"] == 1
    args = make(tmp_path / "conflict", rows=[ROWS[0], ROWS[0].replace("1.1005", "1.1006"), *ROWS[1:]])
    with pytest.raises(SI.StrategyInputError, match="CONFLICTING_DUPLICATE_MINUTE"):
        prepare(args)


def test_out_of_order_minutes_are_not_silently_sorted(tmp_path):
    args = make(tmp_path, rows=[ROWS[1], ROWS[0], *ROWS[2:]])
    with pytest.raises(SI.StrategyInputError, match="SOURCE_MINUTES_NOT_ORDERED"):
        prepare(args)


def test_future_perturbation_does_not_change_closed_hour(tmp_path):
    first = make(tmp_path / "a")
    later_rows = [*ROWS[:3], ROWS[3].replace("1.1015;1.1050;1.1010;1.1040", "1.15;1.2;1.1;1.18")]
    second = make(tmp_path / "b", rows=later_rows, price=[1.1015, 1.18])
    prepare(first)
    prepare(second)
    with np.load(first[2] / "STRATEGY_BARS.npz") as a, np.load(second[2] / "STRATEGY_BARS.npz") as b:
        for key in a.files:
            assert a[key][0] == b[key][0]
        assert a["open"][0] == 1.1  # never use the later hour's opening price


def test_idempotent_existing_output_is_read_only(tmp_path):
    args = make(tmp_path)
    first = prepare(args)
    snapshot = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in args[2].iterdir()}
    assert prepare(args) == first
    assert snapshot == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in args[2].iterdir()}


@pytest.mark.parametrize("name", ["STRATEGY_BARS.npz", "CALENDAR.json", "RECONCILIATION.json", "extra"])
def test_existing_output_never_overwritten(tmp_path, name):
    args = make(tmp_path)
    prepare(args)
    target = args[2] / name
    target.write_bytes(b"existing-tampered-or-unrelated")
    snapshot = {p.name: p.read_bytes() for p in args[2].iterdir()}
    with pytest.raises(SI.StrategyInputError, match="OUTPUT_CONFLICT"):
        prepare(args)
    assert snapshot == {p.name: p.read_bytes() for p in args[2].iterdir()}


def test_empty_existing_directory_is_not_repurposed(tmp_path):
    args = make(tmp_path)
    args[2].mkdir()
    with pytest.raises(SI.StrategyInputError, match="OUTPUT_CONFLICT"):
        prepare(args)
    assert not list(args[2].iterdir())


@pytest.mark.parametrize("ts,prices,code", [
    ([TS[1], TS[0]], [1.104, 1.1015], "INPUT_TIMESTAMPS_INVALID"),
    ([TS[0], TS[0]], [1.1015, 1.1015], "INPUT_TIMESTAMPS_INVALID"),
    ([1735689600], [1.1], "INPUT_TEST_FORBIDDEN"),
    ([TS[0]], [float("nan")], "INPUT_PRICES_INVALID"),
    ([1700000000], [1.1], "INPUT_2024_EMPTY"),
])
def test_retained_input_validation(tmp_path, ts, prices, code):
    args = make(tmp_path, ts=np.array(ts, dtype=np.int64), price=prices)
    with pytest.raises(SI.StrategyInputError, match=code):
        prepare(args)


def test_preparation_pin_must_agree_when_present(tmp_path):
    args = make(tmp_path)
    (args[1] / "PREPARATION.json").write_text(json.dumps(dict(archives=[dict(year="2024", sha256="0" * 64)])))
    with pytest.raises(SI.StrategyInputError, match="PREPARATION_ARCHIVE_MISMATCH"):
        prepare(args)


def test_cli_emits_prepared_receipt_without_model_import(tmp_path):
    archive, inp, out, digest = make(tmp_path)
    command = [sys.executable, "-m", "tools.i6dmo.strategy_inputs", "--archive", str(archive),
               "--expected-sha256", digest, "--input-dir", str(inp), "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PREPARED"


def test_partial_year_source_reconciles_only_actual_2024_input(tmp_path):
    args = make(tmp_path, ts=np.array([1672531200, *TS]), price=[1.05, 1.1015, 1.104])
    report = prepare(args)
    assert report["reconciliation"]["retained_2024_rows"] == 2
    assert report["reconciliation"]["ignored_pre2024_rows"] == 1
    assert report["all_2024_history_proven"] is False


def test_approved_clock_helper_is_used_for_summer_dst(tmp_path, monkeypatch):
    rows = ["20240702 000000;1.1;1.2;1.0;1.15;0"]
    endpoint = int(__import__("datetime").datetime.fromisoformat("2024-07-02T05:00:00+00:00").timestamp())
    args = make(tmp_path, rows=rows, ts=np.array([endpoint]), price=[1.15])
    original = SI.clock.eu_clock_series
    calls = []
    def mapped(hourly):
        calls.append(hourly.index[0])
        return original(hourly)
    monkeypatch.setattr(SI.clock, "eu_clock_series", mapped)
    prepare(args)
    assert len(calls) == 1
    with np.load(args[2] / "STRATEGY_BARS.npz") as z:
        assert z["close_time"].tolist() == [endpoint]


def test_empty_zip_and_multiple_csv_members_rejected(tmp_path):
    args = make(tmp_path)
    with zipfile.ZipFile(args[0], "a") as z:
        z.writestr("other.csv", ROWS[0])
    with pytest.raises(SI.StrategyInputError, match="SOURCE_ZIP_MEMBER_INVALID"):
        SI.prepare_strategy_bars(args[0], args[1], args[2], sha(args[0]))


def test_existing_output_input_drift_is_not_accepted(tmp_path):
    args = make(tmp_path)
    prepare(args)
    original = {p.name: p.read_bytes() for p in args[2].iterdir()}
    (args[1] / "CALENDAR.json").write_bytes(b"changed calendar")
    with pytest.raises(SI.StrategyInputError, match="OUTPUT_CONFLICT"):
        prepare(args)
    assert original == {p.name: p.read_bytes() for p in args[2].iterdir()}
