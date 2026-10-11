"""Synthetic transport/price fixtures; never evidence of a resolved clock."""

from datetime import date
import hashlib
import json
import lzma
import struct

import pytest

from tools import i6n_external_clock_audit as audit


def payload():
    return lzma.compress(b"".join(struct.pack(">5if", minute * 60,
        108000 + minute, 108000 + minute, 108000 + minute, 108000 + minute, 1.)
        for minute in range(1440)), format=lzma.FORMAT_ALONE)


def test_dates_frozen_and_pre_test():
    dates = audit.planned_dates()
    assert len(dates) == 18 and len(set(dates)) == 18
    assert {d.year for d in dates} == set(range(2019, 2025))
    assert {d.month for d in dates} == {1, 3, 7}
    assert all(d.weekday() == 0 and 15 <= d.day <= 21 for d in dates)
    assert date(2024, 3, 18) in dates


def test_decoder_hour_ends_and_layout():
    series = audit.decode(payload(), date(2024, 3, 18))
    assert len(series) == 24
    assert str(series.index[0]) == "2024-03-18 01:00:00+00:00"
    assert str(series.index[-1]) == "2024-03-19 00:00:00+00:00"
    assert series.iloc[0] == (108000 + 59) / 1e5
    assert series.iloc[-1] == (108000 + 1439) / 1e5


@pytest.mark.parametrize("data", [b"junk", payload()[:-1], payload() + b"extra",
    lzma.compress(b"a" * 34561), lzma.compress(b"a" * 24)])
def test_invalid_binary_refused(data):
    with pytest.raises((ValueError, lzma.LZMAError)):
        audit.decode(data, date(2024, 3, 18))


@pytest.mark.parametrize("index,value", [(0, 1), (1, -1), (3, 999999), (5, float("nan"))])
def test_invalid_record_refused(index, value):
    raw = bytearray(lzma.decompress(payload()))
    fields = list(struct.unpack(">5if", raw[:24]))
    fields[index] = value
    raw[:24] = struct.pack(">5if", *fields)
    with pytest.raises(ValueError):
        audit.decode(lzma.compress(raw), date(2024, 3, 18))


def test_comparison_winter_dst_and_contradiction():
    day = date(2024, 3, 18)
    reference = audit.decode(payload(), day)
    wrong = reference + .001
    result = audit.compare_case(day, reference, reference, wrong)
    assert result["status"] == "PASS"
    assert result["fixed_est"]["matched_1e12"] == 24
    assert result["legacy_ny"]["mean_abs_error"] > 0
    assert audit.compare_case(day, reference, wrong, reference)["status"] == "INCONCLUSIVE"
    assert audit.compare_case(day, reference, reference, reference)["status"] == "INCONCLUSIVE"
    winter = date(2024, 1, 15)
    reference = audit.decode(payload(), winter)
    assert audit.compare_case(winter, reference, reference, reference)["status"] == "PASS"


def test_no_posthoc_tolerance_or_missing_row_relaxation():
    day = date(2024, 7, 15)
    reference = audit.decode(payload(), day)
    shifted = reference.copy()
    shifted.iloc[0] += .00001
    assert audit.compare_case(day, reference, shifted, reference + .001)["status"] == "INCONCLUSIVE"
    result = audit.compare_case(day, reference, reference.iloc[:-1], reference + .001)
    assert result["status"] == "INCONCLUSIVE"
    assert result["fixed_est"]["overlap"] == 23


class Response:
    status_code = 200
    headers = {}
    def __init__(self, blocks):
        self.blocks = blocks
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_content(self, chunk_size):
        yield from self.blocks


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_transport_single_request_timeout_and_limit():
    session = Session(Response([payload()]))
    body, url = audit.fetch(session, date(2024, 3, 18), audit.time.monotonic() + 30)
    assert body == payload()
    assert url.endswith("/2024/02/18/BID_candles_min_1.bi5")
    assert session.calls[0][1] == dict(timeout=(5, 15), stream=True, allow_redirects=False)
    assert len(session.calls) == 1
    for response in [Response([b"x" * 1_000_001]), Response([b"x" * 500001] * 2)]:
        with pytest.raises(ValueError, match="byte"):
            audit.fetch(Session(response), date(2024, 3, 18), audit.time.monotonic() + 30)


def test_redirect_error_and_test_dates_never_downloaded():
    response = Response([])
    response.status_code = 302
    session = Session(response)
    with pytest.raises(ValueError, match="200"):
        audit.fetch(session, date(2024, 3, 18), audit.time.monotonic() + 30)
    assert len(session.calls) == 1
    with pytest.raises(ValueError, match="2019"):
        audit.fetch(session, date(2025, 1, 20), audit.time.monotonic() + 30)
    assert len(session.calls) == 1


def test_manifest_pin_and_report_seal(tmp_path):
    report = {"status": "INCONCLUSIVE", "clock_policy": "FIXED_EST_BAR_START",
              "input_manifest_sha256": "a" * 64, "cases": []}
    audit.seal_report(tmp_path, report)
    raw = (tmp_path / "REPORT.json").read_bytes()
    assert (tmp_path / "REPORT.sha256").read_text().strip() == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        audit.seal_report(tmp_path, report)


def test_only_complete_all_pass_plan_verifies():
    cases = [dict(date=d.isoformat(), status="PASS") for d in audit.planned_dates()]
    assert audit.terminal_status(cases) == "VERIFIED"
    assert audit.terminal_status(cases[:-1]) == "INCONCLUSIVE"
    cases[-1]["status"] = "INCONCLUSIVE"
    assert audit.terminal_status(cases) == "INCONCLUSIVE"
    cases[-1] = cases[0]
    assert audit.terminal_status(cases) == "INCONCLUSIVE"


def test_manifest_mismatch_seals_inconclusive_without_network(tmp_path, monkeypatch):
    import requests
    annual = tmp_path / "annual"
    annual.mkdir()
    filename = "HISTDATA_COM_ASCII_EURUSD_M1_2019.zip"
    (annual / filename).write_bytes(b"wrong archive")
    manifest = json.dumps({"attempts": [dict(status="DOWNLOADED", month="2019",
                           file=filename, zip_sha256="0" * 64)]}).encode()
    (annual / "MANIFEST.json").write_bytes(manifest)
    calls = []
    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("must not download with invalid source manifest")
    monkeypatch.setattr(requests.Session, "get", refuse)
    report = audit.run(annual, tmp_path / "report")
    assert calls == []
    assert report["status"] == "INCONCLUSIVE"
    assert len(report["cases"]) == 18
    assert report["clock_policy"] == "FIXED_EST_BAR_START"
    assert report["input_manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert "manifest pin" in report["cases"][0]["reason"]
    with pytest.raises(FileExistsError):
        audit.run(annual, tmp_path / "report")
