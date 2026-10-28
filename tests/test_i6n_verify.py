"""Finite verification automation must not confuse tests with a campaign."""

import json
from pathlib import Path

import pytest

from tools import i6n_verify as v


def test_no_state_is_not_complete(tmp_path):
    report, code = v.status(tmp_path, "identity")
    assert code != 0
    assert report["state"] == "NOT_RUN"
    assert not list(tmp_path.iterdir())


def test_junit_rejects_failures_skips_and_empty_suite(tmp_path):
    for attrs in ('tests="1" failures="1"', 'tests="1" skipped="1"', 'tests="0"'):
        path = tmp_path / "junit.xml"
        path.write_text(f'<testsuites><testsuite {attrs}/></testsuites>')
        with pytest.raises(ValueError):
            v.junit_count(path)


def test_junit_accepts_executed_tests(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite tests="12" failures="0" errors="0" skipped="0"/></testsuites>')
    assert v.junit_count(path) == 12


def test_identity_mismatch_does_not_write(tmp_path):
    path = tmp_path / "STATUS.json"
    value = json.dumps({"identity": "old", "state": "COMPLETED"})
    path.write_text(value)
    report, code = v.status(tmp_path, "new")
    assert code != 0
    assert report["state"] == "IDENTITY_MISMATCH"
    assert path.read_text() == value


def test_status_requires_every_step_receipt(tmp_path):
    (tmp_path / "STATUS.json").write_text(json.dumps(
        {"identity": "same", "state": "COMPLETED", "steps": []}))
    report, code = v.status(tmp_path, "same")
    assert code != 0
    assert report["state"] == "DAMAGED"


def test_cached_verification_checks_retained_bytes(tmp_path):
    receipts = []
    for name, _ in v.STEPS:
        xml = tmp_path / f"{name}.xml"
        log = tmp_path / f"{name}.log"
        xml.write_text('<testsuites><testsuite tests="1"/></testsuites>')
        log.write_text("passed")
        receipts.append({"name": name, "state": "COMPLETED", "tests": 1,
                         "xml_sha256": v.file_hash(xml), "log_sha256": v.file_hash(log)})
    record = {"identity": "same", "state": "COMPLETED", "steps": receipts}
    (tmp_path / "STATUS.json").write_text(json.dumps(record))
    report, code = v.status(tmp_path, "same")
    assert code == 0
    assert report["campaign_ready"] is False
    (tmp_path / f"{v.STEPS[0][0]}.log").write_text("changed")
    assert v.status(tmp_path, "same")[1] != 0
