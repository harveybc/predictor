"""Offline evidence gate tests; invented receipts are not clock observations."""

from copy import deepcopy
from datetime import date
import json

import pytest

from tools import i6n_external_clock_audit_v2_verify as gate


@pytest.fixture
def sealed(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    manifest = tmp_path / "MANIFEST.json"
    attempts = [dict(month=str(y), status="DOWNLOADED", zip_sha256=str(y)[-1] * 64) for y in range(2019, 2025)]
    manifest.write_text(json.dumps(dict(attempts=attempts)))
    manifest_hash = gate.source.digest(manifest.read_bytes())
    replay = [d.isoformat() for d in gate.v1.planned_dates() if d.isoformat() != "2019-01-21"]
    planned = replay + [d.isoformat() for d in gate.v2.holdout_dates()]
    cases = []
    for i, stamp in enumerate(planned):
        day = date.fromisoformat(stamp)
        raw = f"fabricated-price-bytes-{stamp}".encode()
        case = dict(date=stamp, status="PASS", evidence_role="replay" if i < 17 else "holdout",
                    available_endpoints=24, matched_exact=24, mean_abs_error=0., max_abs_error=0.,
                    absent_endpoint_count=0, absent_endpoints_utc=[],
                    source_zip_sha256=str(day.year)[-1] * 64, dukascopy_sha256=gate.source.digest(raw),
                    url=f"{gate.v1.BASE}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5")
        (output / f"{stamp}.bi5").write_bytes(raw)
        (output / f"{stamp}.json").write_text(json.dumps(case))
        cases.append(case)
    report = dict(schema="i6n.external_clock_audit.v2.transport_resume", status="VERIFIED",
                  clock_policy=gate.v2.CLOCK_POLICY, input_manifest_sha256=manifest_hash,
                  planned_dates=planned, cases=cases, acceptance="frozen synthetic fixture",
                  annual_mapping=[dict(year=y, status="PASS", never_moves_earlier=True,
                                       unique=True, ordered=True, source_zip_sha256=str(y)[-1] * 64)
                                  for y in range(2019, 2025)])
    parent = deepcopy(report)
    parent["status"] = "INCONCLUSIVE"
    parent["cases"][17] = dict(date="2019-10-28", status="INCONCLUSIVE", evidence_role="holdout",
                                 reason="ReadTimeout: timeout", dukascopy_sha256=None)
    parent_raw = json.dumps(parent).encode()
    parent_hash = gate.source.digest(parent_raw)
    (output / "report_chain").mkdir()
    (output / "report_chain" / f"{parent_hash}.json").write_bytes(parent_raw)
    report["prior_v2_report_sha256"] = parent_hash
    gate.add_evidence_pins(report)
    gate.v1.seal_report(output, report)
    return output, manifest, report


def reseal(output, report):
    raw = json.dumps(report).encode()
    (output / "REPORT.json").write_bytes(raw)
    (output / "REPORT.sha256").write_text(gate.source.digest(raw))


def test_complete_grouped_hashed_chain_accepts(sealed):
    output, manifest, _ = sealed
    result = gate.verify_audit(output, manifest)
    assert result["status"] == "VERIFIED" and result["case_count"] == 23


@pytest.mark.parametrize("change", ["missing_case", "group", "trace_pin", "artifact", "ancestor", "manifest", "status"])
def test_preparation_gate_rejects_missing_or_tampered_evidence(sealed, change):
    output, manifest, report = sealed
    if change == "missing_case":
        report["cases"].pop()
    elif change == "group":
        report["case_groups"]["holdout"].pop()
    elif change == "trace_pin":
        report["transport_chain_sha256"] = "0" * 64
    elif change == "artifact":
        (output / "2019-03-18.bi5").write_bytes(b"altered")
    elif change == "ancestor":
        next((output / "report_chain").glob("*.json")).write_bytes(b"{}")
    elif change == "manifest":
        manifest.write_bytes(b"{}")
    else:
        report["status"] = "INCONCLUSIVE"
    reseal(output, report)
    with pytest.raises(ValueError):
        gate.verify_audit(output, manifest)
