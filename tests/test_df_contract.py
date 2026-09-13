"""C129: the common contract records what is known, says UNKNOWN for the
rest, keeps original fields, freezes partitions, and never grants."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_contract", ROOT / "tools/df_contract.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

SHA = "a" * 64


def public(**over):
    ds = "public.example.electricity_hourly.v1"
    doc = {
        "schema": C.DATASET_SCHEMA, "dataset_id": ds, "version": "1", "bank": "PUBLIC",
        "files": [{"name": "electricity.tsf", "bytes": 10, "sha256": SHA, "role": "DATA"}],
        "content_sha256": "", "contract_sha256": "",
        "source": {"provider": "Zenodo", "official_url": "https://zenodo.org/records/4656140",
                   "citation": "Godahewa et al. (2020)", "doi": "10.5281/zenodo.4656140",
                   "upstream_owner": "UCI"},
        "license": {"state": "OPEN_ATTRIBUTION", "id": "cc-by-4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/", "text_sha256": "UNAVAILABLE",
                    "attribution_required": "YES", "redistribution": "ALLOWED_WITH_ATTRIBUTION",
                    "derivatives": "ALLOWED", "evidence": [{"source": "zenodo record 4656140", "sha256": SHA}]},
        "time": {"frequency_nominal_seconds": 3600, "timezone": C.UNKNOWN, "timestamp_meaning": C.UNKNOWN,
                 "range_start": "2012-01-01", "range_end": "2014-12-31", "availability_rule": C.UNKNOWN,
                 "availability_delay_seconds": C.UNKNOWN},
        "panel": {"aligned_common_grid": True, "n_series": 2, "alignment_rule": "shared start and frequency"},
        "partitions": {"scheme": "CHRONOLOGICAL_FRACTIONS",
                       "fractions": {"train": 0.6, "calibration": 0.2, "confirmation": 0.2},
                       "boundaries": C.chronological_partitions(100), "sealed_periods_excluded": [],
                       "frozen_before_profile": True},
        "dependence": [],
        "variables": [C.variable(ds, "T1"), C.variable(ds, "T2")],
        "original_fields": {"@frequency": "hourly"},
    }
    doc.update(over)
    return doc


def test_a_public_contract_seals_and_validates():
    doc = C.seal(public())
    assert C.validate(doc) == []
    assert doc["variables"][0]["variable_id"] == C.variable_id_for(doc["dataset_id"], "T1")


def test_unknown_is_a_valid_answer_everywhere():
    v = C.variable("d", "x")
    assert v["unit"]["value"] == C.UNKNOWN and v["license_state"] == C.UNKNOWN and v["role"] == C.UNKNOWN


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["license"].update(evidence=[]), "needs evidence"),
    (lambda d: d["variables"][0].update(role="PUBLICLY_ELIGIBLE"), "is not one of"),
    (lambda d: d["original_fields"].update(status="PUBLICLY_ELIGIBLE"), "never grants"),
    (lambda d: d["panel"].update(n_series=True), "expected an integer"),
    (lambda d: d["time"].update(frequency_nominal_seconds=float("nan")), "finite number"),
    (lambda d: d["partitions"]["fractions"].update(train=0.7), "sum to 1"),
    (lambda d: d["partitions"]["boundaries"].update(calibration=[59, 80]), "contiguous"),
    (lambda d: d["partitions"].update(frozen_before_profile=False), "frozen before"),
    (lambda d: d["variables"].append(copy.deepcopy(d["variables"][0])), "duplicate"),
    (lambda d: d["variables"][0].update(dataset_id="other"), "another dataset"),
    (lambda d: d["variables"][0]["unit"].update(value="kW"), "declared unit needs evidence"),
    (lambda d: d["source"].update(official_url=str(Path.home()) + "/x"), "home path"),
    (lambda d: d.pop("original_fields"), "keys differ"),
    (lambda d: d["license"].update(state="NOT_APPLICABLE_GENERATED"), "only generated data"),
])
def test_refusals(mutate, needle):
    doc = public()
    mutate(doc)
    with pytest.raises(C.ContractRefusal) as exc:
        C.seal(doc)
    assert any(needle in p for p in exc.value.problems), exc.value.problems


def test_tampering_after_sealing_is_detected():
    doc = C.seal(public())
    doc["source"]["citation"] = "changed"
    assert "contract_sha256 does not re-derive" in C.validate(doc)
    doc = C.seal(public())
    doc["files"][0]["sha256"] = "b" * 64
    assert "content_sha256 does not re-derive from the files" in C.validate(doc)


def test_synthetic_contract_rules():
    ds = "synthetic.v2.sine_white"
    doc = public(dataset_id=ds, bank="SYNTHETIC", files=[],
                 license={"state": "NOT_APPLICABLE_GENERATED", "id": "NOT_APPLICABLE", "url": "NOT_APPLICABLE",
                          "text_sha256": "UNAVAILABLE", "attribution_required": "NO", "redistribution": "NOT_APPLICABLE",
                          "derivatives": "NOT_APPLICABLE", "evidence": []},
                 variables=[C.variable(ds, "x0", license_state="NOT_APPLICABLE_GENERATED")])
    assert C.validate(C.seal(doc)) == []
    doc["variables"][0]["license_state"] = "OPEN_ATTRIBUTION"
    with pytest.raises(C.ContractRefusal, match="synthetic license state"):
        C.seal(doc)


def test_financial_may_stay_internal_research_only_without_evidence():
    doc = public(bank="FINANCIAL")
    doc["license"].update(state="INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE", evidence=[])
    assert C.validate(C.seal(doc)) == []


def test_chronological_partitions():
    assert C.chronological_partitions(10) == {"train": [0, 6], "calibration": [6, 8], "confirmation": [8, 10]}
    for bad in ((0.5, 0.5, 0.1), (1.0, 0.0, 0.0)):
        with pytest.raises(C.ContractRefusal):
            C.chronological_partitions(10, bad)
    with pytest.raises(C.ContractRefusal):
        C.chronological_partitions(True)


def test_unknown_variable_field_refuses():
    with pytest.raises(C.ContractRefusal, match="not in the contract"):
        C.variable("d", "x", eligibility="PUBLICLY_ELIGIBLE")
