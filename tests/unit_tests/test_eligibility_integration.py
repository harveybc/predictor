"""The gate as the consumers actually meet it.

`app/main.py` asks once, before preprocessing builds a window and
before a model is fitted. These tests exercise that contract
directly: the stamp a run carries, the refusal that stops a run,
and the fact that the shipped template grants nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eligibility import gate  # noqa: E402
from eligibility import integration as integ  # noqa: E402

TEMPLATE = (REPO /
            "examples/research/ELIGIBILITY_MANIFEST_TEMPLATE"
            ".v1.json")


def _entry(subject_id, decision="PUBLICLY_ELIGIBLE",
           scope="forecasting", kind="variable", **over):
    e = {
        "subject_kind": kind, "subject_id": subject_id,
        "version": "1",
        "io_schema": {"input": ["float"], "output": ["float"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t+0"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "none",
        "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "p" * 64,
                    "evidence": "e" * 64},
        "measured_cost": {"fit_seconds": 0.0},
        "decision": decision, "decision_scope": scope,
        "decision_reason": "reviewed", "reviewer": "double",
        "reviewed_at": "2026-09-09T00:00:00Z",
    }
    e.update(over)
    return e


@pytest.fixture()
def manifest(tmp_path):
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "double", "scope": "test",
           "entries": [_entry("var.ok"),
                       _entry("var.no", decision="REJECTED")]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = tmp_path / "ELIGIBILITY.json"
    p.write_text(json.dumps(doc))
    return p, doc


def test_run_without_manifest_is_stamped_non_authoritative():
    config = {}
    stamp = integ.gate_subjects(config, consumer="test")
    assert stamp["eligibility_status"] == integ.STATUS_LEGACY
    assert "not gated evidence" in stamp["reason"]
    # the stamp travels in the effective config the run saves
    assert config[integ.KEY_STAMP] is stamp
    assert "LEGACY_NON_AUTHORITATIVE" in integ.describe(stamp)


def test_gated_run_records_the_manifest_it_obeyed(manifest):
    p, doc = manifest
    config = {integ.KEY_MANIFEST: str(p),
              integ.KEY_SCOPE: "forecasting"}
    stamp = integ.gate_subjects(config, subject_ids=["var.ok"],
                                consumer="test")
    assert stamp["eligibility_status"] == integ.STATUS_GATED
    assert stamp["manifest_sha256"] == doc["manifest_sha256"]
    assert stamp["subjects_used"] == ["var.ok"]
    assert stamp["universe_size"] == 1
    assert "GATED" in integ.describe(stamp)


def test_ineligible_subject_stops_the_run_before_any_work(
        manifest):
    p, _ = manifest
    config = {integ.KEY_MANIFEST: str(p),
              integ.KEY_SCOPE: "forecasting"}
    with pytest.raises(SystemExit, match="REJECTED"):
        integ.gate_subjects(config, subject_ids=["var.no"],
                            consumer="test")
    with pytest.raises(SystemExit, match="unlisted subject"):
        integ.gate_subjects(config,
                            subject_ids=["var.never_seen"],
                            consumer="test")


def test_configured_manifest_without_scope_refuses(manifest):
    p, _ = manifest
    with pytest.raises(SystemExit, match="never global"):
        integ.gate_subjects({integ.KEY_MANIFEST: str(p)},
                            consumer="test")


def test_pinned_digest_mismatch_stops_the_run(manifest):
    p, _ = manifest
    config = {integ.KEY_MANIFEST: str(p),
              integ.KEY_SCOPE: "forecasting",
              integ.KEY_SHA: "0" * 64}
    with pytest.raises(SystemExit, match="not the pinned one"):
        integ.gate_subjects(config, subject_ids=["var.ok"],
                            consumer="test")


def test_operator_gate_refuses_a_borrowed_plugin_name(tmp_path):
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "double", "scope": "test",
           "entries": [_entry("op.x", kind="operator",
                              parameters={
                                  "plugin_name": "real_op"})]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    config = {integ.KEY_MANIFEST: str(p),
              integ.KEY_SCOPE: "forecasting"}
    ok = integ.gate_operator(config, operator_id="op.x",
                             version="1", code_digest="c" * 64,
                             plugin_name="real_op",
                             consumer="test")
    assert ok["eligibility_status"] == integ.STATUS_GATED
    with pytest.raises(SystemExit, match="borrowing a name"):
        integ.gate_operator(config, operator_id="op.x",
                            version="1", code_digest="c" * 64,
                            plugin_name="impostor",
                            consumer="test")


def test_operator_without_manifest_is_experimental_not_licensed():
    out = integ.gate_operator({}, operator_id="op.x",
                              version="1",
                              code_digest="c" * 64,
                              consumer="test")
    assert out["eligibility_status"] == integ.STATUS_LEGACY
    assert "not licensed" in out["reason"]


def test_shipped_template_grants_nothing():
    assert TEMPLATE.is_file(), "the template must ship"
    with pytest.raises(SystemExit):
        gate.load_manifest(TEMPLATE)
    raw = json.loads(TEMPLATE.read_text())
    assert "_template_note" in raw
    assert "NON-AUTHORIZING" in raw["_template_note"]


def test_main_asks_the_gate_before_the_pipeline_runs():
    """The choke point is in app/main.py and precedes the
    pipeline call — a reader can verify it without running a
    model."""
    src = (REPO / "app/main.py").read_text()
    assert "gate_subjects(" in src
    assert src.index("gate_subjects(") < src.index(
        "pipeline_plugin.run_prediction_pipeline(")
