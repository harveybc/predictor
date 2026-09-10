"""The six mandatory regressions of the eligibility gate, plus
the bindings the order requires.

Each test is named after the failure it forbids. The gate is the
single point where every repository asks whether a variable or
operator may be used, so a hole here is a hole everywhere.
"""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eligibility import gate  # noqa: E402


def _entry(subject_id="var.close", kind="variable",
           decision="PUBLICLY_ELIGIBLE", scope="forecasting",
           version="1", **over):
    e = {
        "subject_kind": kind,
        "subject_id": subject_id,
        "version": version,
        "io_schema": {"input": ["float"], "output": ["float"]},
        "unit": "price_quote_currency",
        "temporal_availability": {
            "event_time": "bar_close_utc",
            "available_time": "bar_close_utc_plus_0s",
            "min_latency_minutes": 0,
        },
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "state_carried_forward_only",
        "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "p" * 64,
                    "evidence": "e" * 64},
        "measured_cost": {"fit_seconds": 0.1,
                          "transform_seconds": 0.01},
        "decision": decision,
        "decision_scope": scope,
        "decision_reason": "reviewed against the public bank",
        "reviewer": "external-review-double",
        "reviewed_at": "2026-09-09T00:00:00Z",
    }
    e.update(over)
    return e


def _manifest(entries=None, issued_at=None):
    doc = {
        "schema": gate.MANIFEST_SCHEMA,
        "issued_at": issued_at or "2026-09-10T00:00:00Z",
        "issuer": "external-review-double",
        "scope": "test",
        "entries": entries if entries is not None else [_entry()],
    }
    doc["manifest_sha256"] = gate._self_sha(doc)
    return doc


@pytest.fixture()
def manifest_file(tmp_path):
    def _write(doc):
        p = tmp_path / "ELIGIBILITY.json"
        p.write_text(json.dumps(doc))
        return p
    return _write


# ==============================================================
# regression 1: absent, stale or different-digest manifest refuses
# ==============================================================

def test_absent_manifest_refuses(tmp_path):
    with pytest.raises(SystemExit, match="ABSENT"):
        gate.load_manifest(tmp_path / "nope.json")


def test_stale_manifest_refuses(manifest_file):
    old = (datetime.now(timezone.utc)
           - timedelta(days=40)).isoformat()
    p = manifest_file(_manifest(issued_at=old))
    gate.load_manifest(p)                      # no limit: fine
    with pytest.raises(SystemExit, match="STALE"):
        gate.load_manifest(p, max_age_days=30)


def test_altered_manifest_refuses(manifest_file):
    doc = _manifest()
    doc["entries"][0]["decision_scope"] = "everything"
    p = manifest_file(doc)                      # digest not fixed
    with pytest.raises(SystemExit, match="does not re-derive"):
        gate.load_manifest(p)


def test_different_manifest_refuses_when_pinned(manifest_file):
    a = _manifest()
    b = _manifest(entries=[_entry(subject_id="var.other")])
    pa = manifest_file(a)
    gate.load_manifest(pa, expected_sha256=a["manifest_sha256"])
    with pytest.raises(SystemExit, match="not the pinned one"):
        gate.load_manifest(pa,
                           expected_sha256=b["manifest_sha256"])


def test_wrong_schema_refuses(manifest_file, tmp_path):
    doc = _manifest()
    doc["schema"] = "something.else.v1"
    doc["manifest_sha256"] = gate._self_sha(doc)
    with pytest.raises(SystemExit, match="schema"):
        gate.load_manifest(manifest_file(doc))


# ==============================================================
# regression 2: an ineligible variable never returns by fallback
# ==============================================================

def test_rejected_variable_is_never_returned(manifest_file):
    doc = _manifest(entries=[
        _entry(subject_id="var.good"),
        _entry(subject_id="var.bad", decision="REJECTED"),
    ])
    doc["manifest_sha256"] = gate._self_sha(doc)
    m = gate.load_manifest(manifest_file(doc))
    assert gate.eligible_universe(m, scope="forecasting") == [
        "var.good"]
    with pytest.raises(SystemExit, match="REJECTED"):
        gate.require_eligible(m, "var.bad", scope="forecasting")
    # and a filter can only ever REMOVE
    kept = gate.filter_to_eligible(
        m, ["var.good", "var.bad", "var.unlisted"],
        scope="forecasting")
    assert kept == ["var.good"]


def test_unlisted_subject_is_not_eligible_by_omission(
        manifest_file):
    m = gate.load_manifest(manifest_file(_manifest()))
    with pytest.raises(SystemExit, match="unlisted subject"):
        gate.require_eligible(m, "var.never_reviewed",
                              scope="forecasting")
    assert gate.is_eligible(m, "var.never_reviewed",
                            scope="forecasting") is False


def test_gate_offers_no_default_allow_parameter():
    import inspect
    sig = inspect.signature(gate.require_eligible)
    assert "default" not in sig.parameters
    assert "allow_missing" not in sig.parameters


# ==============================================================
# regression 3: an empty group never reactivates everything
# ==============================================================

def test_empty_group_yields_empty_universe(manifest_file):
    doc = _manifest(entries=[_entry(subject_id=f"var.{i}")
                             for i in range(5)])
    doc["manifest_sha256"] = gate._self_sha(doc)
    m = gate.load_manifest(manifest_file(doc))
    assert len(gate.eligible_universe(m, scope="forecasting")) == 5
    assert gate.eligible_universe(m, scope="forecasting",
                                  group=[]) == []
    assert gate.eligible_universe(m, scope="forecasting",
                                  group=["var.2"]) == ["var.2"]


def test_group_cannot_widen_the_universe(manifest_file):
    doc = _manifest(entries=[
        _entry(subject_id="var.in"),
        _entry(subject_id="var.out", decision="REJECTED"),
    ])
    doc["manifest_sha256"] = gate._self_sha(doc)
    m = gate.load_manifest(manifest_file(doc))
    got = gate.eligible_universe(
        m, scope="forecasting",
        group=["var.in", "var.out", "var.ghost"])
    assert got == ["var.in"]


# ==============================================================
# regression 4: an operator never enters by plugin name
# ==============================================================

def test_operator_requires_id_version_and_code_bytes(
        manifest_file):
    doc = _manifest(entries=[
        _entry(subject_id="op.ewma", kind="operator",
               parameters={"plugin_name": "ewma_denoise",
                           "alpha": 0.4}),
    ])
    doc["manifest_sha256"] = gate._self_sha(doc)
    m = gate.load_manifest(manifest_file(doc))
    ok = gate.require_operator(m, operator_id="op.ewma",
                               version="1", code_digest="c" * 64,
                               scope="forecasting",
                               plugin_name="ewma_denoise")
    assert ok["subject_id"] == "op.ewma"
    # a different plugin claiming the reviewed name
    with pytest.raises(SystemExit, match="borrowing a name"):
        gate.require_operator(m, operator_id="op.ewma",
                              version="1", code_digest="c" * 64,
                              scope="forecasting",
                              plugin_name="ewma_denoise_v2")
    # the reviewed name attached to different code
    with pytest.raises(SystemExit, match="different code"):
        gate.require_operator(m, operator_id="op.ewma",
                              version="1", code_digest="f" * 64,
                              scope="forecasting",
                              plugin_name="ewma_denoise")
    # a version bump is a new review
    with pytest.raises(SystemExit, match="new review"):
        gate.require_operator(m, operator_id="op.ewma",
                              version="2", code_digest="c" * 64,
                              scope="forecasting")


def test_unreviewed_operator_id_refuses(manifest_file):
    m = gate.load_manifest(manifest_file(_manifest()))
    with pytest.raises(SystemExit, match="unlisted subject"):
        gate.require_operator(m, operator_id="op.smuggled",
                              version="1", code_digest="c" * 64,
                              scope="forecasting")


# ==============================================================
# regression 5: swapping evidence under a positive label refuses
# ==============================================================

def test_replaced_evidence_refuses(manifest_file):
    m = gate.load_manifest(manifest_file(_manifest()))
    gate.require_eligible(m, "var.close", scope="forecasting",
                          evidence_digest="e" * 64)
    with pytest.raises(SystemExit,
                       match="replacing the evidence"):
        gate.require_eligible(m, "var.close",
                              scope="forecasting",
                              evidence_digest="0" * 64)


def test_positive_label_with_rewritten_entry_refuses(
        manifest_file, tmp_path):
    """Rewriting the evidence digest inside the manifest while
    keeping the decision breaks the manifest digest."""
    doc = _manifest()
    p = manifest_file(doc)
    loaded = json.loads(p.read_text())
    loaded["entries"][0]["digests"]["evidence"] = "9" * 64
    p.write_text(json.dumps(loaded))
    with pytest.raises(SystemExit, match="altered after review"):
        gate.load_manifest(p)


# ==============================================================
# regression 6: same manifest, same ordered universe, fresh
# process
# ==============================================================

def test_universe_is_deterministic_in_a_fresh_process(
        manifest_file, tmp_path):
    import subprocess
    doc = _manifest(entries=[
        _entry(subject_id=s) for s in
        ("var.zulu", "var.alpha", "var.mike", "var.bravo")])
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = manifest_file(doc)
    m = gate.load_manifest(p)
    here = gate.eligible_universe(m, scope="forecasting")
    driver = tmp_path / "fresh.py"
    driver.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from eligibility import gate\n"
        f"m = gate.load_manifest({str(p)!r})\n"
        "print(json.dumps(gate.eligible_universe("
        "m, scope='forecasting')))\n")
    rc = subprocess.run([sys.executable, str(driver)],
                        capture_output=True, text=True,
                        timeout=120)
    assert rc.returncode == 0, rc.stderr[-400:]
    there = json.loads(rc.stdout.strip())
    assert here == there
    assert here == sorted(here), "universe order is not stable"


# ==============================================================
# the required bindings
# ==============================================================

@pytest.mark.parametrize("field", gate.REQUIRED_ENTRY_FIELDS)
def test_every_required_binding_is_mandatory(field,
                                             manifest_file):
    e = _entry()
    del e[field]
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "test", "entries": [e]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    with pytest.raises(SystemExit, match="missing required"):
        gate.load_manifest(manifest_file(doc))


@pytest.mark.parametrize("kind", gate.REQUIRED_DIGEST_KINDS)
def test_every_digest_kind_is_mandatory(kind, manifest_file):
    e = _entry()
    del e["digests"][kind]
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "test", "entries": [e]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    with pytest.raises(SystemExit, match=f"no \\['{kind}'\\]"):
        gate.load_manifest(manifest_file(doc))


def test_event_and_available_time_are_both_required(
        manifest_file):
    e = _entry()
    del e["temporal_availability"]["available_time"]
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "test", "entries": [e]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    with pytest.raises(SystemExit,
                       match="never the same field"):
        gate.load_manifest(manifest_file(doc))


def test_reviewer_must_record_scope_and_reason(manifest_file):
    for field in ("decision_reason", "decision_scope"):
        e = _entry(**{field: "   "})
        doc = {"schema": gate.MANIFEST_SCHEMA,
               "issued_at": "2026-09-10T00:00:00Z",
               "issuer": "x", "scope": "test", "entries": [e]}
        doc["manifest_sha256"] = gate._self_sha(doc)
        with pytest.raises(SystemExit, match="recorded no"):
            gate.load_manifest(manifest_file(doc))


def test_duplicate_subject_refuses(manifest_file):
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "test",
           "entries": [_entry(), _entry(decision="REJECTED")]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    with pytest.raises(SystemExit, match="declares a subject "
                                         "twice"):
        gate.load_manifest(manifest_file(doc))


def test_unknown_fit_scope_refuses(manifest_file):
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "test",
           "entries": [_entry(fit_scope="WHATEVER")]}
    doc["manifest_sha256"] = gate._self_sha(doc)
    with pytest.raises(SystemExit, match="unknown fit scope"):
        gate.load_manifest(manifest_file(doc))


# ==============================================================
# scope is never global
# ==============================================================

def test_eligibility_is_scoped(manifest_file):
    m = gate.load_manifest(manifest_file(_manifest()))
    gate.require_eligible(m, "var.close", scope="forecasting")
    with pytest.raises(SystemExit, match="never global"):
        gate.require_eligible(m, "var.close",
                              scope="live_trading")
    assert gate.eligible_universe(m, scope="live_trading") == []


def test_multi_scope_entry_is_honoured(manifest_file):
    doc = _manifest(entries=[
        _entry(scope=["forecasting", "representation"])])
    doc["manifest_sha256"] = gate._self_sha(doc)
    m = gate.load_manifest(manifest_file(doc))
    gate.require_eligible(m, "var.close", scope="forecasting")
    gate.require_eligible(m, "var.close", scope="representation")
    with pytest.raises(SystemExit, match="never global"):
        gate.require_eligible(m, "var.close", scope="rl_policy")


def test_non_positive_decisions_never_pass(manifest_file):
    for decision in ("REJECTED", "INCONCLUSIVE",
                     "LAB_CALIBRATED", "DOMAIN_REVALIDATED"):
        doc = _manifest(entries=[_entry(decision=decision)])
        doc["manifest_sha256"] = gate._self_sha(doc)
        m = gate.load_manifest(manifest_file(doc))
        if decision == "PUBLICLY_ELIGIBLE":
            continue
        with pytest.raises(SystemExit, match=decision):
            gate.require_eligible(m, "var.close",
                                  scope="forecasting")
        assert gate.eligible_universe(m,
                                      scope="forecasting") == []


def test_lab_calibrated_is_not_public_eligibility(manifest_file):
    """A synthetic calibration result must not open the public
    gate — the bank-index rule, enforced again at the gate."""
    doc = _manifest(entries=[_entry(decision="LAB_CALIBRATED")])
    doc["manifest_sha256"] = gate._self_sha(doc)
    m = gate.load_manifest(manifest_file(doc))
    assert gate.eligible_universe(m, scope="forecasting") == []
