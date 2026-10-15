"""Launcher status must surface execution truth without rewriting its journal."""
import copy
import json

import pytest

from tools.i6dmo import raw_long_campaign as L, status as S


@pytest.fixture
def journal(tmp_path):
    doc = dict(schema="i6dmo.raw_long_automation_status.v1", state="DISPATCHED", phase="dispatch",
               eta="UNMEASURED", completed=4, total=6, as_of=1., retained_marker="launcher-proof")
    (tmp_path / "AUTOMATION_STATUS.json").write_text(json.dumps(doc))
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "PLAN.json").write_text("{}")
    return doc


def test_closed_execution_supersedes_stale_journal_read_only(tmp_path, journal, monkeypatch):
    eta = {phase: dict(kind="DONE", seconds=0.) for phase in ("fit", "score", "publication")}
    execution = dict(state="CLOSED", counts=dict(COMPLETE=6, RUNNING=0, FAILED=0),
                     expected=dict(validation_units=6), eta=eta, generated_at=20.)
    before_execution = copy.deepcopy(execution)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    def compute(root, level):
        assert root == tmp_path / "campaign" and level == "campaign"
        return execution
    monkeypatch.setattr(S, "compute", compute)
    result = L.status(tmp_path)
    assert result["state"] == "CLOSED"
    assert result["eta"] == eta
    assert (result["completed"], result["total"]) == (6, 6)
    assert result["phase"] == "closed" and result["as_of"] == 20.
    assert result["launcher_journal"] == journal
    assert result["execution"] == before_execution == execution
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("state", ["RUNNING", "WAITING_FOR_WORKER", "STALE", "FAILED", "SERVICE_FAILED", "HELD"])
def test_active_execution_is_not_dispatch_or_false_completion(tmp_path, journal, monkeypatch, state):
    eta = dict(fit=dict(kind="UNMEASURED", seconds=None, cause="no own-arm timing"),
               score=dict(kind="MEASURED", seconds=12., basis_n=2),
               publication=dict(kind="UNMEASURED", seconds=None))
    execution = dict(state=state, counts=dict(COMPLETE=2, RUNNING=1, WAIT_DEPENDENCY=3),
                     expected=dict(validation_units=6), eta=eta, generated_at=30.)
    monkeypatch.setattr(S, "compute", lambda *a, **k: execution)
    # Do not change or inherit _existing_status's run/resume state normalization.
    monkeypatch.setattr(L, "_existing_status", lambda *a: pytest.fail("run/resume helper must stay separate"))
    result = L.status(tmp_path)
    assert result["state"] == state
    assert result["eta"] == eta
    assert (result["completed"], result["total"]) == (2, 6)
    assert result["phase"] == "execution" and result["as_of"] == 30.
    assert result["launcher_journal"] == journal
    assert result["execution"] == execution


def test_journal_only_and_unprepared_status_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "compute", lambda *a, **k: pytest.fail("no campaign to query"))
    assert L.status(tmp_path) == dict(state="NOT_PREPARED", eta="UNMEASURED")
    doc = dict(state="PREPARED", phase="prepared", completed=2, total=6, eta="UNMEASURED")
    (tmp_path / "AUTOMATION_STATUS.json").write_text(json.dumps(doc))
    assert L.status(tmp_path) == doc


def test_closed_label_does_not_invent_missing_unit_completions(tmp_path, journal, monkeypatch):
    execution = dict(state="CLOSED", counts=dict(COMPLETE=5, FAILED=1),
                     expected=dict(validation_units=6), eta=dict(fit=dict(kind="DONE", seconds=0.)))
    monkeypatch.setattr(S, "compute", lambda *a, **k: execution)
    result = L.status(tmp_path)
    assert result["state"] == "CLOSED"
    assert (result["completed"], result["total"]) == (5, 6)
    assert result["execution"]["counts"]["FAILED"] == 1
