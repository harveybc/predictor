"""Reporting repair keeps an authenticated fit's original identity and never refits it."""
import json
from types import SimpleNamespace

import pytest

from tests.i6dmo import ops_fakes
from tests.i6dmo.test_ops_market_time_pair import _drive
from tools.i6dmo import campaign, recovery


def _source(tmp_path):
    source = ops_fakes.make_market_time_campaign(tmp_path, name="source")
    counter = tmp_path / "fits.txt"
    assert ops_fakes.run_worker(source, "preferred", counter)["result"] == "COMPLETE"
    return source, counter


def test_successor_imports_original_fit_and_runs_only_the_long_fit(tmp_path, monkeypatch):
    source, counter = _source(tmp_path)
    original = campaign.load_plan(source)["units"][0]
    stored = source / "terminals" / original["identity_sha256"] / "TERMINAL.json"
    before = stored.read_bytes()
    monkeypatch.setattr(ops_fakes, "CODE", "a" * 40)
    monkeypatch.setattr(recovery, "reporting_only_changes", lambda *a: ["tools/i6dmo/reporting.py"])
    successor = ops_fakes.make_market_time_campaign(tmp_path, name="successor", completed_short_fit_source=source)
    units = {u["unit_id"]: u for u in campaign.load_plan(successor)["units"]}
    assert units["short.fit"]["identity"] == original["identity"]
    assert units["short.fit"]["execution_policy"] == "ADOPT_ONLY"
    assert units["short.report"]["identity"]["code_commit"] == "a" * 40
    assert units["short.report"]["identity"]["upstream"] == [original["identity_sha256"]]
    assert stored.read_bytes() == before
    q = campaign.open_queue(successor)
    try:
        assert q.unit("short.fit")["state"] == "COMPLETE"
        assert q.terminal("short.fit")["identity_sha256"] == original["identity_sha256"]
    finally:
        q.close()
    states = _drive(successor, counter)
    assert states["long.fit"] == states["long.report"] == "COMPLETE"
    assert ops_fakes.fit_count(counter) == 2  # original short + successor long only


def test_reuse_rejects_scientific_contract_change(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    target = campaign.load_effective(source)
    target["seed"] += 1
    monkeypatch.setattr(recovery, "reporting_only_changes", lambda *a: [])
    with pytest.raises(campaign.Refused, match="FIT_REUSE_CONTRACT_DIFFERS:seed"):
        recovery.prepare(source, target)


def test_reuse_rejects_corrupted_source_terminal(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    original = campaign.load_plan(source)["units"][0]
    child = source / "terminals" / original["identity_sha256"] / "CHILD_TERMINAL.json"
    child.write_bytes(child.read_bytes() + b" ")
    monkeypatch.setattr(recovery, "reporting_only_changes", lambda *a: [])
    with pytest.raises(ValueError, match="CHILD_TERMINAL_DIGEST"):
        recovery.prepare(source, campaign.load_effective(source))


@pytest.mark.parametrize("changed", ["tools/i6dmo/train.py", "docs/i6n/direct_multi_output_static/EXPERIMENT.mg2.json"])
def test_reuse_rejects_training_code_changes(monkeypatch, changed):
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=changed + "\n"))
    with pytest.raises(ValueError, match="FIT_REUSE_TRAINING_CODE_CHANGED"):
        recovery.reporting_only_changes("source", "target")


def test_reuse_allows_only_the_named_progress_metadata(monkeypatch):
    progress = "docs/audits/evidence/canonical_20261003/MASTER_MILESTONE_STATUS.json"
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=progress + "\n"))
    assert recovery.reporting_only_changes("source", "target") == [progress]
