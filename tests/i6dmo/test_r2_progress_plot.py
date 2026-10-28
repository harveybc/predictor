"""Public reporting counts adoption without exporting private launcher logs."""
import json

from tools import plot_i6dmo_r2_progress as P


def test_snapshot_keeps_retained_milestones_and_excludes_private_state(tmp_path, monkeypatch):
    milestones = tmp_path / "milestones.json"
    rows = [dict(id="M4", title="Branches", progress=48)]
    milestones.write_text(json.dumps(dict(milestones=rows)))
    monkeypatch.setattr(P.S, "status", lambda root: dict(action="WAIT_SHORT", families=dict(
        short=dict(state="RUNNING", campaign_id="fixture"), long=None),
        launcher_journal=dict(error="PRIVATE_LOG_NOT_FOR_EXPORT")))
    monkeypatch.setattr(P.STATUS, "compute", lambda *a, **kw: dict(units=[
        dict(kind="pretrain", state="COMPLETE", reuse=dict(source="original"), unit_id="branch"),
        dict(kind="fit", state="RUNNING", reuse=None, unit_id="r0.fit")]))
    doc = P.snapshot(tmp_path, milestones)
    assert "PRIVATE" not in json.dumps(doc)
    assert doc["milestones"] == rows
    assert doc["families"]["short"]["adopted"] == 1
    assert doc["families"]["short"]["steps"][0] == dict(label="Donantes", complete=1, total=19)
    assert doc["families"]["short"]["running"] == ["r0.fit"]
    assert doc["families"]["long"]["state"] == "PENDING_SHORT_CLOSE"
    assert doc["test_open"] is False
