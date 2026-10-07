"""The successor cannot create a weekly queue while a wave task is pending."""

from tools import fs4_wave_successor as successor


def test_pending_wave_does_not_open_any_input(monkeypatch, tmp_path):
    monkeypatch.setattr(successor.progress, "summarize", lambda *_: {
        "counts": {arm: {"PENDING": 1, "LEASED": 0} for arm in successor.partial.ARMS}})
    monkeypatch.setattr(successor.full, "TaskStore", lambda *_: (_ for _ in ()).throw(AssertionError("store opened")))
    monkeypatch.setattr(successor.weekly, "initialize", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("weekly opened")))
    result = successor.tick(queue=tmp_path / "none", manifest_path=tmp_path / "none",
                            receipt_root=tmp_path / "none", report_path=tmp_path / "none",
                            consolidated_paths=[], out_dir=tmp_path, weekly_db=tmp_path / "none")
    assert result["stage"] == "WAITING_FOR_WAVE"
