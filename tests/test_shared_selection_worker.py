import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "shared_selection_worker", Path(__file__).parents[1] / "tools/shared_selection_worker.py"
)
worker = importlib.util.module_from_spec(SPEC)


@pytest.fixture(autouse=True, scope="module")
def module():
    SPEC.loader.exec_module(worker)


def test_latest_base_start_invalidates_old_done():
    log = "t1 base shard 0/2 start\nt2 base shard 0/2 DONE\nt3 base shard 0/2 start\n"
    assert not worker.base_marker_ready(log, "0/2")
    assert worker.base_marker_ready(log + "t4 base shard 0/2 DONE\n", "0/2")


def test_other_stage_and_shard_do_not_close_base():
    log = "t1 base shard 0/2 start\nt2 contrast shard 0/2 DONE\nt3 base shard 1/2 DONE\n"
    assert not worker.base_marker_ready(log, "0/2")


def test_process_query_is_not_a_scientific_worker():
    assert not worker.is_science_process(["bash", "-c", "pgrep -f 'pool_selection_v2.py run --stage base'"], "/queue")
    assert worker.is_science_process(["/env/bin/python", "tools/pool_selection_v2.py", "run", "--stage", "base", "--out", "/queue/out_full"], "/queue")
    assert not worker.is_science_process(["python", "tools/pool_selection_v2.py", "run", "--out", "/other/out_full"], "/queue")


def test_ledger_last_attempt_wins(tmp_path):
    p = tmp_path / "ledger"
    p.write_text('\n'.join(json.dumps(r) for r in [{"key": "a", "error": "bad"}, {"key": "a", "val_loss": 1.0}]) + '\n')
    rows, unique = worker.read_ledger(p)
    assert not unique
    assert worker.successful(rows, ["a"])


@pytest.mark.parametrize("row", [{"key": "a", "error": "bad"}, {"key": "a"}, {"key": "a", "val_loss": float("nan")}, {"key": "a", "val_loss": None}])
def test_exit_zero_or_bad_numeric_is_not_success(row):
    assert not worker.successful({"a": row}, ["a"])


def test_missing_or_failed_key_blocks_stage():
    assert not worker.successful({"a": {"val_loss": 0.1}}, ["a", "b"])
    assert worker.successful({}, [])


def test_partial_line_is_not_silently_adopted(tmp_path):
    p = tmp_path / "ledger"
    p.write_text('{"key":"a"')
    with pytest.raises(json.JSONDecodeError):
        worker.read_ledger(p)


def test_global_index_sharding_is_preserved():
    units = [("b0", "base"), ("b1", "base"), ("c0", "contrast"), ("c1", "contrast"), ("r0", "rescue")]
    assert worker.stage_keys(units, "contrast", "0/2") == ["c0"]
    assert worker.stage_keys(units, "rescue", "0/2") == ["r0"]
    assert worker.stage_keys(units, "contrast") == ["c0", "c1"]


def test_pin_mismatch_rejected(tmp_path):
    p = tmp_path / "input"
    p.write_bytes(b"original")
    cfg = {"pins": {str(p): worker.sha256(p)}}
    worker.verify_pins(cfg)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        worker.verify_pins(cfg)


def test_command_keeps_exact_scientific_arguments(tmp_path):
    cfg = {"root": str(tmp_path), "python": "/env/python"}
    cmd = worker.science_command(cfg, "contrast")
    assert cmd[:4] == ["/env/python", "tools/pool_selection_v2.py", "run", "--stage"]
    assert cmd[cmd.index("--shard") + 1] == "0/2"
    assert cmd.count("--store") == 2
    assert "--fit-end" not in cmd  # preserve the pinned runner's default


def test_new_stage_waits_for_global_prior_stage(tmp_path):
    cfg = {"root": str(tmp_path), "base_keys": ["b0", "b1"]}
    out = tmp_path / "out_full"
    out.mkdir()
    (tmp_path / "LOG").write_text("t base shard 0/2 start\nt base shard 0/2 DONE\n")
    (out / "LEDGER.jsonl").write_text(json.dumps({"key": "b0", "val_loss": 1}) + "\n")
    assert worker.barrier(cfg, "contrast")[0] == "WAIT_BASE_MERGE"
    with (out / "LEDGER.jsonl").open("a") as f:
        f.write(json.dumps({"key": "b1", "val_loss": 1}) + "\n")
    assert worker.barrier(cfg, "contrast")[0] == "READY"


def test_rescue_waits_for_unique_merged_contrasts(tmp_path):
    cfg = {"root": str(tmp_path), "base_keys": ["b"]}
    (tmp_path / "out_full").mkdir()
    (tmp_path / "musashi_ops").mkdir()
    (tmp_path / "LOG").write_text("t base shard 0/2 start\nt base shard 0/2 DONE\n")
    (tmp_path / "musashi_ops/contrast_KEYS.json").write_text(json.dumps({"all_keys": ["c0", "c1"]}))
    p = tmp_path / "out_full/LEDGER.jsonl"
    rows = [{"key": k, "val_loss": 1} for k in ["b", "c0", "c1"]]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows + [rows[-1]]))
    assert worker.barrier(cfg, "rescue")[0] == "WAIT_UNIQUE_MERGE"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert worker.barrier(cfg, "rescue")[0] == "READY"


def test_atomic_json_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    worker.atomic_json(p, {"stage": "contrast"})
    assert json.loads(p.read_text()) == {"stage": "contrast"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("complete", [False, True])
def test_runner_checks_ledger_even_when_child_exits_zero(tmp_path, monkeypatch, complete):
    ops = tmp_path / "musashi_ops"
    ops.mkdir()
    (tmp_path / "code").mkdir()
    (tmp_path / "out_full").mkdir()
    (ops / "contrast_KEYS.json").write_text(json.dumps({"own_keys": ["c0"]}))
    row = {"key": "c0", "val_loss": 0.2} if complete else {"key": "c0", "error": "synthetic"}
    (tmp_path / "out_full/LEDGER.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    cfg = {"root": str(tmp_path), "pins": {}, "python": "python"}
    assert worker.run_stage(cfg, "contrast") == (0 if complete else 1)
    state = json.loads((ops / "contrast_RUN.json").read_text())
    assert state["attempt"] == 1
    assert state["state"] == ("DONE" if complete else "RETRY_PENDING")
    assert (" DONE" in (tmp_path / "LOG_full").read_text()) == complete


def test_tick_does_not_probe_or_dispatch_while_base_running(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "science_processes", lambda root: [123])
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: pytest.fail("must not dispatch"))
    worker.tick({"root": str(tmp_path)}, "/unused")
    assert json.loads((tmp_path / "musashi_ops/STATUS.json").read_text())["state"] == "WAIT_ACTIVE_SELECTION"


def test_tick_waits_for_peer_after_own_contrast(tmp_path, monkeypatch):
    (tmp_path / "musashi_ops").mkdir()
    (tmp_path / "out_full").mkdir()
    (tmp_path / "LOG").write_text("t base shard 0/2 start\nt base shard 0/2 DONE\n")
    (tmp_path / "musashi_ops/contrast_KEYS.json").write_text(json.dumps({"own_keys": ["c0"], "all_keys": ["c0", "c1"]}))
    (tmp_path / "musashi_ops/contrast_RUN.json").write_text(json.dumps({"state": "DONE"}))
    (tmp_path / "out_full/LEDGER.jsonl").write_text(''.join(json.dumps({"key": k, "val_loss": 1}) + '\n' for k in ["b", "c0"]))
    monkeypatch.setattr(worker, "science_processes", lambda root: [])
    monkeypatch.setattr(worker, "unit_state", lambda unit: "inactive")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: pytest.fail("must await merge"))
    worker.tick({"root": str(tmp_path), "base_keys": ["b"]}, "/unused")
    status = json.loads((tmp_path / "musashi_ops/STATUS.json").read_text())
    assert status["state"] == "WAIT_CONTRAST_MERGE"


@pytest.mark.parametrize("active", [False, True])
def test_tick_dispatches_only_one_admitted_unit(tmp_path, monkeypatch, active):
    ops = tmp_path / "musashi_ops"
    ops.mkdir()
    (tmp_path / "out_full").mkdir()
    (tmp_path / "out_full/LEDGER.jsonl").write_text("")
    (ops / "contrast_KEYS.json").write_text(json.dumps({"own_keys": ["c0"], "all_keys": ["c0", "c1"]}))
    monkeypatch.setattr(worker, "science_processes", lambda root: [])
    monkeypatch.setattr(worker, "unit_state", lambda unit: "active" if active else "inactive")
    calls = []
    monkeypatch.setattr(worker.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    cfg = {"root": str(tmp_path), "pins": {}, "admission": "/admit", "python": "/python"}
    worker.tick(cfg, "/config")
    dispatched = [c for c in calls if c[0] == "systemd-run"]
    assert len(dispatched) == (0 if active else 1)
    if dispatched:
        assert dispatched[0][dispatched[0].index("-m") + 1] == "3G"
        assert "--collect" in dispatched[0]
        assert "run-stage" in dispatched[0]
