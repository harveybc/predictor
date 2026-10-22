import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import collect_expert_package as collector
from collect_expert_package import digest, identity, validate_mirror


def test_explicit_local_source_needs_no_ssh_and_requires_absolute_path():
    assert collector.source_location("local", "/worker/package") == "/worker/package/"
    assert collector.source_location("worker", "/worker/package") == "worker:/worker/package/"
    with pytest.raises(ValueError):
        collector.source_location("local", "relative")
    with pytest.raises(ValueError):
        collector.source_location("bad host", "/worker")


def test_remote_replica_requires_readback_not_just_successful_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *a, **k: '{"count": 0}')
    with pytest.raises(ValueError, match="REPLICA_MISMATCH"):
        collector.replicate(tmp_path, "worker", "/replica", "/verifier", {"count": 1})
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *a, **k: '{"count": 1, "closed": false}')
    assert collector.replicate(tmp_path, "worker", "/replica", "/verifier", {"count": 1, "closed": False})["verified"]


def example(tmp_path):
    (tmp_path / "ops").mkdir()
    c = {"root": "/worker/package", "pins": {}}
    (tmp_path / "ops/CONFIG.private.json").write_text(json.dumps(c))
    out = tmp_path / "out/job"
    out.mkdir(parents=True)
    (out / "model.pt").write_bytes(b"retained")
    receipt = {"job": {"name": "job"}, "config_sha256": identity(c),
               "hashes": {"model.pt": digest(out / "model.pt")}}
    (tmp_path / "ops/job.RESULT.json").write_text(json.dumps(receipt))
    return c, out


def test_mirror_verifies_retained_hashes_not_just_file_presence(tmp_path):
    c, out = example(tmp_path)
    assert validate_mirror(tmp_path)["count"] == 1
    (out / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="OUTPUT_CHANGED"):
        validate_mirror(tmp_path)


def test_mirror_close_requires_exact_count_and_consumer_hash(tmp_path):
    c, out = example(tmp_path)
    (tmp_path / "CONSUMERS.json").write_text("{}")
    close = {"config_sha256": identity(c), "done": 1, "pending": 0,
             "consumer_sha256": digest(tmp_path / "CONSUMERS.json")}
    (tmp_path / "ops/CLOSE.json").write_text(json.dumps(close))
    assert validate_mirror(tmp_path)["closed"]
    close["done"] = 2
    (tmp_path / "ops/CLOSE.json").write_text(json.dumps(close))
    with pytest.raises(ValueError, match="CLOSURE"):
        validate_mirror(tmp_path)


def test_absent_config_is_not_mirror_success(tmp_path):
    with pytest.raises(ValueError, match="CONFIG_NOT_COLLECTED"):
        validate_mirror(tmp_path)


def test_missing_mount_does_not_create_destination(tmp_path):
    destination = tmp_path / "missing"
    with pytest.raises(ValueError, match="MOUNT_REQUIRED"):
        collector.require_storage(destination, tmp_path)
    assert not destination.exists()


def test_destination_cannot_escape_declared_mount(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.os.path, "ismount", lambda p: True)
    with pytest.raises(ValueError, match="OUTSIDE_MOUNT"):
        collector.require_storage(tmp_path.parent / "other", tmp_path)


def test_present_writable_mount_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.os.path, "ismount", lambda p: True)
    collector.require_storage(tmp_path / "results", tmp_path)


def test_readonly_mount_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(collector.os.path, "ismount", lambda p: True)
    monkeypatch.setattr(collector.os, "statvfs", lambda p: SimpleNamespace(f_flag=collector.os.ST_RDONLY))
    with pytest.raises(ValueError, match="READ_ONLY"):
        collector.require_storage(tmp_path / "results", tmp_path)
