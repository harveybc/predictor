import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from collect_expert_package import digest, identity, validate_mirror


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
