"""Acquisition is byte download and hash check. It does not import the model."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import df_b77_acquire_20260930 as acquire  # noqa: E402
import df_b77_executable_request_20260930 as b77  # noqa: E402

ACQUIRE_SRC = ROOT / "tools/df_b77_acquire_20260930.py"
EXEC_SRC = ROOT / "tools/df_b77_executable_request_20260930.py"


class NetworkUsed(AssertionError):
    pass


def _refuse_network(*_args, **_kwargs):
    raise NetworkUsed("network")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket, "socket", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_network)


class _Body:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status
        self._off = 0

    def read(self, n=-1):
        if self._off >= len(self._data):
            return b""
        if n is None or n < 0:
            n = len(self._data) - self._off
        chunk = self._data[self._off:self._off + n]
        self._off += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_source_does_not_import_the_model_or_tensorflow():
    text = ACQUIRE_SRC.read_text() + EXEC_SRC.read_text()
    for banned in (
        "import tensorflow",
        "import torch",
        "import sentence_transformers",
        "import modeling_jina_embeddings_v5",
        "import custom_st",
        "from transformers import AutoModel",
        "trust_remote_code=True",
    ):
        assert banned not in text
    exec_source = EXEC_SRC.read_text()
    assert "trust_remote_code=True" not in exec_source
    assert "import urllib" not in exec_source


def test_git_blob_matches_the_object_header():
    data = b"abc"
    assert acquire.git_blob_sha1(data) == hashlib.sha1(b"blob 3\0abc").hexdigest()


def test_a_mismatch_deletes_the_file(tmp_path):
    entry = {"path": "note.txt", "bytes": 3, "git_oid": "0" * 40, "lfs_oid": None, "lfs": False}
    dest = tmp_path / "note.txt"
    with pytest.raises(acquire.AcquisitionRefusal):
        acquire.place_file(entry, dest, url="https://example.invalid/note.txt",
                           opener=lambda _url: _Body(b"abc"))
    assert not dest.exists()


def test_a_matching_blob_is_kept(tmp_path):
    data = b"abc"
    entry = {
        "path": "note.txt",
        "bytes": 3,
        "git_oid": acquire.git_blob_sha1(data),
        "lfs_oid": None,
        "lfs": False,
    }
    dest = tmp_path / "note.txt"
    placed = acquire.place_file(entry, dest, url="https://example.invalid/note.txt",
                                opener=lambda _url: _Body(data))
    assert placed["sha256"] == hashlib.sha256(data).hexdigest()
    assert dest.read_bytes() == data
    assert placed["executed"] is False


def test_lfs_mismatch_is_not_accepted(tmp_path):
    data = b"abc"
    entry = {
        "path": "w.bin",
        "bytes": 3,
        "git_oid": "1" * 40,
        "lfs_oid": "2" * 64,
        "lfs": True,
    }
    dest = tmp_path / "w.bin"
    with pytest.raises(acquire.AcquisitionRefusal, match="lfs"):
        acquire.place_file(entry, dest, url="https://example.invalid/w.bin",
                           opener=lambda _url: _Body(data))
    assert not dest.exists()


def test_jsonl_rows_are_data_not_code(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "a", "label": "x"}\n{"text": "b", "label": "x"}\n{"text": "c", "label": "y"}\n')
    counted = acquire.count_jsonl(path)
    assert counted["rows"] == 3
    assert counted["distinct_labels"] == 2
    assert counted["min_support"] == 1
    assert counted["max_support"] == 2
    assert counted["executed_python"] is False


def test_check_does_not_fetch(tmp_path):
    model = tmp_path / "model" / "jina-embeddings-v5-text-small"
    corpus = tmp_path / "corpus" / "mteb-banking77"
    model.mkdir(parents=True)
    corpus.mkdir(parents=True)
    payload = b"pinned-bytes\n"
    (model / "config.json").write_bytes(payload)
    (corpus / "train.jsonl").write_bytes(payload)
    (model / "REVISION").write_text(acquire.MODEL_REVISION + "\n")
    (corpus / "REVISION").write_text(acquire.CORPUS_REVISION + "\n")
    digest = hashlib.sha256(payload).hexdigest()
    record = {
        "model": {
            "revision": acquire.MODEL_REVISION,
            "files": [{"path": "config.json", "bytes": len(payload), "sha256": digest}],
        },
        "corpus": {
            "revision": acquire.CORPUS_REVISION,
            "files": [{"path": "train.jsonl", "bytes": len(payload), "sha256": digest}],
        },
    }
    assert acquire.check_tree(tmp_path, record)["bytes_fetched"] == 0
    (model / "config.json").write_bytes(payload + b"x")
    with pytest.raises(acquire.AcquisitionRefusal, match="does not match"):
        acquire.check_tree(tmp_path, record)


def test_execution_module_still_refuses_without_the_owner_gate():
    trees = b77.local_trees()
    assert trees["loads_model"] is False
    assert trees["downloads"] is False
    assert trees["trust_remote_code_set_by_this_lane"] is False
    assert trees["snapshot"] == b77.LOCAL_SNAPSHOT_DIR
    assert trees["corpus_repository"] == "mteb/banking77"
    assert "OWNER_TRUST_REMOTE_CODE" in b77.owner_gates_missing({})
    with pytest.raises(SystemExit, match="NOT_RUN"):
        b77.main([])


def test_acquired_tree_matches_the_record_and_fetches_nothing():
    dest = ROOT / ".b77-acquired-20260930"
    record_path = ROOT / "docs/audits/evidence/banking77_request_20260930/ACQUISITION.json"
    result = acquire.check_tree(dest, record_path)
    assert result["bytes_fetched"] == 0
    doc = json.loads(record_path.read_text())
    assert doc["status"] == "ACQUIRED_NOT_EXECUTED"
    assert doc["trial_status"] == "NOT_RUN"
    assert doc["trust_remote_code_set_by_this_lane"] is False
    assert doc["bytes_downloaded"] == 1371334178
    assert doc["model"]["hub_bytes"] == 1369721378
    assert doc["model"]["python_executed"] is False
    assert doc["corpus"]["hub_bytes"] == 1612800
    assert doc["corpus"]["prepare_data_py_executed"] is False
    assert doc["corpus"]["splits"]["train.jsonl"]["rows"] == 10003
    assert doc["corpus"]["splits"]["test.jsonl"]["rows"] == 3080
    assert doc["corpus"]["splits"]["test.jsonl"]["min_support"] == 40
    assert doc["corpus"]["splits"]["train.jsonl"]["min_support"] == 35


def test_corpus_oid_list_is_the_pinned_revision():
    assert acquire.CORPUS_REVISION == "0fd18e25b25c072e09e0d92ab615fda904d66300"
    assert set(acquire.CORPUS_GIT_OIDS) == {
        ".gitattributes", "prepare_data.py", "test.jsonl", "train.jsonl",
    }
    text = (ROOT / "docs/audits/work_plan/RETSU_BANKING77_EXECUTABLE_REQUEST_2026_09_30.md").read_text()
    joined = text + (ROOT / "docs/audits/evidence/banking77_request_20260930/REQUEST.json").read_text()
    assert "/home/" not in joined
    assert os.uname().nodename not in joined
