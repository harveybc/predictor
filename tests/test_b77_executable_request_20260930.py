"""The 3.12 request is closed offline. A missing snapshot file is a refusal, not a download."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import df_b77_executable_request_20260930 as b77  # noqa: E402

EVIDENCE = ROOT / "docs/audits/evidence/banking77_request_20260930"
REQUEST = EVIDENCE / "REQUEST.json"
PINS = EVIDENCE / "pins-cp312.txt"
SYNTHETIC = EVIDENCE / "SYNTHETIC_METRIC_RECORD.json"
DOC = ROOT / "docs/audits/work_plan/RETSU_BANKING77_EXECUTABLE_REQUEST_2026_09_30.md"

CP312_TOTAL = 3230416669
REUSABLE = 2284173652
INDEX_BYTES = 946243017
WEIGHTS = 1369721378
WITH_HUB = CP312_TOTAL + WEIGHTS + WEIGHTS


class NetworkUsed(AssertionError):
    pass


def _refuse_network(*_args, **_kwargs):
    raise NetworkUsed("network")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if b77.is_credential_key(key):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(socket, "socket", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_network)


def test_process_env_is_clean():
    b77.require_clean_process_env()


def test_missing_file_is_a_refusal_not_a_download(tmp_path):
    b77.build_synthetic_snapshot(tmp_path)
    target = tmp_path / "adapters/classification/adapter_model.safetensors"
    target.unlink()
    ledger = b77.DownloadLedger()
    with pytest.raises(b77.LocalSnapshotRefusal, match="not a download") as caught:
        b77.close_load_path(tmp_path, pinned_revision=b77.PINNED_REVISION, ledger=ledger)
    assert "adapters/classification/adapter_model.safetensors" in caught.value.missing
    assert ledger.events == []
    assert ledger.bytes_fetched == 0


def test_hash_mismatch_is_a_refusal_not_a_download(tmp_path):
    written = b77.build_synthetic_snapshot(tmp_path)
    path = tmp_path / "config.json"
    path.write_bytes(path.read_bytes() + b"\n")
    ledger = b77.DownloadLedger()
    expected = {name: item["sha256"] for name, item in written.items()}
    with pytest.raises(b77.LocalSnapshotRefusal, match="not a download"):
        b77.close_load_path(
            tmp_path,
            pinned_revision=b77.PINNED_REVISION,
            ledger=ledger,
            expected_sha256=expected,
        )
    assert ledger.bytes_fetched == 0
    assert ledger.events == []


def test_unpinned_snapshot_download_cannot_fetch():
    ledger = b77.DownloadLedger()
    with pytest.raises(b77.UnpinnedSnapshotDownloadRefusal):
        b77.snapshot_download_without_revision(
            "jinaai/jina-embeddings-v5-text-small",
            allow_patterns=["adapters/*"],
            revision=None,
            ledger=ledger,
        )
    assert ledger.bytes_fetched == 0
    assert ledger.events[0]["revision"] is None
    assert ledger.events[0]["fetched"] is False


def test_a_different_revision_cannot_be_fetched(tmp_path):
    b77.build_synthetic_snapshot(tmp_path)
    ledger = b77.DownloadLedger()
    with pytest.raises(b77.UnpinnedSnapshotDownloadRefusal):
        b77.close_load_path(tmp_path, pinned_revision="a" * 40, ledger=ledger)
    assert ledger.bytes_fetched == 0
    assert ledger.events[0]["revision"] == "a" * 40


def test_repo_id_is_the_refused_download_branch(tmp_path):
    ledger = b77.DownloadLedger()
    missing = tmp_path / "not-a-snapshot"
    with pytest.raises(b77.UnpinnedSnapshotDownloadRefusal):
        b77.close_load_path(missing, pinned_revision=b77.PINNED_REVISION, ledger=ledger)
    assert ledger.events[0]["revision"] is None
    assert ledger.events[0]["allow_patterns"] == ["adapters/*"]
    assert ledger.bytes_fetched == 0


def test_complete_synthetic_snapshot_hashes_every_file_and_imports_nothing(tmp_path):
    written = b77.build_synthetic_snapshot(tmp_path)
    before = set(sys.modules)
    manifest = b77.close_load_path(
        tmp_path,
        pinned_revision=b77.PINNED_REVISION,
        expected_sha256={name: item["sha256"] for name, item in written.items()},
    )
    assert set(sys.modules) == before
    assert manifest["trust_remote_code"] is False
    assert manifest["imported_remote_modules"] is False
    assert manifest["snapshot_download_invoked"] is False
    assert manifest["bytes_fetched"] == 0
    assert manifest["security_certification"] is False
    paths = [item["path"] for item in manifest["files"]]
    assert paths == sorted(written)
    for item in manifest["files"]:
        assert item["revision"] == b77.PINNED_REVISION
        assert item["sha256"] == written[item["path"]]["sha256"]
        assert item["bytes_are_the_pinned_revision_blobs"] is False
    inert = (tmp_path / "modeling_jina_embeddings_v5.py").read_bytes()
    assert inert.startswith(b"INERT ")
    assert b"trust_remote_code" not in inert


def test_synthetic_record_is_not_a_banking77_score():
    record = b77.synthetic_metric_record()
    committed = json.loads(SYNTHETIC.read_text())
    assert record == committed
    assert record["is_banking77_score"] is False
    assert record["model"] is None
    assert record["real_trial_status"] == "NOT_RUN"
    assert record["main_score"] == 0.75
    assert record["scores"]["test"][0]["main_score"] == 0.75
    assert record["scores"]["test"][0]["ap"] is None
    assert "banking" not in record["corpus"]
    assert record["gold"] == list(b77.GOLD)
    assert record["predicted"] == list(b77.PREDICTED)


def test_request_is_not_run_and_pins_stay_on_3_12():
    document = json.loads(REQUEST.read_text())
    text = REQUEST.read_text() + DOC.read_text() + PINS.read_text() + SYNTHETIC.read_text()
    assert "/home/" not in text
    assert os.uname().nodename not in text
    assert str(Path.home()) not in text
    assert document["status"] == "NOT_RUN"
    assert document["executed_by_this_lane"] is False
    assert document["any_dependency_requires_python_ge_3_13"] is False
    assert document["cp312_pin_list_status"] == "CLOSED"
    assert document["versions_changed"] == []
    assert document["licence"] == "CC BY-NC 4.0"
    assert document["trust_remote_code_set_by_this_lane"] is False
    assert document["device"] == "UNDECIDED"
    assert document["device_decision"] == "PROPOSAL"
    assert document["gpu_requested"] is False
    assert document["absence_of_subprocess_eval_exec_is_not_a_security_certification"] is True
    pins = document["pins"]
    assert len(pins) == 87
    assert [item["pin"] for item in pins if item["pin"].startswith("mteb==")] == ["mteb==2.9.0"]
    assert any(item["pin"] == "sentence-transformers==5.1.2" for item in pins)
    assert any(item["pin"] == "torch==2.14.0" for item in pins)
    assert any(item["pin"] == "transformers==4.57.6" for item in pins)
    measured = indexed = 0
    for item in pins:
        spec = item["requires_python"]
        if spec is not None:
            parsed = SpecifierSet(spec)
            assert parsed.contains("3.12.7")
            assert parsed.contains("3.12.13")
        assert item["provenance"] in {"MEASURED", "INDEX_METADATA"}
        assert "cp313" not in item["filename"]
        assert "py313" not in item["filename"]
        if item["provenance"] == "MEASURED":
            measured += item["bytes"]
            assert item["locally_hashed"] is True
        else:
            indexed += item["bytes"]
            assert item["locally_hashed"] is False
            assert item["downloaded_by_this_lane"] is False
    assert measured == REUSABLE
    assert indexed == INDEX_BYTES
    assert measured + indexed == CP312_TOTAL
    disk = document["disk"]
    assert disk["weight_tree_bytes"] == WEIGHTS
    assert disk["weight_tree_label"] == "MEASURED"
    assert disk["banking77_corpus_bytes"] == 1612800
    assert disk["banking77_corpus_label"] == "MEASURED"
    assert disk["this_phase_bytes_downloaded"] == 1371334178
    assert disk["total_including_hub_cache_copy_bytes"] == WITH_HUB
    assert disk["hub_cache_copy_label"] == "NOT_PRESENT"
    assert disk["pip_cache_label"] == "NOT_PRESENT"
    assert disk["installed_site_packages_label"] == "NOT_PRESENT"
    assert document["model"]["revision"] == b77.PINNED_REVISION
    assert document["model"]["weight_tree_bytes"] == WEIGHTS
    assert sum(item["bytes"] for item in document["model"]["files"]) == WEIGHTS
    assert any(
        item["path"] == "model.safetensors"
        and item["presence"] == "MEASURED"
        and item["sha256"] == "045fa75ff963a528cda2589fb1ca0a9ad848b53511780ed4f08f6fe10f6167c3"
        for item in document["model"]["files"]
    )
    assert all(item["presence"] == "MEASURED" and item["executed"] is False for item in document["model"]["files"])
    command = document["command"]
    assert "OWNER_TRUST_REMOTE_CODE" in command
    assert "OWNER_RUN_RESEARCH" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "does not set CUDA_VISIBLE_DEVICES empty for it" in command
    assert ".b77-acquired-20260930/model/jina-embeddings-v5-text-small" in command
    assert "0fd18e25b25c072e09e0d92ab615fda904d66300" in command
    assert "--model-revision" in command
    assert b77.PINNED_REVISION in command
    assert "trust_remote_code=True" not in command
    assert command in DOC.read_text()
    for figure in (str(CP312_TOTAL), str(REUSABLE), str(INDEX_BYTES), str(WEIGHTS), str(WITH_HUB)):
        assert figure in DOC.read_text()
    assert "not a security certification" in DOC.read_text()
    assert "NOT_RUN" in DOC.read_text()
    pin_text = PINS.read_text()
    assert pin_text.count("--hash=sha256:") == 87
    for item in pins:
        assert f"--hash=sha256:{item['sha256']}" in pin_text


def test_the_command_refuses_before_any_hub_call():
    command = json.loads(REQUEST.read_text())["command"]
    env = {"PATH": "/usr/bin:/bin"}
    parsed = subprocess.run(["bash", "-n"], input=command, text=True, env=env, check=False)
    assert parsed.returncode == 0
    refused = subprocess.run(["bash", "-s"], input=command, text=True, env=env, check=False, capture_output=True)
    assert refused.returncode != 0
    assert "NOT_RUN" in refused.stderr
    assert "files.pythonhosted.org" not in refused.stderr
    assert "huggingface.co" not in refused.stderr


def test_module_source_does_not_set_trust_remote_code_or_import_the_hub():
    source = (ROOT / "tools/df_b77_executable_request_20260930.py").read_text()
    assert "trust_remote_code=True" not in source
    assert "huggingface_hub" not in source
    assert "import urllib" not in source
    assert "import requests" not in source
    without_refusal = source.replace("snapshot_download_without_revision(", "")
    assert "snapshot_download(" not in without_refusal
