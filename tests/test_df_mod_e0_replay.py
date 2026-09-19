"""RP19: weight reproduction with code scope and valid numbers. A real attempt (real training with a
tiny allowance) is replayed in a fresh process through the real CLI; then, on in-memory copies of the
replay document: the dictum's NaN restore mutation, and every field separately (NaN / inf / missing /
bool / partial document / non-zero process exit) is refused. The cache reuses a document only for the
same bytes AND the same replay identity (implementation + numeric environment); a document claiming
another implementation, helper or environment is replayed again; a partial file with a non-zero exit
is not a success."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name, where=TOOLS):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("df_mod_e0")
CLOSE = _load("df_mod_e0_close")


@pytest.fixture(scope="module")
def attempt(tmp_path_factory):
    """One real H3 extractor + one sequence arm (a donor pair), trained with 4 updates."""
    root = tmp_path_factory.mktemp("replay") / "attempts"
    ext = root / "ext"
    job = {"kind": "mod_e0_cell", "cell_id": "ext", "hypothesis": "H3", "level": 2, "r": 1, "seed": 1, "arm": "extractor", "role": "CELL", "arch": "A", "training": E.TRAINING, "window": E.WINDOW,
           "max_updates_override": 4, "descriptors": False, "attempt_dir": str(ext)}
    ext.mkdir(parents=True)
    (ext / "job.json").write_text(json.dumps(job))
    E.worker_main(ext / "job.json")
    arm = root / "arm"
    job2 = {**job, "cell_id": "arm", "arm": "sequence", "extractor_weights": str(ext / "weights.weights.h5"), "depends_on": "ext", "donor": "sequence",
            "attempt_dir": str(arm)}
    arm.mkdir()
    (arm / "job.json").write_text(json.dumps(job2))
    E.worker_main(arm / "job.json")
    for a in (ext, arm):
        digest = hashlib.sha256((a / "cell.json").read_bytes()).hexdigest()
        (a / "outcome.json").write_text(json.dumps({"status": "COMPLETED", "verified": {"output_sha256": digest, "rows_written": 1},
                                                    "summary": {"outcome": "COMPLETED", "reason": "", "cost": {"cpu_seconds": 1.0}, "output_sha256": digest}}))
    return {"ext": ext, "arm": arm, "job": job2}


def _replay(attempt_dir, out):
    proc = subprocess.run([sys.executable, "-B", str(TOOLS / "df_mod_e0.py"), "--replay", str(attempt_dir), "--out", str(out)],
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}, capture_output=True, text=True, timeout=900)
    return proc.returncode, json.loads(out.read_text())


def _expected_job(attempt):
    job = attempt["job"]
    return {k: job.get(k) for k in CLOSE.CONSUMED} | {"role": "CELL", "training": job.get("training"), "arm": "sequence", "cell_id": "arm"}


def test_RP19_an_intact_real_replay_verifies_and_carries_its_identity(attempt, tmp_path):
    code, doc = _replay(attempt["arm"], tmp_path / "arm.json")
    assert code == 0 and doc["schema"] == "df_mod_e0_replay.v2" and doc["problems"] == []
    assert set(CLOSE.REPLAY_REQUIRED) <= set(doc) and doc["identity"]["df_mod_e0_sha256"] == hashlib.sha256((TOOLS / "df_mod_e0.py").read_bytes()).hexdigest()
    assert doc["identity"]["numpy"] == np.__version__ and "tensorflow" in doc["identity"]
    job = json.loads((attempt["arm"] / "job.json").read_text())
    e = CLOSE.verify_attempt(attempt["arm"], job, doc, CLOSE.DEFAULT_TOLERANCE, donor_attempt=attempt["ext"])
    assert e["status"] == CLOSE.VERIFIED and e["replay_scope"] == "CURRENT_CODE", e["problems"]
    assert doc["prediction_max_abs_diff"]["validation"] <= 1e-5 and doc["extractor_weights_unequal_layers"] == []
    assert sorted(doc["adapter_activation_max_abs_diff"]) == ["g0_adapt", "g1_adapt"]


@pytest.mark.parametrize("mutation", [
    {"restore_abs_diff": float("nan"), "recorded_best_validation_loss": float("nan")},     # the dictum's mutation
    {"restore_abs_diff": float("nan")}, {"recorded_best_validation_loss": float("inf")}, {"restored_validation_loss": None},
    {"restore_abs_diff": True}, {"prediction_max_abs_diff": {"train": 0.0, "validation": float("nan"), "test": 0.0}},
    {"prediction_max_abs_diff": {"train": 0.0, "test": 0.0}}, {"scale_recomputed_equal": 1}, {"scale_recomputed_equal": None},
    {"adapter_activation_max_abs_diff": {"g0_adapt": float("nan"), "g1_adapt": 0.0}}, {"adapter_activation_max_abs_diff": {"g0_adapt": 0.0}},
    {"adapter_activation_max_abs_diff": None}, {"extractor_weights_unequal_layers": None}, {"_process_exit": 1},
])
def test_RP19_every_non_finite_missing_bool_or_partial_field_is_refused_separately(attempt, tmp_path, mutation):
    code, doc = _replay(attempt["arm"], tmp_path / "arm.json")
    job = json.loads((attempt["arm"] / "job.json").read_text())
    bad = copy.deepcopy(doc)
    bad.update(mutation)
    e = CLOSE.verify_attempt(attempt["arm"], job, bad, CLOSE.DEFAULT_TOLERANCE, donor_attempt=attempt["ext"])
    assert e["status"] == CLOSE.PROBLEMS and any("replay" in q for q in e["problems"]), (mutation, e["problems"])
    # the intact document still verifies (the copy was mutated, not the file)
    ok = CLOSE.verify_attempt(attempt["arm"], job, doc, CLOSE.DEFAULT_TOLERANCE, donor_attempt=attempt["ext"])
    assert ok["status"] == CLOSE.VERIFIED


def test_RP19_a_partial_document_missing_required_keys_is_refused(attempt, tmp_path):
    code, doc = _replay(attempt["arm"], tmp_path / "arm.json")
    job = json.loads((attempt["arm"] / "job.json").read_text())
    for key in ("inputs", "restore_abs_diff", "scale_recomputed_equal"):
        bad = {k: v for k, v in doc.items() if k != key}
        e = CLOSE.verify_attempt(attempt["arm"], job, bad, CLOSE.DEFAULT_TOLERANCE, donor_attempt=attempt["ext"])
        assert e["status"] == CLOSE.PROBLEMS and any("incomplete" in q or key in q for q in e["problems"]), key


def test_RP19_the_cache_binds_implementation_and_environment_not_only_bytes(attempt, tmp_path, monkeypatch):
    out = tmp_path / "replays"
    meta = CLOSE.run_replays([attempt["arm"]], out, workers=1)
    assert meta["notes"]["arm"] == "exit 0" and meta["scopes"]["arm"] == "CURRENT_CODE"
    # same bytes, same identity -> reused without a process
    meta2 = CLOSE.run_replays([attempt["arm"]], out, workers=1)
    assert meta2["notes"]["arm"] == "reused" and meta2["cpu_seconds_children"] < meta["cpu_seconds_children"] + 1.0
    # a document claiming another implementation (helper/constructor), same hashes: NOT reused
    doc = json.loads((out / "arm.json").read_text())
    doc["identity"]["df_mod_e0_sha256"] = "f" * 64
    (out / "arm.json").write_text(json.dumps(doc))
    meta3 = CLOSE.run_replays([attempt["arm"]], out, workers=1)
    assert meta3["notes"]["arm"] == "exit 0" and json.loads((out / "arm.json").read_text())["identity"]["df_mod_e0_sha256"] != "f" * 64
    # another numeric environment (thread count) -> another identity -> replayed again
    meta4 = CLOSE.run_replays([attempt["arm"]], out, workers=1, threads=1)
    assert meta4["notes"]["arm"] == "exit 0" and meta4["identity"]["omp_threads"] == "1"
    # a v1 document (no identity) is never reused as current; with --historic it is reused ONLY under the historic scope
    doc = json.loads((out / "arm.json").read_text())
    doc.pop("identity")
    hist = tmp_path / "historic"
    hist.mkdir()
    (hist / "arm.json").write_text(json.dumps(doc))
    (out / "arm.json").write_text(json.dumps(doc))
    meta5 = CLOSE.run_replays([attempt["arm"]], out, workers=1, historic=hist, only={"other"})
    assert meta5["notes"]["arm"] == "historic" and meta5["scopes"]["arm"] == "HISTORIC_v1_UNBOUND_CODE"
    meta6 = CLOSE.run_replays([attempt["arm"]], out, workers=1, historic=hist, only={"arm"})
    assert meta6["notes"]["arm"] == "exit 0" and meta6["scopes"]["arm"] == "CURRENT_CODE"


def test_RP19_a_process_that_exits_non_zero_after_writing_a_partial_file_is_not_a_success(attempt, tmp_path, monkeypatch):
    out = tmp_path / "replays"
    real_run = CLOSE.subprocess.run

    def fake_run(cmd, **kw):
        if "--replay" in cmd:
            target = Path(cmd[cmd.index("--out") + 1])
            target.write_text(json.dumps({"schema": "df_mod_e0_replay.v2", "attempt": "arm", "problems": [], "inputs": {}, "identity": {}}))
            return subprocess.CompletedProcess(cmd, 1, "", "Traceback: boom")
        return real_run(cmd, **kw)
    monkeypatch.setattr(CLOSE.subprocess, "run", fake_run)
    meta = CLOSE.run_replays([attempt["arm"]], out, workers=1)
    doc = meta["docs"]["arm"]
    assert doc["_process_exit"] == 1 and any("exited 1" in p for p in doc["problems"])
    job = json.loads((attempt["arm"] / "job.json").read_text())
    e = CLOSE.verify_attempt(attempt["arm"], job, doc, CLOSE.DEFAULT_TOLERANCE, donor_attempt=attempt["ext"])
    assert e["status"] == CLOSE.PROBLEMS and any("exited 1" in q for q in e["problems"])
    # the cache does not reuse a failed document either
    monkeypatch.setattr(CLOSE.subprocess, "run", real_run)
    meta2 = CLOSE.run_replays([attempt["arm"]], out, workers=1)
    assert meta2["notes"]["arm"] == "exit 0" and meta2["docs"]["arm"]["problems"] == []
