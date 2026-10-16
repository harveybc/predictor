"""Attempt-local authenticated donor transport, with no model or GPU import."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools.i6dmo import donor_transport as DT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    source = tmp_path / "coordinator-only"
    source.mkdir()
    weights = source / "donor.npz"
    weights.write_bytes(b"authenticated weight fixture")
    terminal = source / "TERMINAL.json"
    terminal.write_text(json.dumps(dict(feature="feature_a", artifacts={"donor.npz": sha(weights)})))
    return dict(regime="R2", r2_donors=dict(
        donors={"feature_a": dict(terminal=dict(path=str(terminal), sha256=sha(terminal)))},
        objective={"name": "retained_objective"}, provenance={"campaign_id": "original"}))


def test_relocated_attempt_imports_without_original_coordinator_path(tmp_path):
    spec = fixture(tmp_path)
    original = copy.deepcopy(spec)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    staged = DT.stage(spec, attempt)
    assert spec == original
    target = tmp_path / "worker-attempt"
    shutil.copytree(attempt, target)
    shutil.rmtree(tmp_path / "coordinator-only")
    resolved = DT.resolve(staged, target)
    ref = resolved["r2_donors"]["donors"]["feature_a"]["terminal"]
    assert Path(ref["path"]).is_relative_to(target)
    assert sha(ref["path"]) == original["r2_donors"]["donors"]["feature_a"]["terminal"]["sha256"]
    assert resolved["r2_donors"]["provenance"] == original["r2_donors"]["provenance"]


@pytest.mark.parametrize("name", ["TERMINAL.json", "donor.npz"])
def test_changed_worker_bytes_are_refused_before_import(tmp_path, name):
    spec = fixture(tmp_path)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    staged = DT.stage(spec, attempt)
    (attempt / "staged_donors/00" / name).write_bytes(b"forged")
    with pytest.raises(DT.TransportError):
        DT.resolve(staged, attempt)


def test_source_weight_hash_must_match_retained_terminal(tmp_path):
    spec = fixture(tmp_path)
    (tmp_path / "coordinator-only/donor.npz").write_bytes(b"changed")
    with pytest.raises(DT.TransportError, match="WEIGHTS_DIGEST"):
        DT.stage(spec, tmp_path / "attempt")


@pytest.mark.parametrize("replacement", ["../outside/TERMINAL.json", "/outside/TERMINAL.json"])
def test_staged_path_escape_refused(tmp_path, replacement):
    staged = DT.stage(fixture(tmp_path), tmp_path / "attempt")
    staged["r2_donors"]["donors"]["feature_a"]["terminal"]["path"] = replacement
    with pytest.raises(DT.TransportError, match="PATH"):
        DT.resolve(staged, tmp_path / "attempt")


def test_symlinked_worker_weights_refused(tmp_path):
    attempt = tmp_path / "attempt"
    staged = DT.stage(fixture(tmp_path), attempt)
    weights = attempt / "staged_donors/00/donor.npz"
    outside = tmp_path / "outside.npz"
    outside.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(outside)
    with pytest.raises(DT.TransportError, match="PATH"):
        DT.resolve(staged, attempt)


def test_r0_never_gets_donor_binding(tmp_path):
    spec = fixture(tmp_path)
    spec["regime"] = "R0"
    with pytest.raises(DT.TransportError, match="R0_HAS_DONORS"):
        DT.stage(spec, tmp_path / "attempt")


def test_nonconsuming_unit_unchanged(tmp_path):
    spec = dict(kind="report", regime="R0")
    assert DT.stage(spec, tmp_path) == spec
    assert DT.resolve(spec, tmp_path) == spec


def test_real_child_entry_resolves_relocated_donors_before_fit(tmp_path, monkeypatch):
    from tools.i6dmo import unit_runner as U
    spec = fixture(tmp_path)
    spec.update(kind="fit", unit_id="r2.fit", identity_sha256="a" * 64)
    staged = DT.stage(spec, tmp_path / "coordinator-attempt")
    target = tmp_path / "remote-attempt"
    shutil.copytree(tmp_path / "coordinator-attempt", target)
    shutil.rmtree(tmp_path / "coordinator-only")
    path = target / "UNIT_SPEC.json"
    path.write_text(json.dumps(staged))
    calls = []
    def fit(received, out, progress):
        ref = received["r2_donors"]["donors"]["feature_a"]["terminal"]
        assert Path(ref["path"]).is_relative_to(target)
        assert sha(ref["path"]) == ref["sha256"]
        calls.append(received)
        return 0
    monkeypatch.setattr(U, "run_fit", fit)
    argv = ["--kind", "fit", "run-unit", "--spec", str(path), "--out", str(target / "out"),
            "--progress", str(target / "PROGRESS.json")]
    assert U.main(argv) == 0 and len(calls) == 1
    (target / "staged_donors/00/donor.npz").write_bytes(b"mutation")
    assert U.main(argv) == 0 and len(calls) == 1
    terminal = json.loads((target / "out/TERMINAL.json").read_text())
    assert terminal["status"] == "FAILED"
    assert terminal["disposition"] == "R2_TRANSPORT_WEIGHTS_DIGEST"
