"""Independent R2 release probes. Temporary synthetic custody; no fits or services.

Run with --noconftest, hidden CUDA, crispdm-run -m 2G -t 180s.
Deployment is a capture boundary, never an executable service operation.
"""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from tests import test_r2_recovery as recovery_fixture
from tests.i6dmo.ops_fakes import private_spec
from tools.i6dmo import campaign as C, design as D, donor_transport as DT
from tools.i6dmo import r2_design as RD, r2_recovery as RR, worker as W
from tools.i6dmo import r2_downsampled_campaign as SEQ, deploy
from tools.i6n_swarm.common import atomic_json, file_sha256

BASE = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static"


@pytest.fixture
def custody(tmp_path):
    root, kwargs = recovery_fixture.custody.__wrapped__(tmp_path)
    # The helper-only fixture omits OPS scheduling fields. Supply them in this
    # temporary source capsule before sealing it for actual factory consumption.
    plan = json.loads((root / "PLAN.json").read_text())
    db = sqlite3.connect(root / "backup/queue.sqlite3")
    try:
        for unit in plan["units"]:
            if unit["kind"] == "pretrain":
                unit.update(stage=C.R2_STAGES[0], device="gpu", test_open=False, depends_on=[])
                db.execute("UPDATE units SET spec=? WHERE unit_id=?", (json.dumps(unit), unit["unit_id"]))
        db.commit()
    finally:
        db.close()
    atomic_json(root / "PLAN.json", plan)
    kwargs["source_pins"]["plan_sha256"] = file_sha256(root / "PLAN.json")
    for mirror in kwargs["mirror_roots"]:
        shutil.copy2(root / "PLAN.json", mirror / "PLAN.json")
        shutil.copy2(root / "backup/queue.sqlite3", mirror / "backup/queue.sqlite3")
    target = kwargs["r0_target"]
    changed_code = Path(target["code_root"]) / "tools/i6dmo/model.py"
    atomic_json(changed_code, dict(successor_training_code="changed, requiring a fresh R0"))
    target["expected_code_files"]["tools/i6dmo/model.py"] = file_sha256(changed_code)
    return root, kwargs


def _new_campaign(tmp_path, root):
    contract_path = BASE / "matched_short/R2_CONTRAST_DESIGN.ruled.json"
    contract = json.loads(contract_path.read_text())
    spec_path = tmp_path / "source-deployment.json"
    atomic_json(spec_path, private_spec(tmp_path))
    pins = dict(common_population=dict(base="experiment_storage", relpath="population", sha256="b" * 64,
        population_sha256=contract["shared"]["population_sha256"]),
        matched_input=dict(relpath="input", sha256="c" * 64))
    C.init_r2_contrast(contract_path, BASE / "matched_short/EXPERIMENT.mg2m.json", spec_path,
        root, code_commit="d" * 40, data_pins=pins)
    return spec_path


def _sequence_fixture(tmp_path, monkeypatch, custody):
    work = tmp_path / "sequence"
    root = work / "short/campaign"
    prepared = _recovered_factory(custody, tmp_path, monkeypatch, target=root)
    spec = Path(prepared["deployment"])
    config = dict(schema="i6dmo.r2_downsampled_sequence.v1", work_root=str(work),
        source_deployment=str(spec), short_source_campaign=str(tmp_path / "original"),
        short_source_pins=dict(effective_sha256="e" * 64, plan_sha256="f" * 64, code_commit="c" * 40),
        long_source_work_root=str(tmp_path / "original-long"), input_dir=str(tmp_path / "input"))
    identity = dict(config=config, code_commit="d" * 40,
        deployment_source_sha256=file_sha256(spec), code_files={})
    atomic_json(work / "SETUP.json", identity)
    # Release-file approval is outside this interrupted-deployment experiment.
    monkeypatch.setattr(SEQ, "release", lambda: dict(code_files={}))
    monkeypatch.setattr(SEQ, "_setup_identity", lambda *a: copy.deepcopy(identity))
    monkeypatch.setattr(SEQ.MR, "validate", lambda *a: dict(state="VALID"))
    binding = SEQ._binding(prepared, "short", "d" * 40, {})
    atomic_json(root.parent / "INITIALIZATION.json", binding)
    atomic_json(root / "INITIALIZED.json", SEQ._verify_campaign(root, binding, initialized=False))
    calls = []
    monkeypatch.setattr(SEQ.L, "_verify_runtime", lambda *a, **kw: calls.append("attest"))
    def capture_deploy(root):
        calls.append("deploy")
        atomic_json(Path(root) / "DEPLOY.json", dict(state="DEPLOYED"))
    monkeypatch.setattr(deploy, "deploy", capture_deploy)
    return config, root, calls


def test_sequence_resumes_real_initialized_queue_before_first_deploy(tmp_path, monkeypatch, custody):
    config, root, calls = _sequence_fixture(tmp_path, monkeypatch, custody)
    assert (root / "PLAN.json").is_file() and not (root / "DEPLOY.json").exists()
    q = C.open_queue(root)
    try:
        assert len(q.units()) == 25
    finally:
        q.close()
    SEQ.run(config)
    assert calls.count("deploy") == 1, "initialized queue must not wait forever before deployment"
    assert calls.count("attest") == 3
    SEQ.run(config)
    assert calls.count("deploy") == 1, "already-deployed work must not be enabled again"


@pytest.mark.parametrize("changed", ["code_commit", "data_pins", "donor_recovery", "physical_custody", "queue"])
def test_sequence_rejects_existing_campaign_with_different_execution_identity(
        tmp_path, monkeypatch, custody, changed):
    config, root, calls = _sequence_fixture(tmp_path, monkeypatch, custody)
    if changed == "queue":
        q = C.open_queue(root)
        try:
            q._db.execute("UPDATE units SET spec='{}' WHERE unit_id='r0.fit'")
        finally:
            q.close()
    elif changed == "physical_custody":
        (tmp_path / "REMOTE_CUSTODY.synthetic.json").write_bytes(b"changed")
    else:
        doc = C.load_effective(root)
        if changed == "donor_recovery":
            doc["r2"]["donor_recovery"]["physical_custody_sha256"] = "0" * 64
        else:
            doc[changed] = "a" * 40 if changed == "code_commit" else {}
        atomic_json(root / "EFFECTIVE_CONFIG.json", C._seal(doc))
    with pytest.raises((ValueError, C.Refused), match="SEQUENCE_"):
        SEQ.run(config)
    assert "deploy" not in calls


def _binding(manifest, root):
    refs = {}
    for i, (feature, ref) in enumerate(manifest["pretraining_refs"].items()):
        child = json.loads(Path(ref["path"]).read_text())["terminal"]
        art = child["artifacts"]["donor_terminal"]
        source = Path(ref["path"]).parent / "out" / art["path"]
        target = root / str(i) / "TERMINAL.json"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        shutil.copy2(source.parent / "donor.npz", target.parent / "donor.npz")
        refs[feature] = dict(terminal=dict(path=str(target), sha256=art["sha256"]))
    return dict(regime="R2", r2_donors=dict(donors=refs,
        objective=manifest["donor_context"]["objective"], provenance=manifest["donor_provenance"]))


def test_relocated_all_19_real_npz_load_and_import_without_original_paths(custody, tmp_path):
    root, kwargs = custody
    manifest = RR.build_manifest(root, **kwargs)
    original = tmp_path / "coordinator-only"
    spec = _binding(manifest, original)
    attempt = tmp_path / "attempt"
    staged = DT.stage(spec, attempt)
    relocated = tmp_path / "other-host/attempt"
    shutil.copytree(attempt, relocated)
    shutil.rmtree(original)
    shutil.rmtree(attempt)
    resolved = DT.resolve(staged, relocated)
    from tools.i6dmo import model as M
    context = manifest["donor_context"]
    config = D.effective_config(D.load_design(BASE / "matched_short/EXPERIMENT.mg2m.json"))["model"]
    config["features"] = context["features"]
    bundle = M.build_dmo(config)
    expected = M.branch_donor_context(bundle, population_sha256=context["population_sha256"],
        scaler_sha256=context["scaler_sha256"], objective=context["objective"],
        provenance=context["provenance"])
    before = M._non_branch_digest(bundle)
    record = M.import_branch_donors(bundle, resolved["r2_donors"]["donors"], expected)
    assert len(record["branches"]) == 19 and record["trainable"] is True
    assert M._non_branch_digest(bundle) == before
    for i, feature in enumerate(context["features"]):
        np.testing.assert_array_equal(bundle.branch_models[feature].get_weights()[0],
                                      np.full((3, 1, 4), i, np.float32))


@pytest.mark.parametrize("mutation", ["terminal", "weights", "parent_symlink"])
def test_relocated_donor_refuses_tampering_before_loader(custody, tmp_path, mutation):
    root, kwargs = custody
    manifest = RR.build_manifest(root, **kwargs)
    spec = _binding(manifest, tmp_path / "coordinator-only")
    attempt = tmp_path / "attempt"
    staged = DT.stage(spec, attempt)
    first = next(iter(staged["r2_staging"]["donors"].values()))
    terminal = attempt / first["path"]
    if mutation == "parent_symlink":
        parent = terminal.parent
        other = tmp_path / "external"
        shutil.move(parent, other)
        parent.symlink_to(other, target_is_directory=True)
    else:
        (terminal if mutation == "terminal" else terminal.parent / "donor.npz").write_bytes(b"tampered")
    with pytest.raises(DT.TransportError):
        DT.resolve(staged, attempt)


def _recovered_factory(custody, tmp_path, monkeypatch, target=None):
    source, kwargs = custody
    # Synthetic custody uses opaque features and an opaque objective. Only its
    # canonical contract authority is bypassed; recovery, adoption and queues are real.
    manifest = RR.build_manifest(source, **kwargs)
    design_path = BASE / "matched_short/EXPERIMENT.mg2m.json"
    design = D.load_design(design_path)
    adapter_effective = D.effective_config(design)
    adapter_effective["feature_manifest"]["features"] = manifest["donor_context"]["features"]
    contract = json.loads((BASE / "matched_short/R2_CONTRAST_DESIGN.ruled.json").read_text())
    contract["shared"].update(population_sha256=manifest["donor_context"]["population_sha256"])
    contract["pretraining"]["objective"] = manifest["donor_context"]["objective"]
    monkeypatch.setattr(RD, "validate_r2_design", lambda *a, **kw: None)
    adapter = SimpleNamespace(validate_design=C._adapter(None).validate_design,
        effective_config=lambda d: copy.deepcopy(adapter_effective))
    contract_path = tmp_path / "synthetic-contract.json"
    atomic_json(contract_path, contract)
    spec_path = tmp_path / "deployment.json"
    deployment = private_spec(tmp_path)
    deployment["roles"]["coordinator"]["experiment_storage_root"] = str(tmp_path / "experiment_storage")
    atomic_json(spec_path, deployment)
    path = tmp_path / "RECOVERY.json"
    atomic_json(path, manifest)
    proof = tmp_path / "REMOTE_CUSTODY.synthetic.json"
    files = {str(p.relative_to(kwargs["mirror_roots"][1])): file_sha256(p)
             for p in kwargs["mirror_roots"][1].rglob("*") if p.is_file()}
    atomic_json(proof, dict(schema="i6dmo.remote_custody_view.v1", source_pins=kwargs["source_pins"],
        role="synthetic-second-host", files=files,
        verification="REMOTE_SHA_BEFORE_AND_AFTER_COPY_EQUALS_LOCAL_BYTES"))
    recovery = dict(path=str(path), sha256=file_sha256(path), r0_target=kwargs["r0_target"],
        physical_custody=dict(path=str(proof), sha256=file_sha256(proof)))
    pins = dict(common_population=dict(base="experiment_storage", relpath="population", sha256="b" * 64,
        population_sha256=manifest["donor_context"]["population_sha256"]),
        matched_input=dict(relpath="input", sha256="c" * 64))
    target = target or tmp_path / "successor"
    C.init_r2_contrast(contract_path, design_path, spec_path, target, adapter=adapter,
        code_commit="d" * 40, data_pins=pins, donor_recovery=recovery,
        r0_reuse=dict(decision="FRESH_R0_REQUIRED"))
    effective, plan = C.load_effective(target), C.load_plan(target)
    assert effective["r2"]["strict_reports"] is True
    assert C.r2_provenance(effective, plan) == manifest["donor_provenance"]
    q = C.open_queue(target)
    try:
        units = q.units()
        donors = [u for u in units if u["kind"] == "pretrain"]
        assert len(donors) == 19 and all(u["state"] == "COMPLETE" for u in donors)
        for unit in donors:
            ref = manifest["pretraining_refs"][unit["spec"]["feature"]]
            assert unit["identity_sha256"] == ref["identity_sha256"]
            assert q.terminal(unit["unit_id"])["source"] == "ADOPTED"
        fit = next(u for u in plan["units"] if u["unit_id"] == "r2.fit")
        extra = W.r2_unit_spec(target, q, plan, effective, fit)
        assert len(extra["r2_donors"]["donors"]) == 19
        assert W.r2_unit_spec(target, q, plan, effective,
            next(u for u in plan["units"] if u["unit_id"] == "r0.fit")).get("r2_donors") is None
        assert not any(u["test_open"] for u in plan["units"])
    finally:
        q.close()
    assert len(manifest["donors"]) == 19
    prepared_path = tmp_path / "prepared.synthetic.json"
    atomic_json(prepared_path, dict(data_pins=pins))
    return dict(state="PREPARED", contract=str(contract_path), design=str(design_path),
        deployment=str(spec_path), prepared=str(prepared_path), recovery=recovery)


def test_recovery_factory_adopts_all_original_queue_identities(custody, tmp_path, monkeypatch):
    _recovered_factory(custody, tmp_path, monkeypatch)
