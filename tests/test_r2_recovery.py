"""Offline failed-campaign custody recovery. Synthetic bytes, never fits or services."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import numpy as np
import pytest

from tools.i6dmo import campaign as C, donor_trust as DT, outbox as O
from tools.i6dmo import r2_recovery as RR
from tools.i6n_swarm.common import canonical


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True))
    return DT._sha(path)


def array_digest(value):
    h = hashlib.sha256(canonical([list(value.shape), str(value.dtype)]).encode())
    h.update(value.tobytes())
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=h.hexdigest())


@pytest.fixture
def custody(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    features = [f"feature.{i:02d}" for i in range(19)]
    runtime = dict(tensorflow="2.21.0", keras="3.13.2")
    model = dict(features=features, slots=13, market_time=dict(rule="MG2_MARKET_EXACT", spacing_open_market_hours=2),
                 branch=dict(filters=4, kernel_size=3, strides=1, activation="relu", padding="causal"))
    objective = dict(name="autoencoder_reconstruction", version="1.0.0", fit_settings=dict(
        monitor="validation_loss_on_purged_INNER_inputs"), identity_sha256="o" * 64)
    design = dict(seal_sha256="z" * 64, sealed_identities=dict(runtime=runtime))
    adapter = dict(effective_config_sha256="e" * 64, feature_manifest=dict(features=features), model=model)
    effective = C._seal(dict(schema="i6dmo.effective_config.v1", mode="R2_CONTRAST", code_commit="c" * 40,
        test_open=False, engineering_cpu_fixture=False, fit_device="gpu", design=dict(splits=dict(test=dict(open=False))),
        arms=dict(mg2m=dict(design=design, design_sha256="d" * 64, adapter_effective=adapter,validator={})),
        data_pins=dict(common_population=dict(population_sha256="p" * 64)),
        r2=dict(contract_seal_sha256="r" * 64, objective=objective, engineering_objective_overrides=None)))
    write(root / "EFFECTIVE_CONFIG.json", effective)
    (root / "backup").mkdir()
    q = sqlite3.connect(root / "backup/queue.sqlite3")
    q.executescript("CREATE TABLE terminals(unit_id TEXT, identity_sha256 TEXT, terminal TEXT, terminal_sha256 TEXT, source TEXT, attempt_id INTEGER);"
                   "CREATE TABLE attempts(attempt_id INTEGER, unit_id TEXT, outcome TEXT, idx INTEGER, kind TEXT, token INTEGER, attempt_dir TEXT, role TEXT);"
                   "CREATE TABLE units(unit_id TEXT, identity_sha256 TEXT, kind TEXT, spec TEXT, state TEXT);")
    box = sqlite3.connect(root / "backup/outbox.sqlite3")
    box.execute("CREATE TABLE items(report_sha256 TEXT, body TEXT, source_unit TEXT, state TEXT, readback_sha256 TEXT)")
    units, reports = [], []

    def stored(uid, kind, artifacts, extra, *, status="COMPLETE", index=None):
        identity = dict(unit=uid, kind=kind, family="short", cadence="NONE", seed=6102026,
            code_commit=effective["code_commit"], effective_config_sha256="e" * 64, data_sha256="f" * 64, upstream=[])
        digest = C.unit_identity(identity)
        unit = dict(unit_id=uid, kind=kind, family="short", arm="mg2m", identity=identity, identity_sha256=digest)
        if index is not None:
            unit.update(feature=features[index], feature_index=index)
        units.append(unit)
        folder = root / "terminals" / digest
        arts = {}
        for name, (rel, body) in artifacts.items():
            path = folder / "out" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(body, bytes):
                path.write_bytes(body)
            else:
                write(path, body)
            arts[name] = dict(path=rel, sha256=DT._sha(path))
        child_reports = []
        if status == "COMPLETE":
            report = dict(experiment_key=f"i6dmo:old:{uid}:cost", experiment_set_key="i6dmo:old", actor="predictor",
                lake="LOCAL_RETAINED", lineage="UNVERIFIED", config_sha256=digest, code_commit="c" * 40,
                project="predictor", phase="I6DMO", tags=dict(unit=uid, weights_sha256="w" * 64),
                metrics=[dict(metric="Fit_Seconds", value=1., unit="seconds", split="train", horizon=None)])
            sealed = O.seal_report(report)
            reports.append(sealed)
            child_reports = [report]
            box.execute("INSERT INTO items VALUES(?,?,?,?,?)", (sealed["report_sha256"], canonical(sealed), uid,
                "DELIVERED", O._rows_sha(O.expected_rows(sealed))))
        child = dict(schema="i6dmo.terminal.v1", unit_id=uid, identity_sha256=digest, status=status,
                     artifacts=arts, reports=child_reports, **extra)
        child_hash = write(folder / "CHILD_TERMINAL.json", child)
        attempt_id = len(units)
        write(folder / "TERMINAL.json", dict(schema="i6dmo.stored_terminal.v1", unit_id=uid,
            identity=identity, identity_sha256=digest, child_terminal_sha256=child_hash, terminal=child,
            attempt=dict(index=1,kind="INITIAL",token=attempt_id,role="coordinator")))
        q.execute("INSERT INTO terminals VALUES(?,?,?,?,?,?)", (uid, digest, canonical(child), child_hash, "LOCAL", attempt_id))
        attempt_dir=f"attempts/{uid}/a01"
        q.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)", (attempt_id, uid, status,1,"INITIAL",attempt_id,attempt_dir,"coordinator"))
        q.execute("INSERT INTO units VALUES(?,?,?,?,?)", (uid, digest, kind, canonical(unit), status))
        if status=="FAILED":
            write(root/attempt_dir/"UNIT_SPEC.json",dict(unit,campaign_id="old",test_open=False,
                effective_config=C.arm_view(effective,"mg2m"),effective_config_sha256=C.sha256_json(effective)))
        return folder

    scaler = "s" * 64
    for i, feature in enumerate(features):
        kernel, bias = np.full((3, 1, 4), i, np.float32), np.zeros(4, np.float32)
        b = io.BytesIO()
        np.savez(b, kernel=kernel, bias=bias)
        corpus = dict(feature_index=i, purge_rule="INNER selection windows whose earliest slot instant precedes the INNER start are purged",
            scaler=[float(i).hex(), float(1).hex()], fit_row_ids="a" * 64, fit_windows="b" * 64,
            selection_row_ids="d" * 64, selection_windows="e" * 64, counts=dict(fit=2, selection=1, purged=1))
        corpus["sha256"] = DT._canonical_sha(corpus)
        manifest = dict(schema="i6dmo.branch_donor.v1", arm="mg2m", rule="MG2_MARKET_EXACT", slots=13,
            spacing_open_market_hours=2, feature_name=feature, feature_index=i,
            feature_names_sha256=DT._canonical_sha(features), branch_model_name=f"branch_{i:02d}",
            conv_layer_name=f"branch_{i:02d}_conv", branch_params={k:model["branch"][k] for k in ("filters","kernel_size","strides","activation")},
            input_shape=[13,1], output_shape=[13,4], scaler_sha256=scaler, population_sha256="p" * 64,
            objective=objective, weights=[array_digest(kernel), array_digest(bias)])
        donor = dict(schema="i6dmo.r2_pretraining_terminal.v1", status="COMPLETE", campaign_id="old", unit_id=f"r2.pretrain.{i:02d}",
            feature=feature, feature_index=i, applied_seed=6102026+i, code_pin="c" * 40,
            ruling_seal_sha256="r" * 64,
            seed_rule="keras.utils.set_random_seed(6102026 + feature_index) immediately before building each autoencoder; applied seed recorded in each donor terminal",
            objective=objective, learned_corpus=corpus, manifest=manifest, fit=dict(stop_reason="patience"),
            artifacts={"donor.npz":hashlib.sha256(b.getvalue()).hexdigest()})
        stored(f"r2.pretrain.{i:02d}", "pretrain", dict(donor_terminal=("donor/TERMINAL.json", donor),
            **{"donor.npz":("donor/donor.npz",b.getvalue())}, corpus=("CORPUS.json",corpus)),
            dict(feature=feature, feature_index=i, applied_seed=6102026+i, learned_corpus_sha256=corpus["sha256"],
                 tf_imported_before_flags=False), index=i)

    code_root = tmp_path / "target-code"
    code_files = {}
    for name in ("tools/i6dmo/train.py", "tools/i6dmo/model.py"):
        code_files[name] = write(code_root / name, dict(original_training_code=name))
    counts = json.loads(DT.CAPACITY_DOC.read_text())["declared_capacity"]["short_mg2"]
    capacity = dict(arm="mg2m", counts=counts)
    capacity["capacity_sha256"] = DT._canonical_sha(capacity)
    identity = dict(design_sha256="d"*64, effective_config_sha256="e"*64, common_population_sha256="p"*64,
        matched_arm="mg2m", capacity_sha256=capacity["capacity_sha256"], scales_sha256="v"*64,
        code_files_sha256=code_files, weights_sha256="w"*64)
    model_bytes = io.BytesIO()
    with zipfile.ZipFile(model_bytes,"w") as z:
        z.writestr("metadata.json", json.dumps(dict(keras_version=runtime["keras"])))
    identity["model_archive_sha256"] = hashlib.sha256(model_bytes.getvalue()).hexdigest()
    monitor = dict(procedure="i6dmo.monitor_eval.v2", partition="sequential_row_id_fixed")
    receipt = dict(status="COMPLETED", scientific=True, overrides={}, device="gpu", identity=identity,
        resources=dict(tensorflow_version=runtime["tensorflow"]), matched=dict(arm="mg2m",capacity=capacity,
        common_population_sha256="p"*64,scaler=dict(fit_row_digest="j"*64)), artifacts={})
    art = {"model/forecast_model.keras":("train/model/forecast_model.keras",model_bytes.getvalue()),
           "model/MODEL.json":("train/model/MODEL.json",dict(weights_sha256="w"*64)),
           "TRAINING.json":("train/TRAINING.json",dict(monitor_evaluation=monitor))}
    for name, (_,body) in art.items():
        receipt["artifacts"][name] = hashlib.sha256(body).hexdigest() if isinstance(body,bytes) else DT._canonical_sha(body)
    # Use the same JSON byte encoding as stored() for receipt artifact hashes.
    for name, (_,body) in art.items():
        if not isinstance(body,bytes):
            receipt["artifacts"][name] = hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    stored("r0.fit","fit",dict(art,RECEIPT=("train/RECEIPT.json",receipt)), dict(trainer_identity=identity))
    stored("r2.fit","fit",{},dict(training=None,costs=dict(fit_seconds=None),disposition="R2_DONOR_TERMINAL_DIGEST"),status="FAILED")
    q.commit();q.close();box.commit();box.close()
    plan = dict(schema="i6dmo.plan.v1", campaign_id="old", mode="R2_CONTRAST", units=units,
        effective_config_file_sha256=DT._sha(root/"EFFECTIVE_CONFIG.json"),effective_config_sha256=C.sha256_json(effective),test_units_enumerated=False)
    write(root/"PLAN.json",plan)
    write(root/"REPORTS.json",reports)
    write(root/"PUBLICATION.json",dict(state="DELIVERED",expected=len(reports),delivered=len(reports),pending=0,mismatch=0))
    mirrors = [tmp_path/"mirror-one",tmp_path/"mirror-two"]
    for mirror in mirrors:
        shutil.copytree(root,mirror)
    source_pins = dict(effective_sha256=DT._sha(root/"EFFECTIVE_CONFIG.json"),plan_sha256=DT._sha(root/"PLAN.json"),code_commit="c"*40)
    target = dict(contract=dict(shared=dict(experiment_design_sha256="d"*64,effective_config_sha256="e"*64,
        population_sha256="p"*64,arm="mg2m",capacity_sha256=capacity["capacity_sha256"],scaler_fit_row_digest="j"*64,runtime=runtime)),
        expected_code_files=code_files,code_root=str(code_root),expected_scales_sha256="v"*64,current_monitor_identity=monitor)
    return root,dict(mirror_roots=mirrors,source_pins=source_pins,r0_target=target)


def test_recovers_original_identities_context_and_r0_read_only(custody):
    root,kwargs=custody
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.parent.rglob("*") if p.is_file()}
    manifest=RR.build_manifest(root,**kwargs)
    assert manifest["status"]=="ADOPTABLE"
    assert len(manifest["pretraining_refs"])==19
    assert manifest["donor_context"]["provenance"]["campaign_id"]=="old"
    assert manifest["donor_context"]["provenance"]["code_pin"]=="c"*40
    assert manifest["donor_provenance"] == manifest["donor_context"]["provenance"]
    for feature, ref in manifest["pretraining_refs"].items():
        assert manifest["unit_specs"][feature]["identity"] == ref["identity"]
    assert manifest["r0_fit"]["unit_id"]=="r0.fit"
    assert manifest["failed_r2"]["terminal"]["training"] is None
    assert RR.verify_manifest(manifest,source_root=root,**kwargs)==manifest
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.parent.rglob("*") if p.is_file()}
    assert not (root/"CLOSE.json").exists()


@pytest.mark.parametrize("relative",["REPORTS.json","backup/outbox.sqlite3","backup/queue.sqlite3"])
def test_mirror_custody_bytes_required_not_claimed_hash(custody,relative):
    root,kwargs=custody
    (kwargs["mirror_roots"][1]/relative).write_bytes(b"changed")
    with pytest.raises(RR.RecoveryError,match="CUSTODY"):
        RR.build_manifest(root,**kwargs)


def test_donor_npz_byte_tampering_refused(custody):
    root,kwargs=custody
    p=next((root/"terminals").glob("*/out/donor/donor.npz"))
    p.write_bytes(b"changed")
    with pytest.raises(RR.RecoveryError): RR.build_manifest(root,**kwargs)


def test_r0_code_incompatibility_never_schedules_fresh_fit(custody):
    root,kwargs=custody
    target=kwargs["r0_target"]
    rel="tools/i6dmo/train.py"
    target["expected_code_files"][rel]=write(Path(target["code_root"])/rel,dict(new_training_code=True))
    result=RR.build_manifest(root,**kwargs)
    assert result["status"]=="DONORS_ADOPTABLE_R0_INCOMPATIBLE"
    assert result["r0_fit"] is None and len(result["pretraining_refs"])==19
    assert "code_files_sha256" in result["r0_reuse"]["failed"]


def test_manifest_self_resealing_cannot_replace_custody(custody):
    root,kwargs=custody
    manifest=RR.build_manifest(root,**kwargs)
    forged=copy.deepcopy(manifest)
    forged["donor_context"]["provenance"]["campaign_id"]="new-campaign"
    forged["manifest_sha256"]=DT._canonical_sha({k:v for k,v in forged.items() if k!="manifest_sha256"})
    with pytest.raises(RR.RecoveryError,match="MANIFEST"):
        RR.verify_manifest(forged,source_root=root,**kwargs)


def test_wrong_source_pins_and_alias_mirrors_refused(custody):
    root,kwargs=custody
    bad=copy.deepcopy(kwargs);bad["source_pins"]["plan_sha256"]="0"*64
    with pytest.raises(RR.RecoveryError,match="SOURCE_PIN"): RR.build_manifest(root,**bad)
    bad=copy.deepcopy(kwargs);bad["mirror_roots"]=[root,root]
    with pytest.raises(RR.RecoveryError,match="MIRROR"): RR.build_manifest(root,**bad)


def test_readback_not_self_asserted_delivered_state(custody):
    root,kwargs=custody
    for base in [root,*kwargs["mirror_roots"]]:
        con=sqlite3.connect(base/"backup/outbox.sqlite3")
        con.execute("UPDATE items SET readback_sha256=?",("0"*64,));con.commit();con.close()
    with pytest.raises(RR.RecoveryError,match="PUBLICATION"): RR.build_manifest(root,**kwargs)


def change_donor_and_reseal_copies(custody, change):
    root, kwargs = custody
    unit = json.loads((root / "PLAN.json").read_text())["units"][0]
    folder = root / "terminals" / unit["identity_sha256"]
    path = folder / "out/donor/TERMINAL.json"
    donor = json.loads(path.read_text())
    change(donor)
    digest = write(path, donor)
    child = json.loads((folder / "CHILD_TERMINAL.json").read_text())
    child["artifacts"]["donor_terminal"]["sha256"] = digest
    child_hash = write(folder / "CHILD_TERMINAL.json", child)
    stored = json.loads((folder / "TERMINAL.json").read_text())
    stored.update(terminal=child, child_terminal_sha256=child_hash)
    write(folder / "TERMINAL.json", stored)
    con = sqlite3.connect(root / "backup/queue.sqlite3")
    con.execute("UPDATE terminals SET terminal=?,terminal_sha256=? WHERE unit_id=?",
                (canonical(child), child_hash, unit["unit_id"]))
    con.commit();con.close()
    for mirror in kwargs["mirror_roots"]:
        shutil.copytree(root, mirror, dirs_exist_ok=True)


@pytest.mark.parametrize("corruption", ["campaign", "feature_order", "objective", "population", "corpus", "weights", "budget"])
def test_semantic_donor_forgery_cannot_pass_resealed_artifacts(custody, corruption):
    def change(donor):
        if corruption == "campaign": donor["campaign_id"] = "successor"
        elif corruption == "feature_order": donor["manifest"]["feature_index"] = 1
        elif corruption == "objective": donor["objective"]["fit_settings"]["monitor"] = "new_mean_monitor"
        elif corruption == "population": donor["manifest"]["population_sha256"] = "x" * 64
        elif corruption == "corpus": donor["learned_corpus"]["fit_windows"] = "0" * 64
        elif corruption == "weights": donor["manifest"]["weights"][0]["sha256"] = "0" * 64
        elif corruption == "budget": donor["fit"]["stop_reason"] = "max_updates"
    change_donor_and_reseal_copies(custody, change)
    with pytest.raises(RR.RecoveryError, match="DONOR"):
        RR.build_manifest(custody[0], **custody[1])


def test_private_manifest_loader_requires_external_file_pin(custody, tmp_path):
    root, kwargs = custody
    manifest = RR.build_manifest(root, **kwargs)
    path = tmp_path / "RECOVERY.private.json"
    pin = write(path, manifest)
    assert RR.load_manifest(path, expected_sha256=pin, r0_target=kwargs["r0_target"]) == manifest
    with pytest.raises(RR.RecoveryError, match="MANIFEST_FILE_DIGEST"):
        RR.load_manifest(path, expected_sha256="0" * 64, r0_target=kwargs["r0_target"])


def test_fake_target_code_hashes_not_trusted(custody):
    root, kwargs = custody
    kwargs["r0_target"]["expected_code_files"]["tools/i6dmo/train.py"] = "0" * 64
    with pytest.raises(RR.RecoveryError, match="TARGET_CODE_BYTES"):
        RR.build_manifest(root, **kwargs)


def test_recovery_never_imports_tensorflow(custody, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kw):
        assert not name.startswith("tensorflow") and name != "tools.i6dmo.model", "recovery imported training runtime"
        return original(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert RR.build_manifest(custody[0], **custody[1])["training_repeated"] is False


def test_production_failed_attempt_without_queue_terminal(custody):
    root, kwargs = custody
    for base in [root,*kwargs["mirror_roots"]]:
        con=sqlite3.connect(base/"backup/queue.sqlite3")
        con.execute("DELETE FROM terminals WHERE unit_id='r2.fit'");con.commit();con.close()
    # Attempt files are not in the physical mirror payload in production.
    for mirror in kwargs["mirror_roots"]:
        shutil.rmtree(mirror/"attempts")
    result=RR.build_manifest(root,**kwargs)
    assert result["failed_r2"]["reference"]["authentication"]=="FAILED_PLAN_QUEUE_ATTEMPT_UNIT_SPEC"
    assert result["failed_r2"]["reference"]["unit_spec"]["sha256"]==DT._sha(root/"attempts/r2.fit/a01/UNIT_SPEC.json")
    assert len(result["pretraining_refs"])==19


def test_failed_attempt_unit_spec_identity_cannot_be_relabelled(custody):
    root, kwargs = custody
    path=root/"attempts/r2.fit/a01/UNIT_SPEC.json"
    spec=json.loads(path.read_text());spec["campaign_id"]="successor";write(path,spec)
    with pytest.raises(RR.RecoveryError,match="FAILED_UNIT_SPEC"):
        RR.build_manifest(root,**kwargs)


def test_plan_binds_complete_sealed_effective_not_inner_seal(custody):
    root, kwargs = custody
    effective = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
    plan = json.loads((root / "PLAN.json").read_text())
    assert plan["effective_config_sha256"] == C.sha256_json(effective)
    assert plan["effective_config_sha256"] != effective["seal_sha256"]
    assert RR.build_manifest(root, **kwargs)["status"] == "ADOPTABLE"
    # Even byte-identical custody copies cannot make the wrong digest semantic valid.
    plan["effective_config_sha256"] = effective["seal_sha256"]
    kwargs["source_pins"]["plan_sha256"] = write(root / "PLAN.json", plan)
    for mirror in kwargs["mirror_roots"]:
        write(mirror / "PLAN.json", plan)
    with pytest.raises(RR.RecoveryError, match="SOURCE_IDENTITY"):
        RR.build_manifest(root, **kwargs)
