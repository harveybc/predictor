"""Item 6 (b327b771): provenance fields on donors and bundles, with a VERSIONED migration.

Every new donor sidecar (schema 2) and bundle (bundle.v2) carries a provenance block: the
conditioning contract (OPERATIONAL / SYNTHETIC_OFFLINE), the learned-corpus provenance and the
reconstruction state. What the producer did not declare is written as UNKNOWN, explicitly; a
historical artifact migrated from schema 1 / bundle.v1 is UNKNOWN, never a silently invented
safe default. Consumers that need OPERATIONAL weights ask for it and UNKNOWN is refused.
Synthetic component checks.
"""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from predictor_plugins.modular_temporal import provenance as pv


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(8)
    yield


def bundle():
    return mt.build_modular(mt.default_config(["a", "b"]))


DECLARED = {"conditioning_contract": "OPERATIONAL",
            "learned_corpus": {"kind": "TRAIN_ONLY", "dataset_id": "data-gov:example",
                               "data_sha256": "a" * 64, "support": "rows [0, 7474) of TRAIN",
                               "pretrained_weights_source": None},
            "reconstruction": {"state": "NOT_APPLICABLE"}}


def test_new_donor_sidecar_is_schema_2_with_explicit_unknowns_when_undeclared(tmp_path):
    b = bundle()
    doc = mt.save_donor(b.branch_models["branch_0"], tmp_path / "d.keras", b.donor_manifest("branch", "branch_0"))
    assert doc["schema"] == 2 and doc["provenance"]["provenance_schema"] == pv.PROVENANCE_SCHEMA
    p = doc["provenance"]
    assert p["conditioning_contract"] == "UNKNOWN"
    assert p["learned_corpus"] == {"kind": "UNKNOWN"} and p["reconstruction"] == {"state": "UNKNOWN"}
    assert pv.donor_provenance(tmp_path / "d.keras")["conditioning_contract"] == "UNKNOWN"


def test_declared_provenance_is_validated_before_anything_is_written(tmp_path):
    b = bundle()
    enc, manifest = b.branch_models["branch_0"], b.donor_manifest("branch", "branch_0")
    doc = mt.save_donor(enc, tmp_path / "ok.keras", manifest, provenance=DECLARED)
    assert doc["provenance"]["learned_corpus"]["kind"] == "TRAIN_ONLY"
    assert doc["provenance"]["reconstruction"] == {"state": "NOT_APPLICABLE"}
    bad = json.loads(json.dumps(DECLARED))
    bad["learned_corpus"]["pretrained_weights_source"] = "org/foreign-model"
    for broken, code in ((bad, "FOREIGN_CORPUS_WEIGHTS_NOT_TRAIN_ONLY"),
                         (dict(DECLARED, conditioning_contract="SAFE"), "CONDITIONING_CONTRACT_INVALID"),
                         (dict(DECLARED, reconstruction={"state": "MEASURED"}), "RECONSTRUCTION_MEASURE_MISSING"),
                         (dict(DECLARED, learned_corpus={"kind": "TRAIN_ONLY"}), "TRAIN_CORPUS_IDENTITY_MISSING")):
        with pytest.raises(ValueError, match=code):
            mt.save_donor(enc, tmp_path / "bad.keras", manifest, provenance=broken)
        assert not (tmp_path / "bad.keras").exists()


def test_schema_1_sidecar_loads_reports_unknown_and_migrates_versioned(tmp_path):
    b = bundle()
    path, manifest = tmp_path / "old.keras", b.donor_manifest("branch", "branch_0")
    doc = mt.save_donor(b.branch_models["branch_0"], path, manifest)
    legacy = {k: doc[k] for k in ("manifest", "manifest_sha256", "model_sha256", "weights_sha256")}
    legacy["schema"] = 1                                           # what earlier engines wrote
    sidecar = path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(legacy))
    mt.load_donor(path, manifest)                                   # still loads
    seen = pv.donor_provenance(path)
    assert seen["conditioning_contract"] == "UNKNOWN" and seen["migrated"] is False
    result = pv.migrate_donor_sidecar(path)
    assert result["status"] == "MIGRATED" and result["from_schema"] == 1 and result["to_schema"] == 2
    migrated = json.loads(sidecar.read_text())
    assert migrated["schema"] == 2 and migrated["provenance"]["learned_corpus"] == {"kind": "UNKNOWN"}
    assert migrated["provenance"]["migration"]["from_schema"] == 1
    assert {k: migrated[k] for k in legacy if k != "schema"} == {k: legacy[k] for k in legacy if k != "schema"}
    assert json.loads(path.with_suffix(".manifest.schema1.json").read_text()) == legacy
    mt.load_donor(path, manifest)
    assert pv.migrate_donor_sidecar(path)["status"] == "ALREADY_CURRENT"


def test_operational_consumers_refuse_unknown_and_synthetic_offline(tmp_path):
    b = bundle()
    enc, manifest = b.branch_models["branch_0"], b.donor_manifest("branch", "branch_0")
    mt.save_donor(enc, tmp_path / "unknown.keras", manifest)
    mt.save_donor(enc, tmp_path / "offline.keras", manifest,
                  provenance=dict(DECLARED, conditioning_contract="SYNTHETIC_OFFLINE"))
    mt.save_donor(enc, tmp_path / "op.keras", manifest, provenance=DECLARED)
    for name in ("unknown", "offline"):
        with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
            mt.load_donor(tmp_path / f"{name}.keras", manifest, require_contract="OPERATIONAL")
    mt.load_donor(tmp_path / "op.keras", manifest, require_contract="OPERATIONAL")


def test_bundle_v2_carries_provenance_and_v1_migrates_to_unknown(tmp_path):
    b = bundle()
    doc = mt.save_bundle(b, tmp_path / "new", provenance={"conditioning_contract": "OPERATIONAL"})
    assert doc["schema"] == "predictor.modular.bundle.v2"
    assert doc["provenance"]["conditioning_contract"] == "OPERATIONAL"
    assert doc["provenance"]["learned_corpus"] == {"kind": "UNKNOWN"}
    mt.load_bundle(tmp_path / "new")
    old = mt.save_bundle(b, tmp_path / "old")
    doc_path = tmp_path / "old" / "bundle.json"
    legacy = json.loads(doc_path.read_text())
    legacy["schema"] = "predictor.modular.bundle.v1"
    legacy.pop("provenance")
    doc_path.write_text(json.dumps(legacy))
    _, seen = mt.load_bundle(tmp_path / "old")
    assert seen["provenance"]["conditioning_contract"] == "UNKNOWN" and seen["provenance"]["migrated"] is False
    result = pv.migrate_bundle(tmp_path / "old")
    assert result["status"] == "MIGRATED"
    assert json.loads(doc_path.read_text())["schema"] == "predictor.modular.bundle.v2"
    assert json.loads(doc_path.read_text())["archive_sha256"] == old["archive_sha256"]
    mt.load_bundle(tmp_path / "old")
    assert pv.migrate_bundle(tmp_path / "old")["status"] == "ALREADY_CURRENT"


def test_legacy_schema_1_and_bundle_v1_load_as_unknown_and_operational_requirement_refuses_them(tmp_path):
    b = bundle()
    path, manifest = tmp_path / "old.keras", b.donor_manifest("branch", "branch_0")
    doc = mt.save_donor(b.branch_models["branch_0"], path, manifest, provenance=DECLARED)
    legacy = {k: doc[k] for k in ("manifest", "manifest_sha256", "model_sha256", "weights_sha256")}
    legacy["schema"] = 1
    path.with_suffix(".manifest.json").write_text(json.dumps(legacy))
    mt.load_donor(path, manifest)
    assert pv.donor_provenance(path)["conditioning_contract"] == "UNKNOWN"
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.load_donor(path, manifest, require_contract="OPERATIONAL")
    mt.save_bundle(b, tmp_path / "v1")
    doc_path = tmp_path / "v1" / "bundle.json"
    old = json.loads(doc_path.read_text())
    old["schema"] = "predictor.modular.bundle.v1"
    old.pop("provenance")
    doc_path.write_text(json.dumps(old))
    _, seen = mt.load_bundle(tmp_path / "v1")
    assert seen["provenance"]["conditioning_contract"] == "UNKNOWN"
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.load_bundle(tmp_path / "v1", require_contract="OPERATIONAL")


def test_facade_save_declares_operational_provenance_from_its_config(tmp_path):
    import numpy as np
    from predictor_plugins.predictor_plugin_modular import Plugin
    model = mt.default_config(["a", "b"])
    plugin = Plugin({"modular": model, "modular_provenance": DECLARED,
                     "modular_training": {"max_epochs": 1, "patience": 1, "batch_size": 8, "seed": 1}})
    x = np.random.default_rng(0).normal(size=(16, 24, 2)).astype("float32")
    plugin.build_model((24, 2), x, {})
    plugin.train(x, {"output_horizon_1": x[:, -1, :1]}, x_val=x[:8], y_val={"output_horizon_1": x[:8, -1, :1]})
    plugin.save(str(tmp_path / "m.keras"))
    _, doc = mt.load_bundle(tmp_path / "m.keras.bundle", require_contract="OPERATIONAL")
    assert doc["provenance"]["learned_corpus"]["kind"] == "TRAIN_ONLY"
    undeclared = Plugin({"modular": model})
    undeclared.build_model((24, 2), x, {})
    undeclared.save(str(tmp_path / "u.keras"))
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.load_bundle(tmp_path / "u.keras.bundle", require_contract="OPERATIONAL")
