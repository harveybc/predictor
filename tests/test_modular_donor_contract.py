"""The engine itself enforces the donor conditioning contract on R1/R2 (M04 finding on 5d85933f).

``build_modular`` loads every R1/R2 donor with ``require_contract`` taken from the config key
``donor_contract``. Absent, it is OPERATIONAL: a schema-1-only donor (UNKNOWN) and a SYNTHETIC_OFFLINE
donor are refused by name; an ALONGSIDE schema-2 OPERATIONAL donor is accepted. Bypassing needs the
explicit value UNKNOWN_ALLOWED, which then sits in the normalized config and so in bundle.json.
The key is optional and not defaulted into the config, so existing config digests do not move.
Synthetic component checks.
"""
import copy
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import tensorflow as tf

from predictor_plugins import modular_config as mc
from predictor_plugins import modular_temporal as mt
from predictor_plugins.modular_temporal import provenance as pv

OPERATIONAL = {"conditioning_contract": "OPERATIONAL",
               "learned_corpus": {"kind": "TRAIN_ONLY", "dataset_id": "fixture:train", "data_sha256": "c" * 64,
                                  "support": "fixture TRAIN rows", "pretrained_weights_source": None},
               "reconstruction": {"state": "NOT_EVALUATED"}}


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(12)
    yield


def base():
    return mt.default_config(["a", "b"])


def r1_with(tmp_path, provenance=None, legacy=False, alongside=None):
    """A branch-0 R1 config whose donor carries the given provenance (or is schema-1 only)."""
    c = base()
    b = mt.build_modular(c)
    path, manifest = tmp_path / "d.keras", b.donor_manifest("branch", "branch_0")
    doc = mt.save_donor(b.branch_models["branch_0"], path, manifest, provenance=provenance)
    if legacy:
        old = {k: doc[k] for k in ("manifest", "manifest_sha256", "model_sha256", "weights_sha256")}
        old.update(schema=1, provenance={"keras_version": doc["provenance"]["keras_version"]})
        path.with_suffix(".manifest.json").write_text(json.dumps(old))
        if alongside is not None:
            pv.write_alongside(path, alongside, {"rule": "fixture", "sources": [{"file": "x", "sha256": "d" * 64}]})
    c["branches"][0].update(regime="R1", donor=str(path))
    return c


def test_schema_1_only_donor_is_refused_by_default(tmp_path):
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.build_modular(r1_with(tmp_path, legacy=True))


def test_alongside_v2_operational_donor_is_accepted(tmp_path):
    bundle = mt.build_modular(r1_with(tmp_path, legacy=True, alongside=OPERATIONAL))
    assert not bundle.branch_models["branch_0"].trainable


def test_unknown_and_synthetic_offline_schema_2_donors_are_refused(tmp_path):
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.build_modular(r1_with(tmp_path))                          # undeclared -> UNKNOWN
    offline = dict(OPERATIONAL, conditioning_contract="SYNTHETIC_OFFLINE")
    (tmp_path / "o").mkdir()
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.build_modular(r1_with(tmp_path / "o", provenance=offline))
    (tmp_path / "p").mkdir()
    mt.build_modular(r1_with(tmp_path / "p", provenance=OPERATIONAL))


def test_bypass_needs_explicit_unknown_allowed_and_is_recorded_in_the_bundle(tmp_path):
    c = r1_with(tmp_path, legacy=True)
    c["donor_contract"] = "UNKNOWN_ALLOWED"
    bundle = mt.build_modular(c)
    doc = mt.save_bundle(bundle, tmp_path / "bundle")
    assert doc["config"]["donor_contract"] == "UNKNOWN_ALLOWED"
    assert json.loads((tmp_path / "bundle" / "bundle.json").read_text())["config"]["donor_contract"] \
        == "UNKNOWN_ALLOWED"
    for bad in ("unknown_allowed", "ANY", "SYNTHETIC_OFFLINE", None, 1):
        with pytest.raises(ValueError, match="donor_contract"):
            mt.build_modular(dict(copy.deepcopy(c), donor_contract=bad))


def test_key_is_optional_digest_neutral_and_reversible(tmp_path):
    plain = base()
    before = mt.config_digest(plain)
    assert "donor_contract" not in mt._normalize(plain) and mt.config_digest(plain) == before
    explicit = dict(copy.deepcopy(plain), donor_contract="OPERATIONAL")
    assert mt.config_digest(explicit) != before                       # declared, so part of identity
    assert mc.unflatten(mc.flatten(explicit)) == mt._normalize(explicit)
    mt.build_modular(plain)                                          # R0 configs never load donors
