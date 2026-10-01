"""Re-manifesting donors written by the pre-c1e035d7 engine (556c5f3e).

The old donors are produced by the OLD engine source (pinned verbatim with its
sha256 under fixtures/engine_556c5f3e) in a separate interpreter, exactly as
M02's running pretraining writes them: explicit params from the nested config.
Synthetic component checks only.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "engine_556c5f3e" / "modular_temporal.py"
FIXTURE_SHA256 = "2daa0d3db2b5b3b301cc43e7638f58855bcb7bc841e3c6c03264b5b4cfbb9c9c"
EXPLICIT_CORE = {"d_model": 16, "heads": 2, "blocks": 1, "ff_dim": 16, "dropout": 0.0, "kernel_size": 3,
                 "stage_channels": [12, 10, 8], "time_factors": [2, 1, 1]}
OLD_CONFIG = {"window": 24, "sample_hours": 1, "feature_names": ["a", "b", "c"],
              "branches": [{"name": "x", "features": ["a"], "params": {"channels": 4, "kernel_size": 3}},
                           {"name": "y", "features": ["b", "c"], "params": {"channels": 3, "kernel_size": 2}}],
              "core": {"params": EXPLICIT_CORE}}

CHILD = r"""
import json, sys, numpy as np, tensorflow as tf
from predictor_plugins import modular_temporal as old
assert old.__file__.startswith(sys.argv[1]), old.__file__
tf.keras.utils.set_random_seed(5)
config = json.loads(sys.argv[2]); out = sys.argv[3]
b = old.build_modular(config)
x = np.random.default_rng(9).normal(size=(4, 24, 3)).astype("float32")
for name, model in b.branch_models.items():   # one pretraining-like step so weights are not fresh
    ae = old.build_autoencoder(model); ae.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
    cols = [config["feature_names"].index(f) for s in config["branches"] if s["name"] == name for f in s["features"]]
    ae.train_on_batch(x[:, :, cols], x[:, :, cols])
    old.save_donor(model, f"{out}/branch_{name}.keras", b.donor_manifest("branch", name))
old.save_donor(b.core_model, f"{out}/core.keras", b.donor_manifest("core"))
np.save(f"{out}/x.npy", x); np.save(f"{out}/encoded.npy", np.asarray(b.encoder_model(x)))
"""


@pytest.fixture(scope="module")
def old_donors(tmp_path_factory):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    root = tmp_path_factory.mktemp("old_engine")
    pkg = root / "pkg" / "predictor_plugins"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "modular_temporal.py").write_bytes(FIXTURE.read_bytes())
    out = root / "donors"
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONPATH=str(root / "pkg"), CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3",
               TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, "-s", "-c", CHILD, str(root / "pkg"), json.dumps(OLD_CONFIG), str(out)],
                   check=True, env=env)
    return out


def _copy_donors(src, dst):
    dst.mkdir()
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


def _r1(config, d):
    c = copy.deepcopy(config)
    for spec in c["branches"]:
        spec.update(regime="R1", donor=str(d / f"branch_{spec['name']}.keras"))
    c["core"].update(regime="R1", donor=str(d / "core.keras"))
    return c


def _donors(d):
    return [d / "branch_x.keras", d / "branch_y.keras", d / "core.keras"]


def test_old_donor_is_accepted_after_remanifest_with_identical_outputs(old_donors, tmp_path):
    from predictor_plugins import modular_temporal as mt
    from tools.modular_donor_remanifest import remanifest
    d = _copy_donors(old_donors, tmp_path / "d")
    with pytest.raises(ValueError, match="onor"):
        mt.build_modular(_r1(OLD_CONFIG, d))                  # refused under the new identity
    originals = {p: p.read_bytes() for p in _donors(d)}
    sidecars = {p: p.with_suffix(".manifest.json").read_bytes() for p in _donors(d)}
    results = [remanifest(p, feature_names=["a", "b", "c"], window=24, sample_hours=1) for p in _donors(d)]
    assert [r["status"] for r in results] == ["REMANIFESTED"] * 3
    for p in _donors(d):
        assert p.read_bytes() == originals[p]                  # weights file untouched
        assert Path(str(p.with_suffix("")) + ".manifest.pre_c1e035d7.json").read_bytes() == sidecars[p]
        log = json.loads(p.with_suffix(".provenance.json").read_text())["remanifest_records"]
        assert len(log) == 1 and log[0]["old_manifest_sha256"] != log[0]["new_manifest_sha256"]
        side = json.loads(p.with_suffix(".manifest.json").read_text())
        assert side["provenance"]["keras_version"] == mt.keras_version()
    b = mt.build_modular(_r1(OLD_CONFIG, d))
    x = np.load(d / "x.npy")
    np.testing.assert_allclose(b.encoder_model(x), np.load(d / "encoded.npy"), atol=1e-6)
    # the effective identity also accepts the same model written with implicit defaults
    implicit = copy.deepcopy(OLD_CONFIG)
    implicit["core"]["params"] = {k: v for k, v in EXPLICIT_CORE.items()
                                  if k not in ("dropout", "kernel_size", "time_factors")}
    mt.build_modular(_r1(implicit, d))


def test_remanifest_is_idempotent(old_donors, tmp_path):
    from tools.modular_donor_remanifest import remanifest
    d = _copy_donors(old_donors, tmp_path / "d")
    for p in _donors(d):
        remanifest(p)
    snapshot = {f: f.read_bytes() for f in d.iterdir()}
    again = [remanifest(p)["status"] for p in _donors(d)]
    assert again == ["ALREADY_CURRENT"] * 3
    assert {f: f.read_bytes() for f in d.iterdir()} == snapshot


def test_tampered_weights_file_is_refused_and_nothing_written(old_donors, tmp_path):
    from tools.modular_donor_remanifest import RemanifestRefusal, remanifest
    d = _copy_donors(old_donors, tmp_path / "d")
    target = d / "branch_y.keras"
    with target.open("ab") as f:
        f.write(b"tamper")
    before = target.with_suffix(".manifest.json").read_bytes()
    with pytest.raises(RemanifestRefusal, match="model_sha256"):
        remanifest(target)
    assert target.with_suffix(".manifest.json").read_bytes() == before
    assert not Path(str(target.with_suffix("")) + ".manifest.pre_c1e035d7.json").exists()
    assert not target.with_suffix(".provenance.json").exists()


@pytest.mark.parametrize("expect", [{"feature_names": ["b", "a", "c"]}, {"window": 48}, {"sample_hours": 2}])
def test_donor_from_a_different_config_is_refused(old_donors, tmp_path, expect):
    from tools.modular_donor_remanifest import RemanifestRefusal, remanifest
    d = _copy_donors(old_donors, tmp_path / "d")
    for p in _donors(d):
        with pytest.raises(RemanifestRefusal):
            remanifest(p, **expect)
        assert not p.with_suffix(".provenance.json").exists()


def test_remanifested_donor_still_rejects_another_config(old_donors, tmp_path):
    from predictor_plugins import modular_temporal as mt
    from tools.modular_donor_remanifest import remanifest
    d = _copy_donors(old_donors, tmp_path / "d")
    for p in _donors(d):
        remanifest(p)
    for mutate in (lambda c: c["branches"][0]["params"].update(channels=5),
                   lambda c: c.update(feature_names=["a", "c", "b"]),
                   lambda c: c["core"]["params"].update(blocks=2)):
        other = copy.deepcopy(OLD_CONFIG)
        mutate(other)
        with pytest.raises(ValueError, match="onor"):
            mt.build_modular(_r1(other, d))


def test_external_or_inconsistent_old_manifest_is_refused(old_donors, tmp_path):
    from predictor_plugins import modular_temporal as mt
    from tools.modular_donor_remanifest import RemanifestRefusal, remanifest
    d = _copy_donors(old_donors, tmp_path / "d")
    side = d / "branch_x.manifest.json"
    doc = json.loads(side.read_text())
    doc["manifest"]["plugin"]["implementation"] = "somebody_else:causal_conv1d"
    doc["manifest_sha256"] = mt._digest(doc["manifest"])
    side.write_text(json.dumps(doc))
    with pytest.raises(RemanifestRefusal, match="re-derivable"):
        remanifest(d / "branch_x.keras")
    doc["manifest"]["plugin"]["implementation"] = "modular_temporal.v1:causal_conv1d"
    doc["manifest"]["input_grid"][3] = 99
    doc["manifest_sha256"] = mt._digest(doc["manifest"])
    side.write_text(json.dumps(doc))
    with pytest.raises(RemanifestRefusal, match="grid"):
        remanifest(d / "branch_x.keras")
