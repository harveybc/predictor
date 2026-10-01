"""RL export spec v0 gaps closed in tools/export_modular_encoder_npz.py (lane G receipt review).

1. A saved bundle is exported only if its provenance is OPERATIONAL (load_bundle(require_contract));
   an explicit --allow-unknown-provenance bypass is recorded in the receipt.
2. The export carries the engine identity: config digest, provenance, regimes and per-component donor
   manifests' digests, so the torch side can bind what it imported.
3. Encoder options the torch consumer does not implement (input_normalization) are refused at export;
   head-only options (target_residual) do not change the encoder and are allowed.
Synthetic component checks.
"""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from tools import export_modular_encoder_npz as ex

OPERATIONAL = {"conditioning_contract": "OPERATIONAL",
               "learned_corpus": {"kind": "TRAIN_ONLY", "dataset_id": "fixture:train", "data_sha256": "f" * 64,
                                  "support": "fixture rows", "pretrained_weights_source": None},
               "reconstruction": {"state": "NOT_EVALUATED"}}


def small():
    c = mt.default_config(["a", "b"])
    c["core"]["params"] = {"d_model": 16, "heads": 2, "blocks": 1, "ff_dim": 32, "stage_channels": [12, 10, 8]}
    return c


def test_config_export_carries_engine_identity(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(small()))
    assert ex.main(["--config", str(cfg), "--out", str(tmp_path / "e.npz")]) == 0
    r = json.loads((tmp_path / "e.npz.receipt.json").read_text())
    assert r["identity"]["config_sha256"] == mt.config_digest(small())
    assert r["identity"]["regimes"]["common"] == "R0"
    assert r["source"]["label"] == "RANDOM_INIT_NOT_PRETRAINED"
    with np.load(tmp_path / "e.npz") as z:
        meta = json.loads(str(z["__meta__"]))
    assert meta["identity"] == r["identity"]


def test_bundle_export_requires_operational_provenance(tmp_path):
    b = mt.build_modular(small())
    mt.save_bundle(b, tmp_path / "unknown")
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        ex.main(["--bundle", str(tmp_path / "unknown"), "--out", str(tmp_path / "u.npz")])
    assert ex.main(["--bundle", str(tmp_path / "unknown"), "--out", str(tmp_path / "u.npz"),
                    "--allow-unknown-provenance"]) == 0
    r = json.loads((tmp_path / "u.npz.receipt.json").read_text())
    assert r["identity"]["provenance_bypass"] == "UNKNOWN_ALLOWED"
    mt.save_bundle(b, tmp_path / "op", provenance=OPERATIONAL)
    assert ex.main(["--bundle", str(tmp_path / "op"), "--out", str(tmp_path / "o.npz")]) == 0
    r = json.loads((tmp_path / "o.npz.receipt.json").read_text())
    assert r["identity"]["provenance"]["conditioning_contract"] == "OPERATIONAL"
    assert r["identity"]["provenance_bypass"] is None


def test_encoder_options_the_torch_consumer_lacks_are_refused(tmp_path):
    c = small()
    c["input_normalization"] = {"kind": "window_mean", "length": 24, "target_features": ["a"]}
    cfg = tmp_path / "n.json"
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="NOT_SUPPORTED_BY_TORCH_CONSUMER"):
        ex.main(["--config", str(cfg), "--out", str(tmp_path / "n.npz")])
    c = small()
    c["target_residual"] = {"kind": "seasonal_naive", "period": 24, "target_features": ["a"]}
    cfg.write_text(json.dumps(c))
    assert ex.main(["--config", str(cfg), "--out", str(tmp_path / "r.npz")]) == 0   # head-only option
