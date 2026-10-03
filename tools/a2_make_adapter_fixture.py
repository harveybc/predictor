#!/usr/bin/env python3
"""Produce the committed test fixture with the REAL adapter (feature-extractor 6fd7601), not a copy.

Writes tests/fixtures/a2_adapter_6fd7601/: train.npz (synthetic, split='train', string row_ids,
24 steps x 3 features), adapter/encoder.keras + adapter/manifest.json emitted by
``app.npz_encoder_adapter.train_encoder`` imported from ``--adapter-module``, and SOURCE.json binding
the adapter source sha256. Tiny by construction (40 rows, filters 8, latent 4, 1 epoch).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--adapter-module", required=True, help="path to feature-extractor app/npz_encoder_adapter.py @6fd7601")
p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests/fixtures/a2_adapter_6fd7601"))
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2021)
n, steps, ch = 40, 24, 3
windows = rng.normal(size=(n, steps, ch)).astype(np.float32)
row_ids = np.array([f"fx:row{i}" for i in range(n)])
np.savez(out / "train.npz", windows=windows, row_ids=row_ids, split=np.array("train"),
         feature_names=np.array(["a", "b", "c"]), dataset_id=np.array("fixture:a2:train"))
spec = importlib.util.spec_from_file_location("npz_encoder_adapter", a.adapter_module)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m = mod.train_encoder(str(out / "train.npz"), str(out / "adapter"), seed=0, latent_dim=4, filters=8,
                      epochs=1, batch_size=16)
src = Path(a.adapter_module).read_bytes()
(out / "SOURCE.json").write_text(json.dumps({
    "producer": "feature-extractor app/npz_encoder_adapter.py", "commit": "6fd7601",
    "adapter_source_sha256": hashlib.sha256(src).hexdigest(),
    "call": "train_encoder(train.npz, adapter, seed=0, latent_dim=4, filters=8, epochs=1, batch_size=16)",
    "manifest": m.__dict__}, indent=1, sort_keys=True) + "\n")
print(json.dumps(m.__dict__, sort_keys=True))
