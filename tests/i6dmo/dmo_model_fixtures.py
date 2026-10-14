"""Invented tiny CPU fixtures for the I6-DMO model lane (no real market data).

Prices are a synthetic random walk; feature values are synthetic noise with a
weak signal. Origins sit inside the declared fit / inner / validation spans so
the model-side boundary and purge checks pass; nothing touches 2025.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4, manifest_sha256

DESIGN = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json"
HORIZONS = [2, 3, 4, 5, 6, 12]
OFFSETS = list(range(-24, 1, 2))


def _epoch(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_arrays(start, rows, seed, *, invalid_rate=0.25, n_features=19):
    """One split: hourly origins from ``start``; NaN only where the mask is false."""
    rng = np.random.default_rng(seed)
    row_id = _epoch(start) + 3600 * np.arange(rows, dtype=np.int64)
    x = rng.normal(size=(rows, len(OFFSETS), n_features)) * (1.0 + np.arange(n_features))
    origin_price = 1.10 + np.cumsum(rng.normal(scale=1e-3, size=rows))
    signal = 2e-4 * x[:, -1, :1]
    target = origin_price[:, None] + signal * np.asarray(HORIZONS)[None, :] \
        + rng.normal(scale=5e-4, size=(rows, len(HORIZONS))) * np.sqrt(HORIZONS)[None, :]
    mask = rng.random((rows, len(HORIZONS))) >= invalid_rate
    mask[:, 0] = True  # every row keeps at least one valid head
    target = np.where(mask, target, np.nan)
    return dict(row_id=row_id, x=x.astype(np.float64), origin_price=origin_price,
                target_price=target, mask=mask)


def write_bundle(root, *, features=None, splits=None, extra_split=None, seed=0,
                 manifest_sha=None):
    """Write BUNDLE.json + one npz per split under ``root`` and return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    features = list(MANIFEST_NO_STLFSI4) if features is None else list(features)
    splits = splits or {
        "fit": split_arrays("2020-01-06T00:00:00", 96, seed + 1, n_features=len(features)),
        "inner": split_arrays("2023-11-13T00:00:00", 32, seed + 2, n_features=len(features)),
        "validation": split_arrays("2024-01-08T00:00:00", 40, seed + 3, n_features=len(features)),
    }
    if extra_split:
        splits = {**splits, **extra_split}
    entries = {}
    for name, arrays in splits.items():
        path = root / f"{name}.npz"
        np.savez(path, **arrays)
        entries[name] = {"path": path.name, "sha256": _sha(path)}
    manifest = {
        "schema": "i6dmo.data_bundle.v1",
        "features": features,
        "feature_manifest_sha256": manifest_sha or manifest_sha256(features),
        "horizons_hours": HORIZONS,
        "offsets_hours": OFFSETS,
        "splits": entries,
    }
    (root / "BUNDLE.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return root


def load_design_doc():
    return json.loads(DESIGN.read_text(encoding="utf-8"))
