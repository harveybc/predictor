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
LONG_HORIZONS = [18, 24, 36, 48, 60, 72]
LONG_OFFSETS = list(range(-144, 1, 12))


def _epoch(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_arrays(start, rows, seed, *, invalid_rate=0.25, n_features=19, horizons=None, steps=13):
    """One split: hourly origins from ``start``; NaN only where the mask is false."""
    horizons = HORIZONS if horizons is None else horizons
    rng = np.random.default_rng(seed)
    row_id = _epoch(start) + 3600 * np.arange(rows, dtype=np.int64)
    x = rng.normal(size=(rows, steps, n_features)) * (1.0 + np.arange(n_features))
    origin_price = 1.10 + np.cumsum(rng.normal(scale=1e-3, size=rows))
    signal = 2e-4 * x[:, -1, :1]
    target = origin_price[:, None] + signal * np.asarray(horizons)[None, :] \
        + rng.normal(scale=5e-4, size=(rows, len(horizons))) * np.sqrt(horizons)[None, :]
    mask = rng.random((rows, len(horizons))) >= invalid_rate
    mask[:, 0] = True  # every row keeps at least one valid head
    target = np.where(mask, target, np.nan)
    return dict(row_id=row_id, x=x.astype(np.float64), origin_price=origin_price,
                target_price=target, mask=mask)


def write_bundle(root, *, features=None, splits=None, extra_split=None, seed=0,
                 manifest_sha=None, horizons=None, offsets=None):
    """Write BUNDLE.json + one npz per split under ``root`` and return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    features = list(MANIFEST_NO_STLFSI4) if features is None else list(features)
    horizons = HORIZONS if horizons is None else horizons
    offsets = OFFSETS if offsets is None else offsets
    kw = dict(n_features=len(features), horizons=horizons, steps=len(offsets))
    splits = splits or {
        "fit": split_arrays("2020-01-06T00:00:00", 96, seed + 1, **kw),
        "inner": split_arrays("2023-11-13T00:00:00", 32, seed + 2, **kw),
        "validation": split_arrays("2024-01-08T00:00:00", 40, seed + 3, **kw),
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
        "horizons_hours": list(horizons),
        "offsets_hours": list(offsets),
        "splits": entries,
    }
    (root / "BUNDLE.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return root


def load_design_doc():
    return json.loads(DESIGN.read_text(encoding="utf-8"))


def short_terminal(path, *, status="COMPLETE", unit_id="short.report", schema="i6dmo.terminal.v1",
                   eligibility=None):
    """An invented ops-style terminal; eligibility/metrics are present but must never be read."""
    doc = {"schema": schema, "unit_id": unit_id, "identity_sha256": "ab" * 32, "status": status,
           "disposition": status, "artifacts": {}, "costs": {"wall_seconds": 1.0},
           "eligibility": eligibility if eligibility is not None else {"2": {"mae": 1.0, "naive": 2.0,
                                                                          "eligible": True, "n": 9}},
           "gates": {}, "reports": []}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")
    return Path(path)
