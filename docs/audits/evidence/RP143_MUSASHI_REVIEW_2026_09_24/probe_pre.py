#!/usr/bin/env python3
"""Read-only, bounded preservation of the two RP143 review CPU probes.

Only NumPy and the standard library are imported. Output is JSON on stdout;
this script never writes artifacts or accesses campaign roots/services.
"""

import os
import resource
import signal
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
resource.setrlimit(resource.RLIMIT_CPU, (10, 15))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
signal.alarm(20)

import hashlib
import importlib.util
import json
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REVIEW_COMMIT = "07140d03a265d235584235216dad3346be569853"
RETAINED = (
    "docs/audits/evidence/d3_k5_20260917/RP140/"
    "EVIDENCE_COMPOSITION.1790211629.json"
)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    R = load("df_sota_repro")
    M = load("df_ecl_modular")
    ident = {
        "checkpoint_sha256": "a" * 64,
        "predictions_sha256_recorded": "b" * 64,
        "record_sha256": "c" * 64,
        "pred_body_sha256_recorded": "d" * 64,
        "author_metric_float32": {"mae": 0.25, "mse": 0.125},
    }
    replay = {
        "unit": "u", "replayed_author_metric": ident["author_metric_float32"],
        "shape": [1, 1, 1], "elements": 1, "allclose_rule": True,
    }
    observations = {
        "wrong_shape_metric_binding": R._bind_replay_record(replay, ident),
        "unverifiable_digest_binding": R._bind_replay_record(
            {"identity": {"checkpoint_sha256": "foreign"}}, {"checkpoint_sha256": None}
        ),
    }
    report = {"verification": {"rows": [{
        "unit": "u", "verified": True, "author_metric_float32": {"mse": 0, "mae": 0},
        "replay": {"device_uuid": "synthetic", "allclose_rule": True},
    }]}}
    claim = R._claim_for(
        report, {"kind": "report", "sha256": "synthetic", "path": "in-memory"}, "u", ident
    )
    observations["report_without_identities"] = {
        key: claim[key] for key in ("bound", "closure", "row_metric")
    }
    binding = R._claim_for(
        {"identities": {"u": {"row_verified": True, "replay": {"allclose_rule": True}}}},
        {"kind": "binding", "sha256": "synthetic", "path": "in-memory"}, "u", ident,
    )
    observations["binding_without_identity_blocks"] = binding["bound"]

    class PyDataset:
        def __init__(self, **kw):
            pass

    class Dataset:
        def __getitem__(self, i):
            return np.ones((8, 4)), np.ones((4, 4)), None, None

    # Only the framework base class and data provider are fake; batch/mask logic is real.
    tf_stub = types.SimpleNamespace(keras=types.SimpleNamespace(
        utils=types.SimpleNamespace(PyDataset=PyDataset)
    ))
    W = M._windows_class(tf_stub)
    seq = W(Dataset(), list(range(4)), seq_len=8, pred_len=4, batch=2, seed=0, masked=0.25)
    x0, y0 = seq[0]
    x1, y1 = seq[0]
    observations["same_validation_batch"] = {
        "equal_masks": bool(np.array_equal(y0[..., 4:], y1[..., 4:])),
        "different_mask_positions": int(np.sum(y0[..., 4:] != y1[..., 4:])),
    }

    metric = {"mae": 0.25, "mse": 0.125}
    history = {"u": {
        "unit": "u", "replayed_author_metric": metric, "shape": [1, 1, 1],
        "elements": 1, "allclose_rule": True, "max_abs_prediction_difference": 0,
    }}
    claim = R._claim_for(
        history, {"kind": "replay_history", "sha256": "synthetic", "path": "in-memory"},
        "u", ident,
    )
    design = {
        "design_sha256": "synthetic", "horizons": [96, 192, 336, 720],
        "lock": {"published": R.PAPER["L96"]}, "cells": [],
    }
    composition = {"cells": {}}
    for horizon in design["horizons"]:
        unit = f"h{horizon}"
        design["cells"].append({"cell_id": unit, "horizon": horizon})
        composition["cells"][unit] = {
            "seed": 2021, "composition_class": "REPLAY_ON_OBSERVED_DEVICE",
            "properties": {
                "accepted_terminal": True, "closure_verified": True, "score": metric,
                "replay": claim["replay"], "training_device_attribution": {"class": "UNKNOWN"},
            },
        }
    # Synthetic closure/terminal flags isolate the pooling gate, not real custody.
    with patch.object(R, "_baselines_from_catalogs", return_value={}):
        table = R.composed_table(Path("/not-accessed"), design, composition)
    observations["metric_only_wrong_shape_replay_reaches_mean"] = table["four_horizon_mean"]["status"]

    actual = json.loads((ROOT / RETAINED).read_text())
    counts = {}
    for cell in actual["cells"].values():
        sha = cell["properties"]["replay"]["from_evidence"]
        selected = next(c for c in cell["admitted"] if c["evidence_sha256"] == sha)
        kind = selected.get("binding_strength") or selected["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    observations["retained_selected_replay_classes"] = counts
    observations["retained_arithmetic"] = {
        metric_name: float(np.mean([
            cell["properties"]["score"][metric_name] for cell in actual["cells"].values()
        ])) for metric_name in ("mse", "mae")
    }
    heavy = sorted(name for name in ("tensorflow", "torch", "pandas") if name in sys.modules)
    assert not heavy, f"Unexpected heavyweight imports: {heavy}"
    result = {
        "schema": "rp143_musashi_lightweight_pre.v1",
        "reviewed_commit": REVIEW_COMMIT,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("tools/df_sota_repro.py", "tools/df_ecl_modular.py", RETAINED)
        },
        "execution": {
            "cpu_soft_limit_seconds": 10, "cpu_hard_limit_seconds": 15,
            "wall_alarm_seconds": 20, "address_space_limit_mib": 512,
            "numpy_threads": 1, "gpu_visible": False, "heavy_modules_imported": heavy,
        },
        "observations": observations,
        "scope": {
            "actual_functions": [
                "df_sota_repro._bind_replay_record", "df_sota_repro._claim_for",
                "df_sota_repro._replay_history_entries", "df_sota_repro.composed_table",
                "df_sota_repro.agreement", "df_ecl_modular._windows_class",
                "df_ecl_modular._windows_class.<locals>._TrainWindows.__getitem__",
            ],
            "mocked": [
                "TensorFlow namespace and PyDataset base class only; no framework execution",
                "Dataset.__getitem__: constant synthetic NumPy arrays",
                "df_sota_repro._baselines_from_catalogs: returns {}, no filesystem read",
            ],
            "synthetic": "All probe identities, reports, histories, design cells, closure and terminal flags",
            "retained_read": "Only the committed composition JSON; counts and score arithmetic, not identity validation",
            "not_verified": [
                "Production artifact identities or terminal custody",
                "Independent adapter target/metric parity (finding 3 was source review only)",
                "AE reload parity or optimizer freeze/update behavior (finding 5 was source review only)",
                "Full compose_evidence execution or any training/model execution",
            ],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
