"""Control evaluator + cell runner + independent scorer on a tiny synthetic NPZ (needs TensorFlow, CPU).

Run on a worker under crispdm-run (e.g. -m 3G): CUDA_VISIBLE_DEVICES="" TF_DETERMINISTIC_OPS=1.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eth_forecast_campaign as fc  # noqa: E402
from tools import eth_forecast_dataset as ds  # noqa: E402
from tools import modular_checkpoint_scorer as scorer  # noqa: E402
from tools.eth_control_evaluator import control_parameters, evaluate_control  # noqa: E402
from tests.test_eth_forecast_dataset import FEATURES, SPLIT, make_view  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data(tmp_path):
    view = tmp_path / "view.csv"
    make_view(view)
    out = tmp_path / "npz"
    ds.build(view, out, features=FEATURES, window=24, horizons=fc.HORIZONS, split=SPLIT, expected_sha=None)
    return out, view


def base_from(out):
    with np.load(out / "validation.npz") as z:
        names = z["feature_names"].astype(str).tolist()
    return {"feature_names": names, "window": 24, "sample_hours": 4.0, "horizons": fc.HORIZONS,
            "target_feature_indices": [names.index("log_return_1")],
            "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False, "unit": "z_train"},
            "evaluator_fixed": {"max_updates": 1000000, "max_seconds": 600.0}}


def test_control_receipt_verifies_exactly_and_counts_parameters(tmp_path, data):
    out, _ = data
    base = base_from(out)
    nested = fc.control_candidate(base, "huber", "adam", 2021, [8, 8])
    nested["evaluator"]["max_epochs"] = 2
    result, prediction = evaluate_control(nested, out / "train.npz", out / "validation.npz", tmp_path / "cell")
    assert result["schema_version"] == "modular.candidate.evaluation.v1" and result["architecture"] == "flatten_mlp_control"
    assert result["control"]["trainable_parameters"] == control_parameters(24, len(FEATURES), [8, 8], 6, 1)
    assert prediction.shape == (result["data"]["validation_rows"], 6, 1)
    assert set(result["per_horizon"]) == {"1", "2", "3", "4", "5", "6"}
    assert result["training"]["settings"]["weight_decay"] == 0.0
    verification = scorer.verify(tmp_path / "cell" / "evaluation.json", out / "validation.npz", tmp_path / "v.json")
    assert verification["verdict"] == "VERIFIED", verification["problems"]
    assert verification["exact_match"] is True


def test_cell_runner_writes_accepted_receipt_with_naives(tmp_path, data):
    out, _ = data
    base = base_from(out)
    nested = fc.control_candidate(base, "mae", "adamw", 2022, [4, 4])
    nested["evaluator"]["max_epochs"] = 1
    (tmp_path / "cand.json").write_text(json.dumps(nested))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_DETERMINISTIC_OPS": "1", "PYTHONPATH": str(ROOT),
           "F2_HOST_ROLE": "test"}
    done = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "eth_cell_runner.py"), "--candidate",
                           str(tmp_path / "cand.json"), "--train", str(out / "train.npz"), "--validation",
                           str(out / "validation.npz"), "--out", str(tmp_path / "run"), "--revision", "r" * 40,
                           "--campaign-id", "t", "--cid", "c" * 64, "--label", "control_mlp_mae_adamw",
                           "--manifest", str(out / "MANIFEST.json"), "--heartbeat-interval", "5"],
                          capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    receipt = json.loads((tmp_path / "run" / "accepted.json").read_text())
    assert receipt["candidate"]["cid"] == "c" * 64 and receipt["bridge"]["host_role"] == "test"
    assert receipt["naives"]["per_naive"]["zero_return"]["6"]["MAE"] > 0
    assert set(receipt["naives"]["beats_zero_return"]) == {"1", "2", "3", "4", "5", "6"}
    assert (tmp_path / "run" / "heartbeat.json").exists() and (tmp_path / "run" / "heartbeat.jsonl").exists()
    pred = np.load(receipt["artifacts"]["predictions_validation"])
    import hashlib
    assert hashlib.sha256(pred.tobytes()).hexdigest() == receipt["digests"]["predictions_sha256"]
    # the independent scorer reproduces exactly these bytes
    verification = scorer.verify(tmp_path / "run" / "accepted.json", out / "validation.npz", tmp_path / "v.json")
    assert verification["verdict"] == "VERIFIED" and verification["exact_match"] is True
    assert verification["digests"]["predictions_sha256"] == receipt["digests"]["predictions_sha256"]
