from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.i6d_prepare_cost_pilot import prepare


def _inputs(tmp_path: Path, rows: int = 200):
    timestamps = np.arange(rows, dtype="int64") * 3600
    windows = np.zeros((rows, 24, 40), dtype="float32")
    source = tmp_path / "windows.npz"
    np.savez_compressed(
        source,
        windows=windows,
        timestamps=timestamps,
        feature_names=np.asarray([f"feature_{index}" for index in range(40)]),
        dataset_id=np.asarray("fixture"),
    )
    targets = tmp_path / "targets.parquet"
    pd.DataFrame(
        {
            "t_decision_utc": pd.to_datetime(timestamps, unit="s", utc=True),
            "row_id": [f"row-{index}" for index in range(rows)],
            "Y_s_1h": np.linspace(-0.01, 0.01, rows),
            "Y_l_24h": np.linspace(-0.02, 0.02, rows),
        }
    ).to_parquet(targets, index=False)
    return source, targets


def test_prepare_builds_purged_train_only_inputs(tmp_path):
    source, targets = _inputs(tmp_path)

    report = prepare(source, targets, tmp_path / "output")

    assert report["scope"] == "TRAIN_ONLY_DIAGNOSTIC"
    assert report["horizon"] == 1
    assert report["test_read"] is False
    assert report["purged_rows"] == 24
    with np.load(tmp_path / "output" / "train.npz", allow_pickle=False) as train:
        assert train["split"].item() == "train"
        train_last_target = train["target_timestamps"][-1, 0]
    with np.load(tmp_path / "output" / "validation.npz", allow_pickle=False) as validation:
        assert validation["split"].item() == "validation"
        assert validation["baseline"].shape == validation["targets"].shape
        validation_first_input = validation["timestamps"][0] - 23 * 3600
    assert train_last_target < validation_first_input


def test_prepare_supports_long_horizon_and_purges_target_support(tmp_path):
    source, targets = _inputs(tmp_path)

    report = prepare(source, targets, tmp_path / "output", horizon=24)

    assert report["target_name"] == "Y_l_24h"
    with np.load(tmp_path / "output" / "train.npz", allow_pickle=False) as train:
        assert train["horizons"].tolist() == [24]
        train_last_target = train["target_timestamps"][-1, 0]
    with np.load(tmp_path / "output" / "validation.npz", allow_pickle=False) as validation:
        validation_first_input = validation["timestamps"][0] - 23 * 3600
    assert train_last_target < validation_first_input


def test_prepare_rejects_wrong_window_contract(tmp_path):
    source, targets = _inputs(tmp_path)
    with np.load(source, allow_pickle=False) as payload:
        malformed = tmp_path / "malformed.npz"
        np.savez_compressed(
            malformed,
            windows=payload["windows"][:, :, :-1],
            timestamps=payload["timestamps"],
            feature_names=payload["feature_names"][:-1],
            dataset_id=payload["dataset_id"],
        )

    with pytest.raises(ValueError, match="shape"):
        prepare(malformed, targets, tmp_path / "output")
