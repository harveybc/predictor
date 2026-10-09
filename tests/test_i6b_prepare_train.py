import json

import numpy as np
import pandas as pd
import pytest

from tools import i6b_prepare_train as subject


def freeze(members):
    document = {
        "schema": "fs4.test_freeze.v1",
        "freeze_sha256": "f" * 64,
        "winners": {"EURUSD": {
            "Y_s_1h": {"RAW": {"members": members[:2]}},
            "Y_l_24h": {"RAW": {"members": members[1:]}},
        }},
    }
    return document


def frame(start="2023-01-01", periods=40):
    ts = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.DataFrame({
        "t_decision_utc": ts,
        "row_id": [f"r{i}" for i in range(periods)],
        "a": np.arange(periods, dtype=float),
        "b": np.arange(periods, dtype=float) * 2,
        "c": np.arange(periods, dtype=float) * -1,
    })


def test_selected_union_is_raw_eurusd_only_and_sorted():
    document = freeze(["c", "a", "b"])
    assert subject.selected_features(document) == ("a", "b", "c")
    document["winners"]["EURUSD"]["Y_s_1h"]["TRAINED_ENCODER"] = {"members": ["forbidden"]}
    assert subject.selected_features(document) == ("a", "b", "c")


def test_prepare_uses_exact_hourly_train_windows_and_value_mask_pairs(tmp_path):
    source = tmp_path / "features.parquet"
    table = frame()
    table.loc[10, "b"] = np.nan
    table.to_parquet(source)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze(["c", "a", "b"])))

    out = tmp_path / "train.npz"
    report = subject.prepare(freeze_path, [source], out, train_end="2024-01-01T00:00:00Z")

    with np.load(out, allow_pickle=False) as data:
        assert data["split"].item() == "train"
        assert data["windows"].shape == (17, 24, 6)
        assert data["feature_names"].tolist() == [
            "a__value", "a__observed", "b__value", "b__observed", "c__value", "c__observed"]
        assert np.isfinite(data["windows"]).all()
        assert data["timestamps"].max() < pd.Timestamp("2024-01-01", tz="UTC").timestamp()
        assert data["windows"][:, :, 3].min() == 0.0
    assert report["selected_features"] == ["a", "b", "c"]
    assert report["validation_read"] is False and report["test_read"] is False
    config = json.loads(out.with_suffix(".config.json").read_text())
    assert [branch["features"] for branch in config["branches"]] == [
        ["a__value", "a__observed"], ["b__value", "b__observed"], ["c__value", "c__observed"]]


def test_prepare_refuses_gaps_duplicates_and_post_train_rows(tmp_path):
    table = frame(periods=26).drop(index=7)
    source = tmp_path / "features.parquet"
    table.to_parquet(source)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze(["a", "b", "c"])))
    with pytest.raises(ValueError, match="NO_COMPLETE_HOURLY_WINDOWS"):
        subject.prepare(freeze_path, [source], tmp_path / "out.npz", train_end="2023-01-02T02:00:00Z")


def test_prepare_refuses_a_feature_missing_from_sources(tmp_path):
    source = tmp_path / "features.parquet"
    frame().drop(columns=["c"]).to_parquet(source)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze(["a", "b", "c"])))
    with pytest.raises(ValueError, match="SELECTED_FEATURE_MISSING"):
        subject.prepare(freeze_path, [source], tmp_path / "out.npz", train_end="2024-01-01T00:00:00Z")
