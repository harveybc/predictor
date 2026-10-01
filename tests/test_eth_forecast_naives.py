"""Naive controls, strict record and prediction CSV for lane F2 (numpy only)."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eth_forecast_naives as nv  # noqa: E402
from tools import modular_forecast_evidence as mfe  # noqa: E402

W, F, H, N = 24, 3, [1, 2, 3, 4, 5, 6], 40
MU, SIGMA = 0.0002, 0.02


def make_npz(path, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N + W + 6, F)).astype(np.float32)  # standardized 1-bar returns in column 1
    origins = np.arange(W - 1, W - 1 + N)
    x = np.stack([z[o - W + 1:o + 1] for o in origins])
    y = np.stack([[[sum(z[o + k, 1] for k in range(1, h + 1))] for h in H] for o in origins]).astype(np.float32)
    ts = 1_700_000_000 + origins * 14400
    arrays = dict(windows=x, targets=y, row_ids=np.array([f"eth4h:row{o}:{t}" for o, t in zip(origins, ts)]),
                  timestamps=ts, target_timestamps=ts[:, None] + np.asarray(H) * 14400, dataset_id=np.array("d"),
                  split=np.array("validation"), feature_names=np.array(["a", "log_return_1", "c"]),
                  target_names=np.array(["log_return_1"]), horizons=np.asarray(H), timestamp_unit=np.array("seconds"),
                  metric_space=np.array("z_train"), scaler_identity=np.array("sid"),
                  scaler_scale=np.array([SIGMA]))
    np.savez(path, **arrays)
    return x, y, z, origins


def test_naive_definitions(tmp_path):
    x, y, z, origins = make_npz(tmp_path / "v.npz")
    data, _ = nv._load_validation(tmp_path / "v.npz")
    preds, horizons = nv.naive_predictions(data, MU, SIGMA, 6)
    assert np.allclose(preds["persistence_last_value"][:, 3, 0], x[:, -1, 1])
    assert np.allclose(preds["zero_return"][:, 5, 0], -6 * MU / SIGMA)
    assert np.all(preds["train_mean"] == 0)
    # seasonal: Y_h(t-6) = sum_{k=1..h} z[t-6+k]
    for n, o in enumerate(origins[:3]):
        for k, h in enumerate(H):
            assert preds["seasonal_6"][n, k, 0] == pytest.approx(sum(z[o - 6 + j, 1] for j in range(1, h + 1)), abs=1e-5)
    table = nv.naive_table(data, MU, SIGMA, 6)
    assert set(table["per_naive"]) == {"persistence_last_value", "zero_return", "train_mean", "seasonal_6"}
    for h in H:
        row = table["strict_minimum"][str(h)]
        assert row["MAE"] == min(r[str(h)]["MAE"] for r in table["per_naive"].values())
        e = np.abs(preds["zero_return"][:, h - 1, 0] - y[:, h - 1, 0]).mean()
        assert table["per_naive"]["zero_return"][str(h)]["MAE"] == pytest.approx(e)


def test_seasonal_not_available_when_outside_window(tmp_path):
    make_npz(tmp_path / "v.npz")
    data, _ = nv._load_validation(tmp_path / "v.npz")
    table = nv.naive_table(data, MU, SIGMA, 30)  # period 30 > window 24: no row one period back
    assert table["per_naive"]["seasonal_30"]["1"]["status"] == "NOT_AVAILABLE"
    assert table["strict_minimum"]["1"]["naive"] in ("persistence_last_value", "zero_return", "train_mean")


def fake_receipt(tmp_path, data_sha):
    per_h = {str(h): {"MAE": 0.5 + 0.01 * h, "MSE": 0.4, "baseline_MAE": 1.0, "baseline_MSE": 1.5} for h in H}
    return {"schema_version": "modular.candidate.evaluation.v1", "status": "completed",
            "data": {"test_used": False, "dataset_id": "d", "target_names": ["log_return_1"],
                     "metric_space": "z_train", "scaler_identity": "sid"},
            "digests": {"validation_sha256": data_sha, "model_sha256": "m" * 64, "weights_sha256": "w" * 64},
            "per_horizon": per_h, "candidate": {"cid": "c" * 64}, "bridge": {"predictor_revision": "r" * 40}}


def test_strict_record_rewrites_naive_and_resigns(tmp_path):
    make_npz(tmp_path / "v.npz")
    sha = hashlib.sha256((tmp_path / "v.npz").read_bytes()).hexdigest()
    data, _ = nv._load_validation(tmp_path / "v.npz")
    record = mfe.build(fake_receipt(tmp_path, sha), tmp_path / "v.npz", campaign_id="t")
    table = nv.naive_table(data, MU, SIGMA, 6)
    strict = nv.strict_record(record, table, seasonal_period=6)
    assert mfe.verify(strict) and strict["evidence_sha256"] != record["evidence_sha256"]
    assert strict["schema"] == mfe.SCHEMA and strict["naive"]["mandatory_first_bar"] == "zero_return"
    for entry in strict["per_horizon"]:
        h = str(entry["horizon"])
        assert entry["naive_MAE"] == table["strict_minimum"][h]["MAE"]
        assert entry["MAE"]["skill"] == pytest.approx(1 - entry["model_MAE"] / entry["naive_MAE"])
        assert "naive_MAE" in entry["seasonal_naive"] and "naive_MAE" in entry["zero_return_naive"]
        assert "naive_MAE" in entry["persistence_naive"] and "beats_zero_return_MSE" in entry
        assert entry["beats_zero_return"] == (entry["model_MAE"] < entry["zero_return_naive"]["naive_MAE"] and entry["model_MSE"] < entry["zero_return_naive"]["naive_MSE"])
    assert strict["seasonal_naive"]["period_steps"] == 6


def test_predictions_csv_binds_digest_and_inverts_scale(tmp_path):
    x, y, z, origins = make_npz(tmp_path / "v.npz")
    data, _ = nv._load_validation(tmp_path / "v.npz")
    pred = np.random.default_rng(1).normal(size=y.shape).astype("<f4")
    sha = hashlib.sha256(pred.tobytes()).hexdigest()
    view = tmp_path / "view.csv"
    lines = ["DATE_TIME,CLOSE,x"] + [f"2024-01-{1 + i // 6:02d} {4 * (i % 6):02d}:00:00,{100 + i},0" for i in range(200)]
    view.write_text("\n".join(lines) + "\n")
    info = nv.predictions_csv(pred, data, MU, SIGMA, view, tmp_path / "p.csv", expected_sha256=sha)
    assert info["predictions_sha256"] == sha and info["rows"] == N
    rows = list(csv.DictReader(open(tmp_path / "p.csv")))
    r0 = rows[0]
    assert r0["row_id"] == f"eth4h:row{origins[0]}:{1_700_000_000 + origins[0] * 14400}"
    assert float(r0["close_origin"]) == 100 + origins[0]
    lr = SIGMA * float(pred[0, 2, 0]) + 3 * MU
    assert float(r0["logret_hat_h3"]) == pytest.approx(lr, abs=1e-9)
    assert float(r0["close_hat_h3"]) == pytest.approx((100 + origins[0]) * np.exp(lr), abs=1e-6)
    with pytest.raises(ValueError, match="differ from the verified digest"):
        nv.predictions_csv(pred, data, MU, SIGMA, view, tmp_path / "q.csv", expected_sha256="0" * 64)
