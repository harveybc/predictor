"""Closure table from synthetic queues and receipts (numpy-free)."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eth_forecast_closure as cl  # noqa: E402
from tools.modular_doin_campaign import SCHEMA_SQL  # noqa: E402

H = ["1", "2", "3", "4", "5", "6"]


def receipt(tmp_path, cid, mae, zero=1.0, persist=1.3, seasonal=1.4):
    per_h = {h: {"MAE": mae + 0.01 * int(h), "MSE": 2 * mae} for h in H}
    naives = {"per_naive": {"persistence_last_value": {h: {"MAE": persist, "MSE": 2.0} for h in H},
                            "zero_return": {h: {"MAE": zero, "MSE": 1.5} for h in H},
                            "train_mean": {h: {"MAE": zero + 0.001, "MSE": 1.5} for h in H},
                            "seasonal_6": {h: {"MAE": seasonal, "MSE": 2.1} for h in H}},
              "strict_minimum": {h: {"naive": "zero_return", "MAE": zero, "MSE": 1.5} for h in H}}
    r = {"objective": {"value": mae}, "per_horizon": per_h, "naives": naives, "candidate": {"cid": cid},
         "training": {"selected_epoch": 3, "observed_updates": 630, "stop_reason": "patience"},
         "digests": {"weights_sha256": "w" * 64, "model_sha256": "m" * 64}}
    p = tmp_path / f"{cid[:30]}_accepted.json"
    p.write_text(json.dumps(r))
    v = tmp_path / f"{cid[:30]}_verification.json"
    v.write_text(json.dumps({"verdict": "VERIFIED", "exact_match": True}))
    return p, v


def make_queue(tmp_path, name, cells):
    db = sqlite3.connect(tmp_path / name)
    db.executescript(SCHEMA_SQL)
    pos = 0
    for label, seed, mae, status in cells:
        cid = f"{label}{seed}".ljust(64, "x")
        db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (cid, pos, label.ljust(64, "c"), seed, label, "{}", "{}", status, None, 0, mae, 0))
        pos += 1
        if status == "verified":
            p, v = receipt(tmp_path, cid, mae)
            db.execute("INSERT INTO attempts(cid, attempt, kind, started, status, output_root, receipt_path, verdict) "
                       "VALUES(?,?,?,?,?,?,?,?)", (cid, 1, "train", 0, "completed", "", str(p), None))
            db.execute("INSERT INTO attempts(cid, attempt, kind, started, status, output_root, receipt_path, verdict) "
                       "VALUES(?,?,?,?,?,?,?,?)", (cid, 1, "verify", 0, "completed", "", str(v), "VERIFIED"))
    db.commit()
    return tmp_path / name


def test_closure_pairs_cells_and_never_says_advantage_within_spread(tmp_path):
    q1 = make_queue(tmp_path, "q1.sqlite", [("grouped32_mae_adam", 2021, 0.90, "verified"),
                                            ("grouped32_mae_adam", 2022, 0.92, "verified"),
                                            ("control_mlp_mae_adam", 2021, 0.95, "verified")])
    q2 = make_queue(tmp_path, "q2.sqlite", [("control_mlp_mae_adam", 2022, 0.93, "verified"),
                                            ("per_feature_mae_adam", 2021, 1.10, "verified"),
                                            ("per_feature_mae_adam", 2022, 1.12, "verified"),
                                            ("per_feature_huber_adam", 2021, 0.5, "queued")])
    table = cl.closure([q1, q2], sigma=0.02)
    g = table["configurations"]["grouped32_mae_adam"]
    assert g["eligible"] and g["mean_MAE_z"] == pytest.approx(0.91) and g["spread"] == pytest.approx(0.02)
    assert g["beats_zero_return_all_horizons_all_seeds"] is True
    assert table["configurations"]["per_feature_mae_adam"]["beats_zero_return_all_horizons_all_seeds"] is False
    row = g["per_seed"][2021]["per_horizon"]["3"]
    assert row["model_MAE_logret"] == pytest.approx(0.93 * 0.02) and row["strict_naive"] == "zero_return"
    assert row["vs_strict"]["skill"] == pytest.approx(1 - 0.93 / 1.0)
    assert table["literature"]["status"] == "NOT_AVAILABLE" and table["comparability"]["status"] == "NOT_COMPARABLE"
    assert [u["label"] for u in table["unverified"]] == ["per_feature_huber_adam"]
    by = {(c["a"], c["b"]): c for c in table["contrasts"]}
    # control (0.95, 0.93) vs grouped32 (0.90, 0.92): mean diff 0.03 > spreads 0.02 -> stated with numbers
    c = by[("control_mlp_mae_adam", "grouped32_mae_adam")]
    assert c["mean_difference"] == pytest.approx(0.03) and "exceeds both seed spreads" in c["label_rule"]
    assert "grouped32_mae_adam" in c["label_rule"] and "advantage" not in c["label_rule"]
    # grouped32 vs per_feature differ by 0.2 but spreads 0.02 -> exceeds; control vs per_feature
    assert "within" not in by[("grouped32_mae_adam", "per_feature_mae_adam")]["label_rule"]
    cl.write_csv(table, tmp_path / "t.csv")
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert len(lines) == 1 + 6 * 6 and lines[0].startswith("configuration,seed,horizon")


def test_refuses_non_exact_verification(tmp_path):
    q = make_queue(tmp_path, "q.sqlite", [("grouped32_mae_adam", 2021, 0.9, "verified")])
    for v in tmp_path.glob("*_verification.json"):
        v.write_text(json.dumps({"verdict": "VERIFIED", "exact_match": False}))
    with pytest.raises(ValueError, match="exact-match"):
        cl.closure([q], sigma=0.02)
