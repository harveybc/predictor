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


def receipt(tmp_path, cid, mae, zero=1.0, persist=1.3, seasonal=1.4,
            campaign_id="f2_eurusd_1h_h1to6_v1"):
    per_h = {h: {"MAE": mae + 0.01 * int(h), "MSE": 2 * mae} for h in H}
    naives = {"per_naive": {"persistence_last_value": {h: {"MAE": persist, "MSE": 2.0} for h in H},
                            "zero_return": {h: {"MAE": zero, "MSE": 1.5} for h in H},
                            "train_mean": {h: {"MAE": zero + 0.001, "MSE": 1.5} for h in H},
                            "seasonal_6": {h: {"MAE": seasonal, "MSE": 2.1} for h in H}},
              "strict_minimum": {h: {"naive": "zero_return", "MAE": zero, "MSE": 1.5} for h in H}}
    r = {"objective": {"value": mae}, "per_horizon": per_h, "naives": naives, "candidate": {"cid": cid},
         "artifact": {"campaign_id": campaign_id},
         "population": {"asset": "EURUSD 1h", "dataset_id": "eurusd:fixture:v1", "sample_hours": 1,
                        "targets": ["log_return_1"], "rows": 120},
         "scale": {"metric_space": "z_train", "scaler_identity": "train-scaler:fixture"},
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
    table = cl.closure([q1, q2], sigma=0.02, seeds=(2021, 2022))
    g = table["configurations"]["grouped32_mae_adam"]
    assert g["eligible"] and g["mean_MAE_z"] == pytest.approx(0.91) and g["spread"] == pytest.approx(0.02)
    assert g["beats_zero_return_all_horizons_all_seeds"] is True
    assert table["configurations"]["per_feature_mae_adam"]["beats_zero_return_all_horizons_all_seeds"] is False
    row = g["per_seed"][2021]["per_horizon"]["3"]
    assert row["model_MAE_logret"] == pytest.approx(0.93 * 0.02) and row["strict_naive"] == "zero_return"
    assert row["vs_strict"]["skill"] == pytest.approx(1 - 0.93 / 1.0)
    assert table["literature"]["status"] == "NOT_AVAILABLE" and table["comparability"]["status"] == "NOT_COMPARABLE"
    assert table["campaign_identity"]["campaign_id"] == "f2_eurusd_1h_h1to6_v1"
    assert "EURUSD 1h" in table["literature"]["reason"]
    assert "ETHUSDT" not in table["literature"]["reason"]
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


def test_refuses_mixed_campaign_identity_before_aggregation(tmp_path):
    q1 = make_queue(tmp_path, "q1.sqlite", [("grouped32_mae_adam", 2021, 0.9, "verified")])
    q2 = make_queue(tmp_path, "q2.sqlite", [("grouped32_mae_adam", 2022, 0.8, "verified")])
    receipt_path = next(p for p in tmp_path.glob("*_accepted.json") if "2022" in p.name)
    changed = json.loads(receipt_path.read_text())
    changed["artifact"]["campaign_id"] = "another_campaign"
    receipt_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="mixed campaign identity"):
        cl.closure([q1, q2], sigma=0.02, seeds=(2021, 2022))


def test_single_seed_cli_retains_existing_flags(tmp_path, monkeypatch):
    q = make_queue(tmp_path, "q.sqlite", [("grouped32_mae_adam", 2021, .9, "verified"),
                                          ("per_feature_mae_adam", 2021, .8, "verified")])
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"target": {"sigma": .02}}))
    out = tmp_path / "closure"
    monkeypatch.setattr(sys, "argv", ["closure", "--queue", str(q), "--manifest", str(manifest), "--out", str(out)])
    cl.main()
    table = json.loads(out.with_suffix(".json").read_text())
    assert table["contrasts"][0]["spread_a"] is None
    assert "unavailable" in table["contrasts"][0]["label_rule"]
    assert len(out.with_suffix(".csv").read_text().splitlines()) == 13


@pytest.mark.parametrize("wrong_digest", [False, True])
def test_historical_receipt_identity_from_bound_declaration(tmp_path, wrong_digest):
    q = make_queue(tmp_path, "queue.sqlite", [("grouped32_mae_adam", 2021, .9, "verified")])
    p = next(tmp_path.glob("*_accepted.json"))
    r = json.loads(p.read_text())
    for key in ("artifact", "population", "scale"):
        r.pop(key)
    r["data"] = {"dataset_id": "eth:fixture", "target_names": ["log_return_1"],
                 "horizons": list(range(1, 7)), "metric_space": "z_train",
                 "scaler_identity": "train:fixture"}
    r["digests"].update(train_sha256="t" * 64, validation_sha256="v" * 64)
    p.write_text(json.dumps(r))
    declaration = {"campaign_id": "eth_fixture", "asset": "ETHUSDT", "base": {"sample_hours": 4},
                   "data_manifest": {"dataset_id": "eth:fixture", "scaler_identity": "train:fixture"},
                   "data": {"train": {"sha256": "x" * 64 if wrong_digest else "t" * 64},
                            "validation": {"sha256": "v" * 64}}}
    (tmp_path / "CAMPAIGN.json").write_text(json.dumps(declaration))
    if wrong_digest:
        with pytest.raises(ValueError, match="digest"):
            cl.closure([q], sigma=.02)
    else:
        identity = cl.closure([q], sigma=.02)["campaign_identity"]
        assert identity["campaign_id"] == "eth_fixture"
        assert identity["sample_hours"] == 4
        assert identity["validation_sha256"] == "v" * 64


def test_duplicate_label_seed_cannot_hide_other_identity(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    q1 = make_queue(a, "q.sqlite", [("grouped32_mae_adam", 2021, .9, "verified")])
    q2 = make_queue(b, "q.sqlite", [("grouped32_mae_adam", 2021, .8, "verified")])
    p = next(b.glob("*_accepted.json"))
    r = json.loads(p.read_text())
    r["artifact"]["campaign_id"] = "different"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        cl.closure([q1, q2], sigma=.02)
