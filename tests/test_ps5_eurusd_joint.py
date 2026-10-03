"""Contract tests for one EURUSD PS5 development comparison."""
import csv
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import ps5_eurusd_joint as ps5


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    ps1, ps2 = tmp_path / "ps1", tmp_path / "batch_001"
    ps1.mkdir(parents=True)
    ps2.mkdir()
    features = ["px.logret_1h", "px.ewma_vol_168", "px.zclose_24"]
    (ps1 / "admissible_features.json").write_text(json.dumps({"features": [
        {"feature_id": f, "admissibility": "ADMISSIBLE"} for f in features]}))
    (ps1 / "digests.json").write_text(json.dumps({"artifacts_sha256": {
        "admissible_features.json": _sha(ps1 / "admissible_features.json")}}))
    (ps1 / "READY").write_text(json.dumps({"batch": "batch_001", "digests_sha256": "a" * 64}))
    with (ps2 / "ps2_status.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "target", "horizon", "status"])
        w.writeheader()
        w.writerows({"feature": f, "target": "Y_l", "horizon": 24,
                     "status": "EXPLORATION" if f == features[-1] else "PROVISIONAL_SURVIVOR"}
                    for f in features)
    with (ps2 / "ps2_synergy.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "b", "target", "horizon", "fold"])
        w.writeheader()
        w.writerow({"a": features[1], "b": features[2], "target": "Y_l", "horizon": 24,
                    "fold": "inner_2019"})
    # A regular elapsed-hour grid, including a deliberate missing market observation.
    ts = np.arange(900, dtype="int64") * 3600 + 1338422400
    cols = {"x__" + f: np.sin(np.arange(len(ts)) / (i + 8)).astype("float32")
            for i, f in enumerate(features)}
    cols["x__" + features[-1]][350] = np.nan
    np.savez(ps2 / "series.npz", timestamps=ts, **cols)
    yy = np.sin(np.arange(len(ts)) / 18).astype("float32")[:, None]
    np.savez(ps2 / "targets.npz", timestamps=ts, Y_l=yy, Y_s=yy)
    manifest = {"schema": "ps2_batch.v1", "asset": "EURUSD", "batch_id": "batch_001",
                "sampling_period_seconds": 3600, "features": features,
                "producer": {"lane_a_batch": "batch_001", "lane_a_ready_digests_sha256": "a" * 64},
                "series": {"file": "series.npz", "sha256": _sha(ps2 / "series.npz")},
                "targets": {"file": "targets.npz", "sha256": _sha(ps2 / "targets.npz")},
                "target_columns": {"Y_l": ["Y_l_24h"], "Y_s": ["Y_s_1h"]},
                "train_end_ts": int(ts[-1]),
                "folds": [{"fold_id": "inner_2019", "split": "train",
                           "fit": [int(ts[200]), int(ts[600])],
                           "val": [int(ts[760]), int(ts[899])]}]}
    (ps2 / "batch_manifest.json").write_text(json.dumps(manifest))
    (ps2 / "ps2_manifest.json").write_text(json.dumps({"batch_id": "batch_001", "train_rows": len(ts),
        "train_last_time": int(ts[-1]), "control_all_admissible": features,
        "provenance": {"lane_a_batch": "batch_001",
        "lane_a_ready_digests_sha256": "a" * 64,
        "lane_a_artifacts_sha256": {"admissible_features.json": _sha(ps1 / "admissible_features.json")}},
        "output_sha256": {n: _sha(ps2 / n) for n in ("ps2_status.csv", "ps2_synergy.csv")}}))
    return ps1, ps2, features


def _load(tmp_path):
    ps1, ps2, features = fixture(tmp_path)
    return ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                             target="Y_l", horizon=24, fold_id="inner_2019"), ps1, ps2, features


def test_retained_feature_ids_digests_and_train_rows(tmp_path):
    data, ps1, ps2, features = _load(tmp_path)
    assert data["features"] == features
    assert data["identity"]["series_sha256"] == _sha(ps2 / "series.npz")
    assert data["identity"]["ps1_features_sha256"] == _sha(ps1 / "admissible_features.json")
    assert len(data["timestamps"]) == 900
    assert data["timestamps"][-1] == data["train_end_ts"]
    man = json.loads((ps2 / "batch_manifest.json").read_text())
    man["features"][0] = "ETH_CLOSE"
    (ps2 / "batch_manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="feature IDs"):
        ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                          target="Y_l", horizon=24, fold_id="inner_2019")


def test_reject_digest_and_external_rows(tmp_path):
    data, ps1, ps2, features = _load(tmp_path)
    with (ps2 / "series.npz").open("ab") as fh:
        fh.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                          target="Y_l", horizon=24, fold_id="inner_2019")
    ps1, ps2, features = fixture(tmp_path / "again")
    man = json.loads((ps2 / "batch_manifest.json").read_text())
    man["train_end_ts"] -= 3600
    (ps2 / "batch_manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="TRAIN"):
        ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                          target="Y_l", horizon=24, fold_id="inner_2019")


def test_reject_ps1_ps2_provenance_and_row_count(tmp_path):
    _, ps1, ps2, features = _load(tmp_path)
    man = json.loads((ps2 / "batch_manifest.json").read_text())
    man["producer"]["lane_a_ready_digests_sha256"] = "b" * 64
    (ps2 / "batch_manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="provenance"):
        ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                          target="Y_l", horizon=24, fold_id="inner_2019")
    ps1, ps2, features = fixture(tmp_path / "again")
    ps2man = json.loads((ps2 / "ps2_manifest.json").read_text())
    ps2man["train_rows"] -= 1
    (ps2 / "ps2_manifest.json").write_text(json.dumps(ps2man))
    with pytest.raises(ValueError, match="row count"):
        ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                          target="Y_l", horizon=24, fold_id="inner_2019")


def test_purged_train_only_common_rows_budget_and_paired_naive(tmp_path):
    data, _, _, features = _load(tmp_path)
    calls = []

    def fit(x_train, y_train, x_stop, y_stop, x_val, *, seed, settings, config, source):
        calls.append((x_train.copy(), x_stop.copy(), x_val.copy(), seed, dict(settings), config))
        assert x_train.shape[1:] == x_stop.shape[1:] == x_val.shape[1:] == (24, 3)
        assert len(y_train) == len(x_train) and len(y_stop) == len(x_stop)
        return np.zeros(len(x_val), dtype="float32"), {"updates": 2, "epochs": 1}

    out = ps5.evaluate_one(data, seeds=[7, 11], window=24, settings={"max_epochs": 2,
                            "max_updates": 8, "batch_size": 16}, fit_fn=fit, source="pin")
    assert len(calls) == 8
    assert [c[3] for c in calls] == [7, 11] * 4
    assert all(c[4] == calls[0][4] and c[5] == calls[0][5] for c in calls)
    assert len({a["rows_sha256"] for a in out["arms"].values()}) == 1
    assert out["naive"]["rows_sha256"] == out["arms"]["base"]["rows_sha256"]
    assert all(a["naive_mae"] == out["naive"]["mae"] for a in out["arms"].values())
    assert out["paired_naive_gate"]["strategy_eligible"] is False
    assert all(a["beats_paired_naive"] is False for a in out["arms"].values())
    assert out["population"]["max_fit_label_ts"] < out["population"]["stop_start_ts"]
    assert out["population"]["max_stop_label_ts"] <= out["population"]["fit_end_ts"]
    assert out["population"]["max_val_label_ts"] <= out["population"]["val_end_ts"]
    assert out["population"]["purge_h"] == 144
    assert out["selection_release"] is None and out["evidence_scope"] == "DEVELOPMENT_INCOMPLETE_POPULATION"
    assert set(out["arms"]) == {"base", "base+a", "base+b", "base+a+b"}
    assert all(c[5]["feature_names"] == features for c in calls)
    assert np.all(calls[0][0][:, :, 1:] == 0)  # fixed architecture; absent inputs masked
    assert np.any(calls[-1][0][:, :, 1:] != 0)


def test_relative_reentry_does_not_pass_paired_naive_gate(tmp_path):
    data, _, _, _ = _load(tmp_path)
    calls = iter((100.0, 90.0, 80.0, 70.0))

    def fit(x_train, y_train, x_stop, y_stop, x_val, **kwargs):
        return np.full(len(x_val), next(calls)), {"updates": 1, "epochs": 1}

    out = ps5.evaluate_one(data, seeds=[7], window=24,
                           settings={"max_epochs": 1}, fit_fn=fit, source="pin")
    assert out["comparison"] == "RE_ENTERS_THIS_FOLD"
    assert out["paired_naive_gate"] == {"status": "FAIL", "joint_beats_naive": False,
                                        "strategy_eligible": False,
                                        "reason": "incomplete selection population"}
    assert all(arm["skill_vs_paired_naive"] < 0 for arm in out["arms"].values())


def test_no_test_read_and_one_pair_only(tmp_path, monkeypatch):
    _, ps1, ps2, features = _load(tmp_path)
    forbidden = ps2 / "external_test.npz"
    forbidden.write_bytes(b"should never be opened")
    old = Path.open

    def guarded(path, *args, **kwargs):
        if path == forbidden:
            raise AssertionError("external test read")
        return old(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    data = ps5.load_retained(ps1, ps2, base=[features[0]], pair=features[1:],
                             target="Y_l", horizon=24, fold_id="inner_2019")
    out = ps5.evaluate_one(data, seeds=[7], window=24,
                           settings={"max_epochs": 1, "max_updates": 3, "batch_size": 16},
                           fit_fn=lambda x, y, sx, sy, vx, **kw: (np.zeros(len(vx)),
                                                                    {"updates": 1, "epochs": 1}), source="pin")
    assert out["pair"] == features[1:] and len(out["arms"]) == 4
    assert out["refits"] == 4


def test_all_admissible_control_and_omitted_states(tmp_path):
    data, _, ps2, features = _load(tmp_path)
    other = ps2.parent / "batch_002"
    other.mkdir()
    (other / "ps2_manifest.json").write_text(json.dumps({"batch_id": "batch_002",
        "control_all_admissible": ["fx.rate", "fx.spread"]}))
    (other / "batch_manifest.json").write_text(json.dumps({"batch_id": "batch_002",
        "features": ["fx.rate"]}))
    pop = ps5.population_control(ps2, data["features"])
    assert pop["all_admissible_count"] == 5
    assert pop["evaluated_all_admissible_control"] is False
    assert pop["all_admissible_control_state"] == "PENDING_NOT_EXECUTED"
    assert {x["state"] for x in pop["omitted_set"]} == {
        "RETAINED_SERIES_PAYLOAD_MISSING", "NO_RETAINED_SERIES_PAYLOAD"}
    assert {x["feature_id"] for x in pop["omitted_set"]} == {"fx.rate", "fx.spread"}
    assert pop["complete_three_batch_population"] is False


def test_canonical_metadata_identity_and_full_pending_denominator():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/audits/evidence/canonical_20261003"
    ps1 = evidence / "laneA/batch_001"
    ps2 = evidence / "laneB/batch_001"
    assert _sha(ps1 / "admissible_features.json") == (
        "438fb732cb071ded24a5107155baabb662373ed504895984d0d8600198742d2a")
    man = json.loads((ps2 / "batch_manifest.json").read_text())
    assert man["series"]["sha256"] == "3e4404d260d9236ca24fc8561eedbfcc9c47a609d3e64b68870c204a3731f025"
    assert man["targets"]["sha256"] == "7dbb0b9552f27873d0bf58604908cd8f6b46c7daa1ee56cb44cd23c42fa9c209"
    pair = ["px.ewma_vol_168", "px.zclose_24"]
    pop = ps5.population_control(ps2, ["px.logret_1h", *pair])
    assert pop["all_admissible_count"] == 366
    assert pop["complete_three_batch_population"] is True
    assert len(pop["omitted_set"]) == 363
    assert pop["all_admissible_control_state"] == "PENDING_NOT_EXECUTED"
    assert not any(x["state"] == "RETAINED_SERIES_NOT_RUN" for x in pop["omitted_set"])


def test_validation_perturbation_cannot_change_train_scaler(tmp_path):
    data, _, _, _ = _load(tmp_path)
    other = copy.deepcopy(data)
    start = other["fold"]["val"][0]
    for values in other["columns"].values():
        values[other["timestamps"] >= start] += 10000
    captured = []

    def fit(x_train, y_train, x_stop, y_stop, x_val, **kwargs):
        captured.append((x_train.copy(), x_stop.copy(), x_val.copy()))
        return np.zeros(len(x_val)), {"updates": 1, "epochs": 1}

    for current in (data, other):
        ps5.evaluate_one(current, seeds=[7], window=24,
                         settings={"max_epochs": 1, "max_updates": 1}, fit_fn=fit, source="pin")
    for i in range(4):
        np.testing.assert_array_equal(captured[i][0], captured[i + 4][0])
        np.testing.assert_array_equal(captured[i][1], captured[i + 4][1])
    assert not np.array_equal(captured[3][2], captured[7][2])


def test_explicit_source_pin_rejects_other_commit(tmp_path, monkeypatch):
    (tmp_path / "predictor_plugins/modular_temporal").mkdir(parents=True)
    (tmp_path / "predictor_plugins/modular_temporal/__init__.py").write_text("")
    monkeypatch.setattr(ps5.subprocess, "check_output", lambda *a, **k: "wrong-commit\n")
    with pytest.raises(ValueError, match="clean commit"):
        ps5.verify_source(tmp_path)
