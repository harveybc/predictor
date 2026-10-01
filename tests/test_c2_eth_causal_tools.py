"""Lane C2 tools, tested first on a planted synthetic world (no real data, CPU, seconds).

What the tests pin:
* the population binding refuses a view, manifest or split whose digest or content disagrees, and reproduces M07's
  origin rule and row-id digest on a synthetic view;
* labels are located at exactly h regular bars (a gap gives no label), in M07's z-units and in raw log return;
* the blocked splits embargo the rows around the held-out block;
* the incremental utility of a planted predictive feature is positive on most blocks, a pure noise feature's is not;
* the leak probe: the vendored producer is the pinned bytes, a stored column produced by it is CAUSAL_BY_RECOMPUTATION
  with no future-row influence, and a stored column shifted from the future is PRODUCER_MISMATCH (and SUSPECT);
* the DML study recovers a planted effect under confounding where the naive slope is biased, the scrambled-label and
  noise controls fail as required, the future-shifted control fires, and a near-deterministic treatment is screened
  as TREATMENT_PREDICTED_BY_CONTROLS;
* the dossier validates against `causal_dossier.v1` with the CONTRACTED_MODEL_READY_VIEW slot, and that slot refuses
  a missing field.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import c2_eth_population as pop_mod  # noqa: E402
from c2_eth_population import (PopulationRefusal, bind_population, blocked_splits, contiguous_blocks,  # noqa: E402
                               m07_origins, m07_row_ids_sha256, sha256_file)

N_ROWS = 1400
TRAIN_END = 1200
BAR = 14400
FEATURES = ["f_signal", "f_noise", "f_dup", "f_leak", "f_vol"]


def _synthetic_view(path: Path, seed=7, gaps=(300, 701)):
    rng = np.random.default_rng(seed)
    times = np.arange(N_ROWS, dtype=np.int64) * BAR + 1_600_000_000
    for g in gaps:  # two irregular steps: everything after g is shifted by one extra bar
        times[g:] += BAR
    z = rng.standard_normal(N_ROWS)
    w = rng.standard_normal(N_ROWS)
    r = np.zeros(N_ROWS)
    # next-bar return depends on z_t (planted signal) and w_t
    r[1:] = 0.004 * z[:-1] + 0.003 * w[:-1] + 0.01 * rng.standard_normal(N_ROWS - 1)
    close = 100 * np.exp(np.cumsum(r))
    frame = pd.DataFrame({
        "DATE_TIME": pd.to_datetime(times, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "typical_price": close, "OPEN": close * (1 + 0.001 * rng.standard_normal(N_ROWS)),
        "HIGH": close * 1.01, "LOW": close * 0.99, "CLOSE": close, "VOLUME": 1000 + 100 * rng.random(N_ROWS),
        "log_return_1": np.concatenate([[0.0], np.diff(np.log(close))]),
        "f_signal": z, "f_noise": rng.standard_normal(N_ROWS), "f_dup": w + 1e-3 * rng.standard_normal(N_ROWS),
        "f_leak": np.concatenate([r[1:], [0.0]]),  # the NEXT bar's return: a leak
        "f_vol": w,
    })
    frame.to_csv(path, index=False)
    return frame, times


def _manifest(path, view_sha, train_end=TRAIN_END):
    doc = {"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "variant": "A_all_admissible_control",
           "resource": {"sha256": view_sha}, "split": {"train": {"rows": [0, train_end]}},
           "admissible_declaration_sha256": pop_mod.DECLARATION_SHA256, "features": FEATURES, "feature_count": len(FEATURES),
           "manifest_sha256_canonical": "0" * 64}
    path.write_text(json.dumps(doc))
    return sha256_file(path)


def _split(path, view_sha, times, lr1, train_end=TRAIN_END, window=24, hmax=6):
    mu, sigma = float(np.mean(lr1[:train_end])), float(np.std(lr1[:train_end]))
    origins, gaps = m07_origins(times, window - 1, train_end - 1 - hmax - window, window, hmax)
    doc = {"schema": "f2.eth_forecast_split.v1", "source_sha256": view_sha, "window": window, "purge_bars": hmax,
           "declared_split": {"train_rows": [0, train_end], "validation_rows": [train_end, N_ROWS - 100], "test_rows": [N_ROWS - 100, N_ROWS]},
           "target": {"definition": "cumulative z", "mu": mu, "sigma": sigma},
           "splits": {"train": {"origin_rows": [int(origins.min()), int(origins.max())], "windows": int(len(origins)),
                                "gap_excluded_windows": int(gaps), "row_ids_sha256": m07_row_ids_sha256(origins, times)}}}
    path.write_text(json.dumps(doc))
    return sha256_file(path), origins


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("c2")
    view = d / "view.csv"
    frame, times = _synthetic_view(view)
    vsha = sha256_file(view)
    msha = _manifest(d / "manifest.json", vsha)
    ssha, origins = _split(d / "split.json", vsha, times, frame["log_return_1"].to_numpy())
    pop = bind_population(view, d / "manifest.json", d / "split.json", expect_view=vsha, expect_manifest=msha, expect_split=ssha)
    return {"dir": d, "pop": pop, "frame": frame, "times": times, "origins": origins, "shas": (vsha, msha, ssha)}


# ---------------------------------------------------------------------------------------------------- binding


def test_binding_reproduces_m07_origins_and_row_ids(world):
    pop = world["pop"]
    assert np.array_equal(pop.origins, world["origins"])
    assert pop.gap_excluded_windows == 2 * (24 + 6 - 1)  # a gap between rows g-1,g excludes origins [g-6, g+22]
    assert pop.bindings["split"]["train_row_ids_sha256"] == m07_row_ids_sha256(pop.origins, pop.times)
    assert pop.train_end == TRAIN_END


def test_binding_refuses_wrong_view_digest(world):
    d = world["dir"]
    with pytest.raises(PopulationRefusal, match="VIEW_SHA256_MISMATCH"):
        bind_population(d / "view.csv", d / "manifest.json", d / "split.json", expect_view="0" * 64,
                        expect_manifest=world["shas"][1], expect_split=world["shas"][2])


def test_binding_refuses_split_that_does_not_reproduce(world):
    d = world["dir"]
    doc = json.loads((d / "split.json").read_text())
    doc["splits"]["train"]["windows"] += 1
    bad = d / "split_bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(PopulationRefusal, match="M07_TRAIN_ORIGINS_NOT_REPRODUCED"):
        bind_population(d / "view.csv", d / "manifest.json", bad, expect_view=world["shas"][0],
                        expect_manifest=world["shas"][1], expect_split=sha256_file(bad))


def test_labels_at_exactly_h_regular_bars(world):
    pop = world["pop"]
    close = world["frame"]["CLOSE"].to_numpy()
    y1 = pop.raw_log_return(1)
    assert np.isnan(y1[299]) and np.isfinite(y1[298])  # the row before the gap has no label at 1 bar
    assert np.isclose(y1[10], np.log(close[11] / close[10]))
    assert np.isnan(y1[TRAIN_END - 1])  # last TRAIN row never reads validation
    y3 = pop.raw_log_return(3)
    assert all(np.isnan(y3[297:300])) and np.isfinite(y3[296])
    z6 = pop.m07_target(6)
    t = 50
    assert np.isclose(pop.z_to_log_return(z6[t], 6), np.log(close[t + 6] / close[t]), atol=1e-9)


def test_blocked_splits_embargo_the_held_out_block(world):
    pop = world["pop"]
    splits = blocked_splits(pop.origins, h=2, purge=6, k=5)
    assert [s["name"] for s in splits] == ["B1", "B2", "B3", "B4", "B5"]
    for s in splits:
        lo, hi = s["eval_rows"]
        assert s["embargo"] == 8
        assert not np.any((s["fit"] >= lo - 8) & (s["fit"] <= hi + 8))
        assert np.all((s["eval"] >= lo) & (s["eval"] <= hi))
    blocks = contiguous_blocks(pop.origins, 5)
    assert sum(hi - lo + 1 >= 1 for _, lo, hi in blocks) == 5


# ---------------------------------------------------------------------------------------------------- deliverable 1


def test_planted_signal_has_positive_incremental_utility(world, tmp_path):
    from c2_feature_contribution import run
    pop = world["pop"]
    table, summary = run(pop, [1], tmp_path, protocols=("blocks5",), skip_hgb=True)
    sig = summary["features"]["f_signal"]["blocks5|h1"]
    noise = summary["features"]["f_noise"]["blocks5|h1"]
    assert sig["n_blocks"] == 5
    assert sig["sign_agreement"] >= 0.8 and sig["delta_mae_z_median"] > 0
    assert abs(noise["delta_mae_z_median"]) < sig["delta_mae_z_median"]
    # f_vol is duplicated by f_dup: its CONDITIONAL incremental utility is ~0 although it is predictive alone
    vol = summary["features"]["f_vol"]["blocks5|h1"]
    assert abs(vol["delta_mae_z_median"]) < 0.1 * sig["delta_mae_z_median"]
    assert vol["only_beats_naive_zero_blocks"] >= 3
    ref = summary["reference"]["blocks5|h1"]
    assert ref["full|selected"]["mae_z_mean"] < ref["naive_zero"]["mae_z_mean"]  # the planted world is predictable
    leak = summary["features"]["f_leak"]["blocks5|h1"]
    assert leak["only_beats_naive_zero_blocks"] == 5
    assert set(table.columns) >= {"population", "view_sha256", "split", "train_rows", "horizon_bars", "n_fit", "n_eval", "mae_z", "mae_log_return"}


# ---------------------------------------------------------------------------------------------------- leak probe


def test_vendored_producer_is_the_pinned_bytes():
    from c2_leak_probe import VENDOR, VENDOR_SHA256
    assert hashlib.sha256(VENDOR.read_bytes()).hexdigest() == VENDOR_SHA256
    prov = json.loads((VENDOR.parent / "PROVENANCE.json").read_text())
    assert prov["sha256"] == VENDOR_SHA256 and prov["commit"].startswith("19fe375a")


def test_leak_probe_passes_a_producer_column_and_catches_a_future_shifted_one(tmp_path):
    import c2_leak_probe as lp
    d = tmp_path
    view = d / "view.csv"
    frame, times = _synthetic_view(view, seed=11, gaps=())
    producer = lp.load_producer()
    raw = lp.raw_frame(pd.read_csv(view, parse_dates=["DATE_TIME"]), N_ROWS)
    tech = producer.compute_technical(raw)
    frame["rsi_14"] = tech["rsi_14"].to_numpy()
    frame["sma_10"] = tech["sma_10"].to_numpy()
    frame["sma_10_future"] = np.concatenate([tech["sma_10"].to_numpy()[1:], [np.nan]])  # shifted from the future
    frame.to_csv(view, index=False)
    vsha = sha256_file(view)
    feats = ["rsi_14", "sma_10", "sma_10_future", "f_leak", "f_noise"]
    doc = {"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "variant": "A", "resource": {"sha256": vsha},
           "split": {"train": {"rows": [0, TRAIN_END]}}, "admissible_declaration_sha256": pop_mod.DECLARATION_SHA256,
           "features": feats, "feature_count": len(feats), "manifest_sha256_canonical": "0" * 64}
    (d / "manifest.json").write_text(json.dumps(doc))
    ssha, _ = _split(d / "split.json", vsha, times, frame["log_return_1"].to_numpy())
    pop = bind_population(view, d / "manifest.json", d / "split.json", expect_view=vsha, expect_manifest=sha256_file(d / "manifest.json"), expect_split=ssha)
    out = lp.run(pop, d / "leak", steps=(600, 900), features=feats, burn_in=300)
    assert out["verdict"]["rsi_14"] == "CAUSAL_BY_RECOMPUTATION"
    assert out["verdict"]["sma_10"] == "CAUSAL_BY_RECOMPUTATION"
    assert out["perturbation"]["sma_10"]["future_rows_influence"] is False
    assert out["perturbation"]["sma_10"]["past_rows_influence"] is True
    # a column the producer cannot reproduce is not certified; the future-shifted copy differs from the producer
    assert out["identity"]["sma_10_future"]["state"] == "PRODUCER_MISMATCH"
    assert out["verdict"]["sma_10_future"].startswith("CAUSALITY_NOT_VERIFIED_PRODUCER_MISMATCH")
    assert "SUSPECT_NEXT_BAR_CORRELATION" in out["verdict"]["f_leak"]
    assert "SUSPECT" not in out["verdict"]["f_noise"]


# ---------------------------------------------------------------------------------------------------- dossier


def _confounded_world(path: Path, theta=0.5, seed=3):
    """X = W + e (confounded by W), Y_{t+1} ~ theta*X + 1.0*W + u: the naive slope of Y on X is biased upward."""
    rng = np.random.default_rng(seed)
    times = np.arange(N_ROWS, dtype=np.int64) * BAR + 1_600_000_000
    w = rng.standard_normal(N_ROWS)
    x = w + 0.8 * rng.standard_normal(N_ROWS)
    x_det = w + 0.05 * rng.standard_normal(N_ROWS)   # nearly determined by the control
    r = np.zeros(N_ROWS)
    r[1:] = 0.01 * (theta * x[:-1] + 1.0 * w[:-1]) + 0.01 * rng.standard_normal(N_ROWS - 1)
    close = 100 * np.exp(np.cumsum(r))
    frame = pd.DataFrame({"DATE_TIME": pd.to_datetime(times, unit="s").strftime("%Y-%m-%d %H:%M:%S"), "typical_price": close,
                          "OPEN": close, "HIGH": close * 1.01, "LOW": close * 0.99, "CLOSE": close, "VOLUME": 1000.0,
                          "log_return_1": np.concatenate([[0.0], np.diff(np.log(close))]), "x_treat": x, "w_conf": w,
                          "x_det": x_det, "noise2": rng.standard_normal(N_ROWS)})
    frame.to_csv(path, index=False)
    return frame, times


@pytest.fixture(scope="module")
def confounded(tmp_path_factory):
    d = tmp_path_factory.mktemp("dml")
    view = d / "view.csv"
    frame, times = _confounded_world(view)
    vsha = sha256_file(view)
    feats = ["x_treat", "w_conf", "x_det", "noise2"]
    doc = {"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "variant": "A", "resource": {"sha256": vsha},
           "split": {"train": {"rows": [0, TRAIN_END]}}, "admissible_declaration_sha256": pop_mod.DECLARATION_SHA256,
           "features": feats, "feature_count": 4, "manifest_sha256_canonical": "0" * 64}
    (d / "manifest.json").write_text(json.dumps(doc))
    ssha, _ = _split(d / "split.json", vsha, times, frame["log_return_1"].to_numpy())
    pop = bind_population(view, d / "manifest.json", d / "split.json", expect_view=vsha, expect_manifest=sha256_file(d / "manifest.json"), expect_split=ssha)
    return {"dir": d, "pop": pop}


def test_dml_recovers_planted_effect_where_naive_slope_is_biased(confounded):
    from c2_causal_dossier import study
    pop = confounded["pop"]
    s = study(pop, "x_treat", 1)
    # units: Y in z-units of log_return_1 (sigma ~ 0.0165): theta_z = 0.01*0.5/sigma
    planted = 0.01 * 0.5 / pop.sigma
    naive_slope = np.polyfit(pop.feature_matrix_train()[pop.origins, 0], np.nan_to_num(pop.m07_target(1)[pop.origins]), 1)[0]
    assert abs(naive_slope - planted) > 3 * s["se"]          # the naive slope is biased by W
    assert abs(s["theta"] - planted) < 2.5 * s["se"]          # the DML estimate is not
    assert s["support_state"] == "SUPPORTED" and s["residual_variance_share"] > 0.2
    assert s["controls"]["scrambled_label"]["failed_as_required"]
    assert s["controls"]["noise_treatment"]["failed_as_required"]
    assert s["controls"]["future_shifted_feature"]["failed_as_required"]          # the shift template fires
    assert s["controls"]["future_shifted_feature"]["causal_column_moved_by_future_rows"] is False
    assert s["residual_variance_share_full_w"] < s["residual_variance_share"] + 1e-9
    assert "w_conf" in s["adjustment"] and "x_treat" not in s["adjustment"]
    # rung 3 under the PLM: delta is exactly theta * (x - x0)
    assert np.isclose(s["rung3"]["delta_mean"], s["theta"] * (np.mean(pop.feature_matrix_train()[pop.origins[np.isfinite(pop.m07_target(1)[pop.origins])], 0]) - s["rung3"]["x0"]), atol=1e-9)


def test_near_deterministic_treatment_is_screened(confounded):
    from c2_causal_dossier import study
    s = study(confounded["pop"], "x_det", 1)
    assert s["support_state"] == "TREATMENT_PREDICTED_BY_CONTROLS"
    assert s["reasons"][0] == "TREATMENT_PREDICTED_BY_CONTROLS"


def test_dossier_validates_against_the_contract(confounded):
    jsonschema = pytest.importorskip("jsonschema")
    from c2_causal_dossier import dossier, study, validate
    pop = confounded["pop"]
    doc = dossier(pop, study(pop, "x_treat", 1), revision="0123456789abcdef")
    schema_path = REPO / "docs" / "contracts" / "causal_dossier.v1.schema.json"
    assert validate(doc, schema_path) == []
    assert doc["rung2"]["state"] == "NOT_IDENTIFIED" and doc["rung2"]["estimate"] is None
    assert doc["rung3"]["label"] == "MODEL_BASED_COUNTERFACTUAL"
    assert doc["data_manifest"]["asset_appearance"]["state"] == "CONTRACTED_MODEL_READY_VIEW"
    assert doc["selection"]["cf_eligible"] is False
    broken = json.loads(json.dumps(doc))
    del broken["data_manifest"]["asset_appearance"]["lake_parent_appearance_id"]
    assert validate(broken, schema_path) != []
    broken2 = json.loads(json.dumps(doc))
    broken2["data_manifest"]["asset_appearance"]["availability_class"] = "GOVERNED"
    assert validate(broken2, schema_path) != []


def test_recommendation_table_builds_from_the_summaries(world, tmp_path):
    from c2_feature_contribution import run
    import c2_leak_probe as lp
    from c2_recommendation_table import build, markdown
    pop = world["pop"]
    _, summary = run(pop, [1], tmp_path / "c", protocols=("blocks5",), skip_hgb=True)
    leak = lp.run(pop, tmp_path / "l", steps=(600,), features=pop.features, burn_in=300)
    table = build(summary, leak)
    assert len(table) == len(pop.features)
    assert {"population", "split", "horizon_bars", "n_eval_rows_total", "naive_zero_mae_z_mean", "leak_verdict"} <= set(table.columns)
    md = markdown(table, summary, leak)
    assert "f_signal" in md and "DEVELOPMENT" in md
