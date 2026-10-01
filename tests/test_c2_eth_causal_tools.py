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
FEATURES = ["f_signal", "f_noise", "f_dup", "f_vol", "f_leak"]  # the exact leak is LAST so max_features=4 excludes it


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
    # without the exact leak column (which alone reproduces the label and would zero every arm's error)
    table, summary = run(pop, [1], tmp_path / "nol", protocols=("blocks5",), skip_hgb=True, max_features=4)
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
    table, summary_all = run(pop, [1], tmp_path / "all", protocols=("blocks5",), skip_hgb=True)
    leak = summary_all["features"]["f_leak"]["blocks5|h1"]
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
    frame["sma_20"] = tech["sma_20"].to_numpy()
    frame["sma_10"] = np.concatenate([tech["sma_10"].to_numpy()[1:], [np.nan]])  # stored under the producer's name, but shifted from the future
    frame.to_csv(view, index=False)
    vsha = sha256_file(view)
    feats = ["rsi_14", "sma_20", "sma_10", "f_leak", "f_noise"]
    doc = {"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "variant": "A", "resource": {"sha256": vsha},
           "split": {"train": {"rows": [0, TRAIN_END]}}, "admissible_declaration_sha256": pop_mod.DECLARATION_SHA256,
           "features": feats, "feature_count": len(feats), "manifest_sha256_canonical": "0" * 64}
    (d / "manifest.json").write_text(json.dumps(doc))
    ssha, _ = _split(d / "split.json", vsha, times, frame["log_return_1"].to_numpy())
    pop = bind_population(view, d / "manifest.json", d / "split.json", expect_view=vsha, expect_manifest=sha256_file(d / "manifest.json"), expect_split=ssha)
    out = lp.run(pop, d / "leak", steps=(600, 900), features=feats, burn_in=300)
    assert out["verdict"]["rsi_14"] == "CAUSAL_BY_RECOMPUTATION"
    assert out["verdict"]["sma_20"] == "CAUSAL_BY_RECOMPUTATION"
    assert out["perturbation"]["sma_20"]["future_rows_influence"] is False
    assert out["perturbation"]["sma_20"]["past_rows_influence"] is True
    # the stored sma_10 is not what the producer emits at t: not certified
    assert out["identity"]["sma_10"]["state"] == "PRODUCER_MISMATCH"
    assert out["verdict"]["sma_10"].startswith("CAUSALITY_NOT_VERIFIED_PRODUCER_MISMATCH")
    # a column the producer never emits is a state of its own, and the exact leak is SUSPECT
    assert out["identity"]["f_leak"]["state"] == "NOT_PRODUCED_BY_PRODUCER"
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


# ---------------------------------------------------------------------------------------------------- calibration


def test_block_interval_is_calibrated_where_short_hac_over_rejects():
    """A persistent null: AR(1) regressor residual (phi 0.97) times a persistent, volatility-clustered null outcome.
    The short HAC (lag 7) rejects far above nominal; the block interval at the measured length must not."""
    from c2_battery_calibration import block_bootstrap_se, integrated_autocorr_length
    from c2_causal_dossier import hac_variance
    rng = np.random.default_rng(5)
    n, cells = 3000, 60
    rej_hac, rej_boot, lengths = 0, 0, []
    for _ in range(cells):
        e = rng.standard_normal(n)
        rx = np.zeros(n)
        for t in range(1, n):
            rx[t] = 0.97 * rx[t - 1] + e[t]
        ry = np.zeros(n)
        u = rng.standard_normal(n)
        for t in range(1, n):
            ry[t] = 0.9 * ry[t - 1] + u[t]
        denom = float(rx @ rx)
        theta = float(rx @ ry) / denom
        score = rx * (ry - theta * rx)
        se_hac = np.sqrt(hac_variance(score, 7) * n) / denom
        L = max(integrated_autocorr_length(rx, n // 20), integrated_autocorr_length(ry, n // 20), 7)
        lengths.append(L)
        se_boot = block_bootstrap_se(score, denom, L, 100, rng)
        rej_hac += abs(theta) > 1.96 * se_hac
        rej_boot += abs(theta) > 1.96 * se_boot
    assert rej_hac / cells > 0.10                      # the short HAC over-rejects under this null
    assert rej_boot / cells < rej_hac / cells            # the block interval rejects less
    assert rej_boot / cells <= 0.12                      # and sits near nominal (binomial noise on 60 cells)
    assert np.median(lengths) > 20


def test_calibration_aggregate_and_patch(confounded, tmp_path):
    from c2_battery_calibration import aggregate, calibrate_cell, patch_dossiers
    from c2_causal_dossier import dossier, study
    pop = confounded["pop"]
    c = calibrate_cell(pop, "x_treat", 1, boot=50)
    assert c["block_length"] >= 7 and set(c["real"]) == set(c["scrambled"])
    assert c["real"]["se_block_bootstrap_L"] > 0
    agg = aggregate([c])
    assert "battery_verdict" in agg and agg["cells"] == 1
    d = tmp_path / "dossiers"
    d.mkdir()
    doc = dossier(pop, study(pop, "x_treat", 1), revision="0123456789abcdef")
    (d / "x.json").write_text(json.dumps(doc))
    (d / "DOSSIER_INDEX.json").write_text(json.dumps({"dossiers": [{"feature": "x_treat", "horizon_bars": 1, "file": "x.json"}]}))
    assert patch_dossiers(d, [c], agg) == 1
    patched = json.loads((d / "x.json").read_text())
    assert patched["rung2"]["sensitivity"]["calibration_block_length"] == c["block_length"]
    assert patched["rung2"]["state"] == "NOT_IDENTIFIED"


# ---------------------------------------------------------------------------------------------------- real-label rejection


def test_trailing_scale_uses_only_rows_up_to_t():
    from c2_real_rejection import trailing_scale, past_return, seasonal_return
    r = np.abs(np.random.default_rng(1).standard_normal(500)) * 0.01
    base = trailing_scale(r)
    r2 = r.copy()
    r2[300:] += 5.0                                   # perturb the future of t = 299
    assert np.allclose(base[:300], trailing_scale(r2)[:300], equal_nan=True)
    assert np.isnan(base[58]) and np.isfinite(base[59])
    logc = np.cumsum(np.random.default_rng(2).standard_normal(500) * 0.01)
    for h in (1, 3, 6):
        pr, se = past_return(logc, h), seasonal_return(logc, h)
        t = 100
        assert np.isclose(pr[t], logc[t] - logc[t - h])              # window ends at t
        assert np.isclose(se[t], logc[t - 6 + h] - logc[t - 6])      # ends at t - 6 + h <= t
        l2 = logc.copy()
        l2[t + 1:] += 9.0
        assert np.isclose(past_return(l2, h)[t], pr[t]) and np.isclose(seasonal_return(l2, h)[t], se[t])


def test_seasonal_test_detects_a_planted_seasonal_dependence_and_not_its_absence(tmp_path):
    from c2_real_rejection import seasonal_cell
    rng = np.random.default_rng(9)
    rows = []
    for planted in (0.0, 0.6):
        d = tmp_path / f"s{planted}"
        d.mkdir()
        n = N_ROWS
        e = rng.standard_normal(n) * 0.01
        r = np.zeros(n)
        r[6:] = e[6:] + planted * e[:-6]                       # next return depends on the one 6 bars earlier
        close = 100 * np.exp(np.cumsum(r))
        times = np.arange(n, dtype=np.int64) * BAR + 1_600_000_000
        frame = pd.DataFrame({"DATE_TIME": pd.to_datetime(times, unit="s").strftime("%Y-%m-%d %H:%M:%S"), "typical_price": close, "OPEN": close,
                              "HIGH": close * 1.01, "LOW": close * 0.99, "CLOSE": close, "VOLUME": 1000.0,
                              "log_return_1": np.concatenate([[0.0], np.diff(np.log(close))]), "noise_a": rng.standard_normal(n), "noise_b": rng.standard_normal(n)})
        view = d / "view.csv"
        frame.to_csv(view, index=False)
        vsha = sha256_file(view)
        feats = ["noise_a", "noise_b"]
        (d / "manifest.json").write_text(json.dumps({"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "variant": "A", "resource": {"sha256": vsha},
                                                    "split": {"train": {"rows": [0, TRAIN_END]}}, "admissible_declaration_sha256": pop_mod.DECLARATION_SHA256,
                                                    "features": feats, "feature_count": 2, "manifest_sha256_canonical": "0" * 64}))
        ssha, _ = _split(d / "split.json", vsha, times, frame["log_return_1"].to_numpy())
        pop = bind_population(view, d / "manifest.json", d / "split.json", expect_view=vsha, expect_manifest=sha256_file(d / "manifest.json"), expect_split=ssha)
        rows.append(seasonal_cell(pop, 1, boot=100))
    null, planted = rows
    assert planted["seasonal_24h"]["reject"] is True and planted["seasonal_24h"]["sign_agreement"] >= 0.8
    assert planted["seasonal_24h"]["theta"] > 0
    assert abs(null["seasonal_24h"]["t"]) < abs(planted["seasonal_24h"]["t"])


def test_regime_cell_returns_the_paired_variants(confounded):
    from c2_real_rejection import regime_cell, summarize_regime
    c = regime_cell(confounded["pop"], "x_treat", 1, boot=50)
    assert {"base", "scaled", "regime_ctrl", "both"} <= set(c) and c["finding"] in {"SURVIVES_REGIME_CONDITIONING", "REGIME_ARTEFACT_CANDIDATE", "NOT_REJECTED_AT_BASE"}
    assert len(c["blocks_theta_base"]) == 5 and c["rows"] > 500
    agg = summarize_regime([c])
    assert agg["cells"] == 1 and agg["nominal"] == 0.05


# ---------------------------------------------------------------------------------------------------- interval rule, evidence split


def test_interval_rule_threshold_and_committed_eth_evidence():
    from c2_interval_rule import check, tolerance
    assert abs(tolerance(498) - 0.0695) < 5e-4
    base = REPO / "docs" / "audits" / "evidence" / "lane_c2_eth_20261001"
    cal = json.loads((base / "calibration" / "BATTERY_CALIBRATION.json").read_text())
    idx = json.loads((base / "dossiers" / "DOSSIER_INDEX.json").read_text())
    res = check(cal, idx)
    assert res["verdict"] == "CONTROLS_FAIL_AS_REQUIRED" and "block_bootstrap_L" in res["calibrated_methods"]
    assert "hac_lag_h6" not in res["calibrated_methods"]            # the uncalibrated interval is refused by the rule
    bad = json.loads(json.dumps(cal))
    bad["aggregate"]["scrambled_rate_block_bootstrap_L"] = 0.181
    assert check(bad, idx)["verdict"] == "BATTERY_SUSPECT"
    bad_idx = json.loads(json.dumps(idx))
    bad_idx["battery"]["future_shift_template_fired"] -= 1
    assert "FUTURE_SHIFT_TEMPLATE_DID_NOT_FIRE_IN_EVERY_CELL" in check(cal, bad_idx)["problems"]


def test_evidence_split_and_time_semantics_validate_and_stay_separate(confounded):
    pytest.importorskip("jsonschema")
    from c2_causal_dossier import dossier, study, validate
    from c2_dossier_evidence_split import derive
    pop = confounded["pop"]
    doc = dossier(pop, study(pop, "x_treat", 1), revision="0123456789abcdef")
    doc.update(derive(doc))
    schema = REPO / "docs" / "contracts" / "causal_dossier.v1.schema.json"
    assert validate(doc, schema) == []
    ec = doc["evidence_classes"]
    assert ec["observed_natural_interventions"]["state"] == "NOT_AVAILABLE"
    assert ec["paired_counterfactuals"]["state"] == "REPORTED_UNDER_DECLARED_MODEL" and doc["rung2"]["state"] == "NOT_IDENTIFIED"
    assert doc["time_semantics"]["availability_time"]["state"] == "UNDECLARED"
    assert doc["time_semantics"]["population"]["rows"] == doc["data_manifest"]["n_episodes"]
    broken = json.loads(json.dumps(doc))
    broken["evidence_classes"]["paired_counterfactuals"]["assumptions"] = []
    assert validate(broken, schema) != []                         # a reported class without assumptions is refused
    broken2 = json.loads(json.dumps(doc))
    del broken2["time_semantics"]["availability_time"]
    assert validate(broken2, schema) != []


# ---------------------------------------------------------------------------------------------------- population spec (EURUSD 1h binding)


def _fx_world(d: Path, seed=21):
    """Hourly OHLC with weekend gaps, no log_return_1 column, split declared by the manifest only (lane B's EURUSD shape)."""
    rng = np.random.default_rng(seed)
    n, step = 2600, 3600
    times = []
    t = 1_600_000_000
    for i in range(n):
        times.append(t)
        t += step if (i % 120) != 119 else step + 48 * 3600       # a weekend-like gap every 120 bars
    times = np.array(times, dtype=np.int64)
    r = 0.0004 * rng.standard_normal(n)
    close = 1.2 * np.exp(np.cumsum(r))
    frame = pd.DataFrame({"DATE_TIME": pd.to_datetime(times, unit="s").strftime("%Y-%m-%d %H:%M:%S"), "OPEN": np.r_[close[0], close[:-1]],
                          "LOW": close * 0.9995, "HIGH": close * 1.0005, "CLOSE": close})
    view = d / "fx.csv"
    frame.to_csv(view, index=False)
    vsha = sha256_file(view)
    doc = {"schema": "selected_feature_manifest.v1", "status": "FROZEN_DEVELOPMENT", "variant": "A_all_admissible_control",
           "resource": {"sha256": vsha}, "admissible_declaration_sha256": "6f8fed4197c0d1824969768b5bd69521da115bc7122f49ccd79f897f74e9ccf0",
           "features": ["OPEN", "LOW", "HIGH", "CLOSE"], "feature_count": 4, "manifest_sha256_canonical": "0" * 64,
           "split": {"declared_by": "coordinator ruling: 70/15/15 chronological", "train": {"rows": [0, 1800]}, "validation": {"rows": [1800, 2200]}, "test": {"rows": [2200, 2600]}}}
    (d / "fx_manifest.json").write_text(json.dumps(doc))
    return view, d / "fx_manifest.json", vsha, sha256_file(d / "fx_manifest.json")


def test_population_spec_binds_a_manifest_split_population_and_refuses_digest_mismatch(tmp_path):
    from c2_eth_population import PopulationSpec, EURUSD_1H_SPEC, ETH_4H_SPEC
    view, man, vsha, msha = _fx_world(tmp_path)
    spec = PopulationSpec(name="FX_test", view_sha256=vsha, manifest_file_sha256=msha, declaration_sha256=EURUSD_1H_SPEC.declaration_sha256,
                          dataset_id="test.fx", view_path="fx.csv", view_commit="0" * 40, bar_seconds=3600, split_mode="manifest", window=1, hmax=6)
    pop = bind_population(view, man, spec=spec)
    assert pop.train_rows == (0, 1800) and pop.bar_seconds == 3600 and "log_return_1" in pop.frame.columns
    assert pop.origins.max() <= 1800 - 1 - 6 and pop.gap_excluded_windows > 0
    assert pop.bindings["split"]["test_status"] == "PROTECTED_NEVER_READ" and pop.bindings["population_spec"] == "FX_test"
    y1 = pop.raw_log_return(1)                                    # a weekend gap row has no 1 h label
    gap_row = 119
    assert np.isnan(y1[gap_row]) and np.isfinite(y1[gap_row - 1])
    z = pop.m07_target(2)
    assert np.isclose(pop.z_to_log_return(z[10], 2), np.log(pop.frame["CLOSE"][12] / pop.frame["CLOSE"][10]), atol=1e-9)
    # the same machinery that reads ETH reads this population: blocks and the calibrated interval run on it
    splits = blocked_splits(pop.origins, 2)
    assert len(splits) == 5
    from dataclasses import replace
    with pytest.raises(PopulationRefusal, match="VIEW_SHA256_MISMATCH"):
        bind_population(view, man, spec=replace(spec, view_sha256="0" * 64))
    with pytest.raises(PopulationRefusal, match="MANIFEST_SHA256_MISMATCH"):
        bind_population(view, man, spec=replace(spec, manifest_file_sha256="0" * 64))
    with pytest.raises(PopulationRefusal, match="UNKNOWN_SPLIT_MODE"):
        bind_population(view, man, spec=replace(spec, split_mode="nope"))
    assert EURUSD_1H_SPEC.view_sha256.startswith("72b8271d") and ETH_4H_SPEC.view_sha256.startswith("1b447c66")


def test_population_measure_runs_on_the_fx_shaped_population_and_applies_the_rule(tmp_path):
    from dataclasses import replace
    from c2_eth_population import EURUSD_1H_SPEC, PopulationSpec
    from c2_population_measure import build_check_documents, measure_cell, references
    from c2_interval_rule import check
    view, man, vsha, msha = _fx_world(tmp_path)
    spec = PopulationSpec(name="FX_test", view_sha256=vsha, manifest_file_sha256=msha, declaration_sha256=EURUSD_1H_SPEC.declaration_sha256, dataset_id="test.fx",
                          view_path="fx.csv", view_commit="0" * 40, bar_seconds=3600, split_mode="manifest", window=1, hmax=6)
    pop = bind_population(view, man, spec=spec)
    ref = references(pop, [1, 2], tmp_path / "o")
    assert {"naive_zero", "naive_train_mean", "naive_seasonal_24h", "full|selected"} <= set(ref[1])
    assert ref[1]["naive_seasonal_24h"]["n_eval_total"] == ref[1]["naive_zero"]["n_eval_total"]     # identical rows
    c = measure_cell(pop, "CLOSE", 1, scrambles=3, boot=30)
    assert c["state"] in {"MEASURED", "TREATMENT_COLLINEAR_WITH_ALL_CONTROLS"}
    cells = [measure_cell(pop, f, h, scrambles=3, boot=30) for f in ("OPEN", "CLOSE") for h in (1, 2)]
    measured = [x for x in cells if x["state"] == "MEASURED"]
    if measured:
        cal, idx = build_check_documents(cells, 3)
        res = check(cal, idx)
        assert res["verdict"] in {"CONTROLS_FAIL_AS_REQUIRED", "BATTERY_SUSPECT"} and res["cells"] == 3 * len(measured)


def test_variant_a_price_levels_are_not_estimable_and_range_family_is(tmp_path):
    from c2_eth_population import EURUSD_1H_RANGE_SPEC, EURUSD_1H_SPEC, PopulationSpec, add_range_features
    from c2_population_measure import build_check_documents, measure_cell
    view, man, vsha, msha = _fx_world(tmp_path)
    spec_a = PopulationSpec(name="FX_A", view_sha256=vsha, manifest_file_sha256=msha, declaration_sha256=EURUSD_1H_SPEC.declaration_sha256, dataset_id="t", view_path="fx.csv",
                            view_commit="0" * 40, bar_seconds=3600, split_mode="manifest", window=1, hmax=6)
    pop_a = bind_population(view, man, spec=spec_a)
    cells = [measure_cell(pop_a, f, 1, scrambles=2, boot=20) for f in pop_a.features]
    assert all(c["state"] == "TREATMENT_COLLINEAR_WITH_ALL_CONTROLS" for c in cells)     # four price levels are one series
    assert build_check_documents(cells, 2) == (None, None)
    doc = json.loads(man.read_text())
    doc["features"] = ["log_high_low", "close_location", "log_close_open"]
    doc["feature_count"] = 3
    doc.pop("admissible_declaration_sha256")
    man2 = tmp_path / "fx_range_manifest.json"
    man2.write_text(json.dumps(doc))
    spec_r = PopulationSpec(name="FX_R", view_sha256=vsha, manifest_file_sha256=sha256_file(man2), declaration_sha256=None, dataset_id="t", view_path="fx.csv",
                            view_commit="0" * 40, bar_seconds=3600, split_mode="manifest", window=1, hmax=6, derived="range_v1")
    pop_r = bind_population(view, man2, spec=spec_r)
    assert pop_r.bindings["manifest"]["derived"] == "range_v1" and pop_r.bindings["manifest"]["derived_close_location_filled_rows"] >= 0
    f = pop_r.frame
    assert np.allclose(f["log_high_low"], np.log(f["HIGH"] / f["LOW"])) and np.isfinite(f["close_location"]).all()
    c = measure_cell(pop_r, "log_high_low", 1, scrambles=2, boot=20)
    assert c["state"] == "MEASURED" and len(c["scrambled"]) == 2 and "future_shift_template_fired" in c
    assert EURUSD_1H_RANGE_SPEC.manifest_file_sha256.startswith("2985f589")


def test_split_variant_selects_the_declared_block_and_never_the_reserve(tmp_path):
    from c2_eth_population import EURUSD_1H_SPEC, EURUSD_LAKE_A_S2_SPEC, EURUSD_LAKE_A_S1_SPEC, PopulationSpec
    from c2_population_measure import measure_cell
    view, man, vsha, msha = _fx_world(tmp_path)
    doc = json.loads(man.read_text())
    doc.pop("split")
    doc["split_variants"] = {"S1": {"train": {"rows": [0, 1800]}, "validation": {"rows": [1800, 2200]}, "test": {"rows": [2200, 2600]}},
                             "S2": {"train": {"rows": [0, 1500]}, "validation": {"rows": [1500, 2000]}, "reserve": {"rows": [2000, 2600], "status": "PROSPECTIVE_CONFIRMATION_PROTECTED"}}}
    m2 = tmp_path / "lake_manifest.json"
    m2.write_text(json.dumps(doc))
    base = dict(view_sha256=vsha, manifest_file_sha256=sha256_file(m2), declaration_sha256=EURUSD_1H_SPEC.declaration_sha256, dataset_id="t", view_path="x", view_commit="0" * 40,
                bar_seconds=3600, split_mode="manifest", window=1, hmax=6)
    p1 = bind_population(view, m2, spec=PopulationSpec(name="S1", split_variant="S1", **base))
    p2 = bind_population(view, m2, spec=PopulationSpec(name="S2", split_variant="S2", **base))
    assert p1.train_rows == (0, 1800) and p2.train_rows == (0, 1500)
    assert p2.bindings["split"]["test_rows"] == [2000, 2600] and p2.origins.max() < 1500
    assert EURUSD_LAKE_A_S2_SPEC.split_variant == "S2_prospective_reserve" and EURUSD_LAKE_A_S1_SPEC.view_sha256.startswith("ab0ada28")
    # per-block sign stability is reported for a measured cell
    doc2 = json.loads(m2.read_text())
    doc2["features"] = ["log_high_low", "close_location", "log_close_open"]; doc2["feature_count"] = 3
    m3 = tmp_path / "lake_range.json"
    m3.write_text(json.dumps(doc2))
    pr = bind_population(view, m3, spec=PopulationSpec(name="R", split_variant="S1", derived="range_v1", **{**base, "manifest_file_sha256": sha256_file(m3)}))
    c = measure_cell(pr, "log_high_low", 1, scrambles=2, boot=20)
    assert len(c["real"]["blocks_theta"]) == 5 and 0.0 <= c["real"]["sign_agreement"] <= 1.0 and isinstance(c["real"]["sign_stable_4of5"], bool)


# ---------------------------------------------------------------------------------------------------- paired loss inference


def test_paired_inference_detects_a_planted_margin_and_not_equality(tmp_path):
    from c2_paired_loss_inference import analyse, block_bootstrap_mean, paired_row
    rng = np.random.default_rng(3)
    d_better = -0.01 + 0.2 * rng.standard_normal(20000)          # model better by 0.01 against noise 0.2
    d_equal = 0.2 * rng.standard_normal(20000)
    b = paired_row(d_better, 12, 400, np.random.default_rng(1))
    e = paired_row(d_equal, 12, 400, np.random.default_rng(1))
    assert b["excludes_zero"] and b["side"] == "MODEL_BETTER" and b["mean_diff"] < 0
    assert not e["excludes_zero"] and e["side"] == "INCLUDES_ZERO"
    # persistence: autocorrelated differences widen the interval at the measured block length relative to block length 1
    ar = np.zeros(20000)
    u = rng.standard_normal(20000)
    for t in range(1, 20000):
        ar[t] = 0.9 * ar[t - 1] + u[t]
    se_short = block_bootstrap_mean(ar, 1, 300, np.random.default_rng(2))[0]
    se_long = block_bootstrap_mean(ar, 60, 300, np.random.default_rng(2))[0]
    assert se_long > 2 * se_short


def test_paired_inference_refuses_evidence_it_cannot_reproduce(tmp_path):
    from c2_eth_population import EURUSD_1H_SPEC, PopulationSpec
    from c2_paired_loss_inference import analyse, validation_targets
    view, man, vsha, msha = _fx_world(tmp_path)
    spec = PopulationSpec(name="FX", view_sha256=vsha, manifest_file_sha256=msha, declaration_sha256=EURUSD_1H_SPEC.declaration_sha256, dataset_id="t", view_path="x",
                          view_commit="0" * 40, bar_seconds=3600, split_mode="manifest", window=1, hmax=6)
    pop = bind_population(view, man, spec=spec)
    rows = np.arange(1810, 1900)
    rows = rows[(pop.times[rows + 2] - pop.times[rows]) == 7200]
    y1 = validation_targets(pop.frame["log_return_1"].to_numpy(), pop.mu, pop.sigma, pop.times, rows, 1, 3600)
    y2 = validation_targets(pop.frame["log_return_1"].to_numpy(), pop.mu, pop.sigma, pop.times, rows, 2, 3600)
    assert np.allclose(y2 - y1, ((pop.frame["log_return_1"].to_numpy()[rows + 2] - pop.mu) / pop.sigma))
    preds = {1: np.zeros(len(rows)), 2: np.zeros(len(rows))}
    zero = {h: float(np.mean(np.abs((-h * pop.mu / pop.sigma) - y))) for h, y in ((1, y1), (2, y2))}
    good = {"per_horizon": [{"horizon": h, "model_MAE": float(np.mean(np.abs(preds[h] - y))), "naive_MAE": zero[h] - 0.001} for h, y in ((1, y1), (2, y2))],
            "naives": {"per_naive": {"zero_return": {str(h): {"MAE": zero[h]} for h in (1, 2)}}}}
    res = analyse(pop, rows, preds, good, boot=50)
    assert set(res["horizons"]) == {1, 2} and res["checks"][1]["agrees"] and "combined_equal_weight" in res
    bad = json.loads(json.dumps(good))
    bad["per_horizon"][0]["model_MAE"] += 0.01
    with pytest.raises(ValueError, match="EVIDENCE_NOT_REPRODUCED"):
        analyse(pop, rows, preds, bad, boot=50)
    # a gap inside the label span is refused
    with pytest.raises(ValueError, match="IRREGULAR_LABEL_SPAN"):
        validation_targets(pop.frame["log_return_1"].to_numpy(), pop.mu, pop.sigma, pop.times, np.array([118]), 3, 3600)


def test_m07_fin_scaler_excludes_row0_and_flat_bars():
    from c2_paired_loss_inference import m07_fin_scaler
    n = 400
    rng = np.random.default_rng(4)
    close = 1.1 * np.exp(np.cumsum(0.001 * rng.standard_normal(n)))
    high, low = close * 1.001, close * 0.999
    low[10] = high[10] = close[10]                       # a flat bar: close_location NaN
    frame = pd.DataFrame({"CLOSE": close, "HIGH": high, "LOW": low})
    mu, sigma, used = m07_fin_scaler(frame, 300)
    lr = np.log(close[1:] / close[:-1])
    keep = np.ones(299, bool)
    keep[9] = False                                      # return at row 10 belongs to the flat bar (index 9 in lr)
    assert used == 298 and np.isclose(mu, lr[:299][keep].mean()) and np.isclose(sigma, lr[:299][keep].std())
