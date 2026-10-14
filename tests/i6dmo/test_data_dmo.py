"""I6-DMO data lane acceptance tests: DMO02, DMO03, DMO05, DMO06, DMO08, DMO09, DMO16.

Every dataset here is a SYNTHETIC fixture written by ``tests/i6dmo/_synth.py``
(no real prices, no 2025 bar). Nothing is trained.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6n_contracts import eligibility as c1

from tools.i6dmo import data, naive, provenance

UTC = timezone.utc
H = 3600
HORIZONS = (2, 3, 4, 5, 6, 12)


def iso(t: int) -> str:
    return datetime.fromtimestamp(int(t), UTC).isoformat()


@pytest.fixture(scope="module")
def design():
    return _synth.load_design()


@pytest.fixture(scope="module")
def full(tmp_path_factory, design):
    """Four TRAIN years + 2024 with a few holes; built once per module."""
    root = tmp_path_factory.mktemp("full")
    missing = {_synth.epoch("2022-06-15T13:00:00+00:00"),   # a target/input hole
               _synth.epoch("2024-03-06T15:00:00+00:00")}
    nonfinite = [(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)]
    src = _synth.make_input(root / "input", missing=missing, nonfinite=nonfinite)
    bundle = data.load_input(src, design)
    table = data.build_origin_table(bundle, design)
    return {"src": src, "bundle": bundle, "table": table, "root": root,
            "missing": missing, "nonfinite": nonfinite}


# --------------------------------------------------------------------------- DMO02

def test_dmo02_window_is_13_exact_timestamps_including_t(full, design):
    spec = data.WindowSpec.from_design(design)
    assert spec.offsets_hours() == tuple(range(-24, 1, 2))
    assert len(spec.offsets_hours()) == 13 and spec.offsets_hours()[-1] == 0
    arrays = data.materialize(full["bundle"], full["table"], "FIT")
    ts, x = full["bundle"].ts, full["bundle"].x
    pos = {int(v): i for i, v in enumerate(ts)}
    for r in (0, len(arrays.origin_ts) // 2, len(arrays.origin_ts) - 1):
        t = int(arrays.origin_ts[r])
        stamps = [t + o * H for o in spec.offsets_hours()]
        assert stamps[-1] == t
        expected = np.stack([x[pos[s]] for s in stamps])
        np.testing.assert_array_equal(arrays.X[r], expected)
        assert arrays.row_ids[r] == f"EURUSD|{iso(t)}"


def test_dmo02_perturbing_t_plus_1_and_later_changes_no_input(full, design, tmp_path):
    arrays = data.materialize(full["bundle"], full["table"], "FIT")
    r = len(arrays.origin_ts) // 3
    t = int(arrays.origin_ts[r])
    src = _synth.make_input(tmp_path / "p", missing=full["missing"], nonfinite=full["nonfinite"])
    with np.load(src / "INPUT.npz") as z:
        ts, x, price = z["ts"], z["x"].copy(), z["price"].copy()
    after = ts > t
    assert int(ts[np.argmax(after)]) == t + H          # t+1 bar exists and is perturbed
    x[after] += 1000.0
    price[after] += 1.0
    _synth.rewrite_input(src, x=x, price=price)
    b2 = data.load_input(src, design)
    t2 = data.build_origin_table(b2, design)
    a2 = data.materialize(b2, t2, "FIT")
    r2 = list(a2.row_ids).index(arrays.row_ids[r])
    np.testing.assert_array_equal(a2.X[r2], arrays.X[r])
    assert a2.price_origin[r2] == arrays.price_origin[r]


def test_dmo02_absent_slot_makes_origin_input_unavailable_never_filled(full, design):
    hole = _synth.epoch("2022-06-15T13:00:00+00:00")
    table = full["table"]
    affected = [t for t in range(hole, hole + 25 * H, H) if t in table.index]
    assert affected, "fixture must contain origins whose window covers the hole"
    for t in affected:
        rec = table.record(t)
        if (t - hole) % (2 * H) == 0 and t - hole <= 24 * H:
            assert rec.input_ok is False
            assert rec.input_reason == "INPUT_SLOT_NOT_OBSERVED"
    arrays = data.materialize(full["bundle"], table, "FIT")
    assert not set(int(v) for v in arrays.origin_ts) & {t for t in affected
                                                        if (t - hole) % (2 * H) == 0}
    assert table.counts["FIT"]["origins"]["INPUT_SLOT_NOT_OBSERVED"] >= 1
    assert table.counts["FIT"]["origins"]["INPUT_VALUE_NOT_FINITE"] >= 1


# --------------------------------------------------------------------------- DMO03

def test_dmo03_exact_h3_h5_targets_and_hourly_origins(full, design):
    arrays = data.materialize(full["bundle"], full["table"], "VALIDATION")
    ts, price = full["bundle"].ts, full["bundle"].price
    pos = {int(v): i for i, v in enumerate(ts)}
    for j, h in enumerate(HORIZONS):
        valid = np.flatnonzero(arrays.mask[:, j])
        assert len(valid) > 0
        for r in valid[:: max(1, len(valid) // 50)]:
            q = int(arrays.origin_ts[r]) + h * H
            assert arrays.y_price[r, j] == price[pos[q]]
        assert np.isnan(arrays.y_price[~arrays.mask[:, j], j]).all()
    o = np.sort(arrays.origin_ts.astype(np.int64))
    steps = np.diff(o)
    assert (steps == H).sum() > 0.8 * len(steps)        # hourly, not thinned to 2 h
    odd = (o // H) % 2 == 1
    assert odd.any() and (~odd).any()


def test_dmo03_missing_endpoint_is_not_bridged_by_next_row(full, design):
    hole = _synth.epoch("2024-03-06T15:00:00+00:00")
    table = full["table"]
    for h in (3, 5):
        rec = table.record(hole - h * H)
        j = HORIZONS.index(h)
        assert rec.label_ok[j] is False
        assert rec.label_reasons[j][0] == "TARGET_NOT_OBSERVED"
        # the other heads of the same origin are retained when valid
        assert any(rec.label_ok[k] for k in range(len(HORIZONS)) if k != j)


# --------------------------------------------------------------------------- DMO05

def test_dmo05_masks_equal_direct_c1_calls_including_dst_weeks(full, design):
    bundle, table = full["bundle"], full["table"]
    ref, weeks = c1.load_calendar(json.loads((full["src"] / "CALENDAR.json").read_text()))
    observed = c1.ObservedTargets({datetime.fromtimestamp(int(v), UTC):
                                   datetime.fromtimestamp(int(v), UTC) for v in bundle.ts})
    splits = data.split_bounds(design)
    chosen = [w for w in weeks if w.week_id in ("2021-03-07", "2021-03-14", "2021-10-31",
                                                "2021-11-07", "2024-03-10", "2024-11-03")]
    assert len(chosen) == 6
    checked = 0
    for w in chosen:
        for t in table.origins_in_week(w.week_id):
            rec = table.record(t)
            t_dt = datetime.fromtimestamp(t, UTC)
            split = next(s for s in splits if s.start <= t_dt < s.end)
            for role, got in ((c1.Role.MANAGEMENT, rec.label_ok), (c1.Role.ENTRY, rec.entry_ok)):
                ref_heads = c1.evaluate_heads(t_dt, HORIZONS, role, w,
                                              input_available=rec.input_ok,
                                              observed_targets=observed, split=split)
                assert tuple(d.label_eligible for d in ref_heads) == tuple(got)
                checked += 1
    assert checked > 100


def test_dmo05_business_rules(full, design):
    table = full["table"]
    week = json.loads((full["src"] / "CALENDAR.json").read_text())["market_calendar"]["weeks"]
    w = next(x for x in week if x["week_id"] == "2022-05-15")
    o, c, k = (_synth.epoch(w[f]) for f in ("market_open", "market_close", "entry_cutoff"))
    a, b = o + 12 * H, c - 12 * H
    # before A: not an operational origin for any head
    rec = table.record(a - H)
    assert not any(rec.label_ok)
    assert rec.label_reasons[0][0] == "OPERATIONALLY_INELIGIBLE"
    # Thursday: after the entry cutoff ENTRY is false while MANAGEMENT is true
    rec = table.record(k + 2 * H)
    assert rec.label_ok[0] is True and rec.entry_ok[0] is False
    rec = table.record(k - 3 * H)
    assert rec.entry_ok[0] is True
    # force flat: target at or after B is masked, strictly before is kept
    rec = table.record(b - 12 * H)
    assert rec.label_ok[HORIZONS.index(12)] is False
    rec = table.record(b - 13 * H)
    assert rec.label_ok[HORIZONS.index(12)] is True
    # weekend: a Friday origin never targets next week's bars
    rec = table.record(b - 2 * H)
    assert rec.label_ok[HORIZONS.index(12)] is False and rec.label_ok[0] is False


def test_dmo05_split_boundaries_purge_cross_boundary_labels(full, design):
    splits = {s.split_id: s for s in data.split_bounds(design)}
    assert splits["FIT"].end == splits["INNER"].start == datetime(2023, 11, 6, tzinfo=UTC)
    assert splits["FIT"].label_available_by == splits["INNER"].start
    assert splits["INNER"].end == splits["VALIDATION"].start == datetime(2024, 1, 1, tzinfo=UTC)
    assert splits["VALIDATION"].end == datetime(2025, 1, 1, tzinfo=UTC)
    # a non-contract boundary placed mid-week shows the purge (test-only override)
    moved = copy.deepcopy(design)
    moved["splits"]["validation"]["start"] = "2024-01-03T12:00:00Z"
    moved["splits"]["train"]["end"] = "2024-01-03T12:00:00Z"
    with pytest.raises(data.DataContractError) as err:
        data.split_bounds(moved)
    assert err.value.code == "SPLIT_NOT_CONTRACT"
    tb = data.build_origin_table(full["bundle"], moved, _allow_noncontract_splits=True)
    boundary = _synth.epoch("2024-01-03T12:00:00+00:00")
    rec = tb.record(boundary - 3 * H)
    assert rec.split == "INNER"
    assert rec.label_ok[HORIZONS.index(2)] is True
    assert rec.label_ok[HORIZONS.index(4)] is False
    assert "TARGET_AT_OR_AFTER_SPLIT_END" in rec.label_reasons[HORIZONS.index(4)]


# --------------------------------------------------------------------------- DMO06

def _perturb_from(src, start_epoch, tmp, design):
    with np.load(src / "INPUT.npz") as z:
        ts, x, price = z["ts"], z["x"].copy(), z["price"].copy()
    late = ts >= start_epoch
    x[late] = x[late] * -3.0 + 7.0
    price[late] = price[late] + 0.5
    dst = tmp
    dst.mkdir()
    for name in ("CLOCK.json", "CALENDAR.json"):
        (dst / name).write_text((src / name).read_text())
    np.savez(dst / "INPUT.npz", ts=ts, x=x, price=price)
    _synth.rewrite_input(dst)
    b = data.load_input(dst, design)
    return b, data.build_origin_table(b, design)


def _same(a, b):
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.entry_mask, b.entry_mask)
    np.testing.assert_array_equal(np.nan_to_num(a.y_price, nan=-1), np.nan_to_num(b.y_price, nan=-1))
    assert list(a.row_ids) == list(b.row_ids)


def test_dmo06_outer_and_inner_perturbation_leave_fit_preprocessing_unchanged(full, design, tmp_path):
    base_fit = data.materialize(full["bundle"], full["table"], "FIT")
    base_inner = data.materialize(full["bundle"], full["table"], "INNER")
    base_stats = data.fit_statistics(full["bundle"], full["table"])
    b, t = _perturb_from(full["src"], _synth.epoch("2023-11-06T00:00:00+00:00"),
                         tmp_path / "a", design)
    _same(base_fit, data.materialize(b, t, "FIT"))
    assert data.fit_statistics(b, t) == base_stats
    b, t = _perturb_from(full["src"], _synth.epoch("2024-01-01T00:00:00+00:00"),
                         tmp_path / "b", design)
    _same(base_fit, data.materialize(b, t, "FIT"))
    _same(base_inner, data.materialize(b, t, "INNER"))
    assert data.fit_statistics(b, t) == base_stats
    # positive control: perturbing inside FIT does change the fit statistics
    b, t = _perturb_from(full["src"], _synth.epoch("2023-06-01T00:00:00+00:00"),
                         tmp_path / "c", design)
    assert data.fit_statistics(b, t) != base_stats


def test_dmo06_fit_statistics_zero_scale_is_typed(full, design, tmp_path):
    stats = data.fit_statistics(full["bundle"], full["table"])
    assert stats["status"] == "OK"
    assert set(stats["naive_mae_fit"]) == {str(h) for h in HORIZONS}
    src = _synth.make_input(tmp_path / "z")
    with np.load(src / "INPUT.npz") as z:
        x = z["x"].copy()
    x[:, 3] = 0.25
    _synth.rewrite_input(src, x=x)
    b = data.load_input(src, design)
    zs = data.fit_statistics(b, data.build_origin_table(b, design))
    assert zs["status"] == "ZERO_SCALE"
    assert zs["zero_scale_features"] == ["fred.rates.dgs7.logret_5d"]
    assert zs["feature_std"]["fred.rates.dgs7.logret_5d"] == 0.0


# --------------------------------------------------------------------------- DMO08

def _predictions(arrays, shift=2e-4):
    pred = np.where(arrays.mask, arrays.price_origin[:, None] + shift, np.nan)
    pred[1::3] = arrays.price_origin[1::3, None] - 3 * shift
    return np.where(arrays.mask, pred, np.nan)


def test_dmo08_mae_and_naive_identical_rows_and_manual_replay(full, design):
    arrays = data.materialize(full["bundle"], full["table"], "VALIDATION")
    pred = _predictions(arrays)
    report = naive.paired_metrics(arrays, pred, design, input_dir=full["src"],
                                  checkpoint="synthetic-checkpoint",
                                  sealed_at="2026-10-14T00:00:00+00:00")
    assert report["agreement"]["status"] == "AGREE"
    replay = naive.naive_replay(full["src"], arrays, HORIZONS)
    for j, h in enumerate(HORIZONS):
        g = report["per_horizon"][str(h)]
        m = arrays.mask[:, j]
        q = arrays.origin_ts[m].astype(np.int64) + h * H
        assert g["n"] == int(m.sum()) == replay[str(h)]["n"]
        manual_model = math.fsum(np.abs(pred[m, j] - arrays.y_price[m, j])) / m.sum()
        manual_naive = math.fsum(np.abs(arrays.y_price[m, j] - arrays.price_origin[m])) / m.sum()
        assert g["mae"] == pytest.approx(manual_model, rel=1e-12)
        assert g["naive_mae"] == pytest.approx(manual_naive, rel=1e-12)
        assert replay[str(h)]["mae"] == pytest.approx(manual_naive, rel=1e-12)
        assert g["relative_error"] == pytest.approx((manual_model - manual_naive) / manual_naive,
                                                    rel=1e-9)
        assert g["row_ids_sha256"] == replay[str(h)]["row_ids_sha256"]
        assert (q - arrays.origin_ts[m]).tolist() == [h * H] * int(m.sum())
    assert report["c2"]["state"] == "COMPLETE"


def test_dmo08_negative_control_replay_detects_tampered_target(full, design, tmp_path):
    arrays = data.materialize(full["bundle"], full["table"], "VALIDATION")
    pred = _predictions(arrays)
    src = tmp_path / "t"
    src.mkdir()
    for name in ("CLOCK.json", "CALENDAR.json", "INPUT.npz"):
        (src / name).write_bytes((full["src"] / name).read_bytes())
    with np.load(src / "INPUT.npz") as z:
        ts, price = z["ts"], z["price"].copy()
    j = HORIZONS.index(5)
    r = int(np.flatnonzero(arrays.mask[:, j])[10])
    price[np.searchsorted(ts, int(arrays.origin_ts[r]) + 5 * H)] += 0.01
    _synth.rewrite_input(src, price=price)
    with pytest.raises(naive.NaiveReplayMismatch):
        naive.paired_metrics(arrays, pred, design, input_dir=src,
                             checkpoint="synthetic-checkpoint",
                             sealed_at="2026-10-14T00:00:00+00:00")


def test_dmo08_missing_prediction_is_failure_not_dropped(full, design):
    arrays = data.materialize(full["bundle"], full["table"], "VALIDATION")
    pred = _predictions(arrays)
    j = HORIZONS.index(2)
    r = int(np.flatnonzero(arrays.mask[:, j])[0])
    pred[r, j] = np.nan
    report = naive.paired_metrics(arrays, pred, design, input_dir=full["src"],
                                  checkpoint="synthetic-checkpoint",
                                  sealed_at="2026-10-14T00:00:00+00:00")
    g = report["per_horizon"]["2"]
    assert g["state"] == "INCOMPLETE" and g["n_failed"] == 1
    assert g["n"] == int(arrays.mask[:, j].sum()) - 1


# --------------------------------------------------------------------------- DMO09

def test_dmo09_weekly_sums_counts_and_explicit_absences():
    w = ["W1", "W1", "W1", "W1", "W2"]
    err = np.array([[1.0, np.nan], [1.0, np.nan], [1.0, np.nan], [1.0, np.nan], [9.0, np.nan]])
    nerr = np.array([[2.0, np.nan], [2.0, np.nan], [2.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    mask = np.array([[True, False]] * 5)
    out = naive.weekly_diagnostics(w, err, nerr, mask, (2, 3), expected_weeks=["W0", "W1", "W2"])
    h2 = out["per_horizon"]["2"]
    assert h2["annual"]["n"] == 5 and h2["annual"]["sum_abs_error"] == 13.0
    assert h2["annual"]["mae"] == 13.0 / 5
    assert h2["annual"]["mae"] != (1.0 + 9.0) / 2          # not a mean of weekly MAEs
    assert h2["weeks"]["W0"] == {"status": "ABSENT", "n": 0}
    assert h2["weeks"]["W1"]["n"] == 4 and h2["weeks"]["W2"]["n"] == 1
    assert h2["unweighted_weekly_mean_relative_error"]["label"] == "DIAGNOSTIC_UNWEIGHTED"
    assert out["per_horizon"]["3"]["annual"] == {"status": "NO_ROWS", "n": 0}
    with pytest.raises(ValueError):
        naive.weekly_diagnostics(["W9"], err[:1], nerr[:1], mask[:1], (2, 3), expected_weeks=["W1"])


def test_dmo09_weekly_from_real_shapes(full, design):
    arrays = data.materialize(full["bundle"], full["table"], "VALIDATION")
    pred = _predictions(arrays)
    report = naive.paired_metrics(arrays, pred, design, input_dir=full["src"],
                                  checkpoint="synthetic-checkpoint",
                                  sealed_at="2026-10-14T00:00:00+00:00")
    weekly = report["weekly"]
    for h in HORIZONS:
        ann = weekly["per_horizon"][str(h)]["annual"]
        assert ann["n"] == report["per_horizon"][str(h)]["n"]
        assert ann["sum_abs_error"] == pytest.approx(
            report["per_horizon"][str(h)]["mae"] * ann["n"], rel=1e-12)
    expected = full["table"].weeks_in_split("VALIDATION")
    assert set(weekly["per_horizon"]["2"]["weeks"]) == set(expected)


# --------------------------------------------------------------------------- DMO16

def test_dmo16_four_calendar_years_and_all_validation_origins(full, design, tmp_path):
    table = full["table"]
    fit = data.materialize(full["bundle"], table, "FIT")
    inner = data.materialize(full["bundle"], table, "INNER")
    val = data.materialize(full["bundle"], table, "VALIDATION")
    years = {datetime.fromtimestamp(int(t), UTC).year for t in np.concatenate([fit.origin_ts,
                                                                             inner.origin_ts])}
    assert years == {2020, 2021, 2022, 2023}
    assert datetime.fromtimestamp(int(fit.origin_ts.min()), UTC) < datetime(2020, 1, 8, tzinfo=UTC)
    assert datetime.fromtimestamp(int(inner.origin_ts.max()), UTC) > datetime(2023, 12, 28, tzinfo=UTC)
    assert datetime.fromtimestamp(int(inner.origin_ts.min()), UTC) >= datetime(2023, 11, 6, tzinfo=UTC)
    # independent recount of every 2024 origin with an available input and >=1 valid head
    cal = json.loads((full["src"] / "CALENDAR.json").read_text())
    ref, weeks = c1.load_calendar(cal)
    ts = full["bundle"].ts
    have = set(int(v) for v in ts)
    xbad = set(int(v) for v in ts[~np.isfinite(full["bundle"].x).all(axis=1)])
    expected = 0
    v0, v1 = _synth.epoch("2024-01-01T00:00:00+00:00"), _synth.epoch("2025-01-01T00:00:00+00:00")
    for w in weeks:
        a, b = int(w.operational_start.timestamp()), int(w.force_flat.timestamp())
        for t in range(a, b, H):
            if not (v0 <= t < v1) or t not in have:
                continue
            slots = [t - k * H for k in range(0, 25, 2)]
            if any(s not in have or s in xbad for s in slots):
                continue
            if any(t + h * H < min(b, v1) and t + h * H in have for h in HORIZONS):
                expected += 1
    assert len(val.origin_ts) == expected
    assert table.counts["VALIDATION"]["rows_materialized"] == expected


def test_dmo16_no_sampling_or_shortening_knobs(full, design, tmp_path):
    for fn in (data.build_frozen_bundle, data.build_origin_table, data.materialize, data.load_input):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"max_rows", "sample", "sample_fraction", "max_steps_train",
                             "max_steps_test", "limit", "stride", "epochs"}
    bad = copy.deepcopy(design)
    bad["max_steps_train"] = 300
    with pytest.raises(data.DataContractError) as err:
        data.load_input(full["src"], bad)
    assert err.value.code == "UNKNOWN_DESIGN_FIELD"
    short = copy.deepcopy(design)
    short["splits"]["train"]["start"] = "2022-01-01T00:00:00Z"
    with pytest.raises(data.DataContractError) as err:
        data.load_input(full["src"], short)
    assert err.value.code == "SPLIT_NOT_CONTRACT"
    thin = copy.deepcopy(design)
    thin["input"]["issuance_hours"] = 2
    with pytest.raises(data.DataContractError) as err:
        data.load_input(full["src"], thin)
    assert err.value.code == "WINDOW_NOT_CONTRACT"


def test_dmo16_incomplete_coverage_is_refused_not_shortened(design, tmp_path):
    src = _synth.make_input(tmp_path / "short", data_end=_synth.epoch("2024-06-01T00:00:00+00:00"))
    with pytest.raises(data.DataContractError) as err:
        data.load_input(src, design)
    assert err.value.code == "COVERAGE_INCOMPLETE"
    late = _synth.make_input(tmp_path / "late")
    with np.load(late / "INPUT.npz") as z:
        keep = z["ts"] >= _synth.epoch("2020-03-01T00:00:00+00:00")
        _synth.rewrite_input(late, ts=z["ts"][keep], x=z["x"][keep], price=z["price"][keep])
    with pytest.raises(data.DataContractError) as err:
        data.load_input(late, design)
    assert err.value.code == "COVERAGE_INCOMPLETE"
    nocal = _synth.make_input(tmp_path / "nocal", first_sunday=datetime(2020, 3, 1, tzinfo=UTC))
    with pytest.raises(data.DataContractError) as err:
        data.load_input(nocal, design)
    assert err.value.code == "CALENDAR_INCOMPLETE"


# --------------------------------------------------------------------------- bundle / provenance

def test_frozen_bundle_digests_and_tamper_detection(full, design, tmp_path):
    out = tmp_path / "bundle"
    manifest = data.build_frozen_bundle(full["src"], design, out)
    assert manifest["test_materialized"] is False
    assert manifest["feature_manifest_sha256"] == provenance.EXPECTED_MANIFEST_SHA256
    assert manifest["availability_contract_sha256"] == provenance.EXPECTED_CONTRACT_SHA256
    assert manifest["confirmatory_test_blocked"] is True
    assert set(manifest["splits"]) == {"fit", "inner", "validation"}
    assert sorted(p.name for p in out.iterdir()) == [
        "BUNDLE.json", "fit.npz", "fit_aux.npz", "inner.npz", "inner_aux.npz",
        "validation.npz", "validation_aux.npz"]
    # model-lane consumed contract (INTERFACE_MODEL.json)
    for key in ("schema", "features", "feature_manifest_sha256", "horizons_hours",
                "offsets_hours", "splits"):
        assert key in manifest
    assert manifest["offsets_hours"] == list(range(-24, 1, 2))
    for name, info in manifest["splits"].items():
        assert not info["path"].startswith("/")
        assert provenance.file_sha256(out / info["path"]) == info["sha256"]
        with np.load(out / info["path"]) as z:
            assert sorted(z.files) == ["mask", "origin_price", "row_id", "target_price", "x"]
            assert z["row_id"].dtype == np.int64 and len(np.unique(z["row_id"])) == len(z["row_id"])
            assert z["x"].shape[1:] == (13, 19)
            assert np.isfinite(z["target_price"][z["mask"]]).all()
    assert set(manifest["fit_statistics"]["naive_mae_fit"]) == {str(h) for h in HORIZONS}
    for value in manifest["inputs"].values():
        assert not str(value.get("name", "")).startswith("/")
    assert data.verify_bundle(out)["status"] == "VERIFIED"
    again = data.build_frozen_bundle(full["src"], design, tmp_path / "bundle2")
    assert again["bundle_sha256"] == manifest["bundle_sha256"]
    with np.load(out / "inner.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["x"][0, 0, 0] += 1e-9
    np.savez(out / "inner.npz", **arrays)
    with pytest.raises(data.DataContractError) as err:
        data.verify_bundle(out)
    assert err.value.code == "BUNDLE_DIGEST_MISMATCH"


def test_test_rows_are_refused_before_materialization(design, tmp_path):
    src = _synth.make_input(tmp_path / "t", last_sunday=datetime(2025, 1, 5, tzinfo=UTC),
                            data_end=_synth.epoch("2025-01-08T00:00:00+00:00"))
    with pytest.raises(data.DataContractError) as err:
        data.load_input(src, design)
    assert err.value.code == "TEST_ROWS_PRESENT"


def test_stlfsi4_and_aliases_rejected(design, tmp_path):
    names = list(_synth.MANIFEST_NO_STLFSI4)
    for alias in ("fred.stress.stlfsi4.level", "STLFSI4", "fred.stress.stlfsi4.level_lag24",
                  "stlfsi_v2"):
        with pytest.raises(provenance.ProvenanceError) as err:
            provenance.check_manifest(names[:-1] + [alias])
        assert err.value.code == "EXCLUDED_FEATURE_ALIAS"
    src = _synth.make_input(tmp_path / "s", features=names[:-1] + ["fred.stress.stlfsi4.level"])
    with pytest.raises(provenance.ProvenanceError):
        data.load_input(src, design)


def test_availability_gate_reports_unverified_and_blocks_confirmatory_test():
    report = provenance.availability_gate(list(_synth.MANIFEST_NO_STLFSI4))
    assert report["status"] == "PASS"
    assert report["contract_sha256"] == provenance.EXPECTED_CONTRACT_SHA256
    assert report["manifest_sha256"] == provenance.EXPECTED_MANIFEST_SHA256
    assert report["confirmatory_test_blocked"] is True
    status = provenance.producer_status()
    assert set(status) == set(provenance.UNRESOLVED_SEVEN)
    for row in status.values():
        assert row["status"] in ("RECOVERED", "STILL_UNRESOLVED")
        assert row["status"] != "RECOVERED" or (row["producer"] and row["reproduction"])
        assert row["status"] != "STILL_UNRESOLVED" or row["missing"]
    with pytest.raises(provenance.ProvenanceError) as err:
        provenance.confirmatory_gate()
    assert err.value.code == "PROVENANCE_UNRESOLVED"
