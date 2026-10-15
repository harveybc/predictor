"""MG2/MG12 training-adapter bundles (order 50d20d01 §1.1; plan §2/§4/§6 data items).

Synthetic fixtures only: no fit, no GPU, no TEST bar, no DOIN. Every assertion
is a data/contract property; none is a predictive measurement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import data, market_grid, naive, provenance

UTC = timezone.utc
H = 3600
SHORT_HEADS = (2, 3, 4, 5, 6, 12)
LONG_HEADS = (18, 24, 36, 48, 60, 72)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    root = tmp_path_factory.mktemp("mb")
    return _synth.make_input(root / "input",
                             missing={_synth.epoch("2022-06-15T13:00:00+00:00")},
                             nonfinite=[(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)])


@pytest.fixture(scope="module")
def built(src, tmp_path_factory):
    out = tmp_path_factory.mktemp("built")
    short = market_grid.build_market_bundle(src, market_grid.market_design(_synth.load_design()),
                                            out / "mg2", code_commit="TEST")
    long_ = market_grid.build_market_bundle(src, market_grid.market_design(
        data.long_data_view(_synth.load_design())), out / "mg12", code_commit="TEST")
    return {"short": short, "long": long_, "short_dir": out / "mg2", "long_dir": out / "mg12",
            "src": src}


# ----------------------------------------------------------------- design seal (§2)

def test_market_design_is_typed_and_rejects_contradictory_fields():
    d = market_grid.market_design(_synth.load_design())
    assert d["schema"] == market_grid.DESIGN_SCHEMA
    assert d["input_sampling"]["rule"] == "MG2_MARKET_EXACT"
    assert d["input_sampling"]["spacing_open_market_hours"] == 2
    assert d["input_sampling"]["span_open_market_hours"] == 24
    assert d["input_sampling"]["observations"] == 13
    assert d["input_sampling"]["elapsed_span_hours_fixed"] is False
    assert "history_span_hours" not in d["input_sampling"]
    assert d["horizons_hours"] == list(SHORT_HEADS)
    assert d["time_encoder"]["adjacent_delta_unit"] == "HOURS"
    assert d["time_encoder"]["model_applies_division_by"] == 24
    assert d["data_identity_sha256"] == market_grid.data_identity_sha256(d)
    long_ = market_grid.market_design(data.long_data_view(_synth.load_design()))
    assert long_["input_sampling"]["rule"] == "MG12_MARKET_EXACT"
    assert long_["data_identity_sha256"] != d["data_identity_sha256"]
    bad = json.loads(json.dumps(d))
    bad["input"] = {"history_span_hours": 24, "spacing_hours": 2}
    with pytest.raises(data.DataContractError) as err:
        market_grid.validate_market_design(bad)
    assert err.value.code == "CONTRADICTORY_TIME_FIELDS"
    worse = json.loads(json.dumps(d))
    worse["input_sampling"]["elapsed_span_hours_fixed"] = True
    with pytest.raises(data.DataContractError):
        market_grid.validate_market_design(worse)


def test_frozen_scope_fields_are_enforced():
    d = market_grid.market_design(_synth.load_design())
    assert d["seed"] == 6102026 and d["repetitions"] == 1
    assert d["splits"]["train"] == {"start": "2020-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"}
    assert d["splits"]["validation"]["end"] == "2025-01-01T00:00:00Z"
    assert d["splits"]["inner_tail_weeks"] == 8
    assert d["feature_policy"]["exclude"] == ["fred.stress.stlfsi4.level"]
    assert len(d["features"]) == 19
    assert d["feature_manifest_sha256"] == provenance.EXPECTED_MANIFEST_SHA256
    assert d["input_sampling"]["no_asof_no_smoothing_no_interpolation_no_synthetic_prices"] is True
    assert d["preprocessing"].startswith("scalers and loss scales fit on FIT rows only")
    assert d["selection_redone"] is False if "selection_redone" in d else True
    assert d["targets"].startswith("exact ELAPSED t+h, unmodified")


# ------------------------------------------------------- per-example time inputs (§3/§6)

def test_per_example_offsets_closure_flags_and_first_last_conventions(built):
    arrays = market_grid.load_market_split(built["short_dir"], "fit")
    off, adj, flag = arrays["offsets_hours"], arrays["adjacent_delta_hours"], arrays["closure_crossed"]
    n = len(arrays["row_id"])
    assert off.shape == adj.shape == flag.shape == (n, 13)
    assert (off[:, -1] == 0).all()                      # last offset is exactly 0
    assert (off[:, :-1] < 0).all()                      # signed, strictly past
    assert (np.diff(off, axis=1) > 0).all()             # chronological
    assert (adj[:, 0] == 0).all() and (flag[:, 0] == 0).all()
    np.testing.assert_array_equal(adj[:, 1:], np.diff(off, axis=1))
    assert set(np.unique(flag).tolist()) <= {0, 1}
    assert np.issubdtype(off.dtype, np.floating) and np.issubdtype(flag.dtype, np.integer)
    # raw HOURS, not divided by 24 (that conversion belongs to the model)
    assert adj[:, 1:].min() >= 2 and float(adj.max()) > 2.0
    crossing = np.flatnonzero(flag.any(axis=1))
    assert len(crossing) > 0
    for r in crossing[:20]:
        j = int(np.flatnonzero(flag[r])[0])
        assert adj[r, j] > 2.0                          # a flagged step is longer than the spacing
    ordinary = np.flatnonzero(~flag.any(axis=1))
    for r in ordinary[:20]:
        np.testing.assert_array_equal(adj[r, 1:], np.full(12, 2.0))
        assert off[r, 0] == -24.0


def test_value_channels_come_from_the_exact_slot_source_rows(built, src):
    bundle = data.load_input(src, _synth.load_design())
    arrays = market_grid.load_market_split(built["short_dir"], "fit")
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    ts_slots = arrays["slot_instants"]
    for r in range(0, len(arrays["row_id"]), max(1, len(arrays["row_id"]) // 30)):
        t = int(arrays["row_id"][r])
        assert int(ts_slots[r, -1]) == t                 # current t included
        for j in range(13):
            s = int(ts_slots[r, j])
            np.testing.assert_array_equal(arrays["x"][r, j], bundle.x[pos[s]])
            assert arrays["offsets_hours"][r, j] == (s - t) / H
        assert np.isfinite(arrays["x"][r]).all()


def test_weekend_dst_and_ordinary_weeks_have_distinct_offset_patterns(built, src):
    arrays = market_grid.load_market_split(built["short_dir"], "fit")
    weeks = json.loads((src / "CALENDAR.json").read_text())["market_calendar"]["weeks"]
    by_origin = {int(v): i for i, v in enumerate(arrays["row_id"])}
    seen = {}
    for wid in ("2021-03-14", "2021-11-07", "2021-02-07"):
        i = next(k for k, w in enumerate(weeks) if w["week_id"] == wid)
        t = _synth.epoch(weeks[i]["market_open"]) + 12 * H
        r = by_origin[t]
        assert arrays["closure_crossed"][r].sum() == 1
        seen[wid] = float(-arrays["offsets_hours"][r, 0])
    assert seen["2021-03-14"] != seen["2021-11-07"]      # 47 h vs 49 h closures
    assert all(v > 24.0 for v in seen.values())          # never labelled a fixed 24 h span


def test_future_perturbation_changes_nothing(built, src, tmp_path):
    """Per-window: nothing after t enters its window. Split-wide: FIT scalers/loss scales
    cannot be moved by inner/development rows."""
    arrays = market_grid.load_market_split(built["short_dir"], "fit")
    r = len(arrays["row_id"]) - 1                      # last FIT origin: all FIT data is <= t
    t = int(arrays["row_id"][r])
    p = _synth.make_input(tmp_path / "f", missing={_synth.epoch("2022-06-15T13:00:00+00:00")},
                          nonfinite=[(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)])
    with np.load(p / "INPUT.npz") as z:
        ts, x, price = z["ts"], z["x"].copy(), z["price"].copy()
    later = ts > t
    x[later] += 1000.0
    price[later] += 1.0
    _synth.rewrite_input(p, x=x, price=price)
    out = tmp_path / "fb"
    market_grid.build_market_bundle(p, market_grid.market_design(_synth.load_design()), out,
                                    code_commit="TEST")
    a2 = market_grid.load_market_split(out, "fit")
    r2 = list(a2["row_id"]).index(t)
    for key in ("x", "offsets_hours", "adjacent_delta_hours", "closure_crossed", "slot_instants"):
        np.testing.assert_array_equal(a2[key][r2], arrays[key][r])
    assert a2["origin_price"][r2] == arrays["origin_price"][r]
    assert (a2["offsets_hours"][r2, -1] == 0) and a2["row_id"][r2] == t
    # FIT scalers / loss scales are fit on FIT rows only: development rows cannot move them
    # (a FIT label matures up to t+h, so only data beyond the split can be called "outer").
    p2 = _synth.make_input(tmp_path / "o", missing={_synth.epoch("2022-06-15T13:00:00+00:00")},
                           nonfinite=[(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)])
    with np.load(p2 / "INPUT.npz") as z:
        ts2, x2, price2 = z["ts"], z["x"].copy(), z["price"].copy()
    outer = ts2 >= _synth.epoch("2024-01-01T00:00:00+00:00")
    x2[outer] += 1000.0
    price2[outer] += 1.0
    _synth.rewrite_input(p2, x=x2, price=price2)
    out2 = tmp_path / "ob"
    m2 = market_grid.build_market_bundle(p2, market_grid.market_design(_synth.load_design()),
                                         out2, code_commit="TEST")
    assert m2["fit_statistics"] == built["short"]["fit_statistics"]
    assert m2["splits"]["fit"]["arrays_sha256"] == \
        built["short"]["splits"]["fit"]["arrays_sha256"]
    assert m2["splits"]["validation"]["arrays_sha256"] != \
        built["short"]["splits"]["validation"]["arrays_sha256"]   # positive control


# -------------------------------------------------------- targets, masks, naive (§4/§6)

def test_targets_are_exact_elapsed_hours_and_masks_are_c1(built, src):
    from tools.i6n_contracts import eligibility as c1
    bundle = data.load_input(src, _synth.load_design())
    arrays = market_grid.load_market_split(built["short_dir"], "validation")
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    for j, h in enumerate(SHORT_HEADS):
        rows = np.flatnonzero(arrays["mask"][:, j])
        assert len(rows) > 0
        for r in rows[:25]:
            q = int(arrays["row_id"][r]) + h * H         # ELAPSED, not market time
            assert arrays["target_price"][r, j] == bundle.price[pos[q]]
        assert np.isnan(arrays["target_price"][~arrays["mask"][:, j], j]).all()
    cal = json.loads((src / "CALENDAR.json").read_text())
    ref, weeks = c1.load_calendar(cal)
    observed = c1.ObservedTargets({datetime.fromtimestamp(int(v), UTC):
                                   datetime.fromtimestamp(int(v), UTC) for v in bundle.ts})
    splits = {s.split_id: s for s in data.split_bounds(_synth.load_design())}
    week = next(w for w in weeks if w.week_id == "2024-03-10")
    checked = 0
    for r, t in enumerate(arrays["row_id"].tolist()):
        if str(arrays["week_ids"][r]) != "2024-03-10":
            continue
        ds = c1.evaluate_heads(datetime.fromtimestamp(int(t), UTC), SHORT_HEADS,
                               c1.Role.MANAGEMENT, week, input_available=True,
                               observed_targets=observed, split=splits["VALIDATION"])
        assert tuple(bool(v) for v in arrays["mask"][r]) == tuple(d.label_eligible for d in ds)
        checked += 1
    assert checked > 10


def test_same_row_per_horizon_naive_matches_independent_replay(built, src):
    manifest = market_grid.load_market_manifest(built["short_dir"])
    arrays = market_grid.load_market_split(built["short_dir"], "fit")

    class A:
        row_ids = arrays["row_ids"]
        mask = arrays["mask"]
    replay = naive.naive_replay(src, A, SHORT_HEADS)
    for h in SHORT_HEADS:
        assert replay[str(h)]["mae"] == pytest.approx(
            manifest["fit_statistics"]["naive_mae_fit"][str(h)], rel=1e-12)
        assert replay[str(h)]["n"] == manifest["splits"]["fit"]["head_valid"][str(h)]
    val = market_grid.load_market_split(built["short_dir"], "validation")
    pred = np.where(val["mask"], val["origin_price"][:, None] + 2e-4, np.nan)
    report = market_grid.paired_metrics(built["short_dir"], "validation", pred, input_dir=src)
    assert report["agreement"]["status"] == "AGREE"
    for h in SHORT_HEADS:
        g = report["per_horizon"][str(h)]
        assert g["n"] == int(val["mask"][:, SHORT_HEADS.index(h)].sum())


def test_split_and_purge_boundaries(built):
    for family, heads in (("short_dir", SHORT_HEADS), ("long_dir", LONG_HEADS)):
        manifest = market_grid.load_market_manifest(built[family])
        bounds = {"fit": ("2020-01-01", "2023-11-06"), "inner": ("2023-11-06", "2024-01-01"),
                  "validation": ("2024-01-01", "2025-01-01")}
        for split, (lo, hi) in bounds.items():
            arrays = market_grid.load_market_split(built[family], split)
            lo_e, hi_e = _synth.epoch(lo + "T00:00:00+00:00"), _synth.epoch(hi + "T00:00:00+00:00")
            assert (arrays["row_id"] >= lo_e).all() and (arrays["row_id"] < hi_e).all()
            for j, h in enumerate(heads):
                q = arrays["row_id"][arrays["mask"][:, j]] + h * H
                assert (q >= lo_e).all() and (q < hi_e).all()   # purged maturity, no crossing
            assert manifest["splits"][split]["n_rows"] == len(arrays["row_id"])


def test_unplanned_gaps_are_not_compressed(built, src):
    manifest = market_grid.load_market_manifest(built["short_dir"])
    hole = _synth.epoch("2022-06-15T13:00:00+00:00")
    fit = manifest["splits"]["fit"]
    assert fit["counts"]["closures_vs_gaps"]["unplanned_missing_expected_hours"] >= 1
    assert fit["counts"]["origins"]["INPUT_SLOT_NOT_OBSERVED"] >= 1
    arrays = market_grid.load_market_split(built["short_dir"], "fit")
    assert hole + 2 * H not in set(int(v) for v in arrays["row_id"])
    # no retained window silently steps over the missing expected hour
    assert hole not in set(int(v) for v in arrays["slot_instants"].reshape(-1))


# -------------------------------------------------------- seals and reconciliation (§1.1)

def test_bundle_seals_identities_timestamps_exclusions_and_support_hashes(built):
    for family, rule in (("short", "MG2_MARKET_EXACT"), ("long", "MG12_MARKET_EXACT")):
        m = built[family]
        assert m["schema"] == market_grid.BUNDLE_SCHEMA != data.SCHEMA
        assert m["input_sampling"]["rule"] == rule
        assert "offsets_hours" not in m                  # no fixed elapsed ladder is published
        assert m["offsets_open_market_hours"][-1] == 0
        assert m["elapsed_offsets_vary_per_row"] is True
        assert m["test_materialized"] is False
        for key in ("design_sha256", "data_identity_sha256", "feature_manifest_sha256",
                    "availability_contract_sha256", "calendar", "source_clock", "inputs",
                    "code", "bundle_sha256", "support_hashes", "reconciliation"):
            assert key in m, (family, key)
        assert m["calendar"]["status"] == "ASSUMED_NOT_BROKER"
        for split, info in m["splits"].items():
            for key in ("path", "sha256", "aux_path", "n_rows", "row_ids_sha256", "arrays_sha256",
                        "head_valid", "head_entry_valid", "counts", "slot_instants_sha256",
                        "offsets_sha256"):
                assert key in info, (family, split, key)
            assert info["counts"]["origins"]["candidates"] >= info["n_rows"]
        assert data.verify_bundle(built[f"{family}_dir"])["status"] == "VERIFIED"


def test_reconciliation_against_the_support_study_is_retained(built):
    for family in ("short", "long"):
        rec = built[family]["reconciliation"]
        assert rec["rule"] == built[family]["input_sampling"]["rule"]
        assert rec["study_sha256"] == market_grid.STUDY_SHA256
        assert rec["basis"].startswith("materialized rows")
        reference = market_grid.STUDY_REFERENCE[rec["rule"]]
        for split in ("fit", "inner", "validation"):
            row = rec["per_split"][split]
            assert row["study_rows"] == reference[split]
            assert row["difference"] == row["materialized"] - row["study_rows"]
            assert row["status"] == ("MATCH" if row["difference"] == 0
                                     else "DIFFERENCE_RETAINED")
            assert row["materialized"] == market_grid.load_market_manifest(
                built[f"{family}_dir"])["splits"][split]["n_rows"]
        # this synthetic fixture is not the retained input, so differences must be RETAINED,
        # never silently zeroed or filled; the real bundles are reconciled in the return.
        assert rec["status"] in ("RECONCILED", "DIFFERENCE_RETAINED_BEFORE_ANY_FIT")
        if any(r["difference"] for r in rec["per_split"].values()):
            assert rec["status"] == "DIFFERENCE_RETAINED_BEFORE_ANY_FIT"
        assert rec["counts_are_not_quotas"] is True
        assert "filling gaps" in rec["note"]
    shared = built["short"]["reconciliation"]["shared_usable_origins"]
    assert set(shared) == {"fit", "inner", "validation"}


def test_study_reference_counts_are_the_real_ones():
    assert market_grid.STUDY_REFERENCE["MG2_MARKET_EXACT"] == {"fit": 16496, "inner": 715,
                                                               "validation": 4772}
    assert market_grid.STUDY_REFERENCE["MG12_MARKET_EXACT"] == {"fit": 13174, "inner": 546,
                                                                "validation": 3804}
    assert market_grid.STUDY_REFERENCE["shared"] == {"fit": 13123, "inner": 546,
                                                     "validation": 3760}


def test_market_bundle_is_not_adoptable_as_a_ds2_bundle(built, src):
    """A market bundle must not satisfy the elapsed-grid contract by shape alone."""
    m = built["short"]
    assert m["schema"] == "i6dmo.market_bundle.v1"
    with pytest.raises(data.DataContractError) as err:
        data.authenticate_sampling(m, _synth.load_design())
    assert err.value.code == "SAMPLING_RULE_MISMATCH"
    with pytest.raises(data.DataContractError) as err:
        market_grid.authenticate_market_bundle(m, market_grid.market_design(
            data.long_data_view(_synth.load_design())))
    assert err.value.code == "MARKET_RULE_MISMATCH"
    assert market_grid.authenticate_market_bundle(
        m, market_grid.market_design(_synth.load_design())) == "MG2_MARKET_EXACT"


def test_test_rows_are_refused(src, tmp_path, built):
    bad = _synth.make_input(tmp_path / "t", last_sunday=datetime(2025, 1, 5, tzinfo=UTC),
                            data_end=_synth.epoch("2025-01-08T00:00:00+00:00"))
    with pytest.raises(data.DataContractError) as err:
        market_grid.build_market_bundle(bad, market_grid.market_design(_synth.load_design()),
                                        tmp_path / "tb")
    assert err.value.code == "TEST_ROWS_PRESENT"
