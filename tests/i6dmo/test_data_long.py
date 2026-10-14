"""I6-DMO data lane, LONG family (I6-DMO-LONG-DS12-STATIC-R0) bundle preparation tests.

Synthetic fixtures only (tests/i6dmo/_synth.py); no fit, no TEST year and no
short-family outcome is read. The long data view is the short design with the
PLAN section 5 long fields (the same derivation as the model lane's
``seal-long-design``); seal fields are accepted but do not enter the data identity.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import data, naive

UTC = timezone.utc
H = 3600
LONG = (18, 24, 36, 48, 60, 72)


def long_design(sealed=False):
    d = _synth.load_design()
    d["experiment_id"] = "I6-DMO-LONG-DS12-STATIC-R0"
    d["horizons_hours"] = list(LONG)
    d["input"]["history_span_hours"] = 144
    d["input"]["spacing_hours"] = 12
    if sealed:
        d["derived_from"] = {"rule": "x", "short_experiment_id": "I6-DMO-SHORT-DS2-STATIC-R0",
                             "short_design_sha256": "0" * 64,
                             "short_terminal": {"unit_id": "short.report", "identity_sha256": "1" * 64,
                                                "terminal_file_sha256": "2" * 64}}
        d["seal_sha256"] = "3" * 64
    return d


def asof_reference(ts, x, s, f):
    k = int(np.searchsorted(ts, s, side="right")) - 1
    while k >= 0 and not np.isfinite(x[k, f]):
        k -= 1
    return k


@pytest.fixture(scope="module")
def lng(tmp_path_factory):
    root = tmp_path_factory.mktemp("long")
    src = _synth.make_input(root / "input",
                            missing={_synth.epoch("2022-06-15T13:00:00+00:00")},
                            nonfinite=[(_synth.epoch("2021-09-08T09:00:00+00:00"), 5)])
    design = long_design()
    bundle = data.load_input(src, design)
    return {"src": src, "design": design, "bundle": bundle,
            "table": data.build_origin_table(bundle, design), "root": root}


def test_long_window_offsets_rule_and_asof_values(lng):
    spec = data.WindowSpec.from_design(lng["design"])
    assert spec.offsets_hours() == tuple(range(-144, 1, 12))
    assert len(spec.offsets_hours()) == 13
    assert lng["bundle"].sampling_rule == "DS12_ASOF_LAST_AVAILABLE"
    assert data.sampling_rule_name(data.WindowSpec.from_design(_synth.load_design())) \
        == "DS2_ASOF_LAST_AVAILABLE"
    arrays = data.materialize(lng["bundle"], lng["table"], "FIT")
    ts, x = lng["bundle"].ts, lng["bundle"].x
    for r in range(0, len(arrays.origin_ts), max(1, len(arrays.origin_ts) // 25)):
        t = int(arrays.origin_ts[r])
        for j, off in enumerate(spec.offsets_hours()):
            s = t + off * H
            for f in (0, 5, 18):
                k = asof_reference(ts, x, s, f)
                assert arrays.X[r, j, f] == x[k, f]
                assert arrays.staleness_s[r, j, f] == s - int(ts[k])


def test_long_weekend_and_dst_staleness_counted(lng):
    cal = json.loads((lng["src"] / "CALENDAR.json").read_text())["market_calendar"]["weeks"]
    closes = [(_synth.epoch(w["market_close"]), _synth.epoch(n["market_open"]))
              for w, n in zip(cal, cal[1:])]
    arrays = data.materialize(lng["bundle"], lng["table"], "FIT")
    spec = data.WindowSpec.from_design(lng["design"])
    slots = arrays.origin_ts[:, None] + np.asarray(spec.offsets_hours()) * H
    weekend = np.zeros(slots.shape, dtype=bool)
    expected = np.zeros(slots.shape, dtype=np.int64)
    for c, o in closes:
        inside = (slots > c) & (slots <= o)
        weekend |= inside
        expected = np.where(inside, slots - c, expected)
    # every 144 h window spans a weekend: weekend slots read the Friday close bar
    assert weekend.any(axis=1).all()
    st = arrays.staleness_s[:, :, 0]
    np.testing.assert_array_equal(st[weekend], expected[weekend])
    asof = lng["table"].counts["FIT"]["asof"]
    assert asof["rule"] == "DS12_ASOF_LAST_AVAILABLE"
    assert asof["closure_threshold_hours"] == 49
    assert asof["origins_with_asof_slot"] == len(arrays.origin_ts)
    # DST: the spring-forward weekend closure is 47 h, the fall-back one 49 h
    for wid, hours in (("2021-03-14", 47), ("2021-11-07", 49)):
        i = next(k for k, w in enumerate(cal) if w["week_id"] == wid)
        gap = (_synth.epoch(cal[i]["market_open"]) - _synth.epoch(cal[i - 1]["market_close"])) // H
        assert gap == hours
        a = _synth.epoch(cal[i]["market_open"]) + 12 * H
        r = list(arrays.origin_ts).index(a)
        j = list(spec.offsets_hours()).index(-24)
        s = a - 24 * H
        assert arrays.staleness_s[r, j, 0] == s - _synth.epoch(cal[i - 1]["market_close"])


def test_long_labels_never_cross_inner_or_validation_boundaries(lng):
    bounds = {"FIT": _synth.epoch("2023-11-06T00:00:00+00:00"),
              "INNER": _synth.epoch("2024-01-01T00:00:00+00:00"),
              "VALIDATION": _synth.epoch("2025-01-01T00:00:00+00:00")}
    starts = {"FIT": _synth.epoch("2020-01-01T00:00:00+00:00"), "INNER": bounds["FIT"],
              "VALIDATION": bounds["INNER"]}
    for split, end in bounds.items():
        arrays = data.materialize(lng["bundle"], lng["table"], split)
        assert len(arrays.origin_ts) > 0
        assert (arrays.origin_ts >= starts[split]).all()
        for j, h in enumerate(LONG):
            q = arrays.origin_ts[arrays.mask[:, j]] + h * H
            assert len(q) > 0, (split, h)
            assert (q < end).all()
            assert (q >= starts[split]).all()
    # an origin 50 h before the inner boundary keeps h18/h24/h36/h48 but not h60/h72
    table = lng["table"]
    cands = [t for t in table.origin_ts
             if bounds["FIT"] - 73 * H <= t < bounds["FIT"] and table.record(t).split == "FIT"]
    for t in cands:
        rec = table.record(t)
        for j, h in enumerate(LONG):
            if t + h * H >= bounds["FIT"]:
                assert rec.label_ok[j] is False


def test_long_naive_replay_and_paired_metrics_agree(lng):
    arrays = data.materialize(lng["bundle"], lng["table"], "VALIDATION")
    pred = np.where(arrays.mask, arrays.price_origin[:, None] + 3e-4, np.nan)
    report = naive.paired_metrics(arrays, pred, lng["design"], input_dir=lng["src"],
                                  checkpoint="synthetic-long", sealed_at="2026-10-14T00:00:00+00:00")
    assert report["agreement"]["status"] == "AGREE"
    replay = naive.naive_replay(lng["src"], arrays, LONG)
    for j, h in enumerate(LONG):
        m = arrays.mask[:, j]
        manual = np.abs(arrays.y_price[m, j] - arrays.price_origin[m]).sum() / m.sum()
        assert replay[str(h)]["mae"] == pytest.approx(manual, rel=1e-12)
        assert report["per_horizon"][str(h)]["naive_mae"] == pytest.approx(manual, rel=1e-12)


def test_long_bundle_identity_history_provenance_and_model_format(lng, tmp_path):
    (lng["src"] / "CLOCK_SCOPE.json").write_text(json.dumps({
        "assumption_id": "SYNTH_DCA", "assumption_sha256": "a" * 64,
        "declared_assumption": {"assumption_id": "SYNTH_DCA", "status": "DECLARED_ASSUMPTION",
                                "span": {"start": "2019-03-11T00:00:00+00:00",
                                         "end": "2019-12-01T00:00:00+00:00"}},
        "base_scope": {"data_start": "2019-12-01T00:00:00Z",
                       "warmup_2019_december": "declared, not quote-verified"}}))
    out = tmp_path / "b"
    m = data.build_frozen_bundle(lng["src"], long_design(), out)
    assert m["offsets_hours"] == list(range(-144, 1, 12))
    assert m["horizons_hours"] == list(LONG)
    assert m["input_sampling"]["rule"] == "DS12_ASOF_LAST_AVAILABLE"
    assert m["experiment_id"] == "I6-DMO-LONG-DS12-STATIC-R0"
    hp = m["history_provenance"]
    assert hp["earliest_consumed_bar"] < "2020-01-01"
    assert hp["consumed_before_train_start"] is True
    assert hp["declared_assumptions"][0]["assumption_id"] == "SYNTH_DCA"
    assert hp["declared_assumptions"][0]["overlaps_consumed_history"] is False
    assert hp["base_scope_warmup"] == "declared, not quote-verified"
    assert hp["consumed_before_base_scope_start"] is False
    # the data identity ignores seal fields; a sealed long design yields the same data contract
    m2 = data.build_frozen_bundle(lng["src"], long_design(sealed=True), tmp_path / "b2")
    assert m2["data_contract_sha256"] == m["data_contract_sha256"]
    assert m2["design_sha256"] != m["design_sha256"]
    assert data.data_contract_sha256(_synth.load_design()) != m["data_contract_sha256"]
    for info in m["splits"].values():
        with np.load(out / info["path"]) as z:
            assert sorted(z.files) == ["mask", "origin_price", "row_id", "target_price", "x"]
            assert z["x"].shape[1:] == (13, 19) and z["mask"].shape[1] == 6
        with np.load(out / info["aux_path"]) as z:
            assert z["slot_staleness_s"].shape[1:] == (13, 19)
    assert data.verify_bundle(out)["status"] == "VERIFIED"


def test_long_requires_144h_history_and_never_shortens(tmp_path):
    src = _synth.make_input(tmp_path / "late", first_sunday=datetime(2019, 12, 1, tzinfo=UTC))
    with np.load(src / "INPUT.npz") as z:
        keep = z["ts"] >= _synth.epoch("2019-12-29T00:00:00+00:00")
        _synth.rewrite_input(src, ts=z["ts"][keep], x=z["x"][keep], price=z["price"][keep])
    data.load_input(src, _synth.load_design())          # short: 24 h of history suffices
    with pytest.raises(data.DataContractError) as err:
        data.load_input(src, long_design())
    assert err.value.code == "COVERAGE_INCOMPLETE"
    bad = long_design()
    bad["input"]["spacing_hours"] = 11
    with pytest.raises(data.DataContractError) as err:
        data.load_input(src, bad)
    assert err.value.code == "WINDOW_NOT_CONTRACT"


def test_long_data_view_matches_plan_derivation():
    view = data.long_data_view(_synth.load_design())
    assert data.data_contract_sha256(view) == data.data_contract_sha256(long_design(sealed=True))
    with pytest.raises(data.DataContractError):
        data.long_data_view(view)
