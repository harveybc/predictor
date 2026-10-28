"""CD01 master/suffix support; CD02 full TRAIN; CD03 calendar folds and no TEST;
CD04 TRAIN-only scales/future invariance; CD05 source/code/array custody;
CD06 typed unsupported TRAIN and empty-period skips. Synthetic data, no fits.
"""
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import continuation_data as C, data, market_grid
from tools.i6n_contracts import eligibility as E

H = 3600
START = _synth.epoch("2020-01-01T00:00:00+00:00")
CUT = _synth.epoch("2024-01-01T00:00:00+00:00")
END = _synth.epoch("2025-01-01T00:00:00+00:00")
MISSING = _synth.epoch("2024-01-10T13:00:00+00:00")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = _synth.make_input(tmp_path_factory.mktemp("continuation-input") / "input")
    calendar = json.loads((root / "CALENDAR.json").read_text())
    chosen = {"2019-12-22", "2019-12-29", "2020-01-05", "2023-10-22", "2023-10-29",
              "2023-11-05", "2023-11-12", "2023-12-17", "2023-12-24", "2023-12-31",
              "2024-01-07", "2024-03-24", "2024-03-31", "2024-06-23", "2024-06-30",
              "2024-09-22", "2024-09-29", "2024-12-22", "2024-12-29"}
    stamps = _synth.bar_ends([w for w in calendar["market_calendar"]["weeks"]
                             if w["week_id"] in chosen])
    with np.load(root / "INPUT.npz", allow_pickle=False) as z:
        keep = np.isin(z["ts"], stamps) & (z["ts"] != MISSING)
        values = {k: z[k][keep] for k in z.files}
    _synth.rewrite_input(root, **values)
    return root


@pytest.fixture(scope="module")
def prepared(source, tmp_path_factory):
    root = tmp_path_factory.mktemp("continuation-output") / "master"
    return root, C.prepare(source, root, _synth.DESIGN_PATH)


def test_cd01_only_four_max_tables_and_one_input_load(source, tmp_path, monkeypatch):
    original_load, original_table = data.load_input, market_grid.build_market_table
    reads, tables = [], []
    def read(*args, **kwargs):
        reads.append(1)
        return original_load(*args, **kwargs)
    def table(bundle, design, **kwargs):
        assert design["input"]["observations"] == 24
        assert kwargs["evaluation_splits"][0].split_id == "FIT"
        tables.append((design["input"]["spacing_hours"], tuple(design["horizons_hours"])))
        return original_table(bundle, design, **kwargs)
    monkeypatch.setattr(data, "load_input", read)
    monkeypatch.setattr(market_grid, "build_market_table", table)
    root = tmp_path / "master"
    manifest = C.prepare(source, root, _synth.DESIGN_PATH)
    assert len(reads) == 1 and len(tables) == 4
    assert manifest["test_read"] is False and manifest["fits_run"] == 0
    monkeypatch.setattr(market_grid, "build_market_table", lambda *a, **k: pytest.fail("rebuilt a fold table"))
    C.load(root, "long", 2, 6, horizons=[72])
    C.load(root, "short", 4, 24, period=C.periods(root, "QUARTERLY")[0])


def test_cd01_all_windows_and_spacings_share_support_scales_and_suffixes(prepared):
    for group in ("short", "long"):
        reference, ref_scales, ref_meta = C.load(prepared[0], group, 2, 24)
        max_by_spacing = {s: C.load(prepared[0], group, s, 24)[0] for s in (2, 4)}
        for spacing in (2, 4):
            for slots in (6, 9, 13, 18, 24):
                arrays, scales, meta = C.load(prepared[0], group, spacing, slots)
                assert scales == ref_scales and len(meta["features"]) == 19
                assert meta["group_support_sha256"] == ref_meta["group_support_sha256"]
                for split in ("FIT", "VALIDATION"):
                    assert set(arrays) == {"FIT", "VALIDATION"}
                    for key in ("row_ids", "origin_ts", "mask", "target_price", "origin_price", "week_ids"):
                        np.testing.assert_array_equal(arrays[split][key], reference[split][key])
                    np.testing.assert_array_equal(arrays[split]["x"], max_by_spacing[spacing][split]["x"][:, -slots:])
                    assert arrays[split]["x"].shape[1:] == (slots, 19)
                    assert (arrays[split]["offsets_hours"][:, -1] == 0).all()


def test_cd02_full_four_year_train_without_inner_and_exact_targets(source, prepared):
    arrays, scales, meta = C.load(prepared[0], "long", 4, 13)
    fit, val = arrays["FIT"], arrays["VALIDATION"]
    assert set(arrays) == {"FIT", "VALIDATION"} and meta["validation_used_for_stopping"] is True
    assert meta["train_start"] == data.iso(START) and meta["train_end"] == data.iso(CUT)
    assert (fit["row_ids"] >= START).all() and (fit["row_ids"] < CUT).all()
    assert (fit["row_ids"] >= _synth.epoch("2023-11-06T00:00:00+00:00")).any()
    with np.load(source / "INPUT.npz", allow_pickle=False) as z:
        ts, price, x = z["ts"], z["price"], z["x"]
    for name, arr in arrays.items():
        assert arr["row_ids"].dtype == np.int64
        assert np.array_equal(arr["origin_ts"], arr["row_ids"])
        instants = arr["origin_ts"][:, None] + (arr["offsets_hours"] * H).astype(np.int64)
        assert not np.isin(instants, MISSING).any()
        np.testing.assert_array_equal(arr["x"], x[np.searchsorted(ts, instants)])
        for j, h in enumerate(meta["horizons"]):
            mask = arr["mask"][:, j]
            targets = arr["row_ids"][mask] + h * H
            np.testing.assert_array_equal(arr["target_price"][mask, j], price[np.searchsorted(ts, targets)])
            assert (targets < (CUT if name == "FIT" else END)).all()
        assert np.array_equal(np.isnan(arr["target_price"]), ~arr["mask"])
    assert (val["row_ids"] >= CUT).all() and (val["row_ids"] < END).all()
    assert np.isfinite(scales["std"]).all() and (np.asarray(scales["std"]) > 0).all()


def test_cd03_periods_follow_operational_calendar_and_preserve_partial_year(source, prepared):
    calendar = json.loads((source / "CALENDAR.json").read_text())
    _, weeks = E.load_calendar(calendar)
    expected = [w for w in weeks if w.operational_start.timestamp() < END and w.force_flat.timestamp() > CUT]
    for cadence in ("QUARTERLY", "MONTHLY", "WEEKLY"):
        periods = C.periods(prepared[0], cadence)
        flattened = [wid for p in periods for wid in p["week_ids"]]
        assert flattened == [w.week_id for w in expected]
        assert len({p["id"] for p in periods}) == len(periods)
        for period in periods:
            week = next(w for w in weeks if w.week_id == period["week_ids"][0])
            cutoff = datetime.fromisoformat(period["train_end"].replace("Z", "+00:00"))
            start = datetime.fromisoformat(period["train_start"].replace("Z", "+00:00"))
            assert cutoff == week.market_open and start == cutoff.replace(year=cutoff.year - 4)
        assert periods[-1]["validation_end"] == data.iso(END)
    first = C.periods(prepared[0], "QUARTERLY")[0]
    assert _synth.epoch(first["train_start"]) < START


@pytest.mark.parametrize("group", ["short", "long"])
@pytest.mark.parametrize("cadence", ["QUARTERLY", "MONTHLY", "WEEKLY"])
def test_cd03_pooled_period_pairs_match_static_whole_year(prepared, group, cadence):
    static, _, _ = C.load(prepared[0], group, 2, 13)
    pieces = []
    for period in C.periods(prepared[0], cadence):
        try:
            arrays, _, meta = C.load(prepared[0], group, 4, 9, period=period)
        except C.ContinuationDataSkip as exc:
            assert exc.code == "EMPTY_VALIDATION_PERIOD"
            continue
        assert meta["period"]["id"] == period["id"]
        pieces.append(arrays["VALIDATION"])
        fit = arrays["FIT"]
        targets = fit["row_ids"][:, None] + np.array(meta["horizons"]) * H
        assert (targets[fit["mask"]] < _synth.epoch(period["train_end"])).all()
    for key in ("row_ids", "mask", "origin_price", "target_price"):
        np.testing.assert_array_equal(np.concatenate([p[key] for p in pieces]), static["VALIDATION"][key])


def test_cd04_train_scaler_union_and_fold_future_invariance(source, prepared, tmp_path):
    period = C.periods(prepared[0], "QUARTERLY")[1]
    arrays, scales, meta = C.load(prepared[0], "long", 2, 24, period=period)
    union = set()
    for spacing in (2, 4):
        fit = C.load(prepared[0], "long", spacing, 24, period=period)[0]["FIT"]
        union.update((fit["origin_ts"][:, None] + fit["offsets_hours"] * H).astype(np.int64).ravel())
    with np.load(source / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    bars = np.array(sorted(union))
    values = x[np.searchsorted(ts, bars)]
    np.testing.assert_allclose(scales["input_mean"], values.mean(axis=0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(scales["std"], values.std(axis=0), rtol=0, atol=1e-14)
    for j, value in enumerate(scales["naive_mae"]):
        fit = arrays["FIT"]
        mask = fit["mask"][:, j]
        assert value == pytest.approx(np.abs(fit["target_price"][mask, j] - fit["origin_price"][mask]).mean())
    alt = tmp_path / "input"
    shutil.copytree(source, alt)
    future = ts >= _synth.epoch(period["train_end"])
    x[future] += 500
    price[future] *= 1.1
    _synth.rewrite_input(alt, x=x, price=price)
    out = tmp_path / "future"
    C.prepare(alt, out, _synth.DESIGN_PATH)
    new_period = next(p for p in C.periods(out, "QUARTERLY") if p["id"] == period["id"])
    changed, new_scales, _ = C.load(out, "long", 2, 24, period=new_period)
    assert new_scales == scales
    for key in arrays["FIT"]:
        np.testing.assert_array_equal(arrays["FIT"][key], changed["FIT"][key])


def test_cd05_relocation_corruption_and_period_identity(source, prepared, tmp_path):
    relocated = tmp_path / "input"
    shutil.copytree(source, relocated)
    expected = C.load(prepared[0], "short", 2, 6)
    got = C.load(prepared[0], "short", 2, 6, input_dir=relocated)
    assert expected[1:] == got[1:]
    with (relocated / "CALENDAR.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(C.ContinuationDataError, match="SOURCE_DIGEST"):
        C.load(prepared[0], "short", 2, 6, input_dir=relocated)
    period = C.periods(prepared[0], "QUARTERLY")[0]
    period["train_end"] = data.iso(END)
    with pytest.raises(C.ContinuationDataError, match="PERIOD_IDENTITY"):
        C.load(prepared[0], "short", 2, 6, period=period)
    corrupt = tmp_path / "corrupt"
    shutil.copytree(prepared[0], corrupt)
    name = next(iter(prepared[1]["files"]))
    with (corrupt / name).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(C.ContinuationDataError, match="FILE_DIGEST"):
        C.load(corrupt, "short", 2, 6)


def test_cd05_runtime_design_relocation_is_hash_checked(prepared, tmp_path, monkeypatch):
    _, manifest = prepared
    relative = manifest["source"]["base_design_relpath"]
    runtime = tmp_path / "runtime"
    for name in (*C.CODE_FILES, relative):
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(C.REPO / name, target)
    monkeypatch.setattr(C, "REPO", runtime)
    assert C.preflight(prepared[0], "long")["state"] == "PASS"
    with (runtime / relative).open("a") as handle:
        handle.write(" ")
    with pytest.raises(C.ContinuationDataError, match="SOURCE_DIGEST"):
        C.load(prepared[0], "long", 2, 13)


def test_cd05_resealed_calendar_cannot_change_authentic_periods(prepared, tmp_path):
    out = tmp_path / "forged"
    shutil.copytree(prepared[0], out)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest["calendar"]["market_calendar"]["version"] = "forged"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = C.sha256_json(manifest)
    (out / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(C.ContinuationDataError, match="SOURCE_CONTRACT_IDENTITY"):
        C.periods(out, "QUARTERLY")


def test_cd03_h72_projection_uses_group_train_statistics(prepared):
    full, scales, _ = C.load(prepared[0], "long", 2, 24)
    projected, reduced, meta = C.load(prepared[0], "long", 4, 6, horizons=[72])
    assert reduced["input_mean"] == scales["input_mean"] and reduced["std"] == scales["std"]
    assert reduced["naive_mae"] == [scales["naive_mae"][-1]]
    assert meta["group_scales_sha256"] == scales["scales_sha256"]
    for split in full:
        keep = full[split]["mask"][:, -1]
        np.testing.assert_array_equal(projected[split]["row_ids"], full[split]["row_ids"][keep])


def test_cd06_unsupported_training_head_is_typed_before_publication(source, tmp_path, monkeypatch):
    original = market_grid.build_market_table
    def build(bundle, design, **kwargs):
        table = original(bundle, design, **kwargs)
        if table.horizons[-1] == 72:
            table.label_ok[table.origin_ts < CUT, -1] = False
        return table
    monkeypatch.setattr(market_grid, "build_market_table", build)
    out = tmp_path / "refused"
    with pytest.raises(C.ContinuationDataError, match="HEAD_UNSUPPORTED:long:FIT:h72"):
        C.prepare(source, out, _synth.DESIGN_PATH)
    assert not out.exists()


def test_cd06_no_test_rows_or_global_fit_statistics(source, prepared, tmp_path):
    manifest = prepared[1]
    assert manifest["train_statistics"] == "ON_DEMAND_PER_FOLD_ONLY"
    assert all("scales" not in info for info in manifest["groups"].values())
    alt = tmp_path / "input"
    shutil.copytree(source, alt)
    with np.load(alt / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    ts[-1] = END
    _synth.rewrite_input(alt, ts=ts, x=x, price=price)
    with pytest.raises(data.DataContractError, match="TEST_ROWS_PRESENT"):
        C.prepare(alt, tmp_path / "refused", _synth.DESIGN_PATH)


def test_cd06_partial_week_empty_heads_allowed_and_h72_empty_period_skips(prepared):
    last = C.periods(prepared[0], "WEEKLY")[-1]
    arrays, _, _ = C.load(prepared[0], "long", 2, 13, period=last)
    assert len(arrays["VALIDATION"]["row_ids"]) > 0
    assert arrays["VALIDATION"]["mask"][:, -1].sum() == 0
    with pytest.raises(C.ContinuationDataSkip, match="EMPTY_VALIDATION_PERIOD"):
        C.load(prepared[0], "long", 2, 13, horizons=[72], period=last)
    assert C.preflight(prepared[0], "long")["state"] == "PASS"


@pytest.mark.parametrize("kind,code", [("std", "FEATURE_SCALE_INVALID"), ("naive", "NAIVE_SCALE_INVALID")])
def test_cd06_degenerate_train_scales_refused(source, tmp_path, kind, code):
    alt = tmp_path / "input"
    shutil.copytree(source, alt)
    with np.load(alt / "INPUT.npz", allow_pickle=False) as z:
        x, price = z["x"], z["price"]
    if kind == "std":
        x[:, 0] = 1
    else:
        price[:] = 1.1
    _synth.rewrite_input(alt, x=x, price=price)
    out = tmp_path / "master"
    C.prepare(alt, out, _synth.DESIGN_PATH)
    with pytest.raises(C.ContinuationDataError, match=code):
        C.preflight(out, "long")


def test_cd01_evaluation_splits_are_explicit_disjoint_and_pretest(source):
    design = _synth.load_design()
    bundle = data.load_input(source, design)
    legacy = market_grid.build_market_table(bundle, design)
    again = market_grid.build_market_table(bundle, design, evaluation_splits=None)
    assert legacy.identity == again.identity and "evaluation_splits" not in legacy.identity
    np.testing.assert_array_equal(legacy.label_ok, again.label_ok)
    broad = E.SplitBounds("FIT", datetime(2020, 1, 1, tzinfo=timezone.utc),
                          datetime(2025, 1, 1, tzinfo=timezone.utc),
                          label_available_by=datetime(2025, 1, 1, tzinfo=timezone.utc))
    table = market_grid.build_market_table(bundle, design, evaluation_splits=(broad,))
    assert table.identity["evaluation_splits"] == [broad.to_json()]
    assert set(table.counts) == {"FIT"}
    for splits in ((), (broad, broad), (E.SplitBounds("FIT", broad.start,
                    datetime(2025, 1, 2, tzinfo=timezone.utc), label_available_by=None),)):
        with pytest.raises(data.DataContractError, match="EVALUATION_SPLITS_INVALID"):
            market_grid.build_market_table(bundle, design, evaluation_splits=splits)
