"""Resolution data acceptance on sparse synthetic calendar fixtures, never fits.

RD01: generic designs/streamed group intersection; RD02: exact past inputs and
targets/business purge; RD03: group FIT-only statistics; RD04: H72 projection;
RD05: future invariance; RD06: authenticated replay, private output and no TEST.
"""
import json
import shutil

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import data, market_grid, resolution_data as R

H = 3600
MISSING = _synth.epoch("2024-01-09T13:00:00+00:00")
NONFINITE = _synth.epoch("2024-01-08T13:00:00+00:00")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = _synth.make_input(tmp_path_factory.mktemp("resolution-input") / "input")
    calendar = json.loads((root / "CALENDAR.json").read_text())
    weeks = calendar["market_calendar"]["weeks"]
    chosen = {"2019-12-22", "2019-12-29", "2020-01-05", "2023-10-22",
              "2023-10-29", "2023-11-05", "2023-11-12", "2024-01-07",
              "2024-01-14", "2024-12-22", "2024-12-29"}
    weeks = [w for w in weeks if w["week_id"] in chosen]
    stamps = _synth.bar_ends(weeks)
    with np.load(root / "INPUT.npz", allow_pickle=False) as z:
        keep = np.isin(z["ts"], stamps) & (z["ts"] != MISSING)
        values = {k: z[k][keep] for k in z.files}
    values["x"][values["ts"] == NONFINITE, 18] = np.nan
    _synth.rewrite_input(root, **values)
    (root / "CALENDAR.json").write_text(json.dumps(_synth.calendar_config(weeks)))
    return root


@pytest.fixture(scope="module")
def prepared(source, tmp_path_factory):
    out = tmp_path_factory.mktemp("resolution-output") / "population"
    manifest = R.prepare(source, out, _synth.DESIGN_PATH)
    return out, manifest


def _design(spacing, slots, heads):
    doc = _synth.load_design()
    doc["input"].update(history_span_hours=spacing * (slots - 1),
                        spacing_hours=spacing, observations=slots)
    doc["horizons_hours"] = list(heads)
    return data.validate_design(doc)


def test_rd01_all_treatments_share_exact_intersection(source, prepared):
    out, manifest = prepared
    assert manifest["test_read"] is False and manifest["fits_run"] == 0
    for group, slots in (("short", (13,)), ("long", (6, 9, 13, 18, 24))):
        heads = manifest["groups"][group]["horizons"]
        reference, scale, _ = R.load(out, group, 2, slots[0])
        expected = None
        for spacing in (2, 4):
            for size in slots:
                design = _design(spacing, size, heads)
                table = market_grid.build_market_table(data.load_input(source, design), design)
                mask = table.label_ok & table.input_ok[:, None]
                expected = mask.copy() if expected is None else expected & mask
                arrays, scales, meta = R.load(out, group, spacing, size)
                assert scales == scale
                assert meta["horizons"] == heads
                assert meta["features"] == list(data.load_input(source, design).features)
                for split, arr in arrays.items():
                    assert arr["x"].shape[1:] == (size, 19)
                    for key in ("row_ids", "mask", "origin_price", "target_price", "week_ids"):
                        np.testing.assert_array_equal(arr[key], reference[split][key])
        for split, arr in reference.items():
            sel = (table.split == split) & expected.any(axis=1)
            np.testing.assert_array_equal(arr["row_ids"], table.origin_ts[sel])
            np.testing.assert_array_equal(arr["mask"], expected[sel])


def test_rd02_exact_inputs_targets_closures_and_business(source, prepared):
    out, _ = prepared
    with np.load(source / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    saw_closure = False
    for group, spacing, slots in (("short", 4, 13), ("long", 2, 6), ("long", 4, 24)):
        arrays, _, meta = R.load(out, group, spacing, slots)
        design = _design(spacing, slots, meta["horizons"])
        table = market_grid.build_market_table(data.load_input(source, design), design)
        for split, arr in arrays.items():
            origins, mask = arr["row_ids"], arr["mask"]
            assert origins.dtype == np.int64 and mask.any(axis=1).all()
            np.testing.assert_array_equal(arr["origin_ts"], origins)
            instants = origins[:, None] + (arr["offsets_hours"] * H).astype(np.int64)
            np.testing.assert_array_equal(instants[:, -1], origins)
            assert (instants <= origins[:, None]).all() and not np.isin(instants, MISSING).any()
            indices = np.searchsorted(ts, instants)
            np.testing.assert_array_equal(ts[indices], instants)
            np.testing.assert_array_equal(arr["x"], x[indices])
            np.testing.assert_array_equal(arr["x"][:, -1], x[np.searchsorted(ts, origins)])
            assert np.isfinite(arr["x"]).all()
            flags = np.zeros(instants.shape, bool)
            flags[:, 1:] = np.diff(instants, axis=1) > spacing * H
            np.testing.assert_array_equal(flags, arr["closure_crossed"])
            saw_closure |= flags.any()
            ti = np.searchsorted(table.origin_ts, origins)
            assert (mask <= table.label_ok[ti]).all()
            for j, h in enumerate(meta["horizons"]):
                used = mask[:, j]
                targets = origins[used] + h * H
                ix = np.searchsorted(ts, targets)
                np.testing.assert_array_equal(ts[ix], targets)
                np.testing.assert_array_equal(arr["target_price"][used, j], price[ix])
                bound = next(b for b in data.split_bounds(design) if b.split_id == split)
                assert (targets < int(bound.end.timestamp())).all()
            assert np.array_equal(np.isnan(arr["target_price"]), ~mask)
    assert saw_closure


def test_rd02_missing_open_bar_and_any_nonfinite_feature_exclude_origin(prepared):
    for group in ("short", "long"):
        for spacing in (2, 4):
            arrays, _, _ = R.load(prepared[0], group, spacing, 13)
            origins = arrays["VALIDATION"]["row_ids"]
            assert MISSING + 4 * H not in origins
            assert NONFINITE + 4 * H not in origins


def test_rd01_only_one_market_table_is_retained(source, tmp_path, monkeypatch):
    import weakref
    original = market_grid.build_market_table
    previous = []
    def build(bundle, design):
        assert all(ref() is None for ref in previous)
        table = original(bundle, design)
        previous.append(weakref.ref(table))
        return table
    monkeypatch.setattr(market_grid, "build_market_table", build)
    R.prepare(source, tmp_path / "streamed", _synth.DESIGN_PATH)
    assert len(previous) == 12 and all(ref() is None for ref in previous)


def test_rd03_fit_only_deduplicated_group_union_and_naive(source, prepared):
    out, manifest = prepared
    with np.load(source / "INPUT.npz", allow_pickle=False) as z:
        ts, x = z["ts"], z["x"]
    for group, slots in (("short", (13,)), ("long", (6, 9, 13, 18, 24))):
        union = set()
        for spacing in (2, 4):
            for size in slots:
                arrays, scales, _ = R.load(out, group, spacing, size)
                fit = arrays["FIT"]
                union.update((fit["row_ids"][:, None] + fit["offsets_hours"] * H).astype(np.int64).ravel())
        bars = np.array(sorted(union), dtype=np.int64)
        assert len(bars) == manifest["groups"][group]["scaler_rows"]
        assert (bars < _synth.epoch("2023-11-06T00:00:00+00:00")).all()
        values = x[np.searchsorted(ts, bars)]
        np.testing.assert_allclose(scales["input_mean"], values.mean(axis=0), rtol=0, atol=1e-14)
        np.testing.assert_allclose(scales["std"], values.std(axis=0), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(scales["std"], scales["input_std"])
        for j, naive in enumerate(scales["naive_mae"]):
            mask = fit["mask"][:, j]
            expected = np.abs(fit["target_price"][mask, j] - fit["origin_price"][mask]).mean()
            assert naive == pytest.approx(expected, rel=1e-14)


def test_rd04_h72_is_fixed_support_projection_not_a_new_scaler(prepared):
    out, _ = prepared
    full, full_scales, _ = R.load(out, "long", 2, 13)
    reference = None
    for spacing in (2, 4):
        for slots in (6, 9, 13, 18, 24):
            arrays, scales, meta = R.load(out, "long", spacing, slots, horizons=[72])
            assert scales["horizons"] == meta["horizons"] == [72]
            assert scales["naive_mae"] == [full_scales["naive_mae"][-1]]
            assert scales["input_mean"] == full_scales["input_mean"]
            assert scales["std"] == full_scales["std"]
            assert meta["group_scales_sha256"] == full_scales["scales_sha256"]
            for split, arr in arrays.items():
                keep = full[split]["mask"][:, -1]
                np.testing.assert_array_equal(arr["row_ids"], full[split]["row_ids"][keep])
                assert arr["mask"].shape == (int(keep.sum()), 1)
                assert arr["mask"].all()
                if reference is not None:
                    np.testing.assert_array_equal(arr["row_ids"], reference[split]["row_ids"])
            reference = arrays


def test_rd05_future_perturbation_cannot_change_fit_or_past_inputs(source, prepared, tmp_path):
    alt = tmp_path / "input"
    shutil.copytree(source, alt)
    cutoff = _synth.epoch("2024-01-10T00:00:00+00:00")
    with np.load(alt / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    x[ts >= cutoff] += 1000
    price[ts >= cutoff] *= 1.5
    _synth.rewrite_input(alt, x=x, price=price)
    changed = tmp_path / "changed"
    R.prepare(alt, changed, _synth.DESIGN_PATH)
    for group in ("short", "long"):
        before, scales, _ = R.load(prepared[0], group, 2, 13)
        after, new_scales, _ = R.load(changed, group, 2, 13)
        assert new_scales == scales
        for split in ("FIT", "INNER"):
            for key in before[split]:
                np.testing.assert_array_equal(before[split][key], after[split][key])
        a, b = before["VALIDATION"], after["VALIDATION"]
        past = a["origin_ts"] < cutoff
        np.testing.assert_array_equal(a["x"][past], b["x"][past])
        assert not np.array_equal(a["target_price"], b["target_price"], equal_nan=True)


def test_rd06_corruption_rejected_and_prepare_idempotent(source, prepared, tmp_path):
    out, manifest = prepared
    assert R.prepare(source, out, _synth.DESIGN_PATH) == manifest
    copy = tmp_path / "corrupt"
    shutil.copytree(out, copy)
    filename = next(name for name in manifest["files"] if name.endswith(".npz"))
    with (copy / filename).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(R.ResolutionDataError, match="FILE_DIGEST"):
        R.load(copy, "short", 2, 13)
    body = json.loads((copy / "MANIFEST.json").read_text())
    body["test_read"] = True
    (copy / "MANIFEST.json").write_text(json.dumps(body))
    with pytest.raises(R.ResolutionDataError):
        R.load(copy, "short", 2, 13)


def test_rd06_runtime_checkout_relocation_preserves_design_hash(prepared, tmp_path, monkeypatch):
    out, manifest = prepared
    relative = manifest["source"]["base_design_relpath"]
    assert not relative.startswith("/") and "base_design_path" not in manifest["source"]
    runtime = tmp_path / "runtime"
    for filename in (*R.CODE_FILES, relative):
        destination = runtime / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(R.REPO / filename, destination)
    replica = tmp_path / "replica"
    shutil.copytree(out, replica)
    expected, expected_scales, _ = R.load(out, "long", 4, 24, horizons=[72])
    monkeypatch.setattr(R, "REPO", runtime)
    arrays, scales, _ = R.load(replica, "long", 4, 24, horizons=[72])
    assert scales == expected_scales
    np.testing.assert_array_equal(arrays["FIT"]["row_ids"], expected["FIT"]["row_ids"])
    with (runtime / relative).open("a") as handle:
        handle.write(" ")
    with pytest.raises(R.ResolutionDataError, match="SOURCE_DIGEST"):
        R.load(replica, "long", 4, 24, horizons=[72])


@pytest.mark.parametrize("filename", ["INPUT.npz", "CLOCK.json", "CALENDAR.json"])
def test_rd06_explicit_input_relocation_requires_identical_source_bytes(source, prepared, tmp_path, filename):
    relocated = tmp_path / "relocated-input"
    shutil.copytree(source, relocated)
    expected, expected_scales, expected_meta = R.load(prepared[0], "long", 2, 13, horizons=[72])
    arrays, scales, meta = R.load(prepared[0], "long", 2, 13, horizons=[72], input_dir=relocated)
    assert scales == expected_scales and meta == expected_meta
    assert meta["source_files"] == prepared[1]["source"]["files"]
    assert str(relocated) not in json.dumps(meta)
    for split in data.SPLITS:
        for key in expected[split]:
            np.testing.assert_array_equal(arrays[split][key], expected[split][key])
    with (relocated / filename).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(R.ResolutionDataError, match="SOURCE_DIGEST"):
        R.load(prepared[0], "long", 2, 13, horizons=[72], input_dir=relocated)
    # The bad replica must not be ignored in favor of the still-valid original.
    _, original_scales, original_meta = R.load(prepared[0], "long", 2, 13, horizons=[72])
    assert original_scales == expected_scales and original_meta == expected_meta


@pytest.mark.parametrize("kind,code", [("std", "FEATURE_SCALE_INVALID"),
                                      ("naive", "NAIVE_SCALE_INVALID")])
def test_rd03_degenerate_statistics_refused_before_publication(source, tmp_path, kind, code):
    alt = tmp_path / "input"
    shutil.copytree(source, alt)
    with np.load(alt / "INPUT.npz", allow_pickle=False) as z:
        x, price = z["x"], z["price"]
    if kind == "std":
        x[:, 0] = 1.0
    else:
        price[:] = 1.1
    _synth.rewrite_input(alt, x=x, price=price)
    out = tmp_path / "refused"
    with pytest.raises(R.ResolutionDataError, match=code) as exc:
        R.prepare(alt, out, _synth.DESIGN_PATH)
    assert exc.value.code == code and not out.exists()


@pytest.mark.parametrize("split", data.SPLITS)
def test_rd03_each_head_requires_support_in_every_split(source, tmp_path, monkeypatch, split):
    original = market_grid.build_market_table
    def build(bundle, design):
        table = original(bundle, design)
        table.label_ok[table.split == split, -1] = False
        return table
    monkeypatch.setattr(market_grid, "build_market_table", build)
    out = tmp_path / "refused"
    with pytest.raises(R.ResolutionDataError, match=f"HEAD_UNSUPPORTED:short:{split}:h12"):
        R.prepare(source, out, _synth.DESIGN_PATH)
    assert not out.exists()


@pytest.mark.parametrize("key,value,code", [("std", 0.0, "FEATURE_SCALE_INVALID"),
    ("std", float("nan"), "FEATURE_SCALE_INVALID"),
    ("std", float("inf"), "FEATURE_SCALE_INVALID"),
    ("std", -1.0, "FEATURE_SCALE_INVALID"),
    ("input_mean", float("inf"), "FEATURE_SCALE_INVALID"),
    ("naive_mae", 0.0, "NAIVE_SCALE_INVALID"),
    ("naive_mae", -1.0, "NAIVE_SCALE_INVALID"),
    ("naive_mae", float("nan"), "NAIVE_SCALE_INVALID"),
    ("naive_mae", float("inf"), "NAIVE_SCALE_INVALID"),
    ("naive_mae", None, "NAIVE_SCALE_INVALID")])
def test_rd03_statistics_gate_refuses_nonfinite_or_nonpositive(prepared, key, value, code):
    import copy
    scales = copy.deepcopy(prepared[1]["groups"]["long"]["scales"])
    scales[key][0] = value
    if key == "std":
        scales["input_std"][0] = value
    with pytest.raises(R.ResolutionDataError, match=code):
        R._validate_scales(scales, "long", R.GROUPS["long"]["horizons"])


def test_rd06_source_and_test_rows_rejected(source, prepared, tmp_path):
    alt = tmp_path / "input"
    shutil.copytree(source, alt)
    with np.load(alt / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    ts[-1] = _synth.epoch("2025-01-01T00:00:00+00:00")
    _synth.rewrite_input(alt, ts=ts, x=x, price=price)
    with pytest.raises(data.DataContractError, match="TEST_ROWS_PRESENT"):
        R.prepare(alt, tmp_path / "refused", _synth.DESIGN_PATH)
    with (alt / "CLOCK.json").open("a") as handle:
        handle.write(" ")
    copy = tmp_path / "source-changed"
    shutil.copytree(prepared[0], copy)
    body = json.loads((copy / "MANIFEST.json").read_text())
    body["source"]["input_dir"] = str(alt)
    body.pop("manifest_sha256")
    body["manifest_sha256"] = R.sha256_json(body)
    (copy / "MANIFEST.json").write_text(json.dumps(body))
    with pytest.raises(R.ResolutionDataError, match="SOURCE_DIGEST"):
        R.load(copy, "long", 2, 13)


@pytest.mark.parametrize("group,spacing,slots,heads", [("short", 2, 6, None),
    ("long", 1, 13, None), ("long", 2, 13, [2]), ("long", 2, 13, []),
    ("long", 2, 13, [72, 72]), ("TEST", 2, 13, None)])
def test_rd06_invalid_requests_refused(prepared, group, spacing, slots, heads):
    with pytest.raises(R.ResolutionDataError):
        R.load(prepared[0], group, spacing, slots, horizons=heads)
