"""RAW long matched control: synthetic-only contracts, no fit or TEST input."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import common_population as CP, data, design as D, market_grid, provenance

DOCS = Path(__file__).resolve().parents[3] / "docs/i6n/direct_multi_output_static"
ARMS = ("mg1l", "mg12m")
HEADS = (18, 24, 36, 48, 60, 72)
CAL = {"calendar_id": "SYNTH_FX_NY17_FIXTURE", "version": "synth-v1"}


def designs():
    base = _synth.load_design()
    dds = {a: CP.matched_data_design(base, a) for a in ARMS}
    exps = {a: D.matched_market_time_design(base, a, dds, calendar=CAL) for a in ARMS}
    return exps, dds


def test_long_seals_and_old_identity_bytes(tmp_path):
    assert tuple(D.MATCHED_SCOPE) == ("mg1", "mg2m")
    assert tuple(D.LONG_MATCHED_SCOPE) == ARMS
    assert set(D.ALL_MATCHED_SCOPE) == set(D.MATCHED_SCOPE) | set(ARMS)
    exps, dds = designs()
    for arm, slots, spacing in (("mg1l", 145, 1), ("mg12m", 13, 12)):
        doc = exps[arm]
        assert D.matched_arm(doc) == arm
        assert doc["horizons_hours"] == list(HEADS)
        assert doc["market_time"]["observations"] == slots
        assert doc["market_time"]["spacing_open_market_hours"] == spacing
        assert doc["market_time"]["span_open_market_hours"] == 144
        assert doc["sealed_identities"]["runtime"] == {"tensorflow": "2.21.0", "keras": "3.13.2"}
        effective = D.effective_config(doc)
        assert effective["matched_population_rules"]["contrast_id"] == D.LONG_MATCHED_CONTRAST_ID
        assert effective["model"]["slots"] == slots
        assert len(dds[arm]["features"]) == 19
    result = D.seal_matched_long(_synth.DESIGN_PATH, tmp_path / "long")
    assert result["contrast_id"] == D.LONG_MATCHED_CONTRAST_ID != D.MATCHED_CONTRAST_ID
    assert D.seal_matched_long(_synth.DESIGN_PATH, tmp_path / "long") == result
    D.seal_matched_short(_synth.DESIGN_PATH, tmp_path / "short")
    for name in ("DESIGN.mg1.json", "DESIGN.mg2m.json", "EXPERIMENT.mg1.json",
                 "EXPERIMENT.mg2m.json", "MATCHED_POPULATION_RULES.json"):
        assert (tmp_path / "short" / name).read_bytes() == (DOCS / "matched_short" / name).read_bytes()


def test_exact_supported_pairs_only(tmp_path):
    exps, dds = designs()
    for arms in (("mg1l",), ("mg1", "mg12m"), ("mg1l", "mg12m", "mg1")):
        with pytest.raises(CP.CommonPopulationError, match="ARMS_REQUIRED"):
            CP.build_common_population(tmp_path, {a: exps.get(a, {}) for a in arms},
                                       {a: dds.get(a, {}) for a in arms})
    bad = copy.deepcopy(dds["mg1l"])
    bad["horizons_hours"] = [2, 3, 4, 5, 6, 12]
    bad["data_identity_sha256"] = CP.data_identity_sha256(bad)
    with pytest.raises(CP.CommonPopulationError, match="MATCHED_DATA_DESIGN_INVALID"):
        CP.validate_matched_data_design(bad)


def test_published_long_designs_replay_production_seal(tmp_path):
    result = D.seal_matched_long(_synth.DESIGN_PATH, tmp_path)
    assert result["fits_run"] == 0
    for name in ("DESIGN.mg1l.json", "DESIGN.mg12m.json", "EXPERIMENT.mg1l.json",
                 "EXPERIMENT.mg12m.json", "MATCHED_POPULATION_RULES.json"):
        assert (tmp_path / name).read_bytes() == (DOCS / "matched_long" / name).read_bytes()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched-long")
    src = _synth.make_input(root / "input", missing={_synth.epoch("2022-06-15T13:00:00+00:00")})
    with np.load(src / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    keep = np.zeros(len(ts), dtype=bool)
    for start, end in (("2019-12-01", "2020-01-15"), ("2022-06-01", "2022-07-01"),
                       ("2023-11-01", "2023-12-01"), ("2024-03-01", "2024-04-01"),
                       ("2024-10-20", "2024-11-10"), ("2024-12-22", "2025-01-01")):
        keep |= (ts >= _synth.epoch(start + "T00:00:00+00:00")) & (ts < _synth.epoch(end + "T00:00:00+00:00"))
    _synth.rewrite_input(src, ts=ts[keep], x=x[keep], price=price[keep])
    exps, dds = designs()
    pop = CP.build_common_population(src, exps, dds, code_commit="SYNTHETIC")
    seal = CP.write_common_population(pop, root / "population")
    return src, exps, dds, pop, seal, root / "population"


def test_long_intersection_scaler_maturity_and_restart(built):
    src, exps, dds, pop, seal, out = built
    assert pop.horizons == HEADS
    tables = {a: market_grid.build_market_table(data.load_input(src, CP._elapsed_shim(dds[a])),
                                               CP._elapsed_shim(dds[a])) for a in ARMS}
    loaded = CP.load_common_population(out, expected_digest=pop.digest, input_dir=src)
    for name in CP.SPLIT_NAMES:
        a, b = (CP.arm_tensors(loaded, exps[arm], name) for arm in ARMS)
        sel = tables[ARMS[0]].split == CP.SOURCE_SPLIT[name]
        common = tables[ARMS[0]].label_ok[sel] & tables[ARMS[1]].label_ok[sel]
        assert np.array_equal(pop.splits[name].head_mask, common[common.any(axis=1)])
        assert np.array_equal(a.row_id, b.row_id)
        assert np.array_equal(a.head_mask, b.head_mask)
        assert np.array_equal(a.entry_mask, b.entry_mask)
        assert np.array_equal(a.management_mask, a.head_mask)
        assert a.x.shape[1:] == (145, 19) and b.x.shape[1:] == (13, 19)
        assert np.array_equal(a.slot_instants[:, ::12], b.slot_instants)
        for arm, arr in zip(ARMS, (a, b)):
            assert np.array_equal(arr.offsets_hours, (arr.slot_instants - arr.row_id[:, None]) / 3600)
            assert np.array_equal(arr.slot_instants[:, -1], arr.row_id)
            assert np.isnan(arr.target_price[~arr.head_mask]).all()
            assert CP.replay_against_input(arr, src)["status"] == "REPLAYED"
            original = CP.arm_tensors(pop, exps[arm], name)
            assert np.array_equal(arr.x, original.x)
        end = int(data.split_bounds(CP._elapsed_shim(dds[ARMS[0]]))[CP.SPLIT_NAMES.index(name)].end.timestamp())
        assert ((a.row_id[:, None] + np.asarray(HEADS) * 3600)[a.head_mask] < end).all()
        assert a.closure_crossed.any()
    bars = np.unique(np.concatenate([pop.arm_arrays[a]["FIT"]["slot_instants"].ravel() for a in ARMS]))
    with np.load(src / "INPUT.npz", allow_pickle=False) as z:
        xs = z["x"][np.searchsorted(z["ts"], bars)]
    np.testing.assert_array_equal(pop.scaler.mean, xs.mean(axis=0))
    np.testing.assert_array_equal(pop.scaler.std, xs.std(axis=0))
    assert set(pop.naive_loss_scale) == set(HEADS)
    assert set(seal["identities"]["arms"]) == set(ARMS)


def test_long_model_capacity_and_hourly_elapsed_view(built):
    from tools.i6dmo import model as M, train as T
    _, exps, _, pop, _, out = built
    records = []
    for arm in ARMS:
        eff = D.effective_config(exps[arm])
        config = M.normalize_model_config(eff["model"])
        assert M.market_family(config) == M.MATCHED_ARMS[arm]
        assert M.declared_parameter_counts(config) == M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]
        records.append({"schema": M.CAPACITY_SCHEMA, "arm": arm,
                        "counts": M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]})
        assert tuple(data.horizons(T._elapsed_view(exps[arm]))) == HEADS
        loaded = T.load_matched_population(out, pop.digest, HEADS, provenance.MANIFEST)
        arrays = T.load_matched_data(loaded, eff)
        scales, _ = T.matched_scales(loaded, eff, arrays["fit"])
        assert scales["naive_mae"] == [pop.naive_loss_scale[h] for h in HEADS]
        assert T.matched_arm(eff) == arm
    assert M.matched_capacity_delta(*records)["declared_head_delta"] == 25344
    bad = copy.deepcopy(D.effective_config(exps["mg1l"])["model"])
    bad["horizons_hours"] = [2, 3, 4, 5, 6, 12]
    with pytest.raises(ValueError, match="FIXED_VALUE"):
        M.normalize_model_config(bad)
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(records[0], {"schema": M.CAPACITY_SCHEMA, "arm": "mg2m",
                                             "counts": M.DECLARED_CAPACITY["short_mg2"]})


def test_long_real_graph_capacity_and_dynamic_context():
    from tools.i6dmo import model as M
    records = []
    for arm, family in (("mg1l", "long_mg1_matched"), ("mg12m", "long_mg12_matched")):
        bundle = M.build_dmo(M.model_config(provenance.MANIFEST, family=family))
        record = M.capacity_record(bundle)
        assert record["arm"] == arm
        assert record["counts"] == M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]
        records.append(record)
        slots = record["counts"]["slots"]
        spacing = 1 if arm == "mg1l" else 12
        offsets = np.arange(-(slots - 1) * spacing, 1, spacing, dtype=np.float32)[None, :]
        closure = np.zeros_like(offsets)
        x = np.zeros((1, slots, 19), dtype=np.float32)
        output = bundle.forecast_model([x, offsets, closure], training=False)
        assert len(output) == 6 and all(tuple(y.shape) == (1, 1) for y in output)
        encoder = bundle.forecast_model.get_layer("physical_time_encoder")
        fused = np.zeros((1, slots, 76), dtype=np.float32)
        plain = encoder([fused, offsets, closure]).numpy()
        shifted, crossed = offsets.copy(), closure.copy()
        shifted[:, :slots // 2] -= 48
        crossed[:, slots // 2] = 1
        actual = encoder([fused, shifted, crossed]).numpy()
        assert plain.shape == actual.shape == (1, slots, 78)
        assert not np.array_equal(plain, actual)
    assert M.matched_capacity_delta(*records)["delta"]["total"] == 25344


def test_long_causality_probe_rejects_lookahead_and_dropped_slots():
    from tools.i6dmo import model as M
    keras = M.keras
    grid = tuple(range(145))
    x = keras.Input((145, 1))
    future = keras.Model(x, keras.layers.Lambda(lambda a: M.tf.reverse(a, axis=[1]))(x))
    dropped = keras.Model(x, keras.layers.Lambda(lambda a: a * 0)(x))
    for model in (future, dropped):
        with pytest.raises(ValueError, match="time alignment violated"):
            M._probe_causal(model, grid, grid, "negative control")


def test_long_future_perturbation_does_not_refit_scaler_or_fit_population(built, tmp_path):
    src, exps, dds, pop, _, _ = built
    changed = tmp_path / "future-input"
    shutil.copytree(src, changed)
    with np.load(changed / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    future = ts >= _synth.epoch("2023-11-06T00:00:00+00:00")
    x[future] += 1e5
    price[future] += 10
    _synth.rewrite_input(changed, x=x, price=price)
    other = CP.build_common_population(changed, exps, dds, code_commit="SYNTHETIC")
    assert other.scaler.fit_row_digest == pop.scaler.fit_row_digest
    np.testing.assert_array_equal(other.scaler.mean, pop.scaler.mean)
    np.testing.assert_array_equal(other.scaler.std, pop.scaler.std)
    assert other.naive_loss_scale == pop.naive_loss_scale
    assert other.splits["FIT"].digest == pop.splits["FIT"].digest
    for arm in ARMS:
        np.testing.assert_array_equal(other.arm_arrays[arm]["FIT"]["x"], pop.arm_arrays[arm]["FIT"]["x"])


def test_long_identity_and_postload_tampering_refused(built, tmp_path):
    from tools.i6dmo import train as T
    src, exps, _, pop, _, out = built
    effective = D.effective_config(exps["mg1l"])
    arrays, scales, matched = T._prepare_matched(
        {"path": out, "expected_digest": pop.digest, "input_dir": src}, effective)
    assert matched["arm"] == "mg1l" and matched["slots"] == 145
    assert set(matched["naive_loss_scale"]) == {str(h) for h in HEADS}
    identity = {"common_population_sha256": pop.digest, "matched_arm": "mg1l"}
    assert T.verify_matched_identity(identity, pop.digest, "mg1l") == identity
    with pytest.raises(T.TrainDisposition, match="NOT_A_MATCHED_FIT"):
        T.verify_matched_identity(identity, pop.digest, "mg1")
    bad = copy.deepcopy(arrays["fit"])
    bad["target_price"][bad["mask"]] += 1
    with pytest.raises(T.TrainDisposition, match="NAIVE_SCALE_MISMATCH"):
        T.matched_scales(pop, effective, bad)
    copied = tmp_path / "population"
    shutil.copytree(out, copied)
    loaded = CP.load_common_population(copied, expected_digest=pop.digest)
    path = copied / "ARM_mg1l_FIT.npz"
    with np.load(path, allow_pickle=False) as z:
        changed = {k: z[k] for k in z.files}
    changed["x"][0, 0, 0] += 1
    np.savez(path, **changed)
    with pytest.raises(CP.CommonPopulationError, match="ARM_ARRAYS_MISMATCH"):
        CP.arm_tensors(loaded, exps["mg1l"], "FIT")
    with pytest.raises(CP.CommonPopulationError, match="ARM_ARRAYS_MISMATCH"):
        CP.load_common_population(copied, expected_digest=pop.digest)


def test_long_missing_bar_exclusion_is_typed_not_filled(built):
    _, _, _, pop, _, _ = built
    origin = _synth.epoch("2022-06-15T14:00:00+00:00")
    exc = pop.exclusion_arrays["FIT"]
    row = np.flatnonzero(exc["candidate_row_ids"] == origin).item()
    codes = pop.report["head_reason_codes"]
    assert "INPUT_INVALID_MG1L_ONLY" in [codes[c] for c in exc["head_reason"][row]]
    assert origin not in pop.splits["FIT"].origin_row_ids
    assert pop.report["mg12m_slots_subset_of_mg1l"] is True


def test_short_capacity_records_replay_frozen_snapshot():
    from tools.i6dmo import model as M
    old = json.loads((DOCS / "matched_short/CAPACITY_MATCHED.json").read_text())
    records = {}
    for arm in D.MATCHED_SCOPE:
        effective = D.effective_config(D.load_design(DOCS / f"matched_short/EXPERIMENT.{arm}.json"))
        records[arm] = M.capacity_record(M.build_dmo(effective["model"]))
        assert records[arm]["counts"] == old["arms"][arm]["counts"]
        assert records[arm]["capacity_sha256"] == old["arms"][arm]["capacity_sha256"]
    assert {M.MATCHED_ARMS[a]: M.DECLARED_CAPACITY[M.MATCHED_ARMS[a]] for a in D.MATCHED_SCOPE} == old["declared_capacity"]
    assert M.matched_capacity_delta(records["mg1"], records["mg2m"])["delta"] == old["delta"]
