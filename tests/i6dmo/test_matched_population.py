"""Matched short MG1/MG2 common population (NEXT_CONTRASTS_PLAN_20261015 section 2; SC01-SC04, SC10).

Synthetic fixtures only (``tests/i6dmo/_synth.py``): CPU, no fit, no GPU, no TEST bar, no DOIN.
Every assertion is a data/contract property, never a predictive measurement.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tests.i6dmo import _synth
from tools.i6dmo import common_population as CP, data, design as D, market_grid, provenance

UTC = timezone.utc
H = 3600
HEADS = (2, 3, 4, 5, 6, 12)
REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs/i6n/direct_multi_output_static"
MATCHED = DOCS / "matched_short"
FIXTURE_CAL = {"calendar_id": "SYNTH_FX_NY17_FIXTURE", "version": "synth-v1"}
RUNTIME = dict(D.MATCHED_RUNTIME)
MISSING = _synth.epoch("2022-06-15T13:00:00+00:00")          # a Wednesday open-session bar
#: Closed identities that must stay byte-identical (seals 4b64b4ee / 97541764).
CLOSED_FILES = {
    "EXPERIMENT.mg2.json": "89dea96d72430f3f3a1233c24698727a74da30a9f7e70cb4bbf4f5525a2f9cd5",
    "EXPERIMENT.mg12.json": "c5e3b3da91a3bfeba98d2908997bfa515d642c823f9422cbecff9542b2125b88",
    "EXPERIMENT.short_hourly.json": "5287aa059968ae1f3320b2d979e68d452a11822719cd29faf65b756ce30726d7",
    "market_grid/DESIGN.mg2.json": "b84b19a2286f66f930e9887f136a3d42bf068d6e87d81e2f19c8c9c8835b9cc6",
}


def _designs(calendar=FIXTURE_CAL, runtime=RUNTIME):
    short = _synth.load_design()
    dds = {a: CP.matched_data_design(short, a) for a in CP.ARMS}
    exps = {a: D.matched_market_time_design(short, a, dds, calendar=calendar, runtime=runtime)
            for a in CP.ARMS}
    return exps, dds


def _copy_input(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return _synth.make_input(tmp_path_factory.mktemp("mp") / "input", missing={MISSING})


@pytest.fixture(scope="module")
def designs():
    return _designs()


@pytest.fixture(scope="module")
def built(src, designs, tmp_path_factory):
    exps, dds = designs
    pop = CP.build_common_population(src, exps, dds, code_commit="TEST")
    out = tmp_path_factory.mktemp("pop") / "population"
    seal = CP.write_common_population(pop, out)
    return {"pop": pop, "seal": seal, "dir": out, "exps": exps, "dds": dds, "src": src}


def _tables(src, dds):
    out = {}
    for arm in CP.ARMS:
        shim = CP._elapsed_shim(dds[arm])
        out[arm] = market_grid.build_market_table(data.load_input(src, shim), shim)
    return out


def _err(exc_info):
    return getattr(exc_info.value, "code", None)


# =============================================================================== SC01

def test_sc01_arms_share_exact_common_masks(built):
    pop, exps = built["pop"], built["exps"]
    assert tuple(pop.splits) == CP.SPLIT_NAMES and pop.schema == CP.SCHEMA
    for name, sp in pop.splits.items():
        a = CP.arm_tensors(pop, exps["mg1"], name)
        b = CP.arm_tensors(pop, exps["mg2m"], name)
        for key in ("row_id", "head_mask", "entry_mask", "management_mask", "row_ids"):
            assert np.array_equal(getattr(a, key), getattr(b, key)), (name, key)
        assert np.array_equal(a.row_id, sp.origin_row_ids)
        assert np.array_equal(a.head_mask, sp.head_mask)
        assert np.array_equal(sp.management_mask, sp.head_mask)
        assert not (sp.entry_mask & ~sp.head_mask).any()
        assert sp.head_mask.any(axis=1).all()                       # >= 1 common head per origin
        assert np.all(np.diff(sp.origin_row_ids) > 0)                # ordered
        assert sp.counts == {h: int(sp.head_mask[:, j].sum()) for j, h in enumerate(HEADS)}
        assert a.x.shape[1:] == (25, 19) and b.x.shape[1:] == (13, 19)
        assert np.array_equal(np.isnan(a.target_price), ~a.head_mask)
        assert np.array_equal(a.target_price[a.head_mask], b.target_price[b.head_mask])


def test_sc01_population_is_the_exact_intersection(built):
    """Independent recomputation from the audited market tables of each arm."""
    pop = built["pop"]
    tables = _tables(built["src"], built["dds"])
    t1, t2 = tables["mg1"], tables["mg2m"]
    for name in CP.SPLIT_NAMES:
        sel = t1.split == CP.SOURCE_SPLIT[name]
        common = t1.label_ok[sel] & t2.label_ok[sel]
        keep = common.any(axis=1)
        sp = pop.splits[name]
        assert np.array_equal(t1.origin_ts[sel][keep], sp.origin_row_ids)
        assert np.array_equal(common[keep], sp.head_mask)
        before = pop.report["before_intersection"][name]
        for j, h in enumerate(HEADS):
            assert sp.counts[h] <= min(before["mg1"]["heads"][str(h)], before["mg2m"]["heads"][str(h)])
            assert before["mg1"]["heads"][str(h)] == int(t1.label_ok[sel][:, j].sum())
            assert before["mg2m"]["heads"][str(h)] == int(t2.label_ok[sel][:, j].sum())


def test_sc01_rebuild_is_identical_and_loader_verifies(built):
    rebuilt = CP.build_common_population(built["src"], built["exps"], built["dds"], code_commit="TEST")
    assert rebuilt.digest == built["pop"].digest
    loaded = CP.load_common_population(built["dir"], expected_digest=built["pop"].digest,
                                       input_dir=built["src"])
    assert loaded.digest == built["seal"]["population_digest"]
    for name in CP.SPLIT_NAMES:
        assert loaded.splits[name].digest == built["pop"].splits[name].digest
        assert np.array_equal(loaded.splits[name].head_mask, built["pop"].splits[name].head_mask)
    a = CP.arm_tensors(loaded, built["exps"]["mg1"], "INNER")
    b = CP.arm_tensors(built["pop"], built["exps"]["mg1"], "INNER")
    assert np.array_equal(a.x, b.x) and np.array_equal(a.slot_instants, b.slot_instants)


def _tamper_copy(built, tmp_path):
    dst = tmp_path / "pop"
    shutil.copytree(built["dir"], dst)
    return dst


def _rewrite_npz(path: Path, **changes):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for key, fn in changes.items():
        arrays[key] = fn(arrays[key].copy())
    np.savez(path, **arrays)


def test_sc01_changed_row_is_refused(built, tmp_path):
    dst = _tamper_copy(built, tmp_path)

    def shift(rid):
        rid[5] += H
        return rid
    _rewrite_npz(dst / "POPULATION_FIT.npz", origin_row_ids=shift)
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(dst, expected_digest=built["pop"].digest)
    assert _err(err) == "ROW_DIGEST_MISMATCH"


def test_sc01_changed_head_is_refused(built, tmp_path):
    dst = _tamper_copy(built, tmp_path)

    def flip(mask):
        r = int(np.flatnonzero(mask[:, 0] & mask[:, 1])[0])
        mask[r, 1] = False
        return mask
    _rewrite_npz(dst / "POPULATION_DEVELOPMENT.npz", head_mask=flip)
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(dst, expected_digest=built["pop"].digest)
    assert _err(err) == "HEAD_MASK_DIGEST_MISMATCH"


def test_sc01_changed_calendar_digest_is_refused(built, tmp_path):
    dst = _tamper_copy(built, tmp_path)
    body = json.loads((dst / "POPULATION.json").read_text())
    body["identities"]["calendar"]["sha256"] = "0" * 64
    (dst / "POPULATION.json").write_text(json.dumps(body))
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(dst, expected_digest=built["pop"].digest)
    assert _err(err) == "CALENDAR_DIGEST_MISMATCH"
    # re-digesting the edited identity only moves the failure to the sealed population digest
    body["identity_digests"]["calendar"] = provenance.canonical_sha256(body["identities"]["calendar"])
    body["identities"]["source"]["inputs"]["CALENDAR.json"] = "0" * 64
    body["identity_digests"]["source"] = provenance.canonical_sha256(body["identities"]["source"])
    (dst / "POPULATION.json").write_text(json.dumps(body))
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(dst, expected_digest=built["pop"].digest)
    assert _err(err) == "POPULATION_DIGEST_MISMATCH"


def test_sc01_changed_calendar_file_and_wrong_expected_digest_are_refused(built, tmp_path):
    alt = _copy_input(built["src"], tmp_path / "input")
    cal = json.loads((alt / "CALENDAR.json").read_text())
    cal["market_calendar"]["version"] = "synth-v2"
    (alt / "CALENDAR.json").write_text(json.dumps(cal))
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(built["dir"], expected_digest=built["pop"].digest, input_dir=alt)
    assert _err(err) == "CALENDAR_DIGEST_MISMATCH"
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(built["dir"], expected_digest="f" * 64)
    assert _err(err) == "POPULATION_DIGEST_MISMATCH"


def test_sc01_changed_arm_tensor_is_refused(built, tmp_path):
    dst = _tamper_copy(built, tmp_path)

    def bump(x):
        x[0, 0, 0] += 1.0
        return x
    _rewrite_npz(dst / "ARM_mg2m_FIT.npz", x=bump)
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(dst, expected_digest=built["pop"].digest)
    assert _err(err) == "ARM_ARRAYS_MISMATCH"


# =============================================================================== SC02

def test_sc02_last_input_is_t_and_every_slot_is_past(built):
    pop, exps = built["pop"], built["exps"]
    with np.load(built["src"] / "INPUT.npz") as z:
        ts, x = z["ts"], z["x"]
    pos = {int(v): i for i, v in enumerate(ts)}
    for arm in CP.ARMS:
        for name in CP.SPLIT_NAMES:
            t = CP.arm_tensors(pop, exps[arm], name)
            assert np.array_equal(t.slot_instants[:, -1], t.row_id)
            assert np.all(t.offsets_hours[:, -1] == 0.0) and np.all(t.offsets_hours <= 0.0)
            assert np.all(np.diff(t.slot_instants, axis=1) > 0)
            assert np.all(t.adjacent_delta_hours[:, 0] == 0) and np.all(t.closure_crossed[:, 0] == 0)
            assert np.array_equal(t.x[:, -1, :], x[[pos[int(v)] for v in t.row_id]])
        CP.replay_against_input(CP.arm_tensors(pop, exps[arm], "DEVELOPMENT"), built["src"])


def test_sc02_future_perturbation_changes_no_input_and_only_its_label(built, tmp_path):
    pop = built["pop"]
    dev = pop.splits["DEVELOPMENT"]
    r0 = int(np.flatnonzero(dev.head_mask.all(axis=1))[len(dev.origin_row_ids) // 3])
    t0 = int(dev.origin_row_ids[r0])
    alt = _copy_input(built["src"], tmp_path / "input")
    with np.load(alt / "INPUT.npz") as z:
        ts, x, price = z["ts"], z["x"].copy(), z["price"].copy()
    future = ts > t0
    x[future] = x[future] * 3.0 + 11.0                   # every feature bar after t, finite
    j3 = HEADS.index(3)
    q = int(np.flatnonzero(ts == t0 + 3 * H)[0])
    price[q] += 0.5                                      # ONLY the exact t0+3h price
    _synth.rewrite_input(alt, x=x, price=price)
    alt_pop = CP.build_common_population(alt, built["exps"], built["dds"], code_commit="TEST")
    for name in CP.SPLIT_NAMES:                          # masks depend on existence only
        assert np.array_equal(alt_pop.splits[name].head_mask, pop.splits[name].head_mask)
        assert np.array_equal(alt_pop.splits[name].origin_row_ids,
                              pop.splits[name].origin_row_ids)
    for arm in CP.ARMS:
        old = CP.arm_tensors(pop, built["exps"][arm], "DEVELOPMENT")
        new = CP.arm_tensors(alt_pop, built["exps"][arm], "DEVELOPMENT")
        upto = old.row_id <= t0
        assert np.array_equal(old.x[upto], new.x[upto])          # no input at or before t0 moved
        assert np.array_equal(old.origin_price[upto], new.origin_price[upto])
        assert new.target_price[r0, j3] == old.target_price[r0, j3] + 0.5
        others = [j for j in range(len(HEADS)) if j != j3]
        assert np.array_equal(new.target_price[r0, others], old.target_price[r0, others])
        assert new.origin_price[r0] == old.origin_price[r0]
        assert not np.array_equal(old.x[~upto], new.x[~upto])     # the perturbation acted
        assert np.array_equal(old.offsets_hours, new.offsets_hours)


# =============================================================================== SC03

def _row(pop, exps, arm, split, t):
    tens = CP.arm_tensors(pop, exps[arm], split)
    r = int(np.flatnonzero(tens.row_id == t)[0])
    return tens, r


@pytest.mark.parametrize(("origin", "closure_step_h", "mg1_span_h"), [
    ("2021-03-15T12:00:00+00:00", 48, 71),   # US DST start weekend: 47 h closure
    ("2021-03-22T12:00:00+00:00", 49, 72),   # ordinary weekend: 48 h closure
    ("2021-11-08T12:00:00+00:00", 50, 73),   # US DST end weekend: 49 h closure
])
def test_sc03_weekend_and_dst_slots_use_expected_open_hours(built, origin, closure_step_h, mg1_span_h):
    pop, exps = built["pop"], built["exps"]
    t = _synth.epoch(origin)
    split = "FIT"
    tens, r = _row(pop, exps, "mg1", split, t)
    steps = np.diff(tens.slot_instants[r]) // H
    assert sorted(set(steps.tolist())) == [1, closure_step_h]
    assert int((steps > 1).sum()) == 1
    k = int(np.flatnonzero(steps > 1)[0]) + 1
    assert tens.closure_crossed[r].tolist() == [1 if i == k else 0 for i in range(25)]
    assert tens.adjacent_delta_hours[r, k] == closure_step_h
    assert -tens.offsets_hours[r, 0] == mg1_span_h                # actual elapsed span varies
    bundle = data.load_input(built["src"], CP._elapsed_shim(built["dds"]["mg1"]))
    grid = market_grid.expected_market_hours(bundle)
    g = int(np.flatnonzero(grid == t)[0])
    assert np.array_equal(tens.slot_instants[r], grid[g - 24: g + 1])   # 24 expected open hours
    two, r2 = _row(pop, exps, "mg2m", split, t)
    assert np.array_equal(two.slot_instants[r2], grid[g - 24: g + 1: 2])
    assert set(two.slot_instants[r2].tolist()) <= set(tens.slot_instants[r].tolist())
    assert int(two.closure_crossed[r2].sum()) == 1
    assert -two.offsets_hours[r2, 0] == mg1_span_h


def test_sc03_missing_open_session_bar_stays_an_exclusion(built):
    pop = built["pop"]
    exc = pop.exclusion_arrays["FIT"]
    rid = exc["candidate_row_ids"]
    code = {c: i for i, c in enumerate(CP.HEAD_REASONS)}
    in_code = {c: i for i, c in enumerate(CP.INPUT_REASONS)}
    assert MISSING not in set(rid.tolist())                       # the absent bar is no origin
    odd = int(np.flatnonzero(rid == MISSING + H)[0])              # missing slot at offset -1
    even = int(np.flatnonzero(rid == MISSING + 2 * H)[0])         # missing slot at offset -2
    assert exc["input_reason_mg1"][odd] == in_code["INPUT_SLOT_NOT_OBSERVED"]
    assert exc["input_reason_mg2m"][odd] == in_code["INPUT_OK"]
    assert exc["input_reason_mg1"][even] == in_code["INPUT_SLOT_NOT_OBSERVED"]
    assert exc["input_reason_mg2m"][even] == in_code["INPUT_SLOT_NOT_OBSERVED"]
    assert set(exc["head_reason"][odd].tolist()) <= {code["INPUT_INVALID_MG1_ONLY"],
                                                     code["OPERATIONALLY_INELIGIBLE"]}
    assert code["INPUT_INVALID_MG1_ONLY"] in exc["head_reason"][odd].tolist()
    assert code["INPUT_INVALID_BOTH"] in exc["head_reason"][even].tolist()
    fit = pop.splits["FIT"].origin_row_ids
    assert MISSING + H not in set(fit.tolist()) and MISSING + 2 * H not in set(fit.tolist())
    # no retained window compresses over the gap: every mg1 window is 25 consecutive grid hours
    tens = CP.arm_tensors(pop, built["exps"]["mg1"], "FIT")
    assert MISSING not in set(tens.slot_instants.reshape(-1).tolist())
    bundle = data.load_input(built["src"], CP._elapsed_shim(built["dds"]["mg1"]))
    grid = market_grid.expected_market_hours(bundle)
    gpos = np.searchsorted(grid, tens.slot_instants)
    assert np.array_equal(grid[gpos], tens.slot_instants)
    assert np.all(np.diff(gpos, axis=1) == 1)
    summary = pop.report["exclusions"]["FIT"]["heads"]["2"]
    assert summary["INPUT_INVALID_MG1_ONLY"] >= 1 and summary["INPUT_INVALID_BOTH"] >= 1


# =============================================================================== SC04

def test_sc04_scaler_and_naive_are_fit_only_and_shared(built, tmp_path):
    pop = built["pop"]
    inner_start = int(data.split_bounds(CP._elapsed_shim(built["dds"]["mg1"]))[1].start.timestamp())
    alt = _copy_input(built["src"], tmp_path / "input")
    with np.load(alt / "INPUT.npz") as z:
        ts, x, price = z["ts"], z["x"].copy(), z["price"].copy()
    late = ts >= inner_start
    x[late] = x[late] * 5.0 - 3.0
    price[late] = price[late] * 1.3
    _synth.rewrite_input(alt, x=x, price=price)
    alt_pop = CP.build_common_population(alt, built["exps"], built["dds"], code_commit="TEST")
    assert np.array_equal(alt_pop.scaler.mean, pop.scaler.mean)
    assert np.array_equal(alt_pop.scaler.std, pop.scaler.std)
    assert alt_pop.scaler.fit_row_digest == pop.scaler.fit_row_digest
    assert alt_pop.scaler.n_rows == pop.scaler.n_rows
    assert alt_pop.naive_loss_scale == pop.naive_loss_scale
    assert alt_pop.splits["FIT"].digest == pop.splits["FIT"].digest
    # the perturbation was effective where it is allowed to act
    old = CP.arm_tensors(pop, built["exps"]["mg1"], "INNER")
    new = CP.arm_tensors(alt_pop, built["exps"]["mg1"], "INNER")
    assert not np.array_equal(old.x, new.x) and alt_pop.digest != pop.digest


def test_sc04_scaler_is_the_union_of_both_arms_fit_source_bars(built):
    pop, exps = built["pop"], built["exps"]
    a = CP.arm_tensors(pop, exps["mg1"], "FIT")
    b = CP.arm_tensors(pop, exps["mg2m"], "FIT")
    inst = np.concatenate([a.slot_instants.reshape(-1), b.slot_instants.reshape(-1)])
    vals = np.concatenate([a.x.reshape(-1, 19), b.x.reshape(-1, 19)])
    uniq, first = np.unique(inst, return_index=True)
    assert CP._scaler_rows_digest(uniq) == pop.scaler.fit_row_digest
    assert len(uniq) == pop.scaler.n_rows
    assert np.allclose(vals[first].mean(axis=0), pop.scaler.mean, rtol=0, atol=1e-12)
    assert np.allclose(vals[first].std(axis=0), pop.scaler.std, rtol=0, atol=1e-12)
    inner0 = int(pop.splits["INNER"].origin_row_ids.min())
    assert int(uniq.max()) < inner0
    # naive loss scale: FIT-only exact |p(t+h)-p(t)| on the common FIT mask
    for j, h in enumerate(HEADS):
        m = a.head_mask[:, j]
        want = float(np.abs(a.target_price[m, j] - a.origin_price[m]).sum() / m.sum())
        assert pop.naive_loss_scale[h] == pytest.approx(want, rel=1e-12)
    z = a.scaled_x(pop.scaler)
    assert np.array_equal(z, (a.x - pop.scaler.mean) / pop.scaler.std)


# =============================================================================== SC10

def test_sc10_no_test_rows_or_units(built):
    pop = built["pop"]
    test_start = _synth.epoch("2025-01-01T00:00:00+00:00")
    assert set(pop.splits) == {"FIT", "INNER", "DEVELOPMENT"}
    for name, sp in pop.splits.items():
        assert int(sp.origin_row_ids.max()) < test_start
        targets = sp.origin_row_ids[:, None] + np.asarray(HEADS) * H
        assert int(targets[sp.head_mask].max()) < test_start
    assert not [f for f in built["seal"]["files"] if "TEST" in f.upper()]
    body = json.loads((built["dir"] / "POPULATION.json").read_text())
    assert body["test_materialized"] is False and "TEST" not in body["splits"]
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.arm_tensors(pop, built["exps"]["mg1"], "TEST")
    assert _err(err) == "SPLIT_UNKNOWN"


def test_sc10_input_holding_a_test_bar_is_refused(designs, tmp_path):
    exps, dds = designs
    bad = _synth.make_input(tmp_path / "bad", last_sunday=datetime(2025, 1, 5, tzinfo=UTC),
                            data_end=_synth.epoch("2025-01-03T00:00:00+00:00"))
    with pytest.raises(data.DataContractError) as err:
        CP.build_common_population(bad, exps, dds)
    assert _err(err) == "TEST_ROWS_PRESENT"


def test_sc10_old_or_changed_designs_cannot_reuse_the_identity(built):
    pop, exps, dds = built["pop"], built["exps"], built["dds"]
    old_mg2 = json.loads((DOCS / "EXPERIMENT.mg2.json").read_text())
    hourly = json.loads((DOCS / "EXPERIMENT.short_hourly.json").read_text())
    for doc in (old_mg2, hourly):
        with pytest.raises(CP.CommonPopulationError) as err:
            CP.arm_tensors(pop, doc, "FIT")
        assert _err(err) == "MATCHED_IDENTITY_REQUIRED"
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.build_common_population(built["src"], {"mg1": exps["mg1"], "mg2m": old_mg2}, dds)
    assert _err(err) == "MATCHED_IDENTITY_REQUIRED"
    old_data = json.loads((DOCS / "market_grid/DESIGN.mg2.json").read_text())
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.build_common_population(built["src"], exps, {"mg1": dds["mg1"], "mg2m": old_data})
    assert _err(err) == "MATCHED_DATA_DESIGN_REQUIRED"
    # a validly re-sealed but CHANGED design (other runtime pin) is a different identity
    changed, _ = _designs(runtime={"tensorflow": "2.19.0", "keras": "3.9.0"})
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.arm_tensors(pop, changed["mg1"], "FIT")
    assert _err(err) == "DESIGN_IDENTITY_MISMATCH"
    # an edited, unsealed design is refused before any identity comparison
    edited = json.loads(json.dumps(exps["mg2m"]))
    edited["training"]["patience"] = 11
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.arm_tensors(pop, edited, "FIT")
    assert _err(err) in ("FIXED_VALUE", "MATCHED_DESIGN_SEAL")
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.build_common_population(built["src"], {"mg1": exps["mg2m"], "mg2m": exps["mg1"]}, dds)
    assert _err(err) == "ARM_MISMATCH"
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.build_common_population(built["src"], exps, {"mg1": dds["mg2m"], "mg2m": dds["mg2m"]})
    assert _err(err) == "ARM_UNKNOWN"


def test_sc10_changed_source_or_calendar_identity_is_refused(built, tmp_path):
    alt = _copy_input(built["src"], tmp_path / "input")
    with np.load(alt / "INPUT.npz") as z:
        price = z["price"].copy()
    price[0] += 1e-6
    _synth.rewrite_input(alt, price=price)
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(built["dir"], expected_digest=built["pop"].digest, input_dir=alt)
    assert _err(err) == "SOURCE_DIGEST_MISMATCH"
    real_cal, _ = _designs(calendar=D.MATCHED_CALENDAR)
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.build_common_population(built["src"], real_cal, built["dds"])
    assert _err(err) == "CALENDAR_IDENTITY_MISMATCH"


# =============================================================================== designs

def test_committed_matched_designs_validate_and_reseal_identically(tmp_path):
    for arm, fam, slots, spacing in (("mg1", "short_mg1_matched", 25, 1),
                                     ("mg2m", "short_mg2_matched", 13, 2)):
        report = D.validate_design_report(MATCHED / f"EXPERIMENT.{arm}.json")
        assert report["status"] == "PASS" and report["family"] == fam
        doc = D.load_design(MATCHED / f"EXPERIMENT.{arm}.json")
        assert D.matched_arm(doc) == arm
        mt = doc["market_time"]
        assert (mt["observations"], mt["spacing_open_market_hours"], mt["span_open_market_hours"]) \
            == (slots, spacing, 24)
        assert doc["seed"] == 6102026 and doc["horizons_hours"] == list(HEADS)
        assert doc["retraining_frequency"] == "NONE" and doc["model"]["regime"] == "R0"
        dd = json.loads((MATCHED / f"DESIGN.{arm}.json").read_text())
        CP.validate_matched_data_design(dd, arm=arm)
        assert doc["sealed_identities"]["data_identity_sha256"] == dd["data_identity_sha256"]
        assert doc["matched_population"]["rules_sha256"] == \
            D.MATCHED_POPULATION_RULES["rules_sha256"]
    for arm in ("mg1", "mg2m"):
        pins = json.loads((MATCHED / f"EXPERIMENT.{arm}.json").read_text())["sealed_identities"]["runtime"]
        assert pins == {"tensorflow": "2.21.0", "keras": "3.13.2"} == D.MATCHED_RUNTIME
    D.seal_matched_short(DOCS / "EXPERIMENT.short.json", tmp_path)       # default = execution pins
    for name in ("DESIGN.mg1.json", "DESIGN.mg2m.json", "EXPERIMENT.mg1.json",
                 "EXPERIMENT.mg2m.json", "MATCHED_POPULATION_RULES.json"):
        assert (tmp_path / name).read_bytes() == (MATCHED / name).read_bytes(), name


def test_closed_identities_are_byte_identical_and_still_their_own_families():
    for rel, sha in CLOSED_FILES.items():
        assert hashlib.sha256((DOCS / rel).read_bytes()).hexdigest() == sha, rel
    mg2 = D.load_design(DOCS / "EXPERIMENT.mg2.json")
    assert D.family(mg2) == "short_mg2" and mg2["seal_sha256"].startswith("4b64b4ee")
    mg12 = D.load_design(DOCS / "EXPERIMENT.mg12.json")
    assert mg12["seal_sha256"].startswith("97541764")
    hourly = D.load_design(DOCS / "EXPERIMENT.short_hourly.json")
    assert D.family(hourly) == "short_hourly"
    for doc in (mg2, mg12, hourly):
        with pytest.raises(D.DesignError) as err:
            D.matched_arm(doc)
        assert err.value.code == "MATCHED_IDENTITY_REQUIRED"


def test_relabelled_closed_mg2_is_not_a_matched_design():
    doc = json.loads((DOCS / "EXPERIMENT.mg2.json").read_text())
    doc["experiment_id"] = D.MG2_MATCHED_ID
    with pytest.raises(D.DesignError) as err:
        D.validate_design(doc)
    assert err.value.code == "MATCHED_SCHEMA_REQUIRED"
    doc["schema"] = D.MATCHED_SCHEMA
    with pytest.raises(D.DesignError) as err:
        D.validate_design(doc)
    assert err.value.code in ("MISSING_FIELD", "UNKNOWN_FIELD")


def test_superseded_runtime_seals_are_refused():
    """The 2.18.0/3.9.0 seals were superseded before execution and can never be re-derived."""
    short = D.load_design(DOCS / "EXPERIMENT.short.json")
    dds = {a: CP.matched_data_design(short, a) for a in CP.ARMS}
    for arm in CP.ARMS:
        with pytest.raises(D.DesignError) as err:
            D.matched_market_time_design(short, arm, dds,
                                         runtime={"tensorflow": "2.18.0", "keras": "3.9.0"})
        assert err.value.code == "MATCHED_IDENTITY_SUPERSEDED"
    note = json.loads((MATCHED / "SUPERSEDED_BEFORE_EXECUTION.json").read_text())
    assert note["status"] == "SUPERSEDED_BEFORE_EXECUTION"
    assert {v["old_seal_sha256"] for v in note["designs"].values()} == set(D.MATCHED_SUPERSEDED_SEALS)
    for arm in CP.ARMS:
        assert note["designs"][arm]["new_seal_sha256"] == \
            json.loads((MATCHED / f"EXPERIMENT.{arm}.json").read_text())["seal_sha256"]
    for arm in CP.ARMS:
        cur = D.load_design(MATCHED / f"EXPERIMENT.{arm}.json")
        assert cur["seal_sha256"] not in D.MATCHED_SUPERSEDED_SEALS


def test_d3_accessor_binds_to_the_verified_load_not_stored_strings(built, tmp_path):
    """Review D3 / p8 h: after a verified load, a tampered ARM file whose stored digest string was
    also updated must be refused by arm_tensors, exactly as a fresh verified load refuses it."""
    dst = _tamper_copy(built, tmp_path)
    pop = CP.load_common_population(dst, expected_digest=built["pop"].digest, input_dir=built["src"])
    clean = CP.arm_tensors(pop, built["exps"]["mg1"], "FIT")
    with np.load(dst / "ARM_mg1_FIT.npz", allow_pickle=False) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["x"][0, 0, 0] += 1.0
    np.savez(dst / "ARM_mg1_FIT.npz", **arrs)
    body = json.loads((dst / "POPULATION.json").read_text())
    body["splits"]["FIT"]["arms"]["mg1"]["arrays_sha256"] = provenance.array_sha256(arrs)
    (dst / "POPULATION.json").write_text(json.dumps(body, indent=1, sort_keys=True))
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.arm_tensors(pop, built["exps"]["mg1"], "FIT")
    assert _err(err) == "ARM_ARRAYS_MISMATCH"
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.load_common_population(dst, expected_digest=built["pop"].digest, input_dir=built["src"])
    assert _err(err) == "POPULATION_DIGEST_MISMATCH"
    assert clean.x[0, 0, 0] == arrs["x"][0, 0, 0] - 1.0
    unverified = CP.CommonPopulation(splits=pop.splits, scaler=pop.scaler,
                                     naive_loss_scale=pop.naive_loss_scale,
                                     identities=pop.identities, digest=pop.digest,
                                     source_dir=str(dst))
    with pytest.raises(CP.CommonPopulationError) as err:
        CP.arm_tensors(unverified, built["exps"]["mg2m"], "INNER")
    assert _err(err) == "POPULATION_NOT_VERIFIED"
