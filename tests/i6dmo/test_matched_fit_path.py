"""Matched MG1/MG2 short control, MODEL lane: SC05, SC06, SC07 and the CPU half of SC11.

Runs against the REAL data-lane module ``tools/i6dmo/common_population.py`` (no shim): a tiny
synthetic common population is built with ``build_common_population`` from a synthetic input
whose calendar covers 2020-2024 but whose bars exist only in a few weeks per split (every other
expected open-market hour is a counted exclusion), with both arms' designs sealed by
``design.matched_market_time_design`` under the synthetic fixture calendar. The fits go through
the REAL trainer CLI (``python -m tools.i6dmo.train``) and ``train_unit``; the reports through
the REAL report hook (``tools.i6dmo.reporting.run_unit``). The committed sealed designs
(matched_short/EXPERIMENT.mg1.json / EXPERIMENT.mg2m.json) are used for the model builds.
No injected arrays loader, no fake trainer, no GPU, no TEST bar, two epochs on CPU.
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6dmo import common_population as CP, design as D, market_grid, model as M, reporting, train as T  # noqa: E402,E501
from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4  # noqa: E402
from tests.i6dmo import _synth  # noqa: E402
from tests.i6dmo.dmo_model_fixtures import DESIGN  # noqa: E402

UTC = timezone.utc
REPO = DESIGN.parents[3]
MATCHED = DESIGN.parent / "matched_short"
SEALED = {"mg1": MATCHED / "EXPERIMENT.mg1.json", "mg2m": MATCHED / "EXPERIMENT.mg2m.json"}
CLOSED_MG2 = DESIGN.parent / "EXPERIMENT.mg2.json"
CAPACITY_DOC = MATCHED / "CAPACITY_MATCHED.json"
FEATURES = list(MANIFEST_NO_STLFSI4)
HORIZONS = [2, 3, 4, 5, 6, 12]
OVERRIDES = {"max_epochs": 2}
FIXTURE_CAL = {"calendar_id": "SYNTH_FX_NY17_FIXTURE", "version": "synth-v1"}
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
#: Weeks (Sunday ids) that carry bars; every other calendar week is an unobserved session.
OBSERVED = (("2019-12-29", "2020-01-12"), ("2023-11-05", "2023-11-19"),
            ("2024-01-07", "2024-01-21"), ("2024-12-29", "2024-12-29"))
TEST_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())


def _sha(path):
    return T.file_sha256(path)


def make_sparse_input(directory, seed=11):
    directory.mkdir(parents=True)
    weeks = _synth.calendar_weeks(datetime(2019, 12, 29, tzinfo=UTC), datetime(2024, 12, 29, tzinfo=UTC))
    keep = {w["week_id"] for w in weeks for a, b in OBSERVED if a <= w["week_id"] <= b}
    ts = _synth.bar_ends([w for w in weeks if w["week_id"] in keep])
    ts = ts[ts < TEST_START]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((len(ts), len(FEATURES)))
    price = 1.10 + np.cumsum(rng.standard_normal(len(ts)) * 1e-3)
    np.savez(directory / "INPUT.npz", ts=ts, x=x, price=price)
    digest = hashlib.sha256((directory / "INPUT.npz").read_bytes()).hexdigest()
    (directory / "CLOCK.json").write_text(json.dumps({
        "schema": "i6n.source_clock.v1", "status": "RESOLVED", "features": FEATURES,
        "input_sha256": digest, "timestamp_semantics": "UTC_BAR_END",
        "availability_policy": "AT_TIMESTAMP", "data_kind": "synthetic_fixture"}, indent=1))
    (directory / "CALENDAR.json").write_text(json.dumps(_synth.calendar_config(weeks), indent=1))
    return directory


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched")
    src = make_sparse_input(root / "input")
    short = _synth.load_design()
    dds = {a: CP.matched_data_design(short, a) for a in CP.ARMS}
    designs = {}
    for arm in CP.ARMS:
        doc = D.matched_market_time_design(short, arm, dds, calendar=FIXTURE_CAL)
        designs[arm] = root / f"EXPERIMENT.{arm}.json"
        designs[arm].write_text(json.dumps(doc, indent=1, sort_keys=True))
    exps = {a: json.loads(designs[a].read_text()) for a in CP.ARMS}
    seal = CP.write_common_population(CP.build_common_population(src, exps, dds, code_commit="TEST"),
                                      root / "population")
    digest = seal["population_digest"]
    pop = CP.load_common_population(root / "population", expected_digest=digest, input_dir=src)
    return {"root": root, "input": src, "designs": designs, "exps": exps,
            "population": root / "population", "digest": digest, "pop": pop}


def _cp(world):
    return {"path": str(world["population"]), "expected_digest": world["digest"],
            "input_dir": str(world["input"])}


def _cli(world, arm, out, *, design=None, overrides=True):
    env = dict(ENV)
    if overrides:
        env["I6DMO_TEST_TRAINING_OVERRIDES"] = json.dumps(OVERRIDES)
    return subprocess.run(
        [sys.executable, "-m", "tools.i6dmo.train", "--design", str(design or world["designs"][arm]),
         "--out", str(out), "--device", "cpu", "--json",
         "--common-population", str(world["population"]),
         "--common-population-digest", world["digest"], "--input-dir", str(world["input"])],
        capture_output=True, text=True, cwd=REPO, check=False, env=env)


def _fit(world, arm, out):
    """In-process train_unit (same function the CLI runs); used where the accessor is wrapped."""
    return T.train_unit(world["designs"][arm], None, out, "cpu", training_overrides=OVERRIDES,
                        common_population=_cp(world))


@pytest.fixture(scope="module")
def fits(world):
    """One matched fit per arm through the REAL trainer CLI (fresh flags-first interpreters)."""
    out = {}
    for arm in CP.ARMS:
        target = world["root"] / f"fit_{arm}" / "out" / "train"
        run = _cli(world, arm, target)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        out[arm] = target
    out["receipts"] = {arm: json.loads((out[arm] / "RECEIPT.json").read_text()) for arm in CP.ARMS}
    return out


def _stored_terminal(fit_out):
    """The ops store layout: <folder>/TERMINAL.json wrapping the terminal, files under out/."""
    folder = Path(fit_out).parents[1]
    receipt = json.loads((Path(fit_out) / "RECEIPT.json").read_text())
    artifacts = {rel: {"path": f"train/{rel}", "sha256": sha}
                 for rel, sha in receipt["artifacts"].items()}
    artifacts["RECEIPT"] = {"path": "train/RECEIPT.json", "sha256": _sha(Path(fit_out) / "RECEIPT.json")}
    stored = {"stored_at": 1760000000, "identity_sha256": "f" * 64,
              "terminal": {"artifacts": artifacts, "trainer_identity": receipt["identity"],
                           "scientific": receipt["scientific"]}}
    (folder / "TERMINAL.json").write_text(json.dumps(stored))
    return folder / "TERMINAL.json"


def _spec(world, arm, terminal, design=None):
    return {"unit_id": f"{arm}.report", "kind": "report", "family": "short",
            "identity_sha256": "r" * 64, "campaign_id": "matched-short-model-lane-test",
            "cadence": "NONE", "test_open": False,
            "effective_config": {"design": design or world["exps"][arm], "code_commit": "c" * 40},
            "upstream": {f"{arm}.fit": {"path": str(terminal), "sha256": _sha(terminal)}},
            "common_population": {"path": str(world["population"]), "sha256": world["digest"]},
            "matched": {"arm": arm, "fit_unit": f"{arm}.fit", "require_scientific_fit": False,
                        "input": {"path": str(world["input"]),
                                  "sha256": _sha(Path(world["input"]) / "INPUT.npz")}}}


def _report(world, arm, fit_out, out):
    return reporting.run_unit(_spec(world, arm, _stored_terminal(fit_out)), out,
                              out.parent / f"{out.name}.progress")


@pytest.fixture(scope="module")
def reports(world, fits):
    out = {}
    for arm in CP.ARMS:
        target = world["root"] / f"report_{arm}"
        result = _report(world, arm, fits[arm], target)
        assert result["status"] == "COMPLETE", result
        out[arm] = {"result": result, "metrics": json.loads((target / "METRICS.json").read_text())}
    return out


# ============================================================================ SC07 capacity

def _eff(arm):
    return T.effective_from_design(SEALED[arm])


def test_sc07_real_sealed_builds_three_inputs_78_channels_and_2304_delta():
    records = {}
    for arm, slots, family in (("mg1", 25, "short_mg1_matched"), ("mg2m", 13, "short_mg2_matched")):
        eff = _eff(arm)
        assert eff["family"] == family and T.matched_arm(eff) == arm
        assert M.normalize_model_config(eff["model"]) == M.model_config(FEATURES, family=family)
        bundle = M.build_dmo(eff["model"])                    # the exact call train_unit makes
        model = bundle.forecast_model
        assert [tuple(t.shape[1:]) for t in model.inputs] == [(slots, 19), (slots,), (slots,)]
        # MEASURED, not assumed: 19 x 4 encoded value channels + 2 context channels.
        assert tuple(bundle.core_model.input_shape) == (None, slots, 78)
        encoder = model.get_layer("physical_time_encoder")
        assert encoder.min_adjacent_hours == float(eff["design"]["market_time"]["spacing_open_market_hours"])
        assert len(model.outputs) == 6
        records[arm] = M.capacity_record(bundle)
        assert records[arm]["arm"] == arm
        literal = M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]
        assert {k: records[arm]["counts"][k] for k in literal} == literal
    delta = M.matched_capacity_delta(records["mg1"], records["mg2m"])
    assert delta["delta"] == {"branches": 0, "fusion": 0, "time_encoder": 0, "core": 0,
                              "heads": 2304, "total": 2304}
    assert (25 - 13) * 4 * 8 * 6 == M.MATCHED_HEAD_DELTA == 2304
    assert records["mg1"]["counts"]["total"] == 7186 and records["mg2m"]["counts"]["total"] == 4882
    # The committed machine-readable disclosure is exactly what the real sealed builds measure.
    doc = json.loads(CAPACITY_DOC.read_text())
    for arm in CP.ARMS:
        assert doc["arms"][arm]["counts"] == records[arm]["counts"]
        assert doc["arms"][arm]["capacity_sha256"] == records[arm]["capacity_sha256"]
        assert doc["arms"][arm]["sealed_design_sha256"] == _eff(arm)["design_sha256"]
    assert doc["delta"] == delta["delta"]
    assert doc["declared_capacity"] == {name: M.DECLARED_CAPACITY[name]
                                         for name in doc["declared_capacity"]}


def _wide_head(*, input_shape, horizon_hours, name, params):
    keras = tf.keras
    inputs = keras.Input(input_shape)
    x = keras.layers.Flatten(name=name + "_flatten")(inputs)
    x = keras.layers.Dense(9, activation="relu", name=name + "_hidden")(x)
    x = keras.layers.Dense(1, name=name + "_out")(x)
    return keras.Model(inputs, x, name=name)


def _wide_core(*, input_shape, time_grid, name, params):
    keras = tf.keras
    inputs = keras.Input(input_shape)
    x = keras.layers.Conv1D(8, 5, padding="causal", activation="relu", name="core_conv_0")(inputs)
    x = keras.layers.Conv1D(4, 3, padding="causal", activation="relu", name="core_conv_1")(x)
    return M.TemporalComponent(keras.Model(inputs, x, name=name), tuple(time_grid))


@pytest.mark.parametrize("patch", ["head", "core", "literal"])
def test_sc07_an_unreported_capacity_change_is_refused(monkeypatch, patch):
    if patch == "head":
        monkeypatch.setattr(M, "dmo_head", _wide_head)
    elif patch == "core":
        monkeypatch.setattr(M, "dmo_core", _wide_core)
    else:
        changed = copy.deepcopy(M.DECLARED_CAPACITY)
        changed["short_mg2"]["per_head"] = 434
        monkeypatch.setattr(M, "DECLARED_CAPACITY", changed)
    bundle = M.build_dmo(_eff("mg2m")["model"])
    with pytest.raises(M.CapacityError, match="CAPACITY_UNREPORTED_CHANGE"):
        M.capacity_record(bundle)


def test_sc07_capacity_delta_refuses_anything_but_the_declared_head_difference():
    a = M.capacity_record(M.build_dmo(_eff("mg1")["model"]))
    b = M.capacity_record(M.build_dmo(_eff("mg2m")["model"]))
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(b, a)                       # arms swapped
    tampered = copy.deepcopy(a)
    tampered["counts"]["core"] += 8
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(tampered, b)


def test_sc07_the_trainer_refuses_a_capacity_change_typed_before_fitting(world, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "dmo_head", _wide_head)
    receipt = _fit(world, "mg1", tmp_path / "unit")
    assert receipt["status"] == "FAILED_TYPED" and receipt["code"] == "CAPACITY_UNREPORTED_CHANGE"
    assert not (tmp_path / "unit" / "TRAINING.json").exists()


# ============================================================================ time encoding seam

def _ladder(rows, slots, spacing, *, start=None):
    offsets = np.tile(np.arange(-(slots - 1), 1, dtype=np.float64) * spacing, (rows, 1))
    if start is not None:
        offsets = np.tile(np.arange(start, start + slots, dtype=np.float64), (rows, 1))
    return offsets


def _arrays(offsets):
    rows, slots = offsets.shape
    return {"x": np.zeros((rows, slots, 19)), "offsets_hours": offsets,
            "closure_crossed": np.zeros((rows, slots))}


def test_25_slot_ladder_passes_the_sealed_spacing_seam_and_bad_ladders_refuse(world):
    mg1, mg2m = _eff("mg1"), _eff("mg2m")
    assert T.market_spacing_hours(mg1) == 1.0 and T.market_spacing_hours(mg2m) == 2.0
    weekend = _ladder(2, 25, 1.0)
    weekend[1, :10] -= 48.0                                   # a weekend between slots 9 and 10
    flags = np.zeros((2, 25))
    flags[1, 10] = 1
    ok = T.market_time_inputs({**_arrays(weekend), "closure_crossed": flags}, design=mg1)
    assert [a.shape for a in ok] == [(2, 25, 19), (2, 25), (2, 25)]
    assert len(T.market_time_inputs(_arrays(_ladder(2, 13, 2.0)), design=mg2m)) == 3
    # The REAL population's per-row actual offsets (closures, DST) pass at 1 h and at 2 h.
    for arm, eff in (("mg1", mg1), ("mg2m", mg2m)):
        tensors = CP.arm_tensors(world["pop"], world["exps"][arm], "FIT")
        arrays = {"x": tensors.x, "offsets_hours": tensors.offsets_hours,
                  "closure_crossed": tensors.closure_crossed}
        assert len(T.market_time_inputs(arrays, design=eff)) == 3
        assert (np.diff(tensors.offsets_hours, axis=1) > float(
            eff["design"]["market_time"]["spacing_open_market_hours"])).any()   # gaps exist
    refusals = [
        (_ladder(2, 25, 1.0, start=0), mg1),                 # ordinal slot indices 0..24
        (_ladder(2, 25, 0.5), mg1),                          # compressed ladder under MG1
        (_ladder(2, 13, 1.0), mg2m),                         # ordinal 13-slot ladder under MG2
        (_ladder(2, 25, 1.0), mg2m),                         # an MG1 ladder under the MG2 design
    ]
    for offsets, eff in refusals:
        with pytest.raises(T.TrainDisposition, match="ORDINAL_OR_INVALID_OFFSETS"):
            T.market_time_inputs(_arrays(offsets), design=eff)
    model = M.build_dmo(mg1["model"]).forecast_model
    x = np.zeros((2, 25, 19), np.float32)
    pred = model([x, weekend.astype(np.float32), flags.astype(np.float32)], training=False)
    assert all(np.isfinite(np.asarray(p)).all() for p in pred)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="ORDINAL_OR_INVALID_OFFSETS"):
        model([x, _ladder(2, 25, 0.5).astype(np.float32), np.zeros((2, 25), np.float32)], training=False)


# ============================================================================ SC06 fit path

def test_sc06_both_real_fit_paths_consume_the_frozen_common_population(world, fits):
    population = world["pop"]
    scales = {}
    for arm in CP.ARMS:
        out, receipt = fits[arm], fits["receipts"][arm]
        assert receipt["status"] == "COMPLETED" and receipt["scientific"] is False
        block = receipt["matched"]
        assert block["arm"] == arm and block["common_population_sha256"] == world["digest"]
        assert block["population_source_verified"] is True
        assert receipt["identity"]["common_population_sha256"] == world["digest"]
        assert receipt["identity"]["matched_arm"] == arm
        assert block["arm_source"]["kind"] == "common_population.arm_tensors"
        assert block["population_arm_identity"]["experiment_design_sha256"] == \
            receipt["identity"]["design_sha256"]
        training = json.loads((out / "TRAINING.json").read_text())
        assert training["rows"]["fit_trainable"] == len(population.splits["FIT"].origin_row_ids)
        assert training["rows"]["inner_trainable"] == len(population.splits["INNER"].origin_row_ids)
        assert training["rows"]["fit_without_valid_head"] == 0
        record = training["monitor_evaluation"]
        # The checkpoint monitor read exactly the common FIT/INNER per-head supports.
        for name, key in (("FIT", "fit"), ("INNER", "inner")):
            assert record[key]["counts"] == [int(c) for c in population.splits[name].head_mask.sum(0)]
        assert "validation" not in record and "development" not in record
        for name, key in T.POPULATION_SPLITS.items():
            sp = population.splits[name]
            with np.load(out / "predictions" / f"{key}.npz") as z:
                assert np.array_equal(z["row_id"], sp.origin_row_ids)
                assert np.array_equal(z["mask"], sp.head_mask)
                assert np.isnan(z["target_price"][~sp.head_mask]).all()
        scales[arm] = json.loads((out / "SCALES.json").read_text())
    # ONE shared FIT-only scaler and naive loss scale: identical across arms, equal to the
    # population's frozen statistics, never an arm's own refit.
    assert scales["mg1"] == scales["mg2m"]
    assert scales["mg1"]["input_mean"] == population.scaler.mean.tolist()
    assert scales["mg1"]["input_std"] == population.scaler.std.tolist()
    assert scales["mg1"]["naive_mae"] == [population.naive_loss_scale[h] for h in HORIZONS]
    arm_only = T.fit_scales(T.load_matched_data(population, _eff_world(world, "mg2m"))["fit"])
    assert arm_only["input_mean"] != scales["mg2m"]["input_mean"]
    assert fits["receipts"]["mg1"]["identity"]["scales_sha256"] == \
        fits["receipts"]["mg2m"]["identity"]["scales_sha256"]
    settings = json.loads((fits["mg1"] / "TRAINING.json").read_text())["settings"]
    assert (settings["learning_rate"], settings["weight_decay"], settings["batch_size"],
            settings["patience"], settings["min_delta"]) == (1e-3, 1e-4, 256, 10, 1e-8)
    for arm in CP.ARMS:
        assert _eff(arm)["training"]["max_epochs"] == 200 and _eff(arm)["seed"] == 6102026


def _eff_world(world, arm):
    return T.effective_from_design(world["designs"][arm])


def _wrap_accessor(monkeypatch, edit):
    """Perturb what the REAL accessor returns (the arm's tensors), nothing else."""
    real = CP.arm_tensors

    def wrapped(pop, design, split):
        tensors = real(pop, design, split)
        target = np.array(tensors.target_price, copy=True)
        edit(split, target, np.asarray(tensors.head_mask, bool))
        return dataclasses.replace(tensors, target_price=target)
    monkeypatch.setattr(CP, "arm_tensors", wrapped)


def test_sc06_masked_out_labels_never_reach_loss_monitor_predictions_or_score(
        world, fits, reports, tmp_path, monkeypatch):
    def fill_masked_out(split, target, mask):
        assert (~mask).any(), split                      # masked-out heads exist in every split
        target[~mask] = 9.99                             # a finite label OUTSIDE the common mask
    _wrap_accessor(monkeypatch, fill_masked_out)
    out = tmp_path / "fit" / "out" / "train"
    receipt = _fit(world, "mg1", out)
    assert receipt["status"] == "COMPLETED", receipt
    base = json.loads((fits["mg1"] / "TRAINING.json").read_text())
    new = json.loads((out / "TRAINING.json").read_text())
    assert [r["monitored"] for r in new["history"]] == [r["monitored"] for r in base["history"]]
    assert [r["weights_sha256"] for r in new["history"]] == [r["weights_sha256"] for r in base["history"]]
    for key in ("fit", "inner", "validation"):
        with np.load(out / "predictions" / f"{key}.npz") as a, \
                np.load(fits["mg1"] / "predictions" / f"{key}.npz") as b:
            assert np.array_equal(a["predicted_price"], b["predicted_price"])
    # The report under the same perturbed accessor scores identically (annual, weekly, selection).
    result = _report(world, "mg1", out, tmp_path / "report")
    assert result["status"] == "COMPLETE", result
    assert result["eligibility"] == reports["mg1"]["result"]["eligibility"]
    assert result["selection"] == reports["mg1"]["result"]["selection"]
    metrics = json.loads((tmp_path / "report" / "METRICS.json").read_text())
    assert metrics["weekly"] == reports["mg1"]["metrics"]["weekly"]


def test_sc06_positive_control_a_common_inner_label_does_move_the_monitor(world, fits, tmp_path, monkeypatch):
    def bump_inner(split, target, mask):
        if split == "INNER":
            r = int(np.flatnonzero(mask[:, 0])[0])
            target[r, 0] += 0.05
    _wrap_accessor(monkeypatch, bump_inner)
    out = tmp_path / "fit" / "out" / "train"
    receipt = _fit(world, "mg1", out)
    assert receipt["status"] == "COMPLETED", receipt
    base = json.loads((fits["mg1"] / "TRAINING.json").read_text())["history"]
    new = json.loads((out / "TRAINING.json").read_text())["history"]
    assert new[0]["fit_loss"] == base[0]["fit_loss"]           # FIT untouched (same weights)
    assert new[0]["inner_loss"] != base[0]["inner_loss"]       # the common INNER label counts


def test_sc06_a_common_fit_label_change_is_caught_by_the_shared_naive_cross_check(world, tmp_path, monkeypatch):
    def bump_fit(split, target, mask):
        if split == "FIT":
            r = int(np.flatnonzero(mask[:, 0])[0])
            target[r, 0] += 0.05
    _wrap_accessor(monkeypatch, bump_fit)
    receipt = _fit(world, "mg2m", tmp_path / "unit")
    assert receipt["status"] == "FAILED_TYPED" and receipt["code"] == "NAIVE_SCALE_MISMATCH"


# ============================================================================ SC05 report path

def test_sc05_report_scores_model_and_naive_on_the_same_common_rows_with_manual_replay(world, fits, reports):
    dev = world["pop"].splits["DEVELOPMENT"]
    for arm in CP.ARMS:
        result, metrics = reports[arm]["result"], reports[arm]["metrics"]
        with np.load(fits[arm] / "predictions" / "validation.npz") as z:
            pred, target, origin = z["predicted_price"], z["target_price"], z["origin_price"]
        rels = []
        for j, h in enumerate(HORIZONS):
            sel = dev.head_mask[:, j]
            mae = np.abs(pred[sel, j] - target[sel, j]).mean()
            naive = np.abs(target[sel, j] - origin[sel]).mean()
            row = result["eligibility"][str(h)]
            assert row["n"] == int(sel.sum()) == dev.counts[h]
            assert row["mae"] == pytest.approx(mae, rel=1e-12)
            assert row["naive"] == pytest.approx(naive, rel=1e-12)
            assert row["relative_error"] == pytest.approx((mae - naive) / naive, rel=1e-10)
            assert row["row_ids_sha256"] == metrics["per_horizon"][str(h)]["row_ids_sha256"]
            rels.append(row["relative_error"])
            assert metrics["weekly"]["per_horizon"][str(h)]["annual"]["n"] == int(sel.sum())
            manual = metrics["matched"]["manual_pooled_replay"][str(h)]
            assert manual["mae"] == pytest.approx(row["mae"], rel=1e-12)
        block = metrics["matched"]
        assert block["population"] == "COMMON_DEVELOPMENT"
        assert block["common_population_sha256"] == world["digest"]
        assert block["development_counts"] == {str(h): dev.counts[h] for h in HORIZONS}
        assert block["input_replay"]["status"] == "REPLAYED"
        assert metrics["agreement"]["status"] == "AGREE"
        assert block["selection"]["status"] == "COMPLETE"
        assert block["selection"]["score"] == pytest.approx(sum(rels) / 6, rel=1e-12)
        assert result["selection"] == block["selection"]
        annual = result["reports"][0]
        assert annual["tags"]["population"] == "COMMON_DEVELOPMENT" and annual["tags"]["arm"] == arm
        assert any(m["metric"] == "Selection_Score_Equal_Head_Relative_MAE" for m in annual["metrics"])
        assert {m["horizon"] for m in annual["metrics"] if m["metric"] == "Relative_Error"} == set(HORIZONS)
    assert reports["mg1"]["metrics"]["matched"]["development_row_ids_sha256"] == \
        reports["mg2m"]["metrics"]["matched"]["development_row_ids_sha256"]


def _rewrite_predictions(fit_out, dst, edit):
    dst = Path(dst)
    shutil.copytree(fit_out, dst)
    path = dst / "predictions" / "validation.npz"
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    edit(arrays)
    np.savez(path, **arrays)
    receipt = json.loads((dst / "RECEIPT.json").read_text())
    receipt["artifacts"]["predictions/validation.npz"] = _sha(path)
    (dst / "RECEIPT.json").write_text(json.dumps(receipt))
    return dst


def test_sc05_masked_out_predictions_do_not_move_the_score_but_valid_ones_do(world, fits, reports, tmp_path):
    mask = world["pop"].splits["DEVELOPMENT"].head_mask
    hidden = np.argwhere(~mask)[0]
    shown = np.argwhere(mask)[0]

    def masked(arrays):
        arrays["predicted_price"][tuple(hidden)] = 123.0

    def valid(arrays):
        arrays["predicted_price"][tuple(shown)] += 0.01

    out = _rewrite_predictions(fits["mg2m"], tmp_path / "a" / "out" / "train", masked)
    result = _report(world, "mg2m", out, tmp_path / "report_a")
    assert result["status"] == "COMPLETE", result
    assert result["eligibility"] == reports["mg2m"]["result"]["eligibility"]
    out = _rewrite_predictions(fits["mg2m"], tmp_path / "b" / "out" / "train", valid)
    result = _report(world, "mg2m", out, tmp_path / "report_b")
    assert result["status"] == "COMPLETE", result
    h = str(HORIZONS[shown[1]])
    assert result["eligibility"][h]["mae"] != reports["mg2m"]["result"]["eligibility"][h]["mae"]


def test_report_refuses_rows_mask_or_prices_that_are_not_the_common_population(world, fits, tmp_path):
    def drop_row(arrays):
        for k in ("row_id", "predicted_price", "target_price", "origin_price", "mask"):
            arrays[k] = arrays[k][1:]

    def widen_mask(arrays):
        arrays["mask"] = np.ones_like(arrays["mask"])

    def move_price(arrays):
        arrays["origin_price"][0] += 1e-4

    for name, edit, code in (("rows", drop_row, "ROW_ID_MISMATCH"), ("mask", widen_mask, "MASK_MISMATCH"),
                             ("price", move_price, "PRICE_REPLAY_MISMATCH")):
        out = _rewrite_predictions(fits["mg1"], tmp_path / name / "out" / "train", edit)
        result = _report(world, "mg1", out, tmp_path / f"report_{name}")
        assert result["status"] == "FAILED" and result["disposition"] == code, result


def test_decision_rule_uses_scores_ties_to_mg2m_and_never_selects_from_failed_heads(reports):
    mg1, mg2m = reports["mg1"]["metrics"], reports["mg2m"]["metrics"]
    decision = reporting.decide_matched(mg1, mg2m, require_scientific=False)
    s1, s2 = mg1["matched"]["selection"]["score"], mg2m["matched"]["selection"]["score"]
    assert decision["capacity_delta"]["delta"]["heads"] == 2304
    assert decision["selected"] == ("mg1" if s1 < s2 - 1e-8 else "mg2m")
    tie = copy.deepcopy(mg1)
    tie["matched"]["selection"]["score"] = s2 + 5e-9
    assert reporting.decide_matched(tie, mg2m, require_scientific=False)["decision"] == "TIE_RETAIN_CHEAPER"
    assert reporting.decide_matched(tie, mg2m, require_scientific=False)["selected"] == "mg2m"
    failed = copy.deepcopy(mg1)
    failed["per_horizon"]["12"]["relative_error"] = None
    failed["matched"]["selection"] = reporting.selection_score(failed["per_horizon"], HORIZONS)
    out = reporting.decide_matched(failed, mg2m, require_scientific=False)
    assert out["decision"] == "INCOMPLETE_COMPARISON" and out["selected"] is None
    assert out["missing"][0]["horizon"] == 12 and out["missing"][0]["arm"] == "mg1"
    other = copy.deepcopy(mg2m)
    other["matched"]["development_row_ids_sha256"] = "0" * 64
    with pytest.raises(reporting.ReportFailure, match="COMMON_POPULATION_MISMATCH"):
        reporting.decide_matched(mg1, other, require_scientific=False)


# ============================================================================ SC06 refusals

def test_sc06_a_cached_full_population_mg2_fit_is_never_a_matched_arm(world, fits, tmp_path, monkeypatch):
    # (a) A REAL ordinary full-population MG2 fit under the closed design, offered as the arm.
    src = _synth.make_input(tmp_path / "full_input")
    market_grid.build_market_bundle(src, market_grid.market_design(_synth.load_design()),
                                    tmp_path / "mg2_bundle", code_commit="TEST")
    monkeypatch.setenv("I6DMO_TEST_MAX_ROWS", "48")
    plain = tmp_path / "plain" / "out" / "train"
    receipt = T.train_unit(CLOSED_MG2, tmp_path / "mg2_bundle", plain, "cpu",
                           training_overrides={"max_epochs": 1})
    monkeypatch.delenv("I6DMO_TEST_MAX_ROWS")
    assert receipt["status"] == "COMPLETED" and "common_population_sha256" not in receipt["identity"]
    result = _report(world, "mg2m", plain, tmp_path / "report_plain")
    assert result["status"] == "FAILED" and result["disposition"] == "NOT_A_MATCHED_FIT", result
    # (b) A matched receipt bound to another population, or to the other arm.
    identity = fits["receipts"]["mg2m"]["identity"]
    with pytest.raises(T.TrainDisposition, match="NOT_A_MATCHED_FIT"):
        T.verify_matched_identity(identity, "0" * 64, "mg2m")
    with pytest.raises(T.TrainDisposition, match="NOT_A_MATCHED_FIT"):
        T.verify_matched_identity(identity, world["digest"], "mg1")
    # (c) The closed MG2 design cannot carry a matched fit, scientific or not; nothing trains.
    for overrides in (False, True):
        out = tmp_path / f"closed_{overrides}"
        run = _cli(world, "mg2m", out, design=CLOSED_MG2, overrides=overrides)
        assert run.returncode == 3, run.stdout[-2000:] + run.stderr[-2000:]
        closed = json.loads((out / "RECEIPT.json").read_text())
        assert closed["code"] == "MATCHED_IDENTITY_REQUIRED"
        assert not (out / "TRAINING.json").exists()
    # (d) A report spec naming the closed design is refused too.
    spec = _spec(world, "mg2m", _stored_terminal(fits["mg2m"]), design=json.loads(CLOSED_MG2.read_text()))
    result = reporting.run_unit(spec, tmp_path / "report_closed", tmp_path / "p.json")
    assert result["status"] == "FAILED" and result["disposition"] == "MATCHED_IDENTITY_REQUIRED", result


def test_population_digest_subset_arm_and_argument_refusals(world, tmp_path, monkeypatch):
    bad = T.train_unit(world["designs"]["mg1"], None, tmp_path / "a", "cpu", training_overrides=OVERRIDES,
                       common_population={"path": str(world["population"]), "expected_digest": "0" * 64})
    assert bad["status"] == "FAILED_TYPED" and "DIGEST" in bad["code"], bad["code"]
    monkeypatch.setenv("I6DMO_TEST_MAX_ROWS", "10")
    assert _fit(world, "mg1", tmp_path / "b")["code"] == "MATCHED_SUBSET_FORBIDDEN"
    monkeypatch.delenv("I6DMO_TEST_MAX_ROWS")
    # The committed sealed MG1 design binds the REAL population, not this synthetic one.
    other = T.train_unit(SEALED["mg1"], None, tmp_path / "c", "cpu", training_overrides=OVERRIDES,
                         common_population=_cp(world))
    assert other["status"] == "FAILED_TYPED" and other["code"] == "DESIGN_IDENTITY_MISMATCH", other["code"]
    with pytest.raises(T.ContractError, match="MATCHED_ARGS"):
        T.train_unit(world["designs"]["mg1"], tmp_path, tmp_path / "d", "cpu", common_population=_cp(world))
    assert T.main(["--design", str(world["designs"]["mg1"]), "--out", str(tmp_path / "e"),
                   "--device", "cpu", "--common-population", "p"]) == 2
    assert T.main(["--design", str(world["designs"]["mg1"]), "--out", str(tmp_path / "f"),
                   "--device", "cpu", "--input-dir", "i"]) == 2


# ============================================================================ SC11 (CPU half)

def test_sc11_cpu_serialization_round_trip_reproduces_saved_predictions_exactly(world, fits):
    population = world["pop"]
    for arm in CP.ARMS:
        out = fits[arm]
        reloaded, doc = M.load_dmo(out / "model")
        assert doc["load_path"] == "rebuild+load_weights"
        assert tuple(reloaded.forecast_model.inputs[0].shape[1:]) == ((25 if arm == "mg1" else 13), 19)
        scales = json.loads((out / "SCALES.json").read_text())
        data = T.load_matched_data(population, _eff_world(world, arm))
        for key in ("fit", "inner", "validation"):
            x, _, _ = T.materialize_monitor_inputs(data[key], scales)
            again = T.to_price(M.predict_matrix(reloaded.forecast_model, x), data[key], scales)
            with np.load(out / "predictions" / f"{key}.npz") as z:
                assert np.array_equal(again, z["predicted_price"]), (arm, key)
        assert M.capacity_record(reloaded) == fits["receipts"][arm]["matched"]["capacity"]
