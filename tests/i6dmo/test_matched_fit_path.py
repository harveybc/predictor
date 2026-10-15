"""Matched MG1/MG2 short control, MODEL lane: SC05, SC06, SC07 and the CPU half of SC11.

REAL seams only: the sealed MG2 design, a REAL MG2 market bundle built by the data lane's
``market_grid`` from synthetic inputs, the REAL trainer (``tools.i6dmo.train.main`` in a fresh
interpreter, and ``train_unit`` in-process), and the REAL report hook
(``tools.i6dmo.reporting.run_unit``). No injected arrays loader, no fake trainer, no GPU, no
TEST rows, tiny CPU fixtures (two epochs).

Coverage boundary, stated plainly: the data lane's ``tools/i6dmo/common_population.py`` does
not exist yet, so the frozen common population, the per-arm accessor for the MG1 arm and the
population's on-disk format come from ``tests/i6dmo/_common_population_shim.py`` (installed as
that module only while it is absent). No sealed MG1 design exists yet either (data lane), so the
MG1 arm runs through ``train_unit`` with a test-derived effective config, which stamps
``scientific: false``; the MG2 arm runs through the real CLI under the sealed MG2 design.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6dmo import design as D, market_grid, model as M, reporting, train as T  # noqa: E402
from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4  # noqa: E402
from tests.i6dmo import _common_population_shim as SHIM, _synth  # noqa: E402
from tests.i6dmo.dmo_model_fixtures import DESIGN  # noqa: E402

REPO = DESIGN.parents[3]
MG2_DESIGN = DESIGN.parent / "EXPERIMENT.mg2.json"
CAPACITY_DOC = DESIGN.parent / "matched_short" / "CAPACITY_MATCHED.json"
FEATURES = list(MANIFEST_NO_STLFSI4)
HORIZONS = [2, 3, 4, 5, 6, 12]
OVERRIDES = {"max_epochs": 2}
REAL_CP = importlib.util.find_spec("tools.i6dmo.common_population") is not None
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
#: The real trainer entrypoint with ONLY the missing data-lane module shimmed in (fresh
#: interpreter; tools.i6dmo.train imports the model module before TensorFlow as in production).
SHIMMED_MAIN = ("import sys\n"
                "from tests.i6dmo import _common_population_shim as s\n"
                "s.install()\n"
                "from tools.i6dmo.train import main\n"
                "sys.exit(main(sys.argv[1:]))\n")


def _sha(path):
    return T.file_sha256(path)


def mg1_effective():
    """TEST-ONLY MG1 effective config (no sealed MG1 design exists yet: data lane).

    The sealed MG2 effective config with the market-time block and model block switched to the
    plan's MG1 values; the model block must equal ``model_config(..., family='short_mg1')``.
    """
    eff = copy.deepcopy(T.effective_from_design(MG2_DESIGN))
    mt = dict(eff["design"]["market_time"], rule="MG1_MARKET_EXACT", observations=25,
              spacing_open_market_hours=1, span_open_market_hours=24)
    eff["design"]["market_time"] = mt
    eff["design"]["experiment_id"] = "TEST-I6-DMO-SHORT-MG1-MATCHED-SHIM-R0"
    eff["experiment_id"] = eff["design"]["experiment_id"]
    eff["market_time"] = dict(mt)
    eff["model"] = dict(eff["model"], slots=25, market_time=dict(mt))
    eff["design_sha256"] = D.design_sha256(eff["design"])
    eff["effective_config_sha256"] = D.effective_config_sha256(eff)
    return eff


@pytest.fixture(scope="module", autouse=True)
def common_population_module():
    if REAL_CP:
        pytest.skip("data lane module present: switch these fixtures to it and delete the shim")
    module = SHIM.install()
    yield module
    sys.modules.pop("tools.i6dmo.common_population", None)
    import tools.i6dmo as pkg
    if getattr(pkg, "common_population", None) is module:
        delattr(pkg, "common_population")


@pytest.fixture(scope="module")
def world(tmp_path_factory, common_population_module):
    root = tmp_path_factory.mktemp("matched")
    src = _synth.make_input(root / "input")
    market_grid.build_market_bundle(src, market_grid.market_design(_synth.load_design()),
                                    root / "mg2_bundle", code_commit="TEST")
    mg1 = mg1_effective()
    mg2_sha = D.design_sha256(json.loads(MG2_DESIGN.read_text()))
    fx = SHIM.build_fixture(root / "fx", mg2_bundle=root / "mg2_bundle", input_dir=src,
                            designs={"MG1": mg1["design_sha256"], "MG2": mg2_sha})
    population = SHIM.load_common_population(fx["population"], expected_digest=fx["digest"])
    return {"root": root, "input": src, "mg2_bundle": root / "mg2_bundle", "mg1": mg1,
            **fx, "pop": population}


def _cp(world):
    return {"path": str(world["population"]), "expected_digest": world["digest"]}


def _mg1_fit(world, out, arm=None):
    return T.train_unit(None, arm or world["arm_mg1"], out, "cpu", training_overrides=OVERRIDES,
                        common_population=_cp(world), effective=world["mg1"])


@pytest.fixture(scope="module")
def fits(world):
    """One matched fit per arm: MG2 through the REAL CLI, MG1 through train_unit."""
    out2 = world["root"] / "fit_mg2" / "out" / "train"
    run = subprocess.run(
        [sys.executable, "-c", SHIMMED_MAIN, "--design", str(MG2_DESIGN),
         "--bundle", str(world["mg2_bundle"]), "--out", str(out2), "--device", "cpu", "--json",
         "--common-population", str(world["population"]),
         "--common-population-digest", world["digest"]],
        capture_output=True, text=True, cwd=REPO, check=False,
        env={**ENV, "I6DMO_TEST_TRAINING_OVERRIDES": json.dumps(OVERRIDES)})
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    out1 = world["root"] / "fit_mg1" / "out" / "train"
    receipt1 = _mg1_fit(world, out1)
    assert receipt1["status"] == "COMPLETED", receipt1
    return {"MG1": out1, "MG2": out2,
            "receipts": {"MG1": receipt1,
                         "MG2": json.loads((out2 / "RECEIPT.json").read_text())}}


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


def _spec(world, arm, terminal):
    design = world["mg1"]["design"] if arm == "MG1" else json.loads(MG2_DESIGN.read_text())
    source = world["arm_mg1"] if arm == "MG1" else world["mg2_bundle"]
    arm_pin = {"path": str(source)}
    if arm == "MG2":
        arm_pin["sha256"] = _sha(Path(source) / "BUNDLE.json")
    return {"unit_id": f"{arm.lower()}.report", "kind": "report", "family": "short",
            "identity_sha256": "r" * 64, "campaign_id": "matched-short-model-lane-test",
            "cadence": "NONE", "test_open": False,
            "effective_config": {"design": design, "code_commit": "c" * 40},
            "upstream": {f"{arm.lower()}.fit": {"path": str(terminal), "sha256": _sha(terminal)}},
            "common_population": {"path": str(world["population"]), "sha256": world["digest"]},
            "matched": {"arm": arm, "fit_unit": f"{arm.lower()}.fit", "arm_source": arm_pin,
                        "input": {"path": str(world["input"]),
                                  "sha256": _sha(Path(world["input"]) / "INPUT.npz")}}}


def _report(world, arm, fit_out, out):
    terminal = _stored_terminal(fit_out)
    result = reporting.run_unit(_spec(world, arm, terminal), out, out.parent / f"{out.name}.progress")
    return result


@pytest.fixture(scope="module")
def reports(world, fits):
    out = {}
    for arm in ("MG1", "MG2"):
        target = world["root"] / f"report_{arm.lower()}"
        result = _report(world, arm, fits[arm], target)
        assert result["status"] == "COMPLETE", result
        out[arm] = {"result": result, "metrics": json.loads((target / "METRICS.json").read_text())}
    return out


# ============================================================================ SC07 capacity

def test_sc07_real_sealed_builds_three_inputs_78_channels_and_2304_delta():
    mg2_eff = T.effective_from_design(MG2_DESIGN)
    mg1_eff = mg1_effective()
    assert M.normalize_model_config(mg1_eff["model"]) == M.model_config(FEATURES, family="short_mg1")
    assert M.normalize_model_config(mg2_eff["model"]) == M.model_config(FEATURES, family="short_mg2")
    records = {}
    for arm, eff, slots in (("MG1", mg1_eff, 25), ("MG2", mg2_eff, 13)):
        bundle = M.build_dmo(eff["model"])
        model = bundle.forecast_model
        assert [tuple(t.shape[1:]) for t in model.inputs] == [(slots, 19), (slots,), (slots,)]
        # MEASURED, not assumed: the core sees 76 encoded value channels + 2 context channels.
        assert tuple(bundle.core_model.input_shape) == (None, slots, 78)
        encoder = model.get_layer("physical_time_encoder")
        assert encoder.min_adjacent_hours == float(eff["design"]["market_time"]["spacing_open_market_hours"])
        assert len(model.outputs) == 6
        records[arm] = M.capacity_record(bundle)
        assert records[arm]["arm"] == arm
        assert {k: records[arm]["counts"][k] for k in M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]} \
            == M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]
    delta = M.matched_capacity_delta(records["MG1"], records["MG2"])
    assert delta["delta"] == {"branches": 0, "fusion": 0, "time_encoder": 0, "core": 0,
                              "heads": 2304, "total": 2304}
    assert (25 - 13) * 4 * 8 * 6 == M.MATCHED_HEAD_DELTA == 2304
    assert records["MG1"]["counts"]["total"] == 7186 and records["MG2"]["counts"]["total"] == 4882
    # The committed machine-readable disclosure is exactly what the real builds measure.
    doc = json.loads(CAPACITY_DOC.read_text())
    assert doc["arms"]["MG1"]["counts"] == records["MG1"]["counts"]
    assert doc["arms"]["MG2"]["counts"] == records["MG2"]["counts"]
    assert doc["delta"] == delta["delta"]
    assert doc["declared_capacity"] == M.DECLARED_CAPACITY


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
    bundle = M.build_dmo(M.model_config(FEATURES, family="short_mg2"))
    with pytest.raises(M.CapacityError, match="CAPACITY_UNREPORTED_CHANGE"):
        M.capacity_record(bundle)


def test_sc07_capacity_delta_refuses_anything_but_the_declared_head_difference():
    a = M.capacity_record(M.build_dmo(M.model_config(FEATURES, family="short_mg1")))
    b = M.capacity_record(M.build_dmo(M.model_config(FEATURES, family="short_mg2")))
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(b, a)                       # arms swapped
    tampered = copy.deepcopy(a)
    tampered["counts"]["core"] += 8
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(tampered, b)


def test_sc07_the_trainer_refuses_a_capacity_change_typed_before_fitting(world, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "dmo_head", _wide_head)
    receipt = _mg1_fit(world, tmp_path / "unit")
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


def test_25_slot_ladder_passes_the_sealed_spacing_seam_and_bad_ladders_refuse():
    mg1, mg2 = mg1_effective(), T.effective_from_design(MG2_DESIGN)
    assert T.market_spacing_hours(mg1) == 1.0 and T.market_spacing_hours(mg2) == 2.0
    weekend = _ladder(2, 25, 1.0)
    weekend[1, :10] -= 48.0                                   # a weekend between slots 9 and 10
    flags = np.zeros((2, 25))
    flags[1, 10] = 1
    ok = T.market_time_inputs({**_arrays(weekend), "closure_crossed": flags}, design=mg1)
    assert [a.shape for a in ok] == [(2, 25, 19), (2, 25), (2, 25)]
    assert len(T.market_time_inputs(_arrays(_ladder(2, 13, 2.0)), design=mg2)) == 3   # 13 @ 2 h
    refusals = [
        (_ladder(2, 25, 1.0, start=0), mg1),                 # ordinal slot indices 0..24
        (_ladder(2, 25, 0.5), mg1),                          # compressed ladder under MG1
        (_ladder(2, 13, 1.0), mg2),                          # ordinal 13-slot ladder under MG2
        (_ladder(2, 25, 1.0), mg2),                          # an MG1 ladder under the MG2 design
    ]
    for offsets, eff in refusals:
        with pytest.raises(T.TrainDisposition, match="ORDINAL_OR_INVALID_OFFSETS"):
            T.market_time_inputs(_arrays(offsets), design=eff)
    # The encoder seam carries the same design-derived bound.
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
    for arm in ("MG1", "MG2"):
        out, receipt = fits[arm], fits["receipts"][arm]
        assert receipt["status"] == "COMPLETED" and receipt["scientific"] is False
        block = receipt["matched"]
        assert block["arm"] == arm and block["common_population_sha256"] == world["digest"]
        assert receipt["identity"]["common_population_sha256"] == world["digest"]
        assert receipt["identity"]["matched_arm"] == arm
        assert block["arm_source"]["kind"] == ("market_bundle" if arm == "MG2"
                                               else "common_population_accessor")
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
    assert scales["MG1"] == scales["MG2"]
    assert scales["MG1"]["input_mean"] == population.scaler.mean.tolist()
    assert scales["MG1"]["input_std"] == population.scaler.std.tolist()
    assert scales["MG1"]["naive_mae"] == [population.naive_loss_scale[h] for h in HORIZONS]
    arm_only = T.fit_scales(T.load_matched_data(world["mg2_bundle"], T.effective_from_design(MG2_DESIGN),
                                                MG2_DESIGN, population)["fit"])
    assert arm_only["input_mean"] != scales["MG2"]["input_mean"]
    assert fits["receipts"]["MG1"]["identity"]["scales_sha256"] == \
        fits["receipts"]["MG2"]["identity"]["scales_sha256"]
    # The training recipe is the frozen one.
    settings = json.loads((fits["MG1"] / "TRAINING.json").read_text())["settings"]
    assert (settings["learning_rate"], settings["weight_decay"], settings["batch_size"],
            settings["patience"], settings["min_delta"]) == (1e-3, 1e-4, 256, 10, 1e-8)
    assert T.effective_from_design(MG2_DESIGN)["training"]["max_epochs"] == 200
    assert T.effective_from_design(MG2_DESIGN)["seed"] == 6102026


def _masked_out_positions(world):
    """(split, row) pairs whose 12 h label is valid under the arm but outside the common mask."""
    j = HORIZONS.index(12)
    edits = []
    for name in ("FIT", "INNER", "DEVELOPMENT"):
        with np.load(Path(world["arm_mg1"]) / f"{name}.npz") as z:
            arm_mask = z["mask"]
            ids = z["row_id"]
        sp = world["pop"].splits[name]
        rows = [r for r in range(len(ids)) if arm_mask[r, j] and not sp.head_mask[r, j]]
        assert rows, f"{name}: the fixture must hold arm-valid labels outside the common mask"
        edits += [(name, "target_price", (r, j), 9.99) for r in rows]
    return edits


def test_sc06_masked_out_labels_never_reach_loss_monitor_predictions_or_score(world, fits, reports, tmp_path):
    perturbed = SHIM.perturb_arm(world["arm_mg1"], tmp_path / "arm", _masked_out_positions(world))
    out = tmp_path / "fit" / "out" / "train"
    receipt = _mg1_fit(world, out, arm=perturbed)
    assert receipt["status"] == "COMPLETED", receipt
    base = json.loads((fits["MG1"] / "TRAINING.json").read_text())
    new = json.loads((out / "TRAINING.json").read_text())
    assert [r["monitored"] for r in new["history"]] == [r["monitored"] for r in base["history"]]
    assert [r["weights_sha256"] for r in new["history"]] == [r["weights_sha256"] for r in base["history"]]
    for key in ("fit", "inner", "validation"):
        with np.load(out / "predictions" / f"{key}.npz") as a, \
                np.load(fits["MG1"] / "predictions" / f"{key}.npz") as b:
            assert np.array_equal(a["predicted_price"], b["predicted_price"])
    # The report on the perturbed arm source scores identically (annual and weekly).
    spec = _spec(world, "MG1", _stored_terminal(out))
    spec["matched"]["arm_source"]["path"] = str(perturbed)
    result = reporting.run_unit(spec, tmp_path / "report", tmp_path / "progress.json")
    assert result["status"] == "COMPLETE", result
    assert result["eligibility"] == reports["MG1"]["result"]["eligibility"]
    assert result["selection"] == reports["MG1"]["result"]["selection"]


def test_sc06_positive_control_a_common_inner_label_does_move_the_monitor(world, fits, tmp_path):
    sp = world["pop"].splits["INNER"]
    r = int(np.flatnonzero(sp.head_mask[:, 0])[0])
    perturbed = SHIM.perturb_arm(world["arm_mg1"], tmp_path / "arm",
                                 [("INNER", "target_price", (r, 0), 9.99)])
    out = tmp_path / "fit" / "out" / "train"
    receipt = _mg1_fit(world, out, arm=perturbed)
    assert receipt["status"] == "COMPLETED", receipt
    base = json.loads((fits["MG1"] / "TRAINING.json").read_text())["history"]
    new = json.loads((out / "TRAINING.json").read_text())["history"]
    assert new[0]["fit_loss"] == base[0]["fit_loss"]           # FIT untouched (same weights)
    assert new[0]["inner_loss"] != base[0]["inner_loss"]       # the common INNER label counts


# ============================================================================ SC05 report path

def test_sc05_report_scores_model_and_naive_on_the_same_common_rows_with_manual_replay(world, fits, reports):
    population = world["pop"]
    dev = population.splits["DEVELOPMENT"]
    for arm in ("MG1", "MG2"):
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
            # The model and naive used the identical row set per head.
            assert row["row_ids_sha256"] == metrics["per_horizon"][str(h)]["row_ids_sha256"]
            rels.append(row["relative_error"])
            weekly = metrics["weekly"]["per_horizon"][str(h)]
            assert weekly["annual"]["n"] == int(sel.sum())
        block = metrics["matched"]
        assert block["population"] == "COMMON_DEVELOPMENT"
        assert block["common_population_sha256"] == world["digest"]
        assert block["development_counts"] == {str(h): dev.counts[h] for h in HORIZONS}
        assert metrics["agreement"]["status"] == "AGREE"
        assert block["selection"]["status"] == "COMPLETE"
        assert block["selection"]["score"] == pytest.approx(sum(rels) / 6, rel=1e-12)
        assert result["selection"] == block["selection"]
        annual = result["reports"][0]
        assert annual["tags"]["population"] == "COMMON_DEVELOPMENT" and annual["tags"]["arm"] == arm
        assert any(m["metric"] == "Selection_Score_Equal_Head_Relative_MAE" for m in annual["metrics"])
        # Per-head numbers stay primary: every head is published individually.
        assert {m["horizon"] for m in annual["metrics"] if m["metric"] == "Relative_Error"} == set(HORIZONS)
    assert reports["MG1"]["metrics"]["matched"]["development_row_ids_sha256"] == \
        reports["MG2"]["metrics"]["matched"]["development_row_ids_sha256"]


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

    out = _rewrite_predictions(fits["MG2"], tmp_path / "a" / "out" / "train", masked)
    result = _report(world, "MG2", out, tmp_path / "report_a")
    assert result["status"] == "COMPLETE", result
    assert result["eligibility"] == reports["MG2"]["result"]["eligibility"]
    out = _rewrite_predictions(fits["MG2"], tmp_path / "b" / "out" / "train", valid)
    result = _report(world, "MG2", out, tmp_path / "report_b")
    assert result["status"] == "COMPLETE", result
    h = str(HORIZONS[shown[1]])
    assert result["eligibility"][h]["mae"] != reports["MG2"]["result"]["eligibility"][h]["mae"]


def test_report_refuses_rows_or_mask_that_are_not_the_common_population(world, fits, tmp_path):
    def drop_row(arrays):
        for k in ("row_id", "predicted_price", "target_price", "origin_price", "mask"):
            arrays[k] = arrays[k][1:]

    def widen_mask(arrays):
        arrays["mask"] = np.ones_like(arrays["mask"])

    for name, edit, code in (("rows", drop_row, "ROW_ID_MISMATCH"), ("mask", widen_mask, "MASK_MISMATCH")):
        out = _rewrite_predictions(fits["MG1"], tmp_path / name / "out" / "train", edit)
        result = _report(world, "MG1", out, tmp_path / f"report_{name}")
        assert result["status"] == "FAILED" and result["disposition"] == code, result


def test_decision_rule_uses_scores_ties_to_mg2_and_never_selects_from_failed_heads(reports):
    mg1, mg2 = reports["MG1"]["metrics"], reports["MG2"]["metrics"]
    decision = reporting.decide_matched(mg1, mg2)
    s1, s2 = mg1["matched"]["selection"]["score"], mg2["matched"]["selection"]["score"]
    assert decision["capacity_delta"]["delta"]["heads"] == 2304
    assert decision["selected"] == ("MG1" if s1 < s2 - 1e-8 else "MG2")
    tie = copy.deepcopy(mg1)
    tie["matched"]["selection"]["score"] = s2 + 5e-9
    assert reporting.decide_matched(tie, mg2)["decision"] == "TIE_RETAIN_CHEAPER"
    assert reporting.decide_matched(tie, mg2)["selected"] == "MG2"
    failed = copy.deepcopy(mg1)
    failed["per_horizon"]["12"]["relative_error"] = None
    failed["matched"]["selection"] = reporting.selection_score(failed["per_horizon"], HORIZONS)
    out = reporting.decide_matched(failed, mg2)
    assert out["decision"] == "INCOMPLETE_COMPARISON" and out["selected"] is None
    assert out["missing"][0]["horizon"] == 12 and out["missing"][0]["arm"] == "MG1"
    other = copy.deepcopy(mg2)
    other["matched"]["development_row_ids_sha256"] = "0" * 64
    with pytest.raises(reporting.ReportFailure, match="COMMON_POPULATION_MISMATCH"):
        reporting.decide_matched(mg1, other)


# ============================================================================ SC06 refusals

def test_sc06_a_cached_full_population_mg2_fit_is_never_a_matched_arm(world, fits, tmp_path, monkeypatch):
    # (a) A REAL ordinary (full-population) MG2 fit on the same bundle, then offered as the arm.
    monkeypatch.setenv("I6DMO_TEST_MAX_ROWS", "48")
    plain = tmp_path / "plain" / "out" / "train"
    receipt = T.train_unit(MG2_DESIGN, world["mg2_bundle"], plain, "cpu",
                           training_overrides={"max_epochs": 1})
    monkeypatch.delenv("I6DMO_TEST_MAX_ROWS")
    assert receipt["status"] == "COMPLETED" and "common_population_sha256" not in receipt["identity"]
    result = _report(world, "MG2", plain, tmp_path / "report_plain")
    assert result["status"] == "FAILED" and result["disposition"] == "NOT_A_MATCHED_FIT", result
    # (b) A matched receipt bound to another population, or to the other arm.
    identity = fits["receipts"]["MG2"]["identity"]
    with pytest.raises(T.TrainDisposition, match="NOT_A_MATCHED_FIT"):
        T.verify_matched_identity(identity, "0" * 64, "MG2")
    with pytest.raises(T.TrainDisposition, match="NOT_A_MATCHED_FIT"):
        T.verify_matched_identity(identity, world["digest"], "MG1")
    # (c) A scientific matched fit under the closed MG2 identity is refused before anything runs.
    out = tmp_path / "closed"
    run = subprocess.run(
        [sys.executable, "-c", SHIMMED_MAIN, "--design", str(MG2_DESIGN),
         "--bundle", str(world["mg2_bundle"]), "--out", str(out), "--device", "cpu", "--json",
         "--common-population", str(world["population"]),
         "--common-population-digest", world["digest"]],
        capture_output=True, text=True, cwd=REPO, check=False, env=ENV)
    assert run.returncode == 3, run.stdout[-2000:] + run.stderr[-2000:]
    closed = json.loads((out / "RECEIPT.json").read_text())
    assert closed["code"] == "CLOSED_IDENTITY_NOT_MATCHED" and closed["scientific"] is True
    assert not (out / "TRAINING.json").exists()


def test_population_digest_subset_and_argument_refusals(world, tmp_path, monkeypatch):
    bad = T.train_unit(None, world["arm_mg1"], tmp_path / "a", "cpu", training_overrides=OVERRIDES,
                       common_population={"path": str(world["population"]),
                                          "expected_digest": "0" * 64},
                       effective=world["mg1"])
    assert bad["status"] == "FAILED_TYPED" and bad["code"] == "COMMON_POPULATION_DIGEST"
    monkeypatch.setenv("I6DMO_TEST_MAX_ROWS", "10")
    sub = _mg1_fit(world, tmp_path / "b")
    assert sub["code"] == "MATCHED_SUBSET_FORBIDDEN"
    monkeypatch.delenv("I6DMO_TEST_MAX_ROWS")
    with pytest.raises(T.ContractError, match="EFFECTIVE_INJECTION"):
        T.train_unit(None, world["arm_mg1"], tmp_path / "c", "cpu", effective=world["mg1"])
    wrong_arm = T.train_unit(None, world["arm_mg2"], tmp_path / "d", "cpu",
                             training_overrides=OVERRIDES, common_population=_cp(world),
                             effective=world["mg1"])
    assert wrong_arm["code"] == "ARM_RULE_MISMATCH"
    assert T.main(["--design", str(MG2_DESIGN), "--bundle", "x", "--out", str(tmp_path / "e"),
                   "--device", "cpu", "--common-population", "p"]) == 2


@pytest.mark.skipif(REAL_CP, reason="the data lane module exists")
def test_real_cli_without_the_data_module_is_a_typed_terminal(world, tmp_path):
    out = tmp_path / "unit"
    run = subprocess.run(
        [sys.executable, "-m", "tools.i6dmo.train", "--design", str(MG2_DESIGN),
         "--bundle", str(world["mg2_bundle"]), "--out", str(out), "--device", "cpu", "--json",
         "--common-population", str(world["population"]),
         "--common-population-digest", world["digest"]],
        capture_output=True, text=True, cwd=REPO, check=False,
        env={**ENV, "I6DMO_TEST_TRAINING_OVERRIDES": json.dumps(OVERRIDES)})
    assert run.returncode == 3
    assert json.loads((out / "RECEIPT.json").read_text())["code"] == "COMMON_POPULATION_MODULE_MISSING"


# ============================================================================ SC11 (CPU half)

def test_sc11_cpu_serialization_round_trip_reproduces_saved_predictions_exactly(world, fits):
    population = world["pop"]
    sources = {"MG1": (world["arm_mg1"], world["mg1"], None),
               "MG2": (world["mg2_bundle"], T.effective_from_design(MG2_DESIGN), MG2_DESIGN)}
    for arm, (source, eff, design_ref) in sources.items():
        out = fits[arm]
        reloaded, doc = M.load_dmo(out / "model")
        assert doc["load_path"] == "rebuild+load_weights"
        assert [tuple(t.shape[1:]) for t in reloaded.forecast_model.inputs][0][0] == \
            (25 if arm == "MG1" else 13)
        scales = json.loads((out / "SCALES.json").read_text())
        data = T.load_matched_data(source, eff, design_ref if design_ref else eff["design"], population)
        for key in ("fit", "inner", "validation"):
            x, _, _ = T.materialize_monitor_inputs(data[key], scales)
            again = T.to_price(M.predict_matrix(reloaded.forecast_model, x), data[key], scales)
            with np.load(out / "predictions" / f"{key}.npz") as z:
                assert np.array_equal(again, z["predicted_price"]), (arm, key)
        assert M.capacity_record(reloaded) == fits["receipts"][arm]["matched"]["capacity"]
