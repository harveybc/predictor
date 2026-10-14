"""DMO04 masked loss, DMO10 early stopping/restore/save-load, DMO18 TRAIN-only scales.

Tiny invented CPU fixtures; non-scientific overrides are stamped in the receipt.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6dmo import model as M  # noqa: E402
from tools.i6dmo import train as T  # noqa: E402
from tests.i6dmo.dmo_model_fixtures import (DESIGN, HORIZONS, split_arrays,  # noqa: E402
                                           write_bundle)

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


# --------------------------------------------------------------------- DMO04

def _manual(y, pred, mask):
    heads = []
    for h in range(y.shape[1]):
        sel = mask[:, h]
        if sel.any():
            heads.append(np.mean(np.abs(pred[sel, h] - y[sel, h])))
    return float(np.mean(heads))


def test_dmo04_mask_retains_other_heads_and_nan_never_enters_loss():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(8, 6))
    pred = rng.normal(size=(8, 6))
    mask = np.ones((8, 6), bool)
    mask[0, 1:] = False          # row 0 keeps only h2: still contributes to h2
    mask[3, :5] = False          # row 3 keeps only h12
    mask[:, 4] = False           # h6 unsupported in this batch: excluded, not zero
    y_nan = np.where(mask, y, np.nan)
    y_inf = np.where(mask, y, np.inf)
    expected = _manual(y, pred, mask)
    for labels in (y_nan, y_inf):
        loss, sums, counts = T.masked_normalized_loss(tf.constant(labels), tf.constant(pred),
                                                      tf.constant(mask))
        assert math.isclose(float(loss), expected, rel_tol=1e-12)
        assert np.asarray(counts).tolist() == mask.sum(axis=0).tolist()
    var = tf.Variable(pred)
    with tf.GradientTape() as tape:
        loss, _, _ = T.masked_normalized_loss(tf.constant(y_nan), var, tf.constant(mask))
    grad = np.asarray(tape.gradient(loss, var))
    assert np.all(np.isfinite(grad))
    assert np.all(grad[~mask] == 0)
    assert np.all(grad[0, :1] != 0) and np.all(grad[3, 5:] != 0)


def test_dmo04_evaluation_uses_global_sums_counts_and_rejects_nan_on_valid_labels():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(30, 6))
    pred = rng.normal(size=(30, 6))
    mask = rng.random((30, 6)) > 0.3
    mask[:, 0] = True
    res = T.masked_split_metrics(np.where(mask, y, np.nan), pred, mask)
    assert math.isclose(res["loss"], _manual(y, pred, mask), rel_tol=1e-12)
    assert res["counts"] == mask.sum(axis=0).tolist()
    bad = np.where(mask, y, np.nan)
    bad[2, 0] = np.nan
    with pytest.raises(T.TrainDisposition, match="NONFINITE_LABEL"):
        T.masked_split_metrics(bad, pred, mask)


def test_dmo04_rows_without_any_valid_head_are_counted_not_trained():
    arrays = split_arrays("2020-01-06T00:00:00", 20, 5)
    arrays["mask"][4] = False
    arrays["target_price"][4] = np.nan
    kept, dropped = T.trainable_rows(arrays)
    assert dropped == 1 and len(kept["row_id"]) == 19
    assert arrays["row_id"][4] not in kept["row_id"]


# --------------------------------------------------------------------- DMO18

def test_dmo18_scales_come_only_from_fit_rows():
    fit = split_arrays("2020-01-06T00:00:00", 50, 1)
    scales = T.fit_scales(fit)
    assert scales["source_split"] == "fit" and scales["rows"] == 50
    x = fit["x"].reshape(-1, 19)
    assert np.allclose(scales["input_mean"], x.mean(axis=0))
    assert np.allclose(scales["input_std"], x.std(axis=0))
    delta = np.abs(fit["target_price"] - fit["origin_price"][:, None])
    naive = [np.mean(delta[fit["mask"][:, h], h]) for h in range(6)]
    assert np.allclose(scales["naive_mae"], naive)
    assert scales["counts"] == fit["mask"].sum(axis=0).tolist()


def test_dmo18_scale_dispositions_are_typed_not_epsilon():
    fit = split_arrays("2020-01-06T00:00:00", 30, 2)
    const = {k: v.copy() for k, v in fit.items()}
    const["x"][:, :, 3] = 7.0
    with pytest.raises(T.TrainDisposition, match="CONSTANT_FEATURE_IN_FIT"):
        T.fit_scales(const)
    zero = {k: v.copy() for k, v in fit.items()}
    zero["target_price"][:, 2] = np.where(zero["mask"][:, 2], zero["origin_price"], np.nan)
    with pytest.raises(T.TrainDisposition, match="ZERO_NAIVE_SCALE"):
        T.fit_scales(zero)
    unsupported = {k: v.copy() for k, v in fit.items()}
    unsupported["mask"][:, 5] = False
    unsupported["target_price"][:, 5] = np.nan
    with pytest.raises(T.TrainDisposition, match="HEAD_UNSUPPORTED_IN_FIT"):
        T.fit_scales(unsupported)


def test_dmo18_bundle_identity_gates(tmp_path):
    from tools.i6n_contracts.feature_availability import LEGACY_MANIFEST_20, MANIFEST_NO_STLFSI4
    eff = T.effective_from_design(DESIGN)
    with pytest.raises(T.TrainDisposition, match="STLFSI4_ALIAS"):
        T.load_data_bundle(write_bundle(tmp_path / "a", features=LEGACY_MANIFEST_20), eff)
    renamed = list(MANIFEST_NO_STLFSI4)
    renamed[4] = "fred.stress.STLFSI4.level"
    with pytest.raises(T.TrainDisposition, match="STLFSI4_ALIAS"):
        T.load_data_bundle(write_bundle(tmp_path / "b", features=renamed), eff)
    with pytest.raises(T.TrainDisposition, match="FEATURE_IDENTITY_MISMATCH"):
        T.load_data_bundle(write_bundle(tmp_path / "c", manifest_sha="0" * 64), eff)
    test_rows = {"test": split_arrays("2025-01-06T00:00:00", 4, 9)}
    with pytest.raises(T.TrainDisposition, match="TEST_FORBIDDEN"):
        T.load_data_bundle(write_bundle(tmp_path / "d", extra_split=test_rows), eff)


@pytest.mark.parametrize(("split", "start", "code"), [
    ("validation", "2024-12-31T20:00:00", "SPLIT_BOUNDARY_VIOLATION"),   # labels reach 2025
    ("fit", "2023-11-05T20:00:00", "PURGE_VIOLATION"),                   # fit labels reach inner
    ("inner", "2023-12-31T22:00:00", "SPLIT_BOUNDARY_VIOLATION"),
    ("validation", "2025-01-01T00:00:00", "TEST_FORBIDDEN"),
])
def test_dmo18_boundary_and_purge_checks(tmp_path, split, start, code):
    eff = T.effective_from_design(DESIGN)
    splits = {"fit": split_arrays("2020-01-06T00:00:00", 40, 1),
              "inner": split_arrays("2023-11-13T00:00:00", 16, 2),
              "validation": split_arrays("2024-01-08T00:00:00", 16, 3)}
    splits[split] = split_arrays(start, 4, 4, invalid_rate=0.0)
    with pytest.raises(T.TrainDisposition, match=code):
        T.load_data_bundle(write_bundle(tmp_path / "x", splits=splits), eff)


# --------------------------------------------------------------------- DMO10

def test_dmo10_stopper_mean_monitor_patience_min_delta_and_restore_choice():
    stop = T.MeanMonitorStopper(patience=3, min_delta=0.1, max_epochs=50)
    seq = [(1.0, 3.0), (1.0, 2.0), (1.0, 1.95), (1.0, 1.96), (1.0, 1.99), (0.9, 2.0)]
    decisions = []
    for e, (f, i) in enumerate(seq, 1):
        decisions.append(stop.update(f, i, weights=[np.array([e])]))
        if decisions[-1]:
            break
    assert [r["monitored"] for r in stop.history] == [0.5 * (f + i) for f, i in seq[:5]]
    # epoch 2 improves by 0.5 (> 0.1); epoch 3 improves by 0.025 (< min_delta: stale but best)
    assert stop.best_epoch == 3
    assert [r["stale"] for r in stop.history] == [0, 0, 1, 2, 3]
    assert decisions[:4] == [False, False, False, False] and decisions[4] is True
    assert len(stop.history) == 5 and stop.stop_reason == "patience"
    assert np.array_equal(stop.best_weights[0], np.array([3]))


def test_dmo10_stopper_hits_ceiling_and_first_tie_wins():
    stop = T.MeanMonitorStopper(patience=10, min_delta=1e-8, max_epochs=3)
    for e in range(1, 4):
        last = stop.update(1.0, 1.0, weights=[np.array([e])])
    assert last is True and stop.stop_reason == "max_epochs" and stop.best_epoch == 1


def test_dmo10_train_unit_end_to_end_restore_and_reload(tmp_path):
    bundle = write_bundle(tmp_path / "bundle")
    out = tmp_path / "unit"
    receipt = T.train_unit(DESIGN, bundle, out, device="cpu",
                           training_overrides={"max_epochs": 25, "patience": 4})
    assert receipt["status"] == "COMPLETED", receipt
    assert receipt["scientific"] is False and receipt["overrides"] == {"max_epochs": 25, "patience": 4}
    training = json.loads((out / "TRAINING.json").read_text())
    hist = training["history"]
    for row in hist:
        assert math.isclose(row["monitored"], 0.5 * (row["fit_loss"] + row["inner_loss"]), rel_tol=0,
                            abs_tol=0)
    best = min(range(len(hist)), key=lambda i: (hist[i]["monitored"], i))
    assert training["best_epoch"] == best + 1
    assert receipt["identity"]["weights_sha256"] == hist[best]["weights_sha256"]
    # Re-evaluate the restored model on the same scaled arrays: identical monitored loss.
    loaded, _ = M.load_dmo(out / "model")
    eff = T.effective_from_design(DESIGN)
    data = T.load_data_bundle(bundle, eff)
    scales = json.loads((out / "SCALES.json").read_text())
    fit_x, fit_y, fit_m = T.transform(T.trainable_rows(data["fit"])[0], scales)
    in_x, in_y, in_m = T.transform(T.trainable_rows(data["inner"])[0], scales)
    print(f"DMO10 provenance: out={out} started={receipt['started_at']} "
          f"finished={receipt['finished_at']} updates={training['updates']} "
          f"epochs={training['epochs_completed']}")
    record = training["monitor_evaluation"]
    assert record["procedure"] == "i6dmo.monitor_eval.v2" and record["batch_size"] == 256
    assert record["monitor_instance"] == "canonical_loader_rebuild"
    assert record["loader_overhead_s"]["epochs"] == training["epochs_completed"]
    assert record["graph"] == "inference" and record["self_check"] == "PASS"
    assert record["partition"] == "sequential_row_id_fixed"  # the shuffle never reaches monitoring
    assert record["best_epoch"] == training["best_epoch"]
    assert record["monitored"] == hist[best]["monitored"]  # the value early stopping read
    witness = record["witness"]
    assert witness["ids"] == "row_id" and witness["code_witness"]
    assert len(witness["fit"]["batches"]) == -(-len(fit_x) // record["batch_size"])
    ids = (T.trainable_rows(data["fit"])[0]["row_id"], T.trainable_rows(data["inner"])[0]["row_id"])
    fm_r, im_r, monitored, tolerated = T.replay_monitored(loaded.forecast_model, (fit_x, fit_y, fit_m),
                                               (in_x, in_y, in_m), record, ids=ids)
    with pytest.raises(T.TrainDisposition, match="MONITOR_PATH_MISMATCH"):  # witness needs the ids
        T.replay_monitored(loaded.forecast_model, (fit_x, fit_y, fit_m), (in_x, in_y, in_m), record)
    import copy as _copy
    corrupted = _copy.deepcopy(record)
    corrupted["witness"]["fit"]["batches"][0]["partial_sums"][2] += 1e-9
    with pytest.raises(T.TrainDisposition, match="MONITOR_WITNESS_DIVERGENCE") as err:
        T.replay_monitored(loaded.forecast_model, (fit_x, fit_y, fit_m), (in_x, in_y, in_m),
                           corrupted, ids=ids)
    assert "batch 0" in str(err.value) and "row_ids" in str(err.value)
    assert monitored == hist[best]["monitored"]  # bitwise: identical evaluation procedure
    assert fm_r["sums"] == record["fit"]["sums"] and fm_r["counts"] == record["fit"]["counts"]
    assert im_r["sums"] == record["inner"]["sums"] and im_r["counts"] == record["inner"]["counts"]
    # Saved predictions equal a fresh prediction from the reloaded artifact.
    with np.load(out / "predictions" / "validation.npz") as pz:
        vx, _, _ = T.transform(data["validation"], scales)
        fresh = T.to_price(M.predict_matrix(loaded.forecast_model, vx), data["validation"], scales)
        assert np.array_equal(pz["predicted_price"], fresh)
        assert np.array_equal(pz["row_id"], data["validation"]["row_id"])
        assert pz["horizons_hours"].tolist() == HORIZONS
    for name, digest in receipt["artifacts"].items():
        assert T.file_sha256(out / name) == digest


def test_dmo18_outer_validation_perturbation_leaves_scales_and_weights_unchanged(tmp_path):
    a = write_bundle(tmp_path / "a", seed=0)
    splits = {"fit": split_arrays("2020-01-06T00:00:00", 96, 1),
              "inner": split_arrays("2023-11-13T00:00:00", 32, 2),
              "validation": split_arrays("2024-01-08T00:00:00", 40, 99)}
    b = write_bundle(tmp_path / "b", splits=splits)
    ra = T.train_unit(DESIGN, a, tmp_path / "ua", device="cpu", training_overrides={"max_epochs": 3})
    rb = T.train_unit(DESIGN, b, tmp_path / "ub", device="cpu", training_overrides={"max_epochs": 3})
    assert ra["identity"]["weights_sha256"] == rb["identity"]["weights_sha256"]
    assert (tmp_path / "ua" / "SCALES.json").read_text() == (tmp_path / "ub" / "SCALES.json").read_text()


def test_cli_entry_point_typed_exit_and_refuses_existing_out(tmp_path):
    from tools.i6n_contracts.feature_availability import LEGACY_MANIFEST_20
    bad = write_bundle(tmp_path / "bad", features=LEGACY_MANIFEST_20)
    cmd = [sys.executable, "-m", "tools.i6dmo.train", "--design", str(DESIGN), "--bundle", str(bad),
           "--out", str(tmp_path / "u"), "--device", "cpu", "--json"]
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 3, run.stderr[-2000:]
    receipt = json.loads(run.stdout)
    assert receipt["status"] == "FAILED_TYPED" and receipt["code"] == "STLFSI4_ALIAS"
    assert json.loads((tmp_path / "u" / "RECEIPT.json").read_text()) == receipt
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 2 and "OUT_DIR_NOT_EMPTY" in run.stdout + run.stderr
    help_text = subprocess.run([sys.executable, "-m", "tools.i6dmo.train", "--help"], capture_output=True,
                               text=True, cwd=REPO, env=ENV, check=False).stdout
    assert "--max-epochs" not in help_text and "--patience" not in help_text


def test_gpu_request_without_gpu_is_typed_no_cpu_fallback(tmp_path):
    bundle = write_bundle(tmp_path / "bundle")
    if tf.config.list_physical_devices("GPU"):
        pytest.skip("a GPU is visible; this negative control needs a CPU-only process")
    receipt = T.train_unit(DESIGN, bundle, tmp_path / "u", device="gpu")
    assert receipt["status"] == "FAILED_TYPED" and receipt["code"] == "NO_GPU_VISIBLE"
    assert not (tmp_path / "u" / "model").exists()


def test_determinism_is_enabled_and_recorded_everywhere(tmp_path):
    """DMO10 GPU identity: op determinism on for training AND inference, and recorded."""
    from tools.i6dmo import model as M
    bundle = write_bundle(tmp_path / "b")
    receipt = T.train_unit(DESIGN, bundle, tmp_path / "u", device="cpu",
                           training_overrides={"max_epochs": 2})
    assert receipt["status"] == "COMPLETED"
    assert receipt["determinism"]["op_determinism"] is True
    assert M.DETERMINISM["op_determinism"] is True  # build/load/predict paths share the switch
    doc = json.loads((tmp_path / "u" / "model" / "MODEL.json").read_text())
    assert doc["determinism"]["op_determinism"] is True
    loaded, document = M.load_dmo(tmp_path / "u" / "model")
    assert document["determinism"]["op_determinism"] is True
    assert M.DETERMINISM["op_determinism"] is True


def test_monitor_evaluation_is_one_shared_procedure_and_path_mismatch_is_refused(tmp_path):
    """DMO10: training monitor and replay share ONE function; a different batching is refused."""
    from tools.i6dmo import model as M
    rng = np.random.default_rng(0)
    cfg = M.model_config(["a", "b", "c"])
    bundle = M.build_dmo(dict(cfg, features=["a", "b", "c"]))
    x = rng.normal(size=(130, 13, 3)).astype("float32")  # two batches at batch_size 64
    y = rng.normal(size=(130, 6))
    mask = np.ones((130, 6), bool)
    fit, inner = (x, y, mask), (x[:40], y[:40], mask[:40])
    record = {"procedure": "i6dmo.monitor_eval.v2", "batch_size": 64,
              "graph": "inference", "partition": "sequential_row_id_fixed",
              "monitor_instance": "canonical_loader_rebuild"}
    fm, im, monitored, _ = T.replay_monitored(bundle.forecast_model, fit, inner, record)
    fm2, im2, monitored2, witness = T.evaluate_monitored(bundle.forecast_model, fit, inner, 64)
    assert witness["code_witness"] and witness["fit"]["batches"][0]["partial_sums"]
    assert monitored == monitored2 and fm["sums"] == fm2["sums"]  # same function, bitwise
    with pytest.raises(T.TrainDisposition, match="MONITOR_PATH_MISMATCH"):
        T.replay_monitored(bundle.forecast_model, fit, inner, record, batch_size=130)
    with pytest.raises(T.TrainDisposition, match="MONITOR_PATH_MISMATCH"):
        T.replay_monitored(bundle.forecast_model, fit, inner, {"procedure": "other", "batch_size": 64})


def test_training_graph_evaluated_monitor_is_refused_and_self_check_is_typed():
    """DMO10: only the inference-path monitor is comparable; the self-check is bitwise."""
    from tools.i6dmo import model as M
    rng = np.random.default_rng(1)
    bundle = M.build_dmo(M.model_config(["a", "b"]))
    x = rng.normal(size=(20, 13, 2)).astype("float32")
    y = rng.normal(size=(20, 6))
    mask = np.ones((20, 6), bool)
    fit, inner = (x, y, mask), (x[:8], y[:8], mask[:8])
    good = {"procedure": "i6dmo.monitor_eval.v2", "batch_size": 20, "graph": "inference",
            "partition": "sequential_row_id_fixed", "monitor_instance": "canonical_loader_rebuild"}
    _, _, monitored, _ = T.replay_monitored(bundle.forecast_model, fit, inner, good)
    with pytest.raises(T.TrainDisposition, match="MONITOR_PATH_MISMATCH"):
        T.replay_monitored(bundle.forecast_model, fit, inner, dict(good, graph="training"))
    for mixed in (dict(good, procedure="i6dmo.monitor_eval.v1"),
                  dict(good, monitor_instance="in_session")):
        with pytest.raises(T.TrainDisposition, match="MONITOR_PATH_MISMATCH"):
            T.replay_monitored(bundle.forecast_model, fit, inner, mixed)
    for bad in (dict(good, partition="seeded_permutation"), {k: v for k, v in good.items()
                                                             if k != "partition"}):
        with pytest.raises(T.TrainDisposition, match="MONITOR_PATH_MISMATCH.*partition|MONITOR_PATH_MISMATCH"):
            T.replay_monitored(bundle.forecast_model, fit, inner, bad)
    T.monitor_self_check(bundle.forecast_model, fit, inner, good, monitored)  # bitwise: passes
    with pytest.raises(T.TrainDisposition, match="MONITOR_SELF_CHECK_FAILED"):
        T.monitor_self_check(bundle.forecast_model, fit, inner, good, monitored + 1e-12)


def test_self_check_is_not_trivial_a_corrupted_history_value_fails(tmp_path, monkeypatch):
    """The self-check compares two SEPARATE computations: corrupting what the stopper
    recorded (the hist value) must make the recording-time recompute fail typed."""
    from tools.i6dmo import model as M

    class Corrupting(T.MeanMonitorStopper):
        def update(self, fit_loss, inner_loss, weights):
            decision = super().update(fit_loss, inner_loss, weights)
            self.history[-1]["monitored"] = self.history[-1]["monitored"] + 1e-9  # corrupt hist
            return decision

    monkeypatch.setattr(T, "MeanMonitorStopper", Corrupting)
    bundle = M.build_dmo(M.model_config(["a", "b"]))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 13, 2)).astype("float32")
    y = rng.normal(size=(40, 6))
    mask = np.ones((40, 6), bool)
    training = dict(T.effective_from_design(DESIGN)["training"], max_epochs=2)
    with pytest.raises(T.TrainDisposition, match="MONITOR_SELF_CHECK_FAILED"):
        T.fit_dmo(bundle, (x, y, mask), (x[:8], y[:8], mask[:8]), training, 1,
                  ids=(np.arange(40, dtype=np.int64), np.arange(8, dtype=np.int64)))


def test_static_algorithm_flags_are_forced_recorded_and_enforced(tmp_path):
    """Process-level GPU autotune is disabled; a record from other flags never scores."""
    import os
    from tools.i6dmo import model as M
    M.enable_determinism()
    flags = T.current_determinism_flags()
    assert flags["TF_CUDNN_USE_AUTOTUNE"] == "0" and flags["TF_DETERMINISTIC_OPS"] == "1"
    assert flags["TF_CUDNN_DETERMINISTIC"] == "1" and flags["op_determinism"] is True
    assert flags["jit_autoclustering"] is False
    assert os.environ["TF_CUDNN_USE_AUTOTUNE"] == "0"
    assert "--tf_xla_auto_jit=0" in os.environ.get("TF_XLA_FLAGS", "")
    assert M.DETERMINISM["env"]["TF_CUDNN_USE_AUTOTUNE"] == "0"
    bundle = M.build_dmo(M.model_config(["a", "b"]))
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 13, 2)).astype("float32")
    y, mask = rng.normal(size=(20, 6)), np.ones((20, 6), bool)
    fit, inner = (x, y, mask), (x[:8], y[:8], mask[:8])
    good = {"procedure": "i6dmo.monitor_eval.v2", "batch_size": 20, "graph": "inference",
            "partition": "sequential_row_id_fixed", "monitor_instance": "canonical_loader_rebuild",
            "determinism_flags": dict(flags, TF_CUDNN_USE_AUTOTUNE="1")}
    with pytest.raises(T.TrainDisposition, match="DETERMINISM_FLAGS_MISMATCH"):
        T.replay_monitored(bundle.forecast_model, fit, inner, good)


def test_predeclared_tolerance_applies_only_with_static_flags_and_small_deltas(tmp_path):
    import copy
    from tools.i6dmo import model as M
    bundle = M.build_dmo(M.model_config(["a", "b"]))
    rng = np.random.default_rng(6)
    x = rng.normal(size=(30, 13, 2)).astype("float32")
    y, mask = rng.normal(size=(30, 6)), np.ones((30, 6), bool)
    fit, inner = (x, y, mask), (x[:8], y[:8], mask[:8])
    base = {"procedure": "i6dmo.monitor_eval.v2", "batch_size": 15, "graph": "inference",
            "partition": "sequential_row_id_fixed", "monitor_instance": "canonical_loader_rebuild",
            "determinism_flags": T.current_determinism_flags()}
    _, _, monitored, witness = T.evaluate_monitored(bundle.forecast_model, fit, inner, 15)
    record = dict(base, witness=witness, monitored=monitored)
    perturbed = copy.deepcopy(record)
    for row in perturbed["witness"]["fit"]["batches"]:
        row["partial_sums"] = [v * (1 + 5e-6) for v in row["partial_sums"]]  # within 1e-5 relative
    fm, im, got, tolerated = T.replay_monitored(bundle.forecast_model, fit, inner, perturbed)
    assert tolerated and all(t["max_relative_delta"] <= 1e-5 for t in tolerated)
    assert got == monitored  # the recompute itself is unchanged; only the comparison tolerates
    big = copy.deepcopy(record)
    big["witness"]["fit"]["batches"][0]["partial_sums"][0] *= 1.001  # beyond the bound
    with pytest.raises(T.TrainDisposition, match="MONITOR_WITNESS_DIVERGENCE"):
        T.replay_monitored(bundle.forecast_model, fit, inner, big)
    unflagged = copy.deepcopy(perturbed)
    del unflagged["determinism_flags"]  # tolerance never activates without confirmed static flags
    with pytest.raises(T.TrainDisposition, match="MONITOR_WITNESS_DIVERGENCE"):
        T.replay_monitored(bundle.forecast_model, fit, inner, unflagged)
