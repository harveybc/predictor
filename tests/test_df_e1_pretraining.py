"""RP36 (dictum F3): the auto-encoder's stopping criterion must be a criterion — the same stimulus at
every epoch — and the restored checkpoint must reproduce the minimum it was chosen by.

The defect: the mask was drawn from (seed, epoch, batch index), so advancing an epoch changed 2 785
mask positions and moved the validation loss from 0.2598 to 0.3265 with the weights untouched. The
rules below are run through the REAL runner path (run_unit / WindowBatches / fit with its callbacks),
not through a helper that resembles it.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_e1_pilot")
E = _load("df_mod_e0")
RG = _load("df_e1_regimes")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rp36")
    rng = np.random.default_rng(0)
    n = 6 * P.DAY
    t = np.arange(n)
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    base = np.sin(2 * np.pi * t / P.DAY) + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({c: base * (i + 1) + rng.normal(size=n) * 0.05 + i for i, c in enumerate(cols)})
    ts = pd.date_range("2009-01-01 00:00", periods=n, freq="min")
    df.insert(0, "timestamp_label", ts.strftime("%d/%m/%Y %H:%M:%S"))
    panel = tmp / "panel.parquet"
    df.to_parquet(panel)
    design = P.seal(window=30, horizon=10, dev_train_days=4, dev_val_days=1, seeds=(1,), max_updates=20, ae_updates=40,
                    batch=32, patience_epochs=5, pilot_updates=10, internal_validation_fraction=0.2,
                    declared_task={"context_physical_seconds": 1800, "horizon_physical_seconds": 600, "purge": 40,
                                   "usable_windows_all_targets_valid": {}})
    design.pop("design_sha256")
    design["governed_bytes"] = {"path": str(panel), "sha256": P.sha_file(panel)}
    design["dev_subpartition"]["rows"] = [0, 5 * P.DAY]
    design["design_sha256"] = E.sha_obj(design)
    root = tmp / "root"
    data = P.prepare(design, root)
    job = {"kind": "ae", "cell_id": "ae_s1", "seed": 1, "max_updates": 40, "design": design,
           "data_npz": str(root / "DATA.npz"), "data_sha256": data["data_sha256"], "run_id": "t", "role": "CELL"}
    rec = P.run_unit(job, root / "attempts" / "ae_s1")
    return {"tmp": tmp, "design": design, "root": root, "data": data, "rec": rec, "job": job}


def test_RP36_the_validation_bank_is_the_same_stimulus_at_every_epoch_and_order(world):
    """With the weights untouched, advancing epochs, reshuffling and re-evaluating must not move the
    criterion by a single mask position."""
    WB = P._dataset_class()
    d = dict(np.load(world["root"] / "DATA.npz"))
    W, h, j = int(d["window"][0]), int(d["horizon"][0]), int(d["target_channel"][0])
    origins = np.asarray(d["train_origins"])[:64]
    ratio = world["design"]["pretraining"]["mask_ratio"]
    kw = dict(scaler_mean=d["scaler_mean"], scaler_sd=d["scaler_sd"])
    bank = WB(d["Xs"], d["Y"], origins, W, h, j, 16, shuffle=False, seed=0, masked=ratio, mask_mode="bank", **kw)
    first = [bank[i] for i in range(len(bank))]
    for _ in range(3):
        bank.on_epoch_end()
    again = [bank[i] for i in range(len(bank))]
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(first, again))
    # a shuffled bank shows the same windows the same masks, only in another order
    shuffled = WB(d["Xs"], d["Y"], origins, W, h, j, 16, shuffle=True, seed=7, masked=ratio, mask_mode="bank", **kw)
    flat_first = np.concatenate([a[1] for a in first])
    flat_shuf = np.concatenate([shuffled[i][1] for i in range(len(shuffled))])
    assert np.allclose(np.sort(flat_first.ravel()), np.sort(flat_shuf.ravel()))
    # the training stream, by contrast, is declared to vary by epoch
    train = WB(d["Xs"], d["Y"], origins, W, h, j, 16, shuffle=False, seed=1, masked=ratio, mask_mode="training", **kw)
    before = train[0][1].copy()
    train.on_epoch_end()
    assert not np.array_equal(before, train[0][1])


def test_RP36_the_measured_defect_is_gone_the_loss_does_not_move_with_the_epoch(world):
    """The dictum's measurement, repeated: same weights, same windows, epoch advanced."""
    WB = P._dataset_class()
    d = dict(np.load(world["root"] / "DATA.npz"))
    W, h, j = int(d["window"][0]), int(d["horizon"][0]), int(d["target_channel"][0])
    ratio = world["design"]["pretraining"]["mask_ratio"]
    ae, _ = RG.build_autoencoder(world["design"]["graph"]["assignment"], W, d["Xs"].shape[1], arch="A", seed=1, mask_ratio=ratio)
    ae.load_weights(str(world["root"] / "attempts" / "ae_s1" / "ae.weights.h5"))
    tf = E._tf()
    p = d["Xs"].shape[1]

    def masked_mse(y_true, y_pred):
        x, m = y_true[..., :p], y_true[..., p:]
        return tf.reduce_sum(m * tf.square(x - y_pred)) / (tf.reduce_sum(m) + 1e-8)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_mse)
    va = np.asarray(np.load(world["root"] / "attempts" / "ae_s1" / "arrays.npz")["ae_validation_origins"])[:64]
    bank = WB(d["Xs"], d["Y"], va, W, h, j, 32, shuffle=False, seed=0, masked=ratio, mask_mode="bank",
              bank_seed=world["design"]["pretraining"]["validation_bank_seed"],
              scaler_mean=d["scaler_mean"], scaler_sd=d["scaler_sd"])
    first = float(ae.evaluate(bank, verbose=0))
    changed = 0
    for _ in range(3):
        before = np.concatenate([bank[i][1] for i in range(len(bank))])
        bank.on_epoch_end()
        after = np.concatenate([bank[i][1] for i in range(len(bank))])
        changed += int((before != after).sum())
    second = float(ae.evaluate(bank, verbose=0))
    assert changed == 0 and abs(first - second) < 1e-9


def test_RP36_the_internal_validation_is_inside_train_purged_and_never_the_dev_validation(world):
    rec, d = world["rec"], dict(np.load(world["root"] / "DATA.npz"))
    arr = dict(np.load(world["root"] / "attempts" / "ae_s1" / "arrays.npz"))
    tr, va, ev = set(map(int, d["train_origins"])), set(map(int, arr["ae_validation_origins"])), set(map(int, d["eval_origins"]))
    assert va and va <= tr and not (va & ev)                       # inside train, disjoint from the DEV validation
    pre = set(map(int, arr["ae_train_origins"]))
    assert pre and not (pre & va) and min(va) - max(pre) >= rec["pretraining"]["purge_between"]
    assert "never read" in rec["pretraining"]["internal_validation"]
    assert rec["pretraining"]["validation_bank_sha256"] == P.mask_bank_digest(
        arr["ae_validation_origins"], int(d["window"][0]), d["Xs"].shape[1],
        rec["pretraining"]["mask_ratio"], bank_seed=world["design"]["pretraining"]["validation_bank_seed"])


def test_RP36_the_restored_checkpoint_reproduces_the_minimum_it_was_chosen_by(world):
    pre = world["rec"]["pretraining"]
    assert pre["evaluation_is_repeatable"], "two evaluations of the same weights on the bank disagreed"
    assert pre["restored_reproduces_best"], (pre["reconstruction_val_mse_masked"], pre["best_validation_loss"])
    assert pre["best_epoch"] == int(np.argmin(pre["curve"]["validation"])) + 1
    assert pre["stop_reason"] in ("UPDATE_BUDGET", "EARLY_STOPPING", "EPOCH_BUDGET")


def test_RP36_a_resumed_pretraining_keeps_the_trajectory_at_the_scope_chosen(world):
    """Resume = reload the saved auto-encoder and evaluate: the criterion is where it was left. The
    optimiser state is NOT saved, which is declared here rather than implied by a green test."""
    d = dict(np.load(world["root"] / "DATA.npz"))
    W = int(d["window"][0])
    p = d["Xs"].shape[1]
    ratio = world["design"]["pretraining"]["mask_ratio"]
    ae, _ = RG.build_autoencoder(world["design"]["graph"]["assignment"], W, p, arch="A", seed=1, mask_ratio=ratio)
    ae.load_weights(str(world["root"] / "attempts" / "ae_s1" / "ae.weights.h5"))
    det = RG.detector_layer_names(ae)
    assert RG.weights_digest(ae, det) == world["rec"]["pretraining"]["detector_digest"]
