#!/usr/bin/env python
"""Bayesian direct multi-output head on frozen 208-d embeddings.

Epoch metrics written to STATUS.json come from the Keras fit logs of that
epoch. This script does not call evaluate or predict to refresh them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

PHASE = Path("/var/tmp/retsu-phase1")
OUT = Path(os.environ.get("RETSU_OUT", str(PHASE)))
MAX_EPOCHS = 2000
BATCH = 256
LR = 3e-4
LR_FLOOR = 4.8e-7
LR_FACTOR = 0.2
LR_PATIENCE = 20
LR_GRACE = 20
ES_PATIENCE = 80
ES_START = 100
MIN_DELTA = 1e-8
KL_TARGET = 1e-3
KL_ANNEAL = 40
SIZES = (1575, 3150, 6300, 12600, 25200)


def write_status(payload: dict) -> None:
    payload = dict(payload)
    payload["updated_at"] = pd_now()
    payload["source"] = "keras_epoch_logs"
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "STATUS.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(OUT / "STATUS.json")


def pd_now() -> str:
    import pandas as pd
    return pd.Timestamp.now(tz="UTC").isoformat()


def price_mae_from_logs(logs: dict, prefix: str, horizons: list[int], scale: np.ndarray) -> dict:
    """Turn the MAE fit just logged into price units. No second forward pass.

    The head trains on (y - mean) / std. Logged loss is MAE in that unit.
    Price MAE is that logged MAE times the train std of the same horizon.
    """
    per = {}
    for i, horizon in enumerate(horizons):
        key = f"{prefix}output_horizon_{horizon}_loss"
        if key in logs:
            per[str(horizon)] = float(logs[key]) * float(scale[i])
    total = float(np.mean(list(per.values()))) if per else float("nan")
    return {"total": total, "per_horizon": per}


class GraceReduceLR(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, grace: int, **kwargs):
        super().__init__(**kwargs)
        self.grace = grace
        self.reductions = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 < self.grace:
            return
        before = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        super().on_epoch_end(epoch, logs)
        after = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if after < before - 1e-20:
            self.reductions += 1


class ChampionStatus(tf.keras.callbacks.Callback):
    def __init__(self, job: dict, early, plateau, kl_var: tf.Variable):
        super().__init__()
        self.job = job
        self.early = early
        self.plateau = plateau
        self.kl_var = kl_var
        self.best = np.inf
        self.best_price = None
        self.best_epoch = None
        self.best_weights = None
        self.best_opt = None
        self.final_lr = None
        self.log_path = OUT / "jobs" / job["name"] / "epochs.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        frac = min(1.0, (epoch + 1) / KL_ANNEAL)
        self.kl_var.assign(KL_TARGET * frac)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = float(logs["val_loss"])
        improved = val < self.best - MIN_DELTA
        if improved:
            self.best = val
            self.best_epoch = int(epoch + 1)
            self.best_weights = self.model.get_weights()
            self.best_opt = [v.numpy() for v in self.model.optimizer.variables]
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.final_lr = lr
        es_wait = int(getattr(self.early, "wait", 0) or 0)
        lr_wait = int(getattr(self.plateau, "wait", 0) or 0)
        scale = np.asarray(self.job["target_std"], dtype=np.float64)
        train = price_mae_from_logs(logs, "", self.job["horizons"], scale)
        valid = price_mae_from_logs(logs, "val_", self.job["horizons"], scale)
        if improved:
            self.best_price = valid["total"]
        record = {
            "state": "running",
            "stage": "train",
            "job": self.job["name"],
            "task": self.job["task"],
            "max_steps": self.job["max_steps"],
            "queue_index": self.job["queue_index"],
            "queue_total": self.job["queue_total"],
            "epoch": int(epoch + 1),
            "epochs_planned": MAX_EPOCHS,
            "early_stopping": {
                "wait": es_wait,
                "patience": ES_PATIENCE,
                "started": (epoch + 1) >= ES_START,
                "best_epoch": self.best_epoch,
                "min_delta": MIN_DELTA,
            },
            "reduce_lr": {
                "wait": lr_wait,
                "patience": LR_PATIENCE,
                "grace": LR_GRACE,
                "reductions": int(self.plateau.reductions),
                "factor": LR_FACTOR,
                "floor": LR_FLOOR,
            },
            "lr": lr,
            "train_mae": train["total"],
            "val_mae": valid["total"],
            "train_mae_per_horizon": train["per_horizon"],
            "val_mae_per_horizon": valid["per_horizon"],
            "best_val_mae": self.best_price,
            "target_mean": self.job["target_mean"],
            "target_std": self.job["target_std"],
            "embedding_scale": self.job["embedding_scale"],
            "mae_unit": os.environ.get("RETSU_MAE_UNIT", "seasonal_price"),
            "kl_weight": float(self.kl_var.numpy()),
            "naive_mae_monitor": self.job["naive_mae_monitor"],
            "champion_restored": False,
        }
        write_status(record)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def restore(self) -> None:
        if self.best_weights is None:
            raise RuntimeError("NO_CHAMPION")
        self.model.set_weights(self.best_weights)
        for variable, value in zip(self.model.optimizer.variables, self.best_opt):
            variable.assign(value)
        self.model.optimizer.learning_rate.assign(self.final_lr)
        champion = OUT / "jobs" / self.job["name"] / "champion.weights.h5"
        self.model.save_weights(champion)


def build(horizons: list[int], kl_var: tf.Variable, use_mean: tf.Variable) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(208,), name="embedding")
    outputs = []

    def tensor_fn(dist):
        return tf.cond(use_mean, lambda: dist.mean(), lambda: dist.sample())

    for horizon in horizons:
        suffix = f"_h{horizon}"
        flip_name = f"flipout{suffix}"

        def posterior(dtype, shape, name, trainable, add_variable_fn, flip_name=flip_name):
            # TFP passes its own name and the layer's add_variable. A raw
            # tf.Variable here is snapshotted in eager mode and never trains.
            del name
            return _posterior(dtype, shape, trainable, add_variable_fn, flip_name)

        def prior(dtype, shape, bias_size, trainable, name, flip_name=flip_name):
            return _prior(dtype, shape, bias_size, trainable, flip_name)

        def divergence(q, p, _ignore, kl_var=kl_var):
            return tfp.distributions.kl_divergence(q, p) * kl_var

        bayes = tfp.layers.DenseFlipout(
            1,
            activation="linear",
            kernel_posterior_fn=posterior,
            kernel_prior_fn=prior,
            kernel_divergence_fn=divergence,
            kernel_posterior_tensor_fn=tensor_fn,
            bias_posterior_fn=None,
            bias_prior_fn=None,
            name=flip_name,
        )(inputs)
        bias = tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=lambda shape, dtype=None: tf.random.normal(shape, stddev=0.05, seed=44, dtype=dtype),
            name=f"bias{suffix}",
        )(inputs)
        outputs.append(tf.keras.layers.Add(name=f"output_horizon_{horizon}")([bayes, bias]))
    model = tf.keras.Model(inputs, outputs, name=f"bayes_head_{len(horizons)}h")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
        loss={f"output_horizon_{h}": "mae" for h in horizons},
    )
    # Training samples the Flipout kernel. The validation step of fit uses the
    # posterior mean, so early stopping sees that mean and not a second pass.
    train_impl = model.train_step
    test_impl = model.test_step

    def train_step(data):
        use_mean.assign(False)
        return train_impl(data)

    def test_step(data):
        use_mean.assign(True)
        return test_impl(data)

    model.train_step = train_step
    model.test_step = test_step
    model.use_posterior_mean = use_mean
    return model


def _posterior(dtype, kernel_shape, trainable, add_variable_fn, name):
    """Mean-field kernel posterior registered on the Flipout layer.

    loc ~ N(0, 0.05), seed 123. The untransformed scale uses seed 124 and
    softplus(x + log(expm1(1))) so the initial scale is about 1. The upper
    clip is 10, not 1: a ceiling of 1 sits on that initializer and zeros the
    scale gradient. The prior remains N(0, 1).
    """
    c = float(np.log(np.expm1(1.0)))
    loc = add_variable_fn(
        name=f"{name}_loc",
        shape=list(kernel_shape),
        initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=123),
        dtype=dtype,
        trainable=trainable,
    )
    raw_scale = add_variable_fn(
        name=f"{name}_untransformed_scale",
        shape=list(kernel_shape),
        initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=124),
        dtype=dtype,
        trainable=trainable,
    )

    def to_scale(raw):
        return tf.clip_by_value(1e-3 + tf.nn.softplus(raw + c), 1e-3, 10.0)

    scale = tfp.util.DeferredTensor(raw_scale, to_scale)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc, scale=scale),
        reinterpreted_batch_ndims=len(kernel_shape),
    )


def assert_posterior_tracked(model: tf.keras.Model, horizons: list[int]) -> None:
    names = [variable.name for variable in model.trainable_weights]
    missing = []
    for horizon in horizons:
        for suffix in ("loc", "untransformed_scale"):
            token = f"flipout_h{horizon}_{suffix}"
            if not any(token in name for name in names):
                missing.append(token)
    if missing:
        raise RuntimeError("POSTERIOR_NOT_TRACKED " + ",".join(missing))


def _prior(dtype, kernel_shape, bias_size, trainable, name):
    del bias_size, trainable, name
    loc = tf.zeros(kernel_shape, dtype=dtype)
    scale = tf.ones(kernel_shape, dtype=dtype)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=loc, scale=scale),
        reinterpreted_batch_ndims=len(kernel_shape),
    )


def y_dict(y: np.ndarray, horizons: list[int]) -> dict[str, np.ndarray]:
    return {f"output_horizon_{h}": y[:, i] for i, h in enumerate(horizons)}


def naive_mae(y: np.ndarray, seasonal_origin: np.ndarray | None, mask: np.ndarray) -> float | None:
    return None


def one_job(data: dict, meta: dict, task: str, size: int, queue_index: int, queue_total: int) -> None:
    horizons = meta["horizons_short"] if task == "short" else meta["horizons_long"]
    if not horizons:
        write_status({
            "state": "running",
            "stage": "train_skipped",
            "job": f"{task}_{size}",
            "reason": "NO_HORIZON_WITH_SUPPORT",
            "queue_index": queue_index,
            "queue_total": queue_total,
        })
        return
    ok = data["ok_short"] if task == "short" else data["ok_long"]
    y_all = data["y_short"] if task == "short" else data["y_long"]
    pool = data["pool"] & ok
    monitor = data["monitor"] & ok
    available = int(np.flatnonzero(pool).size)
    if size > available:
        size = available
    if size < 1 or monitor_idx_empty(monitor):
        write_status({
            "state": "running",
            "stage": "train_skipped",
            "job": f"{task}_{size}",
            "reason": "EMPTY_SPLIT",
            "available": available,
            "queue_index": queue_index,
            "queue_total": queue_total,
        })
        return
    train_idx = np.flatnonzero(pool)[:size]
    monitor_idx = np.flatnonzero(monitor)
    y_train = y_all[train_idx]
    y_monitor = y_all[monitor_idx]
    # Train statistics only. Monitor and 2024 are transformed with these.
    target_mean = y_train.mean(axis=0).astype(np.float64)
    target_std = y_train.std(axis=0).astype(np.float64)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)
    y_train_z = ((y_train - target_mean) / target_std).astype(np.float32)
    y_monitor_z = ((y_monitor - target_mean) / target_std).astype(np.float32)
    # A N(0, 1) kernel on 208 inputs otherwise drowns a unit target in Flipout noise.
    # The scale is the train-set RMS times sqrt(fan-in), fixed before fit.
    z_train = data["z"][train_idx]
    fan_in = float(z_train.shape[1])
    z_scale = float(np.sqrt(np.mean(np.square(z_train)) * fan_in))
    z_scale = z_scale if z_scale > 1e-6 else 1.0
    origin = data["seasonal_origin"][monitor_idx][:, None]
    naive = float(np.mean(np.abs(y_monitor - origin)))
    job = {
        "name": f"{task}_{size}",
        "task": task,
        "max_steps": int(size),
        "horizons": list(horizons),
        "queue_index": queue_index,
        "queue_total": queue_total,
        "naive_mae_monitor": naive,
        "train_rows": int(len(train_idx)),
        "monitor_rows": int(len(monitor_idx)),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "embedding_scale": z_scale,
    }
    tf.keras.backend.clear_session()
    kl_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight")
    use_mean = tf.Variable(False, trainable=False, dtype=tf.bool, name="use_posterior_mean")
    model = build(list(horizons), kl_var, use_mean)
    assert_posterior_tracked(model, list(horizons))
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=ES_PATIENCE,
        min_delta=MIN_DELTA,
        start_from_epoch=ES_START,
        restore_best_weights=False,
    )
    plateau = GraceReduceLR(
        grace=LR_GRACE,
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_FLOOR,
        min_delta=MIN_DELTA,
        verbose=0,
    )
    book = ChampionStatus(job, early, plateau, kl_var)
    history = model.fit(
        z_train / z_scale,
        y_dict(y_train_z, list(horizons)),
        validation_data=(data["z"][monitor_idx] / z_scale, y_dict(y_monitor_z, list(horizons))),
        epochs=MAX_EPOCHS,
        batch_size=BATCH,
        shuffle=True,
        verbose=2,
        callbacks=[plateau, early, book],
    )
    try:
        book.restore()
    except Exception as exc:
        book.model.set_weights(book.best_weights)
        (OUT / "jobs" / job["name"] / "optimizer_restore_error.txt").write_text(repr(exc) + "\n")
    last = book.log_path.read_text().strip().splitlines()[-1]
    record = json.loads(last)
    record["champion_restored"] = True
    record["best_epoch"] = book.best_epoch
    record["epochs_ran"] = int(len(history.history["loss"]))
    record["state"] = "job_done"
    write_status(record)
    # One holdout forward after the champion exists. Not an epoch metric.
    hold = data["holdout"] & ok
    if int(hold.sum()) > 0:
        use_mean.assign(True)
        pred = model.predict(data["z"][hold] / z_scale, batch_size=BATCH, verbose=0)
        if len(horizons) == 1:
            pred = [pred]
        y_hold = y_all[hold]
        hold_mae = {
            str(h): float(np.mean(np.abs(pred[i].reshape(-1) * target_std[i] + target_mean[i] - y_hold[:, i])))
            for i, h in enumerate(horizons)
        }
        record["holdout_2024_mae_per_horizon"] = hold_mae
        record["holdout_2024_mae"] = float(np.mean(list(hold_mae.values())))
        record["holdout_rows"] = int(hold.sum())
        write_status(record)
        (OUT / "jobs" / job["name"] / "holdout.json").write_text(json.dumps(record, indent=2) + "\n")


def monitor_idx_empty(monitor: np.ndarray) -> bool:
    return int(np.flatnonzero(monitor).size) < 1


def already_stopped(name: str) -> bool:
    path = OUT / "jobs" / name / "epochs.jsonl"
    if not path.exists():
        return False
    last = json.loads(path.read_text().strip().splitlines()[-1])
    early = last.get("early_stopping") or {}
    wait, patience = early.get("wait"), early.get("patience")
    if wait is None or patience is None:
        return False
    return int(wait) >= int(patience)


def planned_jobs(data: dict, meta: dict) -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    pairs = [("short", "ok_short", meta.get("horizons_short") or [])]
    if meta.get("horizons_long"):
        pairs.append(("long", "ok_long", meta["horizons_long"]))
    for task, ok_key, horizons in pairs:
        if not horizons:
            continue
        pool = int((data["pool"] & data[ok_key]).sum())
        sizes = [size for size in SIZES if size <= pool]
        if pool >= 1 and pool not in sizes:
            sizes.append(pool)
        jobs.extend((task, size) for size in sizes)
    smoke = int(os.environ.get("SMOKE_EPOCHS", "0"))
    if smoke:
        return jobs[:1]
    return jobs


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    meta = json.loads((PHASE / "dataset_meta.json").read_text())
    dataset_path = Path(os.environ.get("RETSU_DATASET", str(PHASE / "dataset.npz")))
    raw = np.load(dataset_path)
    data = {key: raw[key] for key in raw.files}
    smoke = int(os.environ.get("SMOKE_EPOCHS", "0"))
    jobs = planned_jobs(data, meta)
    if smoke:
        global MAX_EPOCHS
        MAX_EPOCHS = smoke
    write_status({
        "state": "running",
        "stage": "train_start",
        "jobs": [f"{task}_{size}" for task, size in jobs],
        "target": "seasonal_standardized_on_train",
        "dataset": str(dataset_path),
    })
    pending = [(task, size) for task, size in jobs if not already_stopped(f"{task}_{size}")]
    for index, (task, size) in enumerate(pending, start=1):
        stale = OUT / "jobs" / f"{task}_{size}"
        for name in ("epochs.jsonl", "holdout.json"):
            old = stale / name
            if old.exists():
                old.unlink()
        one_job(data, meta, task, size, index, len(pending))
    write_status({
        "state": "queue_done" if not smoke else "smoke_done",
        "stage": "train",
        "jobs": [f"{task}_{size}" for task, size in jobs],
        "target": "seasonal_standardized_on_train",
        "dataset": str(dataset_path),
    })
    print("TRAIN_QUEUE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        write_status({"state": "failed", "stage": "train", "reason": "UNCAUGHT", "error": repr(exc)})
        raise
