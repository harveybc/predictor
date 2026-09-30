"""Staged representation pretraining: branch AEs -> fixed donors -> materialized
fusion -> core AE, with early stopping, reconstruction accounting and identities.

Stages (each one early-stopped on its own internal validation, never on the
outer validation, the external test or any holdout):

1. ``branch_ae``: one autoencoder per branch on the train population only. The
   selected encoder is exported as a donor, reloaded through ``load_donor`` and
   its outputs compared with the in-memory encoder (reload parity). The decoder
   is exported beside it so the reconstruction can be replayed.
2. ``fusion_materialization``: the model is REBUILT with every branch in regime
   R1 from the exported donor files (fixed, frozen weights), and the fused
   sequences of train and internal validation are written in bounded batches to
   memory-mapped ``.npy`` files. The fusion output lies on the branch right-edge
   time grid; its train-only channel statistics are recorded. "Bounded" here is
   bounded memory: nothing is held whole in RAM and the fusion contract forbids
   trainable or value-changing transforms, so the representation is NOT
   range-clipped.
3. ``core_ae``: the core autoencoder trains on the materialized bytes re-read
   from disk (digest re-verified). The exported core donor's manifest binds the
   exact upstream branch manifests, branch weight digests and fusion identity
   (engine contract), and a provenance sidecar binds the branch donor files and
   fused materialization digests. A core donor loaded under any other upstream
   is refused by ``build_modular`` before any fit.

Compression is lossy. Every stage reports reconstruction MAE/MSE on train and
internal validation next to a zero-information reference (per-channel train
mean), and the relative MSE. Downstream utility is measured separately by the
forecasting evaluator; see ``run_synthetic_pilot``.

Only explicit train and internal ``train_validation`` populations are accepted.
Population boundaries are caller declarations; array digests are computed here.
This module does not grant data-governance authority or read an external test.
Run on CPU with CUDA_VISIBLE_DEVICES="" under crispdm-run for bounded fixtures.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = "modular.pretrain.v2"
PROVENANCE = ("synthetic_fixture", "local_file", "governed_resource")


# --------------------------------------------------------------------------- heartbeat
def _resources():
    out = {"pid": os.getpid()}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                out[line.split(":")[0].lower() + "_kib"] = int(line.split()[1])
    except OSError:
        pass
    t = os.times()
    out["cpu_seconds"] = round(t.user + t.system, 3)
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        base = Path("/sys/fs/cgroup") / rel.lstrip("/")
        for name in ("memory.current", "memory.peak", "memory.max"):
            f = base / name
            if f.exists():
                value = f.read_text().strip()
                out["cgroup_" + name.replace(".", "_")] = int(value) if value.isdigit() else value
    except (OSError, IndexError):
        pass
    return out


class Heartbeat:
    """Background writer: one JSON line at most ``interval`` seconds apart.

    A daemon thread writes the latest state even while a batch or materialization
    is in flight, so cadence does not depend on training progress. Each line is
    flushed and fsynced (unbuffered). Fields: stage, epoch/update progress, last
    completed checkpoint, resources and the ETA basis.
    """

    def __init__(self, path, interval=30.0):
        if not (0 < float(interval) <= 60):
            raise ValueError("heartbeat interval must be in (0, 60] seconds")
        self.path, self.interval = Path(path), float(interval)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = {"stage": "starting", "last_checkpoint": None}
        self.started = time.time()
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="m02-heartbeat")
        self.beats = 0

    def __enter__(self):
        self.write()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.update(stage="finished" if exc[0] is None else "failed",
                    error=None if exc[0] is None else f"{exc[0].__name__}: {exc[1]}")
        self._stop.set()
        self._thread.join(timeout=5)
        self.write()
        return False

    def update(self, **fields):
        with self._lock:
            self.state.update(fields)

    def progress(self, stage):
        """Adapter for fit_with_early_stopping(progress=...)."""
        def callback(info):
            eta = None
            if info.get("event") == "monitor" or info.get("updates"):
                done = max(info.get("epoch", 1) - 1, 0)
                per_epoch = info["elapsed_seconds"] / max(done, 1)
                eta = {"basis": "estimate: remaining max_epochs x observed mean epoch seconds, capped by "
                                "remaining max_seconds; patience or max_updates may stop sooner",
                       "seconds_estimate": round(
                           min(per_epoch * (info["max_epochs"] - done),
                               max(info["max_seconds"] - info["elapsed_seconds"], 0)), 1)}
            self.update(stage=stage, fit=info, eta=eta)
        return callback

    def write(self):
        with self._lock:
            line = {"time_unix": round(time.time(), 3), "uptime_seconds": round(time.time() - self.started, 3),
                    "beat": self.beats, **copy.deepcopy(self.state), "resources": _resources()}
            self.beats += 1
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _run(self):
        while not self._stop.wait(self.interval):
            self.write()


class _NullHeartbeat:
    def update(self, **fields):
        pass

    def progress(self, stage):
        return None


# --------------------------------------------------------------------------- identities
def _population_pair(train, validation):
    for pop, expected in ((train, "train"), (validation, "train_validation")):
        if pop.get("split") != expected or pop.get("time_unit") != "seconds":
            raise ValueError("pretraining requires train and internal train_validation in seconds")
        if not isinstance(pop.get("dataset_id"), str) or not pop["dataset_id"]:
            raise ValueError("dataset_id is required")
        for field in ("support_start", "support_end"):
            value = pop.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {field}")
        if pop["support_start"] > pop["support_end"]:
            raise ValueError("reversed support")
        if pop.get("provenance", "undeclared") not in (*PROVENANCE, "undeclared"):
            raise ValueError(f"provenance must be one of {PROVENANCE}")
    if train["dataset_id"] != validation["dataset_id"]:
        raise ValueError("dataset identity differs")
    if train.get("provenance", "undeclared") != validation.get("provenance", "undeclared"):
        raise ValueError("train and internal validation provenance differ")
    if train["support_end"] >= validation["support_start"]:
        raise ValueError("train and internal validation supports overlap")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_identity(values, batch_size=64):
    if len(values.shape) != 3 or not len(values) or any(d <= 0 for d in values.shape):
        raise ValueError("nonempty rank-three windows required")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(values.shape), "dtype": str(values.dtype)}, sort_keys=True).encode())
    for start in range(0, len(values), batch_size):
        block = np.ascontiguousarray(values[start:start + batch_size])
        if not np.issubdtype(block.dtype, np.floating) or not np.isfinite(block).all():
            raise ValueError("finite floating point windows required")
        digest.update(block.tobytes())
    return {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": digest.hexdigest()}


class FeatureView:
    """Select channels only for the batch being consumed."""
    def __init__(self, source, indices):
        self.source, self.indices = source, indices
        self.shape = (*source.shape[:2], len(indices))
        self.dtype = source.dtype

    def __len__(self):
        return len(self.source)

    def __getitem__(self, key):
        return self.source[key][..., self.indices]


# --------------------------------------------------------------------------- measurements
def _channel_stats(values, batch_size=256):
    """Train-only per-channel mean/std/min/max accumulated in float64 batches."""
    channels = values.shape[-1]
    total, squared, count = np.zeros(channels), np.zeros(channels), 0
    low, high = np.full(channels, np.inf), np.full(channels, -np.inf)
    for start in range(0, len(values), batch_size):
        block = np.asarray(values[start:start + batch_size], dtype=np.float64).reshape(-1, channels)
        total += block.sum(0)
        squared += np.square(block).sum(0)
        low, high = np.minimum(low, block.min(0)), np.maximum(high, block.max(0))
        count += len(block)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean ** 2, 0.0))
    return {"mean": mean.tolist(), "std": std.tolist(), "min": low.tolist(), "max": high.tolist(),
            "observations_per_channel": int(count)}


def _reconstruction(model, values, train_mean, train_std, batch_size):
    """Reconstruction error of ``model`` against a per-channel train-mean reference.

    Errors are in the input space the stage consumes. ``relative_MSE`` < 1 means
    the reconstruction beats a zero-information constant; it is never 0 here.
    ``standardized_MSE`` divides each channel by its train std (0-std channels are
    excluded and counted) so wide channels do not dominate the fused-space value.
    """
    mean = np.asarray(train_mean)
    std = np.asarray(train_std)
    usable = std > 0
    abs_sum = sq_sum = base_abs = base_sq = std_sq = 0.0
    per_channel_sq = np.zeros(values.shape[-1])
    count = 0
    for start in range(0, len(values), batch_size):
        block = np.asarray(values[start:start + batch_size], dtype=np.float32)
        recon = np.asarray(model(block, training=False), dtype=np.float64)
        if recon.shape != block.shape or not np.isfinite(recon).all():
            raise ValueError("reconstruction must be finite and match its input")
        err = recon - block
        base = mean - block.astype(np.float64)
        abs_sum += np.abs(err).sum()
        sq_sum += np.square(err).sum()
        base_abs += np.abs(base).sum()
        base_sq += np.square(base).sum()
        per_channel_sq += np.square(err).sum((0, 1))
        if usable.any():
            std_sq += np.square(err[..., usable] / std[usable]).sum()
        count += block.shape[0] * block.shape[1]
    n = count * values.shape[-1]
    mse, bmse = sq_sum / n, base_sq / n
    return {"rows": int(len(values)), "MAE": abs_sum / n, "MSE": mse,
            "reference": "per-channel train mean (zero-information constant)",
            "reference_MAE": base_abs / n, "reference_MSE": bmse,
            "relative_MSE": mse / bmse if bmse else None,
            "standardized_MSE": std_sq / (count * int(usable.sum())) if usable.any() else None,
            "zero_std_channels": int((~usable).sum()),
            "per_channel_MSE": (per_channel_sq / count).tolist()}


def _parity(reference, candidate, values, batch_size):
    worst = 0.0
    for start in range(0, len(values), batch_size):
        block = np.asarray(values[start:start + batch_size], dtype=np.float32)
        a = np.asarray(reference(block, training=False))
        b = np.asarray(candidate(block, training=False))
        if a.shape != b.shape:
            raise ValueError("reload parity: output shapes differ")
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def _materialize(model, values, path, batch_size, beat, label):
    shape = (len(values), *tuple(int(n) for n in model.output_shape[1:]))
    output = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)
    for start in range(0, len(values), batch_size):
        encoded = np.asarray(model(np.asarray(values[start:start + batch_size]), training=False))
        if not np.isfinite(encoded).all():
            raise ValueError("nonfinite fused representation")
        output[start:start + len(encoded)] = encoded
        beat.update(materialization={"split": label, "rows_done": start + len(encoded), "rows": len(values)})
    output.flush()
    del output
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------- pipeline
def _stage_config(fit_config, stage_fit_configs, stage):
    cfg = dict(fit_config)
    cfg.update((stage_fit_configs or {}).get(stage, {}))
    return cfg


def pretrain_components(config, train_x, validation_x, output_dir, fit_config,
                        train_population, validation_population, *,
                        stage_fit_configs=None, seed=None, heartbeat=None):
    """Run branch AE -> fixed-donor fused materialization -> core AE.

    ``fit_config`` applies to every stage; ``stage_fit_configs`` may override it
    per stage ("branch", "core"). Returns the PRETRAIN.json document, including
    ``fine_tune_config`` (every component R2 on the exported donors). The
    caller's config is never mutated.
    """
    _population_pair(train_population, validation_population)
    train_identity = _array_identity(train_x)
    validation_identity = _array_identity(validation_x)
    if train_x.shape[1:] != validation_x.shape[1:]:
        raise ValueError("train/validation window schemas differ")
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    import tensorflow as tf
    from predictor_plugins.modular_temporal import (build_autoencoder, build_modular, load_donor,
                                                    save_donor, weights_hash)
    from tools.modular_candidate_evaluator import fit_with_early_stopping

    beat = heartbeat or _NullHeartbeat()
    base = copy.deepcopy(config)
    if any(c.get("regime", "R0") != "R0" or c.get("donor")
           for c in [*base.get("branches", []), base.get("core", {})]):
        raise ValueError("fresh pretraining requires R0 components without donors")
    if seed is not None:
        tf.keras.utils.set_random_seed(int(seed))
    bundle = build_modular(base)
    if tuple(bundle.forecast_model.input_shape[1:]) != tuple(train_x.shape[1:]):
        raise ValueError("data does not match model input")
    out.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(bundle.config)
    names = resolved["feature_names"]
    grids = {"input_right_edges_hours": [(i + 1) * resolved["sample_hours"] for i in range(resolved["window"])],
             "branch_right_edges_hours": list(bundle.branch_time_grid),
             "core_right_edges_hours": list(bundle.core_time_grid)}

    # ---- stage 1: branch autoencoders ------------------------------------------------
    records = []
    for index, spec in enumerate(resolved["branches"]):
        name = spec["name"]
        stage = f"branch_ae:{name}"
        beat.update(stage=stage, fit=None, eta=None)
        cfg = _stage_config(fit_config, stage_fit_configs, "branch")
        batch = int(cfg.get("batch_size", 32))
        columns = [names.index(feature) for feature in spec["features"]]
        train_view, val_view = FeatureView(train_x, columns), FeatureView(validation_x, columns)
        stats = _channel_stats(train_view)
        encoder = bundle.branch_models[name]
        ae = build_autoencoder(encoder)
        started = time.monotonic()
        training = fit_with_early_stopping(ae, train_view, train_view, val_view, val_view,
                                           {**cfg, "progress": beat.progress(stage)})
        if training["observed_updates"] < 1:
            raise ValueError("branch AE made no optimizer update")
        reconstruction = {
            "train": _reconstruction(ae, train_view, stats["mean"], stats["std"], batch),
            "train_validation": _reconstruction(ae, val_view, stats["mean"], stats["std"], batch)}
        donor = out / f"branch_{index:03d}.keras"
        manifest = bundle.donor_manifest("branch", name)
        sidecar = save_donor(encoder, donor, manifest)
        decoder = ae.layers[-1]
        decoder_path = out / f"branch_{index:03d}.decoder.keras"
        decoder.save(decoder_path)
        loaded = load_donor(donor, manifest)
        parity = _parity(encoder, loaded, val_view, batch)
        if parity > 1e-6 or weights_hash(loaded) != sidecar["weights_sha256"]:
            raise ValueError(f"branch {name} donor reload parity failed ({parity})")
        record = {"stage": "branch_ae", "name": name, "features": spec["features"],
                  "donor": str(donor), "donor_sha256": _file_sha(donor),
                  "donor_sidecar_sha256": _file_sha(donor.with_suffix(".manifest.json")),
                  "donor_weights_sha256": sidecar["weights_sha256"],
                  "donor_manifest_sha256": sidecar["manifest_sha256"],
                  "decoder": str(decoder_path), "decoder_sha256": _file_sha(decoder_path),
                  "encoder_parameters": int(encoder.count_params()),
                  "decoder_parameters": int(decoder.count_params()),
                  "train_channel_stats": stats, "training": training,
                  "reconstruction": reconstruction,
                  "reload_parity": {"passed": True, "max_abs_error": parity, "atol": 1e-6,
                                    "rows": int(len(validation_x)), "split": "train_validation"},
                  "wall_seconds": time.monotonic() - started}
        records.append(record)
        beat.update(last_checkpoint={"stage": "branch_ae", "name": name, "donor_sha256": record["donor_sha256"]})
        spec.update(regime="R2", donor=str(donor))

    # ---- stage 2: fused materialization from FIXED exported branch donors -------------
    beat.update(stage="fusion_materialization", fit=None, eta=None)
    fixed = copy.deepcopy(resolved)
    for spec in fixed["branches"]:
        spec["regime"] = "R1"
    fixed["core"].update(regime="R0", donor=None)
    fixed_bundle = build_modular(fixed)
    for rec, model in zip(records, fixed_bundle.branch_models.values()):
        if weights_hash(model) != rec["donor_weights_sha256"] or model.trainable_weights:
            raise ValueError("materialization branch is not the fixed exported donor")
    if list(fixed_bundle.branch_time_grid) != grids["branch_right_edges_hours"]:
        raise ValueError("fusion grid differs from the branch right-edge grid")
    fused_batch = int(_stage_config(fit_config, stage_fit_configs, "core").get("batch_size", 32))
    started = time.monotonic()
    _materialize(fixed_bundle.fusion_model, train_x, out / "fused_train.npy", fused_batch, beat, "train")
    _materialize(fixed_bundle.fusion_model, validation_x, out / "fused_train_validation.npy",
                 fused_batch, beat, "train_validation")
    fusion_record = {
        "stage": "fusion_materialization", "right_edge_grid_hours": grids["branch_right_edges_hours"],
        "branch_donors": [{"name": r["name"], "donor_sha256": r["donor_sha256"],
                           "donor_sidecar_sha256": r["donor_sidecar_sha256"],
                           "weights_sha256": r["donor_weights_sha256"]} for r in records],
        "fusion_identity": fixed_bundle.donor_manifest("core")["upstream"]["fusion"],
        "inputs": {"train": train_identity, "train_validation": validation_identity},
        "files": {}, "bounded_batch_rows": fused_batch,
        "range_bounded": False,
        "wall_seconds": time.monotonic() - started}
    for split in ("train", "train_validation"):
        path = out / f"fused_{split}.npy"
        fusion_record["files"][split] = {"path": str(path), "file_sha256": _file_sha(path),
                                         **_array_identity(np.load(path, mmap_mode="r"))}
    fused_train_view = np.load(out / "fused_train.npy", mmap_mode="r")
    fused_stats = _channel_stats(fused_train_view)
    fusion_record["train_channel_stats"] = fused_stats
    (out / "FUSION.json").write_text(json.dumps(fusion_record, indent=2, allow_nan=False) + "\n")
    beat.update(last_checkpoint={"stage": "fusion_materialization",
                                 "fused_train_sha256": fusion_record["files"]["train"]["sha256"]})

    # ---- stage 3: core autoencoder on the materialized bytes -------------------------
    beat.update(stage="core_ae", fit=None, eta=None)
    cfg = _stage_config(fit_config, stage_fit_configs, "core")
    batch = int(cfg.get("batch_size", 32))
    fused = {}
    for split in ("train", "train_validation"):
        array = np.load(out / f"fused_{split}.npy", mmap_mode="r")
        if _array_identity(array)["sha256"] != fusion_record["files"][split]["sha256"]:
            raise ValueError("materialized fusion changed on disk")
        fused[split] = array
    core = fixed_bundle.core_model
    core_ae = build_autoencoder(core)
    started = time.monotonic()
    core_training = fit_with_early_stopping(core_ae, fused["train"], fused["train"],
                                           fused["train_validation"], fused["train_validation"],
                                           {**cfg, "progress": beat.progress("core_ae")})
    core_reconstruction = {
        split: _reconstruction(core_ae, fused[split], fused_stats["mean"], fused_stats["std"], batch)
        for split in ("train", "train_validation")}
    donor = out / "core.keras"
    manifest = fixed_bundle.donor_manifest("core")
    sidecar = save_donor(core, donor, manifest)
    decoder_path = out / "core.decoder.keras"
    core_ae.layers[-1].save(decoder_path)
    loaded = load_donor(donor, manifest)
    parity = _parity(core, loaded, fused["train_validation"], batch)
    if parity > 1e-5 or weights_hash(loaded) != sidecar["weights_sha256"]:
        raise ValueError(f"core donor reload parity failed ({parity})")
    provenance = {"schema": "modular.core_donor.provenance.v1", "stage": "core_ae",
                  "core_donor_sha256": _file_sha(donor),
                  "core_manifest_sha256": sidecar["manifest_sha256"],
                  "upstream_branch_donors": fusion_record["branch_donors"],
                  "fusion_identity": fusion_record["fusion_identity"],
                  "fused_materialization": {s: fusion_record["files"][s]["sha256"] for s in fused},
                  "fusion_record_sha256": _file_sha(out / "FUSION.json")}
    provenance_path = out / "core.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    core_record = {"stage": "core_ae", "donor": str(donor), "donor_sha256": provenance["core_donor_sha256"],
                   "donor_sidecar_sha256": _file_sha(donor.with_suffix(".manifest.json")),
                   "donor_weights_sha256": sidecar["weights_sha256"],
                   "donor_manifest_sha256": sidecar["manifest_sha256"],
                   "provenance": str(provenance_path), "provenance_sha256": _file_sha(provenance_path),
                   "decoder": str(decoder_path), "decoder_sha256": _file_sha(decoder_path),
                   "encoder_parameters": int(core.count_params()),
                   "decoder_parameters": int(core_ae.layers[-1].count_params()),
                   "upstream_bound": {"branch_weights_sha256": [b["weights_sha256"] for b in manifest["upstream"]["branches"]],
                                      "fusion": manifest["upstream"]["fusion"]},
                   "training": core_training, "reconstruction": core_reconstruction,
                   "reload_parity": {"passed": True, "max_abs_error": parity, "atol": 1e-5,
                                     "rows": int(len(fused["train_validation"])), "split": "train_validation"},
                   "wall_seconds": time.monotonic() - started}
    beat.update(last_checkpoint={"stage": "core_ae", "donor_sha256": core_record["donor_sha256"]})
    resolved["core"].update(regime="R2", donor=str(donor))
    result = {"schema": SCHEMA, "branches": records, "fusion": fusion_record, "core": core_record,
              "grids": grids, "seed": seed,
              "train_population": train_population, "validation_population": validation_population,
              "train_input": train_identity, "validation_input": validation_identity,
              "fused_train": fusion_record["files"]["train"],
              "fused_validation": fusion_record["files"]["train_validation"],
              "fine_tune_config": resolved,
              "scope": ("representation pretraining; reconstruction is lossy and reported per stage; "
                        "downstream utility is measured only by a separate forecasting evaluation")}
    (out / "PRETRAIN.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def regime_config(fine_tune_config, regime):
    """Common-regime copy of a pretraining result's config: R0 drops donors."""
    config = copy.deepcopy(fine_tune_config)
    for component in [*config["branches"], config["core"]]:
        if regime == "R0":
            component.update(regime="R0", donor=None)
        elif regime in ("R1", "R2"):
            if not component.get("donor"):
                raise ValueError("R1/R2 require donors")
            component["regime"] = regime
        else:
            raise ValueError("regime must be R0, R1 or R2")
    return config


# --------------------------------------------------------------------------- synthetic pilot
def synthetic_series(rows, features, seed):
    """Declared SYNTHETIC fixture: hourly sinusoids + AR(1) noise, cross-coupled."""
    rng = np.random.default_rng(seed)
    t = np.arange(rows, dtype=np.float64)
    series = np.zeros((rows, features))
    noise = np.zeros(features)
    for i in range(rows):
        noise = 0.8 * noise + rng.normal(scale=0.3, size=features)
        series[i] = noise
    for f in range(features):
        series[:, f] += np.sin(2 * np.pi * t / (24 * (1 + f % 3))) + 0.5 * np.sin(2 * np.pi * t / (7 + 5 * f))
    if features > 1:
        series[1:, 1] += 0.6 * series[:-1, 0]
    return series


def _windows(series, starts, window):
    return np.stack([series[s:s + window] for s in starts]).astype(np.float32)


def build_synthetic_splits(*, rows, features, window, horizons, seed, target_index=0, t0=1_700_000_000):
    """Chronological train / outer validation windows with purge gaps. No test rows exist.

    AE internal validation is carved from the END of the forecast train windows,
    with a purge of ``window`` samples so AE train and internal validation share
    no observation. The outer validation (forecast selection) starts after the
    last train target plus a window purge.
    """
    series = synthetic_series(rows, features, seed)
    hmax = max(horizons)
    n_train_end = int(rows * 0.7)
    mean = series[:n_train_end].mean(0)
    std = series[:n_train_end].std(0)
    z = (series - mean) / std
    origins = np.arange(window - 1, rows - hmax)
    train_o = origins[origins + hmax < n_train_end]
    val_o = origins[origins - (window - 1) > n_train_end + window]
    sec = 3600

    def pack(o, split):
        starts = o - (window - 1)
        targets = np.stack([z[o + h, target_index] for h in horizons], axis=1)[..., None].astype(np.float32)
        ts = t0 + o.astype(np.int64) * sec
        return dict(windows=_windows(z, starts, window), targets=targets,
                    row_ids=np.array([f"syn-{int(v)}" for v in o]), timestamps=ts,
                    target_timestamps=ts[:, None] + np.asarray(horizons, dtype=np.int64) * sec,
                    dataset_id=np.array("synthetic_fixture_m02_v1"), split=np.array(split),
                    feature_names=np.array([f"f{i}" for i in range(features)]),
                    target_names=np.array([f"f{target_index}"]), horizons=np.asarray(horizons, dtype=np.int64),
                    timestamp_unit=np.array("seconds"), metric_space=np.array("z_train"),
                    scaler_identity=np.array("synthetic-train-zscore-v1"),
                    scaler_scale=np.array([std[target_index]], dtype=np.float64))

    train, validation = pack(train_o, "train"), pack(val_o, "validation")
    # AE internal split inside forecast train, purged by one window.
    n_ae = int(len(train_o) * 0.8)
    ae_train_idx = np.arange(n_ae)
    ae_val_idx = np.arange(n_ae + window, len(train_o))
    ae_train_end = int(t0 + train_o[n_ae - 1] * sec)
    ae_val_start = int(t0 + (train_o[ae_val_idx[0]] - (window - 1)) * sec)
    populations = (
        {"split": "train", "dataset_id": "synthetic_fixture_m02_v1", "provenance": "synthetic_fixture",
         "support_start": int(t0 + (train_o[0] - (window - 1)) * sec), "support_end": ae_train_end,
         "time_unit": "seconds"},
        {"split": "train_validation", "dataset_id": "synthetic_fixture_m02_v1", "provenance": "synthetic_fixture",
         "support_start": ae_val_start, "support_end": int(t0 + train_o[-1] * sec), "time_unit": "seconds"})
    return train, validation, ae_train_idx, ae_val_idx, populations


def run_synthetic_pilot(out, *, rows=2400, features=3, horizons=(1, 3, 6), seed=7,
                        fit=None, heartbeat_path=None, heartbeat_interval=30.0):
    """End-to-end: pretrain (synthetic, local) then forecast R0/R1/R2 with the real evaluator."""
    out = Path(out)
    if out.exists():
        raise ValueError("output directory must not exist")
    fit = dict(fit or {})
    horizons = list(horizons)
    from predictor_plugins.modular_temporal import default_config
    from tools.modular_candidate_evaluator import evaluate_candidate

    base = default_config([f"f{i}" for i in range(features)])
    base["horizons"] = horizons
    train, validation, ae_tr, ae_va, pops = build_synthetic_splits(
        rows=rows, features=features, window=base["window"], horizons=horizons, seed=seed)
    out.mkdir(parents=True)
    np.savez(out / "train.npz", **train)
    np.savez(out / "validation.npz", **validation)
    beat = Heartbeat(heartbeat_path or out / "heartbeat.jsonl", heartbeat_interval)
    ae_fit = {k: v for k, v in fit.items()}
    started = time.monotonic()
    with beat:
        beat.update(stage="pretraining", eta={"basis": "per-stage upper bounds reported inside each fit"})
        pre = pretrain_components(base, train["windows"][ae_tr], train["windows"][ae_va], out / "pretrain",
                                  ae_fit, *pops, seed=seed, heartbeat=beat)
        forecasts = {}
        for regime in ("R0", "R1", "R2"):
            beat.update(stage=f"forecast:{regime}", fit=None, eta=None)
            model = regime_config(pre["fine_tune_config"], regime)
            evaluator = {k: v for k, v in fit.items() if k != "optimizer"}
            evaluator["seed"] = seed
            candidate = {"model": model, "target_feature_indices": [0], "evaluator": evaluator}
            receipt = evaluate_candidate(candidate, out / "train.npz", out / "validation.npz",
                                         out / f"forecast_{regime}", progress=beat.progress(f"forecast:{regime}"))
            forecasts[regime] = receipt
            beat.update(last_checkpoint={"stage": f"forecast:{regime}",
                                         "model_sha256": receipt["digests"]["model_sha256"]})
    summary = summarize(pre, forecasts)
    summary.update(label="SYNTHETIC FIXTURE, LOCAL COMPONENT RUN - not a forecasting result",
                   total_wall_seconds=time.monotonic() - started,
                   heartbeat={"path": str(beat.path), "beats": beat.beats, "interval_seconds": beat.interval},
                   data={"train_npz_sha256": _file_sha(out / "train.npz"),
                         "validation_npz_sha256": _file_sha(out / "validation.npz"),
                         "ae_train_rows": int(len(ae_tr)), "ae_internal_validation_rows": int(len(ae_va)),
                         "forecast_train_rows": int(len(train["windows"])),
                         "forecast_validation_rows": int(len(validation["windows"])),
                         "test_rows_generated": 0})
    (out / "PILOT.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def _fit_brief(t):
    return {k: t[k] for k in ("observed_updates", "selected_updates", "selected_epoch", "epochs_completed",
                              "monitor_evaluations", "stop_reason", "stop_class", "best_validation_loss",
                              "restored_differs_from_last", "elapsed_seconds")} | {
        "early_stop_sha256": t["early_stop"]["sha256"]}


def summarize(pre, forecasts=None):
    rows = []
    for b in pre["branches"]:
        rows.append({"stage": "branch_ae", "component": b["name"], **_fit_brief(b["training"]),
                     **{f"{s}_{m}": b["reconstruction"][s][m] for s in ("train", "train_validation")
                        for m in ("MSE", "reference_MSE", "relative_MSE")},
                     "reload_max_abs_error": b["reload_parity"]["max_abs_error"]})
    c = pre["core"]
    rows.append({"stage": "core_ae", "component": "temporal_core", **_fit_brief(c["training"]),
                 **{f"{s}_{m}": c["reconstruction"][s][m] for s in ("train", "train_validation")
                    for m in ("MSE", "reference_MSE", "relative_MSE", "standardized_MSE")},
                 "reload_max_abs_error": c["reload_parity"]["max_abs_error"]})
    down = {}
    for regime, r in (forecasts or {}).items():
        down[regime] = {"validation_MAE": r["metrics"]["MAE"], "persistence_MAE": r["metrics"]["baseline_MAE"],
                        "skill_MAE": r["metrics"]["skill_MAE"], "validation_MSE": r["metrics"]["MSE"],
                        "per_horizon_MAE": {h: v["MAE"] for h, v in r["per_horizon"].items()},
                        "initial_weights_sha256": r["digests"]["initial_weights_sha256"],
                        **_fit_brief(r["training"])}
    return {"stages": rows, "downstream": down, "grids": pre["grids"],
            "core_upstream_branch_weights": pre["core"]["upstream_bound"]["branch_weights_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixture", choices=["synthetic"], default="synthetic",
                        help="only a declared synthetic fixture until M03 declares admissible inputs")
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--monitor-every", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--heartbeat", default=None)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    a = parser.parse_args(argv)
    fit = dict(max_epochs=a.max_epochs, patience=a.patience, min_delta=a.min_delta,
               monitor_every=a.monitor_every, max_updates=a.max_updates, max_seconds=a.max_seconds,
               batch_size=a.batch_size, learning_rate=a.learning_rate, loss=a.loss)
    summary = run_synthetic_pilot(a.out, rows=a.rows, features=a.features, seed=a.seed, fit=fit,
                                  heartbeat_path=a.heartbeat, heartbeat_interval=a.heartbeat_interval)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
