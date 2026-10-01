"""PS3-R pilot runner (approved design bf90f1f0 section 3). One shard per crispdm-run child.

    python tools/ps3r_pilot_run.py --data <ethusdt_4h_tech_stat_full_model_ready.csv>
        --selection docs/audits/evidence/lane_a_ps3r_20261001/PILOT_SELECTION_397b67d6.json
        --out <state dir> --shard 0 --shards 2

* data: the immutable ETH 4h model-ready view (sha256 1b447c66...); TRAIN = rows before 2024-01-01
  (13 699 rows); nothing of the outer validation (2024) or the protected test (2025) is read.
* targets: log CLOSE(t+h)/CLOSE(t), the label row located by ELAPSED SECONDS (explicit unit), over TRAIN
  rows, as lane B's corrected construction (feature-eng 1b22c64, target_status a8cb02f9); the missing-label
  counts are asserted equal to a8cb02f9 before anything runs. Y_b is NOT_EVALUATED (no versioned rule).
* folds: lane B's three inner folds (purge 60 rows). Encoder fits on the first 85 % of the fold's train
  windows and early-stops on the rest after a 60-row purge; the per-feature scaler is fitted on the
  encoder-train rows only. Probes fit on the fold's train windows whose label lies inside the fold train,
  and evaluate on fold-validation windows whose label lies inside the fold validation.
* arms per (input, fold, seed): ``ae`` (autoencoder_reconstruction) and ``contrastive``
  (ts2vec_contrastive), both starting from the SAME seeded initial weights; the random arm of every probe
  is that exact initial state (paired); the raw arm is the standardized 24-bar window.
* resumable: one atomic record per (input, fold, arm, seed) under ``records/``; an existing valid record is
  skipped. A heartbeat file is rewritten every <= 15 s from a side thread.
Every number produced here is PILOT_ENGINEERING until tools/ps3r_pilot_contrast.py generates the paired table.
"""
import argparse
import hashlib
import json
import os
import resource
import threading
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

DATA_SHA256 = "1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f"
SELECTION_SHA = "397b67d6f40eaf324c0aa720291f67560c55cf0b216bc53cb9c4886b1ae66175"
TRAIN_ROWS = 13699
FOLDS = [{"name": "inner_1", "train": (0, 7474), "val": (7534, 9589)},
         {"name": "inner_2", "train": (0, 9529), "val": (9589, 11644)},
         {"name": "inner_3", "train": (0, 11584), "val": (11644, 13699)}]
PURGE, WINDOW, STEP_SECONDS = 60, 24, 14400
TARGETS = [("Y_s", 4)] + [("Y_l", h) for h in (24, 48, 72, 96, 120, 144)]
EXPECTED_MISSING = {4: 9, 24: 21, 48: 28, 72: 34, 96: 40, 120: 46, 144: 52}     # a8cb02f9
SEEDS = (2021, 2022)
FIT = {"max_epochs": 30, "patience": 5, "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
       "min_delta": 0.0, "max_updates": 5000}
ARMS = {"ae": {"plugin": "autoencoder_reconstruction"}, "contrastive": {"plugin": "ts2vec_contrastive"}}
TS2VEC_DEVIATIONS = [
    "reference zhihanyue/ts2vec @ b0088e14a99706c05451316dc6db8d3da9351163 (MIT); Yue et al., AAAI 2022",
    "crop applied by zeroing history before the crop start on the fixed 24-step input; loss on the shared suffix",
    "timestamp masking zeroes INPUT steps (no exposed hidden projection in the external branch component)",
    "encoder is the approved causal Conv1D branch 2.0.0, not TS2Vec's dilated CNN; objective is the axis under test",
]


def memory():
    """Own RSS peak and the enclosing cgroup's current/peak bytes (cgroup v2), read now."""
    out = {"rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        base = Path("/sys/fs/cgroup") / rel.lstrip("/")
        for name in ("memory.current", "memory.peak"):
            f = base / name
            if f.is_file():
                out["cgroup_" + name.split(".")[1] + "_bytes"] = int(f.read_text().split()[0])
        out["cgroup"] = rel
    except (OSError, IndexError, ValueError):
        out["cgroup"] = "UNAVAILABLE"
    return out


def _atomic(path, document):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load(data_path):
    import pandas as pd
    raw = Path(data_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise SystemExit("data bytes are not the pinned ETH 4h model-ready view")
    frame = pd.read_csv(data_path)
    stamps = pd.to_datetime(frame["DATE_TIME"])
    train = frame[stamps < pd.Timestamp("2024-01-01")].reset_index(drop=True)
    if len(train) != TRAIN_ROWS:
        raise SystemExit(f"TRAIN has {len(train)} rows, expected {TRAIN_ROWS}")
    ts = ((pd.to_datetime(train["DATE_TIME"]) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy("int64")
    if int(np.median(np.diff(ts))) != STEP_SECONDS:
        raise SystemExit("timestamp step is not 14400 s: unit error")
    return train, ts


def targets(train, ts):
    price = train["CLOSE"].to_numpy("float64")
    out = {}
    for name, h in TARGETS:
        j = np.searchsorted(ts, ts + h * 3600)
        ok = j < len(ts)
        ok[ok] = ts[j[ok]] == ts[ok] + h * 3600
        if int((~ok).sum()) != EXPECTED_MISSING[h]:
            raise SystemExit(f"{name}@{h}h missing {int((~ok).sum())} != a8cb02f9's {EXPECTED_MISSING[h]}")
        values, label = np.full(len(ts), np.nan), np.full(len(ts), -1, dtype=np.int64)
        values[ok] = np.log(price[j[ok]] / price[ok])
        label[ok] = j[ok]
        out[f"{name}@{h}h"] = (values, label)
    return out


def fold_arrays(series, fold, labels):
    te = fold["train"][1]
    vs, ve = fold["val"]
    train_origins = np.arange(WINDOW - 1, te)
    cut = train_origins[int(0.85 * len(train_origins))]
    enc_train = train_origins[train_origins < cut]
    enc_val = train_origins[train_origins >= cut + PURGE]
    mu, sd = float(series[:cut].mean()), float(series[:cut].std())       # encoder-train rows only
    sd = sd if sd > 1e-12 else 1.0
    z = ((series - mu) / sd).astype("float32")

    def win(origins):
        return np.stack([z[t - WINDOW + 1:t + 1] for t in origins])[..., None]
    val_origins = np.arange(vs, ve)
    probe_origins = np.concatenate([train_origins, val_origins])
    y = {}
    for name, (values, label) in labels.items():
        v = values[probe_origins].copy()
        lab = label[probe_origins]
        in_train = probe_origins < te
        v[(in_train & (lab >= te)) | (~in_train & (lab >= ve)) | (lab < 0)] = np.nan
        y[name] = v
    return {"x_enc": win(enc_train), "x_enc_val": win(enc_val), "x_probe": win(probe_origins), "y": y,
            "fit_rows": np.arange(len(train_origins)),
            "eval_rows": np.arange(len(train_origins), len(probe_origins)),
            "ranges": {"encoder_train_origins": [int(enc_train[0]), int(enc_train[-1])],
                       "encoder_val_origins": [int(enc_val[0]), int(enc_val[-1])],
                       "probe_fit_origins": [int(train_origins[0]), int(train_origins[-1])],
                       "probe_eval_origins": [int(val_origins[0]), int(val_origins[-1])],
                       "scaler_rows": [0, int(cut)], "scaler": [mu, sd]}}


class Heartbeat(threading.Thread):
    def __init__(self, path, every):
        super().__init__(daemon=True)
        self.path, self.every, self.state, self.stop = path, every, {}, threading.Event()

    def run(self):
        while not self.stop.is_set():
            _atomic(self.path, {**self.state, "utc_epoch": time.time(), "memory_now": memory()})
            self.stop.wait(self.every)


def _tracing_estimate(receipt):
    epochs = [e for e in receipt.get("history", []) if e.get("seconds")]
    if len(epochs) < 2:
        return None
    steady = float(np.median([e["seconds"] / max(e["updates"], 1) for e in epochs[1:]]))
    return max(0.0, epochs[0]["seconds"] - steady * epochs[0]["updates"])


def _objective_shas(queue):
    from predictor_plugins.modular_temporal import objectives as ob
    queue.put({arm: ob.objective_identity(spec)["sha256"] for arm, spec in ARMS.items()})


def _reusable(path, expected_sha):
    """Reuse a completed record only if it binds the same data bytes and the same objective identity."""
    try:
        record = json.loads(path.read_text())
    except ValueError:
        return False
    return (record.get("status") == "PILOT_ENGINEERING"
            and (record.get("input_identity") or {}).get("data_sha256") == DATA_SHA256
            and (record.get("objective") or {}).get("sha256") == expected_sha)


def _child(data_path, feature, fold_name, seed, arm, records_dir):
    """One record in its own process (FINDING PS3R-MEM-01: memory grew per fit inside one process)."""
    train, ts = load(data_path)
    labels = targets(train, ts)
    fold = next(f for f in FOLDS if f["name"] == fold_name)
    data = fold_arrays(train[feature].to_numpy("float64"), fold, labels)
    records = Path(records_dir)
    try:
        run_one(feature, fold, seed, arm, data, records)
    except ValueError as exc:                         # nonfinite loss etc.: recorded, not hidden
        _atomic(records / f"{feature}__{fold_name}__{arm}__{seed}.json",
                {"schema": "ps3r.pilot.record.v1", "status": "FIT_FAILED", "input": feature, "fold": fold_name,
                 "seed": seed, "arm": arm, "reason": str(exc)[:500], "memory_at_record": memory()})


def run_one(feature, fold, seed, arm, data, records):
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt
    from predictor_plugins.modular_temporal import objectives as ob
    from predictor_plugins.modular_temporal import probes as pb
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    cfg = mt.default_config([feature])
    cfg["sample_hours"] = 4
    bundle = mt.build_modular(cfg)
    enc = bundle.branch_models["branch_0"]
    tf.keras.utils.set_random_seed(seed)
    init_twin = tf.keras.models.clone_model(enc)                       # same seeded initial state ...
    init_twin.set_weights(enc.get_weights())                            # ... exactly (the paired random arm)
    init_hash = mt.weights_hash(enc)
    started = time.monotonic()
    receipt = ob.fit_objective(ARMS[arm], enc, data["x_enc"], data["x_enc_val"], dict(FIT, seed=seed))
    fit_seconds = time.monotonic() - started
    rows = pb.probe_battery(trained=enc, random=init_twin, x=data["x_probe"], targets=data["y"],
                            fit_rows=data["fit_rows"], eval_rows=data["eval_rows"], fold=fold["name"], seed=seed)
    for r in rows:
        name = f"{r['target']}@{r['horizon']}h"
        finite = np.isfinite(data["y"][name])
        eval_idx = data["eval_rows"][finite[data["eval_rows"]]]
        r["eval_rows_sha256"] = hashlib.sha256(eval_idx.astype("int64").tobytes()).hexdigest()
    diag_x = data["x_probe"][data["eval_rows"]]
    record = {"schema": "ps3r.pilot.record.v1", "status": "PILOT_ENGINEERING", "input": feature,
              "input_identity": {"dataset": "financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1",
                                 "data_sha256": DATA_SHA256, "column": feature,
                                 "transform_recipe": "RAW_COLUMN_AS_PUBLISHED", "recipe_id": None,
                                 "note": "lane B recipe ids pending (source-transform addendum 256c61a6); "
                                         "fold-local z-score on encoder-train rows applied here"},
              "fold": fold["name"], "seed": seed, "arm": arm, "objective": receipt["identity"],
              "declared_deviations": TS2VEC_DEVIATIONS if arm == "contrastive" else [],
              "architecture": {"component": "causal_conv1d", "version": mt.describe_component(
                  "branch", "causal_conv1d")["version"], "latent_shape": list(enc.output_shape[1:]),
                  "init_weights_sha256": init_hash, "trained_weights_sha256": mt.weights_hash(enc)},
              "reconstruction": receipt["reconstruction"], "ranges": data["ranges"],
              "fit": {"observed_updates": receipt["observed_updates"],
                      "epochs_completed": receipt.get("epochs_completed"), "stop_reason": receipt.get("stop_reason"),
                      "best_validation_loss": receipt.get("best_validation_loss"),
                      "initial_validation_loss": receipt.get("initial_validation_loss"),
                      "fit_seconds": fit_seconds,
                      "tracing_seconds_estimate": _tracing_estimate(receipt) if arm == "contrastive" else None,
                      "epoch_history": receipt.get("history")},
              "probes": rows,
              "latent_diagnostics": {"trained": pb.latent_diagnostics(enc, diag_x),
                                     "random": pb.latent_diagnostics(init_twin, diag_x)},
              "cost": {"seconds_total": time.monotonic() - started, "memory_at_record": memory(),
                       "device_class": "cpu"}}
    _atomic(records / f"{feature}__{fold['name']}__{arm}__{seed}.json", record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--heartbeat-seconds", type=float, default=15.0)
    p.add_argument("--cap-bytes", type=int, required=True,
                   help="the crispdm-run cap of this child; it stops itself at a record boundary at 90 %%")
    a = p.parse_args()
    selection = json.loads(Path(a.selection).read_text())
    if selection.get("worklist_sha256") != SELECTION_SHA or selection.get("status") != "CURRENT":
        raise SystemExit("selection is not the approved CURRENT draw on 397b67d6")
    features = [s["feature"] for s in selection["sample"]][a.shard::a.shards]
    out = Path(a.out)
    records = out / "records"
    records.mkdir(parents=True, exist_ok=True)
    beat = Heartbeat(out / f"heartbeat_shard{a.shard}.json", a.heartbeat_seconds)
    beat.state = {"stage": "loading", "shard": a.shard, "features": features, "done": 0}
    beat.start()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    helper = ctx.Process(target=_objective_shas, args=(queue,))
    helper.start()
    shas = queue.get(timeout=600)
    helper.join()
    todo = [(f, fold, seed, arm) for f in features for fold in FOLDS for seed in SEEDS for arm in ARMS]
    done = skipped = 0
    peaks, seconds_by_arm = [], {}
    for feature, fold, seed, arm in todo:
        if (out / "STOP").exists():                  # planned stop at a record boundary
            beat.state = {"stage": "stopped_planned", "shard": a.shard, "done": done, "skipped": skipped,
                          "reason": (out / "STOP").read_text()[:300]}
            time.sleep(a.heartbeat_seconds + 1)
            beat.stop.set()
            print(json.dumps({"stopped_planned": True, "done": done, "skipped": skipped}), flush=True)
            raise SystemExit(0)
        path = records / f"{feature}__{fold['name']}__{arm}__{seed}.json"
        if path.is_file():
            if _reusable(path, shas[arm]):
                skipped += 1
                continue
            path.rename(path.with_name(path.name + f".not_reused.{int(time.time())}"))
        remaining = len(todo) - done - skipped
        per_arm = {k: (sum(v) / len(v)) for k, v in seconds_by_arm.items() if v}
        if len(per_arm) == len(ARMS):
            left = {k: sum(1 for (_, _, _, ar) in todo[todo.index((feature, fold, seed, arm)):] if ar == k)
                    for k in ARMS}
            eta = {"state": "MEASURED_RATE", "seconds": sum(per_arm[k] * left[k] for k in ARMS),
                   "basis": {k: {"mean_seconds_per_record": per_arm[k], "records_measured": len(seconds_by_arm[k]),
                                 "records_left_upper_bound": left[k]} for k in ARMS},
                   "note": "upper bound: records found reusable later are skipped"}
        else:
            eta = {"state": "NOT_MEASURED", "reason": "no completed record of every arm in this run yet"}
        beat.state = {"stage": "fitting", "shard": a.shard, "input": feature, "fold": fold["name"],
                      "seed": seed, "arm": arm, "done": done, "skipped": skipped, "total": len(todo),
                      "remaining_upper_bound": remaining, "eta": eta, "label": "PILOT_ENGINEERING"}
        record_started = time.monotonic()
        child = ctx.Process(target=_child, args=(a.data, feature, fold["name"], seed, arm, str(records)))
        child.start()
        child.join()
        if child.exitcode != 0 or not path.is_file():
            beat.state = {"stage": "child_failed", "exitcode": child.exitcode, "input": feature,
                          "fold": fold["name"], "seed": seed, "arm": arm, "memory": memory()}
            time.sleep(a.heartbeat_seconds + 1)
            beat.stop.set()
            raise SystemExit(f"record child failed with exit code {child.exitcode}; stopping at this boundary")
        record = json.loads(path.read_text())
        seconds_by_arm.setdefault(arm, []).append(time.monotonic() - record_started)
        done += 1
        mem = memory()
        child_mem = (record.get("cost") or {}).get("memory_at_record") or record.get("memory_at_record") or {}
        peaks.append({"input": feature, "fold": fold["name"], "seed": seed, "arm": arm,
                      "child_rss_peak_bytes": child_mem.get("rss_peak_bytes"), **mem})
        _atomic(out / f"memory_peaks_shard{a.shard}.json", {"cap_bytes": a.cap_bytes, "records": peaks})
        if max(mem.get("cgroup_peak_bytes", 0), child_mem.get("rss_peak_bytes") or 0) >= 0.9 * a.cap_bytes:
            beat.state = {"stage": "stopped_near_cap", "shard": a.shard, "done": done, "memory": mem,
                          "cap_bytes": a.cap_bytes}
            time.sleep(a.heartbeat_seconds + 1)
            beat.stop.set()
            print(json.dumps({"stopped_near_cap": mem, "cap_bytes": a.cap_bytes}), flush=True)
            raise SystemExit(3)
        fit = record.get("fit") or {}
        print(json.dumps({"done": done, "input": feature, "fold": fold["name"], "seed": seed, "arm": arm,
                          "status": record["status"], "updates": fit.get("observed_updates"),
                          "stop": fit.get("stop_reason"),
                          "child_rss_gb": round((child_mem.get("rss_peak_bytes") or 0) / 2**30, 3),
                          "scope_peak_gb": round(mem.get("cgroup_peak_bytes", 0) / 2**30, 3)}), flush=True)
    beat.state = {"stage": "complete", "shard": a.shard, "done": done, "skipped": skipped, "total": len(todo)}
    time.sleep(a.heartbeat_seconds + 1)
    beat.stop.set()


if __name__ == "__main__":
    main()
