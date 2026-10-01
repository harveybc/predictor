"""Fit and export a LOCAL-SMOKE modular candidate for the LTS paper adapter (lane M05).

This produces the files that ``lts`` needs to exercise its modular
inference/feature/action adapter end to end on recorded inputs:

* ``inference_model.keras`` - one Keras model, two outputs: the forecast head
  ``(batch, horizons, targets)`` and the rank-three encoder bottleneck
  ``(batch, output_steps, output_channels)`` (MS15 export; the RL policy that
  would consume it is a separate owned integration).
* ``contract.json`` - ``lts.modular_inference_contract.v1``: engine identity,
  artifact identity, time/feature/scaler/action contract and evidence flags.
* ``metrics.json`` - validation-only metrics next to the zero and persistence
  naive forecasts on the same rows.  The test partition is never evaluated.
* ``golden.json`` - per as-of window: the scaled input tensor and the outputs
  computed HERE, so the LTS side proves parity instead of assuming it.
* ``candidate.json``/``provenance.json`` - the ``lts.paper_candidate_handoff.v1``
  shape read by LTS's candidate preflight.

It is not a promotion.  ``research_validated``, ``live_inference_eligible`` and
``live_execution_eligible`` are written false, and forecasting error says
nothing about trading profitability.  Run on CPU:

    CUDA_VISIBLE_DEVICES= TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
      python tools/export_modular_paper_candidate.py --bars <recorded.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "predictor_plugins" / "modular_temporal.py"

FEATURE_SCHEMA = "lts.modular_features.closed_bars.v1"
CONTRACT_SCHEMA = "lts.modular_inference_contract.v1"
FEATURES = [
    {"name": "log_return", "kind": "log_return", "params": {}},
    {"name": "range_fraction", "kind": "range_fraction", "params": {}},
    {"name": "log_volume_z20", "kind": "log_volume_z", "params": {"window": 20}},
]
VOLUME_WINDOW = 20


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_bars(path: Path) -> list[dict]:
    """Recorded LTS closed-bar CSV (DateTime, Open, High, Low, Close, Volume)."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["DateTime"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("bar timestamps must be timezone-aware")
            rows.append({"time": stamp.astimezone(timezone.utc),
                         **{k.lower(): float(row[k]) for k in ("Open", "High", "Low", "Close", "Volume")}})
    for a, b in zip(rows, rows[1:]):
        if b["time"] <= a["time"]:
            raise ValueError("bars must be strictly time ordered")
    return rows


def feature_matrix(bars: list[dict]) -> np.ndarray:
    """Unscaled features per bar; row i uses bars[..i] only (NaN until defined).

    The LTS adapter re-implements these definitions independently; golden.json
    is what proves the two implementations agree.
    """
    close = np.array([b["close"] for b in bars], dtype=np.float64)
    high = np.array([b["high"] for b in bars], dtype=np.float64)
    low = np.array([b["low"] for b in bars], dtype=np.float64)
    logv = np.log1p(np.array([b["volume"] for b in bars], dtype=np.float64))
    out = np.full((len(bars), len(FEATURES)), np.nan)
    out[1:, 0] = np.log(close[1:] / close[:-1])
    out[:, 1] = (high - low) / close
    for i in range(VOLUME_WINDOW - 1, len(bars)):
        block = logv[i - VOLUME_WINDOW + 1:i + 1]
        out[i, 2] = (logv[i] - block.mean()) / max(float(block.std()), 1e-12)
    return out


def windows(features: np.ndarray, window: int):
    """As-of indices whose whole window is defined, plus next-bar target."""
    first = max(1, VOLUME_WINDOW - 1) + window - 1
    idx = [i for i in range(first, len(features))]
    x = np.stack([features[i - window + 1:i + 1] for i in idx])
    return np.array(idx), x


def _keras_version():
    import keras

    return keras.__version__


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bars", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-id", default="modular-spy-1d-local-smoke-v1")
    ap.add_argument("--asset-id", default="equity:SPY")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--threshold", type=float, default=0.001,
                    help="action dead-band on the forecast next-session log return")
    ap.add_argument("--golden", type=int, default=12)
    ap.add_argument("--previous-contract", type=Path,
                    help="earlier export of the same candidate recipe; its engine tip, Keras"
                         " version and contract digest are recorded as lineage")
    ap.add_argument("--engine-source-commit",
                    help="commit of the engine tree when it is an exported archive, not a git checkout")
    ap.add_argument("--previous-keras-version",
                    help="Keras version the previous export ran under, when its contract predates"
                         " the keras_version field (observed from that environment, not guessed)")
    args = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("refusing: set CUDA_VISIBLE_DEVICES='' (CPU only)")

    import tensorflow as tf  # noqa: E402  (after the CPU guard)
    sys.path.insert(0, str(ROOT))
    from predictor_plugins import modular_temporal as mt

    tf.keras.utils.set_random_seed(args.seed)
    bars = read_bars(args.bars)
    feats = feature_matrix(bars)
    idx, x_raw = windows(feats, args.window)
    has_target = idx + 1 < len(bars)
    target = np.full(len(idx), np.nan)
    target[has_target] = feats[idx[has_target] + 1, 0]
    n = int(has_target.sum())
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    tr, va = slice(0, n_train), slice(n_train, n_train + n_val)
    # Train-only scaler: rows of the training windows' last step plus earlier steps
    # all lie at or before the last training as-of bar, so nothing leaks forward.
    last_train_row = int(idx[n_train - 1])
    train_rows = feats[max(1, VOLUME_WINDOW - 1):last_train_row + 1]
    mean, scale = train_rows.mean(axis=0), train_rows.std(axis=0)
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise SystemExit("degenerate training scale")
    t_scale = float(scale[0])
    x = ((x_raw - mean) / scale).astype(np.float32)
    y = (target / t_scale).astype(np.float32).reshape(-1, 1, 1)

    config = mt.default_config([f["name"] for f in FEATURES])
    config["sample_hours"] = 24
    config["window"] = args.window
    config = mt._normalize(config)
    bundle = mt.build_modular(config)
    model = bundle.forecast_model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                            min_delta=args.min_delta, restore_best_weights=True)
    hist = model.fit(x[tr], y[tr], validation_data=(x[va], y[va]), epochs=args.epochs,
                     batch_size=64, shuffle=True, verbose=0, callbacks=[stop])
    val_losses = [float(v) for v in hist.history["val_loss"]]
    best_epoch = int(np.argmin(val_losses)) + 1

    pred_val = model.predict(x[va], verbose=0).reshape(-1) * t_scale
    truth = target[va]
    persistence = x_raw[va][:, -1, 0]
    metrics = {
        "schema": "lts.modular_candidate_metrics.v1",
        "evidence_class": "local_smoke",
        "population": {"train_windows": n_train, "validation_windows": n_val,
                       "test_windows_untouched": n - n_train - n_val,
                       "validation_first_asof": bars[int(idx[va][0])]["time"].isoformat(),
                       "validation_last_asof": bars[int(idx[va][-1])]["time"].isoformat()},
        "validation": {
            "mae_model": float(np.mean(np.abs(pred_val - truth))),
            "mae_zero_naive": float(np.mean(np.abs(truth))),
            "mae_persistence_naive": float(np.mean(np.abs(persistence - truth))),
        },
        "fit": {"epochs_run": len(val_losses), "best_epoch": best_epoch,
                "patience": args.patience, "min_delta": args.min_delta,
                "restore_best_weights": True, "loss": "mae", "optimizer": "adam(1e-3)"},
        "note": "Forecast error on one validation block; not a trading or profitability result.",
    }
    metrics["validation"]["skill_vs_zero_naive"] = 1.0 - (
        metrics["validation"]["mae_model"] / metrics["validation"]["mae_zero_naive"])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    inference = tf.keras.Model(model.inputs, [model.outputs[0], bundle.encoder_model.outputs[0]],
                               name="modular_inference")
    model_path = out / "inference_model.keras"
    inference.save(model_path)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    # Golden windows: the most recent recorded as-of points (validation/test region
    # included - outputs only, never used to select anything).
    reloaded = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    golden_rows = []
    for j in range(len(idx) - args.golden, len(idx)):
        fc, bott = reloaded.predict(x[j:j + 1], verbose=0)
        golden_rows.append({
            "last_closed_bar": bars[int(idx[j])]["time"].isoformat(),
            "window_scaled": x[j].astype(np.float64).tolist(),
            "forecast": fc.astype(np.float64).reshape(-1).tolist(),
            "bottleneck": bott.astype(np.float64).reshape(bott.shape[1:]).tolist(),
        })
    golden = {"schema": "lts.modular_golden_parity.v1", "bars_sha256": _sha(args.bars),
              "producer": "predictor tools/export_modular_paper_candidate.py", "rows": golden_rows}
    (out / "golden.json").write_text(json.dumps(golden, sort_keys=True) + "\n")

    contract = {
        "schema": CONTRACT_SCHEMA, "contract_version": 1,
        "model_id": args.model_id, "asset_id": args.asset_id, "timeframe": args.timeframe,
        "engine": {"module": "predictor_plugins/modular_temporal.py", "path": str(ENGINE_PATH),
                   "sha256": _sha(ENGINE_PATH), "source_commit": args.engine_source_commit or _git("rev-parse", "HEAD"),
                   "tensorflow": tf.__version__, "keras_version": _keras_version()},
        "artifact": {"file": model_path.name, "sha256": _sha(model_path),
                     "weights_sha256": mt.weights_hash(inference),
                     "outputs": ["forecast", "bottleneck"]},
        "modular_config": bundle.config,
        "time": {
            "grid": "venue_session_close", "sample_hours": 24, "window": args.window,
            "bar_label": "session_start", "bar_close_offset_hours": 16, "max_gap_hours": 120,
            "note": "Equity sessions are not a uniform 24h clock: sample_hours is the nominal"
                    " engine period and max_gap_hours bounds weekend/holiday gaps. A bar labelled"
                    " at the venue-local session start is complete bar_close_offset_hours later.",
            "streams": [{"name": "bars", "frequency": args.timeframe,
                         "fields": ["open", "high", "low", "close", "volume"],
                         "alignment": "native_grid"}],
        },
        "features": {"schema": FEATURE_SCHEMA, "definitions": FEATURES,
                     "names": [f["name"] for f in FEATURES],
                     "history_bars_required": max(1, VOLUME_WINDOW - 1) + args.window,
                     "scaler": {"fitted_on": "train_partition_only",
                                "last_train_asof": bars[last_train_row]["time"].isoformat(),
                                "mean": mean.tolist(), "scale": scale.tolist()}},
        "outputs": {"forecast": {"horizons": [1], "target": "log_return_next_session",
                                 "scaled_by": t_scale},
                    "bottleneck": {"shape": [config["output_steps"], config["output_channels"]],
                                   "rank": 3, "consumer": "separate_owned_policy_adapter"}},
        "action": {"schema": "lts.modular_action.v1", "rule": "forecast_threshold",
                   "horizon_index": 0, "target_index": 0, "long_above": args.threshold,
                   "short_below": -args.threshold, "otherwise": "hold",
                   "units": "log_return",
                   "note": "Dead-band mapping for shadow inference; not tuned for profit."},
        "evidence": {"class": "local_smoke", "metrics_file": "metrics.json",
                     "metrics_sha256": _sha(out / "metrics.json"),
                     "golden_file": "golden.json", "golden_sha256": _sha(out / "golden.json"),
                     "research_validated": False, "live_inference_eligible": False,
                     "live_execution_eligible": False},
        "bars_source": {"sha256": _sha(args.bars), "rows": len(bars),
                        "first": bars[0]["time"].isoformat(), "last": bars[-1]["time"].isoformat()},
    }
    if args.previous_contract is not None:
        previous = json.loads(args.previous_contract.read_text())
        contract["engine"]["previous"] = {
            "source_commit": previous["engine"].get("source_commit"),
            "sha256": previous["engine"].get("sha256"),
            "tensorflow": previous["engine"].get("tensorflow"),
            "keras_version": previous["engine"].get("keras_version")
            or args.previous_keras_version or "unrecorded",
            "contract_sha256": _sha(args.previous_contract),
            "artifact_sha256": previous["artifact"]["sha256"],
        }
    (out / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    provenance = {"schema": "lts.candidate_provenance.v1", "status": "verified",
                  "scope": "local_hash_binding_only", "model_id": args.model_id,
                  "weights_sha256": _sha(model_path), "metrics_sha256": _sha(out / "metrics.json")}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    candidate = {"schema": "lts.paper_candidate_handoff.v1", "model_id": args.model_id,
                 "family": "modular", "asset_id": args.asset_id, "timeframe": args.timeframe}
    for key, name in (("weights", model_path.name), ("metrics", "metrics.json"),
                      ("provenance", "provenance.json"), ("inference_contract", "contract.json")):
        path = (out / name).resolve()
        candidate[key] = {"path": str(path), "sha256": _sha(path)}
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "model_id": args.model_id, "metrics": metrics["validation"],
                      "fit": metrics["fit"], "contract_sha256": _sha(out / "contract.json")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
