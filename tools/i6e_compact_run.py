"""TRAIN-only cost pilot for origin-anchored compact forecasting.

Runs one 9-token model with the retained 20-feature manifest and four calendar
years preceding 2024. No outer VALIDATION or TEST path is accepted. Atomic
status and checkpoint artifacts permit inspection without log-monitoring agents.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def partition_origins(ts, targets, minimum, cutoff, inner_hours=4 * 7 * 24):
    """Purge each 72h label from the next partition and retain 144h input support."""
    ts, y = np.asarray(ts), np.asarray(targets)
    inner_start = cutoff - inner_hours * 3600
    valid = np.isfinite(y) & np.isin(ts + 72 * 3600, ts) & (ts >= minimum + 144 * 3600)
    fit = np.flatnonzero(valid & (ts + 72 * 3600 < inner_start))
    inner = np.flatnonzero(valid & (ts >= inner_start) & (ts + 72 * 3600 < cutoff))
    if not len(fit) or not len(inner):
        raise ValueError("empty fit or purged inner population")
    return fit, inner


def scaled_inputs(values, fit_indices):
    """Fit source-missing imputation/scaling on fit rows, retaining availability."""
    from tools.fs4_temporal_predictor import Standardiser
    scaler = Standardiser.fit(values[fit_indices])
    result = np.empty((len(values), values.shape[1] * 2), dtype=np.float32)
    result[:, 0::2] = scaler.apply(values)
    result[:, 1::2] = np.isfinite(values)
    return result, scaler


def atomic_json(path, value):
    """Replace one complete JSON status, never expose a partly written record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    """Stream an artifact identity with bounded memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(args):
    """Fit a real compact model on complete TRAIN, with observed GPU execution."""
    from tools.fs4_weekly_wrapper import DataStore
    from tools.i6e_clock_windows import clock_windows
    from tools.i6e_masked_model import build_model
    from tools.modular_candidate_evaluator import fit_with_early_stopping
    import tensorflow as tf

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    terminal = root / "PILOT.json"
    if terminal.exists():
        raise ValueError("pilot terminal already exists; do not repeat completed fitting")
    claim = root / "PILOT.claim"
    with claim.open("x") as stream:
        stream.write(str(args.seed))
    start = time.monotonic()
    state = dict(stage="I6E_TRAIN_COST", status="PREPARING", seed=args.seed,
                 horizon_hours=72, spacing_hours=18, samples=9,
                 test_read=False, outer_validation_read=False)
    atomic_json(root / "STATUS.json", state)
    try:
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise ValueError("one visible GPU required; CPU fallback prohibited")
        tf.config.experimental.set_memory_growth(devices[0], True)
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.seed)
        with tf.device("/GPU:0"):
            probe = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        if "GPU:0" not in probe.device:
            raise ValueError("GPU operation did not execute on GPU")
        parent = json.loads(args.parent_design.read_text())
        from tools.i7_weekly_branch_regimes import _verify_seal
        _verify_seal(parent, "design_sha256")
        store = DataStore.from_train_only("EURUSD", args.feature_parquet,
                                         args.target_parquet, bar_hours=1)
        minimum = int(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp())
        cutoff = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp())
        y = store.targets["Y_l_72h"].copy()
        y[store.targets["Y_l_72h_staleness_h"] != 0] = np.nan
        fi, vi = partition_origins(store.ts, y, minimum, cutoff)
        members = list(parent["members"])
        if len(members) != 20 or len(set(members)) != 20:
            raise ValueError("retained 20-feature manifest required")
        values = store.X[:, [store.col[name] for name in members]]
        scaled, scaler = scaled_inputs(values, fi)
        xf, mf = clock_windows(store.ts, scaled, fi, 18, 9)
        xv, mv = clock_windows(store.ts, scaled, vi, 18, 9)
        xf = np.concatenate([xf, mf[:, :, None]], axis=-1).astype(np.float32)
        xv = np.concatenate([xv, mv[:, :, None]], axis=-1).astype(np.float32)
        center = float(np.median(y[fi]))
        scale = float(1.4826 * np.median(np.abs(y[fi] - center)))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("degenerate fit target scale")
        yf = ((y[fi] - center) / scale).astype(np.float32)[:, None]
        yv = ((y[vi] - center) / scale).astype(np.float32)[:, None]
        model = build_model(9, len(members), 18)
        state.update(status="RUNNING", fit_rows=len(fi), inner_rows=len(vi),
                     parameters=model.count_params(), device=probe.device,
                     fit_market_missing_slots=int((~mf).sum()),
                     inner_market_missing_slots=int((~mv).sum()))
        atomic_json(root / "STATUS.json", state)
        last = [0.]

        def progress(event):
            now = time.monotonic()
            if now - last[0] >= 15 or event["event"] != "update":
                last[0] = now
                atomic_json(root / "STATUS.json", dict(state, heartbeat_utc=dt.datetime.now(
                    dt.timezone.utc).isoformat(), progress=event))

        settings = dict(parent["config"]["fit"], seed=args.seed, max_epochs=50,
                        monitor="train_validation_mean", progress=progress)
        training = fit_with_early_stopping(model, xf, yf, xv, yv, settings)
        artifact = root / "PILOT.keras"
        model.save(artifact)
        prediction = np.concatenate([model(xv[i:i+256], training=False).numpy()
                                     for i in range(0, len(xv), 256)])[:, 0] * scale + center
        result = dict(state, status="COMPLETED", training=training,
                      input_digests=store.digests, members=members,
                      parent_design_sha256=parent["design_sha256"],
                      scaler_sha256=scaler.sha256(), target_center=center, target_scale=scale,
                      inner_mae=float(np.mean(np.abs(prediction-y[vi]))),
                      inner_naive_mae=float(np.mean(np.abs(y[vi]))),
                      inner_rows_sha256=hashlib.sha256(store.ts[vi].astype("<i8").tobytes()).hexdigest(),
                      model_sha256=file_hash(artifact), elapsed_seconds=time.monotonic()-start,
                      meaning="TRAIN-inner diagnostic and cost only, not annual validation quality")
        atomic_json(terminal, result)
        atomic_json(root / "STATUS.json", result)
    except Exception as exc:
        atomic_json(root / "STATUS.json", dict(state, status="FAILED", error=str(exc)))
        raise
    finally:
        claim.unlink(missing_ok=True)


def main():
    """Expose bounded execution and inexpensive status reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot", "status"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-design", type=Path)
    parser.add_argument("--feature-parquet", type=Path, action="append")
    parser.add_argument("--target-parquet", type=Path)
    parser.add_argument("--seed", type=int, default=6102026)
    args = parser.parse_args()
    if args.command == "status":
        print((args.output / "STATUS.json").read_text())
    else:
        if not all((args.parent_design, args.feature_parquet, args.target_parquet)):
            parser.error("pilot requires TRAIN features/targets and parent design")
        run(args)


if __name__ == "__main__":
    main()
