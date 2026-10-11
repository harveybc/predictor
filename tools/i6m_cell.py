"""One bounded fit/prediction, or TRAIN-only cost fit reused by its exact cell.

Inputs and identities are authenticated before TensorFlow import. The pilot
reads TRAIN.npz only; its selected weights are reused without another fit.
All temporal support and scaler fitting are local to the weekly contract.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from tools.i6m_data import epoch, file_hash, load_input, metrics, partitions, price_prediction, targets
from tools.i6m_queue import atomic_json, digest, seal, verify


def peak_bytes():
    """Read the child's own cgroup; RSS is not substituted for its peak."""
    try:
        relative = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
        return int((Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.peak").read_text())
    except (OSError, StopIteration, ValueError):
        return None


def windows(ts, scaled, indices, arm):
    from tools.i6e_clock_windows import clock_windows
    value, present = clock_windows(ts, scaled, indices, arm["spacing_hours"], arm["samples"])
    return np.concatenate([value, present[:, :, None]], axis=-1).astype(np.float32)


def neural(args, job, design, ts, x, price, y, endpoint, fi, vi, si):
    import tensorflow as tf
    from tools.i6m_model import build_model
    from tools.i6e_compact_run import scaled_inputs
    from tools.modular_candidate_evaluator import fit_with_early_stopping, _weight_digest

    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise ValueError("exactly one visible GPU required; CPU fallback prohibited")
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(job["seed"])
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    if "GPU:0" not in probe.device:
        raise ValueError("GPU execution probe failed")
    scaled, scaler = scaled_inputs(x, fi)
    center = float(np.median(y[fi]))
    scale = float(1.4826*np.median(np.abs(y[fi]-center)))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("degenerate target scale")
    settings = design["fit"]
    fit_contract = dict(arm=job["arm"], week=job["week"], seed=job["seed"],
                               fit_ts=digest(ts[fi].tolist()), inner_ts=digest(ts[vi].tolist()),
                               scaler=scaler.sha256(), center=center, scale=scale,
                               settings=settings, code_sha256=design["code_sha256"],
                               train_sha256=design["train_sha256"], members=design["members"])
    fit_identity = digest(fit_contract)
    fit_path = args.output / "FIT.json"
    model = build_model(job["arm"]["samples"], 20, job["arm"]["spacing_hours"])
    if model.count_params() != 34921:
        raise ValueError("model parameter contract changed")
    reused = False
    if fit_path.exists():
        fit = verify(json.loads(fit_path.read_text()))
        allowed = [fit_identity]
        if design.get("receipt_only_predecessor_code_sha256"):
            allowed.append(digest(dict(fit_contract, code_sha256=design["receipt_only_predecessor_code_sha256"])))
        if fit["fit_identity"] not in allowed or file_hash(args.output/"MODEL.keras") != fit["model_sha256"]:
            raise ValueError("retained pilot/checkpoint identity mismatch")
        model = tf.keras.models.load_model(args.output/"MODEL.keras")
        training = fit["training"]
        if _weight_digest(model.get_weights()) != training["restored_weights_sha256"]:
            raise ValueError("selected weights mismatch")
        reused = True
    else:
        xf, xv = windows(ts, scaled, fi, job["arm"]), windows(ts, scaled, vi, job["arm"])
        last = [0.]

        def progress(event):
            if time.time()-last[0] >= 15 or event["event"] != "update":
                last[0] = time.time()
                atomic_json(args.output/"STATUS.json", dict(job_id=job["id"], lease=job["lease"],
                            status="FITTING_TRAIN_ONLY" if args.pilot else "FITTING", heartbeat=time.time(), progress=event))

        training = fit_with_early_stopping(model, xf, ((y[fi]-center)/scale).astype(np.float32)[:, None],
                                          xv, ((y[vi]-center)/scale).astype(np.float32)[:, None],
                                          dict(settings, seed=job["seed"], progress=progress))
        model.save(args.output/"MODEL.keras")
        fit = seal(dict(fit_identity=fit_identity, training=training, model_sha256=file_hash(args.output/"MODEL.keras"),
                        fit_rows=len(fi), inner_rows=len(vi), cgroup_peak_bytes=peak_bytes(),
                        parameters=model.count_params(), device=probe.device, test_read=False,
                        outer_validation_read=not args.pilot))
        atomic_json(fit_path, fit)
        del xf, xv
    if args.pilot:
        return dict(status="PILOT_COMPLETED", **fit)
    xs = windows(ts, scaled, si, job["arm"])
    z = np.concatenate([model(xs[i:i+256], training=False).numpy() for i in range(0, len(xs), 256)])[:, 0]
    prediction = price_prediction(price[si], z, center, scale)
    return dict(prediction=prediction, fit_identity=fit_identity, training=training,
                fit_rows=len(fi), inner_rows=len(vi), parameters=model.count_params(), device=probe.device,
                scaler_sha256=scaler.sha256(), target_center=center, target_scale=scale,
                model_sha256=file_hash(args.output/"MODEL.keras"), reused_fit=reused)


def statistical(args, job, design, ts, price, si):
    """ARIMA(1,1,0) per origin-clock phase; one weekly fit per phase, no drift.

    The Kalman filter (not smoother) conditions on observed values through each
    origin. Dynamic prediction ignores all later observations. Calendar holes
    remain NaN on the physical clock, not compressed away.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    retained = args.output/"PREDICTIONS.npz"
    if retained.exists() and design.get("receipt_only_predecessor_code_sha256"):
        with np.load(retained, allow_pickle=False) as previous:
            _, endpoints = targets(ts, price, job["arm"]["horizon_hours"])
            for key, expected in (("timestamps", ts[si]), ("target_timestamps", ts[endpoints[si]]),
                                  ("current", price[si]), ("actual", price[endpoints[si]])):
                if not np.array_equal(previous[key], expected):
                    raise ValueError("retained ARIMA population mismatch")
            prediction = previous["prediction"].copy()
        if not np.isfinite(prediction).all() or not (args.output/"ARIMA_FITS.json").exists():
            raise ValueError("retained ARIMA fit/predictions incomplete")
        return dict(prediction=prediction, parameters=2, device="CPU", reused_fit=True,
                    phase_fits=len(json.loads((args.output/"ARIMA_FITS.json").read_text())))
    h, delta = job["arm"]["horizon_hours"], job["arm"]["spacing_hours"]
    start, cutoff = epoch(job["week"]["fit_start"]), epoch(job["week"]["fit_cutoff"])
    predictions = np.full(len(si), np.nan)
    fits = []
    for phase in np.unique((ts[si]//3600) % delta):
        first = start + ((int(phase)-start//3600) % delta)*3600
        clock = np.arange(first, ts[si].max()+1, delta*3600, dtype=np.int64)
        source = np.searchsorted(ts, clock)
        present = (source < len(ts)) & (ts[np.minimum(source, len(ts)-1)] == clock)
        values = np.full(len(clock), np.nan)
        values[present] = np.log(price[source[present]])
        train = values[clock < cutoff]
        fit = SARIMAX(train, order=(1,1,0), trend="n", enforce_stationarity=True).fit(disp=False, maxiter=100)
        if not fit.mle_retvals.get("converged", False):
            raise ValueError("ARIMA_FAILED_CONVERGENCE")
        filtered = SARIMAX(values, order=(1,1,0), trend="n", enforce_stationarity=True).filter(fit.params)
        positions = np.flatnonzero((ts[si]//3600) % delta == phase)
        for pos in positions:
            index = int((ts[si[pos]]-first)//(delta*3600))
            steps = h//delta
            log_price = filtered.get_prediction(start=index+1, end=index+steps, dynamic=True).predicted_mean[-1]
            predictions[pos] = float(np.exp(log_price))
        fits.append(dict(phase=int(phase), train_slots=len(train), observed=int(np.isfinite(train).sum()), params=fit.params.tolist()))
        atomic_json(args.output/"STATUS.json", dict(job_id=job["id"], lease=job["lease"], status="ARIMA_PHASES", phases=len(fits), heartbeat=time.time()))
    if not np.isfinite(predictions).all():
        raise ValueError("ARIMA nonfinite forecast; no naive fallback")
    atomic_json(args.output/"ARIMA_FITS.json", fits)
    return dict(prediction=predictions, parameters=2, device="CPU", phase_fits=len(fits))


def run(args):
    start = time.monotonic()
    job = verify(json.loads(args.job.read_text()))
    design = verify(json.loads(args.design.read_text()))
    if job["design_sha256"] != design["sha256"]:
        raise ValueError("job design mismatch")
    from tools.i6m_prepare import code_identity
    if design["code_sha256"] != code_identity():
        raise ValueError("worker code differs from sealed runtime")
    args.output.mkdir(parents=True, exist_ok=True)
    terminal = args.output / ("PILOT.json" if args.pilot else "CELL.json")
    if terminal.exists():
        verify(json.loads(terminal.read_text()))
        return
    state = dict(job_id=job["id"], lease=job["lease"], design_sha256=design["sha256"],
                 seed=job["seed"], arm=job["arm"], week=job["week"], model=job["model"],
                 code_sha256=design["code_sha256"], input_sha256=design["input_sha256"], test_read=False)
    try:
        input_path = args.input
        ts, x, price = load_input(input_path, design["train_sha256"] if args.pilot else design["input_sha256"])
        if args.pilot and (job["week"]["ordinal"] != 0 or ts[-1] >= epoch("2024-01-01T00:00:00Z")):
            raise ValueError("cost pilot must be first-week TRAIN only")
        y, endpoint = targets(ts, price, job["arm"]["horizon_hours"])
        fi, vi, si = partitions(ts, y, job["week"], job["arm"]["horizon_hours"])
        if not args.pilot and not len(si):
            result = dict(status="NOT_APPLICABLE", reason="NO_EXACT_ENDPOINTS_INSIDE_VALIDATION", metrics=None)
        else:
            result = (neural(args, job, design, ts, x, price, y, endpoint, fi, vi, si) if job["model"] == "NEURAL"
                      else statistical(args, job, design, ts, price, si))
            if not args.pilot:
                prediction = result.pop("prediction")
                np.savez_compressed(args.output/"PREDICTIONS.npz", timestamps=ts[si], target_timestamps=ts[endpoint[si]],
                                    current=price[si], actual=price[endpoint[si]], prediction=prediction)
                result.update(status="COMPLETED", metrics=metrics(price[si], price[endpoint[si]], prediction, job["arm"]["horizon_hours"]),
                              rows_sha256=digest(ts[si].tolist()), predictions_sha256=file_hash(args.output/"PREDICTIONS.npz"))
                diagnostics = []
                for horizon in job["arm"]["diagnostic_naive_horizons_hours"]:
                    _, ei = targets(ts, price, horizon)
                    oi = si[ei[si] >= 0]
                    if len(oi):
                        diagnostics.append(metrics(price[oi], price[ei[oi]], price[oi], horizon))
                result["diagnostic_naives"] = diagnostics
        current_peak = peak_bytes()
        result = seal({**state, **{k:v for k,v in result.items() if k != "sha256"},
                       "elapsed_seconds": time.monotonic()-start,
                       "own_cgroup_peak_bytes": current_peak,
                       "cgroup_peak_bytes": max(result.get("cgroup_peak_bytes") or 0, current_peak or 0) or None})
        atomic_json(terminal, result)
        atomic_json(args.output/"STATUS.json", result)
    except Exception as exc:
        atomic_json(args.output/"STATUS.json", dict(state, status="FAILED", error=f"{type(exc).__name__}: {exc}"))
        if not args.pilot:
            atomic_json(terminal, seal(dict(state, status="FAILED", reason=f"{type(exc).__name__}: {exc}",
                                          elapsed_seconds=time.monotonic()-start, cgroup_peak_bytes=peak_bytes(), metrics=None)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
