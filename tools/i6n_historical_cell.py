"""One authenticated historical development cell; no clock guesses or retries."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

from tools.i6m_data import file_hash, load_input, price_prediction
from tools.i6m_queue import atomic_json
from tools.i6e_clock_windows import clock_windows
from tools.i6e_compact_run import scaled_inputs
from tools.i6n_contracts import business_metrics as bm, eligibility as c1, paired_metrics as c2
from tools.i6n_model import normalize_config
from tools.i6n_training import _settings
from tools.i6n_weekly_data import WeeklyDataError, build_weekly_partitions

BASE = Path(__file__).resolve().parents[1]
KEYS = set("schema input source_clock calendar evaluation_week_id evaluation_split coverage_mode role horizon_hours samples spacing_hours features model training resources device asset unit".split())
LIMITS = set("max_input_bytes max_rows max_samples max_materialized_bytes max_batch_bytes max_updates max_parameters max_seconds".split())
CLOCK_KEYS = set("schema status coverage_mode data_kind input_sha256 features calendar_sha256 timestamp_semantics availability_policy clock_policy".split())


def stamp(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc)


def output_offsets(cfg):
    """Exact elapsed-hour head offsets [dy, 2dy, ..., h] implied by n_horizons.

    One head is the endpoint h. n heads require dy = h / n to be an integer;
    dy and the input spacing are independent (no coupling to spacing_hours).
    """
    h, n = cfg["horizon_hours"], cfg["model"]["n_horizons"]
    if type(n) is not int or n < 1 or h % n:
        raise ValueError("trajectory heads must split the horizon into equal whole hours")
    dy = h // n
    return tuple(dy * k for k in range(1, n + 1))


def authenticated(cfg, trajectory=False):
    """Validate explicit configuration and clock attestation without decoding NPZ.

    ``trajectory=False`` (single-week cell, base configs) keeps exactly one
    endpoint head. The historical evaluator passes ``trajectory=True`` for a
    candidate with heads [dy, ..., h] and equal per-head loss weights.
    """
    if not isinstance(cfg, dict) or set(cfg) != KEYS or cfg["schema"] != "i6n.historical_cell.v1":
        raise ValueError("historical cell config schema mismatch")
    if cfg["coverage_mode"] != "complete":
        raise ValueError("complete historical coverage required")
    for name in ("asset", "unit", "evaluation_week_id"):
        if not isinstance(cfg[name], str) or not cfg[name].strip() or cfg[name] != cfg[name].strip():
            raise ValueError(f"invalid {name}")
    if cfg["device"] not in ("cpu", "gpu"):
        raise ValueError("explicit cpu or gpu required")
    c1.Role(cfg["role"])
    c1.validate_horizon(cfg["horizon_hours"])
    for key in ("samples", "spacing_hours"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "samples" else 1):
            raise ValueError(f"invalid {key}")
    features = cfg["features"]
    if (not isinstance(features, list) or len(features) != 20
            or any(not isinstance(f, str) or not f.strip() for f in features) or len(set(features)) != 20):
        raise ValueError("exactly 20 unique selected features required")
    normalize_config(cfg["model"])
    _settings(cfg["training"])
    if not trajectory:
        if cfg["model"]["n_horizons"] != 1 or len(cfg["training"]["head_weights"]) != 1:
            raise ValueError("exactly one horizon required")
    else:
        offsets = output_offsets(cfg)
        weights = cfg["training"]["head_weights"]
        if len(weights) != len(offsets):
            raise ValueError("head_weights must match trajectory heads")
        if len(set(weights)) != 1:
            raise ValueError("trajectory heads require equal loss weights")
        for offset in offsets:
            c1.validate_horizon(offset)
    limits = cfg["resources"]
    if set(limits) != LIMITS or any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("explicit positive integer resource limits required")
    if cfg["samples"] > limits["max_samples"]:
        raise ValueError("samples exceed resource limit")
    if set(cfg["calendar"]) != {"market_calendar", "entry_delay_after_open_hours", "force_flat_before_close_hours"}:
        raise ValueError("explicit C1 calendar required")
    _, weeks = c1.load_calendar(cfg["calendar"])
    if any(w.market_open.timestamp() % 3600 for w in weeks):
        raise ValueError("calendar opens must align with whole UTC hours")
    split = c1._split(cfg["evaluation_split"])
    if split.split_id.upper() == "TEST" or split.end > stamp(1735689600):
        raise ValueError("TEST is forbidden")
    for name in ("input", "source_clock"):
        binding = cfg[name]
        if set(binding) != {"path", "sha256"} or file_hash(binding["path"]) != binding["sha256"]:
            raise ValueError(f"{name} digest mismatch")
    clock = json.loads(Path(cfg["source_clock"]["path"]).read_text())
    expected = dict(schema="i6n.source_clock.v1", status="RESOLVED", coverage_mode="complete",
                    data_kind="historical", input_sha256=cfg["input"]["sha256"], features=features,
                    calendar_sha256=c1.canonical_digest(cfg["calendar"]),
                    timestamp_semantics="UTC_BAR_END", availability_policy="AT_TIMESTAMP")
    if (set(clock) != CLOCK_KEYS or any(clock.get(k) != v for k, v in expected.items())
            or not isinstance(clock.get("clock_policy"), str) or not clock["clock_policy"].strip()):
        raise ValueError("source clock unresolved or binding/availability contract mismatch")
    return weeks, split


def _head_labels(ts, price, origins, offset):
    """Exact elapsed-hour log-returns; ``found`` marks an observed t and t+o."""
    origins = np.asarray(origins, np.int64)
    oi = np.searchsorted(ts, origins)
    ti = np.searchsorted(ts, origins + offset * 3600)
    found = ((oi < len(ts)) & (ti < len(ts))
             & (ts[np.minimum(oi, len(ts)-1)] == origins)
             & (ts[np.minimum(ti, len(ts)-1)] == origins + offset * 3600))
    labels = np.zeros(len(origins), np.float64)
    labels[found] = np.log(price[ti[found]]) - np.log(price[oi[found]])
    return labels, found


def prepare(cfg, trajectory=False):
    """Build complete support and TRAIN-only transforms; never fit a model.

    Heads are ``output_offsets(cfg)``; each keeps its own C1 mask and TRAIN-only
    target center/scale. Feature scaling and the governing support use the
    endpoint column only, so endpoint rows are identical with or without
    intermediate heads. An intermediate head without TRAIN support (or with a
    degenerate scale) is marked untrainable, never scored as zero error.
    """
    weeks, split = authenticated(cfg, trajectory)
    offsets = output_offsets(cfg)
    limits, source = cfg["resources"], cfg["input"]
    with zipfile.ZipFile(source["path"]) as archive:
        if max(Path(source["path"]).stat().st_size, sum(i.file_size for i in archive.infolist())) > limits["max_input_bytes"]:
            raise ValueError("input bytes exceed resource limit")
    ts, raw, price = load_input(source["path"], source["sha256"])
    if len(ts) > limits["max_rows"] or raw.dtype.kind not in "fi":
        raise ValueError("source rows or numeric features invalid")
    count = sum(math.ceil((w.market_close-w.market_open).total_seconds()/3600) for w in weeks)
    if count > limits["max_rows"]:
        raise ValueError("calendar grid rows exceed resource limit")
    origins = tuple(stamp(t) for w in weeks for t in range(int(w.market_open.timestamp()),
                    math.ceil(w.market_close.timestamp()), 3600))
    observed = {stamp(t): stamp(t) for t in ts}
    span = (cfg["samples"]-1)*cfg["spacing_hours"]
    available = {t: t for t in observed if t.timestamp() >= ts[0]+span*3600}
    parts = build_weekly_partitions(origins=origins, weeks=weeks, observed_targets=observed,
        input_available_at=available, horizons=list(offsets),
        evaluation_week_id=cfg["evaluation_week_id"], evaluation_split=split,
        source_coverage_start=stamp(ts[0]), history_years=4, inner_days=28,
        history_span_hours=span, role=c1.Role(cfg["role"]), coverage_mode="complete")
    for name, part in (("TRAIN", parts.train), ("INNER", parts.inner)):
        if not any(m[-1] for m in part.mask):
            raise WeeklyDataError(f"NO_ELIGIBLE_ENDPOINT_{name}")
    populations = (parts.train, parts.inner, parts.evaluation)
    rows = [sum(1 for m in p.mask if any(m)) for p in populations]
    row_bytes = cfg["samples"]*41*4 + 8 + len(offsets)*(1 + 8)
    batch = cfg["training"]["batch_size"]
    if sum(rows)*row_bytes > limits["max_materialized_bytes"]:
        raise ValueError("materialized bytes exceed resource limit")
    # Values, indices, desired timestamps, masks, and concatenation coexist.
    if batch*cfg["samples"]*(2*41*4 + 2*8 + 2) > limits["max_batch_bytes"]:
        raise ValueError("window batch bytes exceed resource limit")
    if math.ceil(rows[0]/batch)*cfg["training"]["max_epochs"] > limits["max_updates"]:
        raise ValueError("optimizer updates exceed resource limit")
    # Feature scaler rows: the endpoint TRAIN support only (identical to endpoint mode).
    fit = np.searchsorted(ts, [t.timestamp() for t, m in zip(parts.train.origins, parts.train.mask) if m[-1]])
    scaled, scaler = scaled_inputs(np.where(np.isfinite(raw), raw, np.nan), fit)
    center, scale = np.zeros(len(offsets)), np.ones(len(offsets))
    status, trainable = [], np.zeros(len(offsets), bool)
    train_mask = np.asarray(parts.train.mask, bool)
    train_origins = np.array([int(t.timestamp()) for t in parts.train.origins], np.int64)
    for j, offset in enumerate(offsets):
        selected = train_origins[train_mask[:, j]]
        labels, found = _head_labels(ts, price, selected, offset)
        if not found.all():
            raise ValueError("eligible TRAIN label without exact observed bars")
        if not len(labels):
            status.append("NO_TRAIN_SUPPORT")
        elif not np.isfinite(labels.std()) or labels.std() <= 0:
            status.append("DEGENERATE_TRAIN_SCALE")
        else:
            center[j], scale[j], trainable[j] = labels.mean(), labels.std(), True
            status.append("TRAINED")
    if not trainable[-1]:
        raise ValueError("degenerate TRAIN target scale")
    return dict(ts=ts, price=price, scaled=scaled, scaler=scaler, fit_indices=fit,
                target_center=center, target_scale=scale, parts=parts, available=available,
                observed=observed, weeks=weeks, materialized_bytes=sum(rows)*row_bytes,
                offsets=offsets, head_status=status, head_trainable=trainable)


def windows(ts, scaled, origins, spacing_hours, samples):
    """Bridge arbitrary C1 origins to the exact-time observed-origin sampler."""
    index = np.searchsorted(ts, origins)
    present = (index < len(ts)) & (ts[np.minimum(index, len(ts)-1)] == origins)
    result = np.zeros((len(origins), samples, 41), np.float32)
    if present.any():
        values, mask = clock_windows(ts, scaled, index[present], spacing_hours, samples)
        result[present, :, :-1], result[present, :, -1] = values, mask
    return result


def materialize(data, part, cfg):
    """Rows with at least one trainable eligible head; labels per head.

    Untrainable heads are masked out everywhere. Invalid labels are finite
    zeros under a False mask (selected away before residuals), never NaN.
    Every True mask entry must have exact observed t and t+offset bars.
    """
    offsets = data.get("offsets", (cfg["horizon_hours"],))
    trainable = data.get("head_trainable", np.ones(len(offsets), bool))
    mask = np.asarray(part.mask, bool).reshape(len(part.origins), len(offsets)) & trainable[None, :]
    selected = mask.any(axis=1)
    origins = np.array([int(t.timestamp()) for t in part.origins], np.int64)[selected]
    mask = mask[selected]
    x = np.empty((len(origins), cfg["samples"], 41), np.float32)
    batch = cfg["training"]["batch_size"]
    for i in range(0, len(origins), batch):
        x[i:i+batch] = windows(data["ts"], data["scaled"], origins[i:i+batch], cfg["spacing_hours"], cfg["samples"])
    labels = np.zeros(mask.shape, np.float64)
    for j, offset in enumerate(offsets):
        values, found = _head_labels(data["ts"], data["price"], origins, offset)
        if (mask[:, j] & ~found).any():
            raise ValueError("eligible label without exact observed bars at t and t+offset")
        labels[:, j] = values
    y = np.where(mask, (labels-data["target_center"])/data["target_scale"], 0.).astype(np.float32)
    return x, y, mask, origins


def evaluation_support(data, cfg):
    part = data["parts"].evaluation
    week = next(w for w in cfg["calendar"]["market_calendar"]["weeks"] if w["week_id"] == cfg["evaluation_week_id"])
    raw = dict(cfg["calendar"], schema_version=c1.CONFIG_SCHEMA_VERSION,
        market_calendar={**cfg["calendar"]["market_calendar"], "weeks": [week]},
        roles=[cfg["role"]], horizons_hours=[cfg["horizon_hours"]],
        origin_grid={"kind": "EXPLICIT", "timestamps": [c1.iso(t) for t in part.origins]},
        input_unavailable_origins=[c1.iso(t) for t in part.origins if t not in data["available"]],
        splits=[cfg["evaluation_split"]], label_evidence={"observed_targets": [
            {"timestamp": c1.iso(t), "available_at": c1.iso(a)} for t, a in data["observed"].items()]})
    plan = c1.prepare_support_plan(raw)
    sealed = c1.iso(datetime.now(timezone.utc))
    pop = bm.build_opportunity_population(plan, asset=cfg["asset"], split_id=cfg["evaluation_split"]["split_id"], sealed_at=sealed)
    if {c2.parse_utc("origin", k["origin"]) for k in pop["keys"]} != {t for t, m in zip(part.origins, part.mask) if any(m)}:
        raise ValueError("C1/C2 population differs from weekly per-head M")
    return raw, plan, pop


def device_context(device):
    import tensorflow as tf
    tf.config.set_soft_device_placement(False)
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) != 1:
            raise ValueError("exactly one visible GPU required; no fallback")
        tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.enable_op_determinism()
    return tf, f"/{device.upper()}:0"


def execute(cfg, root, progress):
    """Real masked fit and reload, then original-price C2 evaluation on sealed M."""
    from tools.i6n_model import build_model
    from tools.i6n_training import fit_masked
    from tools.modular_candidate_evaluator import _weight_digest
    data = prepare(cfg)
    raw_plan, plan, pop = evaluation_support(data, cfg)
    atomic_json(root / "calendar_input.json", raw_plan)
    atomic_json(root / "population.json", pop)
    train, inner, evaluation = [materialize(data, p, cfg) for p in
                               (data["parts"].train, data["parts"].inner, data["parts"].evaluation)]
    tf, placement = device_context(cfg["device"])
    tf.keras.utils.set_random_seed(cfg["training"]["seed"])
    with tf.device(placement):
        probe = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        if not probe.device.endswith("/device:"+placement[1:]):
            raise ValueError("device probe placement mismatch")
        model = build_model(cfg["samples"], 20, cfg["spacing_hours"], cfg["model"])
        if model.count_params() > cfg["resources"]["max_parameters"]:
            raise ValueError("parameters exceed resource limit")
        if any(not v.value.handle.device.endswith("/device:"+placement[1:]) for v in model.trainable_variables):
            raise ValueError("model variable placement mismatch")
        audit = fit_masked(model, *train[:3], *inner[:3], cfg["training"], progress)
        model.save(root / "checkpoint.tmp.keras")
        (root / "checkpoint.tmp.keras").replace(root / "model.keras")
        restored = tf.keras.models.load_model(root / "model.keras", compile=False)
        if _weight_digest(restored.get_weights()) != audit["restored_weights_sha256"]:
            raise ValueError("checkpoint reload weight digest mismatch")
        predicted = []
        for i in range(0, len(evaluation[0]), cfg["training"]["batch_size"]):
            xb = evaluation[0][i:i+cfg["training"]["batch_size"]]
            original, replay = model(xb, training=False), restored(xb, training=False)
            if not replay.device.endswith("/device:"+placement[1:]):
                raise ValueError("model output placement mismatch")
            np.testing.assert_allclose(replay, original, rtol=1e-6, atol=1e-7)
            predicted.extend(np.asarray(replay)[:, 0].tolist())
    oi = np.searchsorted(data["ts"], evaluation[3])
    ti = np.searchsorted(data["ts"], evaluation[3]+cfg["horizon_hours"]*3600)
    prices = price_prediction(data["price"][oi], predicted, data["target_center"][0], data["target_scale"][0])
    forecasts = [dict(key, target_timestamp=c1.iso(stamp(t+cfg["horizon_hours"]*3600)),
        price_origin=float(data["price"][a]), price_target=float(data["price"][b]), prediction=float(p),
        checkpoint=audit["restored_weights_sha256"], unit=cfg["unit"], scale="ORIGINAL_PRICE")
        for key, t, a, b, p in zip(pop["keys"], evaluation[3], oi, ti, prices)]
    document = dict(schema_version=c2.INPUT_SCHEMA_VERSION, opportunity_population=pop,
        forecasts=forecasts, forecast_failures=[], usage_receipts=[], contract=dict(
        roles=[cfg["role"].lower()+"_origin"], horizons_hours=[cfg["horizon_hours"]],
        calendar_id=bm.calendar_identity(plan), protocol_id="i6n.historical_cell.v1",
        config_id=c1.canonical_digest(cfg), unit=cfg["unit"], scale="ORIGINAL_PRICE"))
    metrics = bm.score_business_metrics(plan, document, asset=cfg["asset"],
        split_id=cfg["evaluation_split"]["split_id"], sealed_at=pop["sealed_at"])
    atomic_json(root / "metric_input.json", document)
    parts = data["parts"]
    return dict(training=audit, metrics=metrics, coverage_complete=parts.coverage_complete,
        checkpoint_reload_verified=True, evaluation_pairs=parts.evaluation.n_eligible,
        support_sha256=parts.support_sha256, fit_cutoff=c1.iso(parts.fit_cutoff),
        history_start=c1.iso(parts.history_start), inner_start=c1.iso(parts.inner_start),
        materialized_bytes=data["materialized_bytes"], parameters=model.count_params(),
        feature_scaler={k: getattr(data["scaler"], k).tolist() for k in ("median", "mean", "sd")},
        feature_scaler_sha256=data["scaler"].sha256(), target_center=data["target_center"].tolist(),
        target_scale=data["target_scale"].tolist(), device=placement,
        partitions={name: dict(origins=len(p.origins), eligible=p.n_eligible, exclusions=p.exclusions)
                    for name, p in (("train", parts.train), ("inner", parts.inner), ("evaluation", parts.evaluation))})


def identity(cfg):
    names = "i6n_historical_cell i6n_weekly_data i6n_training i6n_model i6m_data i6m_queue i6e_clock_windows i6e_compact_run i6e_masked_model fs4_temporal_predictor fs4_hourly_support modular_candidate_evaluator"
    paths = [BASE / "tools" / (n+".py") for n in names.split()]
    paths += list((BASE / "tools/i6n_contracts").glob("*.py"))
    paths += list((BASE / "predictor_plugins/modular_temporal").rglob("*.py"))
    packages = {n: importlib.metadata.version(n) for n in ("numpy", "tensorflow", "keras")}
    return dict(config=cfg, code={str(p.relative_to(BASE)): file_hash(p) for p in paths},
                python=sys.version, packages=packages, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    for key in ("input", "source_clock"):
        cfg[key]["path"] = str((Path(path).resolve().parent / cfg[key]["path"]).resolve())
    authenticated(cfg)
    return cfg


def verified(root, expected):
    state = json.loads((root / "STATUS.json").read_text())
    if state["identity"] != expected:
        raise ValueError("retained identity mismatch")
    if state["state"] != "COMPLETED":
        raise ValueError("retained run cannot retry: " + state["state"])
    if file_hash(root / "RESULT.json") != state["result_sha256"]:
        raise ValueError("result digest mismatch")
    result = json.loads((root / "RESULT.json").read_text())
    if result["identity"] != expected or result["state"] != "COMPLETED":
        raise ValueError("result identity mismatch")
    required = {"model.keras", "calendar_input.json", "metric_input.json", "population.json"}
    if set(result["artifacts"]) != required:
        raise ValueError("artifact digest set mismatch")
    for name, sha in result["artifacts"].items():
        if file_hash(root / name) != sha:
            raise ValueError(f"{name} digest mismatch")
    document = json.loads((root / "metric_input.json").read_text())
    cfg = expected["config"]
    if document["contract"]["config_id"] != c1.canonical_digest(cfg):
        raise ValueError("metric config identity mismatch")
    if document["opportunity_population"] != json.loads((root / "population.json").read_text()):
        raise ValueError("population identity mismatch")
    plan = c1.prepare_support_plan(json.loads((root / "calendar_input.json").read_text()))
    bm.score_business_metrics(plan, document, asset=cfg["asset"], split_id=cfg["evaluation_split"]["split_id"],
        sealed_at=document["opportunity_population"]["sealed_at"], stored_result=result["metrics"])
    return result


def run(config_path, output):
    cfg = read_config(config_path)
    expected, root = identity(cfg), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "STATUS.json").exists():
            return verified(root, expected)
        if any(p.name != "RUN.lock" for p in root.iterdir()):
            raise ValueError("orphan artifacts prohibit fitting")
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise ValueError("existing wall timer; use an isolated worker")
        started = time.monotonic()
        state = dict(state="RUNNING", identity=expected, scope="HISTORICAL_SINGLE_WEEK_DEVELOPMENT",
                     started_at=c1.iso(datetime.now(timezone.utc)), elapsed_seconds=0., eta_seconds=None)
        atomic_json(root / "STATUS.json", state)
        def progress(event):
            state.update(progress=event, elapsed_seconds=time.monotonic()-started)
            if event["event"] == "epoch":
                state["eta_seconds"] = state["elapsed_seconds"]/event["epoch"]*(cfg["training"]["max_epochs"]-event["epoch"])
            atomic_json(root / "STATUS.json", state)
            if state["elapsed_seconds"] > cfg["resources"]["max_seconds"]:
                raise TimeoutError("cell wall budget exceeded")
        def expired(*_):
            raise TimeoutError("cell wall budget exceeded")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, cfg["resources"]["max_seconds"])
        try:
            result = execute(cfg, root, progress)
            authenticated(cfg)
            if identity(cfg) != expected:
                raise ValueError("identity changed during run")
            result.update(state="COMPLETED", identity=expected, scope=state["scope"], eta_seconds=0,
                elapsed_seconds=time.monotonic()-started, artifacts={n: file_hash(root / n) for n in
                    ("model.keras", "calendar_input.json", "metric_input.json", "population.json")})
            atomic_json(root / "RESULT.json", result)
            state.update(state="COMPLETED", result_sha256=file_hash(root / "RESULT.json"), eta_seconds=0,
                         elapsed_seconds=result["elapsed_seconds"])
            atomic_json(root / "STATUS.json", state)
            return result
        except BaseException as exc:
            state.update(state="FAILED", error=f"{type(exc).__name__}: {exc}", eta_seconds=None)
            atomic_json(root / "STATUS.json", state)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def status(config_path, output):
    """Read-only progress or complete identity/artifact/metric verification."""
    expected, root = identity(read_config(config_path)), Path(output)
    if not (root / "STATUS.json").exists():
        return dict(state="NOT_RUN"), 3
    state = json.loads((root / "STATUS.json").read_text())
    if state["identity"] != expected:
        raise ValueError("retained identity mismatch")
    if state["state"] == "COMPLETED":
        return verified(root, expected), 0
    if state["state"] == "RUNNING":
        with (root / "RUN.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                state.update(state="INTERRUPTED", eta_seconds=None)
    return state, 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result, code = ((run(args.config, args.output), 0) if args.command == "run" else
                        status(args.config, args.output))
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps(dict(state="REFUSED", error=str(exc))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
