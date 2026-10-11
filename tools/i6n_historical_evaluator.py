"""Endpoint-first, fixed-support historical callback for the shared I6N adapter.

No search, deployment, TEST authorization, clock inference or automatic retry.
The caller supplies the resolved cell config and external resource admission.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from tools import i6n_historical_cell as cell, i6n_search as search
from tools.i6n_contracts import business_metrics as bm, eligibility as c1, paired_metrics as c2

SCHEMA = "i6n.historical_evaluator.v1"
SPLITS = ("train", "inner")
CONTEXT_KEYS = {"schema", "base_config", "base_config_sha256", "frozen_support",
                "frozen_support_sha256", "output_root"}
ARTIFACTS = {"model.keras", "scores.npz", "train.input.json", "inner.input.json",
             "train.c2.json", "inner.c2.json"}


def candidate_config(base, effective):
    """Reject ignored/forged genes; map every supported model/window gene."""
    try:
        genome = {k: effective[k] for k in search._GENOME_KEYS}
        canonical = search.effective_config(genome)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid effective genome") from exc
    if c1.canonical_digest(canonical) != c1.canonical_digest(effective):
        raise ValueError("noncanonical or modified effective genome")
    if canonical["output_mode"] != "endpoint":
        raise ValueError("TRAJECTORY_UNSUPPORTED: independent delta_y supervision is not implemented")
    if canonical["horizon"] != base["horizon_hours"]:
        raise ValueError("candidate horizon differs from frozen support")
    return dict(deepcopy(base), samples=canonical["samples"], spacing_hours=canonical["delta_x"],
                horizon_hours=canonical["horizon"], model=deepcopy(canonical["core"]))


def transform_windows(x, kind):
    """Backward difference of TRAIN-scaled levels on this exact input grid."""
    if kind == "identity":
        return x
    if kind != "difference":
        raise ValueError("INPUT_TRANSFORM_UNSUPPORTED")
    result = np.zeros_like(x)
    market = (x[:, 1:, -1] == 1) & (x[:, :-1, -1] == 1)
    pairs = (x[:, 1:, 1:-1:2] == 1) & (x[:, :-1, 1:-1:2] == 1) & market[..., None]
    result[:, 1:, 0:-1:2] = np.where(pairs, x[:, 1:, 0:-1:2]-x[:, :-1, 0:-1:2], 0)
    result[:, 1:, 1:-1:2], result[:, 1:, -1] = pairs, market
    return result


def observations(data, cfg, split):
    part = getattr(data["parts"], split)
    origins = np.array([int(t.timestamp()) for t, mask in zip(part.origins, part.mask) if any(mask)], np.int64)
    oi = np.searchsorted(data["ts"], origins)
    ti = np.searchsorted(data["ts"], origins+cfg["horizon_hours"]*3600)
    return origins, data["price"][oi].astype(np.float64), data["price"][ti].astype(np.float64)


def support(data, cfg):
    """Bind full ordered masks, not a candidate-selected intersection."""
    result = dict(counts={}, naive_sums={}, support_digest={})
    governing = {k: cfg[k] for k in ("input", "source_clock", "calendar", "features", "asset", "unit", "role", "horizon_hours")}
    for name in SPLITS:
        part = getattr(data["parts"], name)
        origins, current, future = observations(data, cfg, name)
        result["counts"][name] = len(origins)
        result["naive_sums"][name] = math.fsum(np.abs(current-future).tolist())
        result["support_digest"][name] = c1.canonical_digest(dict(governing=governing, split=name,
            history_start=c1.iso(data["parts"].history_start), inner_start=c1.iso(data["parts"].inner_start),
            fit_cutoff=c1.iso(data["parts"].fit_cutoff), origins=[c1.iso(t) for t in part.origins],
            targets=[[c1.iso(t) for t in row] for row in part.target_instants], mask=part.mask))
    return search._support(result)


def split_plan(data, cfg, name, sealed):
    parts, part = data["parts"], getattr(data["parts"], name)
    start, end = ((parts.history_start, parts.inner_start) if name == "train" else (parts.inner_start, parts.fit_cutoff))
    targets = {row[0] for row in part.target_instants}
    raw = dict(cfg["calendar"], schema_version=c1.CONFIG_SCHEMA_VERSION, roles=[cfg["role"]],
        horizons_hours=[cfg["horizon_hours"]], origin_grid=dict(kind="EXPLICIT", timestamps=[c1.iso(t) for t in part.origins]),
        input_unavailable_origins=[c1.iso(t) for t in part.origins if t not in data["available"]],
        splits=[dict(split_id=name.upper(), start=c1.iso(start), end=c1.iso(end), label_available_by=c1.iso(end))],
        label_evidence=dict(observed_targets=[dict(timestamp=c1.iso(t), available_at=c1.iso(data["observed"][t]))
            for t in sorted(targets & data["observed"].keys())]))
    plan = c1.prepare_support_plan(raw)
    pop = bm.build_opportunity_population(plan, asset=cfg["asset"], split_id=name.upper(), sealed_at=sealed)
    expected = [c2.iso_utc(t) for t, mask in zip(part.origins, part.mask) if any(mask)]
    if [key["origin"] for key in pop["keys"]] != expected:
        raise ValueError("C1 population differs from full weekly endpoint support")
    return dict(plan=raw, population=pop)


def freeze(config_path, output):
    """Seal full TRAIN/inner endpoint S before proposals, without model fitting."""
    path, root = Path(config_path).resolve(), Path(output).resolve()
    cfg = cell.read_config(path)
    search.default_genome(cfg["horizon_hours"])
    data = cell.prepare(cfg)
    sealed = c1.iso(datetime.now(timezone.utc))
    frozen = dict(schema=SCHEMA, support=support(data, cfg),
                  splits={s: split_plan(data, cfg, s, sealed) for s in SPLITS})
    root.mkdir(parents=True, exist_ok=False)
    cell.atomic_json(root / "FROZEN.json", frozen)
    context = dict(schema=SCHEMA, base_config=str(path), base_config_sha256=cell.file_hash(path),
        frozen_support=str(root / "FROZEN.json"), frozen_support_sha256=cell.file_hash(root / "FROZEN.json"), output_root=str(root))
    binding = dict(i6n_horizon=cfg["horizon_hours"], i6n_support=frozen["support"],
                   i6n_evaluation_context=context,
                   i6n_evaluator_command=[sys.executable, "-m", "tools.i6n_historical_evaluator", "evaluate"],
                   i6n_evaluator_timeout=cfg["resources"]["max_seconds"] + 120)
    cell.atomic_json(root / "BINDING.json", binding)
    return binding


def load_request(request):
    if set(request) != {"config", "generation", "context", "support"}:
        raise ValueError("shared evaluator request schema mismatch")
    if type(request["generation"]) is not int or request["generation"] < 0:
        raise ValueError("invalid generation")
    ctx = request["context"]
    if set(ctx) != CONTEXT_KEYS or ctx["schema"] != SCHEMA:
        raise ValueError("historical context schema mismatch")
    for key in ("base_config", "frozen_support"):
        if cell.file_hash(ctx[key]) != ctx[key+"_sha256"]:
            raise ValueError(f"{key} digest mismatch")
    cfg = candidate_config(cell.read_config(ctx["base_config"]), request["config"])
    frozen = json.loads(Path(ctx["frozen_support"]).read_text())
    if frozen["schema"] != SCHEMA or search._support(request["support"]) != frozen["support"]:
        raise ValueError("request differs from frozen support")
    return cfg, frozen


def identity(cfg, request):
    return dict(cell=cell.identity(cfg), effective_config=request["config"], context=request["context"],
        support=request["support"], bridge_sha256=cell.file_hash(__file__), search_sha256=cell.file_hash(search.__file__))


def fit_candidate(data, cfg, effective, root, progress):
    from tools.i6n_model import build_model
    from tools.i6n_training import fit_masked
    from tools.modular_candidate_evaluator import _weight_digest
    arrays, batch = {}, cfg["training"]["batch_size"]
    for name in SPLITS:
        arrays[name] = cell.materialize(data, getattr(data["parts"], name), cfg)
        x = arrays[name][0]
        for i in range(0, len(x), batch):
            x[i:i+batch] = transform_windows(x[i:i+batch], effective["input_transform"])
    tf, placement = cell.device_context(cfg["device"])
    tf.keras.utils.set_random_seed(cfg["training"]["seed"])
    with tf.device(placement):
        probe = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        if not probe.device.endswith("/device:"+placement[1:]):
            raise ValueError("device placement mismatch; no fallback")
        model = build_model(cfg["samples"], 20, cfg["spacing_hours"], cfg["model"])
        if model.count_params() > cfg["resources"]["max_parameters"]:
            raise ValueError("parameters exceed resource limit")
        if any(not v.value.handle.device.endswith("/device:"+placement[1:]) for v in model.trainable_variables):
            raise ValueError("model variable placement mismatch")
        audit = fit_masked(model, *arrays["train"][:3], *arrays["inner"][:3], cfg["training"], progress)
        model.save(root / "model.tmp.keras")
        (root / "model.tmp.keras").replace(root / "model.keras")
        restored = tf.keras.models.load_model(root / "model.keras", compile=False)
        if _weight_digest(restored.get_weights()) != audit["restored_weights_sha256"]:
            raise ValueError("checkpoint reload weight digest mismatch")
        scores = {}
        for name in SPLITS:
            predictions = []
            for i in range(0, len(arrays[name][0]), batch):
                x = arrays[name][0][i:i+batch]
                expected, actual = model(x, training=False), restored(x, training=False)
                if not actual.device.endswith("/device:"+placement[1:]):
                    raise ValueError("prediction device mismatch")
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                predictions.extend(np.asarray(actual)[:, 0].tolist())
            origins, current, future = observations(data, cfg, name)
            predicted = cell.price_prediction(current, predictions, data["target_center"][0], data["target_scale"][0])
            scores.update({name+"_"+key: value for key, value in
                dict(origins=origins, current=current, future=future, prediction=predicted).items()})
    np.savez(root / "scores.tmp.npz", **scores)
    (root / "scores.tmp.npz").replace(root / "scores.npz")
    return dict(training=audit, checkpoint_reload_verified=True, parameters=model.count_params(),
        effective_config=effective, artifact_directory=str(root), feature_scaler_sha256=data["scaler"].sha256(),
        target_center=data["target_center"].tolist(), target_scale=data["target_scale"].tolist(),
        coverage_complete=data["parts"].coverage_complete, history_start=c1.iso(data["parts"].history_start),
        inner_start=c1.iso(data["parts"].inner_start), fit_cutoff=c1.iso(data["parts"].fit_cutoff),
        scientific_scope="RETROSPECTIVE_TRAIN_INNER_CONDITIONAL_SOURCE_AVAILABILITY",
        search_space_scope="ENDPOINT_ONLY")


def score_receipt(data, cfg, frozen, root, run_id, metadata, reuse=False):
    """C2 owns arithmetic; both forecasts and naive use the same sealed S."""
    receipt = dict(deepcopy(frozen["support"]), status="ok", error_sums={}, metadata=deepcopy(metadata))
    receipt["metadata"].update(endpoint_metrics={}, c2_receipts={})
    with np.load(root / "scores.npz", allow_pickle=False) as scores:
        for name in SPLITS:
            origins, current, future = observations(data, cfg, name)
            for key, actual in dict(origins=origins, current=current, future=future).items():
                if not np.array_equal(scores[name+"_"+key], actual):
                    raise ValueError("retained scores differ from full source support")
            prediction = scores[name+"_prediction"]
            if prediction.shape != current.shape or not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise ValueError("invalid full-support predictions")
            plan = c1.prepare_support_plan(frozen["splits"][name]["plan"])
            pop = frozen["splits"][name]["population"]
            forecasts = [dict(key, target_timestamp=c1.iso(cell.stamp(t+cfg["horizon_hours"]*3600)),
                price_origin=float(a), price_target=float(b), prediction=float(p), unit=cfg["unit"],
                scale="ORIGINAL_PRICE", checkpoint=metadata["training"]["restored_weights_sha256"])
                for key, t, a, b, p in zip(pop["keys"], origins, current, future, prediction)]
            document = dict(schema_version=c2.INPUT_SCHEMA_VERSION, opportunity_population=pop,
                forecasts=forecasts, forecast_failures=[], usage_receipts=[], contract=dict(
                roles=[cfg["role"].lower()+"_origin"], horizons_hours=[cfg["horizon_hours"]],
                calendar_id=bm.calendar_identity(plan), protocol_id=SCHEMA, config_id=run_id, unit=cfg["unit"], scale="ORIGINAL_PRICE"))
            stored = None
            if reuse:
                if document != json.loads((root / f"{name}.input.json").read_text()):
                    raise ValueError("retained metric input differs from checkpoint score rows")
                stored = json.loads((root / f"{name}.c2.json").read_text())
            result = bm.score_business_metrics(plan, document, asset=cfg["asset"], split_id=name.upper(),
                                               sealed_at=pop["sealed_at"], stored_result=stored)
            group = next(g for g in result["groups"] if g["population_kind"] == "S" and g["role"] != "ALL_ROLES")
            if (group["n"] != receipt["counts"][name] or group["coverage"] != 1
                    or group["naive_sum_abs_error"] != receipt["naive_sums"][name]):
                raise ValueError("C2 receipt changed frozen full support")
            receipt["error_sums"][name] = group["sum_abs_error"]
            receipt["metadata"]["endpoint_metrics"][name] = {k: group[k] for k in
                ("n", "mae", "mse", "naive_mae", "naive_mse", "skill_mae", "unit", "scale")}
            if not reuse:
                cell.atomic_json(root / f"{name}.input.json", document)
                cell.atomic_json(root / f"{name}.c2.json", result)
            receipt["metadata"]["c2_receipts"][name] = dict(path=str(root / f"{name}.c2.json"), sha256=cell.file_hash(root / f"{name}.c2.json"))
    receipt["fitness"] = search._score(receipt, frozen["support"])[0]
    return receipt


def retained(root, expected, data, cfg, frozen):
    state = json.loads((root / "STATUS.json").read_text())
    if state["identity"] != expected:
        raise ValueError("retained identity mismatch")
    if state["state"] != "COMPLETED":
        raise ValueError("retained candidate cannot retry: "+state["state"])
    if cell.file_hash(root / "RESULT.json") != state["result_sha256"]:
        raise ValueError("result digest mismatch")
    result = json.loads((root / "RESULT.json").read_text())
    if result["identity"] != expected or set(result["artifacts"]) != ARTIFACTS:
        raise ValueError("result identity or artifact digest set mismatch")
    for name, digest in result["artifacts"].items():
        if cell.file_hash(root / name) != digest:
            raise ValueError(name+" digest mismatch")
    receipt = score_receipt(data, cfg, frozen, root, root.name, result["receipt"]["metadata"], reuse=True)
    if receipt != result["receipt"]:
        raise ValueError("receipt differs from fresh C2 rescore")
    return receipt


def evaluate(request):
    """Shared-adapter callback. Generation changes do not authorize another fit."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("historical evaluator requires the main thread; use i6n_evaluator_command")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("existing wall timer; use an isolated evaluator process")
    cfg, frozen = load_request(request)
    expected = identity(cfg, request)
    root = Path(request["context"]["output_root"]) / "candidates" / c1.canonical_digest(expected)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / "STATUS.json").exists() and any(p.name != "RUN.lock" for p in root.iterdir()):
            raise ValueError("orphan candidate artifacts prohibit fitting")
        started = time.monotonic()
        state = dict(state="RUNNING", identity=expected, elapsed_seconds=0., eta_seconds=None)
        prior = (root / "STATUS.json").exists()
        if not prior:
            cell.atomic_json(root / "STATUS.json", state)
        def progress(event):
            state.update(progress=event, elapsed_seconds=time.monotonic()-started)
            if event["event"] == "epoch":
                state["eta_seconds"] = state["elapsed_seconds"]/event["epoch"]*(cfg["training"]["max_epochs"]-event["epoch"])
            cell.atomic_json(root / "STATUS.json", state)
        def expired(*_):
            raise TimeoutError("historical evaluator wall budget exceeded")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, cfg["resources"]["max_seconds"])
        try:
            data = cell.prepare(cfg)
            if support(data, cfg) != frozen["support"]:
                raise ValueError("candidate full support differs; intersection/drop forbidden")
            if prior:
                return retained(root, expected, data, cfg, frozen)
            metadata = fit_candidate(data, cfg, request["config"], root, progress)
            receipt = score_receipt(data, cfg, frozen, root, root.name, metadata)
            load_request(request)
            if identity(cfg, request) != expected:
                raise ValueError("evaluator identity changed during fit")
            result = dict(identity=expected, receipt=receipt, artifacts={n: cell.file_hash(root / n) for n in sorted(ARTIFACTS)})
            cell.atomic_json(root / "RESULT.json", result)
            state.update(state="COMPLETED", eta_seconds=0, elapsed_seconds=time.monotonic()-started,
                         result_sha256=cell.file_hash(root / "RESULT.json"))
            cell.atomic_json(root / "STATUS.json", state)
            return receipt
        except BaseException as exc:
            if not prior:
                state.update(state="FAILED", eta_seconds=None, error=f"{type(exc).__name__}: {exc}")
                cell.atomic_json(root / "STATUS.json", state)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze" and (args.config is None or args.output is None):
        parser.error("freeze requires --config and --output")
    result = freeze(args.config, args.output) if args.command == "freeze" else evaluate(json.load(sys.stdin))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
