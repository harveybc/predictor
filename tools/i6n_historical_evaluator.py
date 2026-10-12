"""Fixed-support historical callback for the shared I6N adapter (lane A).

Endpoint and real output-trajectory supervision [dy, 2dy, ..., h]. The
comparison metric is ALWAYS the endpoint h on the frozen fold support;
intermediate heads have their own C1 mask, TRAIN-only scale and naive and are
stored separately as diagnostics. They never change endpoint rows.

No search, deployment, TEST authorization, clock inference or automatic retry.
A linked recovery attempt (at most three physical attempts per identity) runs
only when the caller supplies the previous terminal status digest. The caller
supplies the resolved cell config and external resource admission.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import signal
import sys
import threading
import time

import numpy as np

from tools import i6n_historical_cell as cell, i6n_search as search
from tools.i6n_contracts import business_metrics as bm, eligibility as c1, paired_metrics as c2

SCHEMA = "i6n.historical_evaluator.v2"
# v1 contexts (the retained endpoint-only terminals) are verifiable, never fitted.
LEGACY_SCHEMA = "i6n.historical_evaluator.v1"
LEGACY_CONTEXT_KEYS = {"schema", "base_config", "base_config_sha256", "frozen_support",
                       "frozen_support_sha256", "output_root"}
RECEIPT_SCHEMA = "i6n.candidate_receipt.v1"
SPLITS = ("train", "inner")
CONTEXT_KEYS = {"schema", "base_config", "base_config_sha256", "frozen_support",
                "frozen_support_sha256", "output_root", "experiment", "fold"}
ARTIFACTS = {"model.keras", "scores.npz", "train.input.json", "inner.input.json",
             "train.c2.json", "inner.c2.json"}
# Plan section 2: mean TRAIN/inner monitor, patience 5, min_delta 1e-6,
# at most 50 epochs, best checkpoint restored; seed 6102026, one repetition.
PLAN_STOPPING = dict(max_epochs=50, patience=5, min_delta=1e-6, seed=6102026)
MAX_ATTEMPTS = 3
ATTEMPT_KEYS = {"index", "parent_status_sha256", "fencing_token"}
OPTIONAL_ATTEMPT_KEYS = {"parent"}
# A.3: the swarm ledger's record of the previous physical attempt (authority for lineage).
LEDGER_PARENT_KEYS = {"fit_id", "attempt_index", "fencing_token", "state", "state_sha256"}
LEDGER_PARENT_STATES = ("FAILED", "INTERRUPTED")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def plan_stopping(cfg):
    """Refuse any training block that differs from plan section 2 stopping."""
    training = cfg["training"]
    wrong = {k: training.get(k) for k, v in PLAN_STOPPING.items()
             if type(training.get(k)) is not type(v) or training.get(k) != v}
    if wrong:
        raise ValueError(f"PLAN_STOPPING: training differs from plan section 2: {wrong}")
    return dict(PLAN_STOPPING)


def _label(value, name):
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise ValueError(f"{name} must be a short identifier [A-Za-z0-9._:-]")
    return value


def _hex(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _token_order(parent, child):
    """Fencing tokens must increase; integer tokens compare numerically."""
    if parent.isdigit() and child.isdigit():
        return int(child) > int(parent)
    return child != parent


def normalize_attempt(value):
    """Attempt 1 by default; k>1 carries lineage (A.3).

    Lineage authority for k>1 is the swarm ledger's ``parent`` record
    (fit_id, attempt_index=k-1, parent fencing token, FAILED|INTERRUPTED state,
    optional recorded status digest), so recovery works on a host that never
    saw the parent. A local ``parent_status_sha256`` remains accepted (A.2)
    and is checked against the local file whenever that file exists.
    """
    if value is None:
        return dict(index=1, parent_status_sha256=None, fencing_token=None, parent=None)
    if (not isinstance(value, dict) or not ATTEMPT_KEYS <= set(value)
            or set(value) - ATTEMPT_KEYS - OPTIONAL_ATTEMPT_KEYS):
        raise ValueError(f"attempt requires exactly {sorted(ATTEMPT_KEYS)} plus optional {sorted(OPTIONAL_ATTEMPT_KEYS)}")
    index, local, token = value["index"], value["parent_status_sha256"], value["fencing_token"]
    parent = value.get("parent")
    if type(index) is not int or not 1 <= index <= MAX_ATTEMPTS:
        raise ValueError(f"attempt index must be 1..{MAX_ATTEMPTS} physical attempts")
    if token is not None and (not isinstance(token, str) or not token.strip()):
        raise ValueError("attempt fencing_token must be nonempty text or null")
    if local is not None and not _hex(local):
        raise ValueError("attempt parent_status_sha256 must be a sha256 hex digest or null")
    if index == 1:
        if local is not None or parent is not None:
            raise ValueError("attempt 1 has no parent status or ledger parent")
    elif parent is None and local is None:
        raise ValueError("attempt parent_status_sha256 or ledger parent required for a linked recovery attempt")
    if parent is not None:
        if not isinstance(parent, dict) or set(parent) != LEDGER_PARENT_KEYS:
            raise ValueError(f"ledger parent record requires exactly {sorted(LEDGER_PARENT_KEYS)}")
        if not _hex(parent["fit_id"]):
            raise ValueError("ledger parent fit_id must be a sha256 hex digest")
        if type(parent["attempt_index"]) is not int or parent["attempt_index"] != index - 1:
            raise ValueError("ledger parent attempt_index must equal attempt index - 1")
        if parent["state"] not in LEDGER_PARENT_STATES:
            raise ValueError(f"ledger parent state must be one of {LEDGER_PARENT_STATES}")
        if parent["state_sha256"] is not None and not _hex(parent["state_sha256"]):
            raise ValueError("ledger parent state_sha256 must be a sha256 hex digest or null")
        ptoken = parent["fencing_token"]
        if not isinstance(ptoken, str) or not ptoken.strip():
            raise ValueError("ledger parent fencing_token must be nonempty text")
        if token is None or not _token_order(ptoken, token):
            raise ValueError("attempt fencing_token must increase over the ledger parent's token")
        parent = dict(parent)
    return dict(index=index, parent_status_sha256=local, fencing_token=token, parent=parent)


def fit_id(request):
    """Lane B/E fit_id of this request (tools.i6n_swarm.common.fit_id formula)."""
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    ctx = request["context"]
    return sha(dict(experiment=ctx["experiment"], fold=ctx["fold"],
                    effective_config_sha256=sha(request["config"]),
                    context_sha256=sha(ctx), support_sha256=sha(request["support"])))


def candidate_config(base, effective):
    """Reject ignored/forged genes; map every supported model/window gene."""
    try:
        genome = {k: effective[k] for k in search._GENOME_KEYS}
        canonical = search.effective_config(genome)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid effective genome") from exc
    if c1.canonical_digest(canonical) != c1.canonical_digest(effective):
        raise ValueError("noncanonical or modified effective genome")
    if canonical["horizon"] != base["horizon_hours"]:
        raise ValueError("candidate horizon differs from frozen support")
    offsets = canonical["output_offsets"]
    cfg = dict(deepcopy(base), samples=canonical["samples"], spacing_hours=canonical["delta_x"],
               horizon_hours=canonical["horizon"], model=deepcopy(canonical["core"]))
    # Equal loss weights between heads; empty heads drop out per batch/split.
    cfg["training"] = dict(cfg["training"], head_weights=[1.] * len(offsets))
    if list(cell.output_offsets(cfg)) != offsets:
        raise ValueError("effective trajectory offsets differ from model heads")
    return cfg


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


PART = {"train": "train", "inner": "inner", "validation": "evaluation"}


def _offsets(data, cfg):
    return tuple(data.get("offsets", (cfg["horizon_hours"],)))


def observations(data, cfg, split, column=-1):
    """Ordered C1-eligible origins and original prices of one head (endpoint default)."""
    part = getattr(data["parts"], PART[split])
    offset = _offsets(data, cfg)[column]
    origins = np.array([int(t.timestamp()) for t, mask in zip(part.origins, part.mask) if mask[column]], np.int64)
    oi = np.searchsorted(data["ts"], origins)
    ti = np.searchsorted(data["ts"], origins+offset*3600)
    if len(origins) and (not np.array_equal(data["ts"][oi], origins)
                         or not np.array_equal(data["ts"][np.minimum(ti, len(data["ts"])-1)], origins+offset*3600)):
        raise ValueError("eligible head without exact observed bars")
    return origins, data["price"][oi].astype(np.float64), data["price"][ti].astype(np.float64)


def support(data, cfg):
    """Bind the endpoint column's full ordered mask, not a candidate intersection.

    Intermediate heads are projected away, so an endpoint-only candidate and a
    trajectory candidate of the same fold bind byte-identical support.
    """
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
            targets=[[c1.iso(row[-1])] for row in part.target_instants], mask=[[m[-1]] for m in part.mask]))
    return search._support(result)


def split_plan(data, cfg, name, sealed, column=-1):
    """Independent C1 plan/population for one head; must equal its weekly mask."""
    parts, part = data["parts"], getattr(data["parts"], name)
    start, end = ((parts.history_start, parts.inner_start) if name == "train" else (parts.inner_start, parts.fit_cutoff))
    targets = {row[column] for row in part.target_instants}
    raw = dict(cfg["calendar"], schema_version=c1.CONFIG_SCHEMA_VERSION, roles=[cfg["role"]],
        horizons_hours=[_offsets(data, cfg)[column]], origin_grid=dict(kind="EXPLICIT", timestamps=[c1.iso(t) for t in part.origins]),
        input_unavailable_origins=[c1.iso(t) for t in part.origins if t not in data["available"]],
        splits=[dict(split_id=name.upper(), start=c1.iso(start), end=c1.iso(end), label_available_by=c1.iso(end))],
        label_evidence=dict(observed_targets=[dict(timestamp=c1.iso(t), available_at=c1.iso(data["observed"][t]))
            for t in sorted(targets & data["observed"].keys())]))
    plan = c1.prepare_support_plan(raw)
    pop = bm.build_opportunity_population(plan, asset=cfg["asset"], split_id=name.upper(), sealed_at=sealed)
    expected = [c2.iso_utc(t) for t, mask in zip(part.origins, part.mask) if mask[column]]
    if [key["origin"] for key in pop["keys"]] != expected:
        raise ValueError("C1 population differs from full weekly per-head support")
    return dict(plan=raw, population=pop)


def outer_plan(data, cfg, sealed, column=-1):
    """C1 plan/population of the outer development week for one head.

    Used only for weekly-refit units (freeze --score-outer). Sealed before any
    fit; scored after the fit as VALIDATION diagnostics, never in fitness or
    early stopping.
    """
    part = data["parts"].evaluation
    offset = _offsets(data, cfg)[column]
    week = next(w for w in cfg["calendar"]["market_calendar"]["weeks"] if w["week_id"] == cfg["evaluation_week_id"])
    targets = {row[column] for row in part.target_instants}
    raw = dict(cfg["calendar"], schema_version=c1.CONFIG_SCHEMA_VERSION,
        market_calendar={**cfg["calendar"]["market_calendar"], "weeks": [week]},
        roles=[cfg["role"]], horizons_hours=[offset],
        origin_grid=dict(kind="EXPLICIT", timestamps=[c1.iso(t) for t in part.origins]),
        input_unavailable_origins=[c1.iso(t) for t in part.origins if t not in data["available"]],
        splits=[cfg["evaluation_split"]], label_evidence=dict(observed_targets=[
            dict(timestamp=c1.iso(t), available_at=c1.iso(data["observed"][t]))
            for t in sorted(targets & data["observed"].keys())]))
    plan = c1.prepare_support_plan(raw)
    pop = bm.build_opportunity_population(plan, asset=cfg["asset"],
                                          split_id=cfg["evaluation_split"]["split_id"], sealed_at=sealed)
    expected = [c2.iso_utc(t) for t, mask in zip(part.origins, part.mask) if mask[column]]
    if [key["origin"] for key in pop["keys"]] != expected:
        raise ValueError("C1 population differs from the outer week per-head support")
    return dict(plan=raw, population=pop)


def freeze(config_path, output, experiment, fold, score_outer=False):
    """Seal full TRAIN/inner endpoint S of one (horizon, fold), without fitting.

    ``score_outer`` (weekly-refit units only) also seals the outer week's
    endpoint S, count and naive sum; those rows are reported as VALIDATION
    diagnostics after the fit and never enter fitness or stopping.
    """
    if type(score_outer) is not bool:
        raise ValueError("score_outer must be a boolean")
    path, root = Path(config_path).resolve(), Path(output).resolve()
    _label(experiment, "experiment")
    _label(fold, "fold")
    cfg = cell.read_config(path)
    plan_stopping(cfg)
    search.default_genome(cfg["horizon_hours"])
    data = cell.prepare(cfg)
    sealed = c1.iso(datetime.now(timezone.utc))
    frozen = dict(schema=SCHEMA, support=support(data, cfg),
                  splits={s: split_plan(data, cfg, s, sealed) for s in SPLITS})
    if score_outer:
        _, current, future = observations(data, cfg, "validation")
        frozen["outer"] = dict(outer_plan(data, cfg, sealed), split_id=cfg["evaluation_split"]["split_id"],
                               count=len(current), naive_sum=math.fsum(np.abs(current-future).tolist()))
    root.mkdir(parents=True, exist_ok=False)
    cell.atomic_json(root / "FROZEN.json", frozen)
    context = dict(schema=SCHEMA, base_config=str(path), base_config_sha256=cell.file_hash(path),
        frozen_support=str(root / "FROZEN.json"), frozen_support_sha256=cell.file_hash(root / "FROZEN.json"), output_root=str(root),
        experiment=experiment, fold=fold)
    binding = dict(i6n_horizon=cfg["horizon_hours"], i6n_support=frozen["support"],
                   i6n_evaluation_context=context,
                   i6n_evaluator_command=[sys.executable, "-m", "tools.i6n_historical_evaluator", "evaluate"],
                   i6n_evaluator_timeout=cfg["resources"]["max_seconds"] + 120)
    cell.atomic_json(root / "BINDING.json", binding)
    return binding


def load_request(request):
    """Return (candidate cell config, frozen support); validates any attempt.

    v2 contexts require experiment/fold labels and plan section 2 stopping.
    v1 contexts load for read-only verification of retained terminals only;
    ``evaluate`` refuses to fit them.
    """
    keys = {"config", "generation", "context", "support"}
    if isinstance(request, dict) and "resume_checkpoint" in request:
        raise ValueError("RESUME_UNSUPPORTED: no optimizer/RNG/epoch checkpoint is written; "
                         "use one linked RECOVERY attempt")
    if not isinstance(request, dict) or set(request) - {"attempt"} != keys:
        raise ValueError("shared evaluator request schema mismatch")
    normalize_attempt(request.get("attempt"))
    if type(request["generation"]) is not int or request["generation"] < 0:
        raise ValueError("invalid generation")
    ctx = request["context"]
    schema = ctx.get("schema") if isinstance(ctx, dict) else None
    required = {SCHEMA: CONTEXT_KEYS, LEGACY_SCHEMA: LEGACY_CONTEXT_KEYS}.get(schema)
    if required is None or set(ctx) != required:
        raise ValueError("historical context schema mismatch")
    for key in ("base_config", "frozen_support"):
        if cell.file_hash(ctx[key]) != ctx[key+"_sha256"]:
            raise ValueError(f"{key} digest mismatch")
    base = cell.read_config(ctx["base_config"])
    if schema == SCHEMA:
        _label(ctx["experiment"], "experiment")
        _label(ctx["fold"], "fold")
        plan_stopping(base)
    cfg = candidate_config(base, request["config"])
    frozen = json.loads(Path(ctx["frozen_support"]).read_text())
    if frozen["schema"] != schema or search._support(request["support"]) != frozen["support"]:
        raise ValueError("request differs from frozen support")
    return cfg, frozen


def identity(cfg, request):
    """Full identity; generation and attempt are deliberately excluded."""
    return dict(cell=cell.identity(cfg), effective_config=request["config"], context=request["context"],
        support=request["support"], bridge_sha256=cell.file_hash(__file__), search_sha256=cell.file_hash(search.__file__))


def attempt_root(context, digest, index):
    base = Path(context["output_root"]) / "candidates"
    return base / digest if index == 1 else base / f"{digest}.attempt{index}"


def candidate_root(request, index=1):
    """Directory of one physical attempt of the request's candidate identity."""
    cfg, _ = load_request(request)
    return attempt_root(request["context"], c1.canonical_digest(identity(cfg, request)), index)


def _state(root):
    path = root / "STATUS.json"
    return json.loads(path.read_text())["state"] if path.exists() else None


def _check_parent(parent_root, attempt, request):
    """Lineage check for attempt k>1 (A.3).

    The ledger parent record is the authority: it must name this request's
    fit_id (same experiment, fold, config, context and support). When the
    parent's local directory exists (same host), its STATUS is an additional
    check: it must not be running, must not be COMPLETED (that would have been
    adopted), and must match every digest the request quotes.
    """
    ledger = attempt["parent"]
    if ledger is not None and ledger["fit_id"] != fit_id(request):
        raise ValueError("ledger parent fit_id differs from this request's identity/fold")
    path = parent_root / "STATUS.json"
    if not path.exists():
        if ledger is None:
            raise ValueError("attempt parent status digest mismatch (no local parent; supply the ledger parent)")
        return
    digest = cell.file_hash(path)
    for quoted in (attempt["parent_status_sha256"], ledger and ledger["state_sha256"]):
        if quoted is not None and quoted != digest:
            raise ValueError("attempt parent status digest mismatch (local parent status differs)")
    state = _state(parent_root)
    if state == "RUNNING":
        with (parent_root / "RUN.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("attempt parent is still running") from None
            fcntl.flock(lock, fcntl.LOCK_UN)
    elif state != "FAILED":
        raise ValueError(f"attempt parent is not an interrupted/failed terminal: {state}")


def fit_candidate(data, cfg, effective, root, progress, outer=False):
    """Real masked multi-head fit, checkpoint reload, per-head price predictions."""
    from tools.i6n_model import build_model, verify_placement
    from tools.i6n_training import fit_masked
    from tools.modular_candidate_evaluator import _weight_digest
    offsets = _offsets(data, cfg)
    arrays, batch = {}, cfg["training"]["batch_size"]
    names = SPLITS + (("validation",) if outer else ())
    for name in names:
        arrays[name] = cell.materialize(data, getattr(data["parts"], PART[name]), cfg)
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
        placement_report = verify_placement(model, placement, arrays["train"][0][:1])
        started = time.monotonic()
        audit = fit_masked(model, *arrays["train"][:3], *arrays["inner"][:3], cfg["training"], progress)
        fit_seconds = time.monotonic() - started
        model.save(root / "model.tmp.keras")
        (root / "model.tmp.keras").replace(root / "model.keras")
        restored = tf.keras.models.load_model(root / "model.keras", compile=False)
        if _weight_digest(restored.get_weights()) != audit["restored_weights_sha256"]:
            raise ValueError("checkpoint reload weight digest mismatch")
        scores = {}
        for name in names:  # the outer week is predicted only after the fit is final
            x, _, mask, origins = arrays[name]
            outputs = []
            for i in range(0, len(x), batch):
                expected, actual = model(x[i:i+batch], training=False), restored(x[i:i+batch], training=False)
                if not actual.device.endswith("/device:"+placement[1:]):
                    raise ValueError("prediction device mismatch")
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                outputs.append(np.asarray(actual))
            outputs = np.concatenate(outputs) if outputs else np.zeros((0, len(offsets)))
            for j, offset in enumerate(offsets):
                if not data["head_trainable"][j]:
                    continue
                selected = mask[:, j]
                head_origins, current, future = observations(data, cfg, name, column=j)
                if not np.array_equal(origins[selected], head_origins):
                    raise ValueError("materialized head rows differ from C1 head support")
                predicted = (cell.price_prediction(current, outputs[selected, j], data["target_center"][j],
                                                   data["target_scale"][j]) if len(current) else np.zeros(0))
                prefix = name if j == len(offsets) - 1 else f"{name}_h{offset}"
                scores.update({f"{prefix}_{key}": value for key, value in
                    dict(origins=head_origins, current=current, future=future, prediction=predicted).items()})
    np.savez(root / "scores.tmp.npz", **scores)
    (root / "scores.tmp.npz").replace(root / "scores.npz")
    return dict(training=audit, checkpoint_reload_verified=True, parameters=model.count_params(),
        effective_config=effective, artifact_directory=str(root), feature_scaler_sha256=data["scaler"].sha256(),
        target_center=data["target_center"].tolist(), target_scale=data["target_scale"].tolist(),
        coverage_complete=data["parts"].coverage_complete, history_start=c1.iso(data["parts"].history_start),
        inner_start=c1.iso(data["parts"].inner_start), fit_cutoff=c1.iso(data["parts"].fit_cutoff),
        scientific_scope="RETROSPECTIVE_TRAIN_INNER_CONDITIONAL_SOURCE_AVAILABILITY",
        search_space_scope="ENDPOINT_AND_TRAJECTORY", output_offsets=list(offsets),
        head_status=list(data["head_status"]), fit_seconds=fit_seconds, placement=placement_report,
        rows={name: int(len(arrays[name][3])) for name in names}, device=placement)


def effective_config_sha256(config):
    """Lane E definition: canonical JSON of request.config, UTF-8, SHA-256."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def receipt_header(cfg, request, digest, attempt, metadata, elapsed):
    """Per-candidate receipt fields known after the fit (rows added by scoring)."""
    effective, audit = request["config"], metadata["training"]
    genome = {k: effective[k] for k in search._GENOME_KEYS}
    offsets, h = metadata["output_offsets"], cfg["horizon_hours"]
    heads = [dict(offset_hours=o, head_role="ENDPOINT" if o == h else "INTERMEDIATE", status=st,
                  train_target_center=metadata["target_center"][j] if st == "TRAINED" else None,
                  train_target_scale=metadata["target_scale"][j] if st == "TRAINED" else None,
                  loss_weight=1. if st == "TRAINED" else 0.)
             for j, (o, st) in enumerate(zip(offsets, metadata["head_status"]))]
    rule = audit["early_stop"]
    return dict(
        schema_version=RECEIPT_SCHEMA, experiment=request["context"]["experiment"],
        fold=request["context"]["fold"], candidate_identity_sha256=digest,
        genotype_canonical_sha256=search.effective_identity(genome),
        effective_config_sha256=effective_config_sha256(effective),
        cell_config_sha256=c1.canonical_digest(cfg), horizon_hours=h,
        output_mode=effective["output_mode"], delta_x_hours=effective["delta_x"],
        delta_y_hours=effective["delta_y"], history_hours=effective["history_hours"],
        samples=effective["samples"], final_steps=effective["final_steps"], role=cfg["role"],
        heads=heads, stopping=dict(rule_sha256=rule["sha256"], monitor=rule["rule"]["monitor"],
            max_epochs=rule["rule"]["max_epochs"], patience=rule["rule"]["patience"],
            min_delta=rule["rule"]["min_delta"], restore=rule["rule"]["restore"],
            plan_conformant=all(cfg["training"][k] == v for k, v in PLAN_STOPPING.items()),
            epochs_completed=audit["epochs_completed"], best_epoch=audit["best_epoch"],
            stop_reason=audit["stop_reason"], restored_weights_sha256=audit["restored_weights_sha256"]),
        seed=cfg["training"]["seed"],
        cost=dict(elapsed_seconds=elapsed, fit_seconds=metadata["fit_seconds"],
                  parameters=metadata["parameters"], observed_updates=audit["observed_updates"],
                  train_rows=metadata["rows"]["train"], inner_rows=metadata["rows"]["inner"],
                  materialized_bytes=metadata["materialized_bytes"],
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  device=metadata["device"]),
        attempt=attempt, placement=metadata["placement"], rows=[])


def origins_sha256(origins):
    """Lane C support digest: canonical sha256 of sorted epoch-second origins."""
    return c1.canonical_digest(sorted(int(t) for t in origins))


def _row(header, cfg, offset, split, n, status, group=None, digest=None, support_sha256=None):
    sums = ("sum_abs_error", "sum_squared_error", "naive_sum_abs_error", "naive_sum_squared_error")
    values = [group[k] for k in sums] if group is not None else [None] * 4
    return dict(schema_version=RECEIPT_SCHEMA, experiment=header["experiment"],
        candidate_identity_sha256=header["candidate_identity_sha256"], fold=header["fold"],
        horizon_hours=cfg["horizon_hours"], head_offset_hours=offset,
        head_role="ENDPOINT" if offset == cfg["horizon_hours"] else "INTERMEDIATE",
        comparison=offset == cfg["horizon_hours"], role=cfg["role"], c2_role=cfg["role"].lower()+"_origin",
        population="S", split=split.upper(), metric="ABS_AND_SQUARED_ERROR_SUMS", unit=cfg["unit"],
        scale="ORIGINAL_PRICE", n=n, sum_abs_error=values[0], sum_sq_error=values[1],
        naive_sum_abs_error=values[2], naive_sum_sq_error=values[3], c2_receipt_sha256=digest,
        status=status, support_sha256=support_sha256)


def _c2(data, cfg, root, run_id, checkpoint, name, column, rows, plan_raw, pop, reuse, protocol, split_id=None):
    """Native C2 scoring of one head/split on its own C1 S (both forecast and naive)."""
    offset = _offsets(data, cfg)[column]
    stem = name if offset == cfg["horizon_hours"] else f"{name}.h{offset}"
    plan = c1.prepare_support_plan(plan_raw)
    origins, current, future, prediction = rows
    if prediction.shape != current.shape or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError("invalid full-support predictions")
    forecasts = [dict(key, target_timestamp=c1.iso(cell.stamp(t+offset*3600)),
        price_origin=float(a), price_target=float(b), prediction=float(p), unit=cfg["unit"],
        scale="ORIGINAL_PRICE", checkpoint=checkpoint)
        for key, t, a, b, p in zip(pop["keys"], origins, current, future, prediction)]
    document = dict(schema_version=c2.INPUT_SCHEMA_VERSION, opportunity_population=pop,
        forecasts=forecasts, forecast_failures=[], usage_receipts=[], contract=dict(
        roles=[cfg["role"].lower()+"_origin"], horizons_hours=[offset],
        calendar_id=bm.calendar_identity(plan), protocol_id=protocol, config_id=run_id, unit=cfg["unit"], scale="ORIGINAL_PRICE"))
    stored = None
    if reuse:
        if document != json.loads((root / f"{stem}.input.json").read_text()):
            raise ValueError("retained metric input differs from checkpoint score rows")
        stored = json.loads((root / f"{stem}.c2.json").read_text())
    result = bm.score_business_metrics(plan, document, asset=cfg["asset"], split_id=split_id or name.upper(),
                                       sealed_at=pop["sealed_at"], stored_result=stored)
    group = next(g for g in result["groups"] if g["population_kind"] == "S" and g["role"] != "ALL_ROLES")
    if group["n"] != len(origins) or group["coverage"] != 1:
        raise ValueError("C2 receipt changed the head's full C1 support")
    if not reuse:
        cell.atomic_json(root / f"{stem}.input.json", document)
        cell.atomic_json(root / f"{stem}.c2.json", result)
    return group, root / f"{stem}.c2.json"


def score_receipt(data, cfg, frozen, root, run_id, metadata, reuse=False):
    """C2 owns arithmetic. Endpoint TRAIN/inner on sealed S drives fitness.

    Intermediate heads and (weekly units) the outer VALIDATION week are
    diagnostics with their own C1 S and naive; they never enter fitness.
    """
    receipt = dict(deepcopy(frozen["support"]), status="ok", error_sums={}, metadata=deepcopy(metadata))
    receipt["metadata"].update(endpoint_metrics={}, c2_receipts={})
    header = receipt["metadata"].get("candidate_receipt")  # None for retained v1 terminals
    offsets, checkpoint = _offsets(data, cfg), metadata["training"]["restored_weights_sha256"]
    protocol, outer = frozen["schema"], frozen.get("outer")
    if header is None and (len(offsets) != 1 or outer is not None):
        raise ValueError("trajectory or outer receipt requires a candidate receipt header")
    if outer is not None:
        receipt["metadata"]["outer_metrics"] = {}
    table = []
    with np.load(root / "scores.npz", allow_pickle=False) as scores:
        for name in SPLITS + (("validation",) if outer is not None else ()):
            sealed_split = outer if name == "validation" else frozen["splits"][name]
            split_id = outer["split_id"] if name == "validation" else None
            origins, current, future = observations(data, cfg, name)
            for key, actual in dict(origins=origins, current=current, future=future).items():
                if not np.array_equal(scores[name+"_"+key], actual):
                    raise ValueError("retained scores differ from full source support")
            group, path = _c2(data, cfg, root, run_id, checkpoint, name, -1,
                              (origins, current, future, scores[name+"_prediction"]),
                              sealed_split["plan"], sealed_split["population"], reuse, protocol, split_id)
            metrics = {k: group[k] for k in ("n", "mae", "mse", "naive_mae", "naive_mse", "skill_mae", "unit", "scale")}
            if name == "validation":
                if group["n"] != outer["count"] or group["naive_sum_abs_error"] != outer["naive_sum"]:
                    raise ValueError("C2 receipt changed the sealed outer-week support")
                receipt["metadata"]["outer_metrics"] = metrics
            else:
                if group["n"] != receipt["counts"][name] or group["naive_sum_abs_error"] != receipt["naive_sums"][name]:
                    raise ValueError("C2 receipt changed frozen full support")
                receipt["error_sums"][name] = group["sum_abs_error"]
                receipt["metadata"]["endpoint_metrics"][name] = metrics
            receipt["metadata"]["c2_receipts"][name] = dict(path=str(path), sha256=cell.file_hash(path))
            if header is None:
                continue
            rows = {offsets[-1]: _row(header, cfg, offsets[-1], name, group["n"], "SCORED", group,
                                      cell.file_hash(path), origins_sha256(origins))}
            part = getattr(data["parts"], PART[name])
            for j, offset in enumerate(offsets[:-1]):
                prefix = f"{name}_h{offset}"
                if data["head_status"][j] != "TRAINED":
                    raw = [int(t.timestamp()) for t, m in zip(part.origins, part.mask) if m[j]]
                    rows[offset] = _row(header, cfg, offset, name, len(raw), "HEAD_NOT_TRAINED",
                                        support_sha256=origins_sha256(raw))
                    continue
                head = tuple(scores[f"{prefix}_{k}"] for k in ("origins", "current", "future", "prediction"))
                for key, actual in zip(("origins", "current", "future"), observations(data, cfg, name, column=j)):
                    if not np.array_equal(scores[f"{prefix}_{key}"], actual):
                        raise ValueError("retained head scores differ from C1 head support")
                if not len(head[0]):
                    rows[offset] = _row(header, cfg, offset, name, 0, "NO_SPLIT_SUPPORT",
                                        support_sha256=origins_sha256([]))
                    continue
                stem = root / f"{name}.h{offset}.input.json"
                sealed = (json.loads(stem.read_text())["opportunity_population"]["sealed_at"] if reuse
                          else c1.iso(datetime.now(timezone.utc)))
                planned = (outer_plan(data, cfg, sealed, column=j) if name == "validation"
                           else split_plan(data, cfg, name, sealed, column=j))
                group, path = _c2(data, cfg, root, run_id, checkpoint, name, j, head,
                                  planned["plan"], planned["population"], reuse, protocol, split_id)
                rows[offset] = _row(header, cfg, offset, name, group["n"], "SCORED", group,
                                    cell.file_hash(path), origins_sha256(head[0]))
            table.extend(rows[o] for o in offsets)
    if header is not None:
        header["rows"] = table
    receipt["fitness"] = search._score(receipt, frozen["support"])[0]
    return receipt


def artifact_names(receipt):
    """Fixed endpoint artifacts plus one input/C2 pair per scored intermediate head/split."""
    names = set(ARTIFACTS)
    for row in receipt["metadata"].get("candidate_receipt", {}).get("rows", []):
        if row["status"] == "SCORED":
            stem = row["split"].lower() + ("" if row["head_role"] == "ENDPOINT" else f".h{row['head_offset_hours']}")
            names |= {stem + ".input.json", stem + ".c2.json"}
    return names


def retained(root, expected, data, cfg, frozen):
    state = json.loads((root / "STATUS.json").read_text())
    if state["identity"] != expected:
        raise ValueError("retained identity mismatch")
    if state["state"] != "COMPLETED":
        raise ValueError("retained candidate cannot retry: "+state["state"])
    if cell.file_hash(root / "RESULT.json") != state["result_sha256"]:
        raise ValueError("result digest mismatch")
    result = json.loads((root / "RESULT.json").read_text())
    if result["identity"] != expected or set(result["artifacts"]) != artifact_names(result["receipt"]):
        raise ValueError("result identity or artifact digest set mismatch")
    for name, digest in result["artifacts"].items():
        if cell.file_hash(root / name) != digest:
            raise ValueError(name+" digest mismatch")
    receipt = score_receipt(data, cfg, frozen, root, root.name.split(".")[0], result["receipt"]["metadata"], reuse=True)
    if receipt != result["receipt"]:
        raise ValueError("receipt differs from fresh C2 rescore")
    return receipt


def evaluate(request):
    """Shared-adapter callback. Generation/attempt changes never authorize a refit.

    A verified COMPLETED attempt of the identity is adopted. Otherwise the
    requested physical attempt runs once; attempt k>1 requires the exact
    terminal (FAILED or lock-free RUNNING = INTERRUPTED) status digest of
    attempt k-1. FAILED/RUNNING/orphaned directories never retry in place.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("historical evaluator requires the main thread; use i6n_evaluator_command")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("existing wall timer; use an isolated evaluator process")
    cfg, frozen = load_request(request)
    if request["context"]["schema"] != SCHEMA:
        raise ValueError("LEGACY_CONTEXT_READ_ONLY: v1 contexts are verified, never fitted")
    attempt = normalize_attempt(request.get("attempt"))
    expected = identity(cfg, request)
    digest = c1.canonical_digest(expected)
    roots = [attempt_root(request["context"], digest, k) for k in range(1, MAX_ATTEMPTS + 1)]
    completed = [r for r in roots if _state(r) == "COMPLETED"]
    if completed:
        root = completed[0]
    else:
        root = roots[attempt["index"] - 1]
        if attempt["index"] > 1 and not (root / "STATUS.json").exists():
            _check_parent(roots[attempt["index"] - 2], attempt, request)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / "STATUS.json").exists() and any(p.name != "RUN.lock" for p in root.iterdir()):
            raise ValueError("orphan candidate artifacts prohibit fitting")
        started = time.monotonic()
        state = dict(state="RUNNING", identity=expected, attempt=attempt, elapsed_seconds=0., eta_seconds=None)
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
            data = cell.prepare(cfg, trajectory=True)
            if support(data, cfg) != frozen["support"]:
                raise ValueError("candidate full support differs; intersection/drop forbidden")
            if prior:
                return retained(root, expected, data, cfg, frozen)
            metadata = fit_candidate(data, cfg, request["config"], root, progress, "outer" in frozen)
            metadata["materialized_bytes"] = data["materialized_bytes"]
            metadata["candidate_receipt"] = receipt_header(cfg, request, digest, attempt, metadata,
                                                           time.monotonic()-started)
            receipt = score_receipt(data, cfg, frozen, root, digest, metadata)
            load_request(request)
            if identity(cfg, request) != expected:
                raise ValueError("evaluator identity changed during fit")
            result = dict(identity=expected, receipt=receipt,
                          artifacts={n: cell.file_hash(root / n) for n in sorted(artifact_names(receipt))})
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
    parser.add_argument("--experiment")
    parser.add_argument("--fold")
    parser.add_argument("--score-outer", action="store_true",
                        help="weekly-refit unit: also seal and report the outer week (diagnostic only)")
    args = parser.parse_args()
    if args.command == "freeze" and None in (args.config, args.output, args.experiment, args.fold):
        parser.error("freeze requires --config, --output, --experiment and --fold")
    result = (freeze(args.config, args.output, args.experiment, args.fold, args.score_outer)
              if args.command == "freeze"
              else evaluate(json.load(sys.stdin)))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
