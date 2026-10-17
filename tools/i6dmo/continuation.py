"""Explicit finite successor: heads -> joint window/core grid -> cadence -> decision.

Uses the resolution batch's existing admitted worker, outbox, mirrors and timers.
2024 is checkpoint-selection/development data, never an independent holdout.
"""
import math
from pathlib import Path
import shutil

from tools.i6dmo import resolution_batch as B

PROTOCOL = "i6dmo.predictive_continuation.v1"
CORES = ([2], [4], [4, 2], [8, 4])
HEADS = ("conv", "dense", "lstm")
CADENCES = ("quarterly", "monthly", "weekly")


def cells(periods):
    out = []
    def add(stage, name, group, **kwargs):
        out.append(dict(id=name, stage=stage, group=group, period=None,
                        horizons=B.SHORT if group == "short" else B.LONG, **kwargs))
    for head in HEADS:
        for group in ("short", "long"):
            for spacing in (2, 4):
                add("heads", f"dmo_{group}_ds{spacing}_{head}", group, kind="conv",
                    head=head, core=[8, 4], slots=13, spacing=spacing)
    for slots in B.LENGTHS:
        for kind in ("linear", "dlinear", "conv"):
            cell = dict(id=f"single72_n{slots}_{kind}", stage="single72", group="long",
                        period=None, horizons=[72], kind=kind, head="BEST_HEAD", core=[8, 4],
                        slots=slots, spacing="BEST_SPACING")
            out.append(cell)
    for core in CORES:
        for slots in B.LENGTHS:
            for spacing in (2, 4):
                for group in ("short", "long"):
                    for head in HEADS:
                        out.append(dict(id=f"joint_{group}_s{spacing}_n{slots}_c"+"_".join(map(str, core))+"_"+head,
                            stage="joint", group=group, period=None, kind="conv", head=head,
                            core=list(core), slots=slots, spacing=spacing, horizons=[4] if group == "short" else [72]))
    for group in ("short", "long"):
        add("transfer", "transfer_"+group, group, kind="BEST_REPRESENTATIVE", head="SELECTED",
            core="SELECTED", slots="SELECTED", spacing="SELECTED")
    for cadence in CADENCES:
        for group in ("short", "long"):
            for period in periods[cadence]:
                if not isinstance(period.get("id"), str) or any(ch in period["id"] for ch in ("/", "\\", "\x00")):
                    raise ValueError("CONTINUATION_PERIOD_ID_INVALID")
                out.append(dict(id=f"cad_{group}_{cadence}_{period['id']}", stage="cadence", group=group,
                    cadence=cadence, period=period, horizons=B.SHORT if group == "short" else B.LONG,
                    kind="TRANSFER", head="TRANSFER", core="TRANSFER", slots="TRANSFER", spacing="TRANSFER"))
    return out


def score(rows):
    active = [r["relative_error"] for r in rows if r["n"]]
    if not active or any(v is None or not math.isfinite(v) for v in active):
        raise ValueError("CONTINUATION_SCORE_UNAVAILABLE")
    return math.fsum(active)/len(active)


def best(spec, results, predicate):
    candidates = [c for c in spec["cells"] if predicate(c)]
    if not candidates or any(c["id"] not in results or results[c["id"]]["state"] != "COMPLETE"
                             for c in candidates):
        return None
    rows = [results[c["id"]] for c in candidates]
    reference = rows[0]["metrics"]["VALIDATION"]
    for r in rows[1:]:
        other = r["metrics"]["VALIDATION"]
        if [(x["horizon"], x["n"], x["naive_mae"], x["pairs_sha256"]) for x in reference] != \
                [(x["horizon"], x["n"], x["naive_mae"], x["pairs_sha256"]) for x in other]:
            raise ValueError("CONTINUATION_UNPAIRED_RANKING")
    minimum = min(score(r["metrics"]["VALIDATION"]) for r in rows)
    tied = [r for r in rows if score(r["metrics"]["VALIDATION"]) <= minimum+1e-8]
    return min(tied, key=lambda r: (r["parameters"], spec["cells"].index(r["cell"])))


def aggregate(spec, results, group, cadence):
    import hashlib
    import numpy as np
    expected = [c for c in spec["cells"] if c["stage"] == "cadence" and c["group"] == group
                and c["cadence"] == cadence]
    if not expected or any(c["id"] not in results or results[c["id"]]["state"] not in
                          ("COMPLETE", "SKIPPED_NO_EVAL_ROWS") for c in expected):
        return None
    measured = [c for c in expected if results[c["id"]]["state"] == "COMPLETE"]
    rows = []
    hs = B.SHORT if group == "short" else B.LONG
    # Reported sums preserve full-year pooling; paired origins are verified from retained arrays.
    root = Path(spec.get("gpu_root", ""))
    if spec.get("aggregation_root"):
        root = Path(spec["aggregation_root"])
    for j, h in enumerate(hs):
        pieces = [results[c["id"]]["metrics"]["VALIDATION"][j] for c in measured]
        n = sum(r["n"] for r in pieces)
        if not n:
            return None
        se = math.fsum(r["sum_abs_error"] for r in pieces)
        sn = math.fsum(r["sum_abs_naive_error"] for r in pieces)
        origins = []
        for c in measured:
            with np.load(root/"results"/c["id"]/"VALIDATION.npz", allow_pickle=False) as z:
                origins.extend(z["origin_ts"][z["mask"][:, j]].tolist())
        if len(origins) != n or len(set(origins)) != n:
            raise ValueError("CONTINUATION_PERIOD_OVERLAP")
        rows.append(dict(horizon=h, n=n, mae=se/n, naive_mae=sn/n,
                         relative_error=(se-sn)/sn, sum_abs_error=se, sum_abs_naive_error=sn,
                         pairs_sha256=B.digest(sorted(origins))))
    static = results.get("transfer_"+group)
    if not static or static["state"] != "COMPLETE":
        return None
    for a, b in zip(static["metrics"]["VALIDATION"], rows):
        if (a["n"], a["pairs_sha256"]) != (b["n"], b["pairs_sha256"]) or \
                not math.isclose(a["naive_mae"], b["naive_mae"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("CONTINUATION_CADENCE_ANNUAL_SUPPORT_DIFFERS")
    return rows


def resolve(spec, cell, results):
    """Resolve only completed predecessors; no chat, implicit defaults or partial rankings."""
    stage, group = cell["stage"], cell["group"]
    if stage == "heads":
        return dict(kind=cell["kind"], head=cell["head"], core=cell["core"],
                    slots=cell["slots"], spacing=cell["spacing"])
    if stage == "joint":
        return dict(kind=cell["kind"], head=cell["head"], core=cell["core"],
                    slots=cell["slots"], spacing=cell["spacing"])
    if stage == "single72":
        winner = best(spec, results, lambda c: c["stage"] == "heads" and c["group"] == group)
        if winner is None:
            return dict(disposition="BLOCKED_DEPENDENCY", reason="complete head comparison required")
        config = dict(winner["resolved"])
        config.update(kind=cell["kind"], core=cell["core"], slots=cell["slots"])
        return config
    if stage == "transfer":
        complete_scopes = []
        for scope in (("joint", "single72") if group == "long" else ("joint",)):
            candidate = best(spec, results, lambda c: c["group"] == group and c["stage"] == scope)
            if candidate is not None:
                complete_scopes.append(candidate["cell"]["id"])
        winner = best(spec, results, lambda c: c["id"] in complete_scopes)
        if winner is None:
            return dict(disposition="BLOCKED_DEPENDENCY", reason="complete representative grid required")
        return dict(winner["resolved"], representative_source=winner["cell"]["id"],
                    selection_scope="COMPLETE_REPRESENTATIVE_SUBGRIDS_ONLY")
    if stage == "cadence":
        transfer = results.get("transfer_"+group)
        if not transfer or transfer["state"] != "COMPLETE":
            return dict(disposition="BLOCKED_DEPENDENCY", reason="complete static transfer required")
        cadence = cell["cadence"]
        if cadence != "quarterly":
            previous = "quarterly" if cadence == "monthly" else "monthly"
            previous_rows = aggregate(spec, results, group, previous)
            reference = transfer["metrics"]["VALIDATION"] if cadence == "monthly" else aggregate(spec, results, group, "quarterly")
            if previous_rows is None or reference is None:
                states = [results[c["id"]]["state"] for c in spec["cells"] if c["stage"] == "cadence"
                          and c["group"] == group and c["cadence"] == previous and c["id"] in results]
                if states and all(s == "SKIPPED_NOT_BENEFICIAL" for s in states):
                    return dict(disposition="SKIPPED_NOT_BENEFICIAL", reason="earlier cadence not beneficial")
                return dict(disposition="BLOCKED_DEPENDENCY", reason="complete prior annual cadence required")
            if score(previous_rows) >= score(reference)-1e-8:
                return dict(disposition="SKIPPED_NOT_BENEFICIAL", reason=previous+" did not improve the preceding annual score")
        return dict(transfer["resolved"])
    raise ValueError("CONTINUATION_STAGE_UNKNOWN")


def next_cell(spec, results):
    return next((c for c in spec["cells"] if c["id"] not in results), None)


def measured_metrics(arrays, prediction, horizons):
    import numpy as np
    rows = []
    for j, h in enumerate(horizons):
        m = arrays["mask"][:, j]
        n = int(m.sum())
        if not n:
            rows.append(dict(horizon=h, n=0, mae=None, naive_mae=None, relative_error=None,
                predicted_movement_mae=None, sum_abs_error=0., sum_abs_naive_error=0.,
                pairs_sha256=B.digest([]), disposition="NO_VALID_TARGET_IN_PERIOD"))
            continue
        y, p, origin = arrays["target_price"][m, j], prediction[m, j], arrays["origin_price"][m]
        if not np.isfinite(p).all():
            raise ValueError("CONTINUATION_NONFINITE_PREDICTION")
        se = math.fsum(np.abs(p-y).tolist())
        sn = math.fsum(np.abs(origin-y).tolist())
        if sn <= 0:
            raise ValueError("CONTINUATION_ZERO_NAIVE")
        rows.append(dict(horizon=h, n=n, mae=se/n, naive_mae=sn/n, relative_error=(se-sn)/sn,
            predicted_movement_mae=math.fsum(np.abs(p-origin).tolist())/n,
            sum_abs_error=se, sum_abs_naive_error=sn, pairs_sha256=B.digest(arrays["origin_ts"][m].tolist())))
    return rows


def fit_cell(spec, cell, results, folder):
    from tools.i6dmo import train as T, continuation_models as M, continuation_data as D
    from predictor_plugins.modular_temporal import weights_hash
    import numpy as np
    import tensorflow as tf
    import importlib.metadata
    resolved = resolve(spec, cell, results)
    if "disposition" in resolved:
        return dict(batch_id=spec["batch_id"], cell=cell, state=resolved["disposition"],
                    reason=resolved["reason"], files={})
    if tf.__version__ != "2.21.0" or importlib.metadata.version("keras") != "3.13.2" \
            or not tf.config.list_physical_devices("GPU"):
        raise ValueError("CONTINUATION_RUNTIME_OR_GPU_REQUIRED")
    try:
        arrays, scales, meta = D.load(spec["gpu_data"], cell["group"], resolved["spacing"], resolved["slots"],
            horizons=cell["horizons"], input_dir=spec["gpu_input"], period=cell["period"])
    except D.ContinuationDataSkip as exc:
        return dict(batch_id=spec["batch_id"], cell=cell, state="SKIPPED_NO_EVAL_ROWS", reason=str(exc), files={})
    if set(arrays) != {"FIT", "VALIDATION"} or meta["manifest_sha256"] != spec["population_digest"]:
        raise ValueError("CONTINUATION_FULL_TRAIN_VALIDATION_REQUIRED")
    config = M.build_config(meta["features"], resolved["slots"], resolved["spacing"], cell["horizons"],
                           kind=resolved["kind"], seed=B.SEED, head=resolved["head"], core=resolved["core"])
    key = B.digest(dict(model=config, population=meta, training=B.TRAINING, protocol=PROTOCOL))
    # Identical logical cells share authenticated immutable fits, not another seed run.
    donor = next((r for r in results.values() if r["state"] == "COMPLETE" and r.get("fit_key") == key), None)
    if donor is not None:
        source = Path(spec["gpu_root"])/"results"/donor["cell"]["id"]
        for name in donor["files"]:
            shutil.copy2(source/name, folder/name)
        original = B.read(folder/"CONFIG.json")
        B.put(folder/"CONFIG.json", dict(original, cell=cell, reused_from=donor["cell"]["id"]))
        files = {p.name: B.file_sha256(p) for p in folder.iterdir() if p.name in donor["files"]}
        return dict({k: v for k, v in donor.items() if k != "sha256"}, cell=cell,
                    reused_from=donor["cell"]["id"], training_seconds=0., files=files)
    bundle = M.build(config)
    prepared = {}
    for name, a in arrays.items():
        x, y, mask = T.transform(a, scales)
        prepared[name] = ([x, a["offsets_hours"].astype("float32"), a["closure_crossed"].astype("float32")], y, mask)
    B.put(folder/"CONFIG.json", dict(cell=cell, resolved=resolved, model=config, training=B.TRAINING,
        population=meta, protocol=PROTOCOL, monitor_splits=["TRAIN", "VALIDATION"], test_read=False))
    def update(p):
        p = dict(p)
        p["train_loss"] = p.pop("fit_loss")
        p["validation_loss"] = p.pop("inner_loss")
        B.progress(spec["gpu_root"], state="RUNNING", cell=cell["id"], stage=cell["stage"], **p)
    B.progress(spec["gpu_root"], state="RUNNING", cell=cell["id"], stage=cell["stage"], epoch=0)
    training = T.fit_dmo(bundle, prepared["FIT"], prepared["VALIDATION"], B.TRAINING, B.SEED,
        ids=(arrays["FIT"]["row_ids"], arrays["VALIDATION"]["row_ids"]), record_device="gpu", progress=update)
    training["split_roles"] = dict(fit="TRAIN_ALL_FOUR_YEARS", inner="VALIDATION_EVALUATION_PERIOD",
        extra_holdout=False, qualification="DEVELOPMENT_CHECKPOINT_TUNING_NOT_INDEPENDENT_OOS")
    bundle.forecast_model.save_weights(folder/"model.weights.h5")
    bundle.forecast_model.save(folder/"forecast_model.keras")
    fresh = M.build(config)
    fresh.forecast_model.load_weights(folder/"model.weights.h5")
    if weights_hash(fresh.forecast_model) != training["restored_weights_sha256"]:
        raise ValueError("CONTINUATION_RELOAD_MISMATCH")
    rows = {}
    for name, a in arrays.items():
        p = T.monitor_forward(fresh.forecast_model, prepared[name][0], B.TRAINING["batch_size"])
        price = T.to_price(p, a, scales)
        rows[name] = measured_metrics(a, price, cell["horizons"])
        np.savez_compressed(folder/(name+".npz"), prediction=price, origin_ts=a["origin_ts"],
            origin_price=a["origin_price"], target_price=a["target_price"], mask=a["mask"])
    B.put(folder/"TRAINING.json", training)
    peak = None
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            path = Path('/sys/fs/cgroup')/line[3:].lstrip('/')/'memory.peak'
            if path.exists():
                peak = int(path.read_text())
    files = {p.name: B.file_sha256(p) for p in folder.iterdir()
             if p.is_file() and p.name not in ("STARTED.json", "RESULT.json")}
    return dict(batch_id=spec["batch_id"], cell=cell, state="COMPLETE", resolved=resolved,
        fit_key=key, actual_spacing=resolved["spacing"], seed=B.SEED,
        parameters=int(bundle.forecast_model.count_params()), training_seconds=training["fit_seconds"],
        epochs=training["epochs_completed"], best_epoch=training["best_epoch"], stop_reason=training["stop_reason"],
        metrics=rows, population=meta, initial_weights_sha256=training["initial_weights_sha256"],
        restored_weights_sha256=training["restored_weights_sha256"], files=files, cgroup_peak_bytes=peak,
        tensorflow=tf.__version__, keras=importlib.metadata.version("keras"), test_read=False)


def selections(spec, results, artifact_root):
    context = dict(spec, aggregation_root=str(artifact_root))
    chosen = {}
    for group in ("short", "long"):
        baseline = results.get("transfer_"+group)
        if not baseline or baseline["state"] != "COMPLETE":
            chosen[group] = dict(state="INCOMPLETE")
            continue
        current = dict(unit_ids=[baseline["cell"]["id"]], cadence="static", metrics=baseline["metrics"]["VALIDATION"])
        for cadence in CADENCES:
            rows = aggregate(context, results, group, cadence)
            if rows is not None and score(rows) < score(current["metrics"])-1e-8:
                current = dict(unit_ids=[c["id"] for c in spec["cells"] if c["stage"] == "cadence"
                    and c["group"] == group and c["cadence"] == cadence and results[c["id"]]["state"] == "COMPLETE"],
                    cadence=cadence, metrics=rows)
        chosen[group] = dict(current, state="COMPLETE")
    return chosen


def decision_reports(spec, selection, decision):
    rows = []
    for group, data in selection.items():
        for r in data.get("metrics", []):
            for name, key, unit in (("Annual_MAE", "mae", "USD_per_EUR"),
                ("Annual_Naive_MAE", "naive_mae", "USD_per_EUR"),
                ("Annual_Relative_Error", "relative_error", "ratio"), ("Annual_Pairs", "n", "count")):
                rows.append(dict(metric=group+"_"+name, value=r[key], unit=unit,
                                 horizon=r["horizon"], split="validation"))
    for name, value in (decision.get("metrics") or {}).items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            rows.append(dict(metric="Strategy_"+name, value=value, unit="as_recorded",
                             horizon=None, split="validation"))
    rows.append(dict(metric="Pair_Eligible", value=int(decision.get("disposition") == "PAIR_ELIGIBLE"),
                     unit="bool", horizon=None, split="validation"))
    return [dict(experiment_key="i6dmo:"+spec["batch_id"]+":final_decision",
        experiment_set_key="i6dmo:"+spec["batch_id"], actor="predictor", lake="LOCAL_RETAINED",
        lineage="UNVERIFIED", config_sha256=B.digest(dict(selection=selection, decision=decision)),
        code_commit=spec["code_commit"], project="predictor", phase="I6DMO_CONTINUATION",
        tags=dict(protocol=PROTOCOL, test_read="false", evaluation="RETROSPECTIVE_DEVELOPMENT_TUNING",
                  disposition=decision.get("disposition", "INCOMPLETE")), metrics=rows)]


def bootstrap(config_path):
    """Prepare ALL phases and calendar periods, then arm the shared finite worker."""
    from tools.i6dmo import continuation_data as D, raw_long_campaign as L, context_bridge_campaign as CB
    from tools.i6dmo import campaign as C
    config = B.read(config_path)
    required = {"root", "source_deployment", "input_dir"}
    if not required.issubset(config) or set(config)-required-{"strategy_source"}:
        raise ValueError("CONTINUATION_PRIVATE_CONFIG")
    root = Path(config["root"])
    if any(not isinstance(v, str) or not Path(v).is_absolute() for v in config.values()):
        raise ValueError("CONTINUATION_ABSOLUTE_PATH_REQUIRED")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    specpath = root/"BATCH.private.json"
    if specpath.exists():
        spec = B.read(specpath)
        if spec["launch_config"] != config or spec["protocol"] != PROTOCOL:
            raise ValueError("CONTINUATION_CONFIG_CHANGED")
        role = B.read(spec["deployment"])["roles"]["preferred"]
        L._remote(role, ["mkdir", "-p", spec["gpu_root"]])
        B.command(["rsync", "-a", "--checksum", str(specpath), role["ssh_alias"]+":"+spec["gpu_spec"]])
        return B.install(specpath)
    if B.command(["git", "-C", str(B.REPO), "status", "--porcelain"]).strip():
        raise ValueError("CONTINUATION_SOURCE_NOT_FROZEN")
    deployment = B.read(config["source_deployment"])
    deployment["storage"] = dict(estimate_bytes=6*1024**3, reserve_bytes=5*1024**3)
    CB.storage_preflight(deployment, root)
    code = C._git_head()
    files = {str(p.relative_to(B.REPO)): B.file_sha256(p) for d in ("tools/i6dmo", "tools/i6n_contracts",
        "tools/i6n_archive", "tools/i6n_swarm", "olap/store/src", "predictor_plugins/modular_temporal")
        for p in sorted((B.REPO/d).rglob("*.py"))}
    local = Path(deployment["roles"]["coordinator"]["experiment_storage_root"])
    pop = local/"i6dmo"/root.name/"data"
    manifest = D.prepare(config["input_dir"], pop, B.REPO/"docs/i6n/direct_multi_output_static/EXPERIMENT.short.json")
    periods = {c: D.periods(pop, c) for c in CADENCES}
    prepared = dict(launch_config=dict(input_dir=config["input_dir"]), population_dir=str(pop),
                    population_relpath=str(pop.relative_to(local)))
    deployment = L._provision(deployment, prepared, code, reviewed_files=files)
    dp = root/"DEPLOY.private.json"
    B.put(dp, deployment)
    role = deployment["roles"]["preferred"]
    gpu_root = "/var/tmp/i6dmo/"+root.name+"-gpu"
    name = "i6dmo-continuation-"+B.digest(dict(code=code, launch=config, population=manifest["manifest_sha256"]))[:12]
    gpu = next(g for g in role["gpus"] if g.get("heavy") and g["class"] == "rtx5090")
    spec = dict(protocol=PROTOCOL, batch_id=name, root=str(root), artifacts=str(local/"i6dmo"/root.name/"artifacts"),
        gpu_root=gpu_root, gpu_data=str(Path(role["experiment_storage_root"])/prepared["population_relpath"]),
        gpu_input=str(Path(role["data_root"])/Path(config["input_dir"]).relative_to(
            Path(deployment["roles"]["coordinator"]["data_root"]))), population_digest=manifest["manifest_sha256"],
        gpu_spec=gpu_root+"/BATCH.private.json", gpu_uuid=gpu["uuid"], cells=cells(periods), periods=periods,
        training=B.TRAINING, seed=B.SEED, code_files=files, code_commit=code, deployment=str(dp),
        deployment_sha256=B.file_sha256(dp), runtime=role["workdir"], launch_config=config)
    if config.get("strategy_source"):
        source = B.read(config["strategy_source"])["raw_pair"]
        pins = {k: source["strategy_"+k] for k in ("bars", "calendar")}
        for pin in pins.values():
            if not Path(pin["path"]).is_absolute() or B.file_sha256(pin["path"]) != pin["sha256"]:
                raise ValueError("CONTINUATION_STRATEGY_PIN_MISMATCH")
        spec["strategy_inputs"] = pins
        spec["strategy_source_sha256"] = B.file_sha256(config["strategy_source"])
    B.put(specpath, spec)
    L._remote(role, ["mkdir", "-p", gpu_root])
    B.command(["rsync", "-a", "--checksum", str(specpath), role["ssh_alias"]+":"+spec["gpu_spec"]])
    result = B.install(specpath)
    B.put(root/"LAUNCH.json", dict(**result, code_commit=code, logical_cells=len(spec["cells"]), test_open=False))
    return result
