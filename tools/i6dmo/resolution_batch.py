"""Finite sequential DS2/DS4 and h72 study; no agent monitoring or DOIN changes."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

from tools.i6n_swarm.common import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[2]
SEED = 6102026
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
LENGTHS = [6, 9, 13, 18, 24]
TRAINING = dict(batch_size=256, learning_rate=.001, weight_decay=.0001,
                patience=10, min_delta=1e-8, max_epochs=200)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    atomic_json(Path(path), value, mode=0o600)


def cells():
    out = [dict(id=f"{group}_ds{spacing}", group=group, slots=13, spacing=spacing,
                horizons=SHORT if group == "short" else LONG, kind="conv")
           for group in ("short", "long") for spacing in (2, 4)]
    out += [dict(id=f"h72_n{slots}_{kind}", group="long", slots=slots,
                 spacing="SELECTED_LONG", horizons=[72], kind=kind)
            for slots in LENGTHS for kind in ("linear", "dlinear", "conv")]
    return out


def verify_code(spec):
    for rel, sha in spec["code_files"].items():
        path = (REPO / rel).resolve()
        if not path.is_relative_to(REPO) or file_sha256(path) != sha:
            raise ValueError("BATCH_CODE_CHANGED:" + rel)
    if spec["cells"] != cells() or spec["seed"] != SEED or spec["training"] != TRAINING:
        raise ValueError("BATCH_PLAN_CHANGED")
    if "deployment_sha256" in spec and Path(spec["deployment"]).is_file() \
            and file_sha256(spec["deployment"]) != spec["deployment_sha256"]:
        raise ValueError("BATCH_DEPLOYMENT_CHANGED")


def terminal(root, cell, spec):
    path = Path(root) / "results" / cell["id"] / "RESULT.json"
    if not path.exists():
        return None
    d = read(path)
    body = {k: v for k, v in d.items() if k != "sha256"}
    if d.get("sha256") != digest(body) or d.get("cell") != cell or d.get("batch_id") != spec["batch_id"]:
        raise ValueError("BATCH_TERMINAL_IDENTITY:" + cell["id"])
    if d.get("state") == "COMPLETE":
        required = {"CONFIG.json", "TRAINING.json", "model.weights.h5", "FIT.npz", "INNER.npz", "VALIDATION.npz"}
        if set(d.get("files", {})) != required:
            raise ValueError("BATCH_TERMINAL_ARTIFACTS:" + cell["id"])
        if d.get("population", {}).get("manifest_sha256") != spec["population_digest"]:
            raise ValueError("BATCH_TERMINAL_POPULATION:" + cell["id"])
    for rel, sha in d.get("files", {}).items():
        p = (path.parent / rel).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_sha256(p) != sha:
            raise ValueError("BATCH_TERMINAL_FILE:" + cell["id"])
    return d


def selected_spacing(results):
    """Same frozen DEVELOPMENT rows; a missing/failed arm never wins."""
    scores = {}
    for spacing in (2, 4):
        d = results.get(f"long_ds{spacing}")
        if not d or d["state"] != "COMPLETE":
            return None
        rows = d["metrics"]["VALIDATION"]
        if [r["horizon"] for r in rows] != LONG:
            raise ValueError("BATCH_LONG_HEADS_MISSING")
        scores[spacing] = math.fsum(r["relative_error"] for r in rows) / len(LONG)
    a, b = (results[f"long_ds{s}"] for s in (2, 4))
    if a.get("initial_weights_sha256") != b.get("initial_weights_sha256"):
        raise ValueError("BATCH_INITIAL_WEIGHTS_DIFFER")
    for x, y in zip(a["metrics"]["VALIDATION"], b["metrics"]["VALIDATION"]):
        if (x["n"], x["naive_mae"], x["pairs_sha256"]) != (y["n"], y["naive_mae"], y["pairs_sha256"]):
            raise ValueError("BATCH_UNPAIRED_SELECTION")
    return 4 if scores[4] < scores[2] - 1e-8 else 2


def next_cell(spec, results):
    for cell in spec["cells"]:
        if cell["id"] in results:
            continue
        if cell["spacing"] == "SELECTED_LONG" and selected_spacing(results) is None:
            return None
        return cell
    return None


def metrics(arrays, price_pred, horizons):
    """Original-scale, globally pooled, per-horizon MAE and its own exact naive."""
    import numpy as np
    rows = []
    for j, h in enumerate(horizons):
        m = arrays["mask"][:, j]
        y = arrays["target_price"][m, j].astype("float64")
        p = price_pred[m, j].astype("float64")
        origin = arrays["origin_price"][m].astype("float64")
        if not len(y) or not np.isfinite(p).all():
            raise ValueError("BATCH_UNSUPPORTED_OR_NONFINITE_HEAD")
        mae = math.fsum(np.abs(p-y).tolist()) / len(y)
        naive = math.fsum(np.abs(origin-y).tolist()) / len(y)
        if naive <= 0:
            raise ValueError("BATCH_ZERO_NAIVE")
        rows.append(dict(horizon=h, n=len(y), mae=mae, naive_mae=naive,
                         relative_error=(mae-naive)/naive,
                         predicted_movement_mae=math.fsum(np.abs(p-origin).tolist())/len(y),
                         pairs_sha256=digest(arrays["origin_ts"][m].tolist())))
    return rows


def progress(root, **fields):
    put(Path(root) / "PROGRESS.json", dict(at=time.time(), **fields))


def fit_next(spec):
    """One admitted GPU process, one cell, one terminal. No in-process fit retries."""
    verify_code(spec)
    root = Path(spec["gpu_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {c["id"]: t for c in spec["cells"] if (t := terminal(root, c, spec))}
        cell = next_cell(spec, results)
        if cell is None:
            if len(results) < len(spec["cells"]):
                for blocked in spec["cells"]:
                    if blocked["id"] not in results:
                        result = dict(batch_id=spec["batch_id"], cell=blocked, state="BLOCKED_DEPENDENCY",
                                      reason="long DS2/DS4 comparison has a failed prerequisite", files={})
                        result["sha256"] = digest(result)
                        put(root/"results"/blocked["id"]/"RESULT.json", result)
            progress(root, state="GPU_DONE" if len(results) == len(spec["cells"]) else "BLOCKED_DEPENDENCY")
            return
        # No competing compute job on either shared-RAM GPU. No reset or process killing.
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=20)
        if out.stdout.strip():
            progress(root, state="WAIT_GPU", next_cell=cell["id"])
            return
        visible = subprocess.run(["nvidia-smi", "-i", spec["gpu_uuid"], "--query-gpu=name",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if visible.returncode or "5090" not in visible.stdout:
            progress(root, state="WAIT_GPU_HARDWARE", next_cell=cell["id"])
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = spec["gpu_uuid"]
        folder = root / "results" / cell["id"]
        folder.mkdir(parents=True, exist_ok=True)
        started = folder / "STARTED.json"
        if started.exists():
            # Recovered after power loss/timeout: preserve partials; never duplicate a fit.
            result = dict(batch_id=spec["batch_id"], cell=cell, state="FAILED_INTERRUPTED",
                          reason="previous attempt has no terminal; no automatic retraining", files={})
        else:
            put(started, dict(batch_id=spec["batch_id"], cell=cell, at=time.time()))
            try:
                result = fit_cell(spec, cell, results, folder)
            except Exception as exc:
                result = dict(batch_id=spec["batch_id"], cell=cell, state="FAILED",
                              reason=type(exc).__name__ + ":" + str(exc)[:400], files={})
        result["sha256"] = digest(result)
        put(folder / "RESULT.json", result)
        progress(root, state=result["state"], cell=cell["id"])


def fit_cell(spec, cell, results, folder):
    # train imports the pinned determinism initializer before TensorFlow.
    from tools.i6dmo import train as T, resolution_models as M, resolution_data as D
    from predictor_plugins.modular_temporal import weights_hash
    import numpy as np
    import tensorflow as tf
    import importlib.metadata
    if tf.__version__ != "2.21.0" or importlib.metadata.version("keras") != "3.13.2":
        raise ValueError("BATCH_RUNTIME_PIN")
    if not tf.config.list_physical_devices("GPU"):
        raise ValueError("BATCH_GPU_REQUIRED")
    spacing = selected_spacing(results) if cell["spacing"] == "SELECTED_LONG" else cell["spacing"]
    arrays, scales, meta = D.load(spec["gpu_data"], cell["group"], spacing, cell["slots"],
                                 horizons=cell["horizons"], input_dir=spec["gpu_input"])
    if meta["manifest_sha256"] != spec["population_digest"]:
        raise ValueError("BATCH_POPULATION_CHANGED")
    config = M.build_config(meta["features"], cell["slots"], spacing, cell["horizons"],
                            kind=cell["kind"], seed=SEED)
    bundle = M.build(config)
    params = int(bundle.forecast_model.count_params())
    prepared = {}
    for name, a in arrays.items():
        x, y, mask = T.transform(a, scales)
        prepared[name] = ([x, a["offsets_hours"].astype("float32"),
                          a["closure_crossed"].astype("float32")], y, mask)
    put(folder / "CONFIG.json", dict(cell=cell, spacing=spacing, model=config,
                                     training=TRAINING, population=meta))
    progress(spec["gpu_root"], state="RUNNING", cell=cell["id"], epoch=0)
    training = T.fit_dmo(bundle, prepared["FIT"], prepared["INNER"], TRAINING, SEED,
                        ids=(arrays["FIT"]["row_ids"], arrays["INNER"]["row_ids"]),
                        record_device="gpu", progress=lambda p: progress(spec["gpu_root"],
                            state="RUNNING", cell=cell["id"], **p))
    bundle.forecast_model.save_weights(folder / "model.weights.h5")
    fresh = M.build(config)
    fresh.forecast_model.load_weights(folder / "model.weights.h5")
    if weights_hash(fresh.forecast_model) != weights_hash(bundle.forecast_model):
        raise ValueError("BATCH_RELOAD_WEIGHTS")
    reports = {}
    for name, a in arrays.items():
        normalized = T.monitor_forward(fresh.forecast_model, prepared[name][0], TRAINING["batch_size"])
        pred = T.to_price(normalized, a, scales)
        reports[name] = metrics(a, pred, cell["horizons"])
        np.savez_compressed(folder / (name + ".npz"), prediction=pred, origin_ts=a["origin_ts"],
                            origin_price=a["origin_price"], target_price=a["target_price"], mask=a["mask"])
    put(folder / "TRAINING.json", training)
    peak = None
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            p = Path('/sys/fs/cgroup')/line[3:].lstrip('/')/'memory.peak'
            if p.exists():
                peak = int(p.read_text().strip())
    files = {p.name: file_sha256(p) for p in folder.iterdir()
             if p.is_file() and p.name not in ("RESULT.json", "STARTED.json")}
    return dict(batch_id=spec["batch_id"], cell=cell, state="COMPLETE", actual_spacing=spacing,
                seed=SEED, parameters=params, training_seconds=training["fit_seconds"],
                epochs=training["epochs_completed"], best_epoch=training["best_epoch"],
                stop_reason=training["stop_reason"], metrics=reports, population=meta,
                initial_weights_sha256=training["initial_weights_sha256"],
                restored_weights_sha256=training["restored_weights_sha256"], files=files,
                cgroup_peak_bytes=peak, tensorflow=tf.__version__,
                keras=importlib.metadata.version("keras"), test_read=False)


def report(spec, result):
    if result["state"] != "COMPLETE":
        return dict(experiment_key=f"i6dmo:{spec['batch_id']}:{result['cell']['id']}",
                    experiment_set_key="i6dmo:"+spec["batch_id"], actor="predictor", lake="LOCAL_RETAINED",
                    lineage="UNVERIFIED", config_sha256=digest(result["cell"]), code_commit=spec["code_commit"],
                    project="predictor", phase="I6DMO_RESOLUTION_BATCH",
                    tags=dict(cell=result["cell"]["id"], disposition=result["state"], test_read="false",
                              evidence_sha256=result["sha256"]),
                    metrics=[dict(metric="Cell_Complete", value=0, unit="bool", horizon=None, split="validation")])
    rows = []
    for split, values in result["metrics"].items():
        for r in values:
            for metric, key, unit in (("MAE", "mae", "USD_per_EUR"),
                                      ("Naive_MAE", "naive_mae", "USD_per_EUR"),
                                      ("Relative_Error_Vs_Naive", "relative_error", "ratio"),
                                      ("Prediction_Movement_MAE", "predicted_movement_mae", "USD_per_EUR"),
                                      ("N_Paired", "n", "count")):
                rows.append(dict(metric=metric, value=r[key], unit=unit, horizon=r["horizon"],
                                 split="validation" if split == "VALIDATION" else split.lower()))
    rows.extend(dict(metric=k, value=result[key], unit=unit, horizon=None, split="train")
                for k, key, unit in (("Parameters", "parameters", "count"),
                                     ("Fit_Seconds", "training_seconds", "seconds"),
                                     ("Epochs", "epochs", "count"), ("Best_Epoch", "best_epoch", "count")))
    if result.get("cgroup_peak_bytes") is not None:
        rows.append(dict(metric="Cgroup_Peak_Bytes", value=result["cgroup_peak_bytes"], unit="bytes",
                         horizon=None, split="train"))
    return dict(experiment_key=f"i6dmo:{spec['batch_id']}:{result['cell']['id']}",
                experiment_set_key="i6dmo:" + spec["batch_id"], actor="predictor", lake="LOCAL_RETAINED",
                lineage="UNVERIFIED", config_sha256=digest(dict(cell=result["cell"],
                    spacing=result["actual_spacing"], population=result["population"], seed=SEED,
                    training=TRAINING, code=spec["code_commit"])), code_commit=spec["code_commit"],
                project="predictor", phase="I6DMO_RESOLUTION_BATCH", metrics=rows,
                tags=dict(cell=result["cell"]["id"], kind=result["cell"]["kind"],
                          spacing=str(result["actual_spacing"]), slots=str(result["cell"]["slots"]),
                          seed=str(SEED), test_read="false", evaluation="DEVELOPMENT_TUNING",
                          evidence_sha256=result["sha256"]))


def generated_status(spec, root):
    root = Path(root)
    results = {c["id"]: t for c in spec["cells"] if (t := terminal(root, c, spec))}
    good = [r for r in results.values() if r["state"] == "COMPLETE"]
    failed = [r["cell"]["id"] for r in results.values() if r["state"] != "COMPLETE"]
    p = read(root / "PROGRESS.json") if (root / "PROGRESS.json").exists() else {}
    remaining = len(spec["cells"]) - len(results)
    eta = dict(kind="DONE", seconds=0) if remaining == 0 else dict(kind="UNMEASURED")
    if good and remaining:
        eta = dict(kind="PROVISIONAL_MIXED_MODEL_MEAN", basis_n=len(good),
                   seconds=remaining * math.fsum(r["training_seconds"] for r in good)/len(good),
                   note="fit-only extrapolation; window/model costs vary; excludes delivery waits")
    nxt = next_cell(spec, results)
    details = {c["id"]: dict(state=results[c["id"]]["state"],
                            parameters=results[c["id"]].get("parameters"),
                            fit_seconds=results[c["id"]].get("training_seconds"),
                            development=results[c["id"]].get("metrics", {}).get("VALIDATION"))
               if c["id"] in results else dict(state="PENDING") for c in spec["cells"]}
    return dict(schema="i6dmo.resolution_batch_status.v1", batch_id=spec["batch_id"], at=time.time(),
                state="COMPUTE_DONE" if not remaining else "STALE" if p.get("state") == "RUNNING" and
                time.time()-p.get("at", 0)>180 else "RUNNING" if p.get("state") == "RUNNING"
                else p["state"] if p.get("state", "").startswith(("WAIT_", "FAILED_"))
                else "BLOCKED_DEPENDENCY" if nxt is None else "PENDING",
                complete=len(good), terminal=len(results), total=len(spec["cells"]), failed=failed,
                progress=p, eta=eta, selected_long_spacing=selected_spacing(results), cells=details, test_read=False)


def summarize(spec, root):
    """Small navigable result ledger; diagnostics never confer trading eligibility."""
    root = Path(root)
    results = {c["id"]: t for c in spec["cells"] if (t := terminal(root, c, spec))}
    lines = ["# Resolution Batch Results", "", "DEVELOPMENT 2024; TEST closed; one seed 6102026.", "",
             "Relative error = (MAE_h-naive_h)/naive_h; negative is better.", "",
             "| Cell | State | Parameters | Fit Seconds | Per-Horizon Relative Error |",
             "|---|---|---:|---:|---|"]
    for c in spec["cells"]:
        r = results.get(c["id"], {})
        values = "; ".join(f"h{v['horizon']}: {100*v['relative_error']:+.6f}%"
                           for v in r.get("metrics", {}).get("VALIDATION", []))
        lines.append(f"| {c['id']} | {r.get('state', 'PENDING')} | {r.get('parameters', '')} | "
                     f"{r.get('training_seconds', '')} | {values} |")
    lines.extend(["", "Long spacing chosen automatically: " + str(selected_spacing(results)), "",
                  "The DLinear arm is a multivariate adaptation, not a literature reproduction.",
                  "No change to feature selection, pretrained extractors, DOIN or business masks.",
                  "An unsupported or failed head never becomes eligible from the aggregate."])
    path = root / "RESULTS.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines)+"\n")
    tmp.replace(path)


def command(argv, timeout=180):
    r = subprocess.run([str(x) for x in argv], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError("BATCH_COMMAND_FAILED:" + Path(str(argv[0])).name + ":" + str(r.returncode)
                           + ":" + r.stderr[-500:])
    return r.stdout


def mirror_tree(root, destination, role=None):
    """Copy only immutable artifacts and verify each byte; SQLite stays on ext4."""
    from tools.i6dmo import raw_long_campaign as L
    root, destination = Path(root), Path(destination)
    files = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*")
             if p.is_file() and p.name not in ("mirror.lock",)}
    if role and role.get("ssh_alias"):
        L._remote(role, ["mkdir", "-p", str(destination)])
        command(["rsync", "-a", "--checksum", str(root)+"/", role["ssh_alias"]+":"+str(destination)+"/"], 600)
        script = ("import pathlib,hashlib,json; r=pathlib.Path("+repr(str(destination))+"); "
                  "d="+repr(files)+"; print(json.dumps([n for n,h in d.items() "
                  "if not (r/n).is_file() or hashlib.sha256((r/n).read_bytes()).hexdigest()!=h]))")
        bad = json.loads(L._remote(role, ["python3", "-c", script], 120))
    else:
        destination.mkdir(parents=True, exist_ok=True)
        command(["rsync", "-a", "--checksum", str(root)+"/", str(destination)+"/"], 600)
        bad = [n for n, h in files.items() if file_sha256(destination/n) != h]
    if bad:
        raise ValueError("BATCH_MIRROR_MISMATCH")
    return dict(state="VERIFIED", count=len(files), manifest_sha256=digest(files))


def tick(spec):
    """One bounded sync/publication pass. Never fits or imports TensorFlow."""
    from tools.i6dmo import raw_long_campaign as L
    from tools.i6dmo.outbox import Outbox, HttpWarehouse
    verify_code(spec)
    root = Path(spec["root"])
    with (root / "tick.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "CLOSE.json").exists():
            stop_timers(spec)
            return read(root / "STATUS.json")
        deploy = read(spec["deployment"])
        role = deploy["roles"]["preferred"]
        artifacts = Path(spec["artifacts"])
        artifacts.mkdir(parents=True, exist_ok=True)
        # rsync temp files are not visible under RESULT names before rename.
        command(["rsync", "-a", "--checksum", "--exclude=*.lock", role["ssh_alias"]+":"+
                 spec["gpu_root"]+"/", str(artifacts)+"/"], 600)
        state = generated_status(spec, artifacts)
        summarize(spec, artifacts)
        box = Outbox(root / "outbox.sqlite3")
        try:
            for c in spec["cells"]:
                r = terminal(artifacts, c, spec)
                if r:
                    box.enqueue(report(spec, r), source_unit=c["id"])
            wh = deploy["warehouse"]
            pub = box.pump(HttpWarehouse(wh["url"], wh["token_env"]))
            state["publication"] = pub
            put(root / "STATUS.json", state)
            if state["terminal"] == state["total"] and pub["pending"] == 0 and pub["mismatch"] == 0:
                put(artifacts / "PUBLICATION.json", pub)
                put(artifacts / "READBACK.json", [dict(report_sha256=x["report_sha256"],
                    state=x["state"], readback_sha256=x["readback_sha256"]) for x in box.items()])
                put(artifacts / "BATCH_PUBLIC.json", {k: spec[k] for k in
                    ("batch_id", "cells", "seed", "training", "code_files", "code_commit", "population_digest")})
                close_input = dict(batch_id=spec["batch_id"], state="COMPUTE_AND_DELIVERY_COMPLETE",
                                   population_digest=spec["population_digest"], failed=state["failed"],
                                   expected_cells=len(spec["cells"]), publication=pub,
                                   results={c["id"]: terminal(artifacts, c, spec)["sha256"] for c in spec["cells"]},
                                   finalization="master CLOSE requires both verified mirrors of this receipt",
                                   test_read=False)
                put(artifacts / "CLOSE_RECEIPT.json", close_input)
                second = deploy["roles"]["secondary"]
                local_mirror = Path(deploy["roles"]["coordinator"]["experiment_storage_root"])/"i6dmo-mirrors"/spec["batch_id"]
                second_mirror = Path(second["experiment_storage_root"])/"i6dmo-mirrors"/spec["batch_id"]
                mirrors = dict(local=mirror_tree(artifacts, local_mirror),
                               second_host=mirror_tree(artifacts, second_mirror, second))
                state.update(state="CLOSED" if not state["failed"] else "CLOSED_WITH_FAILURES", mirrors=mirrors,
                             close_receipt_sha256=file_sha256(artifacts/"CLOSE_RECEIPT.json"))
                put(root / "CLOSE.json", state)
                put(root / "STATUS.json", state)
                stop_timers(spec)
            return state
        finally:
            box.close()


def stop_timers(spec):
    from tools.i6dmo import raw_long_campaign as L
    d = read(spec["deployment"])
    name = spec["batch_id"]
    L._remote(d["roles"]["preferred"], ["systemctl", "--user", "disable", "--now", name+"-gpu.timer"])
    command(["systemctl", "--user", "disable", "--now", name+"-tick.timer"])


def install(spec_path):
    """Install only uniquely named owned timers; credentials loaded only by CPU tick."""
    from tools.i6dmo import raw_long_campaign as L
    spec_path = Path(spec_path).resolve()
    s = read(spec_path)
    verify_code(s)
    d = read(s["deployment"])
    gpu = d["roles"]["preferred"]
    name = s["batch_id"]
    for target, role, action in (("gpu", gpu, "fit-next"), ("tick", d["roles"]["coordinator"], "tick")):
        path = s["gpu_spec"] if target == "gpu" else str(spec_path)
        run = [d["launcher"]["crispdm_run"], "-m", "4G" if target == "gpu" else "1G", "-t",
               "1800s" if target == "gpu" else "600s", "-n", name+"-"+target, "--", role["python"],
               "-B", "-m", "tools.i6dmo.resolution_batch", action, "--spec", path]
        env = ("EnvironmentFile="+d["warehouse"]["environment_file"]+"\n") if target == "tick" else ""
        text = ("[Unit]\nDescription=Finite resolution batch "+target+"\nX-I6DMO-Batch="+name+
                "\n[Service]\nType=oneshot\nWorkingDirectory="+s["runtime"]+"\n"+env+
                "Environment=PYTHONPATH="+s["runtime"]+"\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
                "Environment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\n"
                "Environment=CUDA_VISIBLE_DEVICES="+(s["gpu_uuid"] if target == "gpu" else "")+"\n"
                "ExecStart="+shlex.join(run)+"\nMemoryMax="+("5G" if target == "gpu" else "2G")+
                "\nMemorySwapMax=0\nTimeoutStartSec=1900\nNice=10\n")
        timer = ("[Unit]\nDescription=Automatic sequential resolution study\nX-I6DMO-Batch="+name+
                 "\n[Timer]\nOnBootSec=10s\nOnUnitInactiveSec=15s\nAccuracySec=1s\nPersistent=true\n"
                 "Unit="+name+"-"+target+".service\n[Install]\nWantedBy=timers.target\n")
        script = ("import pathlib; d=pathlib.Path.home()/'.config/systemd/user'; d.mkdir(parents=True,exist_ok=True); "
                  +"\n".join("p=d/"+repr(name+"-"+target+"."+suffix)+
                    "; assert not p.exists() or "+repr("X-I6DMO-Batch="+name)+" in p.read_text(); p.write_text("+repr(body)+")"
                    for suffix, body in (("service", text), ("timer", timer))))
        L._remote(role, ["python3", "-c", script])
        L._remote(role, ["systemctl", "--user", "daemon-reload"])
        L._remote(role, ["systemctl", "--user", "enable", "--now", name+"-"+target+".timer"])
    return dict(state="ARMED", batch_id=name)


def bootstrap(config_path):
    """Top-level idempotent prepare -> replicate pinned code/data -> arm both timers."""
    from tools.i6dmo import resolution_data as D, raw_long_campaign as L, context_bridge_campaign as CB
    from tools.i6dmo import campaign as C
    config = read(config_path)
    if set(config) != {"root", "source_deployment", "input_dir"} or any(
            not isinstance(v, str) or not Path(v).is_absolute() for v in config.values()):
        raise ValueError("BATCH_PRIVATE_CONFIG")
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    specpath = root / "BATCH.private.json"
    if specpath.exists():
        s = read(specpath)
        if s["launch_config"] != config:
            raise ValueError("BATCH_CONFIG_CHANGED")
        d = read(s["deployment"])
        role = d["roles"]["preferred"]
        L._remote(role, ["mkdir", "-p", s["gpu_root"]])
        command(["rsync", "-a", "--checksum", str(specpath), role["ssh_alias"]+":"+s["gpu_spec"]])
        return install(specpath)
    if command(["git", "-C", str(REPO), "status", "--porcelain"]).strip():
        raise ValueError("BATCH_SOURCE_NOT_FROZEN")
    deployment = read(config["source_deployment"])
    deployment["storage"] = dict(estimate_bytes=2*1024**3, reserve_bytes=5*1024**3)
    CB.storage_preflight(deployment, root)
    code = C._git_head()
    code_files = {str(p.relative_to(REPO)): file_sha256(p) for directory in
                  ("tools/i6dmo", "tools/i6n_contracts", "tools/i6n_archive", "tools/i6n_swarm",
                   "olap/store/src", "predictor_plugins/modular_temporal")
                  for p in sorted((REPO/directory).rglob("*.py"))}
    local_storage = Path(deployment["roles"]["coordinator"]["experiment_storage_root"])
    label = root.name
    pop = local_storage / "i6dmo" / label / "data"
    population = D.prepare(config["input_dir"], pop, REPO/"docs/i6n/direct_multi_output_static/EXPERIMENT.short.json")
    prepared = dict(launch_config=dict(input_dir=config["input_dir"]),
                    population_dir=str(pop), population_relpath=str(pop.relative_to(local_storage)))
    CB.storage_preflight(deployment, root, population_relpath=prepared["population_relpath"])
    deployment = L._provision(deployment, prepared, code, reviewed_files=code_files)
    deploypath = root / "DEPLOY.private.json"
    put(deploypath, deployment)
    role = deployment["roles"]["preferred"]
    gpu_root = "/var/tmp/i6dmo/"+label+"-gpu"
    name = "i6dmo-resolution-"+digest(dict(code=code, launch=config))[:12]
    heavy = [g for g in role["gpus"] if g.get("heavy")]
    if len(heavy) != 1 or heavy[0]["class"] != "rtx5090":
        raise ValueError("BATCH_5090_REQUIRED")
    spec = dict(batch_id=name, root=str(root), artifacts=str(local_storage/"i6dmo"/label/"artifacts"),
                gpu_root=gpu_root, gpu_data=str(Path(role["experiment_storage_root"])/prepared["population_relpath"]),
                gpu_input=str(Path(role["data_root"])/Path(config["input_dir"]).relative_to(
                    Path(deployment["roles"]["coordinator"]["data_root"]))),
                population_digest=population["manifest_sha256"],
                gpu_spec=gpu_root+"/BATCH.private.json", gpu_uuid=heavy[0]["uuid"], cells=cells(),
                training=TRAINING, seed=SEED, code_files=code_files, code_commit=code,
                deployment=str(deploypath), deployment_sha256=file_sha256(deploypath),
                runtime=role["workdir"], launch_config=config)
    put(specpath, spec)
    # Keep secret-free private paths in the remote spec, never public version control.
    L._remote(role, ["mkdir", "-p", gpu_root])
    command(["rsync", "-a", "--checksum", str(specpath), role["ssh_alias"]+":"+spec["gpu_spec"]])
    result = install(specpath)
    put(root/"LAUNCH.json", dict(**result, code_commit=code, seed=SEED, scientific_fits=19, test_open=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("bootstrap", "fit-next", "tick", "install", "status"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--cell", help="status for one exact cell id")
    a = p.parse_args()
    spec = read(a.spec)
    if a.action == "status":
        path = Path(spec["root"]) / "STATUS.json"
        state = read(path) if path.exists() else dict(state="NOT_STARTED", eta="UNMEASURED")
        if a.cell:
            state = dict(batch_id=spec["batch_id"], cell=a.cell,
                         status=state.get("cells", {}).get(a.cell, dict(state="PENDING")),
                         progress=state.get("progress", {}) if state.get("progress", {}).get("cell") == a.cell else {})
        print(json.dumps(state, sort_keys=True))
        return
    try:
        if a.action == "bootstrap":
            print(json.dumps(bootstrap(a.spec), sort_keys=True))
        elif a.action == "fit-next":
            fit_next(spec)
        elif a.action == "tick":
            print(json.dumps(tick(spec), sort_keys=True))
        elif a.action == "install":
            print(json.dumps(install(a.spec), sort_keys=True))
    except Exception as exc:
        if a.action == "fit-next":
            progress(spec["gpu_root"], state="FAILED_GUARD", error=type(exc).__name__+":"+str(exc)[:400])
        elif a.action == "tick":
            put(Path(spec["root"])/"STATUS.json", dict(state="WAIT_OPERATIONAL_RETRY", at=time.time(),
                error=type(exc).__name__+":"+str(exc)[:400], eta=dict(kind="UNMEASURED"), test_read=False))
        raise


if __name__ == "__main__":
    main()
