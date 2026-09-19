#!/usr/bin/env python3
"""RP34: the closure of an E1 pilot run — what the previous one only pretended to do.

The defect it replaces (dictum F2): the old check recomputed MASE from arrays the SAME record
supplied, so a forged MAE, a NaN MASE, foreign origins, a transplanted unit and a design with a
zeroed digest and another horizon were all accepted, and the published table carried the forged
horizon with the same means.

Here the population and the task come from the INDEPENDENT register — the sealed design, the
prepared DATA and the run's own job files — never from the record being judged, and the verdict is
四 separate facts, each with its own reason, because they can hold apart:

    metrics     the record's numbers are recomputed from the unit's arrays AND those arrays are
                bound to the prepared DATA: the evaluation origins, their labels and the
                denominator must BE the prepared ones, element by element, not merely the same
                length; every value finite, no bool arrays, nothing empty
    inference   the saved weights are reloaded in a FRESH process, the same windows are gathered
                from DATA, and the predictions must reproduce the stored ones inside a declared
                tolerance. A gradient summary is never accepted in place of this
    regime      R0/R1/R2 are judged from digests recomputed off the saved weight files: the
                detector R1/R2 imported must BE the bytes of that seed's auto-encoder, R1's
                detector must be unchanged after the fit and R0/R2's must have moved
    governance  whether a governed delivery, campaign and accepted terminal existed WHEN the unit
                ran. The 2026-09-19 pilot has none, so its units close as HISTORICAL_UNGOVERNED
                and are never presented as prospective governed evidence

    python tools/df_e1_close.py --root RUN_ROOT --out CLOSE.json [--replay-units N] [--no-replay]
    python tools/df_e1_close.py --replay JOB_FILE        # internal: the fresh-process reload
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA = "df_e1_close.v1"
VERIFIED, REFUSED, NOT_APPLICABLE, NOT_ATTEMPTED = "VERIFIED", "REFUSED", "NOT_APPLICABLE", "NOT_ATTEMPTED"
HISTORICAL = "HISTORICAL_UNGOVERNED"
GOVERNED = "GOVERNED"
#: the declared tolerance of the fresh-process reload, in the target's own units (kW)
PREDICTION_ATOL = 1e-5


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_numeric(arr, name: str) -> str | None:
    a = np.asarray(arr)
    if a.dtype == bool or not np.issubdtype(a.dtype, np.number):
        return f"{name}: not a numeric array ({a.dtype})"
    if a.size == 0:
        return f"{name}: empty"
    if not np.isfinite(a).all():
        return f"{name}: non-finite values"
    return None


def _number(value, name: str) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name}: not a number ({value!r})"
    if not np.isfinite(value):
        return f"{name}: non-finite ({value!r})"
    return None


# --- the independent register --------------------------------------------------------------------

def register(root: Path) -> dict:
    """Everything the verdict is measured against, read from the run's own sealed files and NEVER
    from the record under judgement. A design whose self-digest does not recompute is refused here,
    so a forged DESIGN (old digest or recomputed digest) never reaches a unit's verdict."""
    E = _load("df_mod_e0")
    root = Path(root)
    problems = []
    design = json.loads((root / "DESIGN.json").read_text())
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    recomputed = E.sha_obj(body)
    if recomputed != design.get("design_sha256"):
        problems.append(f"design self-digest: recomputed {recomputed} != declared {design.get('design_sha256')}")
    data_json = json.loads((root / "DATA.json").read_text())
    data_npz = root / "DATA.npz"
    data_digest = sha_file(data_npz)
    if data_digest != data_json.get("data_sha256"):
        problems.append("DATA.npz bytes are not the ones DATA.json recorded")
    if data_json.get("design_sha256") != design.get("design_sha256"):
        problems.append("DATA.json was prepared under another design")
    panel = Path(design["governed_bytes"]["path"])
    panel_now = sha_file(panel) if panel.is_file() else None
    if panel_now != data_json.get("panel_sha256"):
        problems.append("the panel's bytes are not the ones DATA was prepared from")
    z = np.load(data_npz)
    data = {k: z[k] for k in z.files}
    population = [c["cell_id"] for c in design["pilots"]] + [c["cell_id"] for c in design["cells"]]
    return {"design": design, "design_sha256": design["design_sha256"], "data": data, "data_json": data_json,
            "data_sha256": data_digest, "panel_sha256": panel_now, "population": population,
            "cells_by_id": {c["cell_id"]: c for c in design["pilots"] + design["cells"]},
            "problems": problems, "root": root}


def expected_job(reg: dict, cell: dict, root: Path) -> dict:
    """The job this cell MUST have run under, derived from the sealed design — the same construction
    the runner uses, so a transplanted or re-labelled attempt cannot match."""
    job = {"kind": cell["kind"], "cell_id": cell["cell_id"], "seed": cell["seed"], "max_updates": cell.get("max_updates", 0),
           "data_sha256": reg["data_sha256"], "role": cell.get("role", "CELL")}
    if "regime" in cell:
        job["regime"] = cell["regime"]
    if cell.get("depends_on"):
        # compared by the donor it names, not by an absolute path: a copy of the run is still the same
        # experiment, while a donor from another cell or another file is not
        job["pretrained_npz"] = f"{cell['depends_on']}/detector_pretrained.npz"
    return job


# --- the fresh-process reload ---------------------------------------------------------------------

def replay_worker(job_file: Path) -> int:
    """Reload the saved weights in THIS fresh process, gather the same windows from DATA and predict."""
    job = json.loads(Path(job_file).read_text())
    E = _load("df_mod_e0")
    RG = _load("df_e1_regimes")
    P = _load("df_e1_pilot")
    z = np.load(job["data_npz"])
    Xs, W, h, j = z["Xs"], int(z["window"][0]), int(z["horizon"][0]), int(z["target_channel"][0])
    origins = np.asarray(job["origins"], dtype=np.int64)
    design = json.loads(Path(job["design"]).read_text())
    model = P._model_for_target(design["graph"]["assignment"], W, Xs.shape[1], j, int(job["seed"]))
    model.load_weights(job["weights"])
    X = P._gather(Xs, origins, W)
    pred_scaled = np.asarray(model.predict(X, verbose=0))[:, 0]
    pred = pred_scaled * float(z["scaler_sd"][j]) + float(z["scaler_mean"][j])
    det = RG.detector_layer_names(model)
    if job.get("regime") == "R1":                       # the sealed regime's own configuration, so the counts are comparable
        for layer in model.layers:
            if layer.name in det:
                layer.trainable = False
    out = {"schema": "df_e1_replay.v1", "regime": job.get("regime"), "cell_id": job["cell_id"], "origins": origins.tolist(),
           "prediction": pred.tolist(), "detector_digest": RG.weights_digest(model, det),
           "non_detector_digest": RG.weights_digest(model, RG.non_detector_weighted_layer_names(model)),
           "detector_layers": det, "parameters": E.count_params(model),
           "weights_sha256": sha_file(Path(job["weights"])),
           "identity": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "tensorflow": __import__("tensorflow").__version__,
                        "process": os.getpid(), "threads": os.environ.get("OMP_NUM_THREADS")}}
    Path(job["out"]).write_text(json.dumps(out))
    return 0


def replay(root: Path, cell_id: str, weights: Path, origins: np.ndarray, seed: int, out_dir: Path, regime: str | None = None) -> dict:
    job = {"cell_id": cell_id, "weights": str(weights), "origins": origins.tolist(), "seed": int(seed), "regime": regime,
           "data_npz": str(Path(root) / "DATA.npz"), "design": str(Path(root) / "DESIGN.json"),
           "out": str(out_dir / f"replay_{cell_id}.json")}
    job_file = out_dir / f"replay_job_{cell_id}.json"
    job_file.write_text(json.dumps(job))
    proc = subprocess.run([sys.executable, "-B", str(HERE / "df_e1_close.py"), "--replay", str(job_file)],
                          capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout)[-600:]}
    return json.loads(Path(job["out"]).read_text())


# --- per-unit verdict -----------------------------------------------------------------------------

def verify_unit(reg: dict, cell_id: str, *, do_replay: bool = True, work: Path | None = None) -> dict:
    root = reg["root"]
    cell = reg["cells_by_id"].get(cell_id)
    entry = {"cell_id": cell_id, "metrics": NOT_ATTEMPTED, "inference": NOT_ATTEMPTED, "regime": NOT_ATTEMPTED,
             "governance": HISTORICAL, "problems": [], "facts": {}}
    prob = entry["problems"].append
    if cell is None:
        entry.update(metrics=REFUSED, inference=REFUSED, regime=REFUSED)
        prob("not a member of the sealed population")
        return _finish(entry, None)
    attempt = root / "attempts" / cell_id
    if not (attempt / "cell.json").is_file() or not (attempt / "outcome.json").is_file():
        entry.update(metrics=REFUSED, inference=REFUSED, regime=REFUSED)
        prob("absent: no attempt with a record")
        return _finish(entry, None)
    body = (attempt / "cell.json").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    outcome = json.loads((attempt / "outcome.json").read_text())
    result = json.loads((attempt / "result.json").read_text()) if (attempt / "result.json").is_file() else {}
    if digest != result.get("output_sha256") or digest != (outcome.get("verified") or {}).get("output_sha256"):
        entry.update(metrics=REFUSED, inference=REFUSED, regime=REFUSED)
        prob("the record's bytes are not the ones the child declared and the runner verified")
        return _finish(entry, None)
    rec = json.loads(body)
    entry["facts"]["record_sha256"] = digest
    # --- identity against the independent register (never against the record's own claims) -----
    if rec.get("schema") != "df_e1_cell.v1":
        prob(f"schema {rec.get('schema')!r}")
    if rec.get("cell_id") != cell_id:
        prob(f"the record calls itself {rec.get('cell_id')!r}")
    if rec.get("design_sha256") != reg["design_sha256"]:
        prob("the record was produced under another design")
    if rec.get("data_sha256") != reg["data_sha256"]:
        prob("the record was produced from other prepared data")
    if int(rec.get("seed", -1)) != int(cell["seed"]) or rec.get("kind") != cell["kind"]:
        prob("seed or kind differs from the sealed cell")
    if "regime" in cell and rec.get("regime") != cell["regime"]:
        prob(f"regime {rec.get('regime')!r} is not the sealed {cell['regime']!r}")
    if rec.get("task") != reg["design"]["task"]["id"]:
        prob(f"task {rec.get('task')!r} is not the sealed {reg['design']['task']['id']!r}")
    if rec.get("exposure") != "NO_TEST_ACCESS" or "test" in (rec.get("scores") or {}).get("validation", {}):
        prob("exposure is not NO_TEST_ACCESS")
    job_path = attempt / "job.json"
    if not job_path.is_file():
        prob("the attempt records no job")
    else:
        job = json.loads(job_path.read_text())
        want = expected_job(reg, cell, root)
        job = dict(job)
        if job.get("pretrained_npz"):
            parts = Path(job["pretrained_npz"]).parts
            job["pretrained_npz"] = "/".join(parts[-2:])
        diff = [k for k, v in want.items() if job.get(k) != v]
        entry["facts"]["job_fields_differing"] = diff
        if diff:
            prob(f"the job the attempt ran under differs from the design's: {diff}")
        if (job.get("design") or {}).get("design_sha256") != reg["design_sha256"]:
            prob("the job carried another design")
    arrays_path = attempt / "arrays.npz"
    if not arrays_path.is_file() or sha_file(arrays_path) != rec.get("arrays_sha256"):
        entry.update(metrics=REFUSED)
        prob("arrays absent or altered")
        return _finish(entry, rec)
    arr = dict(np.load(arrays_path))
    d = reg["data"]
    h = int(d["horizon"][0])
    ev = np.asarray(d["eval_origins"], dtype=np.int64)
    # --- metrics: the arrays must BE the prepared evaluation set, and every number recomputes ----
    if "eval_origins" not in arr or not np.array_equal(np.asarray(arr["eval_origins"], dtype=np.int64), ev):
        prob("the unit's evaluation origins are not the prepared ones")
    if "denominator" not in arr or not np.array_equal(np.asarray(arr["denominator"], dtype=float), np.asarray(d["denominator"], dtype=float)):
        prob("the unit's denominator is not the prepared one")
    if int(rec.get("common_evaluation_set_size", -1)) != int(ev.size):
        prob("the record's evaluation-set size is not the prepared one")
    if "validation_y" in arr:
        want_y = d["Y"][ev + h][:, None]
        if arr["validation_y"].shape != want_y.shape or not np.allclose(np.asarray(arr["validation_y"], dtype=float), want_y, atol=0, rtol=0, equal_nan=False):
            prob("the unit's labels are not the prepared targets at this horizon")
    _verify_metrics(entry, rec, arr, d)
    # --- inference and regime ---------------------------------------------------------------------
    if rec["kind"] == "fit":
        _verify_fit(entry, reg, rec, arr, attempt, cell, do_replay=do_replay, work=work)
    elif rec["kind"] == "ae":
        entry["inference"] = NOT_APPLICABLE
        det_file = attempt / "detector_pretrained.npz"
        if not det_file.is_file() or sha_file(det_file) != rec.get("detector_sha256"):
            prob("the pre-trained detector file is absent or not the recorded bytes")
        if not (attempt / "decoder.npz").is_file():
            prob("the separate decoder was not saved")
        entry["regime"] = REFUSED if entry["problems"] else VERIFIED
        entry["facts"]["detector_sha256"] = rec.get("detector_sha256")
    else:
        entry["inference"] = NOT_APPLICABLE
        _verify_controls(entry, reg, rec, arr)
    # --- governance ---------------------------------------------------------------------------------
    entry["governance"], entry["facts"]["governance"] = _governance(root, cell_id, rec)
    return _finish(entry, rec)


def _finish(entry: dict, rec: dict | None) -> dict:
    """The four facts are reported apart, and the scope NAMES what was verified. A unit whose weights
    were not reloaded is not 'verified with a caveat': its inference and regime are NOT_ATTEMPTED and
    the scope says so, so no reader can mistake a metric recomputation for a reproduced experiment."""
    for key in ("metrics", "inference", "regime"):
        if entry[key] == NOT_ATTEMPTED and entry["problems"] and key == "metrics":
            entry[key] = REFUSED
        elif entry[key] == NOT_ATTEMPTED and key == "metrics":
            entry[key] = VERIFIED
    clean = not entry["problems"]
    entry["verified"] = (entry["metrics"] == VERIFIED and entry["regime"] != REFUSED
                         and entry["inference"] != REFUSED and clean)
    replayed = entry["inference"] in (VERIFIED, NOT_APPLICABLE) and entry["regime"] in (VERIFIED, NOT_APPLICABLE)
    if not entry["verified"]:
        entry["scope"] = "REFUSED"
    elif not replayed:
        entry["scope"] = "METRICS_VERIFIED_INFERENCE_NOT_REPLAYED"
    else:
        entry["scope"] = ("SCIENTIFICALLY_VERIFIED_HISTORICAL_UNGOVERNED" if entry["governance"] == HISTORICAL
                          else "VERIFIED_AND_GOVERNED")
    return entry


def _verify_metrics(entry: dict, rec: dict, arr: dict, data: dict) -> None:
    E = _load("df_mod_e0")
    prob = entry["problems"].append
    scores = (rec.get("scores") or {}).get("validation") or {}
    denom = np.asarray(data["denominator"], dtype=float).tolist()
    checked = {}
    for name, block in scores.items():
        key = "validation_pred" if name == "model" else f"validation_pred_{name}"
        if key not in arr:
            if rec["kind"] == "ae":
                continue
            prob(f"{name}: the record reports a score with no predictions in the arrays")
            continue
        bad = _finite_numeric(arr[key], f"{key}") or _finite_numeric(arr["validation_y"], "validation_y")
        if bad:
            prob(bad)
            continue
        got = E.mase(np.asarray(arr[key], dtype=float), np.asarray(arr["validation_y"], dtype=float), denom)
        for metric in ("mase_mean", "mae_mean", "mse_mean", "rmse_mean"):
            claimed, actual = block.get(metric), got.get(metric)
            if claimed is None and actual is None:
                continue
            issue = _number(claimed, f"{name}.{metric}")
            if issue:
                prob(issue)
                continue
            if actual is None or abs(float(claimed) - float(actual)) > 1e-9:
                prob(f"{name}.{metric}: recorded {claimed!r} is not the recomputed {actual!r}")
        if block.get("status") != got.get("status"):
            prob(f"{name}.status: recorded {block.get('status')!r} is not the recomputed {got.get('status')!r}")
        checked[name] = {k: got.get(k) for k in ("mase_mean", "mae_mean", "rmse_mean", "status")}
    if rec["kind"] != "ae" and not checked:
        prob("no score could be recomputed")
    entry["facts"]["recomputed"] = checked
    entry["metrics"] = REFUSED if entry["problems"] else VERIFIED


def _verify_fit(entry: dict, reg: dict, rec: dict, arr: dict, attempt: Path, cell: dict, *, do_replay: bool, work: Path | None) -> None:
    RG = _load("df_e1_regimes")
    prob = entry["problems"].append
    root, d = reg["root"], reg["data"]
    weights = attempt / "weights.weights.h5"
    if not weights.is_file():
        prob("no saved weights: the inference cannot be reproduced")
        entry["inference"] = REFUSED
    ev = np.asarray(d["eval_origins"], dtype=np.int64)
    n = min(int(os.environ.get("DF_E1_REPLAY_WINDOWS", "512")), ev.size)
    if do_replay and weights.is_file():
        out = replay(root, cell["cell_id"], weights, ev[:n], int(cell["seed"]), work or root / "replays", regime=cell.get("regime"))
        if "error" in out:
            prob(f"replay failed: {out['error'][:200]}")
            entry["inference"] = REFUSED
        else:
            got = np.asarray(out["prediction"], dtype=float)
            stored = np.asarray(arr["validation_pred"], dtype=float)[:n, 0]
            diff = float(np.max(np.abs(got - stored))) if got.size == stored.size else None
            entry["facts"]["replay"] = {"windows": int(n), "max_abs_diff_kw": diff, "tolerance_kw": PREDICTION_ATOL,
                                        "identity": out["identity"], "weights_sha256": out["weights_sha256"],
                                        "detector_digest": out["detector_digest"], "parameters": out["parameters"]}
            if diff is None or diff > PREDICTION_ATOL:
                prob(f"the reloaded weights do not reproduce the stored predictions (max |diff| {diff})")
                entry["inference"] = REFUSED
            else:
                entry["inference"] = VERIFIED
            if out["parameters"] != rec.get("parameters"):
                prob("the reloaded graph's parameter counts are not the recorded ones")
            # the regime, judged on the RELOADED detector, not on the record's own summary
            setup = (rec.get("regime_setup") or {})
            regime = cell["regime"]
            after = out["detector_digest"]
            if regime == "R1":
                if after != setup.get("detector_digest_after_setup"):
                    prob("R1: the detector in the saved weights is not the one imported before the fit")
                if not rec.get("detector_unchanged"):
                    prob("R1: the record itself says the detector moved")
            else:
                if after == setup.get("detector_digest_after_setup"):
                    prob(f"{regime}: the detector in the saved weights never moved")
            if regime in ("R1", "R2"):
                ae_attempt = root / "attempts" / cell["depends_on"] / "detector_pretrained.npz"
                if not ae_attempt.is_file():
                    prob(f"{regime}: the auto-encoder's detector file is absent")
                elif sha_file(ae_attempt) != rec.get("pretrained_sha256"):
                    prob(f"{regime}: the imported detector is not the bytes of this seed's auto-encoder")
            entry["regime"] = REFUSED if entry["problems"] else VERIFIED
    else:
        entry["inference"] = NOT_ATTEMPTED if not do_replay else entry["inference"]
        entry["regime"] = NOT_ATTEMPTED if not do_replay else REFUSED
        if not do_replay:
            entry["facts"]["replay"] = "not attempted (--no-replay): inference and regime are NOT verified"
    tr = rec.get("training") or {}
    for key in ("updates", "epochs", "steps_per_epoch"):
        if not isinstance(tr.get(key), int) or tr[key] < 0:
            prob(f"training.{key} is not a count")
    if int(tr.get("updates", 0)) > int(cell.get("max_updates", 0)):
        prob("the unit ran past the sealed update ceiling")
    curve = (tr.get("curve") or {}).get("validation") or []
    if not curve or any(not np.isfinite(v) for v in curve):
        prob("the validation curve is empty or not finite")


def _verify_controls(entry: dict, reg: dict, rec: dict, arr: dict) -> None:
    """The controls are recomputed from the prepared DATA itself: persistence and the daily seasonal
    naive exactly, the ridge by refitting it with the design's own rule."""
    P = _load("df_e1_pilot")
    prob = entry["problems"].append
    d, design = reg["data"], reg["design"]
    W, h, j = int(d["window"][0]), int(d["horizon"][0]), int(d["target_channel"][0])
    Y, ev, tr = d["Y"], np.asarray(d["eval_origins"], dtype=np.int64), np.asarray(d["train_origins"], dtype=np.int64)
    want = {"persistence": Y[ev], "seasonal_naive_daily": Y[ev + h - P.DAY]}
    m_t, s_t = float(d["scaler_mean"][j]), float(d["scaler_sd"][j])
    nfeat = W * d["Xs"].shape[1] + 1
    G, b = np.zeros((nfeat, nfeat)), np.zeros(nfeat)
    for i in range(0, tr.size, 1024):
        o = tr[i:i + 1024]
        Xf = np.concatenate([P._gather(d["Xs"], o, W).reshape(o.size, -1).astype(np.float64), np.ones((o.size, 1))], axis=1)
        G += Xf.T @ Xf
        b += Xf.T @ ((Y[o + h] - m_t) / s_t)
    lam = float(design["ridge_lambda"]) * tr.size
    reg_m = lam * np.eye(nfeat)
    reg_m[-1, -1] = 0.0
    beta = np.linalg.solve(G + reg_m, b)
    Xe = np.concatenate([P._gather(d["Xs"], ev, W).reshape(ev.size, -1).astype(np.float64), np.ones((ev.size, 1))], axis=1)
    want["linear_ridge"] = (Xe @ beta) * s_t + m_t
    diffs = {}
    for name, values in want.items():
        key = f"validation_pred_{name}"
        if key not in arr:
            prob(f"{name}: no stored predictions")
            continue
        got = np.asarray(arr[key], dtype=float)[:, 0]
        diffs[name] = float(np.max(np.abs(got - values))) if got.size == values.size else None
        if diffs[name] is None or diffs[name] > 1e-6:
            prob(f"{name}: the stored control predictions are not the ones DATA implies (max |diff| {diffs[name]})")
    entry["facts"]["controls_recomputed_max_abs_diff"] = diffs
    entry["regime"] = NOT_APPLICABLE
    entry["inference"] = VERIFIED if not entry["problems"] else REFUSED


def _governance(root: Path, cell_id: str, rec: dict) -> tuple:
    """Was there a governed delivery, campaign and ACCEPTED terminal when this unit ran? A local
    document with the terminal schema is not an accepted terminal and never counts as one."""
    local = root / "TERMINALS" / f"{cell_id}.json"
    proposal = root / "CAMPAIGN_PROPOSAL.json"
    facts = {"local_terminal_document": local.is_file(),
             "campaign_proposal": json.loads(proposal.read_text()).get("status") if proposal.is_file() else None,
             "delivery_receipt": (root / "DELIVERIES.json").is_file(),
             "accepted_terminal_receipt": (root / "TERMINAL_RECEIPTS.json").is_file()}
    governed = facts["delivery_receipt"] and facts["accepted_terminal_receipt"]
    facts["reading"] = ("a governed delivery and an accepted terminal exist for this unit" if governed else
                        "no governed delivery and no accepted terminal existed when this unit ran: the local document "
                        "carries the terminal schema but was never submitted to, or accepted by, data-gov")
    return (GOVERNED if governed else HISTORICAL), facts


# --- the whole closure ----------------------------------------------------------------------------

def close(root: Path, *, do_replay: bool = True, out: Path | None = None, replay_units=None) -> dict:
    root = Path(root)
    reg = register(root)
    work = root / "closure_replays"
    work.mkdir(exist_ok=True)
    units, missing = {}, []
    for cell_id in reg["population"]:
        if not (root / "attempts" / cell_id / "cell.json").is_file():
            missing.append(cell_id)
        this_replay = do_replay and (replay_units is None or cell_id in set(replay_units))
        units[cell_id] = verify_unit(reg, cell_id, do_replay=this_replay, work=work)
    strangers = sorted({p.name for p in (root / "attempts").iterdir() if p.is_dir()} - set(reg["population"]))
    verified = [c for c, u in units.items() if u["verified"]]
    doc = {"schema": SCHEMA, "at": now_iso(), "root": str(root), "design_sha256": reg["design_sha256"],
           "data_sha256": reg["data_sha256"], "panel_sha256": reg["panel_sha256"],
           "register_problems": reg["problems"],
           "population": {"declared": reg["population"], "present": sorted(set(reg["population"]) - set(missing)),
                          "absent_ids": missing, "strangers_on_disk": strangers,
                          "rule": "the population is the sealed design's enumeration; absence is declared by id, never by a count"},
           "units": units,
           "counts": {"declared": len(reg["population"]), "verified": len(verified),
                      "refused": len([u for u in units.values() if not u["verified"]]),
                      "metrics_verified": len([u for u in units.values() if u["metrics"] == VERIFIED]),
                      "inference_verified": len([u for u in units.values() if u["inference"] == VERIFIED]),
                      "regime_verified": len([u for u in units.values() if u["regime"] == VERIFIED])},
           "governance": {"units_governed": len([u for u in units.values() if u["governance"] == GOVERNED]),
                          "units_historical_ungoverned": len([u for u in units.values() if u["governance"] == HISTORICAL]),
                          "meaning": "HISTORICAL_UNGOVERNED units are preserved development evidence: they were not "
                                     "delivered, registered or accepted by governance when they ran and are never "
                                     "presented as prospective governed evidence"},
           "replay": {"performed": bool(do_replay), "windows_per_unit": int(os.environ.get("DF_E1_REPLAY_WINDOWS", "512")),
                      "tolerance_kw": PREDICTION_ATOL,
                      "rule": "a fresh process reloads the saved weights and regathers the windows from DATA; a gradient "
                              "summary is never accepted in place of this"},
           "verdict": _verdict(units, verified, missing, reg)}
    if out:
        Path(out).write_text(json.dumps(doc, indent=1, default=float))
    return doc


def _verdict(units: dict, verified: list, missing: list, reg: dict) -> str:
    """ALL_VERIFIED is claimed only when every unit verified AND every unit that could be replayed was."""
    if missing or reg["problems"] or len(verified) != len(reg["population"]):
        return "PARTIAL" if verified else "NOT_VERIFIED"
    if any(u["scope"] == "METRICS_VERIFIED_INFERENCE_NOT_REPLAYED" for u in units.values()):
        return "ALL_METRICS_VERIFIED_NO_REPLAY"
    return "ALL_VERIFIED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--replay", type=Path, default=None, help="internal: the fresh-process reload of one unit")
    a = ap.parse_args(argv)
    if a.replay is not None:
        return replay_worker(a.replay)
    if a.root is None:
        ap.error("--root is required")
    doc = close(a.root, do_replay=not a.no_replay, out=a.out)
    print(json.dumps({"verdict": doc["verdict"], "counts": doc["counts"], "governance": doc["governance"]["units_historical_ungoverned"],
                      "absent": doc["population"]["absent_ids"], "register_problems": doc["register_problems"],
                      "refused": {c: u["problems"][:2] for c, u in doc["units"].items() if not u["verified"]}}, indent=1)[:3000])
    return 0 if doc["verdict"] != "NOT_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
