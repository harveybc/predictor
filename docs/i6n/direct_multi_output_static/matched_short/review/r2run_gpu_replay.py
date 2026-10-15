"""R2 runner review (a5ac7607): leakage check on the REAL population + ONE bounded ENGINEERING GPU replay.

LABEL: ENGINEERING_REPLAY_NOT_SCIENTIFIC. Uses the model lane's REAL functions
(r2_runner.feature_corpus/corpora, model.pretrain_branch, write_branch_donor_terminal,
import_branch_donors, train.fit_dmo, save_dmo/load_dmo). Writes only into --work (scratch).

Modes:
  leak       CPU: spy the population accessor while computing all 19 learned corpora; rebuild the
             population from a DEVELOPMENT-perturbed copy of the retained input; every corpus digest
             must be bitwise unchanged; writes the feature-0 slices (original and perturbed) and the
             19 slice corpora used below.
  pretrain   GPU: ONE real pretraining unit body for feature 0 on the slice (max_epochs 2), donor
             terminal written; --variant perturbed repeats it on the perturbed slice.
  importfit  GPU: 19 donor terminals (feature 0 pretrained; the other 18 engineering scaffolds at
             seed weights, labelled) imported by import_branch_donors; donor bytes survive bitwise;
             then a tiny 2-epoch fit, save -> same-process reload (bitwise).
  replay     GPU, fresh process: reload; predictions within the v2 policy; inputs identical.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

REPO = Path(os.environ.get("RETSU_CODE_ROOT") or Path(__file__).resolve().parents[5])
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

MATCHED = REPO / "docs/i6n/direct_multi_output_static/matched_short"
SEAL = json.loads((MATCHED / "POPULATION_SEAL.json").read_text())
CONTRACT = json.loads((MATCHED / "R2_CONTRAST_DESIGN.ruled.json").read_text())
DESIGN = MATCHED / "EXPERIMENT.mg2m.json"
N_FIT, N_SEL = 2048, 256
PROC = {"uuid": uuid.uuid4().hex, "pid": os.getpid()}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fsha(p):
    return sha(Path(p).read_bytes())


def effective():
    from tools.i6dmo import train as T
    return T.effective_from_design(str(DESIGN))


def objective():
    obj = json.loads(json.dumps(CONTRACT["pretraining"]["objective"]))
    obj["fit_settings"]["max_epochs"] = 2          # ENGINEERING bound (never scientific)
    return obj


def population(args):
    from tools.i6dmo import train as T
    eff = effective()
    return T.load_matched_population(args.population, SEAL["population_digest"], eff["horizons_hours"],
                                     eff["feature_manifest"]["features"], input_dir=args.input_dir)


def mode_leak(args):
    from tools.i6dmo import common_population as CP, design as D, provenance, r2_runner, train as T
    work = Path(args.work)
    eff = effective()
    pop = population(args)
    asked, real = [], CP.arm_tensors

    def spy(p, design, split):
        asked.append(split)
        return real(p, design, split)
    CP.arm_tensors = spy
    try:
        orig = r2_runner.corpora(pop, eff)
        fit_w, sel_w, corpus0 = r2_runner.feature_corpus(pop, eff, 0)
        slices = {}
        for j, f in enumerate(eff["feature_manifest"]["features"]):
            fw, sw, _ = r2_runner.feature_corpus(pop, eff, j)
            slices[f] = (fw[:N_FIT], sw[:N_SEL])
    finally:
        CP.arm_tensors = real
    # perturb DEVELOPMENT bars of a COPY of the input and rebuild the population (data lane, real)
    pin = work / "input_dev_perturbed"
    if pin.exists():
        shutil.rmtree(pin)
    shutil.copytree(args.input_dir, pin)
    with np.load(pin / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"].copy(), z["x"].copy(), z["price"].copy()
    dev_start = int(np.datetime64("2024-01-01T00:00:00", "s").astype(np.int64))
    sel = ts >= dev_start
    x[sel] = x[sel] * 1.37 + 0.5
    price[sel] = price[sel] * 1.01
    np.savez(pin / "INPUT.npz", ts=ts, x=x, price=price)
    clock = json.loads((pin / "CLOCK.json").read_text())
    clock["input_sha256"] = provenance.file_sha256(pin / "INPUT.npz")
    (pin / "CLOCK.json").write_text(json.dumps(clock))
    exps = {a: D.load_design(MATCHED / f"EXPERIMENT.{a}.json") for a in CP.ARMS}
    dds = {a: json.loads((MATCHED / f"DESIGN.{a}.json").read_text()) for a in CP.ARMS}
    ppop = CP.build_common_population(pin, exps, dds, code_commit=SEAL["identities"]["code"]["commit"])
    pert = r2_runner.corpora(ppop, eff)
    pfit, psel, _ = r2_runner.feature_corpus(ppop, eff, 0)
    a_dev = CP.arm_tensors(ppop, json.loads(DESIGN.read_text()), "DEVELOPMENT")
    o_dev = T.arm_split(pop, json.loads(DESIGN.read_text()), "DEVELOPMENT")
    np.savez(work / "slice_original.npz", fit=fit_w[:N_FIT], sel=sel_w[:N_SEL])
    np.savez(work / "slice_perturbed.npz", fit=pfit[:N_FIT], sel=psel[:N_SEL])
    np.savez(work / "slices_all.npz", **{f"{i}_fit": slices[f][0] for i, f in enumerate(slices)},
             **{f"{i}_sel": slices[f][1] for i, f in enumerate(slices)})
    shutil.rmtree(pin)
    return {"accessor_splits_requested": sorted(set(asked)), "accessor_calls": len(asked),
            "dev_bars_perturbed": int(sel.sum()),
            "population_digest_changed": ppop.digest != pop.digest,
            "development_x_changed": not np.array_equal(a_dev.x, o_dev.x),
            "corpora_equal_all_19": orig == pert, "n_features": len(orig),
            "feature0_slice_bitwise_equal": fit_w[:N_FIT].tobytes() == pfit[:N_FIT].tobytes()
            and sel_w[:N_SEL].tobytes() == psel[:N_SEL].tobytes(),
            "feature0_corpus_counts": corpus0["counts"]}


def _context(bundle, corpus_by_feature, pop_digest, scaler_sha):
    from tools.i6dmo import campaign as C, model as M
    eff = effective()
    feats = eff["feature_manifest"]["features"]
    prov = {"campaign_id": "ENGINEERING_REPLAY_NOT_SCIENTIFIC", "ruling_seal_sha256": CONTRACT["seal_sha256"],
            "code_pin": "a5ac7607", "units": {f: C.r2_pretrain_unit_id(i) for i, f in enumerate(feats)},
            "corpus": corpus_by_feature}
    return M.branch_donor_context(bundle, population_sha256=pop_digest, scaler_sha256=scaler_sha,
                                  objective=objective(), provenance=prov)


def _slice_corpus(fit, sel, j):
    body = {"feature_index": j, "engineering_slice": True, "fit_windows": sha(np.ascontiguousarray(fit).tobytes()),
            "selection_windows": sha(np.ascontiguousarray(sel).tobytes()), "n": [len(fit), len(sel)]}
    return dict(body, sha256=sha(json.dumps(body, sort_keys=True).encode()))


def _ids():
    return SEAL["population_digest"], json.loads((Path(os.environ["RETSU_WORK"]) / "SCALER_SHA.json").read_text())


def mode_pretrain(args):
    # mirrors r2_runner.run_pretrain's import order at a5ac7607 (TensorFlow BEFORE tools.i6dmo.model)
    import tensorflow as tf
    from tools.i6dmo import model as M, train as T
    flags = M.enable_determinism()
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("NO_GPU_VISIBLE")
    work = Path(args.work)
    eff = effective()
    feats = eff["feature_manifest"]["features"]
    allz = np.load(work / "slices_all.npz")
    corpora = {f: _slice_corpus(allz[f"{i}_fit"], allz[f"{i}_sel"], i) for i, f in enumerate(feats)}
    z = np.load(work / f"slice_{args.variant}.npz")
    fit, sel = z["fit"], z["sel"]
    with tf.device("/GPU:0"):
        bundle = M.build_dmo(eff["model"])
        ctx = _context(bundle, {f: c["sha256"] for f, c in corpora.items()}, *_ids())
        t0 = time.monotonic()
        pre = M.pretrain_branch(bundle, feats[0], fit, sel, objective())
        secs = time.monotonic() - t0
    out = work / f"donors_{args.variant}" / feats[0]
    if out.exists():
        shutil.rmtree(out)
    ref = M.write_branch_donor_terminal(out, bundle=bundle, feature=feats[0], context=ctx,
                                        unit_id="r2.pretrain.00", corpus=corpora[feats[0]], pretrained=pre)
    return {"variant": args.variant, "feature": feats[0], "applied_seed": pre["applied_seed"],
            "fit": {k: pre["fit"].get(k) for k in ("selected_epoch", "observed_updates", "stop_reason",
                                                   "best_validation_loss")},
            "reconstruction": pre.get("reconstruction"), "pretrain_seconds": secs,
            "donor_npz_sha256": fsha(out / "donor.npz"), "donor_terminal_sha256": ref["sha256"],
            "determinism": flags, "process": PROC, "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
            "tf": tf.__version__, "cgroup_memory_peak_bytes": T._cgroup_peak(),
            "init_order": M.INIT_ORDER}


def mode_importfit(args):
    from tools.i6dmo import model as M, train as T  # static-algorithm flags BEFORE TensorFlow
    import tensorflow as tf
    M.enable_determinism()
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("NO_GPU_VISIBLE")
    work = Path(args.work)
    eff = effective()
    feats = eff["feature_manifest"]["features"]
    allz = np.load(work / "slices_all.npz")
    corpora = {f: _slice_corpus(allz[f"{i}_fit"], allz[f"{i}_sel"], i) for i, f in enumerate(feats)}
    seed_build = M.build_dmo(eff["model"])
    ctx = _context(seed_build, {f: c["sha256"] for f, c in corpora.items()}, *_ids())
    refs = {}
    donor0 = work / "donors_original" / feats[0]
    refs[feats[0]] = {"terminal": {"path": str(donor0 / "TERMINAL.json"), "sha256": fsha(donor0 / "TERMINAL.json")}}
    for i, f in enumerate(feats[1:], start=1):   # ENGINEERING scaffold donors: seed weights, never trained
        out = work / "donors_original" / f
        if out.exists():
            shutil.rmtree(out)
        w = [np.array(v) for v in seed_build.branch_models[f].get_weights()]
        pre = {"weights": w, "applied_seed": M.r2_seed(i), "fit": {"stop_reason": "max_epochs",
                                                                    "engineering_scaffold": True},
               "reconstruction": None}
        ref = M.write_branch_donor_terminal(out, bundle=seed_build, feature=f, context=ctx,
                                            unit_id=M_unit(i), corpus=corpora[f], pretrained=pre)
        refs[f] = {"terminal": {"path": ref["path"], "sha256": ref["sha256"]}}
    with tf.device("/GPU:0"):
        bundle = M.build_dmo(eff["model"])
        r0 = M.build_dmo(eff["model"])
        record = M.import_branch_donors(bundle, refs, _context(bundle, {f: c["sha256"] for f, c in corpora.items()},
                                                               *_ids()))
        with np.load(donor0 / "donor.npz") as z:
            donor_w = [z["kernel"], z["bias"]]
        survived = all(a.tobytes() == b.tobytes() for a, b in zip(bundle.branch_models[feats[0]].get_weights(), donor_w))
        others_seed = all(all(a.tobytes() == b.tobytes() for a, b in zip(bundle.branch_models[f].get_weights(),
                                                                          r0.branch_models[f].get_weights()))
                          for f in feats[1:])
        non_branch = M._non_branch_digest(bundle) == M._non_branch_digest(r0)
        # tiny fit on a real-population slice (2 epochs, engineering)
        data, scales, _ = T._prepare_matched({"path": args.population, "expected_digest": SEAL["population_digest"],
                                              "input_dir": args.input_dir}, eff)
        sl = {k: {n: (v[:m] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(data[k]["row_id"]) else v)
                  for n, v in data[k].items()} for k, m in (("fit", N_FIT), ("inner", N_SEL))}
        fit_in = T.materialize_monitor_inputs(sl["fit"], scales)
        inn_in = T.materialize_monitor_inputs(sl["inner"], scales)
        training = {**eff["training"], "max_epochs": 2}
        result = T.fit_dmo(bundle, fit_in, inn_in, training, eff["seed"],
                           ids=(sl["fit"]["row_id"], sl["inner"]["row_id"]), record_device="gpu")
        pred = M.predict_matrix(bundle.forecast_model, inn_in[0])
    unit = work / "r2fit"
    if unit.exists():
        shutil.rmtree(unit)
    doc = M.save_dmo(bundle, unit / "model")
    loaded, ldoc = M.load_dmo(unit / "model")
    with tf.device("/GPU:0"):
        pred2 = M.predict_matrix(loaded.forecast_model, inn_in[0])
    np.savez(unit / "pred_inner.npz", pred=pred)
    (unit / "TRAINING.json").write_text(json.dumps({"monitor_evaluation": result["monitor_evaluation"]}, default=str))
    inputs = {"x": [sha(np.ascontiguousarray(p).tobytes()) for p in inn_in[0]]}
    (unit / "INPUTS.json").write_text(json.dumps(inputs))
    return {"import_sha256": record["import_sha256"], "donor_weights_survive_import_bitwise": survived,
            "other_18_equal_seed_build": others_seed, "non_branch_equal_seed_build": non_branch,
            "branches_trainable": record["trainable"], "trainable_parameters": record["trainable_parameters"],
            "fit_epochs": result["epochs_completed"], "self_check": result["monitor_evaluation"]["self_check"],
            "weights_saved_eq_loaded": doc["weights_sha256"] == ldoc["weights_sha256"],
            "same_process_reload_bitwise": pred.tobytes() == pred2.tobytes(), "process": PROC,
            "cgroup_memory_peak_bytes": T._cgroup_peak()}


def M_unit(i):
    from tools.i6dmo import campaign as C
    return C.r2_pretrain_unit_id(i)


def mode_replay(args):
    from tools.i6dmo import model as M, train as T  # static-algorithm flags BEFORE TensorFlow
    import tensorflow as tf
    M.enable_determinism()
    work = Path(args.work)
    eff = effective()
    data, scales, _ = T._prepare_matched({"path": args.population, "expected_digest": SEAL["population_digest"],
                                          "input_dir": args.input_dir}, eff)
    inner = {n: (v[:N_SEL] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(data["inner"]["row_id"]) else v)
             for n, v in data["inner"].items()}
    inn_in = T.materialize_monitor_inputs(inner, scales)
    unit = work / "r2fit"
    loaded, _ = M.load_dmo(unit / "model")
    with tf.device("/GPU:0"):
        pred = M.predict_matrix(loaded.forecast_model, inn_in[0])
    ref = np.load(unit / "pred_inner.npz")["pred"]
    diff = np.abs(pred - ref)
    allowed = T.MONITOR_TOLERANCE["atol"] + T.MONITOR_TOLERANCE["rtol"] * np.abs(ref)
    inputs = {"x": [sha(np.ascontiguousarray(p).tobytes()) for p in inn_in[0]]}
    return {"inputs_equal_fit_process": inputs == json.loads((unit / "INPUTS.json").read_text()),
            "predictions_bitwise_equal": pred.tobytes() == ref.tobytes(), "max_abs_diff": float(diff.max()),
            "within_v2_floor": bool((diff <= allowed).all()), "process": PROC,
            "cgroup_memory_peak_bytes": T._cgroup_peak()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("leak", "pretrain", "importfit", "replay"))
    p.add_argument("--variant", default="original", choices=("original", "perturbed"))
    p.add_argument("--population", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--work", required=True)
    a = p.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    os.environ["RETSU_WORK"] = a.work
    if a.mode == "leak":
        from tools.i6dmo import r2_runner
        out = mode_leak(a)
        pop = population(a)
        (Path(a.work) / "SCALER_SHA.json").write_text(json.dumps(r2_runner.scaler_sha256(pop)))
    else:
        out = {"pretrain": mode_pretrain, "importfit": mode_importfit, "replay": mode_replay}[a.mode](a)
    out["label"] = "ENGINEERING_REPLAY_NOT_SCIENTIFIC"
    name = f"{a.mode.upper()}{'_' + a.variant if a.mode == 'pretrain' else ''}.json"
    (Path(a.work) / name).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps({"wrote": name}))


if __name__ == "__main__":
    sys.exit(main())
