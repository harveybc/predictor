"""P7 (SC11): ONE bounded ENGINEERING replay (never scientific) of both matched arms.

Uses the REAL sealed MG1/MG2m designs, the REAL frozen population (digest-verified load with the
retained input) through the real matched data path (train._prepare_matched), then a small row
slice of FIT/INNER/DEVELOPMENT, at most 2 epochs (train.fit_dmo with the sealed recipe otherwise),
model.save_dmo -> model.load_dmo, train.replay_monitored. Writes only into --out (scratch).

Modes:
  inputs  (CPU, no TensorFlow fit) the exact materialized model inputs of the slice -> digests
  fit     (GPU) fit + save + SAME-PROCESS reload; predictions/monitor must be EXACT
  replay  (GPU, fresh process) reload the saved models; predictions and the monitored value are
          compared with the sealed i6dmo.monitor_numeric_equivalence.v2 floor
          (|a - r| <= 1e-9 + 5e-5 |r|); inputs must equal the CPU digests byte for byte

  python p7_gpu_replay.py --mode {inputs,fit,replay} --population P --input-dir I --out O
         [--rows-fit 4096 --rows-inner 512 --rows-dev 512]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO = Path(os.environ.get("RETSU_CODE_ROOT") or Path(__file__).resolve().parents[5])
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

DIGEST = json.loads((REPO / "docs/i6n/direct_multi_output_static/matched_short/POPULATION_SEAL.json")
                    .read_text())["population_digest"]
DESIGNS = {a: REPO / f"docs/i6n/direct_multi_output_static/matched_short/EXPERIMENT.{a}.json"
           for a in ("mg1", "mg2m")}
MAX_EPOCHS = 2


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def arr_digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return sha_bytes(json.dumps([list(a.shape), str(a.dtype)]).encode() + a.tobytes())


def slice_rows(d, n):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["row_id"]):
            out[k] = v[:n].copy()
        else:
            out[k] = v
    return out


def prepared(args, arm):
    from tools.i6dmo import train as T
    effective = T.effective_from_design(DESIGNS[arm])
    data, scales, matched = T._prepare_matched(
        {"path": args.population, "expected_digest": DIGEST, "input_dir": args.input_dir}, effective)
    n = {"fit": args.rows_fit, "inner": args.rows_inner, "validation": args.rows_dev}
    sl = {k: slice_rows(data[k], n[k]) for k in ("fit", "inner", "validation")}
    inputs = {k: T.materialize_monitor_inputs(sl[k], scales) for k in sl}
    digests = {}
    for k, (x, y, m) in inputs.items():
        parts = x if isinstance(x, (list, tuple)) else [x]
        digests[k] = {"x_parts": [arr_digest(p) for p in parts], "y": arr_digest(y), "mask": arr_digest(m),
                      "row_id": arr_digest(sl[k]["row_id"]), "dtypes": [str(p.dtype) for p in parts],
                      "shapes": [list(p.shape) for p in parts]}
    return effective, sl, scales, matched, inputs, digests


def process_identity():
    return {"uuid": uuid.uuid4().hex, "pid": os.getpid()}


def gpu_identity():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    names = []
    for g in gpus:
        try:
            names.append(tf.config.experimental.get_device_details(g).get("device_name"))
        except Exception:  # noqa: BLE001
            names.append(None)
    return {"visible_gpus": len(gpus), "device_names": names, "tensorflow": tf.__version__}


def mode_inputs(args):
    out = {"mode": "inputs", "process": process_identity(), "arms": {}}
    for arm in ("mg1", "mg2m"):
        _, sl, scales, matched, _, digests = prepared(args, arm)
        out["arms"][arm] = {"input_digests": digests, "scales_sha256": scales["scales_sha256"],
                            "population": matched["common_population_sha256"]}
    return out


def mode_fit(args):
    from tools.i6dmo import model as M  # flags before TF
    from tools.i6dmo import train as T
    import tensorflow as tf
    flags = M.enable_determinism()
    out = {"mode": "fit", "process": T.PROCESS_IDENTITY, "gpu": gpu_identity(), "determinism": flags,
           "scientific": False, "label": "ENGINEERING_REPLAY_NOT_SCIENTIFIC", "max_epochs": MAX_EPOCHS, "arms": {}}
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("NO_GPU_VISIBLE: this replay must run on the pinned GPU")
    for arm in ("mg1", "mg2m"):
        effective, sl, scales, matched, inputs, digests = prepared(args, arm)
        training = {**effective["training"], "max_epochs": MAX_EPOCHS}
        with tf.device("/GPU:0"):
            bundle = M.build_dmo(effective["model"])
            capacity = M.capacity_record(bundle)
            result = T.fit_dmo(bundle, inputs["fit"], inputs["inner"], training, effective["seed"],
                               ids=(sl["fit"]["row_id"], sl["inner"]["row_id"]), record_device="gpu")
            pred = M.predict_matrix(bundle.forecast_model, inputs["validation"][0])
        unit = Path(args.out) / arm
        unit.mkdir(parents=True, exist_ok=True)
        doc = M.save_dmo(bundle, unit / "model", extra={"scales_sha256": scales["scales_sha256"]})
        loaded, ldoc = M.load_dmo(unit / "model")
        with tf.device("/GPU:0"):
            pred_reload = M.predict_matrix(loaded.forecast_model, inputs["validation"][0])
            _, _, mon_same, rep_same = T.replay_monitored(loaded.forecast_model, inputs["fit"], inputs["inner"],
                                                          result["monitor_evaluation"],
                                                          ids=(sl["fit"]["row_id"], sl["inner"]["row_id"]))
        price = T.to_price(pred, sl["validation"], scales)
        np.savez(unit / "predictions_dev.npz", row_id=sl["validation"]["row_id"], predicted=pred,
                 predicted_price=price, mask=sl["validation"]["mask"])
        (unit / "TRAINING.json").write_text(json.dumps({"monitor_evaluation": result["monitor_evaluation"]},
                                                       default=str))
        out["arms"][arm] = {
            "input_digests": digests, "scales_sha256": scales["scales_sha256"],
            "capacity_total": capacity["counts"]["total"], "capacity_heads": capacity["counts"]["heads"],
            "epochs_completed": result["epochs_completed"], "best_epoch": result["best_epoch"],
            "monitored": result["monitor_evaluation"]["monitored"], "self_check": result["monitor_evaluation"]["self_check"],
            "weights_sha256_saved": doc["weights_sha256"], "weights_sha256_loaded": ldoc["weights_sha256"],
            "load_path": ldoc.get("load_path"),
            "same_process_reload_predictions_bitwise": pred.tobytes() == pred_reload.tobytes(),
            "same_process_reload_max_abs_diff": float(np.max(np.abs(pred - pred_reload))),
            "same_process_monitor_replay": {"status": rep_same.get("status"), "monitored": mon_same,
                                            "recorded": result["monitor_evaluation"]["monitored"],
                                            "exact": mon_same == result["monitor_evaluation"]["monitored"]},
            "predictions_sha256": hashlib.sha256((unit / "predictions_dev.npz").read_bytes()).hexdigest(),
            "origin_support": {"dev_row_ids_equal_population_slice": bool(np.array_equal(
                sl["validation"]["row_id"], np.asarray(matched_rows(args, "DEVELOPMENT")[:args.rows_dev]))),
                "dev_mask_rows": int(sl["validation"]["mask"].any(axis=1).sum()),
                "dev_mask_cells": int(sl["validation"]["mask"].sum())},
        }
    out["resources"] = {"cgroup_memory_peak_bytes": T._cgroup_peak(),
                        "tf_gpu_memory_peak_bytes": _gpu_peak()}
    return out


def matched_rows(args, split):
    with np.load(Path(args.population) / f"POPULATION_{split}.npz", allow_pickle=False) as z:
        return z["origin_row_ids"].copy()


def _gpu_peak():
    import tensorflow as tf
    try:
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:  # noqa: BLE001
        return None


def mode_replay(args):
    from tools.i6dmo import model as M  # flags before TF
    from tools.i6dmo import train as T
    import tensorflow as tf
    M.enable_determinism()
    fit_doc = json.loads((Path(args.out) / "FIT.json").read_text())
    out = {"mode": "replay", "process": T.PROCESS_IDENTITY, "gpu": gpu_identity(),
           "fit_process": fit_doc["process"], "label": "ENGINEERING_REPLAY_NOT_SCIENTIFIC", "arms": {}}
    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("NO_GPU_VISIBLE")
    for arm in ("mg1", "mg2m"):
        effective, sl, scales, matched, inputs, digests = prepared(args, arm)
        unit = Path(args.out) / arm
        saved = np.load(unit / "predictions_dev.npz")
        record = json.loads((unit / "TRAINING.json").read_text())["monitor_evaluation"]
        loaded, ldoc = M.load_dmo(unit / "model")
        with tf.device("/GPU:0"):
            pred = M.predict_matrix(loaded.forecast_model, inputs["validation"][0])
            fm, im, mon, rep = T.replay_monitored(loaded.forecast_model, inputs["fit"], inputs["inner"], record,
                                                  ids=(sl["fit"]["row_id"], sl["inner"]["row_id"]))
        ref = saved["predicted"]
        diff = np.abs(pred - ref)
        allowed = T.MONITOR_TOLERANCE["atol"] + T.MONITOR_TOLERANCE["rtol"] * np.abs(ref)
        out["arms"][arm] = {
            "inputs_equal_fit_process": digests == fit_doc["arms"][arm]["input_digests"],
            "weights_sha256_loaded": ldoc["weights_sha256"],
            "weights_equal_saved": ldoc["weights_sha256"] == fit_doc["arms"][arm]["weights_sha256_saved"],
            "predictions_bitwise_equal": pred.tobytes() == ref.tobytes(),
            "predictions_max_abs_diff": float(diff.max()),
            "predictions_max_rel_diff": float(np.max(diff / np.maximum(np.abs(ref), 1e-300))),
            "predictions_within_v2_floor": bool((diff <= allowed).all()),
            "monitor_replay_status": rep.get("status"), "monitor_within_floor": rep.get("within_floor"),
            "monitored": mon, "recorded": record["monitored"], "monitored_abs_delta": abs(mon - record["monitored"]),
            "monitor_context": rep.get("context"),
            "row_ids_equal": bool(np.array_equal(saved["row_id"], sl["validation"]["row_id"])),
            "mask_equal": bool(np.array_equal(saved["mask"], sl["validation"]["mask"])),
        }
    out["floor_protocol"] = T.floor_protocol_identity()
    out["resources"] = {"cgroup_memory_peak_bytes": T._cgroup_peak(), "tf_gpu_memory_peak_bytes": _gpu_peak()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("inputs", "fit", "replay"))
    p.add_argument("--population", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows-fit", type=int, default=4096)
    p.add_argument("--rows-inner", type=int, default=512)
    p.add_argument("--rows-dev", type=int, default=512)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    doc = {"inputs": mode_inputs, "fit": mode_fit, "replay": mode_replay}[args.mode](args)
    doc["wall_seconds"] = time.time() - t0
    doc["rows"] = {"fit": args.rows_fit, "inner": args.rows_inner, "dev": args.rows_dev}
    name = {"inputs": "INPUTS_CPU.json", "fit": "FIT.json", "replay": "REPLAY.json"}[args.mode]
    (Path(args.out) / name).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(json.dumps({"wrote": name, "wall_seconds": doc["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
