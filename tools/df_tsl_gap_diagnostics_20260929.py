#!/usr/bin/env python3
"""RB02 AUDIT, the one-sided gap: DIAGNOSTICS, not a re-measurement and not a tuning pass.

The sealed measurement is what it is. This module never replaces it, never widens the frozen agreement margin, never
selects a variant because it looks better, and never touches the test population, the model, the checkpoint rule, the
batch semantics, the target or the dtype of the scored arrays.

What it does do, for one cell at a time, is reload the retained checkpoint through the author's own evaluation path and
then evaluate the SAME predictions under a small, closed, PREDECLARED set of alternative REDUCTIONS, so that the
question "could the reduction convention explain a systematic +1 to +2 % gap?" is answered with a number instead of a
guess. Every variant is reported; none is adopted; the sealed one is named in every record.

The predeclared variants, and why each is in the set:

  author_full_mean          the sealed reduction: the mean over every window x step x channel element. THE MEASUREMENT.
  batch_means_unweighted    the unweighted mean of per-batch means at the author's own batch size. A codebase that
                            accumulates `total_loss` per batch and divides by the batch count computes this; it differs
                            from the sealed one only through the ragged last batch.
  drop_last_batch           the sealed reduction with the final partial batch removed. Older Time-Series-Library
                            revisions built the TEST loader with drop_last=True; the pinned revision does not, so this
                            variant bounds how much a published number could move if its producer did.
  per_channel_then_mean     the mean per target channel, then the unweighted mean over channels. Identical to the
                            sealed one when every channel has the same element count, which it does here; it is kept as
                            a control that must come out equal, so a non-zero difference would indicate a real defect.
  per_step_then_mean        the mean per forecast step, then the unweighted mean over steps. Same control.

A variant that moves the metric by less than the observed gap does not explain the gap. That is the whole argument, and
it is the reason the variants are fixed in this file before any of them is computed.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402
import df_tsl_repro as R                                               # noqa: E402
import df_tsl_execute as X                                             # noqa: E402
import df_tsl_replay as P                                              # noqa: E402

SCHEMA = "df_tsl_gap_diagnostic.v1"

VARIANTS = ("author_full_mean", "batch_means_unweighted", "drop_last_batch", "per_channel_then_mean", "per_step_then_mean")


def reductions(preds: np.ndarray, trues: np.ndarray, batch_size: int) -> dict:
    """Every predeclared reduction of the SAME arrays, in float32 as the author reduces, with a float64 control."""
    out = {}
    d = None

    def _mae_mse(p, t):
        return {"mae": float(np.mean(np.abs(t - p))), "mse": float(np.mean((t - p) ** 2))}

    out["author_full_mean"] = _mae_mse(preds, trues)
    n = preds.shape[0]
    edges = list(range(0, n, batch_size))
    per_batch = [_mae_mse(preds[i:i + batch_size], trues[i:i + batch_size]) for i in edges]
    out["batch_means_unweighted"] = {"mae": float(np.mean([b["mae"] for b in per_batch])),
                                     "mse": float(np.mean([b["mse"] for b in per_batch])),
                                     "batches": len(per_batch),
                                     "last_batch_windows": n - edges[-1]}
    keep = edges[-1] if (n % batch_size) else n
    out["drop_last_batch"] = {**_mae_mse(preds[:keep], trues[:keep]), "windows_kept": int(keep), "windows_dropped": int(n - keep)}
    ch_mae = [float(np.mean(np.abs(trues[:, :, c] - preds[:, :, c]))) for c in range(preds.shape[2])]
    ch_mse = [float(np.mean((trues[:, :, c] - preds[:, :, c]) ** 2)) for c in range(preds.shape[2])]
    out["per_channel_then_mean"] = {"mae": float(np.mean(ch_mae)), "mse": float(np.mean(ch_mse)), "channels": len(ch_mae)}
    st_mae = [float(np.mean(np.abs(trues[:, s, :] - preds[:, s, :]))) for s in range(preds.shape[1])]
    st_mse = [float(np.mean((trues[:, s, :] - preds[:, s, :]) ** 2)) for s in range(preds.shape[1])]
    out["per_step_then_mean"] = {"mae": float(np.mean(st_mae)), "mse": float(np.mean(st_mse)), "steps": len(st_mae)}
    out["float64_control"] = S.float64_metrics(preds, trues)
    del d
    return out


def gaps(design: dict, lane: Path, record: dict) -> dict:
    """Two denominators, never confused. The PER-SEED difference is what this one cell differs from the published value
    by; the CLASS quantity is the THREE-SEED MEAN's difference, which is what the agreement class is actually defined on
    and what §5 of the return argues about. A variant that beats the first can be a rounding of the second."""
    horizon = int(record["horizon_steps"])
    published = design["lock"]["published"]["per_horizon"][str(horizon)]
    seeds = sorted(int(s) for s in design["seeds"])
    mses, maes, present = [], [], []
    for s in seeds:
        p = lane / "CELLS" / f"{design['dataset']}_{design['protocol']}_h{horizon}_s{s}.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())["metric"]["author_float32"]
        mses.append(float(r["mse"])); maes.append(float(r["mae"])); present.append(s)
    rec = record["metric"]["author_float32"]
    per_seed = {"mse": float(rec["mse"]) - float(published["mse"]), "mae": float(rec["mae"]) - float(published["mae"])}
    mean = ({"mse": float(np.mean(mses)) - float(published["mse"]), "mae": float(np.mean(maes)) - float(published["mae"])}
            if present else None)
    return {"published": published, "per_seed": per_seed, "seeds_used_for_the_mean": present,
            "three_seed_mean": mean,
            "class_quantity": "three_seed_mean",
            "reading": "the agreement class is defined on the three-seed mean; the per-seed difference carries no class"}


def diagnose(design: dict, characterization: dict, record: dict, *, data_path: Path, work: Path, lane: Path,
             gpu: int = 0, require_gpu_uuid: str | None = None, device: str = "cuda") -> dict:
    horizon, seed = int(record["horizon_steps"]), int(record["seed"])
    cell = next(c for c in design["cells"] if c["horizon_steps"] == horizon and c["seed"] == seed)
    cell_dir = lane / f"cell_{design['dataset']}_h{horizon}_s{seed}"
    source_ckpt = cell_dir / "checkpoints" / record["setting"] / "checkpoint.pth"
    work.mkdir(parents=True, exist_ok=True)
    staged = P.stage_checkpoint(source_ckpt, work, record["setting"], record["checkpoint_sha256"])
    S.author_env()
    import functools
    import torch
    cross_device_patch = []
    if device == "cpu":
        # the same operational patch the Electricity lane declares: the author's test(test=1) passes no map_location, so a
        # CUDA-saved checkpoint cannot be READ on a CPU-only process. Placement only; no value and no arithmetic changes.
        torch.load = functools.partial(torch.load, map_location=torch.device("cpu"))
        cross_device_patch = [{"what": "torch.load bound to map_location=cpu for the author's own checkpoint reload",
                               "effect": "placement only: the same stored float32 values, on the CPU"}]
    device_check = S.assert_child_device(require_gpu_uuid, gpu) if device != "cpu" else {"asserted": False}
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.time()
    res = S.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=data_path.parent, data_name=data_path.name,
                             work=work, gpu=gpu, use_gpu=(device != "cpu"), log=work / "diag_stdout.log",
                             train=False, bounded=False, dataloader_workers=0)
    preds, trues = np.asarray(res["preds"]), np.asarray(res["trues"])
    pred_sha = S.sha_array(preds)
    if pred_sha != record["population"]["predictions_sha256"] and device != "cpu":
        raise P.ReplayRefusal("REFUSED: the diagnostic is not looking at the audited predictions")
    red = reductions(preds, trues, int(res["args"]["batch_size"]))
    sealed = red["author_full_mean"]
    rec = record["metric"]["author_float32"]
    gp = gaps(design, lane, record)
    published, gap = gp["published"], (gp["three_seed_mean"] or gp["per_seed"])
    moves = {}
    for v in VARIANTS:
        if v == "author_full_mean":
            continue
        dm, da = red[v]["mse"] - sealed["mse"], red[v]["mae"] - sealed["mae"]
        moves[v] = {"mse": dm, "mae": da,
                    "fraction_of_the_class_gap": {"mse": dm / gap["mse"] if gap["mse"] else None,
                                                  "mae": da / gap["mae"] if gap["mae"] else None},
                    "fraction_of_the_per_seed_gap": {
                        "mse": dm / gp["per_seed"]["mse"] if gp["per_seed"]["mse"] else None,
                        "mae": da / gp["per_seed"]["mae"] if gp["per_seed"]["mae"] else None},
                    "could_explain_the_class_gap": bool(abs(dm) >= abs(gap["mse"]) and abs(da) >= abs(gap["mae"]))}
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    out = {"schema": SCHEMA, "kind": "DIAGNOSTIC_NOT_A_MEASUREMENT",
           "reading": ("alternative reductions of the SAME replayed predictions. The sealed reduction is unchanged and "
                       "remains the measurement; nothing here is selected, adopted or substituted, and the frozen "
                       "agreement margin is untouched"),
           "design_sha256": design["design_sha256"], "cell_id": record["cell_id"], "horizon_steps": horizon,
           "seed": seed, "device_requested": device, "device": res["device"], "device_assertion": device_check,
           "checkpoint_staging": staged,
           "predictions_sha256": pred_sha,
           "predictions_are_the_audited_ones": pred_sha == record["population"]["predictions_sha256"],
           "sealed_reduction": R.contract()["reduction"],
           "batch_size": int(res["args"]["batch_size"]), "windows": int(preds.shape[0]),
           "reductions": red,
           "recorded_metric": rec, "published": published, "gaps": gp,
           "variant_moves_relative_to_the_sealed_reduction": moves,
           "verdict": ("NO_PREDECLARED_REDUCTION_EXPLAINS_THE_CLASS_GAP"
                       if not any(m["could_explain_the_class_gap"] for m in moves.values())
                       else "A_REDUCTION_VARIANT_IS_OF_THE_CLASS_GAP'S_MAGNITUDE_AND_MUST_BE_NAMED"),
           "resources": {"wall_seconds": time.time() - t0,
                         "cpu_seconds": (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime),
                         "whole_cgroup_peak_bytes_in_child": X.cgroup_peak_bytes(),
                         "declared_cap_bytes": X.declared_cap_bytes()},
           "operational_patches": cross_device_patch,
           "environment": S.environment(),
           "reservation": {k: os.environ.get(k) for k in ("CRISPDM_JOB_NAME", "INVOCATION_ID")},
           "started_at": S.now_iso()}
    out["record_sha256"] = S.sha_obj(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="weather")
    ap.add_argument("--protocol", default="L96")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--require-gpu-uuid", default=os.environ.get("CRISPDM_REQUIRED_GPU_UUID"))
    a = ap.parse_args(argv)
    lane, out = Path(a.lane).expanduser(), Path(a.out).expanduser()
    design = json.loads((lane / f"DESIGN.{a.dataset}.{a.protocol}.json").read_text())
    characterization = json.loads((lane / f"CHARACTERIZATION.{a.dataset}.json").read_text())
    cid = f"{a.dataset}_{a.protocol}_h{a.horizon}_s{a.seed}"
    record = json.loads((lane / "CELLS" / f"{cid}.json").read_text())
    rep = diagnose(design, characterization, record, data_path=Path(a.data_path).expanduser(),
                   work=out / f"diag_{a.dataset}_h{a.horizon}_s{a.seed}", lane=lane, gpu=a.gpu,
                   require_gpu_uuid=a.require_gpu_uuid, device=a.device)
    path = out / "DIAGNOSTICS" / f"{cid}.{a.device}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"cell": cid, "verdict": rep["verdict"], "gaps": rep["gaps"],
                      "moves": {k: {"mse": v["mse"], "mae": v["mae"]} for k, v in
                                rep["variant_moves_relative_to_the_sealed_reduction"].items()},
                      "record": str(path)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
