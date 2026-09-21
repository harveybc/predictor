"""Read-only forecast comparison; no training, selection or campaign approval."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _vector(value):
    a = np.asarray(value)
    if a.ndim != 1 or not a.size or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("expected a nonempty finite numeric vector")
    return a.astype(np.float64)


def compare(y, predictions, *, reference):
    y = _vector(y)
    if reference not in predictions:
        raise ValueError("reference missing")
    rows = {}
    for name, value in predictions.items():
        p = _vector(value)
        if p.shape != y.shape:
            raise ValueError("prediction and target population differ")
        e = p-y
        rows[name] = {"n": int(y.size), "mae": float(np.mean(np.abs(e))),
                      "rmse": float(np.sqrt(np.mean(e*e))), "bias": float(np.mean(e))}
    baseline = rows[reference]["mae"]
    for row in rows.values():
        row["mae_skill_percent"] = 100*(1-row["mae"]/baseline) if baseline > 0 else None
        row["skill_status"] = "MEASURED" if baseline > 0 else "ZERO_REFERENCE_ERROR"
    return rows


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def legacy_comparison(path):
    """Compare published summary rows only; no assertion of training provenance."""
    with path.open(newline="") as f:
        records = list(csv.DictReader(f))
    rows = {r["Metric"]: float(r["Average"]) for r in records}
    pairs = []
    for key, value in rows.items():
        if key.startswith("Test MAE H"):
            h = key.removeprefix("Test MAE H")
            naive = rows.get(f"Test Naive MAE H{h}")
            if naive is not None:
                pairs.append({"horizon_label":h, "model_mae":value, "naive_mae":naive,
                              "mae_skill_percent":100*(1-value/naive) if naive > 0 else None})
    return {"file":path.name, "sha256":digest(path), "pairs":pairs,
            "scope":"PUBLISHED_ROUNDED_SUMMARIES_ONLY; scale/row identity and producing configuration not independently reconstructed"}


def audit_run(root):
    """Recompute saved validation predictions only; original sources never edited."""
    used = {}

    def pin(path):
        used[str(path.relative_to(root))] = digest(path)
        return path

    with np.load(pin(root/"DATA.npz"), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    h = int(data["horizon"][0])
    w = int(data["window"][0])
    j = int(data["target_channel"][0])
    origins = data["eval_origins"]
    if np.unique(origins).size != origins.size or np.any(np.diff(origins) <= 0):
        raise ValueError("evaluation origins not unique and increasing")
    y = _vector(data["Y"][origins+h])
    tr = data["train_origins"]
    denominator = float(np.mean(np.abs(data["Y"][tr+h]-data["Y"][tr])))
    if not np.isclose(denominator, data["denominator"][0], rtol=1e-12, atol=1e-12):
        raise ValueError("saved denominator does not match train horizon persistence")
    predictions = {"persistence": data["Y"][origins],
                   "seasonal_naive_daily": data["Y"][origins+h-1440]}
    records = {}
    cells = [f"{r}_s{s}" for r in ("R0", "R1", "R2") for s in (1, 2, 3)] + ["controls"]
    for cell in cells:
        d = root/"attempts"/cell
        with np.load(pin(d/"arrays.npz"), allow_pickle=False) as z:
            if not np.array_equal(z["eval_origins"], origins):
                raise ValueError(f"{cell}: origin identity differs")
            if not np.array_equal(z["validation_y"].reshape(-1), y):
                raise ValueError(f"{cell}: targets differ from prepared DATA")
            if not np.array_equal(z["denominator"], data["denominator"]):
                raise ValueError(f"{cell}: denominator differs")
            if cell == "controls":
                for name in ("persistence", "seasonal_naive_daily"):
                    if not np.array_equal(z[f"validation_pred_{name}"].reshape(-1), predictions[name]):
                        raise ValueError(f"{name}: saved control differs from direct reconstruction")
                predictions["linear_ridge"] = z["validation_pred_linear_ridge"].reshape(-1)
            else:
                predictions[cell] = z["validation_pred"].reshape(-1)
        records[cell] = json.loads(pin(d/"cell.json").read_text())
    predictions = {k:_vector(v) for k,v in predictions.items()}
    m, s = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    if s <= 0 or not np.isfinite(s):
        raise ValueError("invalid target scale")
    raw = compare(y, predictions, reference="persistence")
    for cell in cells[:-1]:
        saved = records[cell]["scores"]["validation"]["model"]
        if not np.isclose(raw[cell]["mae"], saved["mae_mean"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{cell}: stored MAE differs from arrays")
        if not np.isclose(raw[cell]["mae"]/denominator, saved["mase_mean"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{cell}: stored scaled error differs")
    scales = {"raw_kW": raw,
              "zscore_train": compare((y-m)/s, {k:(v-m)/s for k,v in predictions.items()}, reference="persistence")}
    if min(float(np.min(v)) for v in [y, *predictions.values()]) > -1:
        scales["log1p_of_kW"] = compare(np.log1p(y), {k:np.log1p(v) for k,v in predictions.items()}, reference="persistence")
    train_start, train_end = int(tr.min()-w+1), int(tr.max()+h+1)
    train_y = data["Y"][train_start:train_end]
    diffs = np.abs(np.diff(train_y))
    valid = np.isfinite(diffs)
    one_step = float(np.mean(diffs[valid]))
    for row in raw.values():
        row["legacy_horizon_scaled_error"] = row["mae"]/denominator
        row["conventional_mase_m1_train_slice"] = row["mae"]/one_step if one_step > 0 else None
    grouped = {}
    for r in ("R0", "R1", "R2"):
        grouped[r] = {
            scale: {key: float(np.mean([table[f"{r}_s{s}"][key] for s in (1,2,3)]))
                    for key in ("mae","rmse","bias","mae_skill_percent")}
            for scale,table in scales.items()
        }
    diagnostic = {}
    for cell in cells[:-1]:
        rec = records[cell]
        diagnostic[cell] = {"training": rec.get("training"), "parameters": rec.get("parameters")}
    changed = [name for name,sha in used.items() if digest(root/name) != sha]
    if changed:
        raise ValueError(f"source files changed during inspection: {changed}")
    return {"scope": "LOCAL_READ_ONLY_REVIEW_OF_PRESERVED_VALIDATION_ARRAYS_NOT_NEW_TRAINING_OR_SCIENTIFIC_APPROVAL",
            "n": int(y.size), "window":w, "horizon":h,
            "scale": {"target_mean":m, "target_sd":s,
                      "train_horizon_persistence":denominator, "train_one_step":one_step,
                      "train_slice":[train_start,train_end], "one_step_pairs_used":int(valid.sum()),
                      "one_step_pairs_excluded_nonfinite":int((~valid).sum()),
                      "log1p_unit_convention":"log(1 + target expressed numerically in kW); existing forecasts transformed, not models trained in log space"},
            "target_quantiles": dict(zip(["min","q25","median","q75","max"],np.quantile(y,[0,.25,.5,.75,1]).tolist())),
            "by_scale":scales, "regime_means":grouped, "training_diagnostics":diagnostic,
            "source_sha256":used, "sources_unchanged":True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--legacy-csv", type=Path)
    args = p.parse_args()
    root = args.run.resolve()
    out = args.out.resolve()
    if out.is_relative_to(root):
        raise ValueError("report must be outside preserved run")
    result = audit_run(root)
    if args.legacy_csv:
        result["legacy_summary"] = legacy_comparison(args.legacy_csv)
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"n":result["n"],"regime_means":result["regime_means"],"scale":result["scale"]},indent=2))


if __name__ == "__main__":
    main()
