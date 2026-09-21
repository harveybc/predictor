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


LEGACY_ERRATUM = {
    "field": "mase_mean",
    "published_as": "MASE",
    "what_it_actually_is": "the validation MAE divided by the mean absolute h-step change over the "
                           "train rows consumed (h = the forecast horizon, 60 minutes)",
    "correct_name": "persistence_scaled_error_horizon_train",
    "version": "successor v2 names it correctly; the historical field is kept unchanged so published "
               "numbers stay readable, and this erratum travels with it",
    "why_it_matters": "a conventional MASE uses a seasonal period m declared on its own grounds, so "
                      "0.8875 is not comparable with a published MASE of another study",
}

SCALED_ERROR_NAMES = {
    "persistence_scaled_error_horizon_train": {
        "formula": "mean(|prediction - target|) over the evaluation rows / mean(|Y[t+h] - Y[t]|) over "
                   "the train origins",
        "denominator_origin": "the train origins of THIS run, never validation or test",
        "m": "not a seasonal period: the horizon h=60 steps",
        "gaps": "train origins are the admissible ones; nothing is interpolated",
    },
    "conventional_mase_m1_train_slice": {
        "formula": "MAE / mean(|Y[t] - Y[t-1]|) over the consumed train slice",
        "m": 1,
        "justification": "the one-step naive of Hyndman/Koehler",
        "gaps": "non-finite consecutive pairs are EXCLUDED and counted, never closed up",
    },
    "conventional_mase_m1440_train_slice": {
        "formula": "MAE / mean(|Y[t] - Y[t-m]|) over the consumed train slice",
        "m": 1440,
        "justification": "one day at one-minute sampling: household load has a daily cycle, which is "
                         "an independent seasonal ground and NOT the horizon",
        "gaps": "non-finite seasonal pairs are EXCLUDED and counted, never closed up",
    },
}


def _local_report_check(root, raw):
    """The run's own RESULTS.json against this recomputation: any difference is kept, not smoothed."""
    path = Path(root)/"RESULTS.json"
    if not path.is_file():
        return {"compared": False, "why": "the run holds no RESULTS.json"}
    doc = json.loads(path.read_text())
    rows, differences = {}, {}
    for cell, entry in (doc.get("cells") or {}).items():
        if "mae" not in entry or cell not in raw:
            continue
        d_mae = float(entry["mae"])-raw[cell]["mae"]
        d_scaled = float(entry["mase"])-raw[cell]["persistence_scaled_error_horizon_train"]
        rows[cell] = {"local_mae": float(entry["mae"]), "recomputed_mae": raw[cell]["mae"],
                      "difference_mae": d_mae,
                      "local_scaled": float(entry["mase"]),
                      "recomputed_scaled": raw[cell]["persistence_scaled_error_horizon_train"],
                      "difference_scaled": d_scaled}
        if d_mae or d_scaled:
            differences[cell] = rows[cell]
    return {"compared": True, "file": str(path), "cells": rows, "differences": differences,
            "identical": not differences,
            "rule": "the exact difference is preserved; a mismatch is reported, never averaged away"}


def compare(y, predictions, *, reference, references=()):
    """Every method on the SAME rows: error, dispersion, and skill against EACH declared reference.

    RP57: one reference was not enough to read the table — a method that beats persistence may lose
    to the linear control on the same rows. A reference whose own error is zero yields an undefined
    skill, never an epsilon that manufactures a win.
    """
    y = _vector(y)
    if reference not in predictions:
        raise ValueError("reference missing")
    wanted = [reference] + [r for r in references if r != reference]
    missing = [r for r in wanted if r not in predictions]
    if missing:
        raise ValueError(f"reference missing: {missing}")
    rows, abs_errors = {}, {}
    for name, value in predictions.items():
        p = _vector(value)
        if p.shape != y.shape:
            raise ValueError("prediction and target population differ")
        e = p-y
        a = np.abs(e)
        abs_errors[name] = a
        rows[name] = {"n": int(y.size), "mae": float(np.mean(a)),
                      "rmse": float(np.sqrt(np.mean(e*e))), "bias": float(np.mean(e)),
                      "abs_error_sd": float(np.std(a, ddof=1)) if a.size > 1 else None,
                      "abs_error_median": float(np.median(a)),
                      "abs_error_q90": float(np.quantile(a, 0.9)),
                      "dispersion_note": "spread of the error over rows; these rows are overlapping "
                                         "windows, so no iid interval is computed from them"}
    baseline = rows[reference]["mae"]
    for row in rows.values():
        row["mae_skill_percent"] = 100*(1-row["mae"]/baseline) if baseline > 0 else None
        row["skill_status"] = "MEASURED" if baseline > 0 else "ZERO_REFERENCE_ERROR"
    for name, row in rows.items():
        skills = {}
        for ref in wanted:
            base = rows[ref]["mae"]
            skills[ref] = {"mae_skill_percent": 100*(1-row["mae"]/base) if base > 0 else None,
                           "status": "MEASURED" if base > 0 else "ZERO_REFERENCE_ERROR",
                           "paired": paired_difference(abs_errors[name], abs_errors[ref])}
        row["versus"] = skills
    return rows


def paired_difference(a, b):
    """The difference of absolute errors ROW BY ROW: the same rows for both, nothing dropped."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError("paired difference needs the same nonempty population")
    d = a-b
    return {"n": int(d.size), "mean_abs_error_difference": float(np.mean(d)),
            "sd_of_difference": float(np.std(d, ddof=1)) if d.size > 1 else None,
            "rows_first_better": int(np.count_nonzero(d < 0)),
            "rows_equal": int(np.count_nonzero(d == 0)),
            "reading": "negative means the first method has the smaller absolute error on average"}


def temporal_blocks(y, predictions, *, block_rows, reference, arms):
    """Paired differences by contiguous blocks of rows, because overlapping windows are not replicas.

    Ten thousand windows that share 59 of 60 minutes are not ten thousand independent situations.
    Contiguous blocks are reported so a difference that holds only in one stretch of the validation
    period is visible as such instead of being averaged away.
    """
    y = _vector(y)
    n = y.size
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    out = []
    for start in range(0, n, block_rows):
        stop = min(start+block_rows, n)
        if stop-start < 2:
            continue
        piece = {name: _vector(v)[start:stop] for name, v in predictions.items()}
        block = {"rows": [start, stop], "n": stop-start,
                 "mae": {name: float(np.mean(np.abs(p-y[start:stop]))) for name, p in piece.items()}}
        block["paired"] = {f"{a}_minus_{b}": paired_difference(np.abs(piece[a]-y[start:stop]),
                                                               np.abs(piece[b]-y[start:stop]))
                           ["mean_abs_error_difference"]
                           for a, b in arms if a in piece and b in piece}
        block["best"] = min(block["mae"], key=block["mae"].get)
        out.append(block)
    return {"block_rows": block_rows, "blocks": out, "reference": reference,
            "rule": "contiguous blocks of the validation period; overlapping windows are not iid replicas"}


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
    references = ("persistence", "seasonal_naive_daily", "linear_ridge")
    raw = compare(y, predictions, reference="persistence", references=references)
    for cell in cells[:-1]:
        saved = records[cell]["scores"]["validation"]["model"]
        if not np.isclose(raw[cell]["mae"], saved["mae_mean"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{cell}: stored MAE differs from arrays")
        if not np.isclose(raw[cell]["mae"]/denominator, saved["mase_mean"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{cell}: stored scaled error differs")
    scales = {"raw_kW": raw,
              "zscore_train": compare((y-m)/s, {k:(v-m)/s for k,v in predictions.items()},
                                      reference="persistence", references=references)}
    log_domain = min(float(np.min(v)) for v in [y, *predictions.values()])
    if log_domain > -1:
        scales["log1p_of_kW"] = compare(np.log1p(y), {k:np.log1p(v) for k,v in predictions.items()},
                                        reference="persistence", references=references)
        log_status = {"applied": True, "smallest_value_seen": log_domain}
    else:
        log_status = {"applied": False, "smallest_value_seen": log_domain,
                      "why": "log1p needs every value above -1; a forecast below it would be undefined, "
                             "and clipping it would invent a number"}
    train_start, train_end = int(tr.min()-w+1), int(tr.max()+h+1)          # the train rows actually consumed
    train_y = data["Y"][train_start:train_end]
    diffs = np.abs(np.diff(train_y))
    valid = np.isfinite(diffs)
    one_step = float(np.mean(diffs[valid]))
    season = 1440                                     # minutes in a day: the load's own cycle, not the horizon
    seasonal_diffs = np.abs(train_y[season:]-train_y[:-season])
    seasonal_valid = np.isfinite(seasonal_diffs)
    one_season = float(np.mean(seasonal_diffs[seasonal_valid])) if seasonal_valid.any() else None
    for row in raw.values():
        # RP57: the successor's own name for the published ratio, and the errata that goes with it
        row["persistence_scaled_error_horizon_train"] = row["mae"]/denominator if denominator > 0 else None
        row["legacy_horizon_scaled_error"] = row["persistence_scaled_error_horizon_train"]
        row["legacy_field_erratum"] = LEGACY_ERRATUM
        row["conventional_mase_m1_train_slice"] = row["mae"]/one_step if one_step > 0 else None
        row["conventional_mase_m1440_train_slice"] = (row["mae"]/one_season
                                                      if one_season and one_season > 0 else None)
    grouped = {}
    for r in ("R0", "R1", "R2"):
        grouped[r] = {
            scale: {**{key: float(np.mean([table[f"{r}_s{s}"][key] for s in (1,2,3)]))
                       for key in ("mae","rmse","bias","mae_skill_percent")},
                    "mae_sd_across_seeds": float(np.std([table[f"{r}_s{s}"]["mae"] for s in (1,2,3)], ddof=1)),
                    "seeds": 3}
            for scale,table in scales.items()
        }
    arm_pairs = [(f"R1_s{s}", f"R0_s{s}") for s in (1,2,3)] + \
                [(f"R2_s{s}", f"R0_s{s}") for s in (1,2,3)] + \
                [(f"R2_s{s}", f"R1_s{s}") for s in (1,2,3)] + \
                [(f"R0_s{s}", "linear_ridge") for s in (1,2,3)]
    paired = {f"{a}_minus_{b}": paired_difference(np.abs(predictions[a]-y), np.abs(predictions[b]-y))
              for a, b in arm_pairs}
    blocks = temporal_blocks(y, predictions, block_rows=1440, reference="persistence",
                             arms=[(f"R1_s{s}", f"R0_s{s}") for s in (1,2,3)] +
                                  [(f"R0_s{s}", "linear_ridge") for s in (1,2,3)])
    diagnostic, costs = {}, {}
    for cell in cells[:-1]:
        rec = records[cell]
        diagnostic[cell] = {"training": rec.get("training"), "parameters": rec.get("parameters")}
    for cell in cells:
        outcome = root/"attempts"/cell/"outcome.json"
        cost = json.loads(outcome.read_text()).get("summary", {}).get("cost", {}) if outcome.is_file() else {}
        costs[cell] = {"cpu_seconds": cost.get("cpu_seconds"), "wall_seconds": cost.get("wall_seconds"),
                       "peak_rss_bytes": cost.get("peak_rss_bytes")}
    for seed in (1, 2, 3):
        ae = root/"attempts"/f"ae_s{seed}"/"outcome.json"
        if ae.is_file():
            c = json.loads(ae.read_text()).get("summary", {}).get("cost", {})
            costs[f"ae_s{seed}"] = {"cpu_seconds": c.get("cpu_seconds"),
                                    "amortised_over": [f"R1_s{seed}", f"R2_s{seed}"],
                                    "note": "the auto-encoder is charged once to the two arms that share it"}
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
            "paired_differences":paired, "temporal_blocks":blocks, "costs":costs,
            "log1p_domain":log_status, "scaled_error_names":SCALED_ERROR_NAMES,
            "local_report_check":_local_report_check(root, raw),
            "source_sha256":used, "sources_unchanged":True}


def markdown(result):
    """The same table a person can read, with every reference in it."""
    lines = ["# Forecast comparison — validation, recomputed from arrays", "",
             f"n = {result['n']} evaluation rows, window {result['window']}, horizon {result['horizon']} "
             f"(one-minute sampling). Scope: {result['scope']}.", "",
             "Improvement is 100·(1 − MAE_method / MAE_reference) on the SAME rows. It is not a hit rate.",
             ""]
    for scale, table in result["by_scale"].items():
        lines += [f"## {scale}", "",
                  "| method | n | MAE | RMSE | bias | vs persistence | vs seasonal naive | vs linear ridge |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in table.items():
            def pct(ref):
                v = row["versus"][ref]["mae_skill_percent"] if "versus" in row else None
                return "undefined" if v is None else f"{v:+.2f}%"
            lines.append(f"| {name} | {row['n']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['bias']:+.6f} | "
                         f"{pct('persistence')} | {pct('seasonal_naive_daily')} | {pct('linear_ridge')} |")
        lines.append("")
    lines += ["## Scaled errors (raw kW)", "",
              "| method | persistence-scaled (horizon, train) | MASE m=1 | MASE m=1440 |",
              "|---|---:|---:|---:|"]
    for name, row in result["by_scale"]["raw_kW"].items():
        def num(key):
            v = row.get(key)
            return "undefined" if v is None else f"{v:.4f}"
        lines.append(f"| {name} | {num('persistence_scaled_error_horizon_train')} | "
                     f"{num('conventional_mase_m1_train_slice')} | {num('conventional_mase_m1440_train_slice')} |")
    lines += ["", "The first column is what earlier reports published as MASE. It is not one: see the erratum "
              "in the JSON report.", "", "## Paired differences (absolute error, row by row)", "",
              "| pair | mean difference | sd | rows where the first is better (of n) |", "|---|---:|---:|---:|"]
    for pair, d in result["paired_differences"].items():
        sd = "—" if d["sd_of_difference"] is None else f"{d['sd_of_difference']:.4f}"
        lines.append(f"| {pair} | {d['mean_abs_error_difference']:+.6f} | {sd} | {d['rows_first_better']} / {d['n']} |")
    lines += ["", "Negative means the first method has the smaller absolute error. These rows are overlapping "
              "windows, so no independent-sample interval is computed from them.", "",
              "## By contiguous block of the validation period", "",
              "| rows | n | best method | " + " | ".join(result["temporal_blocks"]["blocks"][0]["paired"]) + " |",
              "|---|---:|---|" + "---:|"*len(result["temporal_blocks"]["blocks"][0]["paired"])]
    for blk in result["temporal_blocks"]["blocks"]:
        lines.append(f"| {blk['rows'][0]}–{blk['rows'][1]} | {blk['n']} | {blk['best']} | "
                     + " | ".join(f"{v:+.4f}" for v in blk["paired"].values()) + " |")
    lines += ["", "## Cost", "", "| unit | CPU s | wall s | peak RSS MiB |", "|---|---:|---:|---:|"]
    for unit, c in result["costs"].items():
        rss = "—" if not c.get("peak_rss_bytes") else f"{c['peak_rss_bytes']/2**20:.0f}"
        lines.append(f"| {unit} | {c.get('cpu_seconds') or '—'} | {c.get('wall_seconds') or '—'} | {rss} |")
    check = result.get("local_report_check", {})
    if check.get("compared"):
        lines += ["", "## Against the run's own report", "",
                  ("Identical for every cell." if check["identical"] else
                   "DIFFERENT — the exact differences are preserved in the JSON report: "
                   + ", ".join(sorted(check["differences"])))]
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--markdown", type=Path)
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
    if args.markdown:
        md = args.markdown.resolve()
        if md.is_relative_to(root):
            raise ValueError("report must be outside preserved run")
        with md.open("x") as f:
            f.write(markdown(result))
    print(json.dumps({"n":result["n"],"regime_means":result["regime_means"],"scale":result["scale"]},indent=2))


if __name__ == "__main__":
    main()
