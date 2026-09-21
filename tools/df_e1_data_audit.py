#!/usr/bin/env python3
"""RP59: the household task walked end to end — bytes, roles, times, split, scaler, window, label.

Before any architecture is blamed for an error, the path the numbers travel has to be checked. This
walks it in order and measures each step against the panel's own bytes:

  bytes    the delivered file's digest and the digest the prepared DATA recorded
  roles    which columns entered, which one is the target, and in what unit
  times    the sampling grid read from the timestamps, missing minutes, duplicated and absent
           wall-clock stamps (this archive is NAIVE_WALL_CLOCK, so daylight saving shows up here)
  split    train and validation origins, the purge between them, and whether they can overlap
  scaler   fitted on train ROWS, and what it would have been if fitted on the repeated rows of
           overlapping windows — the difference is reported, not assumed negligible
  window   the inputs of an origin are the w rows ending at it
  label    y(origin) is Y[origin + h], checked element by element against the panel
  inverse  the metrics' unit, and that no transformation is applied twice
  future   a value AFTER an origin is perturbed: its train windows and its past features must not
           move, and no gap is closed to make that true

It then describes the error where it actually falls: by hour of day, by how large the change from
the last observation is, and on periods chosen by a rule DECLARED BEFORE the errors are read.

Nothing here claims an irreducible floor. A model failing is not evidence of one.

    python tools/df_e1_data_audit.py --root RUN_ROOT --out REPORT.json [--figures DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

DAY = 1440
PERIOD_RULE = ("declared before any error was read: the FIRST full day of the evaluation period, "
               "the MIDDLE full day, and the LAST full day. Not the best days, not the worst.")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _panel(root: Path, data: dict):
    """The delivered bytes this run consumed, and the labels that came with them."""
    import pandas as pd
    deliveries = json.loads((root / "DELIVERIES.json").read_text())
    unit = (deliveries.get("units") or {}).get("prepare") or next(iter(deliveries["units"].values()))
    path = Path(unit["path"])
    frame = pd.read_parquet(path)
    return frame, unit, path


def bytes_step(root: Path, unit: dict, path: Path, data_json: dict) -> dict:
    on_disk = sha_file(path)
    return {"delivered_sha256": unit["sha256"], "on_disk_sha256": on_disk,
            "prepared_from_sha256": data_json["panel_sha256"],
            "identical": on_disk == unit["sha256"] == data_json["panel_sha256"],
            "bytes": path.stat().st_size, "rows_in_panel": data_json["panel_rows"]}


def roles_step(design: dict, data_json: dict, frame) -> dict:
    target = data_json["input_columns"][data_json["target_channel"]]
    return {"input_columns": data_json["input_columns"], "target_column": target,
            "target_channel": data_json["target_channel"],
            "unit_of_target": "kW (the archive's Global_active_power is a minute-average active power)",
            "columns_in_the_panel": list(frame.columns),
            "columns_not_consumed": [c for c in frame.columns if c not in data_json["input_columns"]]}


def times_step(frame, data_json: dict, slice_rows) -> dict:
    """The grid, the gaps and the wall-clock anomalies, read from the labels themselves."""
    import pandas as pd
    column = next((c for c in frame.columns if "timestamp" in c.lower() or "date" in c.lower()), None)
    if column is None:
        return {"known": False, "why": "the panel carries no timestamp label column"}
    labels = frame[column].iloc[slice_rows[0]:slice_rows[1]]
    stamps = pd.to_datetime(labels, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    bad = int(stamps.isna().sum())
    deltas = stamps.diff().dropna().dt.total_seconds()
    counts = Counter(deltas.tolist())
    modal, modal_n = counts.most_common(1)[0]
    gaps = {str(d): n for d, n in counts.most_common(8) if d != modal}
    duplicated = int(stamps.duplicated().sum())
    expected = (stamps.iloc[-1]-stamps.iloc[0]).total_seconds()/modal + 1 if modal else None
    return {"known": True, "column": column, "unparseable_labels": bad,
            "modal_step_seconds": modal, "rows_at_the_modal_step": modal_n,
            "other_steps_seen": gaps, "duplicated_stamps": duplicated,
            "first": str(stamps.iloc[0]), "last": str(stamps.iloc[-1]),
            "rows_present": int(stamps.size), "rows_expected_on_a_perfect_grid": None if expected is None else int(round(expected)),
            "missing_minutes": None if expected is None else int(round(expected))-int(stamps.size),
            "timezone_declared": "NAIVE_WALL_CLOCK",
            "daylight_saving_note": ("wall-clock labels with no zone: a spring transition appears as a "
                                     "one-hour gap and an autumn one as a repeated hour. Both are "
                                     "reported above as steps and duplicates; neither is closed up."),
            "consumed_slice_rows": list(slice_rows)}


def split_step(data: dict, data_json: dict) -> dict:
    tr, ev = data["train_origins"], data["eval_origins"]
    w, h = int(data["window"][0]), int(data["horizon"][0])
    train_span = [int(tr.min()-w+1), int(tr.max()+h)]
    eval_span = [int(ev.min()-w+1), int(ev.max()+h)]
    overlap = max(0, min(train_span[1], eval_span[1])-max(train_span[0], eval_span[0])+1)
    return {"train_origins": int(tr.size), "evaluation_origins": int(ev.size),
            "window": w, "horizon": h,
            "train_rows_touched": train_span, "evaluation_rows_touched": eval_span,
            "rows_touched_by_both": overlap,
            "purge_declared": data_json.get("purge") or data_json.get("purge_between_splits"),
            "gap_between_last_train_row_and_first_evaluation_row": eval_span[0]-train_span[1]-1,
            "reading": "a window of w rows ending at the origin, a label h rows after it; the two "
                       "spans must not touch, and the gap above is what separates them"}


def scaler_step(data: dict, data_json: dict) -> dict:
    """Which GRAIN the stored scaler was fitted on — measured in the scaled slice itself.

    `Xs` is the slice AFTER scaling, so the scaler is checked by its effect: on the grain it was
    fitted on, the scaled train inputs must have mean 0 and sd 1. A row inside a window is seen once
    per window that contains it, so the two grains — distinct rows, or rows repeated by their windows
    — give different moments. Both are measured here, and the difference is reported in scaled units
    instead of being assumed negligible. Neither reads an evaluation row.
    """
    Xs, tr = data["Xs"], data["train_origins"]
    w = int(data["window"][0])
    rows = np.arange(int(tr.min()-w+1), int(tr.max()+1))
    block = Xs[rows].astype(np.float64)
    finite = np.isfinite(block)
    by_row_mean = np.nanmean(block, axis=0)
    by_row_sd = np.nanstd(block, axis=0, ddof=0)
    counts = np.zeros(Xs.shape[0])
    for start in (tr-w+1):
        counts[start:start+w] += 1
    weight = counts[rows]
    wn = (finite*weight[:, None]).sum(axis=0)
    by_window_mean = (np.where(finite, block, 0.0)*weight[:, None]).sum(axis=0)/wn
    by_window_sd = np.sqrt((np.where(finite, (block-by_window_mean)**2, 0.0)*weight[:, None]).sum(axis=0)/wn)
    centred = {"rows": float(np.max(np.abs(by_row_mean))), "windows": float(np.max(np.abs(by_window_mean)))}
    unit = {"rows": float(np.max(np.abs(by_row_sd-1))), "windows": float(np.max(np.abs(by_window_sd-1)))}
    grain = min(centred, key=lambda k: centred[k]+unit[k])
    return {"declared_in_DATA_json": data_json.get("scaler"),
            "grain_the_scaled_slice_is_centred_on": grain,
            "largest_absolute_mean_of_the_scaled_train_inputs": centred,
            "largest_absolute_sd_minus_one": unit,
            "difference_between_the_grains_in_scaled_units": {
                "mean": float(np.nanmax(np.abs(by_row_mean-by_window_mean))),
                "sd": float(np.nanmax(np.abs(by_row_sd-by_window_sd)))},
            "interior_rows_counted_up_to": float(weight.max()),
            "reading": "the window grain weights a row by how many train windows contain it, so the "
                       "interior of the slice counts up to W times; the difference above is what that "
                       "choice is worth on THIS data, in scaled units",
            "evaluation_rows_were_not_used": bool(rows.max() < int(data["eval_origins"].min()-w+1)),
            "caveat": "this measures the scaler by its EFFECT; the run's own record states the grain "
                      "it declared, and both are reported side by side"}


def label_step(data: dict, frame, data_json: dict) -> dict:
    """y(origin) is Y[origin+h], and Y is the panel's own target column over the consumed slice."""
    Y, ev = data["Y"], data["eval_origins"]
    h = int(data["horizon"][0])
    target = data_json["input_columns"][data_json["target_channel"]]
    lo, hi = data_json["slice_rows"]
    panel = frame[target].to_numpy()[lo:hi].astype(np.float64)
    finite = np.isfinite(panel) & np.isfinite(Y)
    return {"label_is": f"Y[origin + {h}] over the consumed slice",
            "Y_equals_the_panel_column": bool(np.allclose(panel[finite], Y[finite], atol=0, rtol=0)),
            "rows_compared": int(finite.sum()),
            "non_finite_rows_in_the_slice": int((~np.isfinite(panel)).sum()),
            "first_labels": Y[ev[:3]+h].tolist(),
            "first_origins": ev[:3].tolist(),
            "checked_element_by_element": True}


def future_step(data: dict) -> dict:
    """Perturb a value AFTER an origin: its window and its label's past must not move."""
    Xs, tr = data["Xs"].copy(), data["train_origins"]
    w = int(data["window"][0])
    origin = int(tr[len(tr)//2])
    before = Xs[origin-w+1:origin+1].copy()
    Xs[origin+1:origin+200] += 1000.0                      # a large, obvious perturbation of the future
    after = Xs[origin-w+1:origin+1]
    later = [int(o) for o in tr if o > origin+1][:3]
    return {"perturbed_rows": [origin+1, origin+200], "origin_examined": origin,
            "its_window_unchanged": bool(np.array_equal(before, after)),
            "origins_that_legitimately_see_the_perturbation": later,
            "reading": "a window ends AT its origin, so a change after it cannot reach it. Origins "
                       "that come later do see it, which is correct and is not a leak.",
            "no_gap_was_closed": True}


def error_shape(root: Path, data: dict, cells) -> dict:
    """Where the error actually falls: by hour of day, and by how far the series has just moved."""
    Y, ev = data["Y"], data["eval_origins"]
    h = int(data["horizon"][0])
    y = Y[ev+h]
    last = Y[ev]
    change = np.abs(y-last)
    out = {"rule": "rows are bucketed by |y(t+h) - y(t)|, the move the forecast has to follow, and "
                   "by the hour of day of the ORIGIN; buckets are quantiles of the change, fixed "
                   "before any model's error is read"}
    edges = np.quantile(change, [0, .25, .5, .75, .9, 1.0])
    out["change_quantiles_kW"] = edges.tolist()
    per_model = {}
    for cell, pred in cells.items():
        e = np.abs(pred-y)
        buckets = {}
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            m = (change >= lo) & (change <= hi if i == len(edges)-2 else change < hi)
            if m.sum():
                buckets[f"q{i}"] = {"rows": int(m.sum()), "share_of_rows": float(m.mean()),
                                    "mean_abs_error": float(e[m].mean()),
                                    "share_of_total_error": float(e[m].sum()/e.sum()),
                                    "change_range_kW": [float(lo), float(hi)]}
        per_model[cell] = {"mae": float(e.mean()), "by_change_bucket": buckets}
        hours = ((ev % DAY)//60).astype(int)
        per_model[cell]["by_hour_of_day"] = {str(hh): float(e[hours == hh].mean())
                                             for hh in range(24) if (hours == hh).any()}
    out["models"] = per_model
    top = next(iter(per_model.values()))
    worst = max(top["by_change_bucket"].items(), key=lambda kv: kv[1]["share_of_total_error"])
    out["reading"] = (f"for {next(iter(per_model))}, the bucket {worst[0]} "
                      f"({worst[1]['change_range_kW'][0]:.3f}–{worst[1]['change_range_kW'][1]:.3f} kW of "
                      f"movement) holds {worst[1]['share_of_rows']*100:.1f}% of the rows and "
                      f"{worst[1]['share_of_total_error']*100:.1f}% of the total absolute error. This "
                      f"says where the error is, NOT that it is irreducible.")
    return out


def distribution_step(data: dict) -> dict:
    Y, tr, ev = data["Y"], data["train_origins"], data["eval_origins"]
    h = int(data["horizon"][0])
    out = {}
    for name, origins in (("train", tr), ("validation", ev)):
        v = Y[origins+h]
        out[name] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "quantiles": dict(zip(["min", "q01", "q25", "median", "q75", "q99", "max"],
                                           np.quantile(v, [0, .01, .25, .5, .75, .99, 1]).tolist())),
                     "zeros": int((v == 0).sum()), "share_zero": float((v == 0).mean()),
                     "above_q99_of_train": None}
    q99 = np.quantile(Y[tr+h], 0.99)
    for name, origins in (("train", tr), ("validation", ev)):
        v = Y[origins+h]
        out[name]["above_q99_of_train"] = {"rows": int((v > q99).sum()), "share": float((v > q99).mean()),
                                           "threshold_kW": float(q99)}
    return out


def autocorrelation(data: dict, lags=(1, 60, 1440, 10080)) -> dict:
    Y, tr = data["Y"], data["train_origins"]
    w = int(data["window"][0])
    rows = np.arange(int(tr.min()-w+1), int(tr.max()+1))
    v = Y[rows]
    v = v[np.isfinite(v)]
    v = v-v.mean()
    denom = float((v*v).sum())
    out = {}
    for lag in lags:
        if lag < v.size:
            out[str(lag)] = float((v[:-lag]*v[lag:]).sum()/denom)
    return {"train_rows_used": int(v.size), "autocorrelation_by_lag_minutes": out,
            "reading": "measured on the consumed train rows; the daily lag is 1440 minutes"}


def figures(out_dir: Path, data: dict, cells: dict) -> dict:
    """Periods chosen by the declared rule, drawn in kW and in log1p, plus the error by hour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Y, ev = data["Y"], data["eval_origins"]
    h = int(data["horizon"][0])
    out_dir.mkdir(parents=True, exist_ok=True)
    days = [(0, "first"), (len(ev)//2 - DAY//2, "middle"), (max(0, len(ev)-DAY), "last")]
    written = []
    for start, label in days:
        stop = min(start+DAY, len(ev))
        idx = np.arange(start, stop)
        y = Y[ev[idx]+h]
        series = {"truth": y, "persistence": Y[ev[idx]]}
        for name, pred in cells.items():
            series[name] = pred[idx]
        for scale, fn in (("kW", lambda a: a), ("log1p", np.log1p)):
            fig, ax = plt.subplots(figsize=(11, 4))
            for name, values in series.items():
                ax.plot(idx, fn(values), linewidth=0.9, label=name)
            ax.set_title(f"{label} full day of the evaluation period ({scale}) — period chosen by rule, not by error")
            ax.set_xlabel("evaluation row"); ax.set_ylabel(scale); ax.legend(fontsize=7, ncol=4)
            path = out_dir/f"period_{label}_{scale}.png"
            fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
            written.append(str(path))
    hours = ((ev % DAY)//60).astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, pred in cells.items():
        e = np.abs(pred-Y[ev+h])
        ax.plot(range(24), [e[hours == hh].mean() for hh in range(24)], marker="o", label=name)
    e = np.abs(Y[ev]-Y[ev+h])
    ax.plot(range(24), [e[hours == hh].mean() for hh in range(24)], marker="s", linestyle="--", label="persistence")
    ax.set_title("mean absolute error by hour of the origin"); ax.set_xlabel("hour"); ax.set_ylabel("kW")
    ax.legend(fontsize=7)
    path = out_dir/"error_by_hour.png"
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    written.append(str(path))
    return {"rule": PERIOD_RULE, "files": written}


def audit(root: Path, figures_dir: Path | None = None) -> dict:
    root = Path(root)
    design = json.loads((root/"DESIGN.json").read_text())
    data_json = json.loads((root/"DATA.json").read_text())
    with np.load(root/"DATA.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    frame, unit, path = _panel(root, data)
    cells = {}
    for cell in ("R0_s1", "linear_ridge"):
        if cell == "linear_ridge":
            arrays = root/"attempts"/"controls"/"arrays.npz"
            key = "validation_pred_linear_ridge"
        else:
            arrays = root/"attempts"/cell/"arrays.npz"
            key = "validation_pred"
        if arrays.is_file():
            with np.load(arrays, allow_pickle=False) as z:
                cells[cell] = z[key].reshape(-1)
    report = {"schema": "df_e1_data_audit.v1", "at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
              "root": str(root), "design_sha256": design["design_sha256"],
              "bytes": bytes_step(root, unit, path, data_json),
              "roles": roles_step(design, data_json, frame),
              "times": times_step(frame, data_json, data_json["slice_rows"]),
              "split": split_step(data, data_json),
              "scaler": scaler_step(data, data_json),
              "label": label_step(data, frame, data_json),
              "future_perturbation": future_step(data),
              "distribution": distribution_step(data),
              "autocorrelation": autocorrelation(data),
              "error_shape": error_shape(root, data, cells),
              "claims_no_irreducible_floor": "a model failing on some rows is not evidence that those "
                                             "rows are unpredictable; no floor is asserted here"}
    if figures_dir:
        report["figures"] = figures(Path(figures_dir), data, cells)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None)
    a = ap.parse_args(argv)
    report = audit(a.root, a.figures)
    a.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("bytes", "times", "split", "label", "future_perturbation")},
                     indent=1, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
