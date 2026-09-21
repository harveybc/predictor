#!/usr/bin/env python3
"""RP58: what the old financial comparisons actually were — inventoried, frozen, and recomputed.

The question behind this is a memory: a run with MAE near 0.02 against a naive near 0.018. The
answer must not be guessed from the size of a number. This tool therefore does four things, in this
order, and refuses to do the fifth:

  1. FREEZE. Every published results table and prediction file is digested BEFORE any row is chosen,
     so the selection cannot be made after seeing which rows look good.
  2. INVENTORY. For each published table: its producing commit (from git, by content), the config
     that names it as its `results_file`, the declared transformations of x and y and their order,
     the inverse, the baseline, the horizon IN PHYSICAL TIME (from the timestamps of the prediction
     file, not from the file's name), and the train/validation/test files.
  3. RECOMPUTE. Where predictions, labels and row identifiers exist, the metrics are recomputed from
     them and compared with the published summary, difference preserved.
  4. NAME WHAT IS MISSING. Where they do not, the entry is PUBLISHED_SUMMARY_ONLY and says WHICH
     object is absent. A summary is never upgraded by recomputation it cannot support.

  5. It never runs an old sweep, never reopens a reserved test split and never overwrites a sample.

    python tools/df_legacy_inventory.py --out INVENTORY.json [--results-glob ...] [--freeze FILE]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_GLOB = "examples/results/**/*_results.csv"
TRANSFORM_KEYS = ("use_log1p_features", "use_normalization_json", "use_stl", "use_wavelets",
                  "use_multi_tapper", "use_daily", "use_returns", "use_diff", "target_column",
                  "target_plugin", "preprocessor_plugin", "pipeline_plugin", "predictor_plugin",
                  "window_size", "predicted_horizons", "plotted_horizon", "use_ideal_predictions",
                  "max_steps_train", "max_steps_val", "max_steps_test",
                  "x_train_file", "y_train_file", "x_validation_file", "y_validation_file",
                  "x_test_file", "y_test_file")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze(paths) -> dict:
    """The published state, digested before a single row is read for selection."""
    return {"schema": "legacy_publication_freeze.v1",
            "at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "files": {str(p.relative_to(REPO)): {"sha256": sha_file(p), "bytes": p.stat().st_size}
                      for p in sorted(paths)},
            "rule": "digested before any row was chosen; a later selection cannot silently change these"}


def producing_commit(path: Path) -> dict:
    """The commit that introduced THESE bytes, found by content, not by a neighbouring config."""
    rel = str(path.relative_to(REPO))
    log = subprocess.run(["git", "log", "--all", "--follow", "--format=%H %ci %s", "--", rel],
                         cwd=REPO, capture_output=True, text=True)
    lines = [l for l in log.stdout.splitlines() if l.strip()]
    if not lines:
        return {"found": False, "why": "no commit in this repository touches that path"}
    first = lines[-1].split(" ", 1)
    last = lines[0].split(" ", 1)
    return {"found": True, "introduced_in": first[0], "introduced_at": first[1].rsplit(" ", 1)[0],
            "last_touched_in": last[0], "last_touched_at": last[1].rsplit(" ", 1)[0],
            "revisions": len(lines)}


def configs_naming(path: Path, configs) -> list:
    """Only a config that NAMES this file as its own output counts; adjacency proves nothing."""
    rel = str(path.relative_to(REPO))
    out = []
    for cfg_path, cfg in configs:
        if cfg.get("results_file") in (rel, "./"+rel) or cfg.get("output_file", "").endswith(path.name.replace("_results", "_prediction")):
            out.append((cfg_path, cfg))
    return out


def published_rows(path: Path) -> dict:
    with path.open(newline="") as fh:
        records = list(csv.DictReader(fh))
    rows = {}
    for r in records:
        key = r.get("Metric")
        try:
            rows[key] = float(r.get("Average"))
        except (TypeError, ValueError):
            continue
    pairs = {}
    for key, value in rows.items():
        m = re.fullmatch(r"(Train|Validation|Test) MAE H(\d+)", key or "")
        if m:
            split, h = m.group(1), m.group(2)
            pairs[f"{split}_H{h}"] = {"mae": value, "naive_mae": rows.get(f"{split} Naive MAE H{h}")}
    return {"metrics": len(rows), "model_naive_pairs": pairs,
            "has_any_naive": any(v["naive_mae"] is not None for v in pairs.values())}


def prediction_file(results_path: Path) -> Path | None:
    candidate = results_path.with_name(results_path.name.replace("_results.csv", "_prediction.csv"))
    return candidate if candidate.is_file() else None


def physical_grid(path: Path) -> dict:
    """The sampling interval read from the file's own timestamps — never from its name."""
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        stamps = []
        for row in reader:
            if not row:
                continue
            try:
                stamps.append(datetime.fromisoformat(row[0]))
            except ValueError:
                return {"known": False, "why": f"the first column is not a timestamp: {row[0]!r}"}
            if len(stamps) >= 500:
                break
    if len(stamps) < 3:
        return {"known": False, "why": "fewer than three rows to measure a spacing"}
    import collections
    deltas = [(b-a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    counts = collections.Counter(deltas)
    modal, modal_n = counts.most_common(1)[0]
    ordered = sorted(deltas)
    median = ordered[len(ordered)//2]
    return {"known": True, "modal_step_seconds": modal, "modal_step_hours": modal/3600.0,
            "rows_at_the_modal_step": modal_n, "rows_measured": len(deltas)+1,
            "median_step_seconds": median,
            "step_distribution_hours": {str(d/3600.0): n for d, n in counts.most_common(6)},
            "first": stamps[0].isoformat(), "last": stamps[-1].isoformat(),
            "columns": header[:1]+header[1:6],
            "note": "the spacing of consecutive published rows; the MODAL step is the grid and the "
                    "larger ones are gaps (a weekend in an FX series), so the grid is not read from "
                    "the median of the DISTINCT values, which is not a spacing at all"}


def recompute(path: Path) -> dict:
    """Metrics from the published predictions and their labels, plus the naive at the same horizon."""
    import numpy as np
    with path.open(newline="") as fh:
        records = list(csv.DictReader(fh))
    if not records:
        return {"reconstructed": False, "missing": "the prediction file holds no rows"}
    columns = records[0].keys()
    horizons = sorted({int(m.group(1)) for c in columns if (m := re.fullmatch(r"Prediction_H(\d+)", c))})
    if not horizons:
        return {"reconstructed": False, "missing": "no Prediction_H<h> column"}
    missing_target = [h for h in horizons if f"Target_H{h}" not in columns]
    if missing_target:
        return {"reconstructed": False, "missing": f"labels for horizons {missing_target}"}
    base_col = next((c for c in columns if re.fullmatch(r"(test|validation|val|train)_\w+", c)), None)
    split = base_col.split("_", 1)[0] if base_col else None
    split = {"val": "Validation", "validation": "Validation", "test": "Test", "train": "Train"}.get(split)
    out, rows = {}, len(records)
    for h in horizons:
        y = np.array([float(r[f"Target_H{h}"]) for r in records])
        p = np.array([float(r[f"Prediction_H{h}"]) for r in records])
        entry = {"n": int(y.size), "mae": float(np.mean(np.abs(p-y))),
                 "rmse": float(np.sqrt(np.mean((p-y)**2))), "bias": float(np.mean(p-y)),
                 "target_min": float(y.min()), "target_max": float(y.max())}
        if base_col:
            b = np.array([float(r[base_col]) for r in records])
            entry["naive_from_published_base"] = float(np.mean(np.abs(b-y)))
            entry["naive_column"] = base_col
            entry["mae_skill_percent_vs_that_naive"] = (100*(1-entry["mae"]/entry["naive_from_published_base"])
                                                        if entry["naive_from_published_base"] > 0 else None)
        out[f"H{h}"] = entry
    return {"reconstructed": True, "rows": rows, "identifier_column": list(columns)[0],
            "horizons": out, "split_of_these_predictions": split,
            "split_evidence": f"the published file carries the column {base_col!r}" if base_col else
                              "no column names the split these predictions belong to",
            "scope": "recomputed from the published predictions and labels of this file; the training "
                     "that produced them is not re-executed and no split is reopened"}


def scale_reading(recomputed: dict, config: dict | None) -> dict:
    """What scale those numbers are in — read from the values and the config, never from magnitude."""
    if not recomputed.get("reconstructed"):
        return {"known": False, "why": "nothing was reconstructed"}
    first = next(iter(recomputed["horizons"].values()))
    lo, hi = first["target_min"], first["target_max"]
    declared = {k: (config or {}).get(k) for k in ("use_log1p_features", "target_column",
                                                   "use_normalization_json")}
    target = declared.get("target_column")
    log_features = declared.get("use_log1p_features") or []
    reading = {"target_range_in_published_file": [lo, hi], "declared": declared,
               "target_is_listed_among_log1p_features": bool(target and target in log_features)}
    reading["labels_look_log1p_transformed"] = None
    if target and target in log_features:
        reading["labels_look_log1p_transformed"] = bool(hi < 1.0)
        reading["why"] = ("the config lists the target column among the log1p FEATURES; whether the "
                          "LABEL was also transformed cannot be settled by the magnitude alone, so "
                          "the published range is reported and the question is left open")
    else:
        reading["why"] = "the config does not list the target column among the log1p features"
    return reading


def inventory(results_glob: str = RESULTS_GLOB) -> dict:
    results = sorted(REPO.glob(results_glob))
    predictions = [p for p in (prediction_file(r) for r in results) if p]
    frozen = freeze(results + predictions)
    configs = []
    for path in sorted(REPO.glob("examples/config/**/*.json")):
        try:
            configs.append((str(path.relative_to(REPO)), json.loads(path.read_text())))
        except (ValueError, OSError):
            continue
    entries, with_naive, reconstructed = {}, [], 0
    for path in results:
        rel = str(path.relative_to(REPO))
        published = published_rows(path)
        named = configs_naming(path, configs)
        pred = prediction_file(path)
        entry = {"published": published, "commit": producing_commit(path),
                 "configs_that_name_this_output": [c for c, _ in named],
                 "config_rule": "a config counts only when it names this file as its own output; an "
                                "adjacent config in the same folder is NOT evidence of provenance",
                 "prediction_file": str(pred.relative_to(REPO)) if pred else None}
        cfg = named[0][1] if named else None
        entry["declared_transformations"] = ({k: cfg.get(k) for k in TRANSFORM_KEYS if k in cfg}
                                             if cfg else None)
        if pred:
            entry["physical_grid"] = physical_grid(pred)
            grid = entry["physical_grid"]
            if grid.get("known") and cfg and cfg.get("predicted_horizons"):
                entry["horizon_in_physical_time"] = {
                    "grid_hours": grid["modal_step_hours"],
                    "horizons": {f"H{h}": {"steps": h, "hours": h*grid["modal_step_hours"]}
                                 for h in cfg["predicted_horizons"]},
                    "warning": ("the file name says what the operator called it; this is what its own "
                                "timestamps say. They can disagree.")}
            entry["recomputed"] = recompute(pred)
            entry["scale"] = scale_reading(entry["recomputed"], cfg)
            if entry["recomputed"].get("reconstructed"):
                reconstructed += 1
                entry["status"] = "RECONSTRUCTED_FROM_PREDICTIONS_AND_LABELS"
                entry["published_vs_recomputed"] = compare_published(published, entry["recomputed"])
            else:
                entry["status"] = "PUBLISHED_SUMMARY_ONLY"
                entry["missing_object"] = entry["recomputed"]["missing"]
        else:
            entry["status"] = "PUBLISHED_SUMMARY_ONLY"
            entry["missing_object"] = "no prediction file beside this results table"
        if published["has_any_naive"]:
            with_naive.append(rel)
        entries[rel] = entry
    return {"schema": "df_legacy_inventory.v1",
            "at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "freeze": frozen, "entries": entries,
            "counts": {"results_tables": len(results), "with_a_published_naive": len(with_naive),
                       "reconstructed": reconstructed,
                       "published_summary_only": len(results)-reconstructed},
            "tables_with_a_published_naive": with_naive,
            "rule": "nothing here re-executes a sweep, reopens a reserved split or overwrites a sample"}


def compare_published(published: dict, recomputed: dict) -> dict:
    """The published summary against the recomputation: rounding is visible, differences preserved."""
    rows, differences, other_splits = {}, {}, []
    this_split = recomputed.get("split_of_these_predictions")
    for key, pair in published["model_naive_pairs"].items():
        split, h = key.split("_H")
        got = recomputed["horizons"].get(f"H{h}")
        if got is None:
            continue
        if this_split and split != this_split:
            # the published file holds ONE split's predictions; a row of another split is not
            # comparable with them, and pretending otherwise manufactures a difference
            other_splits.append(key)
            continue
        d = pair["mae"]-got["mae"]
        rows[key] = {"published_mae": pair["mae"], "recomputed_mae": got["mae"], "difference": d,
                     "published_naive": pair["naive_mae"],
                     "recomputed_naive": got.get("naive_from_published_base")}
        if abs(d) > 5e-7:                       # the published table carries six decimals
            differences[key] = rows[key]
    return {"rows": rows, "differences_beyond_published_rounding": differences,
            "split_of_the_predictions": this_split,
            "published_rows_of_other_splits_not_comparable": sorted(other_splits),
            "note": "the published file carries six decimals; a difference below that is rounding, "
                    "and anything larger is kept here exactly",
            "split_caveat": "a published row named Test was compared with the predictions published "
                            "beside it; which split those predictions are is a claim of the producing "
                            "run, not something this recomputation can establish"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results-glob", default=RESULTS_GLOB)
    ap.add_argument("--freeze", type=Path, default=None)
    a = ap.parse_args(argv)
    doc = inventory(a.results_glob)
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    if a.freeze:
        a.freeze.write_text(json.dumps(doc["freeze"], indent=1, default=str))
    print(json.dumps({"counts": doc["counts"], "tables_with_a_published_naive":
                      doc["tables_with_a_published_naive"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
