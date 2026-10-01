"""Lane H: build the three-arm result table (CSV rows and markdown) from RESULTS.json evidence. DEVELOPMENT."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pipe = _load("h_kalman_pipeline")


def reading(delta, ci, quarters, naive_mae=None, floor=1e-3):
    """EXCEEDS_SPREAD only when the block-bootstrap interval excludes zero AND |delta| is larger than the range of
    the four contiguous-quarter deltas; a gap below 0.1 percent of the zero-return MAE is reported as negligible whatever its
    interval; otherwise the gap is within the spread and is not called a difference."""
    if naive_mae and abs(delta) / naive_mae < floor:
        return "NEGLIGIBLE_BELOW_0.1_PERCENT_OF_NAIVE_MAE"
    lo, hi = ci
    spread = max(quarters) - min(quarters)
    if (lo > 0 or hi < 0) and abs(delta) > spread:
        return "EXCEEDS_SPREAD"
    return "WITHIN_SPREAD"


def table_rows(results):
    rows = []
    for vname, v in results["variants"].items():
        for lg, blk in v["arms"].items():
            for arm, ev in blk["evaluations"].items():
                paired = {p["horizon"]: p for p in blk["paired_vs_A"].get(arm, [])}
                for r in ev["per_horizon"]:
                    n = r["naive"]
                    seas = [k for k in n if k.startswith("seasonal_")][0]
                    p = paired.get(r["horizon"])
                    row = {"variant": vname, "lags": lg, "arm": arm, "label": ev.get("label"), "eligible": ev["eligible"],
                           "horizon": r["horizon"], "rows": r["rows"], "model_MAE": r["model_MAE"], "model_MSE": r["model_MSE"],
                           "zero_return_MAE": n["zero_return"]["MAE"], "zero_return_MSE": n["zero_return"]["MSE"],
                           "persistence_MAE": n["persistence_last_value"]["MAE"], "persistence_MSE": n["persistence_last_value"]["MSE"],
                           "seasonal_MAE": n[seas]["MAE"], "seasonal_MSE": n[seas]["MSE"], "train_mean_MAE": n["train_mean"]["MAE"],
                           "strict_naive": r["strict_naive"], "skill_vs_zero_return_MAE": n["zero_return"]["skill_MAE"],
                           "skill_vs_zero_return_MSE": n["zero_return"]["skill_MSE"], "alpha": ev.get("alpha"),
                           "channels": ev.get("channels")}
                    if p is not None:
                        row.update({"delta_MAE_vs_A": p["delta_MAE_mean"], "ci95_lo": p["ci95"][0], "ci95_hi": p["ci95"][1],
                                    "quarter_range": max(p["quarter_deltas"]) - min(p["quarter_deltas"])})
                        row["reading"] = reading(p["delta_MAE_mean"], p["ci95"], p["quarter_deltas"], naive_mae=n["zero_return"]["MAE"])
                    else:
                        row["reading"] = "BASE" if arm == "A" else "IDENTITY_CONTROL" if arm == "IDENTITY" else "UNPAIRED"
                    if not ev["eligible"]:
                        row["reading"] = "NON_CAUSAL_REJECTION_CONTROL_NEVER_ELIGIBLE"
                    rows.append(row)
    return rows


def to_markdown(rows, title):
    cols = ["variant", "lags", "arm", "horizon", "model_MAE", "zero_return_MAE", "persistence_MAE", "seasonal_MAE", "train_mean_MAE",
            "skill_vs_zero_return_MAE", "delta_MAE_vs_A", "ci95_lo", "ci95_hi", "quarter_range", "reading"]
    def f(v):
        return f"{v:.5f}" if isinstance(v, float) else ("" if v is None else str(v))
    out = [f"# {title}", "", "DEVELOPMENT. MAE in the train-standardized target space, validation rows, each naive on the same rows. "
           "delta_MAE_vs_A is arm minus A (negative = lower error), 95% moving-block bootstrap interval, quarter_range = range of the "
           "four contiguous-quarter deltas.", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--title", default="Lane H three-arm table")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    rows = table_rows(res)
    Path(a.out_prefix + ".md").write_text(to_markdown(rows, a.title))
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(a.out_prefix + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
