#!/usr/bin/env python3
"""Calibration verdict table from generative_evidence*.csv (stdlib only; no model code, no data)."""
import collections, csv, json, os, sys

D = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p, newline=""))) if os.path.isfile(p) else []


def main():
    feats = load("generative_evidence.csv")
    folds = load("generative_evidence_folds.csv")
    man = json.load(open(os.path.join(D, "run_manifest.json"))) if os.path.isfile(os.path.join(D, "run_manifest.json")) else {}
    out = {"schema": "fs_gen.summary.v1", "denominator_rows": len(feats), "fold_rows": len(folds)}
    out["calibration_state_counts"] = dict(collections.Counter(r["calibration_state"] for r in feats))
    out["knockoff_state_counts"] = dict(collections.Counter(r["knockoff_state"] for r in feats))
    out["by_batch"] = {b: dict(collections.Counter(r["calibration_state"] for r in feats if r["batch"] == b))
                       for b in sorted({r["batch"] for r in feats if r["batch"]})}
    gates = collections.Counter(g for r in folds for g in r["failing_gates"].split(";") if g)
    out["failing_gate_frequency_fold_level"] = dict(gates.most_common())
    out["fold_generator_states"] = dict(collections.Counter(r["generator_state"] for r in folds))
    out["fold_knockoff_states"] = dict(collections.Counter(r["knockoff_state"] for r in folds))
    out["knockoff_reasons_fold_level"] = dict(collections.Counter(r["knockoff_reason"].split(":")[0] for r in folds if r["knockoff_reason"]).most_common())
    run = [r for r in feats if r["knockoff_state"] == "RUN"]
    out["knockoff_run_features"] = [{"feature_id": r["feature_id"], "selected_cells_majority": r["knockoff_selected_cells_majority"],
                                     "selected_cells_any": r["knockoff_selected_cells_any"], "jaccard": r["selection_jaccard_mean"]} for r in run]
    out["features_with_majority_selection"] = sum(1 for r in run if r["knockoff_selected_cells_majority"])
    def med(vals):
        v = sorted(float(x) for x in vals if x not in ("", "nan"))
        return round(v[len(v) // 2], 4) if v else None
    out["fold_metric_medians"] = {k: med(r[k] for r in folds) for k in
                                  ("acf_max_abs_diff_lags_1_48", "psd_log_ratio_rmse", "ks_marginal", "tail_ratio_q99_abs",
                                   "kurtosis_ratio", "regime_coverage", "second_moment_gap", "swap_classifier_auc_batch",
                                   "conditional_independence_abs_z_max")}
    out["cost"] = {"cost_s": man.get("cost_s"), "peak_rss_mb": man.get("peak_rss_mb"), "code_sha256": man.get("code_sha256"),
                   "outputs": man.get("outputs"), "started_utc": man.get("started_utc"), "finished_utc": man.get("finished_utc")}
    json.dump(out, open(os.path.join(D, "SUMMARY.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("calibration_state_counts", "knockoff_state_counts", "failing_gate_frequency_fold_level",
                                           "features_with_majority_selection", "fold_metric_medians")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
