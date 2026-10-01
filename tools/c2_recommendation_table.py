"""Lane C2 deliverable 3: the recommendation table to M03 (selection) and M07 (campaign).

One row per feature x horizon (blocks5 protocol), ranked by the stable incremental utility of deliverable 1, with
every number carrying its population, split, rows and horizon; the reference rows (zero-return, last-return, 24 h
seasonal, train-mean naives; the full ridge; lane B's fixed-alpha ridge; the gradient-boosted full model) on the same
rows; the leak-probe verdict; and the dossier's residual-variance share and control verdicts where a dossier exists.
A second block reports lane B's comparability protocol (3 expanding inner folds, Y_s@4h / Y_l@24h / Y_l@144h).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(summary, leak, dossier_index=None, lane_b_reference=None):
    bind = summary["bindings"]
    pop = f"{bind['view']['dataset_id']} sha {bind['view']['sha256'][:12]}"
    split = f"TRAIN rows [{bind['split']['train_rows'][0]},{bind['split']['train_rows'][1]}) (M07 split {bind['split']['file_sha256'][:8]})"
    rows = []
    dindex = {}
    if dossier_index:
        for d in dossier_index["dossiers"]:
            dindex[(d["feature"], d["horizon_bars"])] = d
    ranking = summary["ranking_by_stable_incremental_utility"]
    for rank, f in enumerate(ranking, 1):
        for key, e in summary["features"][f].items():
            proto, hk = key.split("|")
            h = int(hk[1:])
            ref = summary["reference"][key]
            d = dindex.get((f, h), {})
            rows.append({
                "rank_stable": rank, "feature": f, "protocol": proto, "horizon_bars": h, "horizon_hours": 4 * h,
                "population": pop, "split": split, "n_eval_rows_total": ref["naive_zero"]["n_eval_total"],
                "delta_mae_z_median_over_blocks": e["delta_mae_z_median"], "delta_mae_z_min": e["delta_mae_z_min"],
                "delta_mae_z_max": e["delta_mae_z_max"], "sign_agreement": e["sign_agreement"], "rank_median_in_horizon": e["rank_median"],
                "delta_over_naive_zero_median": float(np.median(list(e["delta_mae_over_naive_zero_by_block"].values()))) if e["delta_mae_over_naive_zero_by_block"] else np.nan,
                "only_feature_mae_z_mean": e["only_mae_z_mean"], "only_beats_naive_zero_blocks": e["only_beats_naive_zero_blocks"],
                "hgb_perm_delta_mae_z_median": e.get("hgb_perm_delta_mae_z_median", np.nan),
                "full_ridge_mae_z_mean": ref["full|selected"]["mae_z_mean"], "full_ridge_alpha1_mae_z_mean": ref.get("full|laneB_alpha1", {}).get("mae_z_mean", np.nan),
                "hgb_full_mae_z_mean": ref.get("hgb_full", {}).get("mae_z_mean", np.nan),
                "naive_zero_mae_z_mean": ref["naive_zero"]["mae_z_mean"], "naive_last_return_mae_z_mean": ref["naive_last_return"]["mae_z_mean"],
                "naive_seasonal_24h_mae_z_mean": ref["naive_seasonal_24h"]["mae_z_mean"], "naive_train_mean_mae_z_mean": ref["naive_train_mean"]["mae_z_mean"],
                "naive_zero_mae_log_return_mean": ref["naive_zero"]["mae_log_return_mean"], "full_ridge_mae_log_return_mean": ref["full|selected"]["mae_log_return_mean"],
                "leak_verdict": leak["verdict"].get(f), "rho_next_bar_return": leak["statistical_flags"][f]["rho_next_bar_return"],
                "dossier_theta_z": d.get("theta_z", np.nan), "dossier_residual_variance_share": d.get("residual_variance_share", np.nan),
                "dossier_support_state": d.get("support_state"), "dossier_controls_failed_as_required": d.get("controls_failed_as_required"),
                "stable_incremental_utility_score": summary["scores"][f]["stable_incremental_utility"],
                "label": "DEVELOPMENT",
            })
    table = pd.DataFrame(rows)
    return table


def markdown(table, summary, leak, lane_b_reference=None, ps4_reference=None):
    bind = summary["bindings"]
    lines = ["# Lane C2: feature recommendation table for M03 (selection) and M07 (campaign) -- DEVELOPMENT", ""]
    lines += [f"Population: `{bind['view']['dataset_id']}`, view sha `{bind['view']['sha256']}` (predictor `{bind['view']['commit'][:8]}`), "
              f"features: variant A, declaration `{bind['manifest']['declaration_sha256'][:8]}`, manifest file `{bind['manifest']['file_sha256'][:8]}`; "
              f"split: {bind['split']['authority']}, file `{bind['split']['file_sha256'][:8]}`, TRAIN rows [{bind['split']['train_rows'][0]},{bind['split']['train_rows'][1]}), "
              f"scored origins [{bind['split']['train_origins'][0]},{bind['split']['train_origins'][1]}] ({bind['split']['train_windows']} windows, {bind['split']['gap_excluded_windows']} gap-excluded), "
              f"train row-ids sha `{bind['split']['train_row_ids_sha256'][:8]}`. Target: {bind['target']['definition']} (mu={bind['target']['mu']:.6g}, sigma={bind['target']['sigma']:.6g}).", ""]
    lines += ["Units: MAE_z is in M07 z-units of the cumulative standardized 1-bar log return; MAE_log_return is the raw log return of CLOSE. "
              "Incremental utility delta = MAE(without feature) - MAE(all 83), positive = the feature helps the held-out ridge. Every row is TRAIN-only, held out within TRAIN.", ""]
    lines += ["## Reference rows on the identical held-out rows (blocks5 protocol, mean over the 5 blocks)", "",
              "| horizon (bars / h) | n eval | naive zero | naive last-return | naive seasonal 24h | naive train-mean | ridge all-83 (alpha selected) | ridge all-83 (alpha=1, lane B) | HGB all-83 |", "|---|---|---|---|---|---|---|---|---|"]
    for key, ref in sorted(summary["reference"].items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1][1:]))):
        proto, hk = key.split("|")
        if proto != "blocks5":
            continue
        h = int(hk[1:])
        g = lambda a: f"{ref[a]['mae_z_mean']:.5f}" if a in ref else "n/a"
        lines.append(f"| {h} / {4 * h} | {ref['naive_zero']['n_eval_total']} | {g('naive_zero')} | {g('naive_last_return')} | {g('naive_seasonal_24h')} | {g('naive_train_mean')} | {g('full|selected')} | {g('full|laneB_alpha1')} | {g('hgb_full')} |")
    lines += ["", "## Lane B comparability protocol (3 expanding inner folds, purge 60; MAE_z mean over folds)", "",
              "| target | horizon bars | n eval | naive zero (ours) | ridge all-83 alpha=1 (ours) | ridge all-83 alpha selected (ours) | lane B C_ALL alpha=1 (its rows, log-return units) | lane B naive zero (its rows) |", "|---|---|---|---|---|---|---|---|"]
    lb = lane_b_reference or {}
    for key, ref in summary["reference"].items():
        proto, hk = key.split("|")
        if proto != "laneB":
            continue
        h = int(hk[1:])
        tname = {1: "Y_s@4h", 6: "Y_l@24h", 36: "Y_l@144h"}.get(h, f"h{h}")
        lines.append(f"| {tname} | {h} | {ref['naive_zero']['n_eval_total']} | {ref['naive_zero']['mae_log_return_mean']:.5f} (log-ret) / {ref['naive_zero']['mae_z_mean']:.5f} (z) | "
                     f"{ref.get('full|laneB_alpha1', {}).get('mae_log_return_mean', float('nan')):.5f} (log-ret) | {ref['full|selected']['mae_log_return_mean']:.5f} (log-ret) | "
                     f"{lb.get(tname, {}).get('C_ALL', float('nan')):.5f} | {lb.get(tname, {}).get('naive', float('nan')):.5f} |")
    if ps4_reference:
        lines += ["", "## Lane B PS4 transform-family probe (feature-eng 2840e52), as reported by lane B", "", "```", json.dumps(ps4_reference, indent=1)[:3000], "```"]
    lines += ["", f"## Rank agreement across the 5 blocks (mean pairwise Spearman of per-block incremental-utility ranks)", ""]
    for key, ra in sorted(summary["rank_agreement"].items()):
        lines.append(f"- {key}: {ra['mean_pairwise_spearman']:.3f} ({ra['n_pairs']} pairs)")
    pos = table[(table.protocol == "blocks5")]
    any_pos = pos[(pos.delta_mae_z_median_over_blocks > 0) & (pos.sign_agreement >= 0.8)]
    lines += ["", "## Features with a positive median incremental utility and sign agreement >= 4/5 (blocks5)", ""]
    if len(any_pos):
        lines += ["| feature | h | delta MAE_z median | min | max | sign agreement | only-feature beats naive zero (blocks of 5) | leak verdict |", "|---|---|---|---|---|---|---|---|"]
        for _, r in any_pos.sort_values(["horizon_bars", "delta_mae_z_median_over_blocks"], ascending=[True, False]).iterrows():
            lines.append(f"| {r.feature} | {r.horizon_bars} | {r.delta_mae_z_median_over_blocks:.6f} | {r.delta_mae_z_min:.6f} | {r.delta_mae_z_max:.6f} | {r.sign_agreement:.1f} | {r.only_beats_naive_zero_blocks} | {r.leak_verdict} |")
    else:
        lines.append("NONE: no feature has a positive median incremental utility with sign agreement >= 4/5 at any horizon.")
    beats = pos[pos.only_beats_naive_zero_blocks >= 3]
    lines += ["", f"## Does ANY single feature beat the zero-return naive on its own (one-feature ridge, >= 3 of 5 blocks)? {'YES: ' + ', '.join(sorted(set(beats.feature))) if len(beats) else 'NO'}", ""]
    lines += ["## Top 20 by stable incremental utility (mean over horizons 1..6 of the median relative delta)", "",
              "| rank | feature | score (delta / naive-zero MAE, mean over h) | mean sign agreement | leak verdict | rho next-bar |", "|---|---|---|---|---|---|"]
    for f in summary["ranking_by_stable_incremental_utility"][:20]:
        sc = summary["scores"][f]
        lines.append(f"| {summary['ranking_by_stable_incremental_utility'].index(f) + 1} | {f} | {sc['stable_incremental_utility']:.5f} | {sc['mean_sign_agreement']:.2f} | {leak['verdict'][f]} | {leak['statistical_flags'][f]['rho_next_bar_return']:.3f} |")
    lines += ["", "## Leak probe summary", "", f"- counts: {leak['counts']}; suspect next-bar correlation: {leak['suspect_count']}",
              f"- producer replayed: {leak['producer']['source']} (sha {leak['producer']['sha256'][:8]}); probe steps {leak['probe_steps']}; burn-in {leak['burn_in_rows']} rows", ""]
    mism = [f for f, v in leak["verdict"].items() if not v.startswith("CAUSAL_BY_RECOMPUTATION")]
    if mism:
        lines.append("- not verified by recomputation: " + ", ".join(f"{f} ({leak['identity'][f].get('state')}, max|diff|/std={leak['identity'][f].get('max_abs_diff_over_std', float('nan')):.3g})" for f in mism))
    lines += ["", "## Recommendation", "",
              "- M03 (selection): no feature in variant A is recommended for inclusion on predictive grounds alone; the table ranks by stable incremental utility so a bounded screen can take the top rows as candidates, each with its leak verdict and its dossier support state.",
              "- M07 (campaign): the paired naive gate (zero-return) is the reference every candidate must beat on the same rows; the reference rows above are the numbers to pair against.",
              "- All numbers are DEVELOPMENT (git-pinned view, undeclared timestamp semantics); nothing here is confirmatory."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--leak", required=True)
    p.add_argument("--dossier-index", default=None)
    p.add_argument("--lane-b-reference", default=None, help="json {target: {C_ALL:, naive:}}")
    p.add_argument("--ps4-reference", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    leak = json.loads(Path(args.leak).read_text(encoding="utf-8"))
    dindex = json.loads(Path(args.dossier_index).read_text(encoding="utf-8")) if args.dossier_index else None
    lb = json.loads(Path(args.lane_b_reference).read_text(encoding="utf-8")) if args.lane_b_reference else None
    ps4 = json.loads(Path(args.ps4_reference).read_text(encoding="utf-8")) if args.ps4_reference else None
    table = build(summary, leak, dindex, lb)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "RECOMMENDATION_TABLE.csv", index=False)
    (out / "RECOMMENDATION_TABLE.md").write_text(markdown(table, summary, leak, lb, ps4), encoding="utf-8")
    print(f"rows={len(table)} -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
