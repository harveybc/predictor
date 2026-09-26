import json, statistics, sys
from pathlib import Path
O = Path(sys.argv[1])
R = json.load(open(O/"RESOLUTION.json"))
M = json.load(open(O/"MATCHED_CONTRAST.json"))
C = json.load(open(O/"CLOSURE_TABLE.json"))

def arm_rows(run, arms):
    """Arm-level closure rows: the arm's mean model error, the naive on the SAME rows, its skill,
    the literature value and the comparability decision -- every field from the closure table's own
    per-cell rows, aggregated only by averaging the model error over the arm's seeds."""
    out = []
    for arm in arms:
        rs = [r for r in C["rows"] if r["run"] == run and r["arm"] == arm and r["model_error"] is not None]
        if not rs: continue
        me = statistics.fmean(r["model_error"] for r in rs)
        ne = rs[0]["naive_error"]
        assert len({round(r["naive_error"],12) for r in rs}) == 1
        assert len({r["n_evaluated"] for r in rs}) == 1
        lit = rs[0]["literature_value_and_source"]
        out.append({"run": run, "arm": arm, "n_seeds": len(rs), "model_kW": me,
                    "model_z": me/(rs[0]["model_error"]/rs[0]["model_error_z"]),
                    "naive_kW": ne, "n": rs[0]["n_evaluated"], "h": rs[0]["model_horizon"],
                    "skill": 1-me/ne, "lit": lit, "comp": rs[0]["comparability_status"],
                    "custody": rs[0]["custody"]["class"], "verified": rs[0]["verified"]})
    return out

LIT = None
print("### Closure table — every arm mean in this document\n")
print("| run | arm | seeds | model error (kW) | model error (z) | naive, same rows (kW), n | h | skill vs naive | literature value & source | comparability | custody | verified |")
print("|---|---|--:|--:|--:|--:|--:|--:|---|---|---|---|")
for run, arms in (("e1_phase1_v1b", ("core_mae","core_mse","tcn_mse")),
                  ("e1_household_successor_v3", ("R0","R1","R2")),
                  ("e1_phase1_matched_v1", ("core_mae","core_mse","tcn_mse"))):
    for r in arm_rows(run, arms):
        LIT = r["lit"]
        print(f"| `{r['run']}` | `{r['arm']}` | {r['n_seeds']} | {r['model_kW']:.9f} | {r['model_z']:.6f} | "
              f"{r['naive_kW']:.9f}, n={r['n']} | {r['h']} | {r['skill']:+.6f} | "
              f"{r['comp']} — see note | {r['comp']} | {r['custody']} | {'yes' if r['verified'] else 'NO'} |")
print()
print("**Literature note, identical for every row above** (the registry decides it from identity fields, "
      "never from a score): source — " + LIT["source"] + ". Published values — " + LIT["published_value"] +
      ". Status **" + LIT["status"] + "**, comparator state `" + str(LIT["comparator_state"]) + "`; "
      "`placed_in_comparison_column: " + str(LIT["placed_in_comparison_column"]).lower() + "`. Reason: " +
      LIT["why_not"] + ". Planned matched comparison: " + LIT["planned_matched_comparison"] + ".")
print()
print("### The retained contrasts against the measured resolution\n")
print("| contrast | effect (kW) | 95% CI (kW) | p | state against the resolution | seeds for this effect |")
print("|---|--:|---|--:|---|--:|")
by = {c["name"]: c for c in R["contrasts"]}
for v in R["effect_verdicts"]:
    c = by[v["contrast"]]
    n = v["seeds_required_for_this_effect"].get("n_per_arm")
    print(f"| `{v['contrast']}` | {v['effect_kW']:+.6f} | [{c['ci95_kW'][0]:+.6f}, {c['ci95_kW'][1]:+.6f}] | "
          f"{c['p_value']:.4f} | {v['state_against_the_resolution']} | {n if n else 'n/a'} |")
print()
print("### The matched re-contrast against the unmatched one\n")
print("| contrast | unmatched effect (kW) | matched effect (kW) | change (kW) | matched 95% CI | matched p | state | erased? |")
print("|---|--:|--:|--:|---|--:|---|---|")
for a in M["against_the_resolution"]:
    u = "—" if a["unmatched_effect_kW"] is None else f"{a['unmatched_effect_kW']:+.6f}"
    ch = "—" if a["change_kW"] is None else f"{a['change_kW']:+.6f}"
    er = "n/a" if a["erased"] is None else ("YES" if a["erased"] else "**NO**")
    print(f"| `{a['contrast']}` | {u} | {a['matched_effect_kW']:+.6f} | {ch} | "
          f"[{a['matched_ci95_kW'][0]:+.6f}, {a['matched_ci95_kW'][1]:+.6f}] | {a['matched_p_value']:.4f} | "
          f"{a['state_against_the_resolution']} | {er} |")
print()
print("### Per-arm means and seed spread, unmatched against matched\n")
print("| arm | mean unmatched (kW) | mean matched (kW) | change (kW) | sd unmatched (kW) | sd matched (kW) | updates unmatched | updates matched |")
print("|---|--:|--:|--:|--:|--:|--:|--:|")
bud = R["budget"]["per_arm"]
for a in ("core_mae","core_mse","tcn_mse"):
    print(f"| `{a}` | {R['arm_means_kW'][a]:.9f} | {M['arm_means_kW'][a]:.9f} | "
          f"{M['arm_means_kW'][a]-R['arm_means_kW'][a]:+.9f} | {R['arm_sds_kW'][a]:.9f} | {M['arm_sds_kW'][a]:.9f} | "
          f"{bud[a]['total_updates']} | {M['budget_audit']['total_updates_by_arm'][a]} |")
print()
L = M["like_for_like_against_the_unmatched_run"]
print("### Like for like: did the repair sharpen the instrument?\n")
print(f"Only `{'`, `'.join(L['arms_present_in_both_runs'])}` were run under BOTH protocols, so only those two answer it.\n")
print("| | sigma (kW) | df | resolution at n=3 (kW) | seeds for a flat 0.01 kW |")
print("|---|--:|--:|--:|--:|")
print(f"| unmatched (RP63) | {L['sigma_unmatched_kW']:.9f} | {L['sigma_df']} | {L['resolution_unmatched_kW']:.6f} | {L['seeds_for_a_flat_0p01_kW_unmatched']['n_per_arm']} |")
print(f"| matched (this round) | {L['sigma_matched_kW']:.9f} | {L['sigma_df']} | {L['resolution_matched_kW']:.6f} | {L['seeds_for_a_flat_0p01_kW_matched']['n_per_arm']} |")
print(f"| ratio | {L['sigma_ratio_matched_over_unmatched']:.6f} | | | |")
