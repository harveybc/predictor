"""Publish, from the artifacts only, what this execution cycle produced:

 - the 21 eligible slots and what each one did (nothing, and why), with the
   calibration dispersion that made it eligible;
 - the 16-slot Holm family and each slot's frozen disposition, with the
   surviving set stated as UNDETERMINED because no p-value exists;
 - the unit ledger: 3024 units, how many were attempted, how many failed and
   the typed reason;
 - the closure table this corpus requires, generated from artifact bytes.

Nothing is typed that can be read. Every number comes from the sealed bytes at
agent-multi@0e99ad1a; the script prints the digest of each file it read.
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

AM = Path("/home/harveybc/Documents/GitHub/agent-multi")
TIP = "0e99ad1add3635dd4f0e151936d6fe4bf6379592"
EVID = Path(__file__).resolve().parent
ADJ = ("docs/audits/evidence/M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_"
       "GOVERNING_2026_09_09.json")
SUCC = "docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json"
CENSUS_SHA = ("12cfd9ad785b41e788ffce575ec575ab"
              "2a78ab772151b5e3b94c8f0c71169ea0")
RUN_ROOT_EXPECTED = "no CONFIRMATION run root was ever created"


def show(rel):
    r = subprocess.run(["git", "show", f"{TIP}:{rel}"], cwd=AM,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read {rel}")
    raw = r.stdout.encode()
    return json.loads(r.stdout), hashlib.sha256(raw).hexdigest()


def main() -> int:
    adj, adj_sha = show(ADJ)
    succ, succ_sha = show(SUCC)
    print(f"read {ADJ}\n  file sha256 = {adj_sha}")
    print(f"read {SUCC}\n  file sha256 = {succ_sha}\n")

    per_slot = succ["confirmation_generators_per_eligible_slot"]
    seeds = succ["nested_seeds_per_generator"]
    units_per_slot = per_slot * seeds

    # ---------- 1. the 21 slots ----------
    slots = []
    for s in sorted(succ["eligible_slots"], key=lambda x: x["cell"]):
        d = s["calibration_dispersion"]
        prefix, w = s["cell"].rsplit("::w", 1)
        slots.append({
            "cell": s["cell"],
            "contrast_fed": f"intervention_effect::{prefix}",
            "width": int(w),
            "confirmation_generators_planned": per_slot,
            "seeds_per_generator": seeds,
            "units_planned": units_per_slot,
            "units_attempted": 0,
            "units_complete": 0,
            "units_failed": 0,
            "what_it_did": "NOTHING — no generator constructed, no unit fitted",
            "why": "the frozen two-record gate never opened, and the frozen "
                   "tools contain no CONFIRMATION unit-execution body",
            "calibration_complete_generators": d["complete_generators"],
            "calibration_sd": d["sd"],
            "calibration_sd_ucb95": d["sd_ucb95"],
            "calibration_precision_supported": d["supported"],
            "confirmation_attrition_floor":
                succ["attrition"]["min_complete_required"],
            "attrition_status": "NOT_APPLICABLE_NO_UNITS",
        })
    assert len(slots) == 21, len(slots)
    inelig = sorted(({"cell": s["cell"], "typed_status": s["typed_status"],
                      "units_planned": 0,
                      "what_it_did": "NOTHING — never constructed by design"}
                     for s in succ["ineligible_slots"]),
                    key=lambda x: x["cell"])

    # ---------- 2. the 16-slot Holm family ----------
    elig_prefixes = {}
    for s in succ["eligible_slots"]:
        prefix, w = s["cell"].rsplit("::w", 1)
        elig_prefixes.setdefault(prefix, []).append(int(w))
    family = []
    for key in succ["contrast_family_16"]:
        row = {"slot": key}
        if key.startswith("intervention_effect::"):
            prefix = key.split("intervention_effect::", 1)[1]
            widths = sorted(elig_prefixes.get(prefix, []))
            row["eligible_widths"] = widths
            if not widths:
                row["frozen_disposition"] = "NOT_EVALUABLE_p1"
                row["frozen_reason"] = "no width is frozen eligible"
            elif len(widths) == 1:
                row["frozen_disposition"] = "SINGLE_WIDTH_USED_AND_NAMED"
                row["frozen_reason"] = f"exactly one width (w{widths[0]})"
            else:
                row["frozen_disposition"] = "TWO_WIDTHS_EQUAL_AVERAGE"
                row["frozen_reason"] = "both frozen widths eligible"
        elif key == "checkpoint_effect::primary_pair":
            row["eligible_widths"] = []
            row["frozen_disposition"] = "CHECKPOINT_PAIR"
            row["frozen_reason"] = "the 15th slot, paired within generator"
        else:
            row["eligible_widths"] = []
            row["frozen_disposition"] = "NON_REJECTING_PLACEHOLDER_p1"
            row["frozen_reason"] = ("M2 failed CALIBRATION with gain "
                                   f"{succ['m2_status']['calibration_gain']}; "
                                   "kept so the family does not shrink")
        row["p_raw"] = None
        row["p_holm"] = None
        row["p_bonferroni"] = None
        row["survived_holm"] = "UNDETERMINED_NO_OBSERVATION"
        family.append(row)
    assert len(family) == 16, len(family)
    n_two = sum(1 for r in family if r["frozen_disposition"]
                == "TWO_WIDTHS_EQUAL_AVERAGE")
    n_one = sum(1 for r in family if r["frozen_disposition"]
                == "SINGLE_WIDTH_USED_AND_NAMED")
    n_ne = sum(1 for r in family if r["frozen_disposition"]
               == "NOT_EVALUABLE_p1")
    assert n_two * 2 + n_one == 21, (n_two, n_one)

    # ---------- 3. the unit ledger ----------
    units = {
        "units_planned": 21 * per_slot * seeds,
        "census_sha256": CENSUS_SHA,
        "pre_result_ledger": RUN_ROOT_EXPECTED,
        "units_attempted": 0,
        "units_complete": 0,
        "units_failed": 0,
        "units_not_attempted": 21 * per_slot * seeds,
        "failure_modes_observed": {},
        "typed_reason_no_unit_was_attempted": [
            "GATE_1_DESIGN_REVIEW_RECORD_ABSENT — the frozen protocol admits "
            "only a record whose role token is EXTERNAL_AUDITOR and whose "
            "decision token approves execution. Authoring it would have been "
            "the executing agent approving its own execution; the host "
            "permission system refused that on 2026-09-26 and the refusal was "
            "not worked around.",
            "GATE_2_OWNER_EXECUTION_RECORD_ABSENT — it chains by digest to the "
            "record above, so it cannot exist first.",
            "BLOCKER_3_NO_EXECUTION_BODY — independently of the gates, the "
            "frozen tools cannot fit a CONFIRMATION unit: the sealed generator "
            "bank refuses CONFIRMATION construction unless allow_confirmation "
            "is passed, the sealed unit runner never passes it, and "
            "execute_confirmation returns right after the pre-result ledger. "
            "Both records installed would yield a 3024-unit PENDING ledger and "
            "zero fitted units.",
        ],
    }

    # ---------- 4. the closure table ----------
    ib = adj["ladder"]["integrated_brier"]
    n_groups = adj["ladder"]["n_groups"]
    def skill(model, naive):
        return round(1.0 - model / naive, 6)
    closure = {
        "generated_from": {
            "governing_adjudication_file_sha256": adj_sha,
            "successor_file_sha256": succ_sha,
            "confirmation_run_artifacts": "NONE — no run root exists",
        },
        "rows": [
            {
                "quantity": "M4 CONFIRMATION screen — primary estimand "
                            "(generator-level paired restricted-endpoint "
                            "difference, calibration_stop minus matched "
                            "initialization)",
                "stage": "CONFIRMATION",
                "model_error_with_scale": "NO_NEW_MEASUREMENT",
                "scale": "restricted acquired associations, integer 0..512 "
                         "over 64 batches of 8",
                "naive_reference_same_rows": "NO_NEW_MEASUREMENT — the naive "
                    "reference IS the matched initialization arm of the same "
                    "generator, tape and compute; neither arm was fitted",
                "skill": "NO_NEW_MEASUREMENT",
                "literature_value": "NOT_CARRIED",
                "literature_source": "NOT_CARRIED — no external benchmark "
                    "exists for a restricted association count on this "
                    "synthetic generator bank",
                "comparability": "NOT_COMPARABLE",
                "comparability_reason": "there is no measurement to compare: "
                    "zero of 3024 units were fitted. This row is "
                    "NO_NEW_MEASUREMENT by the standing closure rule, not an "
                    "unfavourable result.",
            },
        ],
    }
    for model, naive in (("M1", "M0"), ("M2", "M1"), ("M2", "M0")):
        closure["rows"].append({
            "quantity": f"M4 prediction ladder {model} vs {naive} — "
                        "out-of-generator prediction of log1p(restricted "
                        "endpoint), integrated Brier",
            "stage": "CALIBRATION (NOT THIS SCREEN — carried so the scale and "
                     "the M2 exclusion are not asserted without numbers)",
            "model_error_with_scale": f"{ib[model]} integrated Brier "
                                      f"(unitless, 0 is perfect), n_groups "
                                      f"{n_groups}",
            "scale": "integrated Brier over the frozen ladder horizons",
            "naive_reference_same_rows": f"{naive} = {ib[naive]} integrated "
                f"Brier on the SAME {n_groups} unseen-generator groups "
                f"({'parameter count alone' if naive == 'M0' else 'M0 plus checkpoint loss and elapsed updates'})",
            "skill": skill(ib[model], ib[naive]),
            "skill_reading": ("NEGATIVE — the richer model is worse than its "
                              "own naive reference on the same rows"
                              if ib[model] > ib[naive] else "POSITIVE"),
            "literature_value": "NOT_CARRIED",
            "literature_source": "NOT_CARRIED — the endpoint is an internal "
                "restricted association count over a synthetic association "
                "tape; no published value measures the same quantity",
            "comparability": "NOT_COMPARABLE",
            "comparability_reason": "synthetic generator bank, internal "
                "endpoint, no external benchmark; and it is a CALIBRATION "
                "measurement, so it can never stand in for the confirmation "
                "row above",
        })
    closure["note_on_m2"] = (
        f"M2 minus M1 paired gain {adj['ladder']['m2_minus_m1_paired_gain']} "
        f"(t {adj['ladder']['m2_vs_m1_t']}, n_groups {n_groups}) is why M2 is "
        "carried as a non-rejecting placeholder rather than fitted. The table "
        "also shows M1 itself has NEGATIVE skill against M0: on these rows the "
        "parameter count alone predicts the endpoint better than the "
        "measurement-enriched models.")

    out = {
        "schema": "satoshi_m4_confirmation_execution_state.v1",
        "date": "2026-09-26",
        "author": "Satoshi III (Mujuro Utsutsu), successor technical lead",
        "verdict": "NO_NEW_MEASUREMENT",
        "verdict_statement":
            "The M4 CONFIRMATION screen did NOT execute. Zero of 3024 units "
            "were fitted, no CONFIRMATION generator was constructed, no run "
            "root or ledger exists, and no contrast has a p-value. This is not "
            "DOES_NOT_ADVANCE and not ADVANCES: the frozen family is "
            "UNDETERMINED because it has no observation. Two gates and one "
            "code gap are named in the unit ledger.",
        "eligible_slots_21": slots,
        "ineligible_slots_7": inelig,
        "holm_family_16": family,
        "holm_family_topology": {
            "two_width_contrasts": n_two,
            "single_width_contrasts": n_one,
            "not_evaluable_contrasts": n_ne,
            "checkpoint_pair": 1,
            "m2_placeholder": 1,
            "arithmetic": f"{n_two}*2 + {n_one} = 21 eligible slots",
            "survivors_of_holm": "UNDETERMINED — Holm needs 16 p-values and "
                                 "has none",
        },
        "unit_ledger": units,
        "closure_table": closure,
    }
    (EVID / "screen_state.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---------- readable tables ----------
    L = []
    L.append("# M4 CONFIRMATION — what each of the 21 eligible slots did\n")
    L.append("| slot (cell) | contrast it feeds | units planned | attempted | "
             "complete | failed | calibration SD | SD UCB95 | precision |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in slots:
        L.append(f"| `{s['cell']}` | `{s['contrast_fed']}` | "
                 f"{s['units_planned']} | {s['units_attempted']} | "
                 f"{s['units_complete']} | {s['units_failed']} | "
                 f"{s['calibration_sd']} | {s['calibration_sd_ucb95']} | "
                 f"{'supported' if s['calibration_precision_supported'] else 'NOT SUPPORTED'} |")
    L.append(f"\n**Every slot did nothing.** 21 x {per_slot} x {seeds} = "
             f"{units['units_planned']} units planned, "
             f"{units['units_attempted']} attempted. The SD columns are "
             "CALIBRATION dispersion — the evidence that made the slot "
             "eligible — not a confirmation result.\n")
    L.append("## The 7 typed-ineligible slots, never constructed\n")
    L.append("| slot | typed status |")
    L.append("|---|---|")
    for s in inelig:
        L.append(f"| `{s['cell']}` | `{s['typed_status']}` |")
    L.append("\n## The 16-slot Holm family and which contrasts survived\n")
    L.append("| # | slot | frozen disposition | eligible widths | p raw | "
             "p Holm | survived Holm |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(family, 1):
        w = ", ".join(f"w{x}" for x in r["eligible_widths"]) or "—"
        L.append(f"| {i} | `{r['slot']}` | {r['frozen_disposition']} | {w} | "
                 f"— | — | **{r['survived_holm']}** |")
    L.append(f"\n**No contrast survived Holm and none failed it.** Holm is a "
             f"step-down over 16 p-values; zero exist. Topology re-derived: "
             f"{n_two} two-width, {n_one} single-width, {n_ne} NOT_EVALUABLE, "
             f"1 checkpoint pair, 1 M2 placeholder; {n_two}*2 + {n_one} = 21.\n")
    L.append("## Units that failed, and why\n")
    L.append(f"- planned: **{units['units_planned']}**")
    L.append(f"- attempted: **{units['units_attempted']}**")
    L.append(f"- failed in fitting: **{units['units_failed']}** "
             "(nothing reached a fit, so no unit failed numerically)")
    L.append(f"- not attempted: **{units['units_not_attempted']}**, for three "
             "typed reasons:\n")
    for r in units["typed_reason_no_unit_was_attempted"]:
        L.append(f"  - {r}")
    L.append("\n## Closure table\n")
    L.append("| quantity | stage | model error + scale | naive reference, same "
             "rows | skill | literature | comparability |")
    L.append("|---|---|---|---|---|---|---|")
    for r in closure["rows"]:
        lit = r["literature_value"]
        L.append(f"| {r['quantity']} | {r['stage'].split('(')[0].strip()} | "
                 f"{r['model_error_with_scale']} | "
                 f"{r['naive_reference_same_rows']} | {r['skill']} | {lit} | "
                 f"{r['comparability']} |")
    L.append(f"\nComparability reasons, per row, in the order above:\n")
    for i, r in enumerate(closure["rows"], 1):
        L.append(f"{i}. {r['comparability_reason']}")
    L.append(f"\n{closure['note_on_m2']}\n")
    L.append("Generated by `publish_screen_state.py` from "
             f"`{ADJ}` (sha256 `{adj_sha}`) and `{SUCC}` (sha256 "
             f"`{succ_sha}`) at agent-multi@{TIP[:8]}. No value in this file "
             "was typed by hand.\n")
    (EVID / "screen_state.md").write_text("\n".join(L))
    print("\n".join(L[:6]))
    print(f"\nwrote {EVID/'screen_state.json'} and {EVID/'screen_state.md'}")
    print(f"VERDICT: {out['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
