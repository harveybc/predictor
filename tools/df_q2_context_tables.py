#!/usr/bin/env python3
"""Emit the numeric sections of the bounded Q2_CONTEXT publication as markdown, from the run's artifacts only.

The owner's standing rule is that every closure carries the table — model error with its scale, the naive on the SAME rows,
the skill, the literature value with its source, comparability, `NOT_COMPARABLE` with its reason — **generated from
artifacts**. This tool is how those numbers reach the published document. Nothing in it is typed by hand.

It publishes TWO tables, because the corpus's verifier and this run disagree about what may be printed, and the
disagreement is the honest thing to show rather than resolve by preference:

  * **The owner closure table as it landed** (`owner_closure_table.v2`, from `tools/df_closure_table.py`). This run has no
    accepted terminal — no data-gov service key exists on this host — so that tool's policy applies: a score with no
    accepted terminal receipt is printed as **no number at all**. Every error column is `null`, custody is `UNCHECKED`,
    and one problem per unit says why. Published exactly as it landed.
  * **An unanchored measurement table**, `unanchored_measurement_table.v1`, which supplies the five columns the owner's
    rule names, recomputed **independently from the retained `arrays.npz`** of each cell (its `pred`, `y`, `naive` and
    `origins`) and cross-checked against the value the cell record stored. It is NOT an `owner_closure_table.v2` row, it is
    never called verified, and its custody column says `UNANCHORED_NO_ACCEPTED_TERMINAL` in every row.

The contract columns — task, horizon, split, metric, scale, literature value and source, comparability and its reason —
are taken from the landed owner table, which derives them from the sealed contract and therefore carries them even where
it carries no number.

It refuses rather than emit a misleading table: a recomputation that disagrees with the record's stored score; a model and
naive that do not share rows, horizon or scale; an arm whose seeds are not all present; a run whose arrays are not the
record's arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class Refusal(SystemExit):
    pass


def f(x, n=6):
    return "—" if x is None else f"{x:.{n}f}"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(root: Path, unit: str, sd: float) -> dict:
    """MAE of the model and of the naive on the IDENTICAL rows, recomputed from the cell's own retained arrays."""
    d = root / "attempts" / unit
    rec = json.loads((d / "cell.json").read_text())
    if sha_file(d / "arrays.npz") != rec.get("arrays_sha256"):
        raise Refusal(f"REFUSED: {unit}: the arrays on disk are not the record's arrays")
    with np.load(d / "arrays.npz", allow_pickle=False) as z:
        pred, y, naive, origins = z["pred"], z["y"], z["naive"], z["origins"]
    if not (pred.shape == y.shape == naive.shape == origins.shape):
        raise Refusal(f"REFUSED: {unit}: prediction, label, naive and origin arrays are not the same population")
    if not np.isfinite(pred).all():
        raise Refusal(f"REFUSED: {unit}: a non-finite prediction")
    mae = float(np.mean(np.abs(pred - y)))
    nmae = float(np.mean(np.abs(naive - y)))
    if abs(mae - rec["scores"]["mae_kw"]) > 1e-12 or abs(mae / sd - rec["scores"]["mae_z"]) > 1e-12:
        raise Refusal(f"REFUSED: {unit}: the recomputation {mae} does not equal the record's stored score "
                      f"{rec['scores']['mae_kw']}")
    return {"rows": int(origins.size), "mae_kw": mae, "mae_z": mae / sd, "naive_mae_kw": nmae, "naive_mae_z": nmae / sd,
            "skill_vs_naive": 1.0 - mae / nmae, "worse_than_naive": mae > nmae, "record": rec,
            "arrays_sha256": rec["arrays_sha256"], "weights_file_sha256": rec["weights_file_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True, help="the owner_closure_table.v2 JSON for this run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True, help="the unanchored measurement table, as JSON")
    a = ap.parse_args(argv)
    R = a.root
    design = json.loads((R / "DESIGN.json").read_text())
    data = json.loads((R / "BLOCK_DATA.json").read_text())
    pilot = json.loads((R / "REPORT.pilot.json").read_text())
    report = json.loads((R / "REPORT.json").read_text())
    base = json.loads((R / "BASELINES.json").read_text())
    table = json.loads(a.table.read_text())
    if table.get("schema") != "owner_closure_table.v2":
        raise Refusal(f"REFUSED: {a.table} is not an owner_closure_table.v2")

    sd = float(data["sigma_evaluation"])
    seeds = design["seeds"]
    arms = [x["arm"] for x in design["arms"]]
    baseline_arm = design["factorial"]["primary_factorial"][0]
    trows = {r["unit"]: r for r in table["rows"]}
    rrows = {r["cell_id"]: r for r in report["rows"]}
    m = {}
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            if u not in rrows:
                raise Refusal(f"REFUSED: {u} has no closure row")
            m[u] = measure(R, u, sd)
            t = trows.get(u)
            if t is None:
                raise Refusal(f"REFUSED: {u} is absent from the owner closure table")
            if t["model_horizon"] != t["naive_horizon"] or t["model_scale"] != t["naive_scale"]:
                raise Refusal(f"REFUSED: {u}: model and naive do not share horizon and scale")
            if m[u]["rows"] != int(data["common_evaluation"]["n"]):
                raise Refusal(f"REFUSED: {u}: {m[u]['rows']} rows scored, the block's common evaluation has "
                              f"{data['common_evaluation']['n']}")

    L = []
    L.append("### N1. The block as sealed\n")
    L.append(f"* design `{design['design_sha256']}`, schema `{design['schema']}`, block `{design['block']}`, "
             f"state at seal `{design['state']}`, phase `{design['phase']}`")
    L.append(f"* tier: {design['tier']}")
    L.append(f"* prepared data `{data['data_sha256']}`, panel rows {data['rows']['lo']}..{data['rows']['hi']} "
             f"(pad {data['rows']['pad']}), common evaluation **{data['common_evaluation']['n']} origins** "
             f"(panel rows {data['common_evaluation']['first_row']}..{data['common_evaluation']['last_row']}), "
             f"sigma_evaluation {data['sigma_evaluation']} kW")
    L.append(f"* train population `{design['train_population']}`: "
             f"**{data['binding_to_source']['common_train_origins']} origins, identical for every arm**; subset of the "
             f"source run's train origins: {data['binding_to_source']['common_train_subset_of_source']}; the 28 d "
             f"baseline enumeration reproduces the source's train origins: "
             f"{data['binding_to_source']['train_origins_equal_source']}; the common evaluation equals the source's: "
             f"{data['binding_to_source']['common_evaluation_equals_source']}")
    L.append(f"* recipe: {design['recipe']['loss']} loss, {design['recipe']['optimizer']}, lr "
             f"{design['recipe']['learning_rate']}, batch {design['recipe']['batch']}, ceiling "
             f"{design['recipe']['max_updates']} updates, validation every {design['recipe']['validate_every_updates']} "
             f"observed updates, patience {design['recipe']['patience_events']} events, restore_best "
             f"{design['recipe']['restore_best']}, min_delta {design['recipe']['min_delta']}")
    L.append(f"* scaler rule: {design['scaler_rule']}")
    L.append(f"* common evaluation rule: {design['common_evaluation_rule']}")
    L.append("")
    L.append("| arm | window | features | dilations | crop | role | parameters | per-arm train admissible before the intersection |")
    L.append("|---|---:|---|---|---:|---|---:|---:|")
    for x in design["arms"]:
        cap = design["capacity"][x["arm"]]
        fe = data["feasibility"][x["arm"]]
        L.append(f"| `{x['arm']}` | {x['window']} | {x['features']} | "
                 f"{x['dilations'] if x['dilations'] else '—'} | {x['crop'] if x['crop'] else '—'} | "
                 f"{(x.get('role') or 'ARM')} | {cap['parameters']} | {fe['train_admissible_before_intersection']} |")
    L.append("")
    L.append("**Reading rules, verbatim from the sealed design:** " + " · ".join(f"*{r}*" for r in design["reading_rules"]))
    L.append("")
    L.append("**The block's own question, as sealed:** " + design["question"])
    L.append("")
    L.append("**Why the two W1440 full-depth arms of Q2_CONTEXT are not in this block, verbatim from the sealed design:** "
             + design["factorial"]["informed_by"])
    L.append("")

    L.append("### N2. The cost projection the block executed on\n")
    p = pilot["projection"]
    L.append(f"* four cost pilots spent {pilot['spent_cpu_seconds']:.1f} CPU s; projection at the 4 000-update ceiling "
             f"{p['total_at_ceiling_seconds']:.1f} CPU s, with 25 % headroom {p['with_headroom_25_percent']:.1f} CPU s; "
             f"campaign ceiling {design['limits']['campaign_cpu_seconds']} CPU s, closure reserve "
             f"{design['limits']['closure_reserve_seconds']} CPU s")
    L.append(f"* decision **{pilot['decision']}** (`fits_the_ceiling`: {pilot['fits_the_ceiling']}) — contrast with "
             f"Q2_CONTEXT v1's own decision, `BUDGET_LIMITED_BEFORE_ANY_OUTCOME`")
    L.append("")
    L.append("| arm | CPU s per update (pilot) | peak RSS GiB (pilot) | projected CPU s per cell at the ceiling |")
    L.append("|---|---:|---:|---:|")
    for arm, v in p["per_arm"].items():
        L.append(f"| `{arm}` | {v['seconds_per_update']:.4f} | {v['peak_rss_bytes']/2**30:.2f} | "
                 f"{p['per_cell_at_ceiling_seconds'][f'{arm}_s{seeds[0]}']:.1f} |")
    L.append("")

    L.append("### N3. Every fit, as it landed\n")
    L.append("Errors recomputed here from each cell's retained `arrays.npz`, each one cross-checked against the value the "
             "cell record stored (a disagreement above 1e-12 refuses the whole table).\n")
    L.append("| cell | arm | seed | MAE_z | MAE kW | naive MAE kW, same rows | skill vs naive | worse than naive | "
             "stop | censoring | updates | best update | CPU s | peak RSS GiB | reload max err | fresh-process replay |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|")
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            v, r = m[u], rrows[u]
            rep = report["replays"].get(u) or {}
            ok = rep.get("allclose_1e_6")
            L.append(f"| `{u}` | `{arm}` | {s} | {f(v['mae_z'])} | {f(v['mae_kw'])} | {f(v['naive_mae_kw'])} | "
                     f"{f(v['skill_vs_naive'])} | {'**YES**' if v['worse_than_naive'] else 'no'} | {r['stop']} | "
                     f"{r['censoring']} | {r['updates']} | {r['best_update']} | {r['cpu_seconds']:.1f} | "
                     f"{r['peak_rss_bytes']/2**30:.2f} | {r['reload_max_error']:.2e} | "
                     f"{'allclose(1e-6) PASS' if ok else ('FAIL' if ok is False else 'not run')} |")
    L.append("")
    cens = sorted({r["censoring"] for r in rrows.values()})
    L.append(f"Censoring across the twelve fits: {', '.join('`'+c+'`' for c in cens)}. "
             f"Initial-weight digests, per seed: " +
             " · ".join(f"seed {s}: " + ", ".join(f"`{arm}` {rrows[f'{arm}_s{s}']['initial_weights_sha256'][:12]}…"
                                                 for arm in arms) for s in seeds))
    L.append("")

    L.append("### N4. Per arm, and the paired difference against the baseline arm\n")
    per = {arm: [m[f"{arm}_s{s}"]["mae_z"] for s in seeds] for arm in arms}
    kw = {arm: [m[f"{arm}_s{s}"]["mae_kw"] for s in seeds] for arm in arms}
    sk = {arm: [m[f"{arm}_s{s}"]["skill_vs_naive"] for s in seeds] for arm in arms}
    L.append(f"| arm | mean MAE_z | sd (ddof 1) | mean MAE kW | mean skill vs naive | "
             f"paired Δ MAE_z vs `{baseline_arm}`, per seed | mean Δ | signs (+ / −) |")
    L.append("|---|---:|---:|---:|---:|---|---:|---|")
    deltas = {}
    for arm in arms:
        v = per[arm]
        mean = sum(v) / len(v)
        sd_ = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
        if arm == baseline_arm:
            d_txt, dmean, signs = "— (this is the baseline arm)", None, "—"
        else:
            d = [per[arm][i] - per[baseline_arm][i] for i in range(len(seeds))]
            deltas[arm] = d
            d_txt = " · ".join(f"{x:+.6f}" for x in d)
            dmean = sum(d) / len(d)
            signs = f"{sum(1 for x in d if x > 0)} / {sum(1 for x in d if x < 0)}"
        L.append(f"| `{arm}` | {f(mean)} | {f(sd_) if sd_ is not None else '—'} | {f(sum(kw[arm])/len(kw[arm]))} | "
                 f"{f(sum(sk[arm])/len(sk[arm]))} | {d_txt} | {f(dmean) if dmean is not None else '—'} | {signs} |")
    L.append("")
    L.append("Negative Δ = smaller error than the baseline arm. Three paired seeds on one task, one previously inspected "
             "DEV validation week: **development evidence**, as the sealed reading rule says. Both signs are printed and "
             "**no interval is claimed from n = 3**. A difference between two forecast errors is not a verified causal "
             "effect: no causal claim here is verified against a retained-row error, because a causal claim does not "
             "predict a retained row.")
    L.append("")
    null_arm = next((x["arm"] for x in design["arms"] if str(x.get("role") or "").startswith("EXACT_INFORMATION_NULL")), None)
    if null_arm:
        same = [m[f"{null_arm}_s{s}"]["mae_kw"] == m[f"{baseline_arm}_s{s}"]["mae_kw"] for s in seeds]
        inits = [rrows[f"{null_arm}_s{s}"]["initial_weights_sha256"] == rrows[f"{baseline_arm}_s{s}"]["initial_weights_sha256"]
                 for s in seeds]
        L.append(f"**The exact-information null, measured.** `{null_arm}` is the W1440 input cropped to its last 60 rows "
                 f"before the extractor: by RP87 it is the SAME computation as `{baseline_arm}` on the same origins. "
                 f"Measured here rather than assumed: identical initial-weight digest in {sum(inits)} of {len(seeds)} "
                 f"seeds, and MAE equal to the baseline's to the last bit in {sum(same)} of {len(seeds)} seeds. A null "
                 f"that reproduces its treatment exactly is the block's own positive control on its plumbing.")
        L.append("")

    L.append("### N5. The three declared references, on the same rows\n")
    L.append(f"Computed by `df_e1_block.baselines` on the block's {base['common_evaluation_rows']} common evaluation "
             f"origins. The block's closure suppresses these when it fails, so they are published separately.\n")
    L.append("| reference | definition | MAE kW | MAE_z | skill vs persistence |")
    L.append("|---|---|---:|---:|---:|")
    for name, v in base["baselines"].items():
        L.append(f"| `{name}` | {v['definition']} | {f(v['mae_kw'])} | {f(v['mae_z'])} | {f(v['skill_vs_naive'])} |")
    L.append("")
    worse = [n for n, v in base["baselines"].items() if v["skill_vs_naive"] < 0]
    if worse:
        L.append(f"Published exactly as they landed: {', '.join('`'+w+'`' for w in worse)} "
                 f"{'is' if len(worse)==1 else 'are'} **worse** than persistence on these rows.")
        L.append("")

    L.append("### N6. The owner closure table as it landed\n")
    L.append(f"`{table['schema']}` from `tools/df_closure_table.py`, generated {table.get('at')}: "
             f"**{len(table['rows'])} rows, {table['verified_rows']} verified, "
             f"{table['preserved_qualified_rows']} preserved with a qualified scope**; custody classes "
             f"{json.dumps(table['custody_classes'])}; preparation classes {json.dumps(table['preparation_classes'])}.")
    L.append("")
    L.append("**This is the load-bearing fact about this run, and it is printed before any number of mine.** The "
             "verifier's policy is that a score with **no accepted terminal receipt is not reported as a model error at "
             "all** — not as a qualified one. This host holds no data-gov service key, so no terminal was ever accepted, "
             "so every error column below is `null` and every custody is `UNCHECKED`. Published exactly as it landed.")
    L.append("")
    L.append("| unit | task / horizon / split | metric and scale | model error | naive error | skill | "
             "literature value + source | placed in the comparison column | comparability | custody | binding |")
    L.append("|---|---|---|---:|---:|---:|---|---|---|---|---|")
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            r = trows[u]
            lit = r["literature_value_and_source"]
            L.append(f"| `{u}` | {r['task_horizon_split']} | {r['metric_and_scale']} | "
                     f"{'null' if r['model_error'] is None else f(r['model_error'])} | "
                     f"{'null' if r['naive_error'] is None else f(r['naive_error'])} | "
                     f"{(r['skill_vs_naive'] or {}).get('status')} | {lit.get('status')}: {lit.get('source')} — "
                     f"{lit.get('published_value')} ({lit.get('published_or_reproduced')}) | "
                     f"{lit.get('placed_in_comparison_column')} | **{r['comparability_status']}** | "
                     f"{(r.get('custody') or {}).get('class')} | {(r.get('binding') or {}).get('level')} |")
    L.append("")
    L.append("**Why NOT_COMPARABLE, verbatim:** " +
             " · ".join(sorted({r["literature_value_and_source"].get("why_not") or "—" for r in trows.values()})))
    L.append("")
    L.append("**Planned matched comparison, verbatim:** " +
             " · ".join(sorted({r["literature_value_and_source"].get("planned_matched_comparison") or "—" for r in trows.values()})))
    L.append("")
    L.append("**Every problem the table recorded, in full:**")
    L.append("")
    for p_ in table["problems"]:
        L.append(f"* {p_}")
    L.append("")

    L.append("### N7. The unanchored measurement table\n")
    L.append("`unanchored_measurement_table.v1`. It supplies the five columns the owner's rule names, for a run the owner "
             "closure table can only print as `null`. Each error is recomputed from the cell's own retained arrays and "
             "cross-checked against the record's stored score; the contract columns are taken from the landed owner table "
             "above. **It is not an `owner_closure_table.v2` row, it is never verified, and its custody is "
             "`UNANCHORED_NO_ACCEPTED_TERMINAL` in every row.** Nothing may promote, select or rank on it.")
    L.append("")
    L.append("| unit | metric and scale | model error | naive error, SAME rows | skill | rows (model / naive) | "
             "horizon (model / naive) | scale (model / naive) | literature value + source | comparability | custody |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|---|---|---|")
    urows = []
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            v, t = m[u], trows[u]
            lit = t["literature_value_and_source"]
            row = {"unit": u, "arm": arm, "seed": s, "role": "forecast",
                   "task_horizon_split": t["task_horizon_split"], "metric_and_scale": t["metric_and_scale"],
                   "model_error": v["mae_kw"], "model_error_z": v["mae_z"],
                   "naive_error": v["naive_mae_kw"], "naive_error_z": v["naive_mae_z"],
                   "naive_definition": t["naive_definition"],
                   "skill_vs_naive": {"value": v["skill_vs_naive"], "status": "MEASURED",
                                      "reading": "positive = smaller error than the naive; not accuracy, not profit"},
                   "worse_than_naive": v["worse_than_naive"],
                   "model_population": v["rows"], "naive_population": v["rows"],
                   "model_horizon": t["model_horizon"], "naive_horizon": t["naive_horizon"],
                   "model_scale": t["model_scale"], "naive_scale": t["naive_scale"],
                   "literature_value_and_source": lit, "comparability_status": t["comparability_status"],
                   "binding": {"level": "RECORD_DIGEST_LOCAL_ONLY", "arrays_sha256": v["arrays_sha256"],
                               "weights_file_sha256": v["weights_file_sha256"]},
                   "custody": {"class": "UNANCHORED_NO_ACCEPTED_TERMINAL",
                               "why": "no data-gov service key on this host, so no terminal was ever accepted; the arrays "
                                      "are bound to the record that produced them and to nothing else"},
                   "verified": False, "disposition": t["disposition"], "scope": t["scope"]}
            urows.append(row)
            L.append(f"| `{u}` | {row['metric_and_scale']} | {f(row['model_error'])} | {f(row['naive_error'])} | "
                     f"{f(row['skill_vs_naive']['value'])} | {row['model_population']} / {row['naive_population']} | "
                     f"{row['model_horizon']} / {row['naive_horizon']} | {row['model_scale']} / {row['naive_scale']} | "
                     f"{lit.get('status')}: {lit.get('source')} | **{row['comparability_status']}** | "
                     f"{row['custody']['class']} |")
    L.append("")
    nworse = sum(1 for r in urows if r["worse_than_naive"])
    L.append(f"**Fits that landed worse than their naive reference: {nworse} of {len(urows)}.** Every fit's skill is "
             f"printed above whatever its sign.")
    L.append("")

    L.append("### N8. The closure as it landed\n")
    L.append(f"* `{report['schema']}`, design `{report['design_sha256']}`, block `{report['block']}`")
    L.append(f"* **`verified`: {report['verified']}**")
    L.append(f"* common evaluation rows {report['common_evaluation_rows']}; sigma_evaluation "
             f"{report['sigma_evaluation']}; spent CPU {report['spent_cpu_seconds']:.1f} s")
    L.append(f"* closure code drift: `{json.dumps(report['closure_code_drift'])}`")
    L.append(f"* scope: {report['scope']}")
    L.append(f"* disposition: `{json.dumps(report['disposition'])}`; active_selection: "
             f"`{json.dumps(report['active_selection'])}`")
    L.append(f"* `summary`, `paired` and `baselines` are all `None`: a failed closure emits no verified comparator, no "
             f"paired contrast and no selected arm. The per-arm means and the paired differences in N4, and the "
             f"references in N5, are therefore published OUTSIDE the closure, recomputed from the arrays.")
    L.append("")
    L.append("**Every problem the closure recorded, in full:**")
    L.append("")
    for p_ in report["problems"]:
        L.append(f"* {p_}")
    L.append("")
    gov = json.loads((R / "UNGOVERNED_RUN.json").read_text())["steps"][0]
    L.append("### N9. Governance, stated as it is\n")
    L.append(f"* classification **{gov['classification']}**")
    for k, v in gov["governance"].items():
        L.append(f"* {k}: **{v}**")
    for c in gov["consequences"]:
        L.append(f"* {c}")
    L.append(f"* data custody `{gov['delivery']['custody']}`, panel sha256 `{gov['delivery']['sha256']}`, "
             f"{gov['delivery']['bytes']} bytes — {gov['delivery']['why']}")
    L.append(f"* interpreter: Python {gov['interpreter']['python']} (anaconda env `trading-stack`)")
    L.append("")

    a.out.write_text("\n".join(L) + "\n")
    a.json_out.write_text(json.dumps(
        {"schema": "unanchored_measurement_table.v1", "at": table.get("at"),
         "design_sha256": design["design_sha256"], "block": design["block"], "run": table["rows"][0]["run"],
         "what_this_is": "the five columns the owner's closure-table rule names, recomputed from each cell's retained "
                         "arrays for a run with NO accepted terminal, which the owner_closure_table.v2 generator can "
                         "therefore only print as null. Not an owner closure table. No row is verified. Nothing may "
                         "promote, select or rank on it",
         "why_not_an_owner_closure_table": "tools/df_closure_table.py reports no model error for a unit without an "
                                           "accepted terminal receipt; this host holds no data-gov service key",
         "owner_closure_table": {"path": str(a.table), "rows": len(table["rows"]),
                                 "verified_rows": table["verified_rows"],
                                 "custody_classes": table["custody_classes"], "problems": table["problems"]},
         "naive_definition": trows[urows[0]["unit"]]["naive_definition"],
         "sigma_evaluation": sd, "common_evaluation_rows": int(data["common_evaluation"]["n"]),
         "rows": urows, "worse_than_naive_rows": nworse,
         "paired_differences_mae_z_vs_baseline": {"baseline_arm": baseline_arm, "by_arm": deltas,
                                                 "reading": "three paired seeds on one task; both signs reported; no "
                                                            "interval claimed from n = 3; not a verified causal effect"},
         "references": base["baselines"]},
        indent=1, default=str))
    print(json.dumps({"markdown": str(a.out), "json": str(a.json_out), "cells": len(m),
                      "owner_table_rows": len(table["rows"]), "owner_table_verified": table["verified_rows"],
                      "closure_verified": report["verified"], "worse_than_naive": nworse}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
