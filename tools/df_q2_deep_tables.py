#!/usr/bin/env python3
"""Emit the numeric sections of the six-arm Q2_CONTEXT_DEEP publication as markdown, from the run's artifacts only.

Sibling of `tools/df_q2_context_tables.py`, which stays bound to the four-arm bounded block it published. This one adds
what the bounded block could not have: the window x depth CROSSING, its declared contrasts with their per-seed paired
differences and sign counts, the UNDERTRAINED_AT_CEILING verdict per arm computed from each cell's own retained
validation events, and a budget-sensitivity section that measures - for the four arms both blocks share - what the
matched 600-update ceiling costs against the bounded block's 4 000-update ceiling.

It publishes the same TWO tables as its sibling and for the same reason: the owner closure table exactly as
`tools/df_closure_table.py` lands it (this host holds no data-gov service key, so no terminal is ever accepted and that
tool's policy prints no model error at all), and beside it an `unanchored_measurement_table.v1` carrying the real numbers,
recomputed from each cell's retained arrays and never called verified.

Nothing in it is typed by hand. It refuses rather than emit a misleading table: a recomputation that disagrees with the
record; a model and naive that do not share rows, horizon or scale; an arm missing a seed; arrays that are not the
record's arrays; a contrast naming an arm the block does not have; two arms that did not score the identical origins.
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
            "origins": origins, "arrays_sha256": rec["arrays_sha256"], "weights_file_sha256": rec["weights_file_sha256"]}


def tail_improvement(rec: dict):
    """Improvement in validation MAE_z over the LAST `validate_every` observed updates, from the cell's own events:
    val_mae(second-to-last event) - val_mae(last event). Positive = still improving at the ceiling."""
    ev = rec["training"]["events"]
    if len(ev) < 2:
        return None
    return float(ev[-2]["val_mae_scaled"] - ev[-1]["val_mae_scaled"])


def paired(per: dict, a: str, b: str, seeds: list) -> dict:
    """a - b, taken WITHIN each seed. Negative = a has the smaller error."""
    d = [per[a][i] - per[b][i] for i in range(len(seeds))]
    return {"contrast": f"{a} - {b}", "values": d, "mean": sum(d) / len(d),
            "positive": sum(1 for x in d if x > 0), "negative": sum(1 for x in d if x < 0),
            "zero": sum(1 for x in d if x == 0)}


CONTRAST_MEANING = {
    "context_at_matched_depth_10": "input context 1440 vs 60 at depth 10, with capacity identical (12 047 parameters both), "
                                   "depth identical and volume identical: **the context contrast**",
    "context_at_matched_depth_5": "input context at depth 5, where the core reaches only 63 samples: a few extra samples "
                                  "plus the long window's padding, not 1440 samples of information",
    "depth_at_matched_context_60": "core depth 10 vs 5 with the usable context pinned at 60 samples: **the depth contrast**",
    "depth_at_matched_context_1440": "core depth 10 vs 5 with a 1440-row raw window: depth AND the context depth unlocks, "
                                     "jointly, so it identifies neither alone",
    "exact_information_null": "the W1440 input cropped to its last 60 rows: the same computation as the baseline (RP87), "
                              "measured here rather than assumed",
    "causal_channel": "a causal daily-lag channel y(t+h-1440) added to the W60 receiver",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True, help="the owner_closure_table.v2 JSON for this run")
    ap.add_argument("--compare-root", type=Path, help="a previous block's run root whose shared arms were fitted to a "
                                                     "different update ceiling (budget sensitivity section)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    a = ap.parse_args(argv)
    R = a.root
    design = json.loads((R / "DESIGN.json").read_text())
    seal = json.loads((R / "EXTENSION_SEAL.json").read_text())
    data = json.loads((R / "BLOCK_DATA.json").read_text())
    pilot = json.loads((R / "REPORT.pilot.json").read_text())
    report = json.loads((R / "REPORT.json").read_text())
    base = json.loads((R / "BASELINES.json").read_text())
    table = json.loads(a.table.read_text())
    if table.get("schema") != "owner_closure_table.v2":
        raise Refusal(f"REFUSED: {a.table} is not an owner_closure_table.v2")
    if seal["design_sha256"] != design["design_sha256"]:
        raise Refusal("REFUSED: the extension seal record belongs to another design")

    sd = float(data["sigma_evaluation"])
    seeds = design["seeds"]
    arms = [x["arm"] for x in design["arms"]]
    baseline_arm = design["factorial"]["primary_factorial"][0]
    trows = {r["unit"]: r for r in table["rows"]}
    rrows = {r["cell_id"]: r for r in report["rows"]}
    # A cell the block registered but that was NEVER FITTED is carried as missing and named, never quietly dropped and
    # never quietly turned into a smaller design: the design keeps its 18 registered cells whatever landed.
    m, missing = {}, []
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            if not (R / "attempts" / u / "cell.json").is_file():
                missing.append(u)
                continue
            if u not in rrows:
                raise Refusal(f"REFUSED: {u} has a record but no closure row")
            m[u] = measure(R, u, sd)
            t = trows.get(u)
            if t is None:
                raise Refusal(f"REFUSED: {u} is absent from the owner closure table")
            if t["model_horizon"] != t["naive_horizon"] or t["model_scale"] != t["naive_scale"]:
                raise Refusal(f"REFUSED: {u}: model and naive do not share horizon and scale")
            if m[u]["rows"] != int(data["common_evaluation"]["n"]):
                raise Refusal(f"REFUSED: {u}: {m[u]['rows']} rows scored, the common evaluation has "
                              f"{data['common_evaluation']['n']}")
    if not m:
        raise Refusal("REFUSED: not one cell of this block was fitted")
    ref = next(iter(m.values()))["origins"]
    for u, v in m.items():
        if not np.array_equal(v["origins"], ref):
            raise Refusal(f"REFUSED: {u} did not score the same origins as the block's other cells")
    present = sorted({u.rsplit("_s", 1)[0] for u in m})
    complete = [arm for arm in arms if all(f"{arm}_s{s}" in m for s in seeds)]
    partial = [arm for arm in arms if arm in present and arm not in complete]
    if baseline_arm not in complete:
        raise Refusal(f"REFUSED: the baseline arm {baseline_arm} is not complete; no paired contrast can be taken")

    per = {arm: [m[f"{arm}_s{s}"]["mae_z"] for s in seeds] for arm in complete}
    kw = {arm: [m[f"{arm}_s{s}"]["mae_kw"] for s in seeds] for arm in complete}
    sk = {arm: [m[f"{arm}_s{s}"]["skill_vs_naive"] for s in seeds] for arm in complete}
    ceiling = int(design["recipe"]["max_updates"])
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
             f"source run's train origins: {data['binding_to_source']['common_train_subset_of_source']}; the 28 d baseline "
             f"enumeration reproduces the source's train origins: "
             f"{data['binding_to_source']['train_origins_equal_source']}; the common evaluation equals the source's: "
             f"{data['binding_to_source']['common_evaluation_equals_source']}")
    L.append(f"* recipe: {design['recipe']['loss']} loss, {design['recipe']['optimizer']}, lr "
             f"{design['recipe']['learning_rate']}, batch {design['recipe']['batch']}, ceiling "
             f"**{ceiling} updates**, validation every {design['recipe']['validate_every_updates']} observed updates, "
             f"patience {design['recipe']['patience_events']} events, restore_best {design['recipe']['restore_best']}, "
             f"min_delta {design['recipe']['min_delta']}")
    L.append(f"* scaler rule: {design['scaler_rule']}")
    L.append(f"* common evaluation rule: {design['common_evaluation_rule']}")
    L.append("")
    L.append("| arm | raw window | crop | usable context samples | core depth (dilated blocks) | core receptive field | "
             "features | role | parameters | per-arm train admissible before the intersection |")
    L.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|")
    for x in design["arms"]:
        g = seal["arms_derived"][x["arm"]]
        fe = data["feasibility"][x["arm"]]
        L.append(f"| `{x['arm']}` | {g['raw_window']} | {g['crop'] or '—'} | {g['usable_context_samples']} | "
                 f"{g['depth_blocks']} | {g['receptive_field_samples']} | {x['features']} | {(x.get('role') or 'ARM')} | "
                 f"{g['parameters']} | {fe['train_admissible_before_intersection']} |")
    L.append("")
    L.append("**Reading rules, verbatim from the sealed design:** "
             + " · ".join(f"*{r}*" for r in design["reading_rules"]))
    L.append("")
    L.append("**The block's own question, as sealed:** " + design["question"])
    L.append("")
    L.append("**Why this block exists and where its ceiling comes from, verbatim from the sealed design:** "
             + design["factorial"]["informed_by"])
    L.append("")
    L.append("**The budget declaration, verbatim from the sealed recipe:** " + design["recipe"]["budget_declaration"])
    L.append("")

    L.append("### N2. The cost projection the block executed on\n")
    p = pilot["projection"]
    L.append(f"* {len(design['pilots'])} cost pilots spent {pilot['spent_cpu_seconds']:.1f} CPU s; projection at the "
             f"{ceiling}-update ceiling {p['total_at_ceiling_seconds']:.1f} CPU s, with 25 % headroom "
             f"{p['with_headroom_25_percent']:.1f} CPU s; campaign ceiling "
             f"{design['limits']['campaign_cpu_seconds']} CPU s, closure reserve "
             f"{design['limits']['closure_reserve_seconds']} CPU s")
    L.append(f"* decision **{pilot['decision']}** (`fits_the_ceiling`: {pilot['fits_the_ceiling']})")
    L.append("")
    L.append(f"| arm | CPU s per train update (pilot) | peak RSS GiB (pilot) | projected CPU s per cell at the "
             f"{ceiling}-update ceiling | the same projection at a 4 000-update ceiling |")
    L.append("|---|---:|---:|---:|---:|")
    for arm, v in p["per_arm"].items():
        at_ceiling = p["per_cell_at_ceiling_seconds"][f"{arm}_s{seeds[0]}"]
        L.append(f"| `{arm}` | {v['seconds_per_update']:.4f} | {v['peak_rss_bytes']/2**30:.2f} | {at_ceiling:.1f} | "
                 f"{v['seconds_per_update']*4000 + (at_ceiling - v['seconds_per_update']*ceiling)*4000/ceiling:.1f} |")
    L.append("")

    L.append("### N3. Every fit, as it landed\n")
    L.append("Errors recomputed here from each cell's retained `arrays.npz`, each cross-checked against the value the cell "
             "record stored (a disagreement above 1e-12 refuses the whole table), and every arm verified to have scored "
             "the IDENTICAL origin array.\n")
    L.append("| cell | arm | seed | MAE_z | MAE kW | naive MAE kW, same rows | skill vs naive | worse than naive | stop | "
             "censoring | updates | best update | val MAE_z improvement over the last 200 updates | CPU s | "
             "peak RSS GiB | reload max err | fresh-process replay |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            if u not in m:
                L.append(f"| `{u}` | `{arm}` | {s} | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | "
                         f"— | — | — | — | — |")
                continue
            v, r = m[u], rrows[u]
            rep = report["replays"].get(u) or {}
            ok = rep.get("allclose_1e_6")
            ti = tail_improvement(v["record"])
            L.append(f"| `{u}` | `{arm}` | {s} | {f(v['mae_z'])} | {f(v['mae_kw'])} | {f(v['naive_mae_kw'])} | "
                     f"{f(v['skill_vs_naive'])} | {'**YES**' if v['worse_than_naive'] else 'no'} | {r['stop']} | "
                     f"{r['censoring']} | {r['updates']} | {r['best_update']} | "
                     f"{'—' if ti is None else f'{ti:+.6f}'} | {r['cpu_seconds']:.1f} | "
                     f"{r['peak_rss_bytes']/2**30:.2f} | {r['reload_max_error']:.2e} | "
                     f"{'allclose(1e-6) PASS' if ok else ('FAIL' if ok is False else 'not run')} |")
    L.append("")
    cens = sorted({rrows[u]["censoring"] for u in m})
    L.append(f"Censoring across the {len(m)} fits: {', '.join('`'+c+'`' for c in cens)}.")
    L.append("")
    if missing:
        L.append(f"**{len(missing)} of the {len(seeds)*len(arms)} registered cells were NEVER FITTED and are named here, "
                 f"not dropped: {', '.join('`'+u+'`' for u in missing)}.** The design still registers all "
                 f"{len(seeds)*len(arms)}; N13 carries every memory reading that refused to start them. Arms with all "
                 f"{len(seeds)} seeds: {', '.join('`'+x+'`' for x in complete)}"
                 + (f"; arms with SOME seeds: {', '.join('`'+x+'`' for x in partial)}" if partial else "")
                 + ". No contrast is taken across an incomplete arm, and every contrast the missing cells block is printed "
                   "below as `NOT_MEASURED` with the cells that are missing.")
        L.append("")
    L.append("Initial-weight digests, per seed: " + " · ".join(
        f"seed {s}: " + ", ".join(f"`{arm}` {rrows[f'{arm}_s{s}']['initial_weights_sha256'][:12]}…"
                                 for arm in arms if f"{arm}_s{s}" in m)
        for s in seeds))
    L.append("")

    L.append("### N4. Per arm, and the paired difference against the baseline arm\n")
    L.append(f"| arm | mean MAE_z | sd (ddof 1) | mean MAE kW | mean skill vs naive | paired Δ MAE_z vs "
             f"`{baseline_arm}`, per seed | mean Δ | signs (+ / −) |")
    L.append("|---|---:|---:|---:|---:|---|---:|---|")
    deltas = {}
    for arm in complete:
        v = per[arm]
        mean = sum(v) / len(v)
        sd_ = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
        if arm == baseline_arm:
            d_txt, dmean, signs = "— (this is the baseline arm)", None, "—"
        else:
            pd = paired(per, arm, baseline_arm, seeds)
            deltas[arm] = pd["values"]
            d_txt = " · ".join(f"{x:+.6f}" for x in pd["values"])
            dmean, signs = pd["mean"], f"{pd['positive']} / {pd['negative']}"
        L.append(f"| `{arm}` | {f(mean)} | {f(sd_) if sd_ is not None else '—'} | {f(sum(kw[arm])/len(kw[arm]))} | "
                 f"{f(sum(sk[arm])/len(sk[arm]))} | {d_txt} | {f(dmean) if dmean is not None else '—'} | {signs} |")
    L.append("")
    L.append("Negative Δ = smaller error than the baseline arm. Three paired seeds on one task, one previously inspected "
             "DEV validation week: **development evidence**, as the sealed reading rule says. Both signs are printed and "
             "**no interval is claimed from n = 3**. A difference between two forecast errors is not a verified causal "
             "effect: no causal claim here is verified against a retained-row error, because a causal claim does not "
             "predict a retained row.")
    L.append("")

    L.append("### N5. The crossing: input context x core depth, with volume fixed by construction\n")
    grid = seal["crossing_window_x_depth"]
    L.append("Volume is not a factor here: every arm trained on the SAME "
             f"{data['binding_to_source']['common_train_origins']} origins and scored the SAME "
             f"{data['common_evaluation']['n']}, both checked as arrays above. The two factors that remain are the raw "
             "input window and the depth of the causal core.\n")
    L.append("| | depth 5 blocks | depth 10 blocks |")
    L.append("|---|---|---|")
    for w in (60, 1440):
        cells_row = []
        for d in (5, 10):
            nms = grid.get(f"window_{w}__depth_{d}") or []
            txt = []
            for nm in nms:
                g = seal["arms_derived"][nm]
                if nm not in per:
                    got = [f"s{s}" for s in seeds if f"{nm}_s{s}" in m]
                    txt.append(f"`{nm}`<br>**NOT FITTED** ({'no seed' if not got else 'only ' + ', '.join(got)})"
                               f"<br>{g['parameters']} parameters, usable context {g['usable_context_samples']} samples")
                    continue
                mean = sum(per[nm]) / len(per[nm])
                txt.append(f"`{nm}`<br>mean MAE_z **{mean:.6f}**<br>{g['parameters']} parameters, usable context "
                           f"{g['usable_context_samples']} samples")
            cells_row.append("<br>".join(txt) if txt else "**(no arm)**")
        L.append(f"| **raw window {w}** | {cells_row[0]} | {cells_row[1]} |")
    L.append("")
    contrasts = seal["contrasts_declared_before_any_score"]
    computed = {}
    L.append("**The contrasts, exactly as the seal declared them before any fit.** Each is taken within a seed.\n")
    L.append("| declared contrast | what it isolates | per-seed Δ MAE_z | mean Δ | signs (+ / −) |")
    L.append("|---|---|---|---:|---|")
    for key, expr in contrasts.items():
        if " - " not in str(expr):
            continue
        aa, bb = [s.strip() for s in str(expr).split(" - ")]
        bb = bb.split(",")[0].strip()
        if aa not in arms or bb not in arms:
            raise Refusal(f"REFUSED: declared contrast {key} names an arm this block does not have: {expr}")
        if aa not in per or bb not in per:
            gone = [u for u in missing if u.rsplit("_s", 1)[0] in (aa, bb)]
            computed[key] = {"contrast": f"{aa} - {bb}", "status": "NOT_MEASURED", "missing_cells": gone}
            L.append(f"| `{key}`<br>`{aa} - {bb}` | {CONTRAST_MEANING.get(key, '—')} | **NOT_MEASURED** — these cells "
                     f"were never fitted: {', '.join('`'+u+'`' for u in gone)} | — | — |")
            continue
        pd = paired(per, aa, bb, seeds)
        computed[key] = pd
        L.append(f"| `{key}`<br>`{aa} - {bb}` | {CONTRAST_MEANING.get(key, '—')} | "
                 f"{' · '.join(f'{x:+.6f}' for x in pd['values'])} | {pd['mean']:+.6f} | "
                 f"{pd['positive']} / {pd['negative']} |")
    inter = contrasts.get("interaction")
    four = ("long_window_own_depth", "short_window_deep_core", "long_window_local_support_67", "modular_w60")
    if inter and not all(k in per for k in four):
        gone = [u for u in missing if u.rsplit("_s", 1)[0] in four]
        computed["interaction"] = {"contrast": inter, "status": "NOT_MEASURED", "missing_cells": gone}
        L.append(f"| `interaction`<br>`{inter}` | does the context difference itself depend on depth | **NOT_MEASURED** — "
                 f"these cells were never fitted: {', '.join('`'+u+'`' for u in gone)} | — | — |")
    if inter and all(k in per for k in four):
        i_vals = [(per["long_window_own_depth"][i] - per["short_window_deep_core"][i])
                  - (per["long_window_local_support_67"][i] - per["modular_w60"][i]) for i in range(len(seeds))]
        computed["interaction"] = {"contrast": inter, "values": i_vals, "mean": sum(i_vals) / len(i_vals),
                                  "positive": sum(1 for x in i_vals if x > 0),
                                  "negative": sum(1 for x in i_vals if x < 0),
                                  "zero": sum(1 for x in i_vals if x == 0)}
        L.append(f"| `interaction`<br>`{inter}` | does the context difference itself depend on depth | "
                 f"{' · '.join(f'{x:+.6f}' for x in i_vals)} | {sum(i_vals)/len(i_vals):+.6f} | "
                 f"{computed['interaction']['positive']} / {computed['interaction']['negative']} |")
    L.append("")
    L.append("Negative Δ = the first-named arm has the smaller error. **Three seeds is three seeds:** a sign count of "
             "3 / 0 on n = 3 is a direction, not an effect, and no interval is claimed for any row above.")
    L.append("")

    rule = seal["undertrained_at_ceiling_rule"]
    L.append("### N6. The UNDERTRAINED_AT_CEILING verdict, by the rule sealed before any fit\n")
    L.append(f"* statistic: {rule['statistic']}")
    L.append(f"* threshold: **{rule['threshold']}** MAE_z")
    L.append(f"* verdict: {rule['verdict']}")
    L.append(f"* declared: {rule['declared']}")
    L.append("")
    L.append("| arm | improvement over the last 200 updates, per seed | seeds above the threshold | verdict |")
    L.append("|---|---|---:|---|")
    verdicts = {}
    for arm in arms:
        if arm not in complete:
            got = [f"s{s}" for s in seeds if f"{arm}_s{s}" in m]
            verdicts[arm] = {"tail_improvements": [tail_improvement(m[f"{arm}_s{s}"]["record"]) if f"{arm}_s{s}" in m
                                                   else None for s in seeds],
                             "seeds_above_threshold": None, "verdict": "NOT_FITTED_NO_VERDICT",
                             "seeds_present": got}
            L.append(f"| `{arm}` | {' · '.join('—' if v is None else f'{v:+.6f}' for v in verdicts[arm]['tail_improvements'])} "
                     f"| — | **NOT_FITTED_NO_VERDICT** |")
            continue
        vals = [tail_improvement(m[f"{arm}_s{s}"]["record"]) for s in seeds]
        above = sum(1 for v in vals if v is not None and v > rule["threshold"])
        verdicts[arm] = {"tail_improvements": vals, "seeds_above_threshold": above,
                         "verdict": "UNDERTRAINED_AT_CEILING" if above >= 2 else "NOT_UNDERTRAINED_BY_THIS_RULE"}
        L.append(f"| `{arm}` | {' · '.join('—' if v is None else f'{v:+.6f}' for v in vals)} | {above} | "
                 f"**{verdicts[arm]['verdict']}** |")
    L.append("")
    flagged = [k for k, v in verdicts.items() if v["verdict"] == "UNDERTRAINED_AT_CEILING"]
    if flagged:
        L.append(f"**{len(flagged)} arm(s) are UNDERTRAINED_AT_CEILING: {', '.join('`'+x+'`' for x in flagged)}.** By the "
                 f"rule sealed before any fit, no contrast involving them is read as a context or depth effect. The "
                 f"numbers stay published exactly as they landed.")
    else:
        L.append("**No arm is UNDERTRAINED_AT_CEILING by the sealed rule.** Every cell is still `CENSORED_BY_BUDGET` — the "
                 "ceiling was reached — and the rule says only that no arm was still improving faster than the threshold "
                 "when it got there.")
    L.append("")

    L.append("### N7. The three declared references, on the same rows\n")
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

    L.append("### N8. The owner closure table as it landed\n")
    L.append(f"`{table['schema']}` from `tools/df_closure_table.py`, generated {table.get('at')}: "
             f"**{len(table['rows'])} rows, {table['verified_rows']} verified, "
             f"{table['preserved_qualified_rows']} preserved with a qualified scope**; custody classes "
             f"{json.dumps(table['custody_classes'])}; preparation classes {json.dumps(table['preparation_classes'])}.")
    L.append("")
    L.append("**This is the load-bearing fact about this run, and it is printed before any number of mine.** The verifier's "
             "policy is that a score with **no accepted terminal receipt is not reported as a model error at all** — not as "
             "a qualified one. This host holds no data-gov service key, so no terminal was ever accepted, so every error "
             "column below is `null` and every custody is `UNCHECKED`. Published exactly as it landed; the verifier's "
             "custody policy was not touched to make a number appear.")
    L.append("")
    L.append("| unit | task / horizon / split | metric and scale | model error | naive error | skill | "
             "literature value + source | placed in the comparison column | comparability | custody | binding |")
    L.append("|---|---|---|---:|---:|---:|---|---|---|---|---|")
    for s in seeds:
        for arm in arms:
            u = f"{arm}_s{s}"
            if u not in trows:
                L.append(f"| `{u}` | — | — | NOT FITTED | NOT FITTED | — | — | — | — | — | — |")
                continue
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
    L.append("**Why NOT_COMPARABLE, verbatim:** "
             + " · ".join(sorted({r["literature_value_and_source"].get("why_not") or "—" for r in trows.values()})))
    L.append("")
    L.append("**Planned matched comparison, verbatim:** "
             + " · ".join(sorted({r["literature_value_and_source"].get("planned_matched_comparison") or "—"
                                  for r in trows.values()})))
    L.append("")
    L.append("**Every problem the table recorded, in full:**")
    L.append("")
    for p_ in table["problems"]:
        L.append(f"* {p_}")
    L.append("")

    L.append("### N9. The unanchored measurement table\n")
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
            if u not in m:
                continue
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
                   "censoring": rrows[u]["censoring"], "updates": rrows[u]["updates"],
                   "best_update": rrows[u]["best_update"],
                   "val_mae_z_improvement_over_last_200_updates": tail_improvement(v["record"]),
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

    budget = None
    if a.compare_root:
        CR = Path(a.compare_root)
        cdes = json.loads((CR / "DESIGN.json").read_text())
        cdata = json.loads((CR / "BLOCK_DATA.json").read_text())
        csd = float(cdata["sigma_evaluation"])
        shared = [x["arm"] for x in cdes["arms"] if x["arm"] in complete]
        L.append(f"### N10. What the matched {ceiling}-update ceiling costs, measured on the arms both blocks share\n")
        L.append(f"The block `{cdes['block']}` (design `{cdes['design_sha256'][:16]}…`) fitted {len(shared)} of these arms "
                 f"on the SAME rows, the SAME scaler, the SAME seeds and the SAME cadence, but to a "
                 f"**{cdes['recipe']['max_updates']}-update** ceiling with patience {cdes['recipe']['patience_events']} "
                 f"allowed to expire. Its numbers are a direct measurement of what this block's budget costs a cheap arm — "
                 f"and the only honest way to say how far an arm might still have had to travel.\n")
        L.append(f"| arm | mean MAE_z at {ceiling} updates (this block) | mean MAE_z at "
                 f"{cdes['recipe']['max_updates']} updates (that block) | Δ (this − that) | best update there, per seed |")
        L.append("|---|---:|---:|---:|---|")
        rows_b = {}
        for arm in shared:
            this_mean = sum(per[arm]) / len(per[arm])
            thats, bests = [], []
            for s in cdes["seeds"]:
                cm = measure(CR, f"{arm}_s{s}", csd)
                thats.append(cm["mae_z"])
                bests.append(cm["record"]["training"]["best_update"])
            that_mean = sum(thats) / len(thats)
            rows_b[arm] = {"this_block_mean_mae_z": this_mean, "other_block_mean_mae_z": that_mean,
                           "delta": this_mean - that_mean, "other_block_best_updates": bests,
                           "other_block_per_seed_mae_z": thats}
            L.append(f"| `{arm}` | {f(this_mean)} | {f(that_mean)} | {f(this_mean - that_mean)} | "
                     f"{', '.join(str(b) for b in bests)} |")
        L.append("")
        budget = {"compare_root": str(CR), "compare_block": cdes["block"], "compare_design_sha256": cdes["design_sha256"],
                  "compare_max_updates": cdes["recipe"]["max_updates"], "shared_arms": shared, "by_arm": rows_b,
                  "reading": "a measurement of the budget's cost on the arms both blocks share, not a correction applied "
                             "to any number; no arm of this block is adjusted by it and no cross-block difference is used "
                             "as a contrast"}
        L.append("This section compares two BLOCKS, not two arms: it measures the budget, and nothing in N5 is adjusted "
                 "by it.")
        L.append("")

    L.append("### N11. The closure as it landed\n")
    L.append(f"* `{report['schema']}`, design `{report['design_sha256']}`, block `{report['block']}`")
    L.append(f"* **`verified`: {report['verified']}**")
    L.append(f"* common evaluation rows {report['common_evaluation_rows']}; sigma_evaluation "
             f"{report['sigma_evaluation']}; spent CPU {report['spent_cpu_seconds']:.1f} s")
    L.append(f"* closure code drift: `{json.dumps(report['closure_code_drift'])}`")
    L.append(f"* scope: {report['scope']}")
    L.append(f"* disposition: `{json.dumps(report['disposition'])}`; active_selection: "
             f"`{json.dumps(report['active_selection'])}`")
    L.append("* `summary`, `paired` and `baselines` are all `None`: a failed closure emits no verified comparator, no "
             "paired contrast and no selected arm. The per-arm means in N4, the crossing in N5 and the references in N7 "
             "are therefore published OUTSIDE the closure, recomputed from the arrays.")
    L.append("")
    L.append("**Every problem the closure recorded, in full:**")
    L.append("")
    for p_ in report["problems"]:
        L.append(f"* {p_}")
    L.append("")
    reps = {u: r for u, r in report["replays"].items() if "allclose_1e_6" in r}
    if reps:
        diffs = [r.get("max_abs_difference") for r in reps.values() if r.get("max_abs_difference") is not None]
        L.append(f"**Fresh-process replays: {sum(1 for r in reps.values() if r.get('allclose_1e_6'))} of {len(reps)} pass "
                 f"`allclose(1e-6, 1e-6)`"
                 + (f"; the maximum absolute difference over every replayed cell is {max(diffs):.3e} kW.**" if diffs
                    else ".**"))
        L.append("")
        L.append("| cell | fresh-process replay | max abs difference (kW) |")
        L.append("|---|---|---:|")
        for s in seeds:
            for arm in arms:
                u = f"{arm}_s{s}"
                if u not in m:
                    continue
                r = reps.get(u) or {}
                d = r.get("max_abs_difference")
                L.append(f"| `{u}` | {'allclose(1e-6) PASS' if r.get('allclose_1e_6') else 'FAIL/absent'} | "
                         f"{'—' if d is None else f'{d:.3e}'} |")
        L.append("")

    gov = json.loads((R / "UNGOVERNED_RUN.json").read_text())["steps"][0]
    L.append("### N12. Governance, stated as it is\n")
    L.append(f"* classification **{gov['classification']}**")
    for k, v in gov["governance"].items():
        L.append(f"* {k}: **{v}**")
    for c in gov["consequences"]:
        L.append(f"* {c}")
    L.append(f"* data custody `{gov['delivery']['custody']}`, panel sha256 `{gov['delivery']['sha256']}`, "
             f"{gov['delivery']['bytes']} bytes — {gov['delivery']['why']}")
    L.append(f"* interpreter: Python {gov['interpreter']['python']} (anaconda env `trading-stack`)")
    L.append("")

    gate_rows = []
    gate = R / "MEMORY_GATE.jsonl"
    if gate.is_file():
        entries = [json.loads(x) for x in gate.read_text().splitlines() if x.strip()]
        gate_rows = entries
        held = [e for e in entries if e.get("verdict") == "HELD_WAITING_FOR_MEMORY"]
        launched = [e for e in entries if e.get("verdict") == "LAUNCH"]
        fin = [e for e in entries if e.get("verdict") == "FINISHED"]
        never = [e for e in entries if e.get("verdict") == "NOT_STARTED_MEMORY_NEVER_ALLOWED_IT"]
        L.append("### N13. Placement: every memory reading taken before a launch\n")
        L.append(f"* `MEMORY_GATE.jsonl`: {len(entries)} readings — {len(launched)} launches, {len(held)} holds, "
                 f"{len(fin)} finished jobs, {len(never)} units NOT started because memory never allowed it")
        if launched:
            L.append(f"* MemAvailable at launch: min {min(e['mem_available_bytes'] for e in launched)/2**30:.2f} GiB, "
                     f"max {max(e['mem_available_bytes'] for e in launched)/2**30:.2f} GiB")
        if held:
            L.append(f"* MemAvailable while held: min {min(e['mem_available_bytes'] for e in held)/2**30:.2f} GiB, "
                     f"max {max(e['mem_available_bytes'] for e in held)/2**30:.2f} GiB; longest single wait "
                     f"{max(e['waited_seconds'] for e in held):.0f} s")
        L.append("")
        L.append("| label | verdict | MemAvailable GiB | measured pilot peak GiB | required (peak + margin) GiB | cap |")
        L.append("|---|---|---:|---:|---:|---|")
        for e in entries:
            if e.get("verdict") == "FINISHED":
                L.append(f"| `{e['label']}` | FINISHED exit {e['exit_code']} | "
                         f"{e['mem_available_bytes_after']/2**30:.2f} (after) | — | — | — |")
            else:
                L.append(f"| `{e['label']}` | {e['verdict']} | {e['mem_available_bytes']/2**30:.2f} | "
                         f"{e['measured_pilot_peak_bytes']/2**30:.2f} | {e['required_bytes']/2**30:.2f} | {e['cap']} |")
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
         "common_train_origins": data["binding_to_source"]["common_train_origins"],
         "update_ceiling": ceiling,
         "rows": urows, "worse_than_naive_rows": nworse,
         "registered_cells": [f"{arm}_s{s}" for s in seeds for arm in arms],
         "cells_never_fitted": missing, "arms_complete": complete, "arms_partial": partial,
         "paired_differences_mae_z_vs_baseline": {"baseline_arm": baseline_arm, "by_arm": deltas,
                                                 "reading": "three paired seeds on one task; both signs reported; no "
                                                            "interval claimed from n = 3; not a verified causal effect"},
         "crossing": {"grid": grid, "arms_derived": seal["arms_derived"], "contrasts": computed,
                      "reading": "volume fixed by construction (one COMMON_INTERSECTION train origin set); every contrast "
                                 "taken within a seed; three seeds is three seeds"},
         "undertrained_at_ceiling": {"rule": rule, "by_arm": verdicts},
         "budget_sensitivity": budget,
         "memory_gate_readings": gate_rows,
         "references": base["baselines"]},
        indent=1, default=str))
    print(json.dumps({"markdown": str(a.out), "json": str(a.json_out), "cells": len(m),
                      "owner_table_rows": len(table["rows"]), "owner_table_verified": table["verified_rows"],
                      "closure_verified": report["verified"], "worse_than_naive": nworse,
                      "undertrained_arms": flagged, "cells_never_fitted": missing, "arms_complete": complete,
                      "contrasts": {k: {"mean": v["mean"], "signs": f"{v['positive']}/{v['negative']}"}
                                    for k, v in computed.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
