#!/usr/bin/env python3
"""RP18: contrast-bound effects of the ARCH comparison from a closure (CLOSE.json) and its sealed design.

Every effect is a CONTRAST with an exact member list, fixed weights and a denominator derived from the
design (never from what survived verification). The closure must be the closure of the very design
(identity), its population must be the design's, every member must be a VERIFIED cell whose record
agrees with its id and the design, and every value must be finite; a foreign design, a duplicated
member, an unexpected arm or a non-finite value REFUSES. A contrast whose members are not all
present is INCOMPLETE with n expected / observed and the reasons; it is never filled with other
arms' averages. One estimator serves the point estimate and the replicate bootstrap.

Contrasts (per architecture a):
  H2       e_a(h) = mean over replicates of [MASE(profiles) - mean over ALL random_k]; slope over h.
  COMMON   d_a,r = mean over replicates of [MASE(sequence) - MASE(summary)] (donor sequence), at each r;
           gamma_common_pair = d_a,1 - d_a,0 (the SAME pair at both r).
  FACT     2 x 2 fusion x readout at r: fusion = mean(last, gap of sequence fusion) - mean(gap, last of
           summary fusion); readout = mean(sequence, summary_last) - mean(sequence_gap, summary);
           interaction; gamma_factorial = fusion(r=1) - fusion(r=0) only when the 2 x 2 exists at BOTH r.
  READOUT  per fusion at r: sequence - sequence_gap (sequence fusion); summary_last - summary (summary fusion).
  DONOR    delta_a = [seq_dsum - summ_dsum] - [seq - summ] per replicate at r = 1: the SAME common pair.
  DX       adequacy rows only (no effect).
Adequacy is reported apart: persistence naive (the naive of prepare()), the MASE denominator (train
seasonal-naive MAE, a denominator, not a predictor score), the linear reference (point gap AND the
historic +0.03 criterion), the oracle. No `interpretable` flag.

    python tools/df_mod_e0_arch_verify.py --close CLOSE.json --design DESIGN.json --out EFFECTS.json [--tables T.md]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "df_mod_e0_arch_effects.v2"
ESTIMATED, INCOMPLETE, NOT_ESTIMABLE = "ESTIMATED", "INCOMPLETE", "NOT_ESTIMABLE"
LINEAR_TOLERANCE = 0.03


class EffectsRefusal(SystemExit):
    def __init__(self, why: str):
        super().__init__(why)
        self.code = 2
        self.why = why


def _fin(x) -> float:
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise EffectsRefusal(f"REFUSED: non-finite or non-numeric value {x!r}")
    return float(x)


def parse_id(cell_id: str) -> dict:
    parts = cell_id.split("__")
    if parts[0] in ("H2", "H3") and len(parts) >= 5:
        return {"hypothesis": parts[0], "cond": parts[1], "seed": int(parts[2][1:]), "arch": parts[3], "arm": parts[4],
                "donor": "summary" if (len(parts) > 5 and parts[5] == "dsum") else ("sequence" if parts[0] == "H3" else None)}
    if parts[0] == "DX" and len(parts) >= 5:
        return {"hypothesis": "DX", "cond": parts[1], "seed": int(parts[2][1:]), "arch": parts[3], "arm": parts[4], "donor": None}
    raise EffectsRefusal(f"REFUSED: unrecognised cell id {cell_id!r}")


# --- the contracts derived from the design ------------------------------------------------------------------

def contrasts(design: dict) -> dict:
    """Member lists, weights and denominators of every contrast, from the design alone."""
    by_id = {c["cell_id"]: c for c in design["cells"]}
    archs, seeds, levels, rs = design["archs"], sorted(design["replicates"]), sorted(design["levels"]), sorted(design["r_values"])
    ctrl_r = set(design.get("readout_controls_r") or rs)
    ds = design.get("donor_sensitivity") or {}
    out = {"design_sha256": design["design_sha256"], "replicates": seeds, "per_arch": {}}
    for a in archs:
        ent = {"H2": {}, "COMMON": {}, "FACT": {}, "READOUT": {}, "DONOR": None, "DX": []}
        for h in levels:
            ent["H2"][h] = {s: {"profiles": f"H2__h{h}__s{s}__{a}__profiles",
                                "random": [f"H2__h{h}__s{s}__{a}__random_{k}" for k in range(design["random_assignments"])]} for s in seeds}
        for r in rs:
            ent["COMMON"][r] = {s: {"sequence": f"H3__r{r}__s{s}__{a}__sequence", "summary": f"H3__r{r}__s{s}__{a}__summary"} for s in seeds}
            if r in ctrl_r:
                ent["FACT"][r] = {s: {arm: f"H3__r{r}__s{s}__{a}__{arm}" for arm in ("sequence", "sequence_gap", "summary", "summary_last")} for s in seeds}
        if a in (ds.get("archs") or []):
            ent["DONOR"] = {r: {s: {"sequence_dsum": f"H3__r{r}__s{s}__{a}__sequence__dsum", "summary_dsum": f"H3__r{r}__s{s}__{a}__summary__dsum",
                                    "sequence": f"H3__r{r}__s{s}__{a}__sequence", "summary": f"H3__r{r}__s{s}__{a}__summary"} for s in seeds}
                            for r in (ds.get("r") or [])}
        dx = design.get("diagnostic") or {}
        if dx:
            ent["DX"] = [f"DX__{dx['condition']}__s{s}__{a}__{arm}" for s in dx.get("seeds", []) for arm in dx.get("arms", ["profiles"])]
        out["per_arch"][a] = ent
    # every member named must be a design cell
    for a, ent in out["per_arch"].items():
        for group in ("H2", "COMMON", "FACT"):
            for _, by_seed in ent[group].items():
                for _, members in by_seed.items():
                    for m in (members.values() if isinstance(members, dict) else []):
                        for cid in (m if isinstance(m, list) else [m]):
                            if cid not in by_id:
                                raise EffectsRefusal(f"REFUSED: contrast member {cid} is not a design cell")
    out["weights"] = "equal weight per replicate; random assignments averaged within the replicate; readouts averaged within the fusion"
    return out


# --- binding of the closure to the design ---------------------------------------------------------------------

def bind(close_local: dict, design: dict, split: str) -> dict:
    """{cell_id: value} for every VERIFIED cell whose record agrees with its id and the design; refusals on
    identity, population, duplicates, unexpected arms and non-finite values."""
    if close_local.get("design_sha256") != design.get("design_sha256"):
        raise EffectsRefusal(f"REFUSED: the closure is of design {close_local.get('design_sha256')!r}, not {design.get('design_sha256')!r}")
    members = list((close_local.get("population") or {}).get("members") or [])
    expected = [c["cell_id"] for c in design["cells"]]
    if members != expected:
        raise EffectsRefusal("REFUSED: the closure's population is not the design's enumeration")
    if len(set(members)) != len(members):
        raise EffectsRefusal("REFUSED: duplicated member in the population")
    by_id = {c["cell_id"]: c for c in design["cells"]}
    units = close_local.get("units") or {}
    strangers = sorted(k for k, u in units.items() if u.get("role") == "CELL" and k not in by_id)
    if strangers:
        raise EffectsRefusal(f"REFUSED: unexpected cell(s) in the closure: {strangers[:5]}")
    values, states = {}, {}
    for cid, c in by_id.items():
        u = units.get(cid)
        if u is None:
            states[cid] = "ABSENT"
            continue
        if u.get("status") != "VERIFIED" or u.get("role") != "CELL":
            states[cid] = str(u.get("status"))
            continue
        rec = u.get("record") or {}
        meta = parse_id(cid)
        if (str(rec.get("arch")) != str(c["arch"]) or rec.get("arm") != c["arm"] or int(rec.get("seed", -1)) != int(c["seed"])
                or rec.get("hypothesis") != c["hypothesis"] or int(rec.get("level", -1)) != int(c["level"]) or int(rec.get("r", -1)) != int(c["r"])
                or (rec.get("donor") or ("sequence" if c.get("depends_on") else None)) != (c.get("donor") or ("sequence" if c.get("depends_on") else None))):
            raise EffectsRefusal(f"REFUSED: {cid}: the record's identity does not match its id / the design")
        if meta["arm"] != c["arm"]:
            raise EffectsRefusal(f"REFUSED: {cid}: unexpected arm")
        values[cid] = _fin((rec.get("mase") or {}).get(split))
        states[cid] = "VERIFIED"
    return {"values": values, "states": states}


# --- the single estimator --------------------------------------------------------------------------------------

def estimate(spec: dict, values: dict, seeds: list) -> dict:
    """One contrast on a set of replicate ids: per-replicate value from the FULL member list (or the
    replicate is missing), mean and SD; complete only if every replicate's every member is present."""
    per, missing = {}, {}
    for s in seeds:
        members = spec.get(s)
        if members is None:
            missing[s] = ["replicate not in the design"]
            continue
        flat = []
        for m in members.values():
            flat += m if isinstance(m, list) else [m]
        absent = [cid for cid in flat if cid not in values]
        if absent:
            missing[s] = absent
            continue
        per[s] = spec["_f"](members, values)
    n_expected, n_observed = len(seeds), len(per)
    vals = list(per.values())
    return {"state": ESTIMATED if (n_observed == n_expected and n_expected > 0) else (INCOMPLETE if n_observed > 0 or n_expected > 0 else NOT_ESTIMABLE),
            "n_expected": n_expected, "n_observed": n_observed, "complete": n_observed == n_expected and n_expected > 0,
            "value": float(np.mean(vals)) if vals else None, "sd_replicates": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "per_replicate": {int(s): float(v) for s, v in per.items()}, "missing": {int(s): m for s, m in missing.items()}}


def _f_h2(members, values):
    return values[members["profiles"]] - float(np.mean([values[k] for k in members["random"]]))


def _f_common(members, values):
    return values[members["sequence"]] - values[members["summary"]]


def _f_fusion(members, values):
    return float(np.mean([values[members["sequence"]], values[members["sequence_gap"]]]) - np.mean([values[members["summary"]], values[members["summary_last"]]]))


def _f_readout(members, values):
    return float(np.mean([values[members["sequence"]], values[members["summary_last"]]]) - np.mean([values[members["sequence_gap"]], values[members["summary"]]]))


def _f_interaction(members, values):
    return float((values[members["sequence"]] - values[members["sequence_gap"]]) - (values[members["summary_last"]] - values[members["summary"]]))


def _f_readout_seq(members, values):
    return values[members["sequence"]] - values[members["sequence_gap"]]


def _f_readout_sum(members, values):
    return values[members["summary_last"]] - values[members["summary"]]


def _f_donor(members, values):
    return (values[members["sequence_dsum"]] - values[members["summary_dsum"]]) - (values[members["sequence"]] - values[members["summary"]])


def _slope(levels, values):
    if len(levels) < 2:
        return None
    A = np.vstack([np.asarray(levels, dtype=float), np.ones(len(levels))]).T
    return float(np.linalg.lstsq(A, np.asarray(values), rcond=None)[0][0])


def _spec(by_seed: dict, f) -> dict:
    return {**by_seed, "_f": f}


def _difference(a: dict, b: dict, label: str) -> dict:
    """a - b of two ESTIMATED contrasts on the same replicates (paired)."""
    if a["state"] != ESTIMATED or b["state"] != ESTIMATED:
        return {"state": NOT_ESTIMABLE, "value": None, "why": f"{label}: both terms must be complete ({a['state']} / {b['state']})"}
    per = {s: a["per_replicate"][s] - b["per_replicate"][s] for s in a["per_replicate"] if s in b["per_replicate"]}
    vals = list(per.values())
    return {"state": ESTIMATED, "value": float(np.mean(vals)), "sd_replicates": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "per_replicate": per, "n_replicates": len(vals)}


def arch_effects(a: str, ent: dict, values: dict, seeds: list, levels: list, rs: list) -> dict:
    out = {"H2": {"e": {}, "slope": None, "state": None}, "COMMON": {"d": {}}, "FACT": {}, "READOUT": {}, "DONOR": None}
    for h in levels:
        out["H2"]["e"][h] = estimate(_spec(ent["H2"][h], _f_h2), values, seeds)
    est_levels = [h for h in levels if out["H2"]["e"][h]["state"] == ESTIMATED]
    out["H2"]["state"] = ESTIMATED if len(est_levels) == len(levels) else INCOMPLETE
    out["H2"]["slope"] = _slope(est_levels, [out["H2"]["e"][h]["value"] for h in est_levels]) if (len(est_levels) == len(levels) and len(levels) >= 2) else None
    for r in rs:
        out["COMMON"]["d"][r] = estimate(_spec(ent["COMMON"][r], _f_common), values, seeds)
    out["COMMON"]["gamma_common_pair"] = _difference(out["COMMON"]["d"].get(1, {"state": NOT_ESTIMABLE}), out["COMMON"]["d"].get(0, {"state": NOT_ESTIMABLE}), "gamma_common_pair") \
        if (0 in out["COMMON"]["d"] and 1 in out["COMMON"]["d"]) else {"state": NOT_ESTIMABLE, "value": None, "why": "r = 0 and r = 1 both required"}
    for r, by_seed in ent["FACT"].items():
        out["FACT"][r] = {"fusion": estimate(_spec(by_seed, _f_fusion), values, seeds), "readout": estimate(_spec(by_seed, _f_readout), values, seeds),
                          "interaction": estimate(_spec(by_seed, _f_interaction), values, seeds)}
        out["READOUT"][r] = {"sequence_fusion": estimate(_spec(by_seed, _f_readout_seq), values, seeds),
                             "summary_fusion": estimate(_spec(by_seed, _f_readout_sum), values, seeds)}
    if 0 in out["FACT"] and 1 in out["FACT"]:
        out["FACT"]["gamma_factorial"] = _difference(out["FACT"][1]["fusion"], out["FACT"][0]["fusion"], "gamma_factorial")
    else:
        out["FACT"]["gamma_factorial"] = {"state": NOT_ESTIMABLE, "value": None,
                                          "why": f"the 2 x 2 exists at r = {sorted(ent['FACT'])} only; the balanced factorial gamma needs it at both r"}
    if ent["DONOR"]:
        out["DONOR"] = {r: estimate(_spec(by_seed, _f_donor), values, seeds) for r, by_seed in ent["DONOR"].items()}
    return out


def effects(close_local: dict, design: dict, split: str = "validation", n_boot: int = 1000, seed: int = 0) -> dict:
    """Bound, contrast-wise effects with a replicate bootstrap that reuses the same estimator."""
    con = contrasts(design)
    bound = bind(close_local, design, split)
    values = bound["values"]
    seeds, levels, rs = con["replicates"], sorted(design["levels"]), sorted(design["r_values"])
    out = {"schema": SCHEMA, "split": split, "design_sha256": design["design_sha256"], "closure": close_local.get("closure"),
           "population": {"cells": len(design["cells"]), "verified_cells": sum(1 for v in bound["states"].values() if v == "VERIFIED"),
                          "not_verified": {k: v for k, v in bound["states"].items() if v != "VERIFIED"}},
           "replicates": seeds, "weights": con["weights"], "per_arch": {}, "adequacy": {}, "dx": {}, "bootstrap": {}, "definitions": {
               "gamma_common_pair": "d_1 - d_0 with d_r = mean over replicates of MASE(sequence) - MASE(summary), same donor kind, same pair at both r",
               "gamma_factorial": "fusion effect at r=1 minus at r=0, each the balanced 2 x 2 average; NOT_ESTIMABLE unless the 2 x 2 exists at both r",
               "readout": "last-step readout minus pooled readout, per fusion and balanced; descriptive at the r where it exists",
               "donor_delta": "(sequence_dsum - summary_dsum) - (sequence - summary) per replicate: the SAME common pair with the other donor",
               "H2": "profiles minus the mean of ALL random redistributions of the replicate; every control required"}}
    units = close_local.get("units") or {}
    for a in design["archs"]:
        out["per_arch"][a] = arch_effects(a, con["per_arch"][a], values, seeds, levels, rs)
        # adequacy apart: top level H2 profiles and DX, with every reference separately
        rows = []
        for cid in [con["per_arch"][a]["H2"][max(levels)][s]["profiles"] for s in seeds] + con["per_arch"][a]["DX"]:
            u = units.get(cid)
            if not u or u.get("status") != "VERIFIED":
                rows.append({"cell": cid, "state": (u or {}).get("status", "ABSENT")})
                continue
            r = u["record"]
            m, nv, lin, orc = r["mase"][split], r["naive_mase"][split], r["linear_mase"][split], r["oracle_mase"][split]
            rows.append({"cell": cid, "state": "VERIFIED", "model": m, "naive_persistence": nv, "linear": lin, "oracle": orc,
                         "mase_denominator_mean": float(np.mean(r.get("denominator") or [float("nan")])),
                         "seasonal_naive_score": "NOT_MEASURED (the seasonal naive is the MASE denominator on train, not a stored predictor)",
                         "beats_persistence": m < nv, "gap_to_linear": m - lin, "within_linear_plus_0_03": m <= lin + LINEAR_TOLERANCE,
                         "reaches_linear_point": m <= lin})
        out["adequacy"][a] = rows
    # bootstrap over replicates with the SAME estimator
    rng = np.random.default_rng(seed)
    boot = {a: {"H2_slope": [], "gamma_common_pair": [], "d1_common": [], "readout_r1_fusion_effect": []} for a in design["archs"]}
    for _ in range(n_boot):
        pick = [int(s) for s in rng.choice(seeds, size=len(seeds), replace=True)]
        for a in design["archs"]:
            ent = con["per_arch"][a]
            # resampled replicates: duplicate ids are allowed (each draw is a unit)
            vals = []
            for i, s in enumerate(pick):
                vals.append(s)
            e = {}
            for h in levels:
                spec = {i: ent["H2"][h][s] for i, s in enumerate(pick)}
                est = estimate(_spec(spec, _f_h2), values, list(range(len(pick))))
                e[h] = est["value"] if est["state"] == ESTIMATED else None
            if all(v is not None for v in e.values()) and len(levels) >= 2:
                boot[a]["H2_slope"].append(_slope(levels, [e[h] for h in levels]))
            d = {}
            for r in rs:
                est = estimate(_spec({i: ent["COMMON"][r][s] for i, s in enumerate(pick)}, _f_common), values, list(range(len(pick))))
                d[r] = est["value"] if est["state"] == ESTIMATED else None
            if d.get(1) is not None:
                boot[a]["d1_common"].append(d[1])
            if d.get(0) is not None and d.get(1) is not None:
                boot[a]["gamma_common_pair"].append(d[1] - d[0])
            if 1 in ent["FACT"]:
                est = estimate(_spec({i: ent["FACT"][1][s] for i, s in enumerate(pick)}, _f_readout), values, list(range(len(pick))))
                if est["state"] == ESTIMATED:
                    boot[a]["readout_r1_fusion_effect"].append(est["value"])
    out["bootstrap"] = {a: {k: ({"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n": len(v)} if v else None) for k, v in b.items()}
                        for a, b in boot.items()}
    out["bootstrap_note"] = (f"percentile intervals from resampling {len(seeds)} replicate ids with replacement, same estimator as the point; with "
                             f"{len(seeds)} replicates this is a description of two observations, not population precision, equivalence or power")
    return out


def tables(eff: dict, close_local: dict, design: dict) -> str:
    split = eff["split"]
    L = [f"# ARCH comparison — contrast-bound tables ({split}; design {eff['design_sha256'][:12]}; population {eff['population']['cells']} cells, "
         f"verified {eff['population']['verified_cells']}, closure {eff['closure']})", "", f"Replicates: {eff['replicates']} (two observations). {eff['bootstrap_note']}", ""]
    L += ["## Adequacy (H2 profiles at the top level and DX; every reference apart)", "",
          "| arch | cell | model | persistence naive | linear | gap to linear | ≤ linear + 0.03 | reaches linear point | oracle | MASE denom (train seasonal-naive MAE) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a, rows in eff["adequacy"].items():
        for x in rows:
            if x.get("state") != "VERIFIED":
                L.append(f"| {a} | {x['cell']} | {x['state']} | | | | | | | |")
            else:
                L.append(f"| {a} | {x['cell']} | {x['model']:.4f} | {x['naive_persistence']:.4f} | {x['linear']:.4f} | {x['gap_to_linear']:+.4f} | {x['within_linear_plus_0_03']} | "
                         f"{x['reaches_linear_point']} | {x['oracle']:.4f} | {x['mase_denominator_mean']:.4f} |")
    L += ["", "## Contrasts per architecture (state · n expected/observed · value · SD of replicates)", "",
          "| arch | H2 e(h) | H2 slope | d_0 common | d_1 common | gamma common pair | fusion 2×2 r=1 | readout 2×2 r=1 | interaction r=1 | gamma factorial | donor Δ r=1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(e):
        if e is None:
            return "n/a"
        if e.get("state") != ESTIMATED:
            return f"{e.get('state')} ({e.get('n_observed', 0)}/{e.get('n_expected', 0)})" if "n_expected" in e else str(e.get("state"))
        sd = e.get("sd_replicates")
        return f"{e['value']:+.4f} (sd {sd:.4f})" if sd is not None else f"{e['value']:+.4f}"
    for a, p in eff["per_arch"].items():
        e_h = ", ".join(f"h{h}: {cell(v)}" for h, v in p["H2"]["e"].items())
        fact1 = p["FACT"].get(1) or {}
        L.append(f"| {a} | {e_h} | {p['H2']['slope'] if p['H2']['slope'] is None else f'{p['H2']['slope']:+.4f}'} | {cell(p['COMMON']['d'].get(0))} | {cell(p['COMMON']['d'].get(1))} | "
                 f"{cell(p['COMMON']['gamma_common_pair'])} | {cell(fact1.get('fusion'))} | {cell(fact1.get('readout'))} | {cell(fact1.get('interaction'))} | "
                 f"{cell(p['FACT']['gamma_factorial'])} | {cell((p['DONOR'] or {}).get(1))} |")
    L += ["", "## Per-replicate values of the common pair and the readout"]
    for a, p in eff["per_arch"].items():
        L.append(f"- {a}: d_0 {p['COMMON']['d'].get(0, {}).get('per_replicate')}; d_1 {p['COMMON']['d'].get(1, {}).get('per_replicate')}; "
                 f"readout r=1 {(p['FACT'].get(1) or {}).get('readout', {}).get('per_replicate')}; missing: {p['COMMON']['d'].get(0, {}).get('missing')} / {p['COMMON']['d'].get(1, {}).get('missing')}")
    L += ["", "## Bootstrap (same estimator; descriptive)"]
    for a, b in eff["bootstrap"].items():
        L.append(f"- {a}: " + ", ".join(f"{k} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" if v else f"{k} —" for k, v in b.items()))
    L += ["", "## Cells (raw error, MASE, references, denominator, reach, updates, stop, cost)",
          "| cell | arch | arm | status | MAE val | MASE val | persistence | linear | oracle | MASE test | denom mean | reach | updates | stop | cpu s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, u in sorted((close_local.get("units") or {}).items()):
        if u.get("role") != "CELL":
            continue
        r = u.get("record") or {}
        if not r:
            L.append(f"| {k} | | | {u['status']} | | | | | | | | | | | |")
            continue
        L.append(f"| {k} | {r.get('arch')} | {r['arm']} | {u['status']} | {r['mae'][split]:.4f} | {r['mase'][split]:.4f} | {r['naive_mase'][split]:.4f} | "
                 f"{r['linear_mase'][split]:.4f} | {r['oracle_mase'][split]:.4f} | {r['mase'].get('test', float('nan')):.4f} | {np.mean(r['denominator']):.4f} | "
                 f"{r.get('support_reach')} | {r['updates']} | {r['stop_reason']} | {r['cost']['cpu_seconds']:.1f} |")
    return "\n".join(L) + "\n"


def merge_successor(close_local: dict, design: dict, succ_close: dict, succ_design: dict) -> tuple:
    """RP22: the stage plus its readout-completion successor as ONE population for the contrasts. Each closure
    is bound to its own design first (identity, population); the merged design enumerates the stage's cells
    plus the successor's, with the readout controls at every r; inherited donors are the stage's own cells.
    A successor of another parent, or a cell present in both, refuses."""
    if succ_design.get("successor_of") != design.get("design_sha256") or succ_design.get("kind") != "READOUT_COMPLETION":
        raise EffectsRefusal("REFUSED: the successor is not the readout completion of this design")
    if succ_close.get("design_sha256") != succ_design.get("design_sha256"):
        raise EffectsRefusal("REFUSED: the successor closure is of another design")
    ids = {c["cell_id"] for c in design["cells"]}
    dup = [c["cell_id"] for c in succ_design["cells"] if c["cell_id"] in ids]
    if dup:
        raise EffectsRefusal(f"REFUSED: successor cells already in the stage: {dup[:3]}")
    for inh in succ_design.get("inherited") or []:
        if inh["cell_id"] not in ids or inh["inherited_from"]["design_sha256"] != design["design_sha256"]:
            raise EffectsRefusal(f"REFUSED: inherited donor {inh['cell_id']} is not a stage cell")
    merged_design = {**design, "cells": list(design["cells"]) + list(succ_design["cells"]),
                     "readout_controls_r": sorted(set(design.get("readout_controls_r") or design["r_values"]) | {c["r"] for c in succ_design["cells"]}),
                     "design_sha256": "MERGED:" + design["design_sha256"][:16] + "+" + succ_design["design_sha256"][:16],
                     "merged_from": [design["design_sha256"], succ_design["design_sha256"]]}
    merged_design["cells_total"] = len(merged_design["cells"])
    units = dict(close_local["units"])
    for cid, u in succ_close["units"].items():
        if u.get("role") == "INHERITED":
            continue                                          # the stage's own verification of the donor stands; the successor re-verified it apart
        if cid in units:
            raise EffectsRefusal(f"REFUSED: {cid} verified in both closures")
        units[cid] = u
    merged_close = {"design_sha256": merged_design["design_sha256"], "population": {"members": [c["cell_id"] for c in merged_design["cells"]]},
                    "closure": f"{close_local.get('closure')}+{succ_close.get('closure')}", "units": units,
                    "sources": {"stage": {"design": design["design_sha256"], "closure": close_local.get("closure")},
                                "successor": {"design": succ_design["design_sha256"], "closure": succ_close.get("closure"),
                                              "inherited_verified": {k: u["status"] for k, u in succ_close["units"].items() if u.get("role") == "INHERITED"}}}}
    return merged_close, merged_design


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--close", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--successor-close", type=Path, default=None, help="RP22: the readout-completion successor's CLOSE.json")
    parser.add_argument("--successor-design", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=None)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)
    close = json.loads(args.close.read_text())
    local = close.get("local") or close
    design = json.loads(args.design.read_text())
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    try:
        if args.successor_close is not None:
            sc = json.loads(args.successor_close.read_text())
            local, design = merge_successor(local, design, sc.get("local") or sc, json.loads(args.successor_design.read_text()))
        eff = effects(local, design, args.split)
        if "sources" in local:
            eff["sources"] = local["sources"]
        eff["effects_test"] = None
        if args.split != "test":
            t = effects(local, design, "test", n_boot=0)
            eff["effects_test"] = {"per_arch": t["per_arch"], "adequacy": t["adequacy"]}
    except EffectsRefusal as e:
        print(json.dumps({"refused": e.why}))
        return 2
    args.out.write_text(json.dumps(eff, indent=1, default=str) + "\n")
    if args.tables:
        args.tables.write_text(tables(eff, local, design))
    print(json.dumps({a: {"gamma_common_pair": p["COMMON"]["gamma_common_pair"].get("value"), "d": {r: v.get("value") for r, v in p["COMMON"]["d"].items()},
                          "gamma_factorial": p["FACT"]["gamma_factorial"]["state"], "readout_r1": (p["FACT"].get(1) or {}).get("readout", {}).get("value"),
                          "donor_delta_r1": ((p["DONOR"] or {}).get(1) or {}).get("value"), "H2_slope": p["H2"]["slope"]}
                      for a, p in eff["per_arch"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
