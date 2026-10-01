#!/usr/bin/env python3
"""M04 DOIN batch-1 tables GENERATED from the exported queue and the retained verification receipts.

Inputs
  --queue     the exported campaign queue (QUEUE_batch1_v3.json from M04's evidence)
  --receipts  a directory of verification.json receipts named <cid[:16]>.json
Outputs (atomic) in --out-dir
  m04_r0_candidates.{md,csv}   every VERIFIED candidate: config label and key knobs, seed, validation
                               objective, verify verdict and exact_match, host role, cgroup peak, s/update
  m04_incumbents.{md,json}     incumbent history with per-seed aggregate and PER-HORIZON skill rows and
                               the comparability class
Refusals (CampaignRefusal): a verified candidate without its train or verify attempt, a missing receipt,
a receipt whose rescored objective or model digest disagrees with the queue, exact_match not True on a
candidate the queue calls verified, or an incumbent that is not the minimum mean over eligible configs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

COMPARABILITY = ("NOT_COMPARABLE", "ECL L24 -> H1..24, all 321 channels, z_train MAE on the validation split; "
                 "the published TimeFilter/ECL rows are L96 -> H96 on the test split, so no literature value applies")
KNOBS = ("branch.plugin", "branch.grouping_size", "branch.channels", "branch.kernel_size", "core.blocks",
         "core.d_model", "core.heads", "train.loss", "train.learning_rate")


class CampaignRefusal(Exception):
    pass


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def design_summary(flat):
    tf = [flat.get(f"core.time_factor_{i}") for i in range(3) if flat.get(f"core.time_factor_{i}") is not None]
    return (f"{flat.get('branch.plugin')} g{flat.get('branch.grouping_size')} ch{flat.get('branch.channels')} "
            f"k{flat.get('branch.kernel_size')}; steps {flat.get('model.branch_steps')}; core {flat.get('core.blocks')}x"
            f"d{flat.get('core.d_model')} h{flat.get('core.heads')} tf{tf}; {flat.get('train.loss')} lr{flat.get('train.learning_rate')}")


def check_seasonal(diag: dict, queue: dict, receipts: dict) -> dict:
    """A same-row seasonal-naive baseline from a horizon diagnostic, accepted only if it is provably on
    the same rows as the receipts: same population, every verified cid present, and each model's
    per-horizon receipt MAE in the diagnostic equal to the receipt's own."""
    pop = diag.get("population", {})
    vcids = {c["cid"] for c in queue["candidates"] if c["status"] == "verified"}
    dcids = {m["cid"] for m in diag.get("models", [])}
    persist0 = {str(b["horizon"]): b["naive_MAE"] for b in diag["baselines"]}
    for cid in sorted(vcids - dcids):
        # not in the diagnostic: accepted only if its own receipt proves the same rows (identical
        # same-row persistence MAE at every horizon and the same validation row count)
        rc = receipts.get(cid[:16])
        if rc is None:
            raise CampaignRefusal(f"SEASONAL_DIAGNOSTIC_MISSING_CIDS: {cid[:16]} (no receipt)")
        if int(rc.get("validation_rows") or -1) != int(pop.get("rows")) or any(
                abs(rc["per_horizon"][h]["baseline_MAE"] - v) > 1e-9 for h, v in persist0.items()):
            raise CampaignRefusal(f"SEASONAL_DIAGNOSTIC_MISSING_CIDS: {cid[:16]} not proven on the same rows")
    for m in diag["models"]:
        rc = receipts.get(m["cid"][:16])
        if rc is None:
            continue
        if int(rc.get("validation_rows") or pop.get("rows")) != int(pop.get("rows")):
            raise CampaignRefusal(f"SEASONAL_DIAGNOSTIC_POPULATION: {m['cid'][:16]}")
        for ph in m["per_horizon"]:
            mine = rc["per_horizon"][str(ph["horizon"])]["MAE"]
            if abs(mine - ph["receipt_MAE_gpu"]) > 1e-9:
                raise CampaignRefusal(f"SEASONAL_DIAGNOSTIC_NOT_SAME_ROWS: {m['cid'][:16]} h{ph['horizon']} "
                                      f"{mine} vs {ph['receipt_MAE_gpu']}")
    per_h = {str(b["horizon"]): b["seasonal_naive_MAE"] for b in diag["baselines"]}
    persist = {str(b["horizon"]): b["naive_MAE"] for b in diag["baselines"]}
    for rc in receipts.values():
        for h, v in persist.items():
            if abs(rc["per_horizon"][h]["baseline_MAE"] - v) > 1e-9:
                raise CampaignRefusal(f"SEASONAL_DIAGNOSTIC_PERSISTENCE_MISMATCH h{h}")
        break
    return {"per_horizon": per_h, "aggregate": sum(per_h.values()) / len(per_h), "period_hours": 24,
            "label": diag.get("label"), "population": pop}


def build(queue: dict, receipts: dict, comparability=None, seasonal=None) -> dict:
    att = {}
    for a in queue["attempts"]:
        att.setdefault((a["cid"], a["kind"]), []).append(a)
    rows = []
    for c in queue["candidates"]:
        if c["status"] != "verified":
            continue
        tr = [a for a in att.get((c["cid"], "train"), []) if a["status"] == "completed"]
        ve = [a for a in att.get((c["cid"], "verify"), []) if a["status"] == "completed"]
        if not tr or not ve:
            raise CampaignRefusal(f"MISSING_ATTEMPT: {c['cid'][:16]} is verified without a completed train and verify")
        tr, ve = tr[-1], ve[-1]
        rc = receipts.get(c["cid"][:16])
        if rc is None:
            raise CampaignRefusal(f"MISSING_RECEIPT: {c['cid'][:16]}")
        if rc.get("exact_match") is not True or ve.get("verdict") != "VERIFIED":
            raise CampaignRefusal(f"NOT_EXACT: {c['cid'][:16]} exact_match={rc.get('exact_match')} verdict={ve.get('verdict')}")
        if float(rc["objective"]["rescored_value"]) != float(c["objective"]):
            raise CampaignRefusal(f"OBJECTIVE_MISMATCH: {c['cid'][:16]} receipt {rc['objective']['rescored_value']} "
                                  f"queue {c['objective']}")
        if rc["digests"]["model_sha256"] != ve.get("model_sha256"):
            raise CampaignRefusal(f"MODEL_DIGEST_MISMATCH: {c['cid'][:16]}")
        flat = json.loads(c["flat"]) if isinstance(c["flat"], str) else c["flat"]
        rows.append({"cid": c["cid"][:16], "config_id": c["config_id"][:8], "label": c["label"], "seed": c["seed"],
                     "knobs": {k: flat.get(k) for k in KNOBS}, "design": design_summary(flat),
                     "objective": float(c["objective"]),
                     "seasonal_MAE": seasonal["aggregate"] if seasonal else None,
                     "skill_vs_seasonal": (1 - float(c["objective"]) / seasonal["aggregate"]) if seasonal else None,
                     "skill_MAE": rc["metrics"]["skill_MAE"], "verdict": ve["verdict"], "exact_match": True,
                     "train_host": tr.get("host"), "verify_host": ve.get("host"),
                     "cgroup_peak_bytes": tr.get("cgroup_peak_bytes"), "per_update_seconds": tr.get("per_update_seconds"),
                     "observed_updates": tr.get("observed_updates"), "selected_epoch": tr.get("selected_epoch"),
                     "stop_reason": tr.get("stop_reason")})
    # incumbent must be the minimum mean over eligible paired configs
    elig = [s for s in queue["standings"] if s.get("eligible") and s.get("mean_objective") is not None]
    best = min(elig, key=lambda s: s["mean_objective"])
    inc = queue["incumbent"]
    if inc["config_id"] != best["config_id"]:
        raise CampaignRefusal(f"INCUMBENT_NOT_MINIMUM: {inc['config_id'][:8]} vs {best['config_id'][:8]}")
    def per_seed_rows(cids):
        rows = []
        for cid in cids:
            rc = receipts.get(cid[:16])
            if rc is None:
                raise CampaignRefusal(f"MISSING_RECEIPT: {cid[:16]}")
            if rc.get("exact_match") is not True:
                raise CampaignRefusal(f"NOT_EXACT: {cid[:16]}")
            ph = rc["per_horizon"]
            row = {"cid": cid[:16], "MAE": rc["metrics"]["MAE"], "baseline_MAE": rc["metrics"]["baseline_MAE"],
                   "skill_MAE": rc["metrics"]["skill_MAE"],
                   "per_horizon_skill_MAE": {k: ph[k]["skill_MAE"] for k in sorted(ph, key=int)}}
            if seasonal:
                row["seasonal_MAE"] = seasonal["aggregate"]
                row["skill_vs_seasonal"] = 1 - rc["metrics"]["MAE"] / seasonal["aggregate"]
                row["per_horizon_skill_vs_seasonal"] = {k: 1 - ph[k]["MAE"] / seasonal["per_horizon"][k]
                                                        for k in sorted(ph, key=int)}
            rows.append(row)
        return rows
    pairs = []
    for s_ in queue["standings"]:
        if not s_.get("eligible"):
            continue
        cids = [c["cid"] for c in queue["candidates"] if c["config_id"] == s_["config_id"] and c["status"] == "verified"]
        ps = per_seed_rows(cids)
        objs = [r["MAE"] for r in ps]
        pairs.append({"config_id": s_["config_id"][:8], "label": s_["label"], "mean_objective": s_["mean_objective"],
                      "two_seed_spread": (max(objs) - min(objs)) if len(objs) > 1 else None, "per_seed": ps})
    hist = []
    for h in queue["incumbent_history"]:
        cids = json.loads(h["cids"]) if isinstance(h["cids"], str) else h["cids"]
        per_seed = per_seed_rows(cids)
        label = next((s["label"] for s in queue["standings"] if s["config_id"] == h["config_id"]), None)
        seeds = json.loads(h["seeds"]) if isinstance(h["seeds"], str) else h["seeds"]
        hist.append({"seq": h["seq"], "config_id": h["config_id"][:8], "label": label,
                     "mean_objective": h["mean_objective"], "seeds": seeds, "reason": h["reason"],
                     "per_seed": per_seed})
    return {"schema": "m06.m04_campaign_tables.v1", "campaign": queue["campaign"],
            "campaign_sha256": queue["meta"]["campaign_sha256"], "objective": queue["objective"],
            "counts": queue["counts"], "candidates": rows, "incumbents": hist,
            "comparability": comparability or {"class": COMPARABILITY[0], "reason": COMPARABILITY[1]},
            "pairs": sorted(pairs, key=lambda p: p["mean_objective"]),
            "seasonal_naive": seasonal,
            "evidence": "measured validation objective, fresh-process checkpoint rescoring exact_match; NOT test, NOT a published comparison"}


SEASONAL_NOTE = ("Seasonal naive = the value 24 h before the target on the SAME validation rows (period 24 h). On this "
                 "daily-periodic dataset the aggregate skill against last-value persistence OVERSTATES usefulness: "
                 "the seasonal naive is the stronger, relevant baseline, and skill against it is shown separately.")


def _seed_rows(per_seed, hs, seasonal):
    out = []
    for s in per_seed:
        out.append(f"| {s['cid']} | vs persistence | {s['MAE']:.6f} | {s['baseline_MAE']:.6f} | {s['skill_MAE']:+.4f} | "
                   + " | ".join(f"{s['per_horizon_skill_MAE'][k]:+.2f}" for k in hs) + " |")
        if seasonal:
            out.append(f"| {s['cid']} | vs seasonal 24 h | {s['MAE']:.6f} | {s['seasonal_MAE']:.6f} | {s['skill_vs_seasonal']:+.4f} | "
                       + " | ".join(f"{s['per_horizon_skill_vs_seasonal'][k]:+.2f}" for k in hs) + " |")
    return out


def render(t: dict, annotations: dict | None = None, legacy: bool = False, seasonal_na: str | None = None):
    o = t["objective"]
    seasonal = t.get("seasonal_naive")
    c = [("# M04 batch-1 v3: verified R0 candidates (generated)" if legacy else
          f"# {t['campaign']}: verified R0 candidates (generated)"), "",
         f"Campaign `{t['campaign']}`, CAMPAIGN sha `{t['campaign_sha256'][:8]}…`. Objective: {o['metric']} on "
         f"{o['split']}, {o['unit']} (lower is better). Counts: {t['counts']}. Comparability: **{t['comparability']['class']}**: "
         f"{t['comparability']['reason']}.", "",
         ]
    if legacy:
        c += ["| label | cfg | seed | objective | skill MAE | verify | exact | train host | peak GiB | s/update | updates | epoch | stop |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        c += ["| label | cfg | design | seed | objective | persistence skill MAE | seasonal-naive MAE (24 h) | skill vs seasonal | "
              "verify | exact | train host | peak GiB | s/update | updates | epoch | stop |",
              "|---" * 16 + "|"]
    for r in sorted(t["candidates"], key=lambda r: (r["label"], r["seed"])):
        tail = (f"{r['verdict']} | {r['exact_match']} | {r['train_host']} | {(r['cgroup_peak_bytes'] or 0) / 2**30:.2f} | "
                f"{r['per_update_seconds']:.4f} | {r['observed_updates']} | {r['selected_epoch']} | {r['stop_reason']} |")
        if legacy:
            c.append(f"| {r['label']} | {r['config_id']} | {r['seed']} | {r['objective']:.6f} | {r['skill_MAE']:.4f} | " + tail)
        else:
            sm = f"{r['seasonal_MAE']:.6f}" if r.get("seasonal_MAE") is not None else "NOT_AVAILABLE"
            ss = f"{r['skill_vs_seasonal']:+.4f}" if r.get("skill_vs_seasonal") is not None else "NOT_AVAILABLE"
            c.append(f"| {r['label']} | {r['config_id']} | {r.get('design', '')} | {r['seed']} | {r['objective']:.6f} | "
                     f"{r['skill_MAE']:.4f} | {sm} | {ss} | " + tail)
    if seasonal:
        c += ["", SEASONAL_NOTE]
    elif seasonal_na:
        c += ["", f"Seasonal naive (24 h, same rows): **NOT_AVAILABLE**: {seasonal_na}"]
    i = [("# M04 batch-1 v3: incumbent history with per-horizon skill (generated)" if legacy else
          f"# {t['campaign']}: incumbent history and per-horizon skill (generated)"), "",
         f"Comparability: **{t['comparability']['class']}**: {t['comparability']['reason']}.", ""]
    if seasonal:
        i += [SEASONAL_NOTE, ""]
    elif seasonal_na:
        i += [f"Seasonal naive (24 h, same rows): **NOT_AVAILABLE**: {seasonal_na}", ""]
    for h in t["incumbents"]:
        i += [f"## Incumbent {h['seq']}: `{h['config_id']}` {h['label']}, mean validation {t['objective']['metric']} "
              f"{h['mean_objective']:.6f} over seeds {h['seeds']}", f"Reason: {h['reason']}.", ""]
        note = (annotations or {}).get(h["config_id"])
        if note:
            lead = "Architecture class" if legacy else "Annotation"
            i += [f"{lead} (from {note['source']}): **{note['class']}**. {note['text']}", ""]
        hs = list(h["per_seed"][0]["per_horizon_skill_MAE"])
        if legacy:
            i.append("| seed cid | MAE | persistence MAE | skill MAE | " + " | ".join(f"h{k}" for k in hs) + " |")
            i.append("|---" * (4 + len(hs)) + "|")
            for s in h["per_seed"]:
                i.append(f"| {s['cid']} | {s['MAE']:.6f} | {s['baseline_MAE']:.6f} | {s['skill_MAE']:.4f} | "
                         + " | ".join(f"{s['per_horizon_skill_MAE'][k]:+.2f}" for k in hs) + " |")
        else:
            i.append("| seed cid | baseline | MAE | baseline MAE (same rows) | skill | " + " | ".join(f"h{k}" for k in hs) + " |")
            i.append("|---" * (5 + len(hs)) + "|")
            i += _seed_rows(h["per_seed"], hs, seasonal)
        neg = sorted({int(k) for s in h["per_seed"] for k, v in s["per_horizon_skill_MAE"].items() if v < 0})
        i += ["", f"Horizons with NEGATIVE skill (persistence wins) in any seed: {['h%d' % k for k in neg] or 'none'}.", ""]
    if t.get("pairs") and not legacy:
        i += ["## Every verified pair, ranked by mean validation objective", ""]
        for p in t["pairs"]:
            hs = list(p["per_seed"][0]["per_horizon_skill_MAE"])
            sp = f"{p['two_seed_spread']:.6f}" if p["two_seed_spread"] is not None else "n/a"
            i += [f"### `{p['config_id']}` {p['label']}: mean {p['mean_objective']:.6f}, two-seed spread {sp}", "",
                  "| seed cid | baseline | MAE | baseline MAE (same rows) | skill | " + " | ".join(f"h{k}" for k in hs) + " |",
                  "|---" * (5 + len(hs)) + "|"]
            i += _seed_rows(p["per_seed"], hs, seasonal)
            neg = sorted({int(k) for s_ in p["per_seed"] for k, v in s_["per_horizon_skill_MAE"].items() if v < 0})
            i += ["", f"Negative skill vs persistence in any seed: {['h%d' % k for k in neg] or 'none'}."]
            if seasonal:
                negs = sorted({int(k) for s_ in p["per_seed"] for k, v in s_["per_horizon_skill_vs_seasonal"].items() if v < 0})
                i += [f"Negative skill vs seasonal naive in any seed: {['h%d' % k for k in negs] or 'none'}."]
            i += [""]
    return "\n".join(c) + "\n", "\n".join(i) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--annotations", default=None, help="JSON {config_id8: {class, source, text}}")
    ap.add_argument("--prefix", default="m04", help="output file prefix (keeps campaigns' tables apart)")
    ap.add_argument("--seasonal-diagnostic", default=None, help="horizon diagnostic JSON with same-row seasonal naive")
    ap.add_argument("--seasonal-not-available", default=None, help="reason text when no same-row seasonal values exist")
    ap.add_argument("--legacy-layout", action="store_true", help="the OLD tables' original layout")
    ap.add_argument("--comparability-class", default=None)
    ap.add_argument("--comparability-reason", default=None)
    a = ap.parse_args(argv)
    q = json.load(open(a.queue))
    rec = {f[:-5]: json.load(open(os.path.join(a.receipts, f))) for f in os.listdir(a.receipts) if f.endswith(".json")}
    try:
        comp = ({"class": a.comparability_class, "reason": a.comparability_reason}
                if a.comparability_class else None)
        seas = check_seasonal(json.load(open(a.seasonal_diagnostic)), q, rec) if a.seasonal_diagnostic else None
        t = build(q, rec, comp, seas)
    except CampaignRefusal as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return 3
    t["inputs"] = {"queue_sha256": sha_file(a.queue),
                   "receipts_sha256": {f: sha_file(os.path.join(a.receipts, f)) for f in sorted(os.listdir(a.receipts))}}
    os.makedirs(a.out_dir, exist_ok=True)
    ann = json.load(open(a.annotations)) if a.annotations else None
    if ann:
        t["annotations"] = ann
    cmd, imd = render(t, ann, legacy=a.legacy_layout, seasonal_na=a.seasonal_not_available)
    P = a.prefix
    outs = {f"{P}_r0_candidates.md": cmd, f"{P}_incumbents.md": imd, f"{P}_campaign_tables.json": json.dumps(t, indent=1)}
    for name, text in outs.items():
        p = os.path.join(a.out_dir, name)
        open(p + ".tmp", "w").write(text)
        os.replace(p + ".tmp", p)
    with open(os.path.join(a.out_dir, f"{P}_r0_candidates.csv.tmp"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["label", "config_id", "design", "seed", "objective", "skill_MAE", "seasonal_MAE", "skill_vs_seasonal", "verdict", "exact_match", "train_host",
                "verify_host", "cgroup_peak_bytes", "per_update_seconds", "observed_updates", "selected_epoch", "stop_reason",
                *KNOBS, "cid"]
        w.writerow(cols)
        for r in t["candidates"]:
            w.writerow([r[k] if k in r else r["knobs"].get(k) for k in cols])
    os.replace(os.path.join(a.out_dir, f"{P}_r0_candidates.csv.tmp"), os.path.join(a.out_dir, f"{P}_r0_candidates.csv"))
    print(len(t["candidates"]), "candidates;", len(t["incumbents"]), "incumbents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
