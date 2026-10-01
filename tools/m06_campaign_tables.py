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


def build(queue: dict, receipts: dict, comparability=None) -> dict:
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
            rows.append({"cid": cid[:16], "MAE": rc["metrics"]["MAE"], "baseline_MAE": rc["metrics"]["baseline_MAE"],
                         "skill_MAE": rc["metrics"]["skill_MAE"],
                         "per_horizon_skill_MAE": {k: ph[k]["skill_MAE"] for k in sorted(ph, key=int)}})
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
        per_seed = []
        cids = json.loads(h["cids"]) if isinstance(h["cids"], str) else h["cids"]
        for cid in cids:
            rc = receipts.get(cid[:16])
            if rc is None:
                raise CampaignRefusal(f"MISSING_RECEIPT: incumbent {cid[:16]}")
            ph = rc["per_horizon"]
            per_seed.append({"cid": cid[:16], "MAE": rc["metrics"]["MAE"], "baseline_MAE": rc["metrics"]["baseline_MAE"],
                             "skill_MAE": rc["metrics"]["skill_MAE"],
                             "per_horizon_skill_MAE": {k: ph[k]["skill_MAE"] for k in sorted(ph, key=int)}})
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
            "evidence": "measured validation objective, fresh-process checkpoint rescoring exact_match; NOT test, NOT a published comparison"}


def render(t: dict, annotations: dict | None = None):
    o = t["objective"]
    c = [f"# {t['campaign']}: verified R0 candidates (generated)", "",
         f"Campaign `{t['campaign']}`, CAMPAIGN sha `{t['campaign_sha256'][:8]}…`. Objective: {o['metric']} on "
         f"{o['split']}, {o['unit']} (lower is better). Counts: {t['counts']}. Comparability: **{t['comparability']['class']}**: "
         f"{t['comparability']['reason']}.", "",
         "| label | cfg | design | seed | objective | skill MAE | verify | exact | train host | peak GiB | s/update | updates | epoch | stop |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(t["candidates"], key=lambda r: (r["label"], r["seed"])):
        c.append(f"| {r['label']} | {r['config_id']} | {r.get('design', '')} | {r['seed']} | {r['objective']:.6f} | {r['skill_MAE']:.4f} | {r['verdict']} | "
                 f"{r['exact_match']} | {r['train_host']} | {(r['cgroup_peak_bytes'] or 0) / 2**30:.2f} | "
                 f"{r['per_update_seconds']:.4f} | {r['observed_updates']} | {r['selected_epoch']} | {r['stop_reason']} |")
    i = [f"# {t['campaign']}: incumbent history and per-horizon skill (generated)", "",
         f"Comparability: **{t['comparability']['class']}**: {t['comparability']['reason']}.", ""]
    for h in t["incumbents"]:
        i += [f"## Incumbent {h['seq']}: `{h['config_id']}` {h['label']}, mean validation {t['objective']['metric']} "
              f"{h['mean_objective']:.6f} over seeds {h['seeds']}", f"Reason: {h['reason']}.", ""]
        note = (annotations or {}).get(h["config_id"])
        if note:
            i += [f"Architecture class (from {note['source']}): **{note['class']}**. {note['text']}", ""]
        hs = list(h["per_seed"][0]["per_horizon_skill_MAE"])
        i.append("| seed cid | MAE | persistence MAE | skill MAE | " + " | ".join(f"h{k}" for k in hs) + " |")
        i.append("|---" * (4 + len(hs)) + "|")
        for s in h["per_seed"]:
            i.append(f"| {s['cid']} | {s['MAE']:.6f} | {s['baseline_MAE']:.6f} | {s['skill_MAE']:.4f} | "
                     + " | ".join(f"{s['per_horizon_skill_MAE'][k]:+.2f}" for k in hs) + " |")
        neg = sorted({int(k) for s in h["per_seed"] for k, v in s["per_horizon_skill_MAE"].items() if v < 0})
        i += ["", f"Horizons with NEGATIVE skill (persistence wins) in any seed: {['h%d' % k for k in neg] or 'none'}.", ""]
    if t.get("pairs"):
        i += ["## Every verified pair, ranked by mean validation objective", ""]
        for p in t["pairs"]:
            hs = list(p["per_seed"][0]["per_horizon_skill_MAE"])
            sp = f"{p['two_seed_spread']:.6f}" if p["two_seed_spread"] is not None else "n/a"
            i += [f"### `{p['config_id']}` {p['label']}: mean {p['mean_objective']:.6f}, two-seed spread {sp}", "",
                  "| seed cid | MAE | persistence MAE (same rows) | skill MAE | " + " | ".join(f"h{k}" for k in hs) + " |",
                  "|---" * (4 + len(hs)) + "|"]
            for s_ in p["per_seed"]:
                i.append(f"| {s_['cid']} | {s_['MAE']:.6f} | {s_['baseline_MAE']:.6f} | {s_['skill_MAE']:.4f} | "
                         + " | ".join(f"{s_['per_horizon_skill_MAE'][k]:+.2f}" for k in hs) + " |")
            neg = sorted({int(k) for s_ in p["per_seed"] for k, v in s_["per_horizon_skill_MAE"].items() if v < 0})
            i += ["", f"Negative-skill horizons in any seed: {['h%d' % k for k in neg] or 'none'}.", ""]
    return "\n".join(c) + "\n", "\n".join(i) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--annotations", default=None, help="JSON {config_id8: {class, source, text}}")
    ap.add_argument("--prefix", default="m04", help="output file prefix (keeps campaigns' tables apart)")
    ap.add_argument("--comparability-class", default=None)
    ap.add_argument("--comparability-reason", default=None)
    a = ap.parse_args(argv)
    q = json.load(open(a.queue))
    rec = {f[:-5]: json.load(open(os.path.join(a.receipts, f))) for f in os.listdir(a.receipts) if f.endswith(".json")}
    try:
        comp = ({"class": a.comparability_class, "reason": a.comparability_reason}
                if a.comparability_class else None)
        t = build(q, rec, comp)
    except CampaignRefusal as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return 3
    t["inputs"] = {"queue_sha256": sha_file(a.queue),
                   "receipts_sha256": {f: sha_file(os.path.join(a.receipts, f)) for f in sorted(os.listdir(a.receipts))}}
    os.makedirs(a.out_dir, exist_ok=True)
    ann = json.load(open(a.annotations)) if a.annotations else None
    if ann:
        t["annotations"] = ann
    cmd, imd = render(t, ann)
    P = a.prefix
    outs = {f"{P}_r0_candidates.md": cmd, f"{P}_incumbents.md": imd, f"{P}_campaign_tables.json": json.dumps(t, indent=1)}
    for name, text in outs.items():
        p = os.path.join(a.out_dir, name)
        open(p + ".tmp", "w").write(text)
        os.replace(p + ".tmp", p)
    with open(os.path.join(a.out_dir, f"{P}_r0_candidates.csv.tmp"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["label", "config_id", "design", "seed", "objective", "skill_MAE", "verdict", "exact_match", "train_host",
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
