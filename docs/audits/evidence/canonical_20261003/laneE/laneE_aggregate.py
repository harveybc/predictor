#!/usr/bin/env python3
"""Lane E (PS3-R) extractibility aggregator: per-feature results.jsonl -> small tables.

Reads <root>/<batch>/<feature>/{results.jsonl,run_manifest.json,FAILED.json} for the ordered
feature queue in <queue.json> and writes:
  extractibility_matrix.csv  feature x family x metric, status MEASURED/FAILED/NOT_APPLICABLE/PENDING
  feature_cost.csv           per feature: wall, cgroup peak, updates, stop reasons
  probe_summary.csv          per feature x family x target group: skill vs same-row naive, deltas
Reconstruction is diagnostic only. Nothing here selects or drops a feature.
"""
import csv, json, os, statistics as st, sys
from collections import defaultdict

FAMS = ["identity", "random", "ae", "dae"]
TRAINED = ["ae", "dae"]
REC = ["mae_rel_train_constant", "mse_rel_train_constant", "mae_norm", "mae_orig", "mse_orig",
       "acf_l1", "log_psd_l1", "mae_extremes", "dtw_mean"]
GROUPS = ["Y_s", "Y_l", "Y_b"]


def mean(v):
    v = [x for x in v if x is not None]
    return st.fmean(v) if v else None


def load(root, batch, feat):
    d = os.path.join(root, batch, feat)
    man = os.path.join(d, "run_manifest.json")
    fail = os.path.join(d, "FAILED.json")
    if os.path.isfile(man):
        rows = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"))]
        return "DONE", json.load(open(man)), rows
    if os.path.isfile(fail):
        return "FAILED", json.load(open(fail)), []
    return "PENDING", None, []


def main(root, queue_path, out_dir):
    queue = json.load(open(queue_path))["queue"]
    os.makedirs(out_dir, exist_ok=True)
    mat, cost, probe = [], [], []
    for q in queue:
        b, f, stage = q["batch"], q["feature"], q["stage"]
        state, man, rows = load(root, b, f)
        key = dict(batch=b, feature=f, stage=stage)
        if state != "DONE":
            reason = (man or {}).get("reason", "") if state == "FAILED" else ""
            for fam in FAMS:
                mat.append(dict(key, family=fam, metric="*", status=state, value="", n_folds="", note=reason))
            cost.append(dict(key, status=state, note=reason))
            continue
        ff = [r for r in rows if r["kind"] == "fold_family"]
        summ = next((r for r in rows if r["kind"] == "feature_summary"), {})
        for fam in FAMS:
            fr = [r for r in ff if r["family"] == fam]
            recs = [r["reconstruction"] for r in fr]
            rs = {r.get("status") for r in recs}
            for m in REC:
                if rs == {"NOT_APPLICABLE"}:
                    mat.append(dict(key, family=fam, metric="rec_" + m, status="NOT_APPLICABLE", value="",
                                    n_folds=len(fr), note="no decoder"))
                else:
                    v = [r.get(m) for r in recs if r.get("status") == "MEASURED"]
                    stt = "MEASURED" if v and all(x is not None for x in v) and len(v) == len(fr) else \
                          ("FAILED" if "FAILED" in rs else "MEASURED")
                    mat.append(dict(key, family=fam, metric="rec_" + m, status=stt,
                                    value=mean(v), n_folds=len(v), note=""))
            ed = [r["effective_dimension"]["participation_ratio"] for r in fr]
            mat.append(dict(key, family=fam, metric="effective_dim_pr", status="MEASURED", value=mean(ed),
                            n_folds=len(ed), note=""))
            stab = (summ.get("stability") or {}).get(fam, {})
            mat.append(dict(key, family=fam, metric="stability_cka_mean",
                            status=stab.get("status", "FAILED"), value=stab.get("mean", ""),
                            n_folds=len(fr), note=stab.get("reason", "")))
            if fam in TRAINED:
                mat.append(dict(key, family=fam, metric="fit_wall_seconds", status="MEASURED",
                                value=sum(r["cost"]["fit_wall_seconds"] for r in fr), n_folds=len(fr), note=""))
                mat.append(dict(key, family=fam, metric="updates", status="MEASURED",
                                value=sum(r["cost"]["updates"] for r in fr), n_folds=len(fr),
                                note=";".join(sorted({r["fit_report"]["stop_reason"] for r in fr}))))
        # probes
        pr = [r for r in rows if r["kind"] == "probe"]
        pd = [r for r in rows if r["kind"] == "probe_delta"]
        for g in GROUPS:
            for fam in FAMS:
                rr = [r for r in pr if r["target"] == g and r["representation"] == fam]
                ok = [r for r in rr if r["status"] == "MEASURED"]
                if not rr:
                    mat.append(dict(key, family=fam, metric=f"probe_skill_{g}", status="NOT_APPLICABLE",
                                    value="", n_folds=0, note="no target in batch"))
                    continue
                if ok and "mae" in ok[0]:  # emitted rows carry kind="probe"; regression rows have mae
                    skill = mean([r["skill_vs_train_mean"] for r in ok]); naive = "train_mean_same_rows"
                    loss = mean([r["mae"] for r in ok]); nl = mean([r["naive_train_mean_mae"] for r in ok])
                    sz = mean([r["skill_vs_zero"] for r in ok])
                else:
                    skill = mean([r["skill_log_loss_vs_prior"] for r in ok]); naive = "train_prior_same_rows"
                    loss = mean([r["log_loss"] for r in ok]); nl = mean([r["prior_log_loss"] for r in ok]); sz = None
                stt = "MEASURED" if len(ok) == len(rr) else ("FAILED" if not ok else "MEASURED")
                mat.append(dict(key, family=fam, metric=f"probe_skill_{g}", status=stt, value=skill,
                                n_folds=len({r["fold_id"] for r in ok}),
                                note=f"{len(ok)}/{len(rr)} fold-horizon cells; naive={naive}"))
                row = dict(key, family=fam, target=g, n_cells=len(ok), loss=loss, naive_loss=nl,
                           skill_vs_naive=skill, skill_vs_zero=sz, naive=naive)
                if fam in TRAINED:
                    dd = [d for d in pd if d["target"] == g and d["trained"] == fam]
                    dr = [d.get("delta_probe_random_minus_trained") for d in dd]
                    pv = [d.get("preservation_raw_minus_trained") for d in dd]
                    row.update(delta_probe_mean=mean(dr), preservation_mean=mean(pv),
                               frac_beats_random=mean([1.0 if (x or 0) > 0 else 0.0 for x in dr]) if dd else None,
                               frac_beats_raw=mean([1.0 if (x or 0) > 0 else 0.0 for x in pv]) if dd else None,
                               frac_beats_both=mean([1.0 if (a or 0) > 0 and (b or 0) > 0 else 0.0
                                                     for a, b in zip(dr, pv)]) if dd else None)
                    for mname, v in (("delta_probe", row["delta_probe_mean"]),
                                     ("preservation", row["preservation_mean"])):
                        mat.append(dict(key, family=fam, metric=f"{mname}_{g}",
                                        status="MEASURED" if v is not None else "FAILED", value=v,
                                        n_folds=len({d["fold_id"] for d in dd}),
                                        note="positive favours trained"))
                probe.append(row)
        stops = sorted({r["fit_report"]["stop_reason"] for r in ff if r["family"] in TRAINED})
        cost.append(dict(key, status="DONE", wall_seconds=man["wall_seconds"],
                         cgroup_peak_bytes=man.get("cgroup_peak_bytes"), peak_rss_bytes=man.get("peak_rss_bytes"),
                         updates=sum(r["cost"]["updates"] for r in ff),
                         epochs=sum(r["cost"]["epochs_run"] for r in ff), stop_reasons=";".join(stops),
                         code_commit=man.get("code_commit"), series_sha256=man.get("series_sha256")))
    def w(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(out_dir, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, keys); wr.writeheader()
            for r in rows:
                wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    w("extractibility_matrix.csv", mat); w("feature_cost.csv", cost); w("probe_summary.csv", probe)
    n = len(queue); done = sum(1 for c in cost if c["status"] == "DONE")
    print(json.dumps({"features_total": n, "done": done,
                      "failed": sum(1 for c in cost if c["status"] == "FAILED")}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
