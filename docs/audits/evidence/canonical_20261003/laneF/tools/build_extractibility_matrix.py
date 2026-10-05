"""Lane F (PS3-R) extractibility matrix builder.

Reads ut_pilot_run.v1 outputs (results.jsonl + run_manifest.json) for lane F and lane E,
one directory per feature, and writes small CSV tables comparable cell by cell:
  cell = (batch, feature, target, horizon_index, family).
Per cell, folds are aggregated: mean/min over folds of skill (trained, identity, random, same
row naive), Delta_probe = L(random) - L(trained), preservation = L(identity) - L(trained),
number of folds where each is > 0. Nothing here selects a winner or drops a feature.
Paths are given relative to a root; no host names are written.
usage: build_extractibility_matrix.py ROOT_F ROOT_E OUT_DIR batch [batch ...]
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np

TRAINED_F = ("masked_temporal_ae", "past_to_current_siamese")
TRAINED_E = ("ae", "dae")


def load_feature(d):
    rp, mp = os.path.join(d, "results.jsonl"), os.path.join(d, "run_manifest.json")
    if not os.path.isfile(mp):
        return None, None
    man = json.load(open(mp))
    rows = [json.loads(l) for l in open(rp)]
    return man, rows


def runs(root, batch):
    """Yield (feature, man, rows) for every completed run under root/batch/<feature>[/<pass>]."""
    b = os.path.join(root, batch)
    if not os.path.isdir(b):
        return
    for feat in sorted(os.listdir(b)):
        fd = os.path.join(b, feat)
        cands = [fd] + [os.path.join(fd, p) for p in sorted(os.listdir(fd)) if p.startswith("pass_")] \
            if os.path.isdir(fd) else []
        for d in cands:
            man, rows = load_feature(d)
            if man is not None:
                yield feat, man, rows


def cells(lane, batch, feat, man, rows, trained_set):
    out = []
    probes = defaultdict(dict)  # (fold,target,h) -> rep -> row
    for r in rows:
        if r["kind"] == "probe" and r.get("status") == "MEASURED":
            probes[(r["fold_id"], r["target"], r["horizon_index"])][r["representation"]] = r
    ff = {(r["fold_id"], r["family"]): r for r in rows if r["kind"] == "fold_family"}
    summ = [r for r in rows if r["kind"] == "feature_summary"]
    stab = summ[-1]["stability"] if summ else {}
    fams = [f for f in man["families"] if f in trained_set]
    keys = sorted({(t, h) for (_, t, h) in probes})
    for fam in fams:
        frs = [v for (fo, f), v in ff.items() if f == fam]
        cost_s = sum(v["cost"]["fit_wall_seconds"] for v in frs)
        upd = sum(v["cost"]["updates"] for v in frs)
        epochs = [v["cost"]["epochs_run"] for v in frs]
        pr = [v["effective_dimension"]["participation_ratio"] for v in frs]
        rec = [v["reconstruction"] for v in frs]
        rec_status = sorted({x["status"] for x in rec})
        st = stab.get(fam, {})
        for (t, h) in keys:
            per = []
            for (fo, t2, h2), reps in probes.items():
                if (t2, h2) != (t, h) or fam not in reps:
                    continue
                tr, idn, rnd = reps[fam], reps.get("identity"), reps.get("random")
                if tr.get("probe") == "ridge":  # results rows overwrite "kind" with "probe"
                    L = lambda x: x["mae"]; S = lambda x: x["skill_vs_zero"]
                    naive = tr["naive_zero_mae"]
                else:
                    L = lambda x: x["log_loss"]; S = lambda x: x["skill_log_loss_vs_prior"]
                    naive = tr["prior_log_loss"]
                per.append(dict(fold=fo, kind="regression" if tr.get("probe") == "ridge" else "classification", n_val=tr["n_val"], naive=naive,
                                L_tr=L(tr), S_tr=S(tr),
                                L_id=L(idn) if idn else np.nan, S_id=S(idn) if idn else np.nan,
                                L_rnd=L(rnd) if rnd else np.nan, S_rnd=S(rnd) if rnd else np.nan))
            if not per:
                continue
            a = lambda k: np.array([p[k] for p in per], float)
            dp, pv = a("L_rnd") - a("L_tr"), a("L_id") - a("L_tr")
            out.append({
                "lane": lane, "batch": batch, "feature_id": feat, "family": fam,
                "target": t, "horizon_index": h, "loss": "mae" if per[0]["kind"] == "regression" else "log_loss",
                "naive": "zero" if per[0]["kind"] == "regression" else "train_prior",
                "n_folds": len(per), "n_val_total": int(a("n_val").sum()),
                "naive_loss_mean": a("naive").mean(),
                "skill_trained_mean": a("S_tr").mean(), "skill_trained_min": a("S_tr").min(),
                "skill_identity_mean": np.nanmean(a("S_id")), "skill_random_mean": np.nanmean(a("S_rnd")),
                "delta_probe_mean": np.nanmean(dp), "delta_probe_folds_pos": int(np.sum(dp > 0)),
                "preservation_mean": np.nanmean(pv), "preservation_folds_pos": int(np.sum(pv > 0)),
                "stability_cka_mean": st.get("mean", st.get("mean_cka")) if st.get("status") == "MEASURED" else None,
                "eff_dim_pr_mean": float(np.mean(pr)) if pr else None,
                "reconstruction": "/".join(rec_status),
                "epochs_run_mean": float(np.mean(epochs)) if epochs else None,
                "fit_seconds_total": cost_s, "updates_total": upd,
                "run_wall_seconds": man.get("wall_seconds"), "cgroup_peak_bytes": man.get("cgroup_peak_bytes"),
                "code_commit": (man.get("code_commit") or "")[:7],
                "series_sha256": man["series_sha256"][:12],
            })
    return out


def main():
    root_f, root_e, out_dir, batches = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    allc = []
    for b in batches:
        for feat, man, rows in runs(root_f, b):
            allc += cells("F", b, feat, man, rows, TRAINED_F)
        for feat, man, rows in runs(root_e, b):
            allc += cells("E", b, feat, man, rows, TRAINED_E)
    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else v
    cols = list(allc[0].keys()) if allc else []
    with open(os.path.join(out_dir, "extractibility_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for c in allc:
            w.writerow({k: fmt(v) for k, v in c.items()})
    # side-by-side on identical (batch, feature, target, horizon): F families vs E families
    idx = defaultdict(dict)
    for c in allc:
        idx[(c["batch"], c["feature_id"], c["target"], c["horizon_index"])][c["family"]] = c
    fams = list(TRAINED_F) + list(TRAINED_E)
    side = []
    for k in sorted(idx):
        d = idx[k]
        if not any(f in d for f in TRAINED_F):
            continue
        row = {"batch": k[0], "feature_id": k[1], "target": k[2], "horizon_index": k[3]}
        any_c = next(iter(d.values()))
        row["identity_skill_mean"] = any_c["skill_identity_mean"]
        row["random_skill_mean"] = any_c["skill_random_mean"]
        for f in fams:
            row[f"{f}_skill_mean"] = d[f]["skill_trained_mean"] if f in d else ""
            row[f"{f}_delta_probe_mean"] = d[f]["delta_probe_mean"] if f in d else ""
            row[f"{f}_delta_folds_pos"] = d[f]["delta_probe_folds_pos"] if f in d else ""
        e_same = [f for f in TRAINED_E if f in d]
        row["lane_e_present"] = bool(e_same)
        if e_same:  # controls must agree across lanes (same code, seed, rows)
            e_id = d[e_same[0]]["skill_identity_mean"]
            f_id = next(d[f]["skill_identity_mean"] for f in TRAINED_F if f in d)
            row["identity_skill_abs_diff_E_vs_F"] = abs(e_id - f_id)
            e_r = d[e_same[0]]["skill_random_mean"]
            f_r = next(d[f]["skill_random_mean"] for f in TRAINED_F if f in d)
            row["random_skill_abs_diff_E_vs_F"] = abs(e_r - f_r)
        side.append(row)
    cols = sorted({k for r in side for k in r}, key=lambda k: list(side[0].keys()).index(k) if k in side[0] else 99) if side else []
    with open(os.path.join(out_dir, "extractibility_matrix_vs_laneE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for r in side:
            w.writerow({k: fmt(r.get(k, "")) for k in cols})
    # summary per family, all cells and restricted to (batch, feature) pairs measured by BOTH lanes
    both = {(c["batch"], c["feature_id"]) for c in allc if c["lane"] == "F"} & \
           {(c["batch"], c["feature_id"]) for c in allc if c["lane"] == "E"}
    write_summary(allc, os.path.join(out_dir, "extractibility_summary.csv"))
    write_summary([c for c in allc if (c["batch"], c["feature_id"]) in both],
                  os.path.join(out_dir, "extractibility_summary_shared_features.csv"))
    print(json.dumps({"cells": len(allc), "side_rows": len(side), "shared_features": len(both)}))


def write_summary(allc, path):
    summ = defaultdict(lambda: defaultdict(list))
    for c in allc:
        s = summ[(c["lane"], c["batch"], c["family"])]
        s["features"].append(c["feature_id"]); s["dp"].append(c["delta_probe_mean"])
        s["dpos"].append(c["delta_probe_folds_pos"] == c["n_folds"]); s["pv"].append(c["preservation_mean"])
        s["sk"].append(c["skill_trained_mean"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lane", "batch", "family", "n_features", "n_cells", "cells_delta_probe_mean_pos",
                    "cells_delta_probe_pos_all_folds", "cells_preservation_mean_pos", "cells_skill_trained_mean_pos",
                    "median_delta_probe", "median_preservation"])
        for (lane, b, fam), s in sorted(summ.items()):
            dp, pv, sk = np.array(s["dp"]), np.array(s["pv"]), np.array(s["sk"])
            w.writerow([lane, b, fam, len(set(s["features"])), len(dp), int((dp > 0).sum()), int(sum(s["dpos"])),
                        int((pv > 0).sum()), int((sk > 0).sum()), f"{np.nanmedian(dp):.6g}", f"{np.nanmedian(pv):.6g}"])


if __name__ == "__main__":
    main()
