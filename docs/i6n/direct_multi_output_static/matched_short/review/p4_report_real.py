"""P4 (SC05/SC06/SC07 report side): drive the REAL matched report hook (reporting.run_unit ->
run_matched_unit) on the REAL frozen population and real retained input, with a hand-built
upstream fit terminal (no fit runs; predictions are a label-free deterministic function of p(t)).
No arrays_loader is injected: the hook reads the population, the arm source and the input itself.

  * positive: both arms report COMPLETE; per-head pooled MAE / naive / relative error are then
    recomputed INDEPENDENTLY here from the saved predictions file and from INPUT.npz by timestamp.
  * decide_matched on the two METRICS documents: capacity delta 2304 and the decision are
    recomputed independently; tampered capacity / rows are refused.
  * SC06 negatives: closed full-population MG2 identity, another population digest, the closed
    MG2 design, full-population-shaped predictions (extra rows), unreported capacity change.
  * observation: what the hook does with a receipt stamped scientific=false / device=cpu.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p4 -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p4_report_real.py
"""
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

from tools.i6dmo import model as M, reporting, train as T  # noqa: E402

WORK = C.SCRATCH / "p4"
H = 3600
HORIZONS = [2, 3, 4, 5, 6, 12]
CLOSED_MG2_EFF = next(iter(T.CLOSED_FULL_POPULATION_EFFECTIVE))


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_upstream(folder, arm, *, receipt_patch=None, extra_rows=0, pop_digest=None):
    """A stored fit terminal in the worker's layout: folder/TERMINAL.json + folder/out/train/..."""
    if folder.exists():
        shutil.rmtree(folder)
    out = folder / "out" / "train"
    (out / "predictions").mkdir(parents=True)
    a = npz(C.POPULATION / f"ARM_{arm}_DEVELOPMENT.npz")
    row_id, p0, tgt, mask = a["row_id"], a["origin_price"], a["target_price"], a["head_mask"]
    k = np.arange(len(row_id), dtype=np.float64)
    pred = p0[:, None] + 1e-4 * np.sin(k[:, None] * 0.37 + np.arange(6)[None, :])  # label-free
    if extra_rows:
        last = int(row_id[-1])
        row_id = np.concatenate([row_id, last + H * np.arange(1, extra_rows + 1)])
        p0 = np.concatenate([p0, np.full(extra_rows, p0[-1])])
        tgt = np.concatenate([tgt, np.full((extra_rows, 6), np.nan)])
        mask = np.concatenate([mask, np.zeros((extra_rows, 6), bool)])
        pred = np.concatenate([pred, np.repeat(pred[-1:], extra_rows, axis=0)])
    np.savez(out / "predictions" / "validation.npz", row_id=row_id, horizons_hours=np.asarray(HORIZONS, np.int64),
             origin_price=p0, predicted_price=pred, target_price=tgt, mask=mask)
    pred_sha = C.sha(out / "predictions" / "validation.npz")
    family = M.MATCHED_ARMS[arm]
    receipt = {"schema": "i6dmo.train_receipt.v1", "status": "COMPLETED", "code": "COMPLETED",
               "scientific": True, "device": "gpu",
               "identity": {"effective_config_sha256": "e" * 64, "common_population_sha256": pop_digest or C.DIGEST,
                            "matched_arm": arm, "weights_sha256": "w" * 64},
               "matched": {"arm": arm, "common_population_sha256": pop_digest or C.DIGEST,
                           "capacity": {"schema": M.CAPACITY_SCHEMA, "arm": arm,
                                        "counts": copy.deepcopy(M.DECLARED_CAPACITY[family])}},
               "artifacts": {"predictions/validation.npz": pred_sha}}
    if receipt_patch:
        receipt_patch(receipt)
    (out / "RECEIPT.json").write_text(json.dumps(receipt))
    terminal = {"schema": "i6dmo.terminal.v1", "unit_id": f"{arm}.fit", "status": "COMPLETE",
                "artifacts": {"predictions/validation.npz": {"path": "train/predictions/validation.npz",
                                                             "sha256": pred_sha},
                              "RECEIPT": {"path": "train/RECEIPT.json", "sha256": C.sha(out / "RECEIPT.json")}},
                "trainer_identity": receipt["identity"]}
    stored = {"schema": "i6dmo.stored_terminal.v1", "unit_id": f"{arm}.fit", "terminal": terminal,
              "stored_at": 1760000000.0}
    (folder / "TERMINAL.json").write_text(json.dumps(stored))
    return {"path": str(folder / "TERMINAL.json"), "sha256": C.sha(folder / "TERMINAL.json")}


def report(arm, up, *, design=None, name=None):
    design = design or json.loads(C.DESIGNS[arm].read_text())
    out = WORK / f"report_{name or arm}"
    if out.exists():
        shutil.rmtree(out)
    spec = {"kind": "report", "family": "short", "unit_id": f"{arm}.report", "campaign_id": "retsu-probe",
            "identity_sha256": "0" * 64, "effective_config": {"design": design, "code_commit": "probe"},
            "matched": {"arm": arm, "fit_unit": f"{arm}.fit",
                        "input": {"path": str(C.INPUT), "sha256": C.sha(C.INPUT / "INPUT.npz")}},
            "common_population": {"path": str(C.POPULATION), "sha256": C.DIGEST},
            "upstream": {f"{arm}.fit": up}, "pins": {}}
    out.mkdir(parents=True)
    term = reporting.run_unit(spec, out, out / "PROGRESS.json")
    metrics = json.loads((out / "METRICS.json").read_text()) if (out / "METRICS.json").exists() else None
    return term, metrics


def independent(arm, pred_file):
    """Pooled MAE/naive/relative error per head from the saved predictions + INPUT.npz by timestamp."""
    with np.load(C.INPUT / "INPUT.npz", allow_pickle=False) as z:
        ts, price = z["ts"], z["price"]
    pos = {int(v): i for i, v in enumerate(ts)}
    p = npz(pred_file)
    dev = npz(C.POPULATION / "POPULATION_DEVELOPMENT.npz")
    assert np.array_equal(p["row_id"], dev["origin_row_ids"]) and np.array_equal(p["mask"], dev["head_mask"])
    out = {}
    for j, h in enumerate(HORIZONS):
        m = dev["head_mask"][:, j]
        t = dev["origin_row_ids"][m]
        y = price[[pos[int(v) + h * H] for v in t]]
        p0 = price[[pos[int(v)] for v in t]]
        mae = math.fsum(np.abs(p["predicted_price"][m, j] - y).tolist()) / len(t)
        nv = math.fsum(np.abs(y - p0).tolist()) / len(t)
        out[str(h)] = {"n": int(len(t)), "mae": mae, "naive": nv, "relative_error": (mae - nv) / nv}
    return out


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    res = {"positive": {}, "negatives": {}, "observations": {}}
    metrics = {}
    for arm in ("mg1", "mg2m"):
        up = write_upstream(WORK / f"up_{arm}", arm)
        term, met = report(arm, up)
        metrics[arm] = met
        ind = independent(arm, WORK / f"up_{arm}" / "out" / "train" / "predictions" / "validation.npz")
        cmp = {}
        for h in map(str, HORIZONS):
            got = met["per_horizon"][h]
            cmp[h] = {"n_equal": got["n"] == ind[h]["n"], "mae": close(got["mae"], ind[h]["mae"]),
                      "naive": close(got["naive_mae"], ind[h]["naive"]),
                      "relative_error": close(got["relative_error"], ind[h]["relative_error"]),
                      "max_abs_diff": max(abs(got["mae"] - ind[h]["mae"]), abs(got["naive_mae"] - ind[h]["naive"]))}
        res["positive"][arm] = {"status": term["status"], "disposition": term["disposition"],
                                "selection": met["matched"]["selection"]["status"] if met else None,
                                "independent_vs_report": cmp,
                                "all_agree": all(all(v[k] for k in ("n_equal", "mae", "naive", "relative_error"))
                                                 for v in cmp.values()),
                                "naive_equal_across_arms_note": "filled below"}
    # naive per head must be identical across the arms (same rows, same prices)
    res["naive_identical_across_arms"] = all(
        metrics["mg1"]["per_horizon"][h]["naive_mae"] == metrics["mg2m"]["per_horizon"][h]["naive_mae"]
        for h in map(str, HORIZONS))
    # decide_matched
    verdict = reporting.decide_matched(metrics["mg1"], metrics["mg2m"])
    s1 = sum(metrics["mg1"]["per_horizon"][h]["relative_error"] for h in map(str, HORIZONS)) / 6
    s2 = sum(metrics["mg2m"]["per_horizon"][h]["relative_error"] for h in map(str, HORIZONS)) / 6
    expect = "mg2m" if abs(s1 - s2) <= 1e-8 else ("mg1" if s1 < s2 else "mg2m")
    res["decide_matched"] = {"decision": verdict["decision"], "selected": verdict["selected"],
                             "capacity_delta": verdict["capacity_delta"]["delta"],
                             "independent_scores": {"mg1": s1, "mg2m": s2}, "independent_selected": expect,
                             "agrees": verdict["selected"] == expect and close(verdict["scores"]["mg1"], s1)
                             and close(verdict["scores"]["mg2m"], s2),
                             "head_delta_is_2304": verdict["capacity_delta"]["delta"]["heads"] == 2304
                             and verdict["capacity_delta"]["delta"]["total"] == 2304}
    neg = res["negatives"]
    for name, mutate in (
            ("capacity_heads_plus_one", lambda m: m["matched"]["capacity"]["counts"].__setitem__(
                "heads", m["matched"]["capacity"]["counts"]["heads"] + 1)),
            ("capacity_core_changed_both_totals", lambda m: (m["matched"]["capacity"]["counts"].__setitem__(
                "core", 1981), m["matched"]["capacity"]["counts"].__setitem__("total", 7187))),
            ("development_rows_differ", lambda m: m["matched"].__setitem__("development_row_ids_sha256", "0" * 64)),
            ("population_differs", lambda m: m["matched"].__setitem__("common_population_sha256", "1" * 64))):
        bad = copy.deepcopy(metrics["mg1"])
        mutate(bad)
        try:
            v = reporting.decide_matched(bad, metrics["mg2m"])
            neg[f"decide_{name}"] = {"refused": False, "decision": v["decision"]}
        except reporting.ReportFailure as exc:
            neg[f"decide_{name}"] = {"refused": True, "code": exc.code}
    cases = {
        "closed_mg2_effective_identity": dict(receipt_patch=lambda r: r["identity"].__setitem__(
            "effective_config_sha256", CLOSED_MG2_EFF)),
        "other_population_digest": dict(pop_digest="2" * 64),
        "full_population_shaped_predictions_extra_rows": dict(extra_rows=500),
        "receipt_capacity_unreported_change": dict(receipt_patch=lambda r: r["matched"]["capacity"]["counts"]
                                                   .__setitem__("per_head", 818)),
        "receipt_wrong_arm": dict(receipt_patch=lambda r: (r["identity"].__setitem__("matched_arm", "mg2m"),
                                                           r["matched"].__setitem__("arm", "mg2m"))),
    }
    for name, kw in cases.items():
        up = write_upstream(WORK / f"up_neg_{name}", "mg1", **kw)
        term, _ = report("mg1", up, name=f"neg_{name}")
        neg[name] = {"refused": term["status"] != "COMPLETE", "code": term["disposition"]}
    closed = json.loads((C.REPO / "docs/i6n/direct_multi_output_static/EXPERIMENT.mg2.json").read_text())
    up = write_upstream(WORK / "up_neg_closed_design", "mg2m")
    term, _ = report("mg2m", up, design=closed, name="neg_closed_design")
    neg["closed_full_population_mg2_design"] = {"refused": term["status"] != "COMPLETE", "code": term["disposition"]}
    # observation: non-scientific / CPU receipts
    obs = res["observations"]
    for name, patch in (("receipt_scientific_false_device_cpu",
                         lambda r: (r.__setitem__("scientific", False), r.__setitem__("device", "cpu"))),
                        ("receipt_without_scientific_or_device",
                         lambda r: (r.pop("scientific"), r.pop("device")))):
        up = write_upstream(WORK / f"up_obs_{name}", "mg1", receipt_patch=patch)
        term, met = report("mg1", up, name=f"obs_{name}")
        obs[name] = {"report_status": term["status"], "disposition": term["disposition"],
                     "scientific_fit_recorded": (met or {}).get("matched", {}).get("scientific_fit", "ABSENT")}
        if met:
            v = reporting.decide_matched(met, metrics["mg2m"])
            obs[name]["decide_matched_decision"] = v["decision"]
    ok = all(v["all_agree"] and v["status"] == "COMPLETE" for v in res["positive"].values()) and \
        res["decide_matched"]["agrees"] and res["decide_matched"]["head_delta_is_2304"] and \
        res["naive_identical_across_arms"] and all(v["refused"] for v in neg.values())
    res["verdict"] = "PASS" if ok else "DEFECT"
    res["note"] = ("predictions are label-free synthetic values written by this probe; the upstream "
                   "terminal/receipt are hand-built; NO model was fitted")
    C.write("p4_report_real.json", res)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
