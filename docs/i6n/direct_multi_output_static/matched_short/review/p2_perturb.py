"""P2 (SC02/SC04 + reproducibility): rebuild the common population with the REAL adapter
(common_population.build_common_population) from the real retained input and from perturbed
SCRATCH copies of it. No fit, no TEST (the input holds no bar at or after 2025-01-01; load_input
refuses one).

  base    : unmodified input -> must reproduce the sealed population digest exactly.
  gt_t    : every bar with ts > T1 (inside DEVELOPMENT 2024) perturbed (x*1.37+0.5, price*1.01).
            For every origin t <= T1, both arms' inputs must be byte-identical; a label must change
            iff t+h > T1; masks, scaler and naive scales identical.
  inner_dev: every bar with ts >= INNER start perturbed. FIT arrays, the shared scaler (mean, std,
            row digest) and the naive loss scales must be bit-identical.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -W 7200 -m 6G -t 3600s -n i6dmo-review -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p2_perturb.py
"""
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

from tools.i6dmo import common_population as CP, design as D, provenance  # noqa: E402

WORK = C.SCRATCH / "p2"
T1 = int(datetime(2024, 7, 10, 13, 0, tzinfo=timezone.utc).timestamp())
INNER_START = int(datetime(2023, 11, 6, 0, 0, tzinfo=timezone.utc).timestamp())
HOUR = 3600


def designs():
    exps = {a: D.load_design(C.MATCHED / f"EXPERIMENT.{a}.json") for a in CP.ARMS}
    dds = {a: json.loads((C.MATCHED / f"DESIGN.{a}.json").read_text()) for a in CP.ARMS}
    return exps, dds


def perturbed_input(name, cut, inclusive):
    dst = WORK / f"input_{name}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(C.INPUT, dst)
    with np.load(dst / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"].copy(), z["x"].copy(), z["price"].copy()
    sel = ts >= cut if inclusive else ts > cut
    x[sel] = x[sel] * 1.37 + 0.5
    price[sel] = price[sel] * 1.01
    np.savez(dst / "INPUT.npz", ts=ts, x=x, price=price)
    clock = json.loads((dst / "CLOCK.json").read_text())
    clock["input_sha256"] = provenance.file_sha256(dst / "INPUT.npz")
    (dst / "CLOCK.json").write_text(json.dumps(clock))
    return dst, int(sel.sum())


def build(input_dir):
    exps, dds = designs()
    t0 = time.time()
    pop = CP.build_common_population(input_dir, exps, dds, code_commit=C.SEAL["identities"]["code"]["commit"])
    return pop, round(time.time() - t0, 1)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def scaler_and_naive_equal(p, q):
    return {"scaler_mean_bitwise": same(p.scaler.mean, q.scaler.mean),
            "scaler_std_bitwise": same(p.scaler.std, q.scaler.std),
            "scaler_row_digest_equal": p.scaler.fit_row_digest == q.scaler.fit_row_digest,
            "scaler_n_rows_equal": p.scaler.n_rows == q.scaler.n_rows,
            "naive_loss_scale_equal": p.naive_loss_scale == q.naive_loss_scale}


def masks_equal(p, q):
    return {s: all(same(getattr(p.splits[s], k), getattr(q.splits[s], k))
                   for k in ("origin_row_ids", "head_mask", "entry_mask", "management_mask"))
            for s in CP.SPLIT_NAMES}


def gt_t_checks(base, pert):
    out = {}
    hs = np.asarray(CP.HORIZONS, dtype=np.int64) * HOUR
    for arm in CP.ARMS:
        for split in CP.SPLIT_NAMES:
            a, b = base.arm_arrays[arm][split], pert.arm_arrays[arm][split]
            t = a["row_id"]
            le = t <= T1
            x_same_le = bool(np.array_equal(a["x"][le], b["x"][le])) and \
                a["x"][le].tobytes() == b["x"][le].tobytes()
            slots_le_t = bool((a["slot_instants"] <= t[:, None]).all())
            p0_same_le = a["origin_price"][le].tobytes() == b["origin_price"][le].tobytes()
            label_t = t[:, None] + hs[None, :]
            m = a["head_mask"]
            after = m & (label_t > T1) & le[:, None]
            before = m & (label_t <= T1) & le[:, None]
            changed = a["target_price"] != b["target_price"]
            out[f"{arm}.{split}"] = {
                "rows_with_origin_le_T1": int(le.sum()),
                "inputs_byte_identical_for_origin_le_T1": x_same_le,
                "every_slot_instant_le_origin": slots_le_t,
                "origin_price_identical_for_origin_le_T1": p0_same_le,
                "labels_with_t_plus_h_le_T1": int(before.sum()),
                "labels_with_t_plus_h_le_T1_unchanged": bool(not changed[before].any()),
                "labels_with_t_plus_h_gt_T1_and_t_le_T1": int(after.sum()),
                "labels_with_t_plus_h_gt_T1_all_changed": bool(changed[after].all()) if after.any() else None,
                "x_changed_rows_with_origin_gt_T1": int((~np.all(a["x"][~le] == b["x"][~le], axis=(1, 2))).sum()),
            }
    return out


def fit_arrays_equal(base, pert):
    return {arm: all(same(base.arm_arrays[arm]["FIT"][k], pert.arm_arrays[arm]["FIT"][k])
                     for k in base.arm_arrays[arm]["FIT"]) for arm in CP.ARMS}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    res = {"T1": datetime.fromtimestamp(T1, timezone.utc).isoformat(),
           "INNER_START": datetime.fromtimestamp(INNER_START, timezone.utc).isoformat()}
    base, sec = build(C.INPUT)
    res["base"] = {"seconds": sec, "digest": base.digest, "sealed": C.DIGEST,
                   "reproduces_sealed_digest": base.digest == C.DIGEST}
    print("base", res["base"])
    d1, n1 = perturbed_input("gt_t", T1, inclusive=False)
    p1, sec = build(d1)
    res["gt_t"] = {"seconds": sec, "bars_perturbed": n1, "digest_differs": p1.digest != base.digest,
                   "masks_equal": masks_equal(base, p1), **scaler_and_naive_equal(base, p1),
                   "per_arm_split": gt_t_checks(base, p1)}
    print("gt_t done")
    del p1
    shutil.rmtree(d1)
    d2, n2 = perturbed_input("inner_dev", INNER_START, inclusive=True)
    p2, sec = build(d2)
    res["inner_dev"] = {"seconds": sec, "bars_perturbed": n2, "masks_equal": masks_equal(base, p2),
                        **scaler_and_naive_equal(base, p2), "fit_arm_arrays_bitwise": fit_arrays_equal(base, p2),
                        "inner_x_changed": {a: not same(base.arm_arrays[a]["INNER"]["x"], p2.arm_arrays[a]["INNER"]["x"])
                                            for a in CP.ARMS}}
    shutil.rmtree(d2)
    g = res["gt_t"]["per_arm_split"]
    ok = res["base"]["reproduces_sealed_digest"] and all(res["gt_t"]["masks_equal"].values()) and \
        all(v["inputs_byte_identical_for_origin_le_T1"] and v["every_slot_instant_le_origin"]
            and v["labels_with_t_plus_h_le_T1_unchanged"]
            and v["labels_with_t_plus_h_gt_T1_all_changed"] in (True, None) for v in g.values()) and \
        all(res["inner_dev"][k] for k in ("scaler_mean_bitwise", "scaler_std_bitwise", "scaler_row_digest_equal",
                                          "naive_loss_scale_equal")) and \
        all(res["inner_dev"]["fit_arm_arrays_bitwise"].values()) and all(res["inner_dev"]["masks_equal"].values())
    res["verdict"] = "PASS" if ok else "DEFECT"
    C.write("p2_perturb.json", res)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
