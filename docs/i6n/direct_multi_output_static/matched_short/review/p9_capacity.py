"""P9 (SC07): build both REAL sealed arm models on CPU (no fit), count trainable parameters per
component INDEPENDENTLY from the Keras weight shapes, verify the 2304 head delta and that the two
effective model configs differ only in the slot count; then an unreported head change (hidden 9,
or an extra branch filter) must raise CAPACITY_UNREPORTED_CHANGE.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 900s -n i6dmo-review-p9 -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p9_capacity.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

from tools.i6dmo import model as M, train as T  # noqa: E402


def count(weights):
    return int(sum(int(np.prod(w.shape)) for w in weights))


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def main():
    res = {"arms": {}}
    cfg = {}
    for arm in ("mg1", "mg2m"):
        eff = T.effective_from_design(C.DESIGNS[arm])
        cfg[arm] = eff["model"]
        b = M.build_dmo(eff["model"])
        heads = {n: count(h.trainable_weights) for n, h in b.head_models.items()}
        res["arms"][arm] = {"total_trainable_independent": count(b.forecast_model.trainable_weights),
                            "non_trainable": count(b.forecast_model.non_trainable_weights),
                            "branches_independent": sum(count(m.trainable_weights) for m in b.branch_models.values()),
                            "core_independent": count(b.core_model.trainable_weights),
                            "heads_independent": sum(heads.values()), "per_head": sorted(set(heads.values())),
                            "capacity_record_total": M.capacity_record(b)["counts"]["total"]}
    a, b = res["arms"]["mg1"], res["arms"]["mg2m"]
    res["delta_independent"] = {k: a[k] - b[k] for k in ("total_trainable_independent", "branches_independent",
                                                          "core_independent", "heads_independent")}
    res["delta_is_2304_heads_only"] = res["delta_independent"] == {"total_trainable_independent": 2304,
                                                                   "branches_independent": 0, "core_independent": 0,
                                                                   "heads_independent": 2304}
    res["model_config_differences_mg1_vs_mg2m"] = diff(cfg["mg1"], cfg["mg2m"])
    # unreported changes
    for name, mutate in (("head_hidden_9", lambda c: c["heads"].__setitem__("hidden_units", 9)),
                         ("branch_filters_5", lambda c: c["branch"].__setitem__("filters", 5))):
        c = copy.deepcopy(cfg["mg1"])
        try:
            mutate(c)
            bb = M.build_dmo(c)
            M.capacity_record(bb)
            res[f"unreported_{name}"] = "ACCEPTED"
        except M.CapacityError as exc:
            res[f"unreported_{name}"] = f"REFUSED:{exc.code}"
        except Exception as exc:  # noqa: BLE001
            res[f"unreported_{name}"] = f"OTHER:{type(exc).__name__}:{str(exc)[:160]}"
    C.write("p9_capacity.json", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
