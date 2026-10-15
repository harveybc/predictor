"""Independent check of POPULATION_RESEAL_EQUALITY.json on the REAL arrays (numpy + json only for
the comparison; the data lane's verified loader is used only to confirm both seals still load).

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1200s -n i6dmo-review-equality -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/r_equality_796ec8f4.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

OLD_DIGEST = "4ab41bd87987c6e790851fe1262210b13e0fdbabf971deae9b9fa06fc75014f8"


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def same_arrays(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True) or x.tobytes() != y.tobytes():
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def diff_keys(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff_keys(a.get(k), b.get(k), f"{path}.{k}")
    elif a != b:
        out.append(path)
    return out


def main():
    new_dir = C.POPULATION
    old_dir = new_dir.parent / "population"
    old_b, new_b = (json.loads((d / "POPULATION.json").read_text()) for d in (old_dir, new_dir))
    names = sorted(p.name for p in new_dir.glob("*.npz"))
    res = {"new_digest_from_committed_seal": C.DIGEST, "old_digest": OLD_DIGEST,
           "npz_files": names, "npz_files_same_set": names == sorted(p.name for p in old_dir.glob("*.npz")),
           "npz_content_equal": {n: same_arrays(npz(old_dir / n), npz(new_dir / n)) for n in names}}
    res["body_paths_that_differ"] = diff_keys(old_b, new_b)
    res["body_differs_only_in_identity_block_and_digests"] = all(
        p.startswith((".identities.arms.", ".identities.code.", ".identity_digests.arms", ".digest", ".files."))
        for p in res["body_paths_that_differ"])
    res["scaler_equal"] = old_b["scaler"] == new_b["scaler"]
    res["naive_equal"] = old_b["naive_loss_scale"] == new_b["naive_loss_scale"]
    res["splits_equal"] = old_b["splits"] == new_b["splits"]
    res["counts_equal"] = {s: old_b["splits"][s]["counts"] == new_b["splits"][s]["counts"] for s in new_b["splits"]}
    res["report_equal"] = old_b["report"] == new_b["report"]
    res["calendar_source_splits_identity_equal"] = all(old_b["identities"][k] == new_b["identities"][k]
                                                      for k in ("calendar", "source", "splits", "test"))
    res["new_identity_arms_bind_committed_designs"] = {
        a: new_b["identities"]["arms"][a]["experiment_seal_sha256"] ==
        json.loads(C.DESIGNS[a].read_text())["seal_sha256"] for a in ("mg1", "mg2m")}
    # designs: old (at 4a77bea8) vs new differ only in runtime + seal
    dd = {}
    for a in ("mg1", "mg2m"):
        old = json.loads(subprocess.run(["git", "-C", str(C.REPO), "show",
                                         f"4a77bea8:docs/i6n/direct_multi_output_static/matched_short/EXPERIMENT.{a}.json"],
                                        capture_output=True, text=True, check=True).stdout)
        new = json.loads(C.DESIGNS[a].read_text())
        dd[a] = {"paths_that_differ": diff_keys(old, new), "new_runtime": new["sealed_identities"]["runtime"]}
        from tools.i6dmo import design as D
        try:
            D.validate_design(old)
            dd[a]["old_seal_validates"] = "ACCEPTED"
        except D.DesignError as exc:
            dd[a]["old_seal_validates"] = f"REFUSED:{exc.code}"
    res["designs_old_vs_new"] = dd
    from tools.i6dmo import common_population as CP
    for name, path, digest in (("new", new_dir, C.DIGEST), ("old", old_dir, OLD_DIGEST)):
        try:
            CP.load_common_population(path, expected_digest=digest, input_dir=C.INPUT)
            res[f"verified_load_{name}"] = "VERIFIED"
        except CP.CommonPopulationError as exc:
            res[f"verified_load_{name}"] = f"REFUSED:{exc.code}"
    res["verdict"] = "IDENTICAL_EXCEPT_IDENTITY_BLOCK" if (
        all(res["npz_content_equal"].values()) and res["npz_files_same_set"] and res["scaler_equal"]
        and res["naive_equal"] and res["splits_equal"] and all(res["counts_equal"].values())
        and res["report_equal"] and res["calendar_source_splits_identity_equal"]
        and res["body_differs_only_in_identity_block_and_digests"]
        and all(res["new_identity_arms_bind_committed_designs"].values())) else "DIFFERS"
    C.write("equality_796ec8f4.json", res)
    return 0 if res["verdict"] != "DIFFERS" else 1


if __name__ == "__main__":
    sys.exit(main())
