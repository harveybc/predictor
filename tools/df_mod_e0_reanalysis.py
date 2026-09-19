#!/usr/bin/env python3
"""RP12: reanalysis WITHOUT training of the executed MOD-E0-DEV pilot under the corrected trend
descriptor (v2, Var(T + R)) against the executed one (v1, Var(X)).

For every (level, r, seed) the run consumed: the exact ORDERED profile labels, the kept-descriptor
mask, the group sizes, the predefined random redistributions (the candidates the H2 random arms
draw from, after the relabeling filter), the assignment every attempt recorded, the model inputs
(x digest: windows are built from x only) and the H3 donors (the extractor's assignment). Reuse of
the executed results is justified only where every consumed datum is unchanged; anything that
changes is listed so that only the affected evidence is isolated. Originals are read, never written.

    python tools/df_mod_e0_reanalysis.py --root RUN_ROOT --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("df_mod_e0")


def reanalyse(root: Path) -> dict:
    records = {}
    for attempt in sorted((root / "attempts").iterdir()):
        if (attempt / "cell.json").is_file():
            rec = json.loads((attempt / "cell.json").read_bytes())
            records[attempt.name] = rec
    combos = sorted({(r["level"], r["r"], r["seed"]) for r in records.values()})
    out = {"schema": "df_mod_e0_reanalysis.v1", "root": str(root), "attempts": len(records), "combos": [], "descriptor_versions": {"executed": 1, "corrected": 2},
           "changed": {"labels": [], "kept": [], "sizes": [], "candidates": [], "recorded_assignment": [], "x_digest": [], "donor_assignment": []}}
    lo, hi = E.boundaries()["train"]
    for level, r, seed in combos:
        gen = E.generate(level, r, seed)
        x = gen["x"]
        xd = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        p1, p2 = E.profiles(x[lo:hi], 1), E.profiles(x[lo:hi], 2)
        l1, l2 = E.average_linkage(p1["scaled"], 2), E.average_linkage(p2["scaled"], 2)
        s1, s2 = [l1.count(g) for g in sorted(set(l1))], [l2.count(g) for g in sorted(set(l2))]
        c1 = [c for c in E.random_assignments(x.shape[1], 2, s1, 8, seed=1000 + seed) if not E.same_partition(c, l1)]
        c2 = [c for c in E.random_assignments(x.shape[1], 2, s2, 8, seed=1000 + seed) if not E.same_partition(c, l2)]
        cols = [n for n, a, b in zip(p1["names"], p1["raw"].T, p2["raw"].T) if not np.allclose(a, b)]
        members = {k: v for k, v in records.items() if (v["level"], v["r"], v["seed"]) == (level, r, seed)}
        rec_ok = all(v["profiles"]["labels"] == l1 for v in members.values())
        assign_ok = all((v["assignment"] == l1) if (v["arm"] in ("profiles", "extractor", "sequence", "summary")) else (v["assignment"] == c1[int(v["arm"].split("_")[-1])])
                        for v in members.values())
        x_ok = all(v["x_sha256"] == xd for v in members.values())
        donors = {k: v for k, v in members.items() if v["arm"] == "extractor"}
        donor_ok = all(v["assignment"] == l2 for v in donors.values())
        entry = {"level": level, "r": r, "seed": seed, "attempts": sorted(members), "x_sha256": xd, "x_equals_records": x_ok,
                 "labels_executed_v1": l1, "labels_corrected_v2": l2, "labels_ordered_equal": l1 == l2, "same_partition": E.same_partition(l1, l2),
                 "kept_v1": p1["kept"], "kept_v2": p2["kept"], "kept_equal": p1["kept"] == p2["kept"], "sizes_v1": s1, "sizes_v2": s2,
                 "descriptor_columns_changed": cols, "candidates_equal": c1 == c2, "candidates_v2": c2,
                 "records_reproduce_v1_labels": rec_ok, "recorded_assignments_equal_v1_derivation": assign_ok,
                 "donor_assignment_equals_v2": donor_ok, "ari_v1": E.adjusted_rand(l1, [0] * 4 + [1] * 4), "ari_v2": E.adjusted_rand(l2, [0] * 4 + [1] * 4)}
        out["combos"].append(entry)
        key = f"h{level}_r{r}_s{seed}"
        if not entry["labels_ordered_equal"]:
            out["changed"]["labels"].append(key)
        if not entry["kept_equal"]:
            out["changed"]["kept"].append(key)
        if s1 != s2:
            out["changed"]["sizes"].append(key)
        if not entry["candidates_equal"]:
            out["changed"]["candidates"].append(key)
        if not assign_ok or not rec_ok:
            out["changed"]["recorded_assignment"].append(key)
        if not x_ok:
            out["changed"]["x_digest"].append(key)
        if not donor_ok:
            out["changed"]["donor_assignment"].append(key)
    out["summary"] = {"combos": len(combos), "ordered_labels_equal": sum(c["labels_ordered_equal"] for c in out["combos"]),
                      "same_partition": sum(c["same_partition"] for c in out["combos"]),
                      "any_consumed_datum_changed": any(out["changed"].values()),
                      "reuse": "JUSTIFIED: every consumed datum (ordered assignment, sizes, redistributions, inputs, donors) is unchanged under v2"
                               if not any(out["changed"].values()) else "NOT JUSTIFIED for the listed combos: isolate the affected evidence"}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    out = reanalyse(args.root)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"summary": out["summary"], "changed": out["changed"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
