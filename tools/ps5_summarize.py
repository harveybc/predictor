"""Summarize PS5 records: one row per (fold, pair, target) with the four arms, the zero-return naive on
the same rows, the decision, and per-seed spread. DEVELOPMENT evidence (inner folds of TRAIN only)."""
import json
import sys
from pathlib import Path


def counts(r):
    """Coordinator rule: a RE_ENTERS counts only if the joint arm beats the same-row zero-return naive by more
    than its seed half-spread, and every seed of the joint arm is below the naive."""
    if r["decision"] != "RE_ENTERS":
        return False
    a, b = r["pair"]
    seeds = [s["val_mae"] for s in r["arms"][f"base+{a}+{b}"]["per_seed"].values()]
    naive = r["naive"]["mae"]
    return (naive - sum(seeds) / len(seeds) > (max(seeds) - min(seeds)) / 2) and max(seeds) < naive


def main(run_dir):
    rows = []
    for path in sorted(Path(run_dir).glob("ps5_*.json")):
        r = json.loads(path.read_text())
        a, b = r["pair"]
        arms = r["arms"]
        means = {k: arms[k]["mean_val_mae"] for k in ("base", f"base+{a}", f"base+{b}", f"base+{a}+{b}")}
        spread = max(abs(s["val_mae"] - arms["base"]["mean_val_mae"]) for s in arms["base"]["per_seed"].values())
        rows.append({"unit": path.name.split("_")[1], "fold": r["fold"], "pair": f"{a}*{b}", "target": r["target"],
                     "naive_zero_return_MAE": r["naive"]["mae"], "val_rows": next(iter(arms["base"]["per_seed"].values()))["val_rows"],
                     **{k.replace(f"+{a}", "+a").replace(f"+{b}", "+b"): v for k, v in means.items()},
                     "best_arm_beats_naive": min(means.values()) < r["naive"]["mae"],
                     "base_seed_half_spread": spread,
                     "counts": counts(r), "decision": r["decision"], "seconds": r["seconds"]})
    out = {"label": "DEVELOPMENT", "units": len(rows),
           "re_enters": sum(r["decision"] == "RE_ENTERS" for r in rows),
           "re_enters_counting": sum(r["counts"] for r in rows),
           "any_arm_beats_zero_return_naive": sum(r["best_arm_beats_naive"] for r in rows), "rows": rows}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
