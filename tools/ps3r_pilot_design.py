"""PS3-R pilot design helper: stratified input draw and CPU timing on synthetic noise (no candidate fit).

    python tools/ps3r_pilot_design.py select --worklist eth_worklist.csv --out selection.json
    CUDA_VISIBLE_DEVICES= python tools/ps3r_pilot_design.py timing --out timing.json

``select`` reads lane B's ETH 4h work list (feature-eng 9c8e1a0, sha256 f75260a6...),
assigns each feature its majority tier over the three inner folds (ties to the
higher tier), treats every feature drawn EXPLORATORY in some fold and NOT
ranked by majority (majority DEFERRED/EXPLORATORY) as the exploratory stratum, and draws a seeded stratified sample of 20:
PRIORITY 5, SYNERGY 4, REPRESENTATIVE 4, EXPLORATORY 3, DEFERRED 4.

``timing`` measures wall seconds per optimizer update for the two trained
families on SYNTHETIC NOISE windows of the pilot's exact shape (24 bars x 1
feature, batch 64) and the probe battery on a synthetic latent. It is an
engineering timing, not a fit of any candidate on any real input.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ORDER = ["PRIORITY", "SYNERGY", "REPRESENTATIVE", "EXPLORATORY", "DEFERRED"]
QUOTA = {"PRIORITY": 5, "SYNERGY": 4, "REPRESENTATIVE": 4, "EXPLORATORY": 3, "DEFERRED": 4}
SEED = 20261001
WORKLIST_SHA256 = "f75260a6e7e30d340d042cb685f5e9889797fd7b0a87b4599ef99d0798670c9a"


def select(path):
    import numpy as np
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != WORKLIST_SHA256:
        raise SystemExit(f"work list sha256 {sha} is not the pinned {WORKLIST_SHA256}")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    tiers, caution, explored = collections.defaultdict(dict), set(), set()
    for r in rows:
        tiers[r["feature"]][r["fold"]] = r["tier"]
        if "CAUTION_PERSISTENT" in r["reasons"]:
            caution.add(r["feature"])
        if r["tier"] == "EXPLORATORY":
            explored.add(r["feature"])
    stratum = {}
    for f, by_fold in tiers.items():
        count = collections.Counter(by_fold.values())
        top = max(count.values())
        stratum[f] = min((t for t in count if count[t] == top), key=ORDER.index)
    for f in explored:                      # the exploratory sample: drawn outside the ranking
        if stratum[f] in ("DEFERRED", "EXPLORATORY"):
            stratum[f] = "EXPLORATORY"
    rng = np.random.default_rng(SEED)
    chosen = []
    for tier in ORDER:
        pool = sorted(f for f, s in stratum.items() if s == tier and f not in chosen)
        if len(pool) < QUOTA[tier]:
            raise SystemExit(f"stratum {tier} has {len(pool)} < {QUOTA[tier]} features")
        draw = sorted(rng.choice(pool, size=QUOTA[tier], replace=False).tolist())
        chosen += draw
    sample = [{"feature": f, "stratum": stratum[f], "tiers_by_fold": tiers[f],
               "caution_persistent_input": f in caution,
               "inclusion_probability": QUOTA[stratum[f]] / sum(1 for s in stratum.values() if s == stratum[f])}
              for f in chosen]
    return {"worklist_sha256": sha, "seed": SEED, "quota": QUOTA,
            "stratum_sizes": dict(collections.Counter(stratum.values())), "sample": sample}


def timing():
    import numpy as np
    import tensorflow as tf
    from predictor_plugins import modular_temporal as mt
    from predictor_plugins.modular_temporal import objectives as ob
    from predictor_plugins.modular_temporal import probes as pb
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1024, 24, 1)).astype("float32")
    vx = rng.normal(size=(256, 24, 1)).astype("float32")
    out = {"scope": "synthetic noise; engineering timing, not a candidate fit", "threads": os.cpu_count()}
    for name in ("autoencoder_reconstruction", "ts2vec_contrastive"):
        enc = mt.build_modular(mt.default_config(["x"])).branch_models["branch_0"]
        settings = {"max_epochs": 3, "patience": 3, "batch_size": 64, "learning_rate": 1e-3, "seed": 0}
        try:
            r = ob.fit_objective({"plugin": name}, enc, x, vx, settings)
            out[name] = {"updates": r["observed_updates"], "seconds": r.get("elapsed_seconds"),
                         "seconds_per_update": (r.get("elapsed_seconds") or float("nan")) / r["observed_updates"],
                         "includes": "per-epoch validation passes and first-call tracing"}
        except ValueError as exc:            # noise may not improve on the untrained loss; still timed
            out[name] = {"refused": str(exc)}
    enc = mt.build_modular(mt.default_config(["x"])).branch_models["branch_0"]
    n = 9589
    xs = rng.normal(size=(n, 24, 1)).astype("float32")
    y = {"Y_s@4h": rng.normal(size=n), "Y_l@24h": rng.normal(size=n)}
    t0 = time.monotonic()
    pb.probe_battery(trained=enc, random=pb.untrained_twin(enc, 1), x=xs, targets=y,
                     fit_rows=range(0, 7474), eval_rows=range(7534, n), fold="timing", seed=0)
    out["probe_battery_seconds_inner1_two_targets"] = time.monotonic() - t0
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--worklist", required=True)
    s.add_argument("--out", required=True)
    t = sub.add_parser("timing")
    t.add_argument("--out", required=True)
    a = p.parse_args()
    result = select(a.worklist) if a.cmd == "select" else timing()
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=1)[:3000])


if __name__ == "__main__":
    main()
