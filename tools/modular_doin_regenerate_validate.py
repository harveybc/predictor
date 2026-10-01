#!/usr/bin/env python3
"""Regenerate every corrected-queue candidate through the integrated grammar and build it once.

For each candidate row of the corrected campaign that is not terminal, the flat
M04 parameters are re-materialized with ``from_flat`` (M04 candidate layer), and
the resulting ``nested["model"]`` is passed through the integrated tip's
``predictor_plugins.modular_config``: ``flatten`` -> ``unflatten`` must return
the normalized model unchanged and ``dumps`` gives its canonical bytes. The model
is then built once with ``build_modular`` (no data, no fit) to prove the engine
accepts it; the parameter count and canonical sha256 are recorded. Any refusal
is reported per candidate and nothing is dispatched. Output: REGENERATION.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkout", required=True, help="integrated tip checkout (engine + modular_config)")
    parser.add_argument("--m04", required=True, help="checkout providing tools/modular_search_space.py")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path[:0] = [args.checkout, args.m04]
    from tools import modular_search_space as ss  # M04 layer
    from predictor_plugins import modular_config as grammar
    from predictor_plugins import modular_temporal as mt

    root = Path(args.campaign)
    decl = json.loads((root / "CAMPAIGN.json").read_text())
    db = sqlite3.connect(f"file:{root / 'queue.sqlite'}?mode=ro", uri=True)
    rows = db.execute("SELECT cid, label, seed, status, flat FROM candidates WHERE status NOT IN "
                      "('SUPERSEDED_OLD_ARCH','CANCELLED_SUPERSEDED') ORDER BY position").fetchall()
    built, results = {}, []
    for cid, label, seed, status, flat in rows:
        entry = {"cid": cid, "label": label, "seed": seed, "status": status}
        try:
            nested = ss.from_flat(json.loads(flat), decl["base"], decl["search_space"])
            normalized = grammar.loads(json.dumps(nested["model"]))
            if grammar.unflatten(grammar.flatten(normalized)) != normalized:
                raise ValueError("modular_config flatten/unflatten is not the identity on this model")
            text = grammar.dumps(normalized)
            entry["model_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            entry["candidate_cid_recomputed"] = ss.digest(nested)
            entry["cid_matches_queue"] = entry["candidate_cid_recomputed"] == cid
            key = entry["model_sha256"]
            if key not in built:  # seeds share the model; build each distinct model once
                bundle = mt.build_modular(normalized)
                built[key] = {"parameters": int(bundle.forecast_model.count_params()),
                              "fused_shape": list(bundle.fusion_model.output_shape[1:]),
                              "latent_shape": list(bundle.encoder_model.output_shape[1:])}
            entry.update(built[key], verdict="BUILT")
        except Exception as exc:  # recorded per candidate; nothing is dispatched
            entry.update(verdict="REFUSED", error=f"{type(exc).__name__}: {exc}")
        results.append(entry)
    report = {"schema": "lane_d.regeneration.v1", "campaign": decl["campaign_id"], "checkout": args.checkout,
              "candidates": results,
              "counts": {v: sum(r["verdict"] == v for r in results) for v in ("BUILT", "REFUSED")}}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
