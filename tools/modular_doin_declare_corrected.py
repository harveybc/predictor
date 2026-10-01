#!/usr/bin/env python3
"""Persist (never execute) the corrected-architecture DOIN queue for lane D.

Builds a new campaign whose search space names the full-grid engine
(``modular_temporal.v2_full_grid``: branches keep the 24-step grid, fusion
(B,24,16F), PE after fusion, causal Transformer blocks, residual Conv1D core
[2,2,1] x [32,16,8]). Every candidate of the given older campaigns is imported
as ``SUPERSEDED_OLD_ARCH`` (terminal, never dispatched, results kept as evidence
of the old design). New candidates are enqueued and held:

* corrected default per-feature R0 (Huber/MAE x paired seeds):
  ``HOLD:AWAIT_LANE_A_AND_COST_PROFILE``
* corrected grouped32 and the four batch-1 draws carried into the corrected
  space where admissible: ``HOLD:AWAIT_LANE_A_INTEGRATED_COMMIT``
* corrected default R1/R2 (branch, core, both): blocked on compatible donors.

The executor pin is ``PENDING_LANE_A_INTEGRATED_COMMIT`` and ``require_pin`` is
true, so a runner refuses to dispatch until a re-pin amendment names the
integrated commit and a measured cap replaces ``PENDING_STAGED_PROFILE``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import modular_doin_campaign as camp  # noqa: E402
from tools import modular_search_space as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def declare(old_declaration, root, superseded_roots, draws_from):
    d = json.loads(Path(old_declaration).read_text())
    space = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_search_space_v2_full_grid.json").read_text())
    default = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_default_r0_v2_full_grid.json").read_text())
    keep = {k: d[k] for k in ("base", "data", "data_location", "paired_seeds", "input_binding", "env_pin",
                              "verification", "hosts") if k in d}
    executor = dict(d["executor"])
    executor.update(predictor_revision="PENDING_LANE_A_INTEGRATED_COMMIT",
                    predictor_checkout="PENDING_LANE_A_INTEGRATED_COMMIT")
    executor.pop("checkouts_by_revision", None)
    decl = {**keep, "campaign_id": "d_ecl_l24_h24_corrected_r0_v1", "search_space": space,
            "default_candidate": default, "default_huber_delta": 1.0, "executor": executor,
            "require_pin": True,
            "resources": {"train": {"cap": "PENDING_STAGED_PROFILE", "wall": "2h", "timeout_seconds": 6600},
                          "verify": {"cap": "PENDING_STAGED_PROFILE", "wall": "20m", "timeout_seconds": 1100}},
            "architecture": {"design": "owner-corrected full-grid modular stack (da4ce7b4; lane A integrated commit)",
                             "branch": "(B,24,1) -> causal Conv1D -> (B,24,16)", "fusion": "(B,24,16*F)",
                             "core": "PE after fusion, projection, 2 causal Transformer blocks, residual Conv1D "
                                     "stages time 24->12->6->6, channels 32->16->8",
                             "head": "flatten completed latent -> horizons x targets"},
            "superseded_old_arch": [str(r) for r in superseded_roots]}
    campaign = camp.Campaign.create(root, decl)
    for source in superseded_roots:
        campaign.import_superseded(
            source, "old architecture (branch_steps 12 default, core [2,1,1] compress); evidence of the old design only")
    added = []
    pairs = [("corrected_default_R0", default),
             ("corrected_grouped32_R0", {**default, "branch.grouping_size": 32})]
    if draws_from:
        db = sqlite3.connect(f"file:{Path(draws_from) / 'queue.sqlite'}?mode=ro", uri=True)
        seen = set()
        for label, flat in db.execute("SELECT label, flat FROM candidates WHERE label LIKE 'draw%_huber' "
                                      "ORDER BY position"):
            name = label.rsplit("_", 1)[0]
            if name in seen:
                continue
            seen.add(name)
            flat = json.loads(flat)
            delta = flat.pop("train.huber_delta")
            flat.pop("train.seed")
            pairs.append((f"corrected_{name}_R0", {**flat, "train.huber_delta": delta}))
    skipped = []
    for label, flat in pairs:
        delta = flat.pop("train.huber_delta", 1.0)
        try:
            h, m = camp.paired_loss_arms({**flat, "model.branch_steps": 24}, delta)
            added += campaign.enqueue(h, label + "_huber") + campaign.enqueue(m, label + "_mae")
        except ss.SearchSpaceError as exc:
            skipped.append({"label": label, "reason": str(exc)})
    for regime in ("R1", "R2"):
        for label, fl in (("branch", {"branch.regime": regime}), ("core", {"core.regime": regime}),
                          ("branch_core", {"branch.regime": regime, "core.regime": regime})):
            added += campaign.enqueue({**default, **fl, "train.huber_delta": 1.0}, f"corrected_default_{label}_{regime}_huber")
    campaign.hold(lambda f: f["branch.grouping_size"] == 1, "AWAIT_LANE_A_AND_COST_PROFILE")
    campaign.hold(lambda f: True, "AWAIT_LANE_A_INTEGRATED_COMMIT")
    counts = dict(campaign.db.execute("SELECT status, COUNT(*) FROM candidates GROUP BY status").fetchall())
    reasons = dict(campaign.db.execute("SELECT blocked_reason, COUNT(*) FROM candidates WHERE status='blocked' "
                                       "GROUP BY blocked_reason").fetchall())
    return {"campaign": decl["campaign_id"], "counts": counts, "blocked_reasons": reasons, "skipped": skipped,
            "pinned": campaign.pinned()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old-declaration", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--superseded", nargs="+", required=True)
    parser.add_argument("--draws-from")
    args = parser.parse_args()
    print(json.dumps(declare(args.old_declaration, args.root, args.superseded, args.draws_from), indent=1))


if __name__ == "__main__":
    main()
