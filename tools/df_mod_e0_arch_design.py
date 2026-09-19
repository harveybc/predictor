#!/usr/bin/env python3
"""MOD-ARCH-COMPARE numerical design (RP14): the approved ARCH-A / ARCH-B / ARCH-C / ARCH-0 comparison
as a DEVELOPMENT successor of MOD-E0-DEV, with H2 and H3 inside every architecture, the readout
control that separates pooling from history preservation, a bounded donor sensitivity, a trend /
event diagnostic condition, host shards, a budget and a review table. Sealed before any outcome.

    python tools/df_mod_e0_arch_design.py --out DESIGN.json [--levels 0,3] [--max-updates 3000] ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

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
D1 = _load("df_mod_e0_design")

DESIGN_SCHEMA = "df_mod_e0_arch_design.v1"
ARCHS = ("A", "B", "C", "0")
READOUT_ARMS = ("sequence", "sequence_gap", "summary", "summary_last")
ROLES = ("COORDINATOR", "WORKER_A", "WORKER_B")
BUDGET = {"aggregate_cpu_seconds": 14400.0, "headroom": 0.25, "scope": "RP9-RP16: ML tests, re-verification inference, instrumentation, pilots, "
                                                                        "failures and this comparison; the executed pilot's history is separate",
          "gpu": "none", "hosts": "three healthy roles under their own identities; memory-aware concurrency per host"}


def build(*, levels=(0, 1, 2, 3), replicates=(1, 2, 3), random_assignments=3, h3_level=2, r_values=(0, 1), archs=ARCHS,
          readout_arms=READOUT_ARMS, readout_controls_r=(0, 1), donor_sensitivity=None, diagnostic=None, max_updates=3000, patience=None,
          hosts=ROLES, host_weights=None, successor_of: str | None = None, reason: str | None = None, stage_note: str | None = None,
          only_hypotheses=None) -> dict:
    """`readout_controls_r`: the r values at which the two readout-control arms run (the other r values
    keep sequence + summary only); `host_weights`: relative concurrency capacity per role, used to deal
    dependency groups (memory-aware placement); `patience`: early-stopping patience in epochs."""
    donor_sensitivity = donor_sensitivity if donor_sensitivity is not None else \
        {"archs": ["A", "B"], "r": [1], "arms": ["sequence", "summary"],
         "why": "bounded: is the H3 contrast sensitive to a donor optimised for the summary receiver? Shared within each pair; "
                "the sequence-trained donor is NOT called neutral"}
    diagnostic = diagnostic if diagnostic is not None else \
        {"condition": "trend_event", "level": 3, "r": 1, "seeds": [1, 2], "arms": ["profiles"],
         "why": "absent from the executed pilot: a deterministic drift and two level shifts (one in train, one in test) test receiver "
                "adequacy and the corrected trend descriptor; no effect is estimated on it"}
    training = dict(E.TRAINING)
    training["max_updates"] = int(max_updates)
    if patience is not None:
        training["early_stopping"] = {**training["early_stopping"], "patience": int(patience)}
    if stage_note:
        training["stage_note"] = stage_note
    host_weights = dict(host_weights or {r: 1 for r in hosts})
    doc = {"schema": DESIGN_SCHEMA, "experiment": "MOD-ARCH-COMPARE", "proposal": "P-MOD", "hypotheses": ["H2", "H3", "READOUT", "DX"],
           "classification": "DEVELOPMENT_ONLY_NO_RESERVED_CONFIRMATION_NO_WINNER_PRESUPPOSED",
           "architectures": {a: E.ARCHITECTURES[a] for a in archs}, "archs": list(archs), "reference": "A",
           "fusions": {f: E.FUSIONS[f] for f in readout_arms}, "readout_arms": list(readout_arms),
           "levels": list(levels), "replicates": list(replicates), "random_assignments": int(random_assignments), "h3_level": int(h3_level),
           "r_values": list(r_values), "readout_controls_r": list(readout_controls_r), "donor_sensitivity": donor_sensitivity, "diagnostic": diagnostic,
           "host_weights": host_weights, "only_hypotheses": list(only_hypotheses) if only_hypotheses else None,
           "p": E.P_VARS, "window": E.WINDOW, "horizon": E.HORIZON, "n_total": E.N_TOTAL, "training": training,
           "hosts": list(hosts), "successor_of": successor_of, "successor_reason": reason,
           "common": {"core": "Conv1D(16,3,causal,ELU) for sequence fusions; Dense(32,ELU) for summary fusions", "head": "Dense(p) + persistence skip",
                      "activation": E.ACTIVATION, "activation_note": "ELU is a DEV decision recorded after the ML07 diagnostic of arch B; "
                                                                      "it is applied to every architecture as a common factor, not as a transferred result",
                      "width": 16, "adapter": "TimeDistributed Dense(8) linear (identity for ARCH-0)",
                      "reach": {a: {"branch": E.branch_reach(a, E.WINDOW), **{f: E.support_reach(a, f, E.WINDOW) for f in readout_arms}} for a in archs},
                      "reach_note": "declared and tested by perturbation (tests/test_df_mod_e0_arch.py); the core sees 3 positions, so ARCH-0 and "
                                    "ARCH-A reach the planted lag tau = 3 (ARCH-0: x[t-2] = a[t+1-tau]) — they do not lose for a blind core; "
                                    "the period 24 is beyond their reach by construction (that is what the comparison measures)",
                      "parameters_differ": "widths are common; parameter counts differ by architecture and are recorded per cell; equal counts "
                                           "would not prove functional equivalence and are not imposed"},
           "estimands": {"H2_within_arch": "e_a(h) = MASE(profiles) - mean MASE(random) per replicate, slope over h, per architecture a",
                         "H3_within_arch": "d_a,r = MASE(sequence-fusion arms) - MASE(summary-fusion arms) per replicate, averaged over readouts; gamma_a = d_1 - d_0",
                         "READOUT": "rho_a,r = MASE(last readouts) - MASE(pooled readouts) per replicate, averaged over fusions: separates the readout from "
                                    "preserving history until fusion (the 2 x 2 fusion x readout within each (arch, r, replicate) shares one donor)",
                         "DONOR": "delta_a = d_a,1(donor summary) - d_a,1(donor sequence) on the bounded subset",
                         "ARCH": "differences of the above effects and of raw MASE / cost between architectures, descriptive; no winner presupposed",
                         "DX": "MASE of the profiles arm vs naive / linear / oracle under trend_event per architecture: adequacy, no effect"},
           "acceptance_before_effects": ["every architecture beats the naive forecaster on validation at h = 3, r = 1 (receiver adequacy), "
                                         "else its effects are reported as NOT_INTERPRETABLE", "reach tests pass", "cost pilots complete without test access"],
           "budget": BUDGET, "review_table": [{"id": i, "obligation": o, "evidence": e, "status": "NOT_TESTED"} for i, o, e in D1.REVIEW],
           "design_sha256": ""}
    doc["pilots"] = [] if only_hypotheses else pilots(doc)     # a hypothesis-only successor inherits its parent's measured pilot costs
    doc["cells"] = cells(doc)
    doc["cells_total"] = len(doc["cells"])
    doc["cells_by_role"] = {r: sum(1 for c in doc["cells"] if c["host_role"] == r) for r in hosts}
    doc["design_sha256"] = E.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def pilots(design: dict) -> list:
    """Cost pilots per architecture (no test access, small update allowance), on the coordinator."""
    out = []
    for a in design["archs"]:
        out.append({"cell_id": f"pilot__{a}__H2_profiles", "hypothesis": "H2", "level": 3, "r": 1, "seed": 1, "arm": "profiles", "arch": a,
                    "campaign": "-mod-e0-cost-pilot", "host_role": "COORDINATOR"})
        out.append({"cell_id": f"pilot__{a}__H3_extractor", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "extractor", "arch": a,
                    "campaign": "-mod-e0-cost-pilot", "host_role": "COORDINATOR"})
        out.append({"cell_id": f"pilot__{a}__H3_sequence", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "sequence", "arch": a,
                    "campaign": "-mod-e0-cost-pilot-arm", "depends_on": f"pilot__{a}__H3_extractor", "host_role": "COORDINATOR"})
    return out


def cells(design: dict) -> list:
    """Deterministic enumeration; every dependency group (an H3 extractor with its arms) lives on
    one host; groups are dealt round-robin over the declared roles."""
    groups = []
    for a in design["archs"]:
        for h in design["levels"]:
            for seed in design["replicates"]:
                g = [{"cell_id": f"H2__h{h}__s{seed}__{a}__profiles", "hypothesis": "H2", "level": h, "r": 1, "seed": seed, "arm": "profiles", "arch": a}]
                for k in range(design["random_assignments"]):
                    g.append({"cell_id": f"H2__h{h}__s{seed}__{a}__random_{k}", "hypothesis": "H2", "level": h, "r": 1, "seed": seed, "arm": f"random_{k}", "arch": a})
                groups.append(g)
        for r in design["r_values"]:
            for seed in design["replicates"]:
                ext = f"H3__r{r}__s{seed}__{a}__extractor"
                g = [{"cell_id": ext, "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed, "arm": "extractor", "arch": a}]
                arms = design["readout_arms"] if r in (design.get("readout_controls_r") or design["r_values"]) else \
                    [x for x in design["readout_arms"] if x in ("sequence", "summary")]
                for arm in arms:
                    g.append({"cell_id": f"H3__r{r}__s{seed}__{a}__{arm}", "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed, "arm": arm,
                              "arch": a, "depends_on": ext, "donor": "sequence"})
                groups.append(g)
    ds = design.get("donor_sensitivity") or {}
    for a in ds.get("archs") or []:
        for r in ds.get("r") or []:
            for seed in design["replicates"]:
                ext = f"H3__r{r}__s{seed}__{a}__extractor_summary"
                g = [{"cell_id": ext, "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed, "arm": "extractor_summary", "arch": a}]
                for arm in ds.get("arms") or []:
                    g.append({"cell_id": f"H3__r{r}__s{seed}__{a}__{arm}__dsum", "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed,
                              "arm": arm, "arch": a, "depends_on": ext, "donor": "summary"})
                groups.append(g)
    dx = design.get("diagnostic") or {}
    if dx:
        for a in design["archs"]:
            for seed in dx.get("seeds") or []:
                g = [{"cell_id": f"DX__{dx['condition']}__s{seed}__{a}__{arm}", "hypothesis": "DX", "level": dx["level"], "r": dx["r"], "seed": seed,
                      "arm": arm, "arch": a, "diagnostic": dx["condition"]} for arm in dx.get("arms") or ["profiles"]]
                groups.append(g)
    only = design.get("only_hypotheses")
    if only:                                                  # a successor that re-runs one hypothesis's cells (e.g. DX after a code defect)
        groups = [g for g in groups if all(c["hypothesis"] in only for c in g)]
    roles = list(design["hosts"])
    weights = design.get("host_weights") or {r: 1 for r in roles}
    load = {r: 0.0 for r in roles}
    out = []
    for g in groups:                                          # weighted dealing: the least loaded role per unit of capacity takes the group
        role = min(roles, key=lambda r: (load[r] / max(float(weights.get(r, 1)), 1e-9), roles.index(r)))
        load[role] += len(g)
        for c in g:
            out.append({**c, "host_role": role})
    return out


def readout_completion(parent: dict, parent_run_id: str, parent_root: str, reason: str, code_equivalence: dict | None = None) -> dict:
    """RP22: the successor that adds ONLY the missing r = 0 readout controls (sequence_gap, summary_last) for
    every architecture and replicate of the parent stage, consuming the parent's ORIGINAL r = 0 extractors as
    inherited donors (never re-trained), with the same data, seeds, training rule and parameters. Inherited
    cells are enumerated per cell with their parent identity; they are not campaign units here."""
    rs0 = [r for r in parent["r_values"] if r not in set(parent.get("readout_controls_r") or parent["r_values"])]
    if not rs0:
        raise ValueError("the parent already has the readout controls at every r")
    cells, inherited = [], []
    for a in parent["archs"]:
        for r in rs0:
            for seed in parent["replicates"]:
                ext = f"H3__r{r}__s{seed}__{a}__extractor"
                inherited.append({"cell_id": ext, "hypothesis": "H3", "level": parent["h3_level"], "r": r, "seed": seed, "arm": "extractor", "arch": a,
                                  "inherited_from": {"run_id": parent_run_id, "design_sha256": parent["design_sha256"], "root": parent_root, "cell_id": ext}})
                for arm in ("sequence_gap", "summary_last"):
                    cells.append({"cell_id": f"H3__r{r}__s{seed}__{a}__{arm}", "hypothesis": "H3", "level": parent["h3_level"], "r": r, "seed": seed, "arm": arm,
                                  "arch": a, "depends_on": ext, "donor": "sequence", "host_role": "COORDINATOR"})
    doc = {k: v for k, v in parent.items() if k not in ("cells", "cells_total", "cells_by_role", "pilots", "design_sha256", "successor_of", "successor_reason")}
    doc.update({"schema": DESIGN_SCHEMA, "kind": "READOUT_COMPLETION", "successor_of": parent["design_sha256"], "successor_reason": reason,
                "parent_run_id": parent_run_id, "parent_root": parent_root, "hosts": ["COORDINATOR"], "host_weights": {"COORDINATOR": 1},
                "readout_controls_r": sorted(set(parent.get("readout_controls_r") or parent["r_values"]) | set(rs0)),
                "only_hypotheses": ["H3"], "pilots": [], "inherited": inherited, "cells": cells, "cells_total": len(cells),
                "cells_by_role": {"COORDINATOR": len(cells)}, "code_equivalence": code_equivalence,
                "decision": "taken AFTER seeing the stage's results: sequential development, not a predeclared factorial nor confirmation",
                "design_sha256": ""})
    doc["design_sha256"] = E.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def pilot_key_for(cell: dict) -> str:
    """Which cost pilot projects a cell's cost."""
    a = cell["arch"]
    if cell["hypothesis"] == "H3":
        return f"pilot__{a}__H3_extractor" if cell["arm"] in ("extractor", "extractor_summary") else f"pilot__{a}__H3_sequence"
    return f"pilot__{a}__H2_profiles"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", default="0,1,2,3")
    parser.add_argument("--replicates", default="1,2,3")
    parser.add_argument("--random-assignments", type=int, default=3)
    parser.add_argument("--max-updates", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--readout-controls-r", default="0,1")
    parser.add_argument("--host-weights", default=None, help="e.g. COORDINATOR=4,WORKER_A=6,WORKER_B=2")
    parser.add_argument("--diagnostic-seeds", default="1,2")
    parser.add_argument("--only-hypotheses", default=None, help="e.g. DX: a successor with that hypothesis's cells only")
    parser.add_argument("--hosts", default="COORDINATOR,WORKER_A,WORKER_B")
    parser.add_argument("--readout-completion-of", type=Path, default=None, help="RP22: parent stage DESIGN.json; emits the 16-cell successor")
    parser.add_argument("--parent-run-id", default=None)
    parser.add_argument("--parent-root", default=None)
    parser.add_argument("--code-equivalence", type=Path, default=None)
    parser.add_argument("--successor-of", default=None)
    parser.add_argument("--reason", default=None)
    parser.add_argument("--stage-note", default=None)
    args = parser.parse_args(argv)
    if args.readout_completion_of is not None:
        parent = json.loads(args.readout_completion_of.read_text())
        ce = json.loads(args.code_equivalence.read_text()) if args.code_equivalence else None
        doc = readout_completion(parent, args.parent_run_id, args.parent_root, args.reason or "readout completion", ce)
        if args.out.exists():
            raise SystemExit(f"REFUSED: {args.out} exists; a design is never written over")
        args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"design_sha256": doc["design_sha256"], "cells": doc["cells_total"], "inherited": len(doc["inherited"])}, indent=1))
        return 0
    weights = dict(kv.split("=") for kv in args.host_weights.split(",")) if args.host_weights else None
    weights = {k: float(v) for k, v in weights.items()} if weights else None
    diagnostic = {"condition": "trend_event", "level": 3, "r": 1, "seeds": [int(v) for v in args.diagnostic_seeds.split(",")], "arms": ["profiles"],
                  "why": "absent from the executed pilot: a deterministic drift and two level shifts (one in train, one in test) test receiver "
                         "adequacy and the corrected trend descriptor; no effect is estimated on it"}
    doc = build(levels=[int(v) for v in args.levels.split(",")], replicates=[int(v) for v in args.replicates.split(",")],
                random_assignments=args.random_assignments, max_updates=args.max_updates, patience=args.patience,
                readout_controls_r=[int(v) for v in args.readout_controls_r.split(",")], host_weights=weights, diagnostic=diagnostic,
                hosts=args.hosts.split(","), successor_of=args.successor_of, reason=args.reason, stage_note=args.stage_note,
                only_hypotheses=args.only_hypotheses.split(",") if args.only_hypotheses else None)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; a design is never written over")
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"design_sha256": doc["design_sha256"], "cells": doc["cells_total"], "pilots": len(doc["pilots"]), "by_role": doc["cells_by_role"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
