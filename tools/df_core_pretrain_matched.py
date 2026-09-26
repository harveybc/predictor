#!/usr/bin/env python3
"""The budget-matched re-contrast — item two of the MOD-CORE-PRETRAIN block.

RP63 compared three arms and moved three things at once: the trained loss, the early-stopping monitor
(which IS the trained loss, so it moved with it), and — as a consequence — the number of optimiser
updates each arm was given. The retained totals are ``core_mae`` 11 762, ``tcn_mse`` 11 762 and
``core_mse`` 10 270: the winning arm ran 14.5% longer than the arm it beat. Both asymmetries come from
one line, repaired in ``df_e1_phase1._fit``; this runner is that repair executed.

What is held fixed here that was not fixed there:

    the monitor         ``val_mae`` in every arm — the measure the question is judged by. The trained
                        loss no longer chooses the stopping epoch or the restored checkpoint
    the budget          every cell runs EXACTLY the ceiling in optimiser updates. Not an equal
                        ceiling, which RP63 already had: an equal count
    everything else     the same prepared DATA by digest, the same train and evaluation origins, the
                        same labels, the same scaler, the same batch, the same learning rate, the same
                        three seeds — inherited unchanged from the sealed phase-1 design

What is NOT claimed. This is a LOCAL run. No governance campaign is registered, no delivery is
acquired and no terminal is reported, because this round may not start or contact a service; the
prepared DATA is consumed from the retained phase-1 root by digest instead. The run therefore carries
``custody: UNGOVERNED_LOCAL_RECONTRAST`` in its own report and in every row of its closure table, and
it is a DIAGNOSTIC re-measurement, not a governed result. That is stated here rather than implied by
an absent field.

    python tools/df_core_pretrain_matched.py --seal DESIGN.json
    python tools/df_core_pretrain_matched.py --design DESIGN.json --root ROOT --cells core_mae_s1 ...
    python tools/df_core_pretrain_matched.py --report --root ROOT --out CONTRAST.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA = "df_core_pretrain_matched_report.v1"
PHASE1_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_phase1_v1b"
CUSTODY = "UNGOVERNED_LOCAL_RECONTRAST"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"_matched_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def seal(source_run: Path = PHASE1_ROOT) -> dict:
    """The matched design: the sealed phase-1 design with the protocol repaired and declared.

    The identity changes, and it must: a run whose monitor and budget rule differ is not the same
    experiment. The parent design's digest is carried so the lineage is checkable.
    """
    P = _module("df_e1_phase1")
    E = _module("df_mod_e0")
    parent = json.loads((Path(source_run) / "DESIGN.json").read_text())
    design = P.seal(Path(parent["source_run"]["root"]), monitor=P.FIXED_MONITOR,
                    budget_match=P.BUDGET_FIXED)
    design.pop("design_sha256", None)
    design["what_this_is"] = ("RP63's recipe and architecture contrast, RE-RUN with the arms matched on "
                              "optimiser updates and one arm-independent monitor; the measurement the "
                              "MOD-CORE-PRETRAIN block names as its second prerequisite")
    design["repairs"] = {
        "parent_design_sha256": parent["design_sha256"],
        "parent_monitor": (parent.get("training") or {}).get("monitor"),
        "parent_total_updates_by_arm": {"core_mae": 11762, "core_mse": 10270, "tcn_mse": 11762},
        "defect_1": "the monitor was the arm's own trained loss, so it chose the stopping epoch",
        "defect_2": "the monitor also chose the restored checkpoint, on a different curve per arm",
        "consequence": "the arms were given different optimiser-update counts and the winner got more",
        "repair": "one fixed monitor (val_mae) for every arm, and a fixed update count for every cell"}
    design["custody"] = {
        "class": CUSTODY,
        "why": ("this round may not start or contact a governance or warehouse service, so no campaign "
                "is registered and no terminal is reported; the prepared DATA is consumed from the "
                "retained phase-1 root by digest"),
        "what_this_forfeits": ["no accepted delivery precedes the fits",
                               "no accepted terminal anchors the metrics",
                               "the rows of its closure table can never read `verified`"],
        "what_it_keeps": ["the same prepared bytes, checked by digest before any fit",
                          "metrics recomputed in float64 from the saved arrays",
                          "a fresh-process replay of every cell from its saved weights"]}
    design["source_run"]["phase1_root"] = str(Path(source_run))
    design["source_run"]["data_sha256_expected"] = parent["source_run"]["data_sha256"]
    design["design_sha256"] = E.sha_obj(design)
    return design


def _data(root: Path, design: dict) -> dict:
    P = _module("df_e1_phase1")
    src = Path(design["source_run"]["phase1_root"]) / "DATA.npz"
    dst = root / "DATA.npz"
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    digest = P.sha_file(dst)
    if digest != design["source_run"]["data_sha256_expected"]:
        raise SystemExit(f"REFUSED: DATA.npz is {digest}, the sealed design names "
                         f"{design['source_run']['data_sha256_expected']}")
    with np.load(dst, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def run_cells(design: dict, root: Path, cell_ids: list) -> dict:
    """Fit the named cells, each to exactly the ceiling, and replay each from its own saved weights."""
    P = _module("df_e1_phase1")
    PI = _module("df_e1_pilot")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "DESIGN.json").exists():
        (root / "DESIGN.json").write_text(json.dumps(design, indent=1))
    data = _data(root, design)
    protocol = P.resolve_protocol(design)
    if not (protocol["monitor_is_arm_independent"] and protocol["budget_is_matched_by_construction"]):
        raise SystemExit("REFUSED: this design does not fix the monitor AND the budget; it cannot "
                         "produce a matched contrast")
    done = []
    for cid in cell_ids:
        cell = next(c for c in design["cells"] if c["cell_id"] == cid)
        out = root / "attempts" / cid
        if (out / "cell.json").is_file():
            done.append({"cell_id": cid, "state": "ALREADY_PRESENT"})
            continue
        rec = P.run_cell(design, data, cell, out, max_updates=design["training"]["max_updates"])
        # the run's own verification standard: a fresh graph reloads the weights and regathers windows
        replay = _replay(design, data, cell, out)
        rec["replay"] = replay
        (out / "cell.json").write_text(json.dumps(rec, indent=1, default=str))
        done.append({"cell_id": cid, "state": "FITTED", "mae": rec["scores"]["validation"]["model"]["mae_mean"],
                     "updates": rec["training"]["updates"], "replay_max_error": replay["max_abs_error"]})
    return {"root": str(root), "cells": done, "protocol": protocol}


def _replay(design: dict, data: dict, cell: dict, out: Path) -> dict:
    """Reload the saved weights into a freshly built graph and re-predict every evaluation row."""
    P = _module("df_e1_phase1")
    PI = _module("df_e1_pilot")
    W, h = int(data["window"][0]), int(data["horizon"][0])
    j = int(data["target_channel"][0])
    Dataset = PI._dataset_class()
    va = Dataset(data["Xs"], data["Y"], data["eval_origins"], W, h, j, design["training"]["batch"],
                 shuffle=False, seed=int(cell["seed"]),
                 scaler_mean=data["scaler_mean"], scaler_sd=data["scaler_sd"])
    model = P._arm_model(cell["arm"], design, data, int(cell["seed"]))
    model.load_weights(out / "weights.weights.h5")
    m, s = float(data["scaler_mean"][j]), float(data["scaler_sd"][j])
    pred = PI._predict(model, va).reshape(-1) * s + m
    with np.load(out / "arrays.npz", allow_pickle=False) as z:
        stored = np.asarray(z["validation_pred"], dtype=np.float64).reshape(-1)
    err = float(np.max(np.abs(pred.astype(np.float64) - stored)))
    return {"rows": int(stored.size), "max_abs_error": err, "tolerance": 1e-5,
            "within_tolerance": err <= 1e-5, "fresh_graph": True}


def report(root: Path, *, resolution: dict | None = None) -> dict:
    """The matched contrast: every arm's mean, the budget audit, the paired differences, the verdict.

    Every metric is recomputed in float64 from the cells' own arrays, and the contrast is REFUSED if
    the arms are not budget-matched — the rule this round exists to add.
    """
    P = _module("df_e1_phase1")
    R = _module("df_core_pretrain_resolution")
    root = Path(root)
    design = json.loads((root / "DESIGN.json").read_text())
    cells, per_arm = {}, {}
    for att in sorted((root / "attempts").iterdir()):
        if not (att / "cell.json").is_file():
            continue
        rec = json.loads((att / "cell.json").read_text())
        m = R.cell_measurement(root, att.name)
        m["arm"] = rec["arm"]
        m["replay"] = rec.get("replay")
        m["monitor"] = (rec["training"] or {}).get("monitor")
        m["budget_match"] = (rec["training"] or {}).get("budget_match")
        cells[att.name] = m
        per_arm.setdefault(rec["arm"], []).append(m)
    audit = P.require_budget_match({k: {"arm": v["arm"], "updates": v["updates"]} for k, v in cells.items()})
    monitors = {v["monitor"] for v in cells.values()}
    if len(monitors) != 1:
        raise P.ProtocolRefusal(f"REFUSED: more than one monitor across the arms: {sorted(monitors)}")
    naive = {k: R.naive_on_the_same_rows(root, k) for k in cells}
    naive_vals = {round(v["naive_kW"], 12) for v in naive.values()}
    if len(naive_vals) != 1:
        raise P.ProtocolRefusal(f"REFUSED: the naive differs across cells claiming the same rows: {sorted(naive_vals)}")
    vals = {a: sorted([c["mae_kW"] for c in sorted(cs, key=lambda x: x["seed"])]) for a, cs in per_arm.items()}
    by_seed = {a: [c["mae_kW"] for c in sorted(cs, key=lambda x: x["seed"])] for a, cs in per_arm.items()}
    contrasts = []
    for a, b, what in (("core_mae", "core_mse", "the recipe contrast, now budget-matched"),
                       ("tcn_mse", "core_mse", "the reference block against ours, now budget-matched"),
                       ("core_mae", "tcn_mse", "our MAE recipe against the reference block")):
        if a in by_seed and b in by_seed and len(by_seed[a]) == len(by_seed[b]) >= 2:
            contrasts.append(R.paired_contrast(by_seed[a], by_seed[b], f"{a}-{b}", what))
    out = {"schema": SCHEMA, "at": now_iso(), "host_kind": "local", "custody": CUSTODY,
           "design_sha256": design["design_sha256"], "repairs": design["repairs"],
           "monitor": sorted(monitors)[0], "budget_match": design["training"]["budget_match"],
           "budget_audit": audit,
           "cells": cells, "naive_same_rows_kW": sorted(naive_vals)[0],
           "evaluation_rows": sorted({c["n_rows"] for c in cells.values()})[0],
           "arm_means_kW": {a: float(np.mean(v)) for a, v in by_seed.items()},
           "arm_sds_kW": {a: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for a, v in by_seed.items()},
           "arm_mae_by_seed_kW": by_seed,
           "replay": {k: v["replay"] for k, v in cells.items()},
           "all_replays_within_tolerance": all((v.get("replay") or {}).get("within_tolerance") for v in cells.values()),
           "contrasts": contrasts,
           "cpu_seconds": sum(float(json.loads((root / "attempts" / k / "cell.json").read_text())
                                    ["cost"]["cpu_seconds_process"]) for k in cells)}
    if resolution:
        r = resolution["resolution"]
        prev = {c["name"]: c for c in resolution["contrasts"]}
        out["against_the_resolution"] = []
        for c in contrasts:
            before = prev.get(c["name"])
            eff = abs(c["mean_difference_kW"])
            out["against_the_resolution"].append({
                "contrast": c["name"],
                "unmatched_effect_kW": None if not before else before["mean_difference_kW"],
                "matched_effect_kW": c["mean_difference_kW"],
                "change_kW": None if not before else c["mean_difference_kW"] - before["mean_difference_kW"],
                "resolution_kW": r["resolution_kW"],
                "state_against_the_resolution": ("ABOVE_THE_RESOLUTION" if eff >= r["resolution_ci95_kW"][1]
                                                 else "AT_THE_RESOLUTION_BOUNDARY" if eff >= r["resolution_kW"]
                                                 else "BELOW_THE_RESOLUTION"),
                "matched_ci95_kW": c["ci95_kW"], "matched_p_value": c["p_value"],
                "erased": None if not before else (abs(c["mean_difference_kW"]) < r["resolution_kW"]
                                                   and abs(before["mean_difference_kW"]) >= r["resolution_kW"])})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", type=Path)
    ap.add_argument("--design", type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--resolution", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    if a.seal:
        d = seal()
        a.seal.write_text(json.dumps(d, indent=1, default=str))
        print(json.dumps({"design_sha256": d["design_sha256"], "monitor": d["training"]["monitor"],
                          "budget_match": d["training"]["budget_match"],
                          "cells": [c["cell_id"] for c in d["cells"]]}, indent=1))
        return 0
    if a.report:
        res = json.loads(a.resolution.read_text()) if a.resolution else None
        doc = report(a.root, resolution=res)
        if a.out:
            a.out.write_text(json.dumps(doc, indent=1, default=str))
        print(json.dumps({"arm_means_kW": doc["arm_means_kW"], "budget_audit": doc["budget_audit"]["total_updates_by_arm"],
                          "matched": doc["budget_audit"]["matched_on_totals"],
                          "all_replays_within_tolerance": doc["all_replays_within_tolerance"],
                          "against_the_resolution": doc.get("against_the_resolution"),
                          "cpu_seconds": doc["cpu_seconds"]}, indent=1, default=str))
        return 0
    design = json.loads(a.design.read_text())
    E = _module("df_mod_e0")
    if E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]:
        raise SystemExit("REFUSED: the design does not re-derive its own digest")
    got = run_cells(design, a.root, a.cells or [c["cell_id"] for c in design["cells"]])
    print(json.dumps(got, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
