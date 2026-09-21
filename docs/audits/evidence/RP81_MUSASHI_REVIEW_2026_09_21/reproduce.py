"""Audit bd6fcf4. Originals/services untouched; mutations only in temporary copies.

Warehouse stand-ins retain the published accepted terminal payloads, unchanged during each probe.
They are not live warehouse queries. No fit or financial-resource read is performed.
"""
import argparse
import contextlib
import copy
from dataclasses import fields
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, a.repo / "tools" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


K, B, T, F, H = [load(n) for n in ("df_e1_block", "df_benchmark_contract", "df_closure_table", "df_fin_runner", "df_e1_huber")]
out = {}
with tempfile.TemporaryDirectory(prefix="rp81-audit-") as scratch:
    root = Path(scratch) / "block"
    root.mkdir()
    for name in ("DESIGN.json", "BLOCK_DATA.json", "BLOCK_DATA.npz", "DATA.json", "DATA.npz", "TERMINAL_RECEIPTS.json"):
        shutil.copy2(a.root / name, root / name)
    for name in ("attempts", "TERMINALS"):
        shutil.copytree(a.root / name, root / name)
    design = json.loads((root / "DESIGN.json").read_text())
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    warehouse = {}
    for unit, receipt in receipts.items():
        p = root / "TERMINALS" / f"{unit}.json"
        if p.is_file():
            terminal = json.loads(p.read_text())
            warehouse.setdefault(receipt["campaign_sha256"], {"current": {}})["current"][unit] = {
                **terminal, "terminal_sha256": receipt["terminal_sha256"]}
    def held(campaign):
        return copy.deepcopy(warehouse.get(campaign, {"current": {}}))
    C = load("df_mod_e0_close")
    C.warehouse_terminals = lambda url, token, campaign: held(campaign)
    token = Path(scratch) / "fixture-token"
    token.write_text("synthetic-not-a-credential")
    args = SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="synthetic://unchanged-payloads")
    with contextlib.redirect_stdout(io.StringIO()):
        before = K.close(args)
    unit = "modular_w60_s1"
    folder = root / "attempts" / unit
    with np.load(folder / "arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    original_arrays = (folder / "arrays.npz").read_bytes()
    original_record = (folder / "cell.json").read_text()
    arrays["pred"] = arrays["y"].copy()
    arrays["reload_pred"] = arrays["y"].copy()
    np.savez(folder / "arrays.npz", **arrays)
    rec = json.loads(original_record)
    rec["arrays_sha256"] = K.sha_file(folder / "arrays.npz")
    rec["scores"] = H.metrics(arrays["pred"], arrays["y"], arrays["naive"], rec["target_sd"])
    (folder / "cell.json").write_text(json.dumps(rec))
    with contextlib.redirect_stdout(io.StringIO()):
        after = K.close(args)
    strict = T.rows_from_run(root, label="fixture", registry=B.registry(), warehouse=held)
    bad = next(r for r in strict if r["unit"] == unit)
    out["block_closure_does_not_use_strict_table"] = {
        "before_verified": before["verified"], "after_verified": after["verified"],
        "before_mae_z": next(r["mae_z"] for r in before["rows"] if r["cell_id"] == unit),
        "after_mae_z": next(r["mae_z"] for r in after["rows"] if r["cell_id"] == unit),
        "strict_table_verified": bad["verified"], "strict_table_problems": bad["problems"]}
    (folder / "arrays.npz").write_bytes(original_arrays)
    (folder / "cell.json").write_text(original_record)

    data_bytes = (root / "DATA.npz").read_bytes()
    with np.load(root / "DATA.npz", allow_pickle=False) as z:
        changed_data = {key: z[key] for key in z.files}
    changed_data["scaler_sd"] = changed_data["scaler_sd"] * 10
    np.savez(root / "DATA.npz", **changed_data)
    scale_rows = T.rows_from_run(root, label="fixture", registry=B.registry(), warehouse=held)
    scale_row = next(r for r in scale_rows if r["unit"] == unit)
    out["unanchored_evaluation_scaler"] = {
        "original_mae_z": next(r["mae_z"] for r in before["rows"] if r["cell_id"] == unit),
        "rewritten_mae_z": scale_row["model_error_z"], "verified": scale_row["verified"],
        "problems": scale_row["problems"], "predictions_record_receipt_and_warehouse_unchanged": True}
    (root / "DATA.npz").write_bytes(data_bytes)

    # Relabel a registered measured arm as a never-trained model; leave old design digest and accepted payloads intact.
    forged = copy.deepcopy(design)
    for cell in forged["cells"]:
        if cell["arm"] == "gru_adapted_w60":
            cell["arm"] = "UNTRAINED_REFERENCE"
    (root / "DESIGN.json").write_text(json.dumps(forged))
    c = B.BenchmarkContract(**{f.name: design["benchmark_contract"][f.name] for f in fields(B.BenchmarkContract)
                               if f.name in design["benchmark_contract"]})
    result = B.reference_evidence(c, root, reference_arm="UNTRAINED_REFERENCE", warehouse=held)
    out["reference_design_relabel"] = {"state": result["state"], "reference_arm": result.get("reference_arm"),
                                      "design_digest_recomputed": False, "original_payloads_unchanged": True}

    # Actual candidate registry, all declared seeds present: A/C/D must not compete inside B.
    allocation = load("df_fin_task").candidate_allocation()
    candidates = [c for key in ("A_fixed_default", "B_equal_budget_lr", "C_decay_factor", "D_delta_factor") for c in allocation[key]]
    selection_root = Path(scratch) / "selection"
    cells = []
    for cand in candidates:
        for seed in (1, 2, 3):
            cell = {"cell_id": f"{cand['id']}_s{seed}", "candidate_id": cand["id"], "fold": 0, "seed": seed}
            cells.append(cell)
            folder = selection_root / "attempts" / cell["cell_id"]
            folder.mkdir(parents=True)
            value = 0.01 if cand["population"] == "D_delta_factor" else 0.1 if cand["population"] == "A_fixed_default" else 0.5
            (folder / "cell.json").write_text(json.dumps({"cell": cell, "candidate": cand,
                "scores": {"validation": {"mae_z": value}, "test": {"mae_z": value + 0.1}}}))
    sel = F.select(selection_root, {"cells": cells, "candidates": candidates, "seeds": [1, 2, 3],
        "folds": {"dev_weeks": 1}, "design_sha256": "synthetic-audit"})
    out["financial_populations_mixed"] = {fam: {k: sel["per_fold"][0][fam][k] for k in ("selected", "n_candidates_compared")}
                                            for fam in ("mae", "huber")}
    out["bootstrap_omits_present_isolated_week"] = F.block_bootstrap(np.array([-100., np.nan] + [1.] * 10), block_len=2, n_boot=100)

# Compare retained training prefixes without attributing the difference to a particular CPU/kernel.
base = a.repo / "docs/audits/evidence/d3_k5_20260917"
prefixes = []
for arm, old_run in {"modular_w60": "e1_block_dev_matched_v2", "gru_adapted_w60": "e1_block_dev_matched_v2",
                     "calendar": "e1_block_q1_calendar_v1", "randomised_calendar_control": "e1_block_q1_calendar_v1"}.items():
    for seed in (1, 2, 3):
        old = json.loads((base / "RP66/blocks" / old_run / "attempts" / f"{arm}_s{seed}" / "cell.json").read_text())
        new = json.loads((a.root / "attempts" / f"{arm}_s{seed}" / "cell.json").read_text())
        prefixes.append({"arm": arm, "seed": seed,
            "initial_weights_equal": old["initial_weights_sha256"] == new["initial_weights_sha256"],
            "max_common_prefix_validation_delta": max(abs(x["val_mae_scaled"] - y["val_mae_scaled"])
                for x, y in zip(old["training"]["events"], new["training"]["events"])),
            "old_mae_z": old["scores"]["mae_z"], "new_mae_z": new["scores"]["mae_z"]})
out["tier_comparison_training_prefixes"] = prefixes
a.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps(out, indent=2, allow_nan=False))
