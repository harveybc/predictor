"""POST of Musashi's RP81 probes (docs/audits/evidence/RP81_MUSASHI_REVIEW_2026_09_21/reproduce.py) on the repaired tree.

His script is untouched; this harness repeats his steps and RECORDS the typed refusal where the repaired closure now
refuses (his script would stop at the exception). Mutations only on a temporary copy; warehouse stand-ins serve the
published accepted payloads; no service, fit or reserve read.
"""
import contextlib, copy, importlib.util, io, json, shutil, sys, tempfile, argparse
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, a.repo/"tools"/f"{name}.py"); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def refused(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return {"outcome": "RETURNED", "value": fn()}
    except BaseException as exc:
        return {"outcome": "REFUSED", "exception": type(exc).__name__, "message": str(exc)[:400]}


K, B, T, F, H = [load(n) for n in ("df_e1_block", "df_benchmark_contract", "df_closure_table", "df_fin_runner", "df_e1_huber")]
out = {"scope": "POST on the repaired tree; typed refusals recorded"}
with tempfile.TemporaryDirectory(prefix="rp82-post-") as scratch:
    root = Path(scratch)/"block"; root.mkdir()
    for name in ("DESIGN.json", "BLOCK_DATA.json", "BLOCK_DATA.npz", "DATA.json", "DATA.npz", "TERMINAL_RECEIPTS.json", "REPLAYS.json"):
        if (a.root/name).is_file():
            shutil.copy2(a.root/name, root/name)
    for name in ("attempts", "TERMINALS"):
        shutil.copytree(a.root/name, root/name)
    design = json.loads((root/"DESIGN.json").read_text())
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    warehouse = {}
    for unit, receipt in receipts.items():
        p = root/"TERMINALS"/f"{unit}.json"
        if p.is_file():
            terminal = json.loads(p.read_text())
            warehouse.setdefault(receipt["campaign_sha256"], {"current": {}})["current"][unit] = {**terminal, "terminal_sha256": receipt["terminal_sha256"], "config_sha256": design["design_sha256"]}
    held = lambda campaign: copy.deepcopy(warehouse.get(campaign, {"current": {}}))
    C = load("df_mod_e0_close"); C.warehouse_terminals = lambda url, token, campaign: held(campaign)
    token = Path(scratch)/"tok"; token.write_text("synthetic-not-a-credential")
    args = SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="synthetic://unchanged-payloads")
    before = refused(lambda: K.close(args))
    unit = "modular_w60_s1"; folder = root/"attempts"/unit
    with np.load(folder/"arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    original_arrays, original_record = (folder/"arrays.npz").read_bytes(), (folder/"cell.json").read_text()
    arrays["pred"] = arrays["y"].copy(); arrays["reload_pred"] = arrays["y"].copy(); np.savez(folder/"arrays.npz", **arrays)
    rec = json.loads(original_record); rec["arrays_sha256"] = K.sha_file(folder/"arrays.npz"); rec["scores"] = H.metrics(arrays["pred"], arrays["y"], arrays["naive"], rec["target_sd"])
    (folder/"cell.json").write_text(json.dumps(rec))
    after = refused(lambda: K.close(args))
    rep_after = json.loads((root/"REPORT.json").read_text())
    out["block_closure_does_not_use_strict_table"] = {"before": {"outcome": before["outcome"], "verified": before.get("value", {}).get("verified") if before["outcome"] == "RETURNED" else None},
                                                      "after": after, "report_after": {"verified": rep_after["verified"], "summary": rep_after["summary"], "paired": rep_after["paired"],
                                                                                        "problems": rep_after["problems"][:3]}}
    (folder/"arrays.npz").write_bytes(original_arrays); (folder/"cell.json").write_text(original_record)
    # tenfold sigma: only the prepared scaler changes (BLOCK_DATA is the preparation's own evidence; DATA.npz is unread)
    for name in ("DATA.npz", "BLOCK_DATA.npz"):
        data_bytes = (root/name).read_bytes()
        with np.load(root/name, allow_pickle=False) as z:
            changed = {key: z[key] for key in z.files}
        changed["scaler_sd"] = changed["scaler_sd"]*10; np.savez(root/name, **changed)
        v = T.verify_run(root, label="fixture", registry=B.registry(), warehouse=held)
        row = next(r for r in v["rows"] if r["unit"] == unit)
        out[f"unanchored_evaluation_scaler_via_{name}"] = {"rewritten_mae_z": row["model_error_z"], "verified": row["verified"], "problems": v["problems"][:3],
                                                            "preparation_custody": v["preparation_custody"]["class"], "denominator": v["denominator"]}
        (root/name).write_bytes(data_bytes)
    forged = copy.deepcopy(design)
    for cell in forged["cells"]:
        if cell["arm"] == "gru_adapted_w60":
            cell["arm"] = "UNTRAINED_REFERENCE"
    (root/"DESIGN.json").write_text(json.dumps(forged))
    c = B.BenchmarkContract(**{f.name: design["benchmark_contract"][f.name] for f in fields(B.BenchmarkContract) if f.name in design["benchmark_contract"]})
    out["reference_design_relabel"] = B.reference_evidence(c, root, reference_arm="UNTRAINED_REFERENCE", warehouse=held)
    forged.pop("design_sha256"); forged["design_sha256"] = K.sha_obj(forged); (root/"DESIGN.json").write_text(json.dumps(forged))
    out["reference_design_relabel_consistently_rehashed"] = B.reference_evidence(c, root, reference_arm="UNTRAINED_REFERENCE", warehouse=held)
    (root/"DESIGN.json").write_text(json.dumps(design))
    out["reference_genuine"] = {k: v for k, v in B.reference_evidence(c, root, reference_arm="gru_adapted_w60", warehouse=held).items() if k != "derived_mae_z"}
    allocation = load("df_fin_task").candidate_allocation()
    candidates = [c_ for key in ("A_fixed_default", "B_equal_budget_lr", "C_decay_factor", "D_delta_factor") for c_ in allocation[key]]
    sel_root = Path(scratch)/"selection"; cells = []
    for cand in candidates:
        for seed in (1, 2, 3):
            cell = {"cell_id": f"{cand['id']}_s{seed}", "candidate_id": cand["id"], "fold": 0, "seed": seed}; cells.append(cell)
            folder = sel_root/"attempts"/cell["cell_id"]; folder.mkdir(parents=True)
            value = 0.01 if cand["population"] == "D_delta_factor" else 0.1 if cand["population"] == "A_fixed_default" else 0.5
            (folder/"cell.json").write_text(json.dumps({"cell": cell, "candidate": cand, "scores": {"validation": {"mae_z": value}, "test": {"mae_z": value+0.1}}}))
    sel = F.select(sel_root, {"cells": cells, "candidates": candidates, "seeds": [1, 2, 3], "folds": {"dev_weeks": 1}, "design_sha256": "synthetic-audit"})
    out["financial_populations_mixed"] = {"B_strata": {k: {kk: v[kk] for kk in ("selected", "n_candidates_compared", "lrs_compared")} for k, v in sel["per_fold"][0]["B_equal_budget_lr"].items()},
                                          "A_reported_not_selected": sorted(sel["per_fold"][0]["A_fixed_default"]), "CD_contrasts": len(sel["per_fold"][0]["CD_sensitivity_contrasts"]),
                                          "consumed": sel["consumed"]["by_population_complete_configs"]}
    out["bootstrap_omits_present_isolated_week"] = F.block_bootstrap(np.array([-100., np.nan]+[1.]*10), block_len=2, n_boot=100)
a.out.write_text(json.dumps(out, indent=2, allow_nan=False, default=str)+"\n")
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, (dict, list)) else "...") for kk, vv in v.items()}) for k, v in out.items()}, indent=1, default=str)[:3000])
