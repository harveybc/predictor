"""Read-only audit of retained results; destructive probes use private temporary copies.

No training, financial bar reads, service mutation or live warehouse query.
Accepted payload stand-ins are fixed before each corruption.
"""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--financial-repo", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def load(name, path=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or a.repo / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


K, F, T, H, C = [load(n) for n in ("df_e1_block", "df_fin_runner", "df_closure_table", "df_e1_huber", "df_mod_e0_close")]
out = {}
with tempfile.TemporaryDirectory(prefix="rp89-audit-") as td:
    tmp = Path(td)
    root = tmp / "context"
    root.mkdir()
    for name in ("DESIGN.json", "BLOCK_DATA.json", "BLOCK_DATA.npz", "TERMINAL_RECEIPTS.json", "REPLAYS.json"):
        shutil.copy2(a.root / name, root / name)
    for name in ("attempts", "TERMINALS"):
        shutil.copytree(a.root / name, root / name)
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    held = {}
    for u, r in receipts.items():
        p = root / "TERMINALS" / f"{u}.json"
        if p.exists():
            held.setdefault(r["campaign_sha256"], {"current": {}})["current"][u] = {
                **json.loads(p.read_text()), "terminal_sha256": r["terminal_sha256"]}
    token = tmp / "token"
    token.write_text("synthetic-fixture-not-a-credential")
    args = SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="synthetic://fixed-payloads")
    with patch.object(C, "warehouse_terminals", lambda url, tok, c: copy.deepcopy(held.get(c, {}))):
        with contextlib.redirect_stdout(io.StringIO()):
            before = K.close(args)
        u = "daily_lag_s1"
        weights = root / "attempts" / u / "weights.weights.h5"
        weights.write_bytes(b"not a checkpoint")
        with contextlib.redirect_stdout(io.StringIO()):
            after = K.close(args)
        out["corrupt_checkpoint_cached_replay"] = {
            "before_verified": before["verified"], "after_verified": after["verified"],
            "checkpoint_replaced_with": "not a checkpoint", "replay": after["replays"][u],
            "accepted_payloads_unchanged": True, "problems": after["problems"]}

    # A preparation metadata edit removes a previously accepted fold from closure.
    fixtures = load("audit_fin_fixtures", a.repo / "tests/test_fin_loss_opt_acceptance.py")
    bars = fixtures._bars()
    design = fixtures._design(bars)
    fin = tmp / "financial"
    rec = F.prepare(design, fin, frame=bars)
    (fin / "DESIGN.json").write_text(json.dumps(design))
    data, _ = F.load_data(fin, design)
    terminals, rr = {}, {}
    def accepted(u, artifacts, tags):
        rr[u] = {"campaign_sha256": f"fixture-{u}", "terminal_sha256": f"terminal-{u}"}
        terminals[f"fixture-{u}"] = {"current": {u: {"status": "COMPLETED", "terminal_sha256": f"terminal-{u}",
            "config_sha256": design["design_sha256"], "tags": tags, "artifacts": artifacts}}}
    def artifact(role, p):
        return {"role": role, "sha256": F.sha_file(p), "bytes": p.stat().st_size}
    accepted("prepare", [artifact("data", fin / "FIN_DATA.npz"), artifact("record", fin / "FIN_DATA.json")], {})
    candidates = {c["id"]: c for c in design["candidates"]}
    for cell in design["cells"]:
        u, k = cell["cell_id"], cell["fold"]
        folder = fin / "attempts" / u
        folder.mkdir(parents=True)
        arr, scores = {}, {}
        sigma = float(data[f"f{k}_target_mean_sigma"][1])
        for split in ("validation", "test"):
            o, t = data[f"f{k}_{split}_origins"], data[f"f{k}_{split}_targets"]
            y, naive = data["y"][t], data["y"][o]
            pred = naive.copy()
            arr.update({f"{split}_origins": o, f"{split}_targets": t, f"{split}_y": y,
                        f"{split}_naive": naive, f"{split}_pred": pred, f"{split}_reload_pred": pred})
            scores[split] = H.metrics(pred, y, naive, sigma)
        np.savez(folder / "arrays.npz", **arr)
        record = {"cell": cell, "candidate": candidates[cell["candidate_id"]], "design_sha256": design["design_sha256"],
                  "sigma_train": sigma, "scores": scores}
        (folder / "cell.json").write_text(json.dumps(record))
        accepted(u, [artifact("predictions", folder / "arrays.npz"), artifact("record", folder / "cell.json")],
                 {"candidate": cell["candidate_id"], "fold": k, "seed": cell["seed"]})
    (fin / "TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": rr}))
    fargs = SimpleNamespace(root=fin, warehouse_token_file=token, warehouse_url="synthetic://fixed-financial")
    with patch.object(C, "warehouse_terminals", lambda url, tok, c: copy.deepcopy(terminals.get(c, {}))):
        before = F.close(fargs)
        rec["folds"][0]["status"] = "INSUFFICIENT_POPULATION"
        (fin / "FIN_DATA.json").write_text(json.dumps(rec))
        removed = []
        for cell in design["cells"]:
            if cell["fold"] == 0:
                (fin / "attempts" / cell["cell_id"] / "arrays.npz").unlink()
                removed.append(cell["cell_id"])
        after = F.close(fargs)
        out["unanchored_fold_metadata"] = {
            "before_verified": before["verified"], "after_verified": after["verified"], "removed_arrays": removed,
            "before_verified_units": before["verification"]["verified_units"],
            "after_verified_units": after["verification"]["verified_units"], "problems": after["problems"],
            "accepted_preparation_record_unchanged": True}

    # Actual parser function from the producer, executing on a fabricated HistData-format zip.
    producer = a.financial_repo / "_scripts/workers/stage13_omega_light_worker.py"
    tree = ast.parse(producer.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_histdata_zip")
    ns = {"pd": pd, "zipfile": zipfile, "Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(producer), "exec"), ns)
    zpath = tmp / "synthetic_histdata.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("DAT_ASCII_EURUSD_M1_202401.csv", "20240102 000000;1.10;1.12;1.09;1.11;0\n20240102 000100;1.11;1.13;1.10;1.12;0\n")
    parsed = ns["parse_histdata_zip"](zpath)
    out["producer_clock_and_columns"] = {
        "source_sha256": F.sha_file(producer), "source_label": "2024-01-02 00:00:00 EST fixed (UTC-05:00)",
        "parsed_label": str(parsed.datetime.iloc[0]), "expected_utc_under_source_spec": "2024-01-02 05:00:00+00:00",
        "columns_after_producer": list(parsed.columns), "pilot_required_columns": load("df_fin_task").INPUT_COLUMNS,
        "source_spec": "https://www.histdata.com/f-a-q/data-files-detailed-specification/",
        "scope": "Synthetic zip; current producer behavior, not proof of historical file lineage"}

    # Cost parser reaches a typed refusal before any model fit, even on aware synthetic bars.
    cost = F.seal_cost_pilot(lake="fixture", resource="fixture.parquet", time_column="datetime", holdout="2025-01-01",
                            range_from="2024-01-01", range_to="2024-05-19", dev_start="2024-06-24", contract=fixtures._contract())
    aware = bars.copy()
    aware["datetime"] = aware.datetime.dt.tz_localize("UTC")
    croot = tmp / "cost"
    croot.mkdir()
    (croot / "DESIGN.json").write_text(json.dumps(cost))
    refusal = F.cost_pilot(cost, croot, frame=aware)
    calls = []
    class FakeGov:
        def report_terminal(self, root, unit, terminal, **kwargs):
            calls.append(terminal)
            return {"flushed": {"sent": 0, "pending": 1, "failures": ["fixture destination unavailable"]}}
    U = load("df_utility_run")
    with patch.object(F, "acquire", lambda *args: None), patch.object(F, "governance_modules", lambda: (FakeGov(), U)), \
         patch.object(F, "cost_pilot", lambda *args: refusal), contextlib.redirect_stdout(io.StringIO()):
        exit_code = F.main(["cost-pilot", "--root", str(croot), "--api-key-file", str(token)])
    out["cost_refusal_reported_completed"] = {"actual_cost_parser_result": refusal["status"], "parser_reason": refusal.get("reason"),
        "main_exit_code": exit_code, "reported_terminal_status": calls[0]["status"], "reported_terminal_reason": calls[0]["reason"],
        "outbox_pending": 1, "fits_executed": 0}

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps(out, indent=2, allow_nan=False))
