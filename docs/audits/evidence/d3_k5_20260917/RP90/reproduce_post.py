"""POST of Musashi's RP89 probes (docs/audits/evidence/RP89_MUSASHI_REVIEW_2026_09_21/reproduce.py) on the repaired tree (RP90).

His script is untouched; this harness repeats his four steps and RECORDS the typed refusal where the repaired code now refuses
(his script would stop at the exception). Mutations only on temporary copies / fabricated data; warehouse stand-ins serve the
published accepted payloads; no service operation, no fit on real data, no reserve read.
"""
import argparse, ast, contextlib, copy, importlib.util, io, json, shutil, sys, tempfile, zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--financial-repo", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def load(name, path=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or a.repo/"tools"/f"{name}.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def refused(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return {"outcome": "RETURNED", "value": fn()}
    except BaseException as exc:
        return {"outcome": "REFUSED", "exception": type(exc).__name__, "message": str(exc)[:500]}


K, F, T, H, C = [load(n) for n in ("df_e1_block", "df_fin_runner", "df_closure_table", "df_e1_huber", "df_mod_e0_close")]
out = {"scope": "POST on the repaired tree (RP90); typed refusals recorded; Musashi's steps repeated, his script untouched"}
with tempfile.TemporaryDirectory(prefix="rp90-post-") as td:
    tmp = Path(td); root = tmp/"context"; root.mkdir()
    for name in ("DESIGN.json", "BLOCK_DATA.json", "BLOCK_DATA.npz", "TERMINAL_RECEIPTS.json", "REPLAYS.json"):
        shutil.copy2(a.root/name, root/name)
    for name in ("attempts", "TERMINALS"):
        shutil.copytree(a.root/name, root/name)
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    held = {}
    for u, r in receipts.items():
        p = root/"TERMINALS"/f"{u}.json"
        if p.exists():
            held.setdefault(r["campaign_sha256"], {"current": {}})["current"][u] = {**json.loads(p.read_text()), "terminal_sha256": r["terminal_sha256"]}
    token = tmp/"token"; token.write_text("synthetic-fixture-not-a-credential")
    args = SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="synthetic://fixed-payloads")
    with patch.object(C, "warehouse_terminals", lambda url, tok, c: copy.deepcopy(held.get(c, {}))):
        before = refused(lambda: K.close(args))
        u = "daily_lag_s1"
        weights = root/"attempts"/u/"weights.weights.h5"
        original_weights_sha = K.sha_file(weights)
        weights.write_bytes(b"not a checkpoint")
        after = refused(lambda: K.close(args))
        rep = json.loads((root/"REPORT.json").read_text())
        out["corrupt_checkpoint_cached_replay"] = {
            "before": {"outcome": before["outcome"], "verified": before.get("value", {}).get("verified") if before["outcome"] == "RETURNED" else None,
                       "replay_of_unit": (before.get("value", {}).get("replays") or {}).get(u) if before["outcome"] == "RETURNED" else None},
            "after": {"outcome": after["outcome"], "exception": after.get("exception"), "message": after.get("message"),
                      "report_verified": rep["verified"], "summary": rep["summary"], "paired": rep["paired"],
                      "problems": rep["problems"], "replay_of_unit": rep["replays"].get(u)},
            "checkpoint_replaced_with": "not a checkpoint", "accepted_payloads_unchanged": True,
            "actual_bytes_sha256_now": K.sha_file(weights), "record_claimed_sha256": json.loads((root/"attempts"/u/"cell.json").read_text()).get("weights_file_sha256"),
            "original_bytes_sha256": original_weights_sha}

    # 2. a preparation metadata edit removes a previously accepted fold from closure
    fixtures = load("audit_fin_fixtures", a.repo/"tests/test_fin_loss_opt_acceptance.py")
    bars = fixtures._bars(); design = fixtures._design(bars)
    fin = tmp/"financial"
    rec = F.prepare(design, fin, frame=bars)
    (fin/"DESIGN.json").write_text(json.dumps(design))
    data, _ = F.load_data(fin, design)
    terminals, rr = {}, {}
    def accepted(u, artifacts, tags):
        rr[u] = {"campaign_sha256": f"fixture-{u}", "terminal_sha256": f"terminal-{u}"}
        terminals[f"fixture-{u}"] = {"current": {u: {"status": "COMPLETED", "terminal_sha256": f"terminal-{u}", "config_sha256": design["design_sha256"], "tags": tags, "artifacts": artifacts}}}
    def artifact(role, p):
        return {"role": role, "sha256": F.sha_file(p), "bytes": p.stat().st_size}
    accepted("prepare", [artifact("data", fin/"FIN_DATA.npz"), artifact("record", fin/"FIN_DATA.json")], {})
    candidates = {c["id"]: c for c in design["candidates"]}
    for cell in design["cells"]:
        u, k = cell["cell_id"], cell["fold"]; folder = fin/"attempts"/u; folder.mkdir(parents=True)
        arr, scores = {}, {}; sigma = float(data[f"f{k}_target_mean_sigma"][1])
        for split in ("validation", "test"):
            o, t = data[f"f{k}_{split}_origins"], data[f"f{k}_{split}_targets"]
            y, naive = data["y"][t], data["y"][o]; pred = naive.copy()
            arr.update({f"{split}_origins": o, f"{split}_targets": t, f"{split}_y": y, f"{split}_naive": naive, f"{split}_pred": pred, f"{split}_reload_pred": pred})
            scores[split] = H.metrics(pred, y, naive, sigma)
        np.savez(folder/"arrays.npz", **arr)
        (folder/"cell.json").write_text(json.dumps({"cell": cell, "candidate": candidates[cell["candidate_id"]], "design_sha256": design["design_sha256"], "sigma_train": sigma, "scores": scores}))
        accepted(u, [artifact("predictions", folder/"arrays.npz"), artifact("record", folder/"cell.json")], {"candidate": cell["candidate_id"], "fold": k, "seed": cell["seed"]})
    (fin/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": rr}))
    fargs = SimpleNamespace(root=fin, warehouse_token_file=token, warehouse_url="synthetic://fixed-financial")
    with patch.object(C, "warehouse_terminals", lambda url, tok, c: copy.deepcopy(terminals.get(c, {}))):
        before = refused(lambda: F.close(fargs))
        rec["folds"][0]["status"] = "INSUFFICIENT_POPULATION"; (fin/"FIN_DATA.json").write_text(json.dumps(rec))
        removed = []
        for cell in design["cells"]:
            if cell["fold"] == 0:
                (fin/"attempts"/cell["cell_id"]/"arrays.npz").unlink(); removed.append(cell["cell_id"])
        after = refused(lambda: F.close(fargs))
        rep = json.loads((fin/"REPORT.json").read_text())
        out["unanchored_fold_metadata"] = {
            "before": {"outcome": before["outcome"], "verified": before["value"]["verified"] if before["outcome"] == "RETURNED" else None,
                       "verified_units": before["value"]["verification"]["verified_units"] if before["outcome"] == "RETURNED" else None,
                       "required_folds_from": before["value"]["verification"].get("required_folds_from") if before["outcome"] == "RETURNED" else None},
            "after": {"outcome": after["outcome"], "exception": after.get("exception"), "message": after.get("message"), "report_verified": rep["verified"],
                      "verified_units": rep["verification"]["verified_units"], "preparation_custody": rep["verification"]["preparation_custody"],
                      "selection": rep["selection"], "problems": rep["problems"]},
            "removed_arrays": removed, "accepted_preparation_record_unchanged": True}

    # 3. the producer's parser on a fabricated HistData zip: unchanged behaviour, recorded (no data corrected, nothing invented)
    producer = a.financial_repo/"_scripts/workers/stage13_omega_light_worker.py"
    tree = ast.parse(producer.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_histdata_zip")
    ns = {"pd": pd, "zipfile": zipfile, "Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(producer), "exec"), ns)
    zpath = tmp/"synthetic_histdata.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("DAT_ASCII_EURUSD_M1_202401.csv", "20240102 000000;1.10;1.12;1.09;1.11;0\n20240102 000100;1.11;1.13;1.10;1.12;0\n")
    parsed = ns["parse_histdata_zip"](zpath)
    out["producer_clock_and_columns"] = {"source_sha256": F.sha_file(producer), "parsed_label": str(parsed.datetime.iloc[0]),
                                         "expected_utc_under_source_spec": "2024-01-02 05:00:00+00:00", "columns_after_producer": list(parsed.columns),
                                         "disposition": "UNCHANGED by RP90: the producer's lineage is not established; no metadata, volume, arrival time or "
                                                        "availability lag is invented; the financial resource stays undeliverable until the producer "
                                                        "lineage is established (a scoped evidence deficit under the deferred financial stage)"}

    # 4. the cost pilot's schema refusal: closes FAILED, non-zero; pending outbox is a refusal
    cost = F.seal_cost_pilot(lake="fixture", resource="fixture.parquet", time_column="datetime", holdout="2025-01-01", range_from="2024-01-01",
                             range_to="2024-05-19", dev_start="2024-06-24", contract=fixtures._contract())
    aware = bars.copy(); aware["datetime"] = aware.datetime.dt.tz_localize("UTC")
    croot = tmp/"cost"; croot.mkdir(); (croot/"DESIGN.json").write_text(json.dumps(cost))
    refusal = F.cost_pilot(cost, croot, frame=aware)
    calls = []
    class FakeGov:
        def report_terminal(self, root, unit, terminal, **kwargs):
            calls.append(("terminal", terminal)); return {"flushed": {"sent": 0, "pending": 1, "failures": ["fixture destination unavailable"]}}
        def report_failed(self, root, unit, reason, **kwargs):
            calls.append(("failed", reason)); return {"flushed": {"sent": 0, "pending": 1, "failures": ["fixture destination unavailable"]}}
    U = load("df_utility_run")
    with patch.object(F, "acquire", lambda *args: None), patch.object(F, "governance_modules", lambda: (FakeGov(), U)), patch.object(F, "cost_pilot", lambda *args: refusal):
        r = refused(lambda: F.main(["cost-pilot", "--root", str(croot), "--api-key-file", str(token)]))
    out["cost_refusal_reported_completed"] = {"actual_cost_parser_result": refusal["status"], "parser_reason": refusal.get("reason"),
                                              "main": r, "calls": [(c[0], (c[1] if c[0] == "failed" else c[1]["status"])) for c in calls],
                                              "reported_terminal_status": "FAILED" if calls and calls[0][0] == "failed" else (calls[0][1]["status"] if calls else None),
                                              "fits_executed": 0}
    measured = {**refusal, "status": "MEASURED", "configs": [], "total_pilot_cpu_seconds": 0.0}
    (croot/"COST_PILOT.json").write_text(json.dumps(measured)); calls.clear()
    with patch.object(F, "acquire", lambda *args: None), patch.object(F, "governance_modules", lambda: (FakeGov(), U)), patch.object(F, "cost_pilot", lambda *args: measured):
        r2 = refused(lambda: F.main(["cost-pilot", "--root", str(croot), "--api-key-file", str(token)]))
    out["cost_pending_outbox"] = {"main": r2, "terminal_offered": bool(calls and calls[0][0] == "terminal"), "exit_claimed_success": r2["outcome"] == "RETURNED"}

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps({k: (v.get("after", v).get("outcome") if isinstance(v, dict) else v) for k, v in out.items()}, indent=1, default=str)[:1500])
