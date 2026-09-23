"""RP106 POST: Musashi's RP105 adversarial scenarios against the CORRECTED closure and deletion entry points.

His probe (docs/audits/evidence/RP105_MUSASHI_REVIEW_2026_09_22/probe.py) is preserved and run unchanged as well; on the corrected
tree its FIRST scenario now refuses (the different-identity copy is not deleted and no deletion marker is written), so its later
steps — which assume that deletion succeeded — stop with a KeyError. This script therefore re-creates the same three scenarios on
top of a deletion that IS valid, plus the two controls, using only disposable fixtures and the real functions.
"""
import argparse, importlib.util, json, os, shutil, sys, tempfile
from pathlib import Path
from types import SimpleNamespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); sys.path.insert(0, str(a.repo))
    test = load("rp106_fixture", a.repo / "tests/test_df_sota_repro.py"); R = test.R
    out = {}
    with tempfile.TemporaryDirectory(prefix="satoshi-rp106-") as tmp:
        base = Path(tmp)

        class Factory:
            def mktemp(self, name):
                p = base / name; p.mkdir(); return p

        world = test.world.__wrapped__(Factory())
        unit = world["cell"]["cell_id"]

        def closed(tag):
            root = base / tag; shutil.copytree(world["root"], root)
            C = load("df_mod_e0_close", a.repo / "tools/df_mod_e0_close.py")
            import unittest.mock as um
            with um.patch.object(C, "warehouse_terminals", lambda url, tok, c: test._wh(world)(c)):
                tok = base / f"tok_{tag}"; tok.write_text("synthetic")
                rep = R.close(SimpleNamespace(root=root, warehouse_token_file=tok, warehouse_url="synthetic://", data_path=world["data"], skip_replay=False, replay_device="cpu"),
                              json.loads((root / "DESIGN.json").read_text()))
            return root, rep, R.sha_file(root / "REPORT.json")

        def verify(root):
            return R.verify_sota_run(root, warehouse=test._wh(world), data_path=world["data"], replay=False)

        # scenario 2 (Musashi #2): an extra-root file of a different identity under the same unit name
        root, rep, report_sha = closed("conflicting")
        other = base / "different_attempt" / "attempts" / unit; other.mkdir(parents=True)
        (other / "arrays.npz").write_bytes(b"different prediction artifact, not the verified identity")
        res = R.delete_predictions(root, [unit], extra_roots=[base / "different_attempt"], accepted_report_sha256=report_sha)
        out["different_copy_refused"] = {"state": res["units"][unit]["state"], "refusals": res["units"][unit]["preflight"]["refusals"],
                                         "verified_arrays_sha256": world["record"]["arrays_sha256"], "other_sha256": R.sha_file(other / "arrays.npz"),
                                         "verified_path_exists_after": (root / "attempts" / unit / "arrays.npz").is_file(),
                                         "other_path_exists_after": (other / "arrays.npz").is_file(), "deleted_digests": [p["sha256"] for p in res["paths"] if p.get("deleted")]}
        # control: an identical copy is deleted with the verified one
        (other / "arrays.npz").write_bytes((root / "attempts" / unit / "arrays.npz").read_bytes())
        res = R.delete_predictions(root, [unit], extra_roots=[base / "different_attempt"], accepted_report_sha256=report_sha)
        out["identical_copy_control"] = {"state": res["units"][unit]["state"], "deleted": [p["path"].split("/")[-3] for p in res["paths"] if p.get("deleted")],
                                         "all_copies_removed": res["units"][unit]["marker"]["all_copies_removed"]}
        # control: a valid retained history verifies with the ORIGINAL closure's score
        hist = verify(root)
        out["retained_history_control"] = {"historically_verified_units": hist["historically_verified_units"], "problems": hist["problems"],
                                           "historical_score": hist["rows"][0]["author_metric_float32"], "basis": hist["rows"][0]["metric_basis"],
                                           "table_score": R.table(world["design"], hist)["rows"][0]["mae"]}
        # scenario 1 (Musashi #1): rewritten metrics, vault and checkpoint removed
        root2, _, report_sha2 = closed("rewritten")
        R.delete_predictions(root2, [unit], accepted_report_sha256=report_sha2)
        folder = root2 / "attempts" / unit
        rec = json.loads((folder / "cell.json").read_text()); rec["author_metric_float32"] = {"mae": 0.0, "mse": 0.0}
        (folder / "cell.json").write_text(json.dumps(rec)); (folder / "METRICS_VAULT.json").unlink(); (folder / "checkpoint.pth").unlink()
        h = verify(root2)
        out["changed_record_without_vault_or_checkpoint"] = {"historically_verified_units": h["historically_verified_units"], "problems": h["problems"],
                                                             "status": h["rows"][0]["status"], "historical_score": h["rows"][0]["author_metric_float32"],
                                                             "current_record_claims": h["rows"][0]["metric_of_current_record"],
                                                             "table_score": R.table(world["design"], h)["rows"][0]["mae"]}
        # scenario 3 (Musashi #1b): the marker names a report digest nothing has, and REPORT.json is gone
        root3, _, report_sha3 = closed("forged")
        R.delete_predictions(root3, [unit], accepted_report_sha256=report_sha3)
        marker_path = root3 / "attempts" / unit / "PREDICTIONS_DELETED.json"
        marker = json.loads(marker_path.read_text()); marker["closure_report_sha256"] = "0" * 64; marker_path.write_text(json.dumps(marker))
        (root3 / "REPORT.json").unlink()
        shutil.rmtree(root3 / "reports")
        h = verify(root3)
        out["nonexistent_historical_report"] = {"historically_verified_units": h["historically_verified_units"], "problems": h["problems"],
                                                "status": h["rows"][0]["status"], "historical_score": h["rows"][0]["author_metric_float32"]}
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
