"""Disposable actual-entry-point probes; no production writes, GPU or benchmark fit."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    T = load("rp131_fixture", args.repo / "tests/test_df_sota_repro.py")
    R = T.R
    import torch
    torch.set_num_threads(1)
    out = {"scope": "disposable actual-author CPU fixtures, stub accepted warehouse", "host_before": R.host_thermals()}
    assert out["host_before"] and max(out["host_before"].values()) < 80
    with tempfile.TemporaryDirectory(prefix="musashi-rp131-") as temp:
        base = Path(temp)

        class Factory:
            def mktemp(self, name):
                path = base / name
                path.mkdir()
                return path

        world = T.world.__wrapped__(Factory())
        unit = world["cell"]["cell_id"]

        def case(name):
            place = base / name
            place.mkdir()
            root = T._copy(world, place)
            token = place / "token"
            token.write_text("fixture")
            C = T._load("df_mod_e0_close")
            with patch.object(C, "warehouse_terminals", lambda *a: T._wh(world)(None)):
                R.close(SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="fixture://",
                                       data_path=world["data"], skip_replay=False, replay_device="cpu"), world["design"])
            R.accept_catalog(root, world["design"], unit, data_path=world["data"])
            return root

        def ready(root, name):
            T._accept_evidence(world, root, "closure", {"closure_report": root / "REPORT.json"}, "closure")
            T._accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                                      "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"}, unit)
            backup = base / ("backup_" + name)
            R.metadata_backup(root, backup)
            return {"accepted_report_sha256": R.sha_file(root / "REPORT.json"), "backup_manifest": backup / "MANIFEST.json",
                    "receipts": json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"], "warehouse": T._wh(world)}

        root = case("positive")
        kwargs = ready(root, "positive")
        ca = json.loads((root / "attempts" / unit / "CATALOG_ACCEPTANCE.json").read_text())
        out["positive"] = {"certificate": R.acceptance_certificate(ca)["class"],
                           "deletion_dry_run": R.delete_predictions(root, [unit], dry_run=True, **kwargs)["units"][unit]["state"]}
        for mutation in ("no_fields", "failed_field"):
            root = case(mutation)
            path = root / "attempts" / unit / "CATALOG_ACCEPTANCE.json"
            ca = json.loads(path.read_text())
            if mutation == "no_fields":
                ca["independent_comparison"]["fields"] = {}
            else:
                field = ca["independent_comparison"]["fields"]["global.mae"]
                field.update(status="DISAGREEMENT", max_abs_difference=1.0, detail="injected field failure; summary left stale")
            write(path, ca)
            kwargs = ready(root, mutation)
            result = R.delete_predictions(root, [unit], **kwargs)
            out[mutation] = {"certificate": R.acceptance_certificate(ca)["class"],
                             "reported_field_count": len(ca["independent_comparison"]["fields"]),
                             "mae_field": ca["independent_comparison"]["fields"].get("global.mae"),
                             "deletion": result["units"][unit]["state"],
                             "array_remaining": (root / "attempts" / unit / "arrays.npz").exists()}

        root = case("revalidation")
        kwargs = ready(root, "revalidation")
        def revalidate(warehouse):
            r = R.revalidate_acceptances(root, world["design"], receipts=kwargs["receipts"], warehouse=warehouse)
            cell = r["cells"][unit]
            return {"summary": r["summary"], "accepted": cell["catalog_acceptance"]["accepted"]["accepted"],
                    "current_cell_verified": cell["cell_verified_by_current_closure"], "eligible": cell["deletion_eligible_today"]}

        out["revalidation_positive"] = revalidate(kwargs["warehouse"])
        out["revalidation_empty_warehouse"] = revalidate(lambda campaign: {"current": {}})
        rep = json.loads((root / "REPORT.json").read_text())
        rep["verification"]["rows"][0]["verified"] = False
        rep["verification"]["rows"][0]["problems"] = ["REPLAY_PENDING: later closure withdrew verification"]
        write(root / "REPORT.json", rep)
        # Revalidation must not treat the earlier certificate's cell flag as the current report.
        out["revalidation_current_report_unverified"] = revalidate(kwargs["warehouse"])
        out["actual_deletion_gate_current_report"] = R.deletion_gate(root, unit)
    out["host_after"] = R.host_thermals()
    write(args.output, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
