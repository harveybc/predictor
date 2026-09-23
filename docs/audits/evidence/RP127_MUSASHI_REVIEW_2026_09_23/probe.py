"""Independent disposable CPU probes of the actual catalog/deletion call chain."""
import argparse
import hashlib
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    T = load("rp127_fixture", args.repo / "tests/test_df_sota_repro.py")
    R = T.R
    import torch
    torch.set_num_threads(1)
    result = {"scope": "disposable actual-author CPU fixture and stub warehouse only",
              "host_before": R.host_thermals()}
    assert result["host_before"] and max(result["host_before"].values()) < 80
    with tempfile.TemporaryDirectory(prefix="musashi-rp127-") as temp:
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
            return root

        root = case("positive")
        ac = R.accept_catalog(root, world["design"], unit, data_path=world["data"])
        result["positive_catalog"] = {"pass": ac["pass"], "fully_independent": ac["independent_comparison"]["fully_independent"]}
        original = R.metrics_vault

        def wrong_acf(*a, **kw):
            vault = original(*a, **kw)
            block = vault["autocorrelation"]["channel_mean_residual_per_step"]
            block["acf_by_lag"] = [[0.0 if x is not None else None for x in row] for row in block["acf_by_lag"]]
            return vault

        with patch.object(R, "metrics_vault", wrong_acf):
            root = case("no_independent_check")
        good_check = R.accept_catalog(root, world["design"], unit, data_path=world["data"])
        result["wrong_acf_with_reference"] = {"pass": good_check["pass"], "refusals": good_check["refusals"]}
        ac = R.accept_catalog(root, world["design"], unit)
        result["wrong_acf_default_without_data"] = {"pass": ac["pass"], "independent_comparison": ac["independent_comparison"],
                                                   "independently_accepted_families": ac["independently_accepted_families"]}
        T._accept_evidence(world, root, "closure", {"closure_report": root / "REPORT.json"}, "closure")
        T._accept_evidence(world, root, "catalog", {"catalog_acceptance": root / "attempts" / unit / "CATALOG_ACCEPTANCE.json",
                                                  "metrics_vault": root / "attempts" / unit / "METRICS_VAULT.json"}, unit)
        backup = base / "bk_no_independent"
        R.metadata_backup(root, backup)
        receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
        deleted = R.delete_predictions(root, [unit], accepted_report_sha256=R.sha_file(root / "REPORT.json"),
                                      backup_manifest=backup / "MANIFEST.json", receipts=receipts, warehouse=T._wh(world))
        result["delete_without_independent_reference"] = {"state": deleted["units"][unit]["state"],
                                                         "array_remaining": (root / "attempts" / unit / "arrays.npz").exists()}

        root = case("foreign_design")
        ready = T._ready(root, unit, world, base / "bk_foreign")
        receipts = ready["receipts"]
        foreign = "f" * 64
        reg = json.loads((root / R.ACCEPTED_EVIDENCE).read_text())["entries"]
        affected = {entry["unit"] for entry in reg.values()}
        for acceptance_unit in affected:
            row = world["held"][acceptance_unit]
            row["config_sha256"] = foreign
            row["tags"]["design_sha256"] = foreign
            row.pop("terminal_sha256")
            row["terminal_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            receipts[acceptance_unit]["terminal_sha256"] = row["terminal_sha256"]
        # Persist/backup the foreign acceptance honestly; do not corrupt receipt hashes.
        receipt_doc = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())
        receipt_doc["units"] = receipts
        write(root / "TERMINAL_RECEIPTS.json", receipt_doc)
        R.metadata_backup(root, base / "bk_foreign_updated")
        ready["backup_manifest"] = base / "bk_foreign_updated" / "MANIFEST.json"
        strict = R.accepted_artifact(root, receipts, T._wh(world), R.sha_file(root / "REPORT.json"),
                                     expect_kind="closure", expect_role="closure_report", design_sha256=world["design"]["design_sha256"])
        result["foreign_design_explicit_check"] = strict
        deleted = R.delete_predictions(root, [unit], **ready)
        result["foreign_design_actual_deletion"] = {"state": deleted["units"][unit]["state"],
                                                  "array_remaining": (root / "attempts" / unit / "arrays.npz").exists()}
        for acceptance_unit in affected:
            row = world["held"][acceptance_unit]
            row.pop("config_sha256", None)
            row["tags"].pop("design_sha256", None)
            row.pop("terminal_sha256")
            row["terminal_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            receipts[acceptance_unit]["terminal_sha256"] = row["terminal_sha256"]
        result["missing_design_explicit_check"] = R.accepted_artifact(root, receipts, T._wh(world), R.sha_file(root / "REPORT.json"),
                    expect_kind="closure", expect_role="closure_report", design_sha256=world["design"]["design_sha256"])

        def boolean_count(*a, **kw):
            vault = original(*a, **kw)
            assert vault["residuals"]["histogram"]["outside_range"] == 0
            vault["residuals"]["histogram"]["outside_range"] = False
            return vault

        with patch.object(R, "metrics_vault", boolean_count):
            root = case("boolean_count")
        ac = R.accept_catalog(root, world["design"], unit, data_path=world["data"])
        result["boolean_as_numeric_zero"] = {"pass": ac["pass"], "comparison": ac["independent_comparison"]["fields"]["residuals.histogram.outside_range"]}
    result["host_after"] = R.host_thermals()
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
