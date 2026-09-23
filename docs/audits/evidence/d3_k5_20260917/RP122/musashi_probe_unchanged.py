"""Disposable actual-author CPU fixtures; never modifies production evidence."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    T = load("rp121_fixture", args.repo / "tests/test_df_sota_repro.py")
    R = T.R
    import torch
    torch.set_num_threads(1)
    result = {"scope": "disposable CPU fixtures; warehouse stub; no production mutation",
              "host_before": R.host_thermals()}
    assert result["host_before"] and max(result["host_before"].values()) < 80
    with tempfile.TemporaryDirectory(prefix="musashi-rp121-") as temp:
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
        positive = R.accept_catalog(root, world["design"], unit, data_path=world["data"], independent=True)
        result["positive_catalog"] = {"pass": positive["pass"], "refusals": positive["refusals"]}
        original = R.metrics_vault
        before = {}

        def plausible_wrong(*a, **kw):
            vault = original(*a, **kw)
            before["acf"] = vault["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"]
            before["quantiles"] = vault["residuals"]["quantiles"].copy()
            before["correlation"] = vault["global"]["corr_pred_true"]
            vault["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"] = [
                [0.0 if x is not None else None for x in row] for row in before["acf"]]
            vault["residuals"]["quantiles"] = {key: 0.0 for key in before["quantiles"]}
            vault["global"]["corr_pred_true"] = None
            return vault

        with patch.object(R, "metrics_vault", plausible_wrong):
            bad = case("plausible_wrong")
            accepted = R.accept_catalog(bad, world["design"], unit, data_path=world["data"], independent=True)
        result["plausible_wrong_catalog"] = {"pass": accepted["pass"], "refusals": accepted["refusals"],
            "original": before, "injected": {"acf": "all defined values replaced by zero", "quantiles": "all zero", "correlation": None},
            "independent_comparison": accepted["independent_vs_catalog"],
            "independent_correlation": accepted["independent_estimators"]["corr_pred_true"]}

        # Immutable warehouse evidence says diagnostic attachment, not closure acceptance.
        T._accept_evidence(world, root, "diagnostic", {"diagnostic_attachment": root / "REPORT.json"}, "unrelated-subject")
        digest = R.sha_file(root / "REPORT.json")
        receipt = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
        result["wrong_kind_before_local_relabel"] = R.accepted_artifact(root, receipt, T._wh(world), digest, expect_kind="closure")
        reg = json.loads((root / R.ACCEPTED_EVIDENCE).read_text())
        entry = reg["entries"][digest]
        held_before = json.dumps(world["held"], sort_keys=True)
        entry.update(kind="closure", role="closure_report", subject="closure")
        write(root / R.ACCEPTED_EVIDENCE, reg)
        result["wrong_kind_after_local_relabel"] = R.accepted_artifact(root, receipt, T._wh(world), digest, expect_kind="closure")
        result["warehouse_unchanged_by_relabel"] = held_before == json.dumps(world["held"], sort_keys=True)

        root = case("deletion_bypass")
        ready = T._ready(root, unit, world, base / "backup")
        # Keep all local backup checks intact, but make remote acceptance unavailable.
        ready["warehouse"] = lambda campaign: {"current": {}}
        refused = R.delete_predictions(root, [unit], **ready)
        result["delete_default_without_accepted_chain"] = refused["units"][unit]["state"]
        bypass = R.delete_predictions(root, [unit], require_acceptance=False, **ready)
        result["delete_bypass_without_accepted_chain"] = bypass["units"][unit]["state"]
        result["bypass_arrays_remaining"] = (root / "attempts" / unit / "arrays.npz").exists()
    result["host_after"] = R.host_thermals()
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
