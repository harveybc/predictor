"""Independent lifecycle counterexamples on disposable actual-author CPU fixtures."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    T = load("rp113_fixture", a.repo / "tests/test_df_sota_repro.py")
    R = T.R
    import numpy as np
    import torch
    torch.set_num_threads(1)
    result = {"scope": "disposable CPU fixtures only; no production artifacts changed", "host_before": R.host_thermals()}
    assert result["host_before"] and max(result["host_before"].values()) < 80
    with tempfile.TemporaryDirectory(prefix="musashi-rp113-") as tmp:
        base = Path(tmp)

        class Factory:
            def mktemp(self, name):
                d = base / name
                d.mkdir()
                return d

        world = T.world.__wrapped__(Factory())
        unit = world["cell"]["cell_id"]

        def case(name, deleted=True):
            place = base / name
            place.mkdir()
            if deleted:
                root, _, _ = T._closed_and_deleted(world, place)
            else:
                root = T._copy(world, place)
                token = place / "token"
                token.write_text("fixture")
                C = T._load("df_mod_e0_close")
                with patch.object(C, "warehouse_terminals", lambda *args: T._wh(world)(None)):
                    R.close(SimpleNamespace(root=root, warehouse_token_file=token, warehouse_url="fixture://",
                                           data_path=world["data"], skip_replay=False, replay_device="cpu"), world["design"])
            return root

        def summary(root):
            v = T._verify(world, root)
            return {"historically_verified_units": v["historically_verified_units"],
                    "problems": v["problems"], "metric": v["rows"][0]["author_metric_float32"]}

        root = case("rehash_report")
        result["baseline"] = summary(root)
        rep = json.loads((root / "REPORT.json").read_text())
        rep["verification"]["rows"][0]["author_metric_float32"] = {"mae": 0.0, "mse": 0.0}
        write(root / "REPORT.json", rep)
        marker_path = root / "attempts" / unit / "PREDICTIONS_DELETED.json"
        marker = json.loads(marker_path.read_text())
        marker["closure_report_sha256"] = R.sha_file(root / "REPORT.json")
        write(marker_path, marker)
        result["rewritten_report_and_pointer"] = summary(root)

        root = case("regeneration")
        folder = root / "attempts" / unit
        rec = json.loads((folder / "cell.json").read_text())
        regen = folder / "regenerated"
        regen.mkdir()
        write(regen / "REGENERATION.json", {"identity": "BIT_IDENTICAL_TO_THE_DELETED_ORIGINAL",
              "pred_sha256": rec["pred_sha256"], "true_sha256": rec["true_sha256"],
              "author_metric": {"mae": 0.0, "mse": 0.0}})
        write(regen / "ACCEPTANCE.json", {"pass": True})
        result["fabricated_regeneration"] = summary(root)

        root = case("no_approval", deleted=False)
        result["delete_without_approval_or_backup"] = R.delete_predictions(root, [unit])["units"][unit]["state"]
        result["delete_without_approval_arrays_remaining"] = (root / "attempts" / unit / "arrays.npz").exists()

        original = R.metrics_vault

        def wrong_estimators(*args, **kwargs):
            v = original(*args, **kwargs)
            v["residuals"]["sd"] = -123.0
            v["residuals"]["entropy_bits"] = 999.0
            v["global"]["mutual_information_bits_pred_true_64x64"] = 999.0
            v["autocorrelation"]["channel_mean_residual_per_step"]["acf_by_lag"][0][0] = 42.0
            return v

        with patch.object(R, "metrics_vault", wrong_estimators):
            root = case("wrong_estimators", deleted=False)
            ac = R.accept_catalog(root, world["design"], unit)
        result["wrong_estimator_catalog"] = {"pass": ac["pass"], "refusals": ac["refusals"],
               "oracles": ac["oracles"], "injected": {"residual_sd": -123, "entropy_bits": 999, "mi_bits": 999, "acf": 42}}

        result["float32_parity"] = []
        for n in (8193, 16777217, 16777219):
            pred = np.zeros(n, dtype=np.float32)
            true = np.ones(n, dtype=np.float32)
            exact = R.author_metric_exact(pred, true, leaf=1 << 20)
            expected = float(np.mean(np.abs(true - pred)))
            result["float32_parity"].append({"n": n, "bounded": exact["mae"], "numpy_mean": expected, "equal": exact["mae"] == expected})
            del pred, true
    result["host_after"] = R.host_thermals()
    write(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
