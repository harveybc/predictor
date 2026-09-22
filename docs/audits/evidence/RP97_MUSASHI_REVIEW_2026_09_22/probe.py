"""Adversarial audit of RP97. Run on a worker; only disposable fixtures change."""
import argparse
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    test = load("audit_sota_fixture", args.repo / "tests/test_df_sota_repro.py")
    R = test.R
    results = {}
    with tempfile.TemporaryDirectory(prefix="musashi-rp97-") as temporary:
        base = Path(temporary)

        class Factory:
            def mktemp(self, name):
                path = base / name
                path.mkdir()
                return path

        world = test.world.__wrapped__(Factory())
        root, unit = world["root"], world["cell"]["cell_id"]

        def verify():
            return R.verify_sota_run(root, warehouse=test._wh(world),
                                     data_path=world["data"], replay=True)

        original = verify()
        assert original["verified_units"] == [unit], original["problems"]
        results["baseline"] = {"verified_units": original["verified_units"], "problems": original["problems"]}
        vault_path = root / "attempts" / unit / "METRICS_VAULT.json"
        vault = json.loads(vault_path.read_text())
        original_vault = vault_path.read_bytes()
        original_mae = vault["global"]["mae"]
        vault["global"]["mae"] = 999.0
        vault["per_step"]["mae"][0] = 999.0
        vault_path.write_text(json.dumps(vault))
        changed = verify()
        results["altered_vault"] = {
            "original_mae": original_mae,
            "persisted_mae": json.loads(vault_path.read_text())["global"]["mae"],
            "verified_units": changed["verified_units"], "problems": changed["problems"],
            "reported_vault_digest_matches_altered_file": changed["rows"][0]["recomputed"]["metrics_vault_sha256"] == R.sha_file(vault_path),
        }
        vault_path.write_bytes(original_vault)
        replay_path = root / "REPLAYS.json"
        replays = json.loads(replay_path.read_text())
        replays[unit]["max_abs_prediction_difference"] = 999.0
        replays[unit]["allclose_rule"] = True
        replay_path.write_text(json.dumps(replays))
        changed = verify()
        results["contradictory_cached_replay"] = {
            "verified_units": changed["verified_units"], "problems": changed["problems"],
            "max_abs_prediction_difference": changed["rows"][0]["replay"]["max_abs_prediction_difference"],
        }

        import torch
        W, T, C = 10, 2, 1
        x = torch.zeros((W, 24, C))
        y = torch.ones((W, T, C))
        pred = np.full((W, T, C), 2.0, dtype=np.float32)
        full = R.metrics_vault(pred, [(x, y, None, None)], pred_len=T, max_lag=2)
        partial = R.metrics_vault(pred, [(x[:5], y[:5], None, None)], pred_len=T, max_lag=2)
        results["short_loader"] = {"full_mae": full["global"]["mae"],
                                   "partial_mae": partial["global"]["mae"],
                                   "claimed_windows": partial["population"]["windows"],
                                   "consumed_windows": 5}
        results["constant_correlation"] = {
            "global": full["global"]["corr_pred_true"],
            "per_channel": full["per_channel"]["corr_pred_true"],
        }
        # Pure table oracle: horizons differ, but all three seeds are identical.
        design = copy.deepcopy(world["design"])
        report = copy.deepcopy(original)
        design["horizons"] = [96, 192, 336, 720]
        design["cells"], report["rows"], report["verified_units"] = [], [], []
        design["lock"]["published"]["per_horizon"] = {}
        for value, horizon in enumerate(design["horizons"], 1):
            design["lock"]["published"]["per_horizon"][str(horizon)] = {"mae": value, "mse": value}
            for seed in (2021, 2022, 2023):
                cell = copy.deepcopy(world["cell"])
                cell.update(cell_id=f"oracle_h{horizon}_s{seed}", horizon=horizon, seed=seed)
                design["cells"].append(cell)
                row = copy.deepcopy(original["rows"][0])
                row.update(unit=cell["cell_id"], cell=cell, author_metric_float32={"mae": value, "mse": value})
                report["rows"].append(row)
                report["verified_units"].append(cell["cell_id"])
        aggregate = R.table(design, report)["average_over_horizons"]["mae"]
        results["horizon_vs_seed_grain"] = {
            "fixture": "four distinct horizon errors, same errors for all three seeds",
            "expected_seed_averages": [2.5, 2.5, 2.5], "expected_seed_sd": 0.0,
            "reported_sd": aggregate["sd_ddof1"], "reported_n_seeds": aggregate["n_seeds"],
        }
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
