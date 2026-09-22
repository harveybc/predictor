"""POST of Musashi's RP97 probe (docs/audits/evidence/RP97_MUSASHI_REVIEW_2026_09_22/probe.py) on the repaired tree (RP98).
His script is untouched; this harness repeats his five cases and RECORDS the typed refusal where the repaired code now refuses.
Only the disposable fixture it creates is touched."""
import argparse, copy, importlib.util, json, sys, tempfile, contextlib, io
from pathlib import Path
import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def outcome(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return {"outcome": "RETURNED", "value": fn()}
    except BaseException as exc:
        return {"outcome": "REFUSED", "exception": type(exc).__name__, "message": str(exc)[:300]}


ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args(); sys.path.insert(0, str(a.repo))
test = load("audit_sota_fixture", a.repo / "tests/test_df_sota_repro.py"); R = test.R
results = {"scope": "POST on the repaired tree; Musashi's steps repeated, typed refusals recorded"}
with tempfile.TemporaryDirectory(prefix="rp98-post-") as td:
    base = Path(td)
    class Factory:
        def mktemp(self, name):
            p = base / name; p.mkdir(); return p
    world = test.world.__wrapped__(Factory()); root, unit = world["root"], world["cell"]["cell_id"]
    verify = lambda: R.verify_sota_run(root, warehouse=test._wh(world), data_path=world["data"], replay=True)
    original = verify(); results["baseline"] = {"verified_units": original["verified_units"], "problems": original["problems"]}
    vault_path = root / "attempts" / unit / "METRICS_VAULT.json"; vault = json.loads(vault_path.read_text()); original_vault = vault_path.read_bytes()
    vault["global"]["mae"] = 999.0; vault["per_step"]["mae"][0] = 999.0; vault_path.write_text(json.dumps(vault))
    changed = verify()
    results["altered_vault"] = {"verified_units": changed["verified_units"], "problems": changed["problems"], "persisted_mae_after_closure": json.loads(vault_path.read_text())["global"]["mae"],
                                "rejected_candidates_preserved": [p.name for p in (root / "attempts" / unit).glob("METRICS_VAULT.rejected.*.json")],
                                "reported_vault_digest_is_the_recomputed_successor": changed["rows"][0]["recomputed"]["metrics_vault_sha256"] == R.sha_file(vault_path),
                                "successor_verifies_next_closure": verify()["verified_units"]}
    replay_path = root / "REPLAYS.json"; hist = json.loads(replay_path.read_text())
    for k in hist[unit]:
        hist[unit][k]["max_abs_prediction_difference"] = 999.0; hist[unit][k]["allclose_rule"] = True
    replay_path.write_text(json.dumps(hist))
    changed = verify()
    results["contradictory_cached_replay"] = {"verified_units": changed["verified_units"], "problems": changed["problems"],
                                              "max_abs_prediction_difference_reported": changed["rows"][0]["replay"]["max_abs_prediction_difference"],
                                              "adopted_from_cache": "adopted_from" in changed["rows"][0]["replay"], "exact_equal_fraction": changed["rows"][0]["replay"]["exact_equal_fraction"]}
    import torch
    W, T, C = 10, 2, 1
    x = torch.zeros((W, 24, C)); y = torch.ones((W, T, C)); pred = np.full((W, T, C), 2.0, dtype=np.float32)
    full = outcome(lambda: R.metrics_vault(pred, [(x, y, None, None)], pred_len=T, max_lag=2))
    partial = outcome(lambda: R.metrics_vault(pred, [(x[:5], y[:5], None, None)], pred_len=T, max_lag=2))
    results["short_loader"] = {"full": {"outcome": full["outcome"], "mae": full["value"]["global"]["mae"] if full["outcome"] == "RETURNED" else None,
                                        "consumed_windows": full["value"]["population"]["consumed_windows"] if full["outcome"] == "RETURNED" else None},
                               "partial": {k: v for k, v in partial.items() if k != "value"}}
    results["constant_correlation"] = {"global": full["value"]["global"]["corr_pred_true"], "per_channel": full["value"]["per_channel"]["corr_pred_true"],
                                       "catalog_state": full["value"]["catalog"]["correlation_r2"]}
    design = copy.deepcopy(world["design"]); report = copy.deepcopy(original)
    design["horizons"] = [96, 192, 336, 720]; design["seeds"] = [2021, 2022, 2023]
    design["cells"], report["rows"], report["verified_units"] = [], [], []; design["lock"]["published"]["per_horizon"] = {}
    for value, horizon in enumerate(design["horizons"], 1):
        design["lock"]["published"]["per_horizon"][str(horizon)] = {"mae": value, "mse": value}
        for seed in (2021, 2022, 2023):
            cell = copy.deepcopy(world["cell"]); cell.update(cell_id=f"oracle_h{horizon}_s{seed}", horizon=horizon, seed=seed); design["cells"].append(cell)
            row = copy.deepcopy(original["rows"][0]); row.update(unit=cell["cell_id"], cell=cell, author_metric_float32={"mae": value, "mse": value}, verified=True, problems=[])
            report["rows"].append(row); report["verified_units"].append(cell["cell_id"])
    agg = R.table(design, report)["average_over_horizons"]["mae"]
    results["horizon_vs_seed_grain"] = {"expected_seed_averages": [2.5, 2.5, 2.5], "expected_seed_sd": 0.0, "reported_values": agg.get("values"), "reported_sd": agg.get("sd_ddof1"),
                                        "reported_n_seeds": agg.get("n_seeds"), "grain": agg.get("grain"), "status_label": agg.get("status")}
a.output.write_text(json.dumps(results, indent=2, default=str) + "\n"); print(json.dumps(results, indent=1, default=str)[:3000])
