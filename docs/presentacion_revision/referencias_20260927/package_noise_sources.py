"""Archive the displayed historical plot and the distinct latest price sweep."""
from pathlib import Path
from shutil import copy2
import hashlib
import json

repo = Path(__file__).resolve().parents[3]
strategy = repo.parent / "heuristic-strategy"
destination = repo.parent / ".worktrees/heuristic-presentation-20260927/docs/presentation_noise_20260927"
destination.mkdir(parents=True, exist_ok=True)
records = []


def retain(source, relative):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    records.append({"path": relative, "sha256": digest, "bytes": target.stat().st_size})


retain(Path(__file__).parent / "image2.png", "historical/figure_as_presented.png")
for filename in ["sweep_noise_results.csv", "sweep_noise.py"]:
    retain(strategy / filename, "historical/" + filename)
retain(strategy / "tests/data/phase_2_3_base_d3.csv", "historical/input.csv")
retain(strategy / "app/plugins/plugin_direction_atr.py", "historical/plugin_direction_atr.py")
run = strategy / "run_out/native_conditional_20260927_full"
for filename in [
    "input.csv", "manifest.json", "naive.csv", "conditional_paired.csv", "conditional_summary.csv",
    "execution_diagnostics.csv", "sweep.csv", "results.json", "completion.json", "curves.json",
    "INDEPENDENT_VERIFICATION.json", "origins.csv", "NATIVE_PLAN.md", "PLAN.md",
    "native_adapter.py", "native_plugin.py", "legacy_policy.py", "runner.py", "execution.py",
    "shared_helpers.py", "standard_normal_seed42.npy", "standard_normal_seed43.npy",
    "standard_normal_seed44.npy", "targets.npy",
]:
    retain(run / filename, "latest_native/" + filename)
retain(strategy / "docs/noise_price_sweep/NATIVE_PLUGIN_RESULTS.md", "latest_native/RESULTS.md")
for filename in ["barrido_estrategia_largo_profit.png", "barrido_estrategia_largo_sharpe.png",
                 "barrido_estrategia_corto_profit.png", "barrido_estrategia_corto_sharpe.png"]:
    retain(repo / "docs" / filename, "latest_native/figures/" + filename)
manifest = {
    "schema": "presentation_sources.v1", "date": "2026-09-27",
    "displayed_experiment": "historical directional oracle; NOT the latest price-forecast sweep",
    "new_simulations": 0,
    "scope": "Exact data, summaries, source snapshots, noise draws and figures; not the full per-cell trade ledger",
    "files": records,
}
(destination / "FILES.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))
