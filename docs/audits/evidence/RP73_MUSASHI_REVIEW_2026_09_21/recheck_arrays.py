"""Independent read-only arithmetic/content check of the 18 new RP73 forecast arrays.

Anchors are the terminal artifacts published at 1880caa, not a local cell's own badge.
This is not a new query of accounting or the live warehouse, nor a model-weight replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--state", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
evidence = a.repo / "docs/audits/evidence/d3_k5_20260917/RP66"
rows, problems = [], []
for run in ("e1_block_dev_matched_v2", "e1_block_q1_calendar_v1", "e1_block_q3_volume_v1"):
    published = evidence / "blocks" / run
    root = a.state / run
    design = json.loads((published / "DESIGN.json").read_text())
    with np.load(root / "BLOCK_DATA.npz", allow_pickle=False) as z:
        target, origins, h = z["Y"], z["common_eval"], int(z["horizon"][0])
        sigma = float(z["scaler_sd"][int(z["target_channel"][0])])
    for cell in design["cells"]:
        unit = cell["cell_id"]
        terminal = json.loads((published / "TERMINALS" / f"{unit}.json").read_text())
        artifacts = {x["role"]: x for x in terminal["artifacts"]}
        folder = root / "attempts" / unit
        issues = []
        for role, name in (("predictions", "arrays.npz"), ("record", "cell.json"), ("weights", "weights.weights.h5")):
            raw = (folder / name).read_bytes()
            actual = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            if any(actual[k] != artifacts[role][k] for k in actual):
                issues.append(f"{role} differs from published terminal artifact")
        with np.load(folder / "arrays.npz", allow_pickle=False) as z:
            if not np.array_equal(z["origins"], origins):
                issues.append("origin population")
            if not np.array_equal(z["y"], target[origins + h]):
                issues.append("target identity")
            if not np.array_equal(z["naive"], target[origins]):
                issues.append("naive identity")
            pred, y, naive = [np.asarray(z[k], dtype=np.float64) for k in ("pred", "y", "naive")]
            if not all(np.isfinite(x).all() for x in (pred, y, naive)):
                issues.append("nonfinite")
            mae = float(np.mean(np.abs(pred - y)))
            naive_mae = float(np.mean(np.abs(naive - y)))
        reported = json.loads((folder / "cell.json").read_text())["scores"]
        if abs(reported["mae_kw"] - mae) > 1e-12 or abs(reported["mae_z"] - mae / sigma) > 1e-12:
            issues.append("metric mismatch")
        row = {"run": run, **cell, "n": int(len(origins)), "mae_z": mae / sigma,
               "naive_mae_z": naive_mae / sigma, "skill": 1 - mae / naive_mae, "issues": issues}
        rows.append(row)
        problems.extend(f"{run}/{unit}: {p}" for p in issues)
summary = {}
for arm in sorted({r["arm"] for r in rows}):
    group = [r for r in rows if r["arm"] == arm]
    summary[arm] = {"n_seeds": len(group), "mean_mae_z": float(np.mean([r["mae_z"] for r in group])),
                    "sd_mae_z": float(np.std([r["mae_z"] for r in group], ddof=1)),
                    "mean_skill": float(np.mean([r["skill"] for r in group]))}
result = {"scope": __doc__, "rows": rows, "summary": summary, "problems": problems,
          "checked_forecasts": len(rows), "new_warehouse_query": False, "fresh_weight_replay": False}
a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"checked_forecasts": len(rows), "problems": problems, "summary": summary}, indent=2))
raise SystemExit(1 if problems else 0)
