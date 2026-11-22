#!/usr/bin/env python
"""Pair the seed-7 encoder control with the pretrained champions."""
import json
from pathlib import Path

PHASE = Path("/var/tmp/retsu-phase1")
CONTROL = Path("/var/tmp/retsu-phase1-control")
rows = []
for path in sorted((PHASE / "jobs").iterdir()):
    other = CONTROL / "jobs" / path.name / "holdout.json"
    if not other.exists():
        continue
    pre = json.loads((path / "holdout.json").read_text())
    ctrl = json.loads(other.read_text())
    rows.append({
        "job": path.name,
        "pretrained_2024_mae": pre.get("holdout_2024_mae"),
        "control_2024_mae": ctrl.get("holdout_2024_mae"),
        "delta_control_minus_pretrained": float(ctrl["holdout_2024_mae"]) - float(pre["holdout_2024_mae"]),
        "naive_mae_monitor": pre.get("naive_mae_monitor"),
        "control_best_val_mae": ctrl.get("best_val_mae"),
        "pretrained_best_val_mae": pre.get("best_val_mae"),
        "control_epoch": ctrl.get("epoch"),
        "control_best_epoch": (ctrl.get("early_stopping") or {}).get("best_epoch"),
        "control_es_wait": (ctrl.get("early_stopping") or {}).get("wait"),
    })
payload = {
    "control": "DoubleTCN built with seed 7, model.pt not loaded",
    "pretrained": "recipe 61 bundle",
    "jobs": rows,
}
(PHASE / "followup" / "control_vs_pretrained.json").write_text(json.dumps(payload, indent=2) + "\n")
print("COMPARE_OK", len(rows))
