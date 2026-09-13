"""Reseal the D2 v2 design under the current lab code (worker fix 535420d), before any fresh unit exists.

Inputs are rebuilt from the sealed v1 document only. The script refuses unless every field of the new design,
except design_id, lab_code_sha256s and design_sha256, equals v1's, and unless df_d2_unit_worker is the only lab code
file whose digest moved. Write-once output root.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
import df_d2_design as D  # noqa: E402
import df_isolated_runner as IR  # noqa: E402

S = Path.home() / ".local/state/crispdm-data-foundation"
V1 = S / "d2_design_c171_v1"
OUT = S / "d2_design_c171_v2"
if OUT.exists():
    raise SystemExit("REFUSED: output root exists (write-once)")
v1 = json.loads((V1 / "D2_DESIGN_V2.json").read_text())
disp = json.loads((V1 / "C137_DISPERSION.json").read_text())
inputs = {"design_id": "D2V2_C171_R2_2026_09_13",
          "cells": v1["bank"]["cells"],
          "operators": [{"spec": o["spec"], "arm_role": o["arm_role"]} for o in v1["operators"]],
          "fit_modes": {o["kind"]: o["fit_mode"] for o in v1["operators"]},
          "dispersion": disp["rows"], "dispersion_source": v1["power"]["dispersion_source"],
          "snr": {"estimators": v1["snr"]["estimators"], "bootstrap": v1["snr"]["bootstrap"]},
          "budget": v1["budget"], "roles": v1["roles"],
          "denoising_rules": v1["denoising_rules"], "snr_rules": v1["snr_rules"]}
v2 = D.build_design(inputs)
skip = {"design_id", "lab_code_sha256s", "design_sha256"}
diff = sorted(k for k in set(v1) | set(v2) if k not in skip and v1.get(k) != v2.get(k))
moved = sorted(k for k in set(v1["lab_code_sha256s"]) | set(v2["lab_code_sha256s"])
               if v1["lab_code_sha256s"].get(k) != v2["lab_code_sha256s"].get(k))
if diff or moved != ["df_d2_unit_worker"]:
    raise SystemExit(f"REFUSED: fields differ {diff}; code digests moved {moved}")
problems = D.validate_design(v2, require_current_code=True)
if problems:
    raise SystemExit(f"REFUSED: {problems[:3]}")
summary = {"schema": "crispdm.data_foundation.d2_design_reseal.v1", "supersedes": {
               "design_id": v1["design_id"], "design_sha256": v1["design_sha256"]},
           "design_id": v2["design_id"], "design_sha256": v2["design_sha256"], "cause": "df_d2_unit_worker turned an unevaluated oracle on a unit refused whole by the declared "
                    "missing-data rule into a whole-root invalidation (fixed at 535420d); resealed before any fresh "
                    "unit exists",
           "identical_fields": sorted(k for k in v2 if k not in skip),
           "code_digests_moved": {k: [v1["lab_code_sha256s"][k], v2["lab_code_sha256s"][k]] for k in moved},
           "regime_status": {s: sum(p.get("status", p.get("regime_status")) == s for p in v2["seeds_per_regime"].values())
                             for s in ("POWERED", "UNDERPOWERED")} if isinstance(v2["seeds_per_regime"], dict) else None,
           "fresh_units_total": sum(p["n_seeds"] for p in v2["seeds_per_regime"].values())
           if isinstance(v2["seeds_per_regime"], dict) else None}
OUT.mkdir(parents=True)
design_file_sha = D.write_design(v2, OUT / "D2_DESIGN_V2.json")
IR.atomic_write_once(OUT / "C137_DISPERSION.json", (V1 / "C137_DISPERSION.json").read_text())
summary["design_file_sha256"] = design_file_sha
IR.atomic_write_once(OUT / "SEAL_SUMMARY.json", json.dumps(summary, indent=1, sort_keys=True) + "\n")
for p in OUT.iterdir():
    p.chmod(0o444)
OUT.chmod(0o555)
print(json.dumps(summary, indent=1))
