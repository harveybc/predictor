"""Orchestration-only child: writes a FAILED terminal with RETSU_FAIL_CODE and (optionally) a nonzero
fit_seconds, as a trainer that refused before training (no fit_seconds) or failed after it would."""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("command")
p.add_argument("--spec", required=True)
p.add_argument("--out", required=True)
p.add_argument("--progress", required=True)
a = p.parse_args()
spec = json.loads(Path(a.spec).read_text())
fit_s = os.environ.get("RETSU_FIT_SECONDS") or None
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
doc = dict(schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
           status="FAILED", disposition=os.environ.get("RETSU_FAIL_CODE", "NONFINITE_LOSS"), artifacts={},
           costs=dict(wall_seconds=1.0, fit_seconds=float(fit_s) if fit_s else None), eligibility={}, gates={},
           reports=[], training=dict(epochs_completed=3) if fit_s else {})
(out / "TERMINAL.json").write_text(json.dumps(doc))
