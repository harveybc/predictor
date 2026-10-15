"""ORCHESTRATION-ONLY adversarial child for the follow-up review (no fit, no data read).

Mode per unit: env RETSU_MODE_<UNIT_ID with '.'->'_'> or RETSU_MODE; a comma list is consumed one
entry per invocation of that unit (the last entry repeats). Every invocation appends
'<unit_id>:<mode>' to RETSU_COUNTER.

  fake                 delegate to tests/i6dmo/ops_matched_fake.py (normal fake terminal)
  refuse75             exit 75 at once, nothing written (a genuine launcher refusal as the worker sees it)
  crash                stdout + out/partial.txt written, exit 1 (a genuine crash after start)
  timeout              exit 124 (the launcher's wall timeout)
  started_then_75      stdout + a COMPLETE out/TERMINAL.json written, then exit 75
  claims_not_started   stderr says "REFUSED nothing was started" but stdout/out/ exist; exit 75
  infra_prefit         FAILED terminal NO_GPU_VISIBLE, no fit_seconds (pre-fit refusal)
  infra_after_fit      FAILED terminal NO_GPU_VISIBLE WITH fit_seconds (post-training failure)
"""
import json
import os
import runpy
import sys
from pathlib import Path

argv = sys.argv[1:]
spec_path = argv[argv.index("--spec") + 1]
out = Path(argv[argv.index("--out") + 1])
spec = json.loads(Path(spec_path).read_text())
unit = spec["unit_id"]
key = "RETSU_MODE_" + unit.replace(".", "_")
modes = (os.environ.get(key) or os.environ.get("RETSU_MODE") or "fake").split(",")
counter = Path(os.environ["RETSU_COUNTER"])
seen = sum(1 for line in (counter.read_text().splitlines() if counter.exists() else []) if line.startswith(unit + ":"))
mode = modes[min(seen, len(modes) - 1)]
with open(counter, "a") as handle:
    handle.write(f"{unit}:{mode}\n")


def terminal(status, disposition, fit_seconds=None, training=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "TERMINAL.json").write_text(json.dumps(dict(
        schema="i6dmo.terminal.v1", unit_id=unit, identity_sha256=spec["identity_sha256"], status=status,
        disposition=disposition, artifacts={}, eligibility={}, gates={}, reports=[],
        costs=dict(wall_seconds=1.0, fit_seconds=fit_seconds), training=training or {})))


if mode == "fake":
    kind = spec["kind"]
    sys.argv = [str(Path(__file__).with_name("ops_matched_fake.py")), "--kind", kind, *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")
elif mode == "refuse75":
    sys.exit(75)
elif mode == "crash":
    print("training epoch 1 ...")
    out.mkdir(parents=True, exist_ok=True)
    (out / "partial.txt").write_text("partial")
    sys.exit(1)
elif mode == "timeout":
    sys.exit(124)
elif mode == "started_then_75":
    print("training epoch 1 ... done")
    terminal("COMPLETE", "RETSU_STARTED_THEN_75", fit_seconds=5.0, training=dict(epochs_completed=2))
    sys.exit(75)
elif mode == "claims_not_started":
    print("training epoch 1 ...")
    print("crispdm-run: REFUSED nothing was started, and no limit, slice or kernel setting was changed",
          file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partial.txt").write_text("partial")
    sys.exit(75)
elif mode == "infra_prefit":
    terminal("FAILED", "NO_GPU_VISIBLE")
elif mode == "infra_after_fit":
    terminal("FAILED", "NO_GPU_VISIBLE", fit_seconds=12.5, training=dict(epochs_completed=3))
else:
    raise SystemExit(f"unknown mode {mode}")
