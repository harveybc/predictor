"""Bounded independent protocol probes; no fitting, GPU, TEST data or services."""
import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools.i6dmo import data, unit_runner  # noqa: E402

# AUD-DMO-03 probes the REAL executor seam: the hook module the worker/unit_runner registration
# resolves for kind final_test (originally tools.i6dmo.strategy_adapter, whose run_unit was the
# unconditional placeholder; the correction re-registers tools.i6dmo.final with typed guards).
final_test_hook = importlib.import_module(unit_runner.hook_module("final_test"))  # noqa: E402


def run():
    design = json.loads((ROOT / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json").read_text())
    origin = 72 * 3600
    ts = np.array([0, 3600, 7200, origin], dtype=np.int64)
    x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., np.nan]])
    offsets = np.arange(-24, 1, 2, dtype=np.int64) * 3600
    indices, slots = data.asof_slot_index(ts, x, np.array([origin]), offsets)
    observed = ts[indices]
    physical = np.broadcast_to(slots[:, :, None], observed.shape)
    mismatch = int(np.count_nonzero(observed != physical))
    current_stale = int(origin - observed[0, -1, 1])
    final_train = {"start": "2021-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"}
    spec = {"kind": "final_refit", "family": "short", "effective_config": {"design": design},
            "window": {"train": list(final_train.values())}}
    selected = unit_runner._family_design(spec)["splits"]["train"]
    with tempfile.TemporaryDirectory() as tmp:
        terminal = final_test_hook.run_unit(
            {"kind": "final_test", "unit_id": "fixture.final", "identity_sha256": "a" * 64},
            Path(tmp) / "out", Path(tmp) / "progress.json")
    probes = [
        {"id": "AUD-DMO-01", "result": "REPRODUCED" if mismatch and current_stale else "NOT_REPRODUCED",
         "declared_exact_timestamps": design["input"]["require_exact_timestamps"],
         "sampler_rule": data.sampling_rule_name(SimpleNamespace(spacing_hours=2)),
         "nonexact_selected_values": mismatch, "current_feature_staleness_hours": current_stale / 3600},
        {"id": "AUD-DMO-02", "result": "REPRODUCED" if selected != final_train else "NOT_REPRODUCED",
         "required_final_train": final_train, "actual_selected_train": selected},
        {"id": "AUD-DMO-03", "result": "REPRODUCED" if terminal["disposition"] == "FINAL_TEST_HOOK_NOT_PREPARED" else "NOT_REPRODUCED",
         "status": terminal["status"], "disposition": terminal["disposition"]},
    ]
    return {"schema": "i6dmo.audit_probes.v1", "audited_commit": "fdaae9d2",
            "fits_run": 0, "test_rows_read": 0, "probes": probes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(0 if all(p["result"] == "REPRODUCED" for p in report["probes"]) else 1)
