"""Audit existing production callables using only synthetic and disposable fixtures."""
import argparse
import copy
import hashlib
import json
import runpy
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    root = parser.parse_args().source_root
    m = runpy.run_path(str(root / "tests/test_df_utility_harness.py"))
    harness, protocol = m["H"], m["PROTO"]
    x = m["fabricated"](truth="noise")
    y = harness.label(x, protocol)
    rep = dict(values=np.nan_to_num(y), available=np.isfinite(y), reach_right=0,
               accepted=True, emitted_at=np.arange(len(x)) + 10000)
    late = harness.contrast(x, rep, protocol)
    rep["emitted_at"] = np.arange(len(x))
    current = harness.contrast(x, rep, protocol)
    print("FUTURE_LABEL_AS_FEATURE", late["outcome"], late["delta_lower"])
    print("IGNORED_EMISSION", late == current)
    print("T_DF1", harness._t_quantile(.975, 1), float(t.ppf(.975, 1)))
    m = runpy.run_path(str(root / "tests/test_d3_matrix_verify.py"))
    with tempfile.TemporaryDirectory(prefix="d3-l-review-") as tmp:
        _, child = m["composite_pair"](Path(tmp))
        child.freeze["operators"] = copy.deepcopy(child.freeze["operators"])
        op = child.freeze["operators"][1]
        op["params"], op["spec_sha256"] = {"window": 999}, "9" * 64
        child.reseal_freeze()
        path = child.root / "REPORT.json"
        record = json.loads(path.read_text())
        record["synthetic_spec"]["freeze_sha256"] = child.freeze["freeze_sha256"]
        config = dict(schema="d3_mechanics_execution.v1", run_id="r",
                      freeze_sha256=child.freeze["freeze_sha256"], design_sha256=m["DESIGN"])
        record["config_sha256"] = hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(record))
        result = m["verify"](child)
        print("CHANGED_UNMEASURED_OPERATOR", json.dumps(dict(
            verified=result["verified"], refusals=result["refusals"],
            verdicts=result["operators"].get("op_b", {}).get("verdicts"))))


if __name__ == "__main__":
    main()
