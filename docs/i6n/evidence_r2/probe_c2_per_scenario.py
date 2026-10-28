"""Per-scenario replay of the three C2 scenarios of the review probe (probe_candidates.py c2).

The review probe has no exception handling, so on corrected code it stops at the
first scenario that is now refused. This harness builds each scenario exactly as
the probe does and records, per scenario, either the observed values or the
typed refusal. Synthetic only; files live in a TemporaryDirectory.

    python docs/i6n/evidence_r2/probe_c2_per_scenario.py --repo <worktree>
"""

import argparse
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


def scenarios(repo):
    from tools.i6n_contracts import paired_metrics as pm
    from tools.i6n_contracts import paired_metrics_cli as cli

    fixtures = runpy.run_path(str(repo / "tests/test_i6n_paired_metrics.py"))
    doc = fixtures["document"]([fixtures["rec"](0, 2)])
    result = pm.score(doc)
    group = fixtures["group"](result, pm.OPPORTUNITY, 2)

    def phantom_population():
        phantom = copy.deepcopy(group)
        phantom["n_declared"] = 5
        phantom["coverage_denominator"] = 5
        phantom["coverage"] = 0.2
        phantom["sha256"] = pm.canonical_sha256({k: v for k, v in phantom.items() if k != "sha256"})
        merged = pm.aggregate_groups([phantom])
        return {k: merged[k] for k in ("state", "n", "n_declared", "n_excluded", "failures", "coverage")}

    def negative_error_sums():
        negative = copy.deepcopy(group)
        for key in pm._SUM_FIELDS:
            negative[key] = -abs(negative[key])
        negative.update(pm._metrics(negative["n"], {k: negative[k] for k in pm._SUM_FIELDS}))
        negative["sha256"] = pm.canonical_sha256({k: v for k, v in negative.items() if k != "sha256"})
        merged = pm.aggregate_groups([negative])
        return {k: merged[k] for k in ("state", "mae", "mse", "naive_mae")}

    def missing_result_status():
        with tempfile.TemporaryDirectory(prefix="i6n-r2-") as temp:
            root = Path(temp)
            config = root / "config.json"
            config.write_text(json.dumps(doc))
            output = root / "metrics"
            assert cli.run(config, output)[0] == 0
            (output / "RESULT.json").unlink()
            code, value = cli.status(output)
            return {"exit_code": code, "state": value["state"], "disposition": value.get("disposition"),
                    "result_state": (value.get("result") or {}).get("result_state")}

    observed = {}
    for name, scenario in (("phantom_population", phantom_population),
                           ("negative_error_sums", negative_error_sums),
                           ("missing_result_status", missing_result_status)):
        try:
            observed[name] = {"outcome": "NO_EXCEPTION", "values": scenario()}
        except pm.PairedMetricsError as error:
            observed[name] = {"outcome": "REFUSED", "code": error.code, "detail": error.detail}
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--",
                                          "tools/i6n_contracts", "tests/test_i6n_paired_metrics.py"], text=True))
    print(json.dumps({"component": "c2", "commit": commit, "component_files_dirty": dirty,
                      "synthetic_only": True, "observed": scenarios(repo)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
