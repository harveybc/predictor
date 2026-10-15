"""ORCHESTRATION-ONLY fake child for the matched short contrast (fit and report kinds).

Stands in for the model lane's matched trainer/report until they land; it proves the runner's
queue/worker/terminal/outbox/mirror/CLOSE plumbing, NOT the science. The reviewer runs the real
adapters. The comparison unit itself is always the REAL ``unit_runner --kind comparison`` hook.

Env:
  I6DMO_FAKE_COUNTER            append '<kind>:<unit_id>' per invocation (counts fits)
  I6DMO_FAKE_REL_<ARM>          relative error of every head of that arm (default mg1 -0.02, mg2 -0.01)
  I6DMO_FAKE_FAIL_UNIT          unit_id that exits 3 (technical failure)
  I6DMO_FAKE_DROP_HEAD          '<arm>:<h>' omitted from that arm's METRICS (missing head)
  I6DMO_FAKE_POPULATION         population digest the report declares (default: the campaign pin)
  I6DMO_FAKE_ROWS_<ARM>         per-head row-id digest the report declares (default: shared)
  I6DMO_FAKE_PEAK               cgroup peak bytes reported by a fit (default 1 GiB)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HORIZONS = (2, 3, 4, 5, 6, 12)
DEFAULT_REL = {"mg1": -0.02, "mg2": -0.01}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("command")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    if spec.get("kind") != args.kind:
        return 4
    if os.environ.get("I6DMO_FAKE_COUNTER"):
        with open(os.environ["I6DMO_FAKE_COUNTER"], "a") as handle:
            handle.write(f"{args.kind}:{spec['unit_id']}\n")
    Path(args.progress).write_text(json.dumps(dict(done=1, total=1, unit="epoch", phase="fake", updated_at=time.time())))
    if os.environ.get("I6DMO_FAKE_FAIL_UNIT") == spec["unit_id"]:
        return 3
    arm = spec["arm"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    view = spec["effective_config"]
    assert view.get("arm") == arm and "short_bundle" in view["data_pins"], "arm view missing"
    population = os.environ.get("I6DMO_FAKE_POPULATION") or view["data_pins"]["common_population"]["population_sha256"]
    artifacts, reports, lineage = {}, [], {}
    costs = dict(wall_seconds=0.1, fit_seconds=0.0, score_seconds=0.1, cgroup_peak_bytes=None)
    if args.kind == "fit":
        (out / "predictions.json").write_text(json.dumps(dict(arm=arm, fake=True)))
        artifacts["predictions"] = dict(path="predictions.json", sha256=_sha(out / "predictions.json"))
        costs.update(fit_seconds=0.5, cgroup_peak_bytes=int(os.environ.get("I6DMO_FAKE_PEAK", 1 << 30)))
    else:
        rel = float(os.environ.get(f"I6DMO_FAKE_REL_{arm.upper()}", DEFAULT_REL[arm]))
        drop = os.environ.get("I6DMO_FAKE_DROP_HEAD", "")
        rows = os.environ.get(f"I6DMO_FAKE_ROWS_{arm.upper()}")
        per_h = {}
        for h in HORIZONS:
            if drop == f"{arm}:{h}":
                continue
            naive = 0.001 * h
            per_h[str(h)] = dict(n=100, mae=naive * (1 + rel), naive_mae=naive, relative_error=rel,
                                 row_ids_sha256=rows or hashlib.sha256(f"common-h{h}".encode()).hexdigest())
        (out / "METRICS.json").write_text(json.dumps(dict(per_horizon=per_h, common_population_sha256=population,
                                                          splits_read=["train", "validation"])))
        artifacts["metrics"] = dict(path="METRICS.json", sha256=_sha(out / "METRICS.json"))
        lineage = dict(common_population_sha256=population)
        reports.append(dict(experiment_key=f"i6dmo:fake:{spec['unit_id']}", experiment_set_key="i6dmo:fake",
                            actor="predictor", lake="LOCAL_RETAINED", lineage="UNVERIFIED",
                            config_sha256=spec["identity_sha256"], code_commit="fake", project="predictor",
                            phase="I6DMO", tags={"unit": spec["unit_id"], "test_read": "false"},
                            metrics=[dict(metric="Price_MAE", value=r["mae"], split="validation", horizon=int(h),
                                          unit="USD_per_EUR") for h, r in per_h.items()]))
    terminal = dict(schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
                    status="COMPLETE", disposition="FAKE_ORCHESTRATION_ONLY", artifacts=artifacts, costs=costs,
                    eligibility={}, gates={}, reports=reports, lineage=lineage)
    (out / "TERMINAL.json").write_text(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
