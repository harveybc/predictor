"""Fake I6-DMO child: honours the ops child protocol without any model or data.

Env I6DMO_FAKE_COUNTER: append '<kind>:<unit_id>' per invocation (counts fits).
Env I6DMO_FAKE_MODE: 'ok' (default), 'fail' (exit 3), 'oom' (exit 137), 'ineligible'.
Env I6DMO_FAKE_PEAK: cgroup peak bytes to report (default 1 GiB).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("command")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    counter = os.environ.get("I6DMO_FAKE_COUNTER")
    if counter:
        with open(counter, "a") as handle:
            handle.write(f"{args.kind}:{spec['unit_id']}\n")
    mode = os.environ.get("I6DMO_FAKE_MODE", "ok")
    if spec.get("kind") != args.kind:
        print("KIND_MISMATCH", file=sys.stderr)
        return 4
    total = 3
    for done in range(1, total + 1):
        _atomic(args.progress, dict(done=done, total=total, unit="epoch", phase="fit", updated_at=time.time()))
    if mode == "fail":
        return 3
    if mode == "oom":
        return 137
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.csv").write_text("row,h2\n0,1.0\n")
    digest = hashlib.sha256((out / "predictions.csv").read_bytes()).hexdigest()
    eligible = mode != "ineligible"
    horizons = [2, 3, 4, 5, 6, 12] if spec.get("family") == "short" else [18, 24, 36, 48, 60, 72]
    terminal = dict(
        schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
        status="COMPLETE", disposition="FAKE_COMPLETE",
        artifacts={"predictions": {"path": "predictions.csv", "sha256": digest}},
        costs=dict(wall_seconds=1.0, fit_seconds=0.5 if args.kind in ("fit", "final_refit") else 0.0,
                   score_seconds=0.2, cgroup_peak_bytes=int(os.environ.get("I6DMO_FAKE_PEAK", 1 << 30)),
                   gpu_uuid=(spec.get("device") or {}).get("gpu_uuid")),
        eligibility={str(h): dict(mae=1.0 if eligible else 2.0, naive=1.5, eligible=eligible, n=100)
                     for h in horizons} if args.kind == "report" else {},
        gates=dict(strategy_version="fake-1", costs_sha256="0" * 64,
                   pair_status="PASS" if eligible else "PAIR_NOT_ELIGIBLE") if args.kind == "pair" else {},
        reports=[dict(experiment_key=f"i6dmo:fake:{spec['unit_id']}", experiment_set_key="i6dmo:fake",
                      actor="predictor", lake="LOCAL_RETAINED", lineage="UNVERIFIED",
                      config_sha256=spec["identity_sha256"], code_commit="fake", project="predictor",
                      phase="I6DMO", tags={"unit": spec["unit_id"], "test_read": str(spec.get("test_open")).lower()},
                      metrics=[dict(metric="Fit_Seconds", value=0.5, split="validation", horizon=2, unit="seconds")])],
    )
    if args.kind == "pair" and not eligible:
        terminal["status"] = "PAIR_NOT_ELIGIBLE"
    _atomic(out / "TERMINAL.json", terminal)
    raw = (out / "TERMINAL.json").read_bytes()
    print(json.dumps(dict(terminal_sha256=hashlib.sha256(raw).hexdigest())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
