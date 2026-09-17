"""Read a pilot calibration; mutate copies only. Resume fixture never launches a child."""
import argparse
import copy
import json
import runpy
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    args = parser.parse_args()
    helpers = runpy.run_path(str(args.source_root / "tests/test_df_utility_harness.py"))
    harness = helpers["H"]
    frozen = json.loads((args.pilot_root / "FREEZE.json").read_text())
    doc = frozen["protocols"]["mad_extremes_trailing"]
    protocol = harness.Protocol(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in doc.items()
        if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")})
    n = protocol.calibration["n"]
    print("ORIGINAL_SUPPORT", harness.calibration_supports(protocol, helpers["MAD"], n))
    changed = copy.deepcopy(protocol.calibration)
    changed["upper_bound"] = 0.0
    print("CHANGED_BOUND_SUPPORT", harness.calibration_supports(
        protocol.with_calibration(changed), helpers["MAD"], n))
    changed["advances"], changed["false_advance_rate"] = 0, 0.0
    print("INCONSISTENT_COUNTS", harness.calibration_record_problems(changed))
    with tempfile.TemporaryDirectory(prefix="utility-resume-review-") as tmp:
        path = Path(tmp)
        (path / "outcome.json").write_text(json.dumps(dict(
            status="COMPLETED", verified={"output_sha256": "a" * 64},
            summary={"outcome": "ADVANCES", "cost": {}})))
        (path / "result.json").write_text(json.dumps(dict(
            output_file="contrast.json", output_sha256="a" * 64, outcome="ADVANCES")))
        result = harness.run_isolated(
            {"contrast_id": "x", "protocol": {}}, attempt_dir=path,
            assigned_bytes=100000000, wall_seconds=1, cpu_seconds=1)
        print("RESUME_MISSING_SCORE", json.dumps(result))


if __name__ == "__main__":
    main()
