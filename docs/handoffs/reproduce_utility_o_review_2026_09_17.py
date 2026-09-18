"""Audit design validation and observe calibration branch selection without scoring."""
import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    root = parser.parse_args().source_root
    m = runpy.run_path(str(root / "tests/test_df_utility_next_design.py"))
    design = m["design"]()
    design["families"][0]["protocol"]["margin"] = 99.0
    design["design_sha256"] = m["D"].sha_obj({
        k: v for k, v in design.items() if k != "design_sha256"})
    m["D"].validate_design(design, inherited_protocol=m["PILOT"], pilot_units=m["PILOT_UNITS"])
    print("CHILD_MARGIN_CHANGED_ACCEPTED")
    m = runpy.run_path(str(root / "tests/test_df_utility_harness.py"))
    harness = m["H"]
    protocol = m["proto"](branches=("raw", "transformed", "raw_wide", "augmented"))
    observed = []
    original = harness.contrast

    def spy(*args, **kwargs):
        observed.append([kwargs.get("branch_a", "raw"), kwargs.get("branch_b", "transformed")])
        return {"delta_lower": -1.0, "delta_mean": -1.0}

    try:
        harness.contrast = spy
        harness.calibrate(protocol, m["MAD"], n_sims=1, n=400,
                          generator="white_null", bound_confidence=.95)
    finally:
        harness.contrast = original
    print("CALIBRATED_BRANCHES", observed)


if __name__ == "__main__":
    main()
