"""Exercise calibration acceptance on fabricated data; no real runs or services."""
import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    root = parser.parse_args().source_root
    m = runpy.run_path(str(root / "tests/test_df_utility_harness.py"))
    harness, protocol, operator = m["H"], m["proto"](), m["MAD"]
    series = harness.series(m["fabricated"]())
    for rate in (0.0, float("nan")):
        candidate = protocol.with_calibration(dict(
            generator="not-a-measured-null", n_sims=0, seed=1,
            false_advance_rate=rate, alpha_adjusted=protocol.alpha_adjusted))
        result = harness.contrast(
            series, operator, candidate, contrast_id=m["FAMILY"][0],
            eligibility=m["record_for"](operator), unit="fab", variable="v0")
        print("EMPTY_CALIBRATION", 0, str(rate), result["outcome"], result.get("delta_lower"))


if __name__ == "__main__":
    main()
