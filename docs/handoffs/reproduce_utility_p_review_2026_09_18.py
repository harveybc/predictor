"""Exercise closure policy on temporary fixtures, never a real campaign."""
import argparse
import json
import runpy
import tempfile
from pathlib import Path

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    source = parser.parse_args().source_root
    m = runpy.run_path(str(source / "tests/test_df_utility_dev_close.py"))
    close, design_module, harness = m["C"], m["DESIGN"], m["H"]
    with tempfile.TemporaryDirectory(prefix="utility-p-empty-") as tmp:
        root = Path(tmp)
        (root / "REPORT.json").write_text(json.dumps({"run_id": "empty"}))
        result = close.close(root, dict(families=[], replication_map={}, operators=[],
                                       hypotheses={}, design_sha256="x"), root, "CC.json")
        print("EMPTY_DESIGN", result["checks"])
    with tempfile.TemporaryDirectory(prefix="utility-p-check-") as tmp:
        with pytest.MonkeyPatch.context() as patch:
            units = ("bumps__s12", "bumps__s13", "sinusoid__s12", "sinusoid__s13")
            outcomes = {(u, k, h): harness.ADVANCES for u in units
                        for k in design_module.OPS for h in ("H_T", "H_A")}
            root, design = m["_fixture"](Path(tmp), patch, outcomes)
            previous = close.RV.reverify

            def failed_check(*args, **kwargs):
                result = previous(*args, **kwargs)
                result["all_verified"] = False
                return result

            patch.setattr(close.RV, "reverify", failed_check)
            result = close.close(root, design, Path(tmp), "CC.json")
            print("UNVERIFIED_PROPOSALS", result["checks"], len(result["proposed_for_review"]))


if __name__ == "__main__":
    main()
