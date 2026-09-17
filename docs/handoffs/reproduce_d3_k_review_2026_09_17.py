"""Read the reviewed fixture helper; exercise production verify on disposable runs."""
import argparse
import hashlib
import json
import runpy
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    helpers = runpy.run_path(str(args.source_root / "tests/test_d3_matrix_verify.py"))
    for case in ("baseline", "missing_contract", "altered_freeze", "duplicate_terminal"):
        with tempfile.TemporaryDirectory(prefix="d3-k-review-") as tmp:
            root = Path(tmp)
            fixture = helpers["Fixture"](root, [("u", 1)])
            fixture.complete("u")
            path = root / "FREEZE.json"
            frozen = json.loads(path.read_text())
            body = {k: v for k, v in frozen.items() if k != "freeze_sha256"}
            frozen["freeze_sha256"] = hashlib.sha256(json.dumps(
                body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            path.write_text(json.dumps(frozen))
            if case == "missing_contract":
                (root / "bank/u/UNIT.json").unlink()
            elif case == "altered_freeze":
                frozen["freeze_sha256"] = "0" * 64
                path.write_text(json.dumps(frozen))
            elif case == "duplicate_terminal":
                shutil.copytree(root / "collected/A/s", root / "collected/B/s")
            result = helpers["verify"](fixture)
            print(case, json.dumps({key: result[key] for key in ("verified", "refusals")}))


if __name__ == "__main__":
    main()
