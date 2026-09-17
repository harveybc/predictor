"""Reproduce two matrix audit gaps against a specified checkout; only temporary data."""
import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "reviewed_matrix", args.source_root / "tools/df_d3_matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    with tempfile.TemporaryDirectory(prefix="d3-matrix-review-") as tmp:
        root = Path(tmp)
        rows = root / "collected/A/s/attempts/u/attempt-1/rows.jsonl"
        rows.parent.mkdir(parents=True)
        row = dict(unit_id="u", variable="v", operator_kind="op", operator_group="g",
                   bank="SYNTHETIC", family="f", test="verdict",
                   outcome="MECHANICALLY_ACCEPTED", value=1, detail="")
        rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
        receipt = dict(run_id="audit", mismatched=0, units=[dict(
            unit="u", role="A", shard="s", output_verified=True, rows=13)])
        (root / "COLLECT.json").write_text(json.dumps(receipt), encoding="utf-8")
        result = matrix.aggregate(root)
        print("MISSING_12_TESTS", json.dumps(dict(
            rows=result["rows"], verdicts=result["operators"]["op"]["verdicts"],
            tests=result["operators"]["op"]["tests"], mismatched=result["mismatched"])))
        rows.unlink()
        result = matrix.aggregate(root)
        print("MISSING_FILE", json.dumps({key: result[key] for key in
              ("units_verified", "units_collected", "mismatched", "rows")}))


if __name__ == "__main__":
    main()
