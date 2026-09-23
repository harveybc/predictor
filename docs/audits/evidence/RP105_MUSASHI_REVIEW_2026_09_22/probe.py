"""RP105 adversarial lifecycle audit, exclusively disposable CPU fixtures."""
import argparse
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    test = load("rp105_test_fixture", args.repo / "tests/test_df_sota_repro.py")
    R = test.R
    result = {}
    with tempfile.TemporaryDirectory(prefix="musashi-rp105-") as tmp:
        base = Path(tmp)

        class Factory:
            def mktemp(self, name):
                path = base / name
                path.mkdir()
                return path

        world = test.world.__wrapped__(Factory())
        root, unit = world["root"], world["cell"]["cell_id"]

        def verify(path):
            return R.verify_sota_run(path, warehouse=test._wh(world), data_path=world["data"], replay=True)

        baseline = verify(root)
        assert baseline["verified_units"] == [unit], baseline["problems"]
        (root / "REPORT.json").write_text(json.dumps({"verification": baseline}))
        result["baseline"] = {"verified_units": baseline["verified_units"], "deletion_gate": R.deletion_gate(root, unit)}
        assert result["baseline"]["deletion_gate"]["pass"]

        candidate = base / "different_attempt"
        folder = candidate / "attempts" / unit
        folder.mkdir(parents=True)
        (folder / "arrays.npz").write_bytes(b"different prediction artifact, not the verified identity")
        other_hash = R.sha_file(folder / "arrays.npz")
        deleted = R.delete_predictions(root, [unit], extra_roots=[candidate])
        result["different_copy_deleted"] = {
            "verified_arrays_sha256": world["record"]["arrays_sha256"],
            "other_sha256": other_hash,
            "other_path_exists_after": (folder / "arrays.npz").exists(),
            "deleted_digests": [p["sha256"] for p in deleted["paths"] if p.get("deleted")],
        }

        folder = root / "attempts" / unit
        rec_path = folder / "cell.json"
        record = json.loads(rec_path.read_text())
        record["author_metric_float32"] = {"mae": 0.0, "mse": 0.0}
        rec_path.write_text(json.dumps(record))
        (folder / "METRICS_VAULT.json").unlink()
        (folder / "checkpoint.pth").unlink()
        history = verify(root)
        result["changed_record_without_vault_or_checkpoint"] = {
            "historically_verified_units": history["historically_verified_units"],
            "problems": history["problems"],
            "historical_score": history["rows"][0]["author_metric_float32"],
            "table_score": R.table(world["design"], history)["rows"][0]["mae"],
        }

        marker_path = folder / "PREDICTIONS_DELETED.json"
        marker = json.loads(marker_path.read_text())
        marker["closure_report_sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker))
        (root / "REPORT.json").unlink()
        history = verify(root)
        result["nonexistent_historical_report"] = {
            "historically_verified_units": history["historically_verified_units"],
            "problems": history["problems"],
            "status": history["rows"][0]["status"],
        }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
