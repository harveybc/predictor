"""Run against the reviewed checkout; only temporary cache fixtures are written."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "q_cache_tests", args.repo / "tests/test_df_utility_calibration_cache.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    rec, body = t._record()
    with tempfile.TemporaryDirectory() as tmp:
        cache = t.CC.CalibrationCache(Path(tmp))
        sha = rec["computation_sha256"]
        cache.store(sha, body, producer={"fixture": True})
        original = type(t.MAD).transform

        def changed_transform(*args, **kwargs):
            raise RuntimeError("different operator implementation")

        try:
            type(t.MAD).transform = changed_transform
            key = t.H.computation_key(t._proto(t.A), t.MAD, t.PLAN,
                                     branch_a="raw", branch_b="transformed", seed=1007)
            try:
                t.MAD.transform(None, None)
            except RuntimeError:
                changed = True
            print(json.dumps({"operator_behavior_changed": changed,
                              "same_key": key == rec["computation"],
                              "stale_cache_hit": cache.lookup(key)[0] == body}))
        finally:
            type(t.MAD).transform = original
        (Path(tmp) / sha / "record.json").unlink()
        print("MISSING_RECORD_LOOKUP", cache.lookup(rec["computation"]))
        try:
            result = cache.store(sha, body, producer={"recovery": True})
            print("RECOVERY", result)
        except OSError as exc:
            print("RECOVERY_EXCEPTION", type(exc).__name__, "errno", exc.errno)


if __name__ == "__main__":
    main()
