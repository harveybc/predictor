"""CPU-only counterexamples; no datasets, TensorFlow or remote model code loaded."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.strategy_root))
    import pandas as pd
    from app.strategy_support import calibration_set, create_elapsed_hour_predictions

    frame = pd.DataFrame(
        {"CLOSE": [1.0, 2.0, 3.0]},
        index=pd.date_range("2019-04-01", periods=3, freq="h"),
    )
    bad_horizons = {}
    for name, horizon in (("bool", True), ("fractional", 1.9), ("text", "1")):
        result = create_elapsed_hour_predictions(frame, [horizon])
        bad_horizons[name] = {
            "requested": horizon,
            "reported_horizons": result.attrs["horizons_hours"],
            "columns": list(result.columns),
        }
        assert result.attrs["horizons_hours"] == (1,)

    duplicates = create_elapsed_hour_predictions(frame, [1, 1])
    assert duplicates.columns.has_duplicates
    nonfinite = frame.copy()
    nonfinite.iloc[1, 0] = float("nan")
    assert create_elapsed_hour_predictions(nonfinite, [1]).isna().any().any()

    crossing = pd.DataFrame(
        {"CLOSE": [1.0, 999.0]},
        index=pd.to_datetime(["2019-05-15 23:00", "2019-05-16 00:00"]),
    )
    predicted = create_elapsed_hour_predictions(crossing, [1])
    accepted_origins = calibration_set(predicted.index)
    assert len(accepted_origins) == 1
    assert predicted.iloc[0, 0] == 999.0

    # Test the loader assumption with a tiny harmless library, not CUDA.
    with tempfile.TemporaryDirectory(prefix="musashi-loader-probe-") as directory:
        library = Path(directory) / "libmusashi_loader_probe.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-x", "c", "-o", str(library), "-"],
            input="int probe(void) { return 42; }\n", text=True, check=True,
            capture_output=True, timeout=20,
        )
        child = (
            "import ctypes,os,sys; "
            "os.environ['LD_LIBRARY_PATH']=sys.argv[1]; "
            "x=ctypes.CDLL('libmusashi_loader_probe.so'); "
            "print(x.probe())"
        )
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        late = subprocess.run([sys.executable, "-c", child, directory], env=env,
                              capture_output=True, text=True, timeout=20)
        env["LD_LIBRARY_PATH"] = directory
        fresh = subprocess.run([sys.executable, "-c", child, directory], env=env,
                               capture_output=True, text=True, timeout=20)
        assert late.returncode != 0 and "cannot open shared object file" in late.stderr
        assert fresh.returncode == 0 and fresh.stdout.strip() == "42"

    print(json.dumps({
        "scope": "Synthetic counterexamples; no financial score or GPU diagnosis",
        "horizon_coercions": bad_horizons,
        "duplicate_horizons_accepted": True,
        "nonfinite_target_accepted": True,
        "origin_only_calibration_accepts_reserved_target": {
            "origin": str(accepted_origins[0]), "target": str(crossing.index[1]),
            "target_value": float(predicted.iloc[0, 0]),
            "limitation": "No noise fitting exists yet; this is a guard gap, not observed fit leakage",
        },
        "dynamic_loader": {
            "late_env_returncode": late.returncode,
            "fresh_env_returncode": fresh.returncode,
            "fresh_result": fresh.stdout.strip(),
            "scope": "glibc loader mechanism, not identification of the missing GPU library",
        },
    }, indent=2))


if __name__ == "__main__":
    main()
