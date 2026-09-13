"""C130-C133 run: profiles are computed from observed data only, every row
is kept with its module, a failed dataset never stops the others, and the
output is write-once."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_profile_run", ROOT / "tools/df_profile_run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["df_profile_run"] = R
spec.loader.exec_module(R)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    out = tmp_path_factory.mktemp("bank") / "bank"
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/df_synthetic_bank.py"), "--out", str(out), "--limit", "4"],
                   check=True, capture_output=True)
    return out


def test_synthetic_units_are_profiled_and_a_failure_is_recorded(bank, tmp_path):
    jobs = R.synthetic_jobs(bank) + [{"bank": "SYNTHETIC", "dir": str(tmp_path / "missing_unit")}]
    receipt = R.run(tmp_path / "out", jobs, workers=1)
    assert receipt["counts"] == {"COMPLETED": 4, "FAILED": 1}
    done = [d for d in receipt["datasets"] if d["status"] == "COMPLETED"]
    rows = [json.loads(line) for line in (tmp_path / "out" / done[0]["file"]).read_text().splitlines()]
    modules = {r["module"] for r in rows}
    assert modules == set(R.MODULES)
    assert all(set(r["row"]) >= {"dataset_id", "partition", "metric", "estimator", "value", "status", "reason"} for r in rows)
    assert "/home/" not in (tmp_path / "out/PROFILE_RUN_RECEIPT.json").read_text()


def test_the_profile_matrix_is_the_observed_signal_never_the_truth(bank):
    import numpy as np
    # A unit with noise: for a noise-free unit observed equals clean exactly.
    noisy = [j for j in R.synthetic_jobs(bank)
             if json.loads((Path(j["dir"]) / "UNIT.json").read_text())["declared_snr_db"] not in ("inf", "nan")]
    assert noisy, "the fixture bank needs one unit with noise"
    job = noisy[0]
    contract, X, ts, skipped = R.load_job(job)
    unit = Path(job["dir"])
    rec = contract["original_fields"]["unit_record"]

    def as_tv(name):
        a = np.load(unit / name)
        return a.T if a.shape == (rec["n_variables"], rec["n_samples"]) else a

    observed, clean = as_tv("observed_signal.npy"), as_tv("clean_signal.npy")
    assert ts is None and not skipped
    assert np.array_equal(X, observed, equal_nan=True)
    assert not np.array_equal(X, clean, equal_nan=True)


def test_outputs_are_write_once(bank, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="write-once"):
        R.run(tmp_path / "out", R.synthetic_jobs(bank)[:1], workers=1)
