"""C139-C140 load: every collected row validates for its table, coverage is
derived from the rows, and a throwaway load is idempotent."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("df_load_d0_d2")
R = _load("df_profile_run")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("d0d2")
    bank = base / "bank"
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/df_synthetic_bank.py"), "--out", str(bank), "--limit", "2"],
                   check=True, capture_output=True)
    R.run(base / "profiles", R.synthetic_jobs(bank), workers=1)
    return argparse.Namespace(public_panels=None, synthetic_bank=bank, financial_contracts=None, lab=None, snr=None,
                              profiles=base / "profiles")


def test_collected_rows_validate_and_coverage_comes_from_rows(inputs):
    tables, cov = D.collect(inputs)
    for t, rows in tables.items():
        bad = [D.L.validate_row(t, r) for r in rows if D.L.validate_row(t, r)]
        assert not bad, (t, bad[:1])
    assert len(tables["df_dim_dataset"]) == 2 and tables["df_fact_variable_profile"]
    # C164: the runner's durable terminals and both stages of memory estimates are collected
    assert len(tables["df_fact_dataset_terminal"]) == 2
    assert {r["stage"] for r in tables["df_fact_resource_estimate"]} == {"PREFLIGHT_METADATA_UPPER_BOUND",
                                                                         "CHILD_RUNTIME"}
    assert cov["cells"] == len(tables["df_fact_coverage"]) and cov["undeclared_rows"] == 0
    assert cov["counts"]["RESULT"] > 0


@pytest.mark.skipif("PGUSER" not in os.environ, reason="no PG credentials")
def test_throwaway_load_is_idempotent(inputs, tmp_path):
    argv = ["--mode", "throwaway", "--receipt", str(tmp_path / "receipt.json"),
            "--synthetic-bank", str(inputs.synthetic_bank), "--profiles", str(inputs.profiles)]
    assert D.main(argv) == 0
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["idempotent"] is True and receipt["throwaway_database_dropped"] is True
    assert all(r["rows_refused"] == 0 for r in receipt["first_load"].values())
    with pytest.raises(SystemExit, match="write-once"):
        D.main(argv)
