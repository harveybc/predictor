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
    # C169: v2 estimates beside v1, one per physical estimate, no identity collapses
    v2 = tables["df_fact_resource_estimate_v2"]
    assert len(v2) == len(tables["df_fact_resource_estimate"])
    assert len({D.L.row_sha256("df_fact_resource_estimate_v2", r) for r in v2}) == len(v2)
    assert {r["identity_kind"] for r in v2 if r["metric"] in ("unit_root_adf", "unit_root_kpss")} == {"UNIT_ROOT_BLOCK"}
    # C170: coverage v2 beside v1, counts from the ledger, every v1 cell mapped
    c2 = cov["v2"]
    assert c2["cells"] == len(tables["df_fact_coverage_v2"]) == len(tables["df_fact_coverage_v1_v2_map"])
    assert c2["counts_verified_member_by_member"] and c2["undeclared_rows"] == 0 and c2["unmapped_v1_cells"] == 0
    assert c2["counts"]["RESULT"] > 0 and c2["counts"]["NOT_APPLICABLE"] > 0


def test_coverage_run_ids_bind_the_coverage_code(inputs):
    """C178: a v1 or v2 coverage matrix computed by other coverage code never shares a run id with this one."""
    tables, _ = D.collect(inputs)
    code12 = D._sha_file(D.HERE / "df_coverage.py")[:12]
    v1 = {r["run_id"] for r in tables["df_fact_coverage"]}
    v2 = {r["run_id"] for r in tables["df_fact_coverage_v2"]}
    assert len(v1) == len(v2) == 1
    assert next(iter(v1)).startswith("c140_") and next(iter(v1)).endswith("_" + code12)
    assert next(iter(v2)).startswith("c170_") and next(iter(v2)).endswith("_" + code12)
    assert {r["v1_run_id"] for r in tables["df_fact_coverage_v1_v2_map"]} == v1


def test_d2_tables_in_the_loader_equal_the_adjudicator_proposal():
    """C178: the loader holds the D2 v2 specs verbatim, without importing the lab code."""
    A = _load("df_d2_adjudicate")
    for t, spec in A.PROPOSED_TABLES.items():
        assert D.L.TABLES[t] == spec, t
    assert set(D.D2_TABLES) == set(A.PROPOSED_TABLES)
    assert "df_fact_d2_decision" in D.L.ddl()


def test_a_d2_table_dir_is_collected_and_its_rows_validate(inputs, tmp_path):
    A = _load("df_d2_adjudicate")
    row = {"run_id": "d2v2_fresh_test", "design_sha256": "a" * 64, "stratum": "FRESH_CONFIRMATION",
           "subject_kind": "OPERATOR", "subject": "ewma", "operator_params": {"alpha": 0.3}, "spec_sha256": "b" * 64,
           "arm_role": "CANDIDATE", "regime": {"family": "bumps", "perturbation": "white", "declared_snr_db": "10",
                                                "length": 2048, "missingness": "none"},
           "decision": "UNDERPOWERED", "is_decision": False, "reasons": ["SEEDS 3 < DESIGN 10"], "evidence": {},
           "n_seeds_design": 10, "n_seeds_valid": 3, "rule_sha256": "c" * 64, "externally_reviewed": False,
           "code_sha256": "d" * 64}
    assert A.validate_proposed_row("df_fact_d2_decision", row) == []
    d = tmp_path / "d2_tables"
    d.mkdir()
    (d / "df_fact_d2_decision.jsonl").write_text(json.dumps(row) + "\n")
    args = argparse.Namespace(**vars(inputs), table_dir=[d])
    tables, _ = D.collect(args)
    assert tables["df_fact_d2_decision"] == [row] and D.L.validate_row("df_fact_d2_decision", row) == []
    assert any(r["module"] == "D2 outputs: d2_tables" and r["run_id"] == "d2v2_fresh_test" for r in tables["df_dim_run"])
    with pytest.raises(SystemExit, match="none of the C164 or D2 tables"):
        D.collect(argparse.Namespace(**vars(inputs), table_dir=[tmp_path]))


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
