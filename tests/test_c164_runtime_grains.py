"""C164 grains and the C146/C161 row builders: strict rows, typed outcomes, and
no terminal, causal test or host check can claim more than it verified."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load("load_data_foundation")
HP = _load("df_host_preflight")
INC = _load("df_c146_incident")
SHA = "a" * 64


def terminal(**over):
    r = {"run_id": "r", "host_role": "WORKER_A", "bank": "FINANCIAL", "dataset_id": "d", "contract_sha256": SHA,
         "code_sha256": SHA, "status": "COMPLETED", "reason": "", "rows_written": 10, "variables_profiled": 1,
         "metrics_completed": 5, "metrics_missing": 0, "planned_peak_bytes": 100, "observed_peak_rss_bytes": 90,
         "memory_limit_bytes": 1000, "limit_mechanism": "systemd-run scope MemoryMax", "wall_seconds": 1.0,
         "cpu_seconds": 1.0, "output_file": "f.jsonl", "output_sha256": SHA, "started_at": "t0", "ended_at": "t1"}
    r.update(over)
    return r


def test_new_tables_are_generated_additively():
    assert set(L.TABLES) >= {"df_fact_resource_estimate", "df_fact_dataset_terminal", "df_fact_causal_test",
                             "df_fact_naming_isolation_decision", "df_fact_host_receipt", "df_fact_incident_attempt"}
    assert L.SQL_FILE.read_text() == L.ddl()


@pytest.mark.parametrize("over, needle", [
    ({"status": "RESOURCE_EXCEEDED", "output_sha256": None}, "needs a reason"),
    ({"status": "COMPLETED", "output_sha256": None}, "binds its verified output"),
    ({"status": "UNCERTAIN", "reason": "no output", "output_sha256": SHA}, "only a COMPLETED"),
    ({"status": "OOM"}, "not in"),
    ({"host_role": "omega"}, "not in"),
    ({"output_sha256": "nope"}, "sha256"),
])
def test_terminal_refusals(over, needle):
    assert L.validate_row("df_fact_dataset_terminal", terminal()) == []
    assert any(needle in p for p in L.validate_row("df_fact_dataset_terminal", terminal(**over)))


def test_every_terminal_status_is_loadable_with_a_reason():
    for s in L.TERMINAL_STATUSES:
        row = terminal(status=s, reason="" if s == "COMPLETED" else "why",
                       output_sha256=SHA if s == "COMPLETED" else None)
        assert L.validate_row("df_fact_dataset_terminal", row) == [], s


def test_causal_test_and_resource_rows():
    ct = {"run_id": "r", "operator_kind": "ewma", "operator_params": {"alpha": 0.3}, "level": None,
          "test_class": "SUFFIX_ADVERSARIAL", "case_id": "zeros", "n": 64, "cuts_tested": 64, "outcome": "PASS",
          "reason": "", "code_sha256": SHA}
    assert L.validate_row("df_fact_causal_test", ct) == []
    assert any("needs a reason" in p for p in L.validate_row("df_fact_causal_test", dict(ct, outcome="NOT_DETECTED")))
    re_ = {"run_id": "r", "bank": "SYNTHETIC", "dataset_id": "d", "variable_id": None, "partition": "train",
           "module": "df_profile_univariate", "metric": "adf_statistic", "estimator": "augmented_dickey_fuller",
           "estimated_peak_bytes": 10, "formula": "n*(lag+2)*8*5.5", "params": {"n": 1}, "budget_bytes": 5,
           "decision": "NOT_RUN_RESOURCE_BOUND", "stage": "CHILD_RUNTIME", "code_sha256": SHA}
    assert L.validate_row("df_fact_resource_estimate", re_) == []
    assert L.validate_row("df_fact_resource_estimate", dict(re_, decision="PUBLICLY_ELIGIBLE"))


def test_host_receipts_verify_only_what_matches():
    exp = {"head": "h", "code": "c", "manifest_sha": "m"}
    good = {"head": "h", "dirty": 0, "code": "c", "manifest_sha": "m", "manifest_bad": 0, "cgroup": "cpu memory pids",
            "gpu": 0, "python": "PRESENT", "mem_kib": 100, "cpus": 4}
    rows = HP.receipts_for("WORKER_B", good, exp, "r")
    assert all(L.validate_row("df_fact_host_receipt", r) == [] for r in rows)
    assert {r["status"] for r in rows if r["check_name"] in HP.JUDGED} == {"VERIFIED"}
    for key, bad in (("head", "other"), ("manifest_bad", 3), ("cgroup", "cpu pids"), ("gpu", 1), ("dirty", 2)):
        rs = HP.receipts_for("WORKER_B", dict(good, **{key: bad}), exp, "r")
        assert any(r["status"] == "MISMATCH" for r in rs if r["check_name"] in HP.JUDGED), key
    err = HP.receipts_for("WORKER_A", {"error": "ssh failed"}, exp, "r")
    assert {r["status"] for r in err} == {"UNAVAILABLE"}
    assert all(L.validate_row("df_fact_host_receipt", r) == [] for r in err)


def test_table_dirs_load_only_c164_grains(tmp_path):
    import argparse
    import json
    ORCH = _load("df_load_d0_d2")
    d = tmp_path / "battery"
    d.mkdir()
    ct = {"run_id": "battery-1", "operator_kind": "ewma", "operator_params": {}, "level": None,
          "test_class": "PREFIX_ALL_T", "case_id": "c", "n": 8, "cuts_tested": 8, "outcome": "PASS",
          "reason": "", "code_sha256": SHA}
    (d / "df_fact_causal_test.jsonl").write_text(json.dumps(ct) + "\n")
    (d / "df_fact_variable_profile.jsonl").write_text("{}\n")  # not a C164 grain: never taken from here
    args = argparse.Namespace(public_panels=None, synthetic_bank=None, financial_contracts=None, lab=None,
                              lab_delay_cost=None, snr=None, profiles=None, table_dir=[d])
    tables, _ = ORCH.collect(args)
    assert tables["df_fact_causal_test"] == [ct] and "df_fact_variable_profile" not in tables
    run = [r for r in tables["df_dim_run"] if r["module"] == "C164 outputs: battery"]
    assert len(run) == 1 and L.validate_row("df_dim_run", run[0]) == []
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="none of the C164 tables"):
        ORCH.collect(argparse.Namespace(**{**vars(args), "table_dir": [empty]}))


def test_incident_rows_are_non_governing_and_bind_the_record():
    doc = {"attempts": [
        {"attempt_id": "C130_C133_PROFILES_ATTEMPT_1", "module": "C130_C133_PROFILES", "terminated_by": "K",
         "event_at": "t1", "victim_pid": 1, "victim_anon_rss_bytes": 2},
        {"attempt_id": "C134_SNR_CALIBRATION_ATTEMPT_1", "module": "C134_SNR_CALIBRATION", "terminated_by": "S",
         "event_at": "t2", "victim_pid": None, "victim_anon_rss_bytes": None},
        {"attempt_id": "C130_C133_PROFILES_ATTEMPT_2", "module": "C130_C133_PROFILES", "terminated_by": "K",
         "event_at": "t3", "victim_pid": 3, "victim_anon_rss_bytes": 4},
        {"attempt_id": "C134_SNR_CALIBRATION_ATTEMPT_2", "module": "C134_SNR_CALIBRATION", "terminated_by": "S",
         "event_at": "t4", "victim_pid": None, "victim_anon_rss_bytes": None}],
        "roots_left": {"C130_C133_PROFILES": {"root_name": "p", "listing_sha256": SHA,
                                              "missing_final_receipts": ["PROFILE_RUN_RECEIPT.json"]},
                       "C134_SNR_CALIBRATION": {"root_name": "s", "listing_sha256": SHA,
                                                "missing_final_receipts": ["SNR_CALIBRATION.v1.json"]}},
        "record_sha256": SHA, "code_sha256": SHA}
    rows = INC.rows(doc, "r")
    assert all(L.validate_row("df_fact_incident_attempt", r) == [] for r in rows)
    assert {r["state"] for r in rows} == {"NON_GOVERNING_ATTEMPT"}
    kept = [r for r in rows if r["root_name"]]
    assert [r["attempt_id"][-1] for r in kept] == ["2", "2"] and all(r["missing_receipts"] for r in kept)
    assert all(r["evidence"]["record_sha256"] == SHA for r in rows)
