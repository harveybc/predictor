"""P4: the closure pairs each selection row with its mapped replica, proposes for review only
what advances in both, aggregates files/parent/accounting/warehouse checks, and never promotes."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("df_utility_dev_close")
H = _load("df_utility_harness")
DESIGN = _load("test_df_utility_next_design", HERE)


def _row(unit, op, h, outcome, delta=-0.1):
    return {"contrast": f"{unit}__v0__{op}__x", "unit": unit, "variable": "v0", "operator": op, "hypothesis": h,
            "loss_raw_mean": 0.2, "loss_transformed_mean": 0.2 - delta, "delta_mean": delta, "delta_lower": delta - 0.05,
            "delta_se": 0.01, "rows_paired": 390, "n": 400, "blocks_used": 4, "cpu_seconds": 1.0,
            "null_scope": {"derived_decision_bound": 0.004}, "reverified_outcome": outcome}


def _fixture(tmp_path, monkeypatch, outcomes, stopped=None, missing_family=None):
    design = DESIGN.design()
    root = tmp_path / "dev"
    for fam in design["families"]:
        if fam["unit"] == missing_family:
            continue
        froot = root / "families" / fam["unit"]
        froot.mkdir(parents=True)
        (froot / "FREEZE.json").write_text("{}")
        (froot / "REPORT.json").write_text(json.dumps({
            "contrasts": {"outcomes": {f"{fam['unit']}__v0__{k}__x": {"outcome": outcomes[(fam["unit"], k, h)]}
                                       for k in design["operators"] for h in ("H_T", "H_A")}},
            "reconciliation": {"contrasts": {"missing_units": []}, "calibration": {"missing_units": []}}}))
        (froot / "CC.json").write_text(json.dumps({"all_equal": True}))
    (root / "REPORT.json").write_text(json.dumps({"run_id": "dev", "stopped": stopped, "spent_cpu_seconds": 100.0,
                                                  "cap_seconds": 1000.0, "projection": {"projected_cpu_seconds": 500.0},
                                                  "families": {missing_family: {"incomplete": "CPU_CAP_EXHAUSTED: x"}} if missing_family else {}}))

    def fake_reverify(froot, repo):
        unit = Path(froot).name
        table = [_row(unit, k, h, outcomes[(unit, k, h)]) for k in design["operators"] for h in ("H_T", "H_A")]
        return {"all_verified": True, "table": table, "decision_delta": [],
                "contrasts": {t["contrast"]: {"original_outcome": t["reverified_outcome"]} for t in table}}
    monkeypatch.setattr(C.RV, "reverify", fake_reverify)
    return root, design


def test_P4_a_pair_is_proposed_only_when_it_advances_in_selection_and_its_mapped_replica(tmp_path, monkeypatch):
    units = ["bumps__s12", "bumps__s13", "sinusoid__s12", "sinusoid__s13"]
    outcomes = {(u, k, h): H.DOES_NOT_ADVANCE for u in units for k in DESIGN.OPS for h in ("H_T", "H_A")}
    outcomes[("bumps__s12", "cusum_causal", "H_A")] = H.ADVANCES
    outcomes[("bumps__s13", "cusum_causal", "H_A")] = H.ADVANCES           # replicated
    outcomes[("sinusoid__s12", "cusum_causal", "H_A")] = H.ADVANCES
    outcomes[("sinusoid__s13", "cusum_causal", "H_A")] = H.INCONCLUSIVE_UNCALIBRATED   # not replicated
    outcomes[("bumps__s13", "delta_run_length", "H_T")] = H.ADVANCES      # replica only
    root, design = _fixture(tmp_path, monkeypatch, outcomes)
    out = C.close(root, design, tmp_path, "CC.json")
    assert out["proposed_for_review"] == [{"operator": "cusum_causal", "hypothesis": "H_A",
                                          "selection": "bumps__s12", "replica": "bumps__s13"}]
    by = {(r["operator"], r["hypothesis"], r["selection_unit"]): r for r in out["table"]}
    assert by[("cusum_causal", "H_A", "sinusoid__s12")]["verdict"] == C.NOT_PROPOSED
    assert by[("delta_run_length", "H_T", "bumps__s12")]["verdict"] == C.NOT_PROPOSED
    assert by[("cusum_causal", "H_A", "bumps__s12")]["replica_unit"] == "bumps__s13"
    assert len(out["table"]) == 12
    assert out["checks"] == {"files_verified": True, "parent_equals_files": True,
                             "accounting_reconciled": True, "warehouse_content_equal": True}
    assert "PROPOSED_FOR_REVIEW is not confirmation" in out["reading"]
    md = C.markdown(out)
    assert "Proposed for review: 1." in md and "PROPOSED_FOR_REVIEW" in md


def test_P4_an_incomplete_family_is_reported_as_such_and_checks_fail(tmp_path, monkeypatch):
    units = ["bumps__s12", "bumps__s13", "sinusoid__s12", "sinusoid__s13"]
    outcomes = {(u, k, h): H.ADVANCES for u in units for k in DESIGN.OPS for h in ("H_T", "H_A")}
    root, design = _fixture(tmp_path, monkeypatch, outcomes, stopped="CPU_CAP_EXHAUSTED: x", missing_family="sinusoid__s13")
    out = C.close(root, design, tmp_path, "CC.json")
    row = [r for r in out["table"] if r["selection_unit"] == "sinusoid__s12"][0]
    assert row["replica_outcome"].startswith("CPU_CAP_EXHAUSTED") and row["verdict"] == C.NOT_PROPOSED
    assert out["checks"]["files_verified"] is False and out["stopped"].startswith("CPU_CAP")
    assert len(out["proposed_for_review"]) == 6          # bumps pairs only
