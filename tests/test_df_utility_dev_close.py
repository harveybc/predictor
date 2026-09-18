"""P4/Q1: the closure is bound to the registered population (design identity, families, members,
pairs, map) and proposes nothing unless every mandatory check of the pair's scope passed.
Empty, omitted, duplicated, unexpected or incoherent → typed refusal, never all([]) == True."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
D = _load("df_utility_next_design")
DESIGN = _load("test_df_utility_next_design", HERE)


def _row(unit, op, h, outcome, delta=-0.1):
    return {"contrast": f"{unit}__v0__{op}__{'augmented' if h == 'H_A' else 'transformed'}", "unit": unit,
            "variable": "v0", "operator": op, "hypothesis": h,
            "loss_raw_mean": 0.2, "loss_transformed_mean": 0.2 - delta, "delta_mean": delta, "delta_lower": delta - 0.05,
            "delta_se": 0.01, "rows_paired": 390, "n": 400, "blocks_used": 4, "cpu_seconds": 1.0,
            "null_scope": {"derived_decision_bound": 0.004}, "reverified_outcome": outcome}


def _fixture(tmp_path, monkeypatch, outcomes, stopped=None, missing_family=None, design=None):
    design = design or DESIGN.design()
    root = tmp_path / "dev"
    root.mkdir(parents=True, exist_ok=True)
    fam_entries = {}
    for fam in design["families"]:
        if fam["unit"] == missing_family:
            fam_entries[fam["unit"]] = {"role": fam["role"], "root": str(root / "families" / fam["unit"]),
                                        "incomplete": "CPU_CAP_EXHAUSTED: x"}
            continue
        froot = root / "families" / fam["unit"]
        froot.mkdir(parents=True, exist_ok=True)
        run_id = f"dev-{fam['role'][:3]}-{fam['unit']}"
        members = [m["contrast_id"] for m in fam["members"]]
        keys = [f"{c['operator']}__{c['hypothesis']}" for c in fam["calibration_contracts"]]
        (froot / "FREEZE.json").write_text(json.dumps({"run_id": run_id, "protocols": {k: {} for k in keys}}))
        (froot / "REPORT.json").write_text(json.dumps({
            "run_id": run_id,
            "calibration": {"campaign": {"key": f"{run_id}-utility-calibration", "campaign_sha256": "a" * 64},
                            **{k: {"outcome": "COMPLETED"} for k in keys}},
            "contrasts": {"campaign": {"key": f"{run_id}-utility-contrasts", "campaign_sha256": "b" * 64},
                          "outcomes": {c: {"outcome": outcomes[(fam["unit"], m["operator"], m["hypothesis"])]}
                                       for c, m in zip(members, fam["members"])}},
            "terminals": [{"unit_id": c, "status": "COMPLETED"} for c in members],
            "reconciliation": {"contrasts": {"http": 200, "missing_units": [], "accounting_only": [], "lake_only": []},
                               "calibration": {"http": 200, "missing_units": [], "accounting_only": [], "lake_only": []}}}))
        (froot / "CC.json").write_text(json.dumps({"schema": "df_utility_content_check.v1", "run_id": run_id, "generation": 1,
                                                   "all_equal": True, "refused": [],
                                                   "units": {u: {"equal": True} for u in members + [f"calibrate__{k}" for k in keys]}}))
        fam_entries[fam["unit"]] = {"role": fam["role"], "replica_of": fam.get("replica_of"), "run_id": run_id, "root": str(froot)}
    (root / "REPORT.json").write_text(json.dumps({"run_id": "dev", "design_sha256": design["design_sha256"], "stopped": stopped,
                                                  "spent_cpu_seconds": 100.0, "cap_seconds": 1000.0,
                                                  "projection": {"projected_cpu_seconds": 500.0}, "families": fam_entries}))

    def fake_reverify(froot, repo):
        unit = Path(froot).name
        fam = [f for f in design["families"] if f["unit"] == unit][0]
        table = [_row(unit, m["operator"], m["hypothesis"], outcomes[(unit, m["operator"], m["hypothesis"])]) for m in fam["members"]]
        return {"all_verified": True, "table": table, "decision_delta": [],
                "calibrations": {f"{c['operator']}__{c['hypothesis']}": {"problems": []} for c in fam["calibration_contracts"]},
                "contrasts": {t["contrast"]: {"original_outcome": t["reverified_outcome"], "problems": []} for t in table}}
    monkeypatch.setattr(C.RV, "reverify", fake_reverify)
    return root, design


UNITS = ["bumps__s12", "bumps__s13", "sinusoid__s12", "sinusoid__s13"]


def _all(outcome):
    return {(u, k, h): outcome for u in UNITS for k in DESIGN.OPS for h in ("H_T", "H_A")}


def _close(root, design, **kw):
    return C.close(root, design, root.parent, "CC.json", inherited_protocol=DESIGN.PILOT, pilot_units=DESIGN.PILOT_UNITS, **kw)


def test_Q1_an_empty_or_unvalidated_design_is_a_typed_refusal_never_all_true(tmp_path):
    (tmp_path / "REPORT.json").write_text(json.dumps({"run_id": "empty"}))
    with pytest.raises(C.ClosureRefusal, match="design"):
        C.close(tmp_path, dict(families=[], replication_map={}, operators=[], hypotheses={}, design_sha256="x"),
                tmp_path, "CC.json", inherited_protocol=DESIGN.PILOT, pilot_units=DESIGN.PILOT_UNITS)


def test_Q1_the_closure_derives_the_population_from_the_design_and_refuses_omissions_duplicates_and_strangers(tmp_path, monkeypatch):
    root, design = _fixture(tmp_path, monkeypatch, _all(H.DOES_NOT_ADVANCE))
    out = _close(root, design)
    assert out["population"] == {"families": 4, "members": 24, "contracts": 24, "pairs": 2}
    report = json.loads((root / "REPORT.json").read_text())
    # the report names another design
    other = dict(report, design_sha256="0" * 64)
    (root / "REPORT.json").write_text(json.dumps(other))
    with pytest.raises(C.ClosureRefusal, match="identity"):
        _close(root, design)
    # a family omitted from the report
    omitted = json.loads(json.dumps(report))
    del omitted["families"]["sinusoid__s13"]
    (root / "REPORT.json").write_text(json.dumps(omitted))
    with pytest.raises(C.ClosureRefusal, match="omitted|missing"):
        _close(root, design)
    # a family the design does not know
    stranger = json.loads(json.dumps(report))
    stranger["families"]["steps__s12"] = dict(stranger["families"]["bumps__s12"])
    (root / "REPORT.json").write_text(json.dumps(stranger))
    with pytest.raises(C.ClosureRefusal, match="not in the design|unexpected"):
        _close(root, design)
    (root / "REPORT.json").write_text(json.dumps(report))
    # a member the design did not foresee in a family's terminals, and a duplicate
    froot = root / "families" / "bumps__s12"
    frep = json.loads((froot / "REPORT.json").read_text())
    frep["terminals"].append({"unit_id": "bumps__s12__v0__cusum_causal__slow", "status": "COMPLETED"})
    (froot / "REPORT.json").write_text(json.dumps(frep))
    with pytest.raises(C.ClosureRefusal, match="unexpected|not a member"):
        _close(root, design)
    frep["terminals"] = frep["terminals"][:-1] + [frep["terminals"][0]]
    (froot / "REPORT.json").write_text(json.dumps(frep))
    with pytest.raises(C.ClosureRefusal, match="duplicate"):
        _close(root, design)


def test_Q1_receipts_are_bound_to_the_campaign_identity_and_the_whole_population_not_only_what_is_present(tmp_path, monkeypatch):
    root, design = _fixture(tmp_path, monkeypatch, _all(H.DOES_NOT_ADVANCE))
    froot = root / "families" / "bumps__s12"
    cc = json.loads((froot / "CC.json").read_text())
    # all_equal True but one member absent from the compared units: population incomplete
    partial = json.loads(json.dumps(cc))
    partial["units"].pop("bumps__s12__v0__cusum_causal__augmented")
    (froot / "CC.json").write_text(json.dumps(partial))
    out = _close(root, design)
    fam = out["families"]["bumps__s12"]
    assert fam["checks"]["warehouse_content_equal"] is False and "cusum_causal__augmented" in json.dumps(fam["problems"])
    # a content check of another run
    foreign = json.loads(json.dumps(cc))
    foreign["run_id"] = "someone-else"
    (froot / "CC.json").write_text(json.dumps(foreign))
    out = _close(root, design)
    assert out["families"]["bumps__s12"]["checks"]["warehouse_content_equal"] is False
    (froot / "CC.json").write_text(json.dumps(cc))
    # missing_units == [] alone does not reconcile: the campaign keys must be this run's
    frep = json.loads((froot / "REPORT.json").read_text())
    frep["contrasts"]["campaign"]["key"] = "other-run-utility-contrasts"
    (froot / "REPORT.json").write_text(json.dumps(frep))
    out = _close(root, design)
    assert out["families"]["bumps__s12"]["checks"]["accounting_reconciled"] is False


def test_Q1_no_proposal_is_emitted_when_a_mandatory_check_of_the_pair_failed(tmp_path, monkeypatch):
    root, design = _fixture(tmp_path, monkeypatch, _all(H.ADVANCES))
    previous = C.RV.reverify

    def failed(*a, **k):
        r = previous(*a, **k)
        r["all_verified"] = False
        return r
    monkeypatch.setattr(C.RV, "reverify", failed)
    out = _close(root, design)
    assert out["checks"]["files_verified"] is False
    assert out["proposed_for_review"] == []
    assert len(out["unverified_candidates"]) == 12
    assert all(r["verdict"] == C.UNVERIFIED_PAIR for r in out["table"])
    assert "PROPOSED_FOR_REVIEW" not in json.dumps(out["table"]) and "PROPOSED_FOR_REVIEW" not in json.dumps(out["proposed_for_review"])
    md = C.markdown(out)
    assert "Proposed for review: 0" in md and "unverified" in md.lower()


def test_Q1_partial_closure_policy_a_pair_needs_both_families_fully_verified(tmp_path, monkeypatch):
    root, design = _fixture(tmp_path, monkeypatch, _all(H.ADVANCES), stopped="CPU_CAP_EXHAUSTED: x", missing_family="sinusoid__s13")
    out = _close(root, design)
    assert out["closure"] == "PARTIAL"
    by = {(r["operator"], r["hypothesis"], r["selection_unit"]): r for r in out["table"]}
    assert by[("cusum_causal", "H_A", "bumps__s12")]["verdict"] == C.PROPOSED
    assert by[("cusum_causal", "H_A", "sinusoid__s12")]["verdict"] == C.UNVERIFIED_PAIR
    assert len(out["proposed_for_review"]) == 6 and all(p["selection"] == "bumps__s12" for p in out["proposed_for_review"])
    assert out["families"]["sinusoid__s13"]["status"].startswith("CPU_CAP_EXHAUSTED")
    assert out["checks"]["files_verified"] is False


def test_Q1_the_cli_refuses_an_incoherent_design_and_carries_no_proposal_on_failure(tmp_path, monkeypatch):
    root, design = _fixture(tmp_path, monkeypatch, _all(H.ADVANCES))
    dpath = tmp_path / "design.json"
    dpath.write_text(json.dumps(design))
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    (pilot / "FREEZE.pre.json").write_text(json.dumps({"protocol_base": DESIGN.PILOT, "units": [{"unit": u} for u in DESIGN.PILOT_UNITS]}))
    previous = C.RV.reverify
    monkeypatch.setattr(C.RV, "reverify", lambda *a, **k: {**previous(*a, **k), "all_verified": False})
    code = C.main(["--root", str(root), "--design", str(dpath), "--pilot-root", str(pilot), "--repo", str(tmp_path),
                   "--content-checks", "CC.json", "--out", "CLOSE.json"])
    assert code != 0
    written = json.loads((root / "CLOSE.json").read_text())
    assert written["proposed_for_review"] == [] and len(written["unverified_candidates"]) == 12
    bad = json.loads(json.dumps(design))
    bad["families"] = []
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="REFUSED"):
        C.main(["--root", str(root), "--design", str(tmp_path / "bad.json"), "--pilot-root", str(pilot),
                "--repo", str(tmp_path), "--content-checks", "CC.json", "--out", "CLOSE.2.json"])
    assert not (root / "CLOSE.2.json").exists()


def test_P4_a_pair_is_proposed_only_when_it_advances_in_selection_and_its_mapped_replica(tmp_path, monkeypatch):
    outcomes = _all(H.DOES_NOT_ADVANCE)
    outcomes[("bumps__s12", "cusum_causal", "H_A")] = H.ADVANCES
    outcomes[("bumps__s13", "cusum_causal", "H_A")] = H.ADVANCES
    outcomes[("sinusoid__s12", "cusum_causal", "H_A")] = H.ADVANCES
    outcomes[("sinusoid__s13", "cusum_causal", "H_A")] = H.INCONCLUSIVE_UNCALIBRATED
    outcomes[("bumps__s13", "delta_run_length", "H_T")] = H.ADVANCES
    root, design = _fixture(tmp_path, monkeypatch, outcomes)
    out = _close(root, design)
    assert out["proposed_for_review"] == [{"operator": "cusum_causal", "hypothesis": "H_A",
                                          "selection": "bumps__s12", "replica": "bumps__s13"}]
    assert out["closure"] == "TOTAL" and all(v is True for v in out["checks"].values())
    assert "PROPOSED_FOR_REVIEW is not confirmation" in out["reading"]
