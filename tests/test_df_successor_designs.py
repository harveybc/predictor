"""C141-C143: successor designs D3, D4 and D5 with no scores.

Builders validate; every refusal is triggered by a mutation that is
re-hashed first, so the refusal comes from the rule and not from the
digest; the CLI is write-once; protocol digests are re-derived from disk."""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("df_successor_designs",
                                                  ROOT / "tools/df_successor_designs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()
DOCS = M.build_all()
IDS = ("D3", "D4", "D5")


def mutated(did, fn, *, rehash=True):
    d = copy.deepcopy(DOCS[did])
    fn(d)
    if rehash:
        d["design_sha256"] = M.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    return d


def refused(did, fn, token, *, rehash=True, repo_root=M.REPO):
    problems = M.validate(mutated(did, fn, rehash=rehash), repo_root=repo_root)
    assert any(token in x for x in problems), (token, problems)
    return problems


def arm(d, arm_id):
    return next(a for a in d["arms"] if a["arm_id"] == arm_id)


# ------------------------------------------------------------------ builders
@pytest.mark.parametrize("did", IDS)
def test_builder_validates(did):
    assert M.validate(DOCS[did]) == []


@pytest.mark.parametrize("did", IDS)
def test_builder_is_deterministic_and_declares_no_scores(did):
    d = M.BUILDERS[did]()
    assert d == DOCS[did]
    assert d["status"] == "DESIGN_NO_SCORES_COMPUTED"
    assert d["license"]["scoring"] == "NOT_GRANTED"
    assert d["decision_vocabulary"]["decisions"] == list(M.LAB_DECISIONS)


@pytest.mark.parametrize("did", IDS)
def test_protocols_bound_by_disk_sha(did):
    for e in DOCS[did]["protocols"]:
        assert e["sha256"] == M.file_sha256(ROOT / e["file"])
        assert not Path(e["file"]).is_absolute()


@pytest.mark.parametrize("did", IDS)
def test_committed_evidence_matches_builder(did):
    path = ROOT / M.EVIDENCE_DIR / M.SPECS[did]["file"]
    if not path.exists():
        pytest.skip("evidence not written yet")
    assert M.validate_file(path) == []
    assert M.strict_json_loads(path.read_bytes()) == DOCS[did]


# ----------------------------------------------------------- design digest
@pytest.mark.parametrize("did", IDS)
def test_design_sha_must_rederive(did):
    refused(did, lambda d: d["questions"][0].update(question="changed"),
            "DESIGN_SHA256_DOES_NOT_REDERIVE", rehash=False)
    refused(did, lambda d: d.update(design_sha256="0" * 64), "DESIGN_SHA256_DOES_NOT_REDERIVE",
            rehash=False)


# ---------------------------------------------------------- raw control
@pytest.mark.parametrize("did", IDS)
def test_missing_raw_arm_refused(did):
    def drop(d):
        d["arms"] = [a for a in d["arms"] if a["arm_id"] != M.RAW_ARM]
    refused(did, drop, "MISSING_RAW_CONTROL")


@pytest.mark.parametrize("did", IDS)
def test_raw_control_not_kept_refused(did):
    refused(did, lambda d: d["raw_control"].update(kept=False), "MISSING_RAW_CONTROL")
    refused(did, lambda d: d["raw_control"].update(replaced_by_transform=True), "MISSING_RAW_CONTROL")
    refused(did, lambda d: d.pop("raw_control"), "MISSING_RAW_CONTROL")


@pytest.mark.parametrize("did", IDS)
def test_raw_control_kind_missing_refused(did):
    def drop(d):
        d["controls"] = [c for c in d["controls"] if c["kind"] != "RAW"]
    refused(did, drop, "MISSING_RAW_CONTROL")


@pytest.mark.parametrize("did", IDS)
def test_null_and_negative_controls_required(did):
    for kind in ("NULL_SURROGATE", "NEGATIVE_CONTROL_OPERATOR"):
        def drop(d, kind=kind):
            d["controls"] = [c for c in d["controls"] if c["kind"] != kind]
        refused(did, drop, f"MISSING_CONTROL: {kind}")


# ------------------------------------------------------------ D4 availability
def test_d4_availability_block_missing_refused():
    refused("D4", lambda d: d.pop("temporal_availability"), "D4_AVAILABILITY_RULE_MISSING")


def test_d4_availability_not_first_refused():
    refused("D4", lambda d: d["temporal_availability"].update(availability_precedes_alignment=False),
            "D4_AVAILABILITY_RULE_MISSING")
    refused("D4", lambda d: d["temporal_availability"]["order"].reverse(), "D4_AVAILABILITY_RULE_MISSING")
    refused("D4", lambda d: d["temporal_availability"].update(rule="lags are corrected first"),
            "D4_AVAILABILITY_RULE_MISSING")


def test_d4_future_samples_refused():
    refused("D4", lambda d: d["temporal_availability"].update(future_samples_in_correction="ALLOWED"),
            "D4_FUTURE_SAMPLES_ALLOWED")
    refused("D4", lambda d: arm(d, "L10_XCORR_TRAIN_LAG").update(causal_contract="ENDPOINT_CAUSAL"),
            "D4_FUTURE_SAMPLES_ALLOWED")


def test_d4_arm_without_verified_availability_refused():
    refused("D4", lambda d: arm(d, "L10_GCC_PHAT_TRAIN_LAG")["requires"].remove(M.VERIFIED_AVAILABILITY),
            "D4_AVAILABILITY_RULE_MISSING")


# ------------------------------------------------------- D5 budget/abstention
def test_d5_block_missing_refused():
    refused("D5", lambda d: d.pop("budget_and_abstention"), "D5_BUDGET_OR_ABSTENTION_MISSING")


def test_d5_abstention_disabled_refused():
    refused("D5", lambda d: d["budget_and_abstention"]["abstention"].update(allowed=False),
            "D5_BUDGET_OR_ABSTENTION_MISSING")
    refused("D5", lambda d: d["budget_and_abstention"].pop("abstention"), "D5_BUDGET_OR_ABSTENTION_MISSING")


def test_d5_abstention_not_to_raw_refused():
    refused("D5", lambda d: d["budget_and_abstention"]["abstention"].update(target_arm="P12_FIXED_MODE"),
            "D5_ABSTENTION_NOT_TO_RAW")
    refused("D5", lambda d: d["budget_and_abstention"]["abstention"].update(
        target_branch="B1_D2_REVIEWED_VIEWS"), "D5_ABSTENTION_NOT_TO_RAW")


def test_d5_branch_cost_cap_required():
    refused("D5", lambda d: d["budget_and_abstention"]["branches"][1].update(cost_cap=0),
            "D5_BRANCH_WITHOUT_COST_CAP")
    refused("D5", lambda d: d["budget_and_abstention"]["branches"][1].update(cost_cap=True),
            "D5_BRANCH_WITHOUT_COST_CAP")

    def uncapped(d):
        for b in d["budget_and_abstention"]["branches"]:
            if b["branch_id"] != "B0_RAW":
                b["arm_ids"] = [x for x in b["arm_ids"] if x != "B13_RESERVE"]
        d["budget_and_abstention"]["branches"][0]["arm_ids"].remove("B13_RESERVE")
    refused("D5", uncapped, "D5_BRANCH_WITHOUT_COST_CAP")
    refused("D5", lambda d: d["budget_and_abstention"]["branches"][1].pop("cost_cap"), "KEYS")


# --------------------------------------------------------------- no scores
@pytest.mark.parametrize("did", IDS)
def test_metric_value_field_refused(did):
    refused(did, lambda d: d["metrics"][0].update(value=0.123), "RESULT_FIELD_IN_DESIGN")
    refused(did, lambda d: d["metrics"][0].update(value=0.123), "NUMERIC_VALUE_OUTSIDE_BUDGET_CAPS")
    refused(did, lambda d: d["metrics"][0].update(score="0.9"), "RESULT_FIELD_IN_DESIGN")


@pytest.mark.parametrize("did", IDS)
def test_numeric_outside_budget_refused(did):
    refused(did, lambda d: d["hypotheses"][0].update(falsified_if=0.05), "NUMERIC_VALUE_OUTSIDE_BUDGET_CAPS")
    refused(did, lambda d: d["arms"][1].update(requires=[]) or d["arms"][1].update(parameters=[3]),
            "NUMERIC_VALUE_OUTSIDE_BUDGET_CAPS")


@pytest.mark.parametrize("did", IDS)
def test_budget_types_strict(did):
    refused(did, lambda d: d["budget"].update(cpu_core_hours_cap=True), "BUDGET")
    refused(did, lambda d: d["budget"].update(accelerator="GPU"), "BUDGET")


# --------------------------------------------------------- public eligibility
@pytest.mark.parametrize("did", IDS)
def test_publicly_eligible_as_decision_refused(did):
    refused(did, lambda d: d["decision_vocabulary"]["decisions"].append("PUBLICLY_ELIGIBLE"),
            "PUBLICLY_ELIGIBLE_AS_DECISION_OR_STATE")
    refused(did, lambda d: d["stopping_rules"][0].update(action="PUBLICLY_ELIGIBLE"),
            "PUBLICLY_ELIGIBLE_AS_DECISION_OR_STATE")
    refused(did, lambda d: d["arms"][1].update(description="arm becomes PUBLICLY_ELIGIBLE on success"),
            "PUBLICLY_ELIGIBLE_AS_DECISION_OR_STATE")
    refused(did, lambda d: d["failure_region_reporting"]["states_recorded"].append(
        "PUBLICLY_ELIGIBLE, never granted"), "PUBLICLY_ELIGIBLE_AS_DECISION_OR_STATE")


@pytest.mark.parametrize("did", IDS)
def test_publicly_eligible_only_in_never_grants_statements(did):
    text = json.dumps(DOCS[did])
    assert "PUBLICLY_ELIGIBLE" in text
    for path, node in M._walk(DOCS[did]):
        if isinstance(node, str) and "PUBLICLY_ELIGIBLE" in node:
            assert M.NEVER_GRANTS.search(node), path


# --------------------------------------------------------------- consumption
@pytest.mark.parametrize("did", IDS)
@pytest.mark.parametrize("state", ["NOT_IDENTIFIABLE", "LAB_REJECTED", "UNREVIEWED"])
def test_consuming_rejected_or_unreviewed_state_refused(did, state):
    refused(did, lambda d: d["consumes"]["accepted_d2_decisions"].append(state),
            "CONSUMES_UNACCEPTABLE_D2_STATE")


@pytest.mark.parametrize("did", IDS)
def test_consuming_without_review_refused(did):
    refused(did, lambda d: d["consumes"].update(external_review_record_required=False), "CONSUMES_UNREVIEWED")
    refused(did, lambda d: d["consumes"].update(unreviewed_d2_output="CONSUMED"), "CONSUMES_UNREVIEWED")
    refused(did, lambda d: d["consumes"]["never_consumed"].remove("LAB_REJECTED"), "CONSUMES_RULE")
    refused(did, lambda d: d["arms"][1]["requires"].append("D2_LAB_REJECTED_OPERATOR"),
            "CONSUMES_UNACCEPTABLE_D2_STATE")


# ------------------------------------------------------------------ protocols
@pytest.mark.parametrize("did", IDS)
def test_protocol_sha_mismatch_refused(did):
    refused(did, lambda d: d["protocols"][0].update(sha256="f" * 64), "PROTOCOL_SHA_MISMATCH")


@pytest.mark.parametrize("did", IDS)
def test_protocol_sha_rechecked_against_disk(did, tmp_path):
    for e in DOCS[did]["protocols"]:
        dst = tmp_path / e["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / e["file"], dst)
    assert M.validate(DOCS[did], repo_root=tmp_path) == []
    first = tmp_path / DOCS[did]["protocols"][0]["file"]
    first.write_bytes(first.read_bytes() + b"\n")
    problems = M.validate(DOCS[did], repo_root=tmp_path)
    assert any("PROTOCOL_SHA_MISMATCH" in x for x in problems), problems
    first.unlink()
    assert any("PROTOCOL_MISSING_ON_DISK" in x for x in M.validate(DOCS[did], repo_root=tmp_path))


@pytest.mark.parametrize("did", IDS)
def test_protocol_for_each_step_required(did):
    refused(did, lambda d: d["protocols"].pop(0), "PROTOCOL_NOT_CITED_FOR")
    refused(did, lambda d: d["protocols"][0].update(file="/abs/" + d["protocols"][0]["file"]),
            "PROTOCOL_PATH_NOT_RELATIVE")


# ------------------------------------------------------------- strictness
@pytest.mark.parametrize("did", IDS)
def test_strict_keys_and_types(did):
    refused(did, lambda d: d.update(extra="x"), "KEYS")
    refused(did, lambda d: d["arms"][1].update(note="x"), "KEYS")
    refused(did, lambda d: d["arms"][1].update(usable_as_input="yes"), "TYPE")
    refused(did, lambda d: d["metrics"][0].update(estimator=""), "METRIC_WITHOUT_ESTIMATOR")
    refused(did, lambda d: d["hypotheses"][0].update(falsified_if=""), "HYPOTHESIS_NOT_FALSIFIABLE")
    refused(did, lambda d: d.update(status="SCORED"), "IDENTITY")


@pytest.mark.parametrize("did", IDS)
def test_partitions_and_sealed_periods(did):
    refused(did, lambda d: d["partitions"].update(sealed_periods="READ"), "PARTITIONS")
    refused(did, lambda d: d["partitions"].update(target_use="SCREENING"), "PARTITIONS")
    refused(did, lambda d: d["synthetic_calibration_first"]["order"].reverse(), "SYNTHETIC_CALIBRATION_FIRST")


@pytest.mark.parametrize("did", IDS)
def test_failure_regions_and_license(did):
    refused(did, lambda d: d["failure_region_reporting"].update(best_case_only="ALLOWED"),
            "FAILURE_REGION_REPORTING")
    refused(did, lambda d: d["license"].update(scoring="GRANTED"), "LICENSE")


@pytest.mark.parametrize("did", IDS)
def test_absolute_home_path_refused(did):
    refused(did, lambda d: d["arms"][1].update(description=str(Path.home()) + "/data.csv"),
            "ABSOLUTE_HOME_PATH")


def test_d3_non_causal_arm_as_input_refused():
    def flip(d):
        arm(d, "T06_CENTERED_CWT")["usable_as_input"] = True
        next(e for e in d["causal_feasibility"]["entries"]
             if e["arm_id"] == "T06_CENTERED_CWT")["usable_as_input"] = True
    refused("D3", flip, "NON_CAUSAL_USED_AS_INPUT")
    refused("D3", lambda d: d["causal_feasibility"].update(rule="transforms may be centered"),
            "D3_NON_CAUSAL_RULE_MISSING")
    refused("D3", lambda d: d.pop("causal_feasibility"), "D3_NON_CAUSAL_RULE_MISSING")
    refused("D3", lambda d: d["causal_feasibility"]["entries"].pop(0), "D3_FEASIBILITY_NOT_DECLARED_FOR")
    refused("D3", lambda d: arm(d, "Q04_QUANTILE").update(fit="NONE"), "D3_QUANTIZER_NOT_TRAIN_FROZEN")


def test_strict_json_refuses_duplicates_and_nan(tmp_path):
    f = tmp_path / "x.json"
    f.write_text('{"a": 1, "a": 2}')
    assert "STRICT_JSON" in M.validate_file(f)[0]
    f.write_text('{"a": NaN}')
    assert "STRICT_JSON" in M.validate_file(f)[0]


# ------------------------------------------------------------------------ CLI
def test_cli_write_once(tmp_path, capsys):
    out = tmp_path / "evidence"
    assert M.main(["--write-dir", str(out)]) == 0
    written = {k: out / s["file"] for k, s in M.SPECS.items()}
    before = {k: p.read_bytes() for k, p in written.items()}
    shas = json.loads(capsys.readouterr().out)["design_sha256"]
    assert shas == {k: DOCS[k]["design_sha256"] for k in IDS}
    assert M.main(["--validate", *map(str, written.values())]) == 0
    capsys.readouterr()
    assert M.main(["--write-dir", str(out)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert {k: p.read_bytes() for k, p in written.items()} == before


def test_cli_refuses_when_any_one_target_exists(tmp_path, capsys):
    out = tmp_path / "evidence"
    out.mkdir()
    (out / M.SPECS["D4"]["file"]).write_text("{}")
    assert M.main(["--write-dir", str(out)]) == 2
    assert not (out / M.SPECS["D3"]["file"]).exists()
    assert not (out / M.SPECS["D5"]["file"]).exists()
    assert (out / M.SPECS["D4"]["file"]).read_text() == "{}"


def test_cli_validate_reports_problems(tmp_path, capsys):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(mutated("D3", lambda d: d["metrics"][0].update(value=1.0))))
    assert M.main(["--validate", str(f)]) == 1
