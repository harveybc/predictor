"""The E1 seal must refuse rather than mislead, and its states must follow evidence.

A seal is only worth the refusals behind it. These tests build a throwaway copy of
exactly the artifacts `tools/df_e1_seal.py` declares, then break one thing at a time
and require the seal to refuse or to change a condition's state. Nothing here fits,
loads, scores or replays a model; the fixtures copy small JSON and Markdown files.

Written to the same shape as the other closure tests in this suite: the positive case
first, then the refusals, then the mutations that prove each state is load-bearing.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools/df_e1_seal.py"
EV = "docs/audits/evidence/d3_k5_20260917"
PROG = "docs/tres_temas_entrevista/program_v3"


def _load():
    spec = importlib.util.spec_from_file_location("df_e1_seal_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S():
    return _load()


@pytest.fixture()
def repo(S, tmp_path) -> Path:
    """A throwaway repository holding exactly the declared artifacts, plus the
    block root the seal claims produced no outcome."""
    root = tmp_path / "repo"
    for _kind, rel, _why in S.INVENTORY:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    for rel, _name, _state in S.NO_OUTCOME_BLOCKS:
        src, dst = REPO / rel, root / rel
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("REPORT*.json"):
            shutil.copy2(p, dst / p.name)
    # the ruling on the reserved external review is read from these, so the
    # throwaway repository carries them too: the fixture declares everything the
    # seal reads, and nothing it does not.
    for rel in (*S.STANDING_IN_DISPOSITIONS, S.PROGRAMME_STATE):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    return root


def _state(repo: Path, S):
    return json.loads((repo / S.PROGRAMME_STATE).read_text())


def _put_state(repo: Path, S, doc) -> None:
    (repo / S.PROGRAMME_STATE).write_text(json.dumps(doc, indent=1))


def _write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=1))


def _design_path(S, name: str) -> str:
    return next(rel for kind, rel, _ in S.INVENTORY
                if kind == "design" and rel.endswith(name))


# ------------------------------------------------------------- the real thing --

def test_the_seal_over_the_real_repository_is_partial_and_names_its_gaps(S):
    doc = S.seal(REPO)
    assert doc["verdict"] == S.PARTIAL
    states = {c["id"]: c["state"] for c in doc["conditions"]}
    assert states == {
        "RP17_RP24_NO_SEAL_13E_V1": S.OBSERVED,
        "ML_BASELINES_CAUSALITY_UNVERIFIED": S.OUTSIDE,
        "ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR": S.UNMET,
        "HUBER_NO_RETURN_TO_R0_R1_R2_YET": S.OBSERVED,
        "POST_HUBER_PHASE2_CORRECTED": S.DISCHARGED,
        "POST_HUBER_EXTERNAL_ACCEPTANCE": S.OUTSIDE,
        "MOD_E1_EXTERNAL_REVIEW": S.BY_GRANT,
        "OWNER_CLOSURE_TABLE_FROM_ARTIFACTS": S.DISCHARGED,
        "RP136_RP139_MATCHED_ECL_ADAPTER": S.UNMET,
    }
    assert doc["gaps"] == [c["id"] for c in doc["conditions"]
                           if c["state"] in (S.UNMET, S.OUTSIDE, S.BY_GRANT)]
    # every gap says what would discharge it, so the reader is never left guessing
    for c in doc["conditions"]:
        if c["state"] in (S.UNMET, S.OUTSIDE, S.BY_GRANT):
            assert (c.get("what_would_discharge_it")
                    or c.get("what_would_discharge_it_fully")
                    or c.get("evidence"))


def test_every_declared_identity_is_recomputed_not_read_back(S):
    doc = S.seal(REPO)
    designs = {rel: e for rel, e in doc["sealed"]["identity"].items()
               if e["kind"] == "design"}
    assert len(designs) >= 12
    for rel, e in designs.items():
        assert e["design_sha256_recomputed"] == e["design_sha256"], rel


def test_the_numbers_carry_their_qualifications(S):
    nums = S.seal(REPO)["sealed"]["numbers_as_measured"]
    for rel, c in nums["closures"].items():
        # ALL_VERIFIED is only readable beside the facts that are NOT_APPLICABLE
        assert c["verdict"] == "ALL_VERIFIED"
        na = c["facts_not_applicable"]
        assert c["counts"]["metrics_verified"] == c["counts"]["declared"]
        assert (c["counts"]["inference_verified"] + len(na["inference"])
                == c["counts"]["declared"]), rel
        assert (c["counts"]["regime_verified"] + len(na["regime"])
                == c["counts"]["declared"]), rel
    assert [b["block"] for b in nums["blocks_without_an_outcome"]] \
        == ["Q2_CONTEXT"]
    assert nums["blocks_without_an_outcome"][0]["cells_with_an_outcome"] == 0


def test_the_owner_closure_table_is_bound_with_model_and_naive_on_the_same_rows(
        S):
    tbl = S.seal(REPO)["sealed"]["numbers_as_measured"]["owner_closure_table"]
    assert tbl["schema"] == "owner_closure_table.v2"
    assert tbl["problems"] == []
    assert tbl["rows_missing_a_required_column"] == []
    assert tbl["rows_total"] == (tbl["verified_rows"]
                                + tbl["preserved_qualified_rows"])
    for key, r in tbl["per_arm"].items():
        assert r["comparability_status"] == ["NOT_COMPARABLE"], key
        assert r["mean_model_error_kw"] is not None
        assert r["mean_naive_error_kw"] is not None
        assert r["mean_skill_vs_naive"] is not None
        assert len(r["evaluation_rows"]) == 1


def test_a_design_digest_with_no_retained_artifact_is_disclosed_never_hidden(S):
    doc = S.seal(REPO)
    disclosed = doc["disclosures"][
        "design_digests_without_a_retained_artifact"]
    assert set(disclosed) == set(S.DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT)
    tbl = doc["sealed"]["numbers_as_measured"]["owner_closure_table"]
    unretained = [run for run, r in tbl["runs"].items()
                  if not r["design_artifact_retained_here"]]
    assert unretained == ["huber_adamw_musashi"]


# ------------------------------------------------------------- the copy works --

def test_the_throwaway_copy_reproduces_the_seal_exactly(S, repo):
    a, b = S.seal(REPO), S.seal(repo)
    assert a["verdict"] == b["verdict"] and a["gaps"] == b["gaps"]
    assert ({c["id"]: c["state"] for c in a["conditions"]}
            == {c["id"]: c["state"] for c in b["conditions"]})


# ------------------------------------------------------------------ refusals --

def test_an_absent_declared_artifact_is_a_refusal(S, repo):
    (repo / _design_path(S, "PHASE1_DESIGN_SEALED.json")).unlink()
    with pytest.raises(S.SealRefusal, match="absent from disk"):
        S.seal(repo)


def test_an_edited_design_never_enters_the_seal(S, repo):
    rel = _design_path(S, "PHASE2_DESIGN_SEALED_v2.json")
    doc = json.loads((repo / rel).read_text())
    doc["purpose"] = "something else entirely"
    _write_json(repo / rel, doc)
    with pytest.raises(S.SealRefusal, match="does not re-derive"):
        S.seal(repo)


def test_a_repaired_declaration_does_not_launder_an_edited_design(S, repo):
    """Editing the body AND rewriting design_sha256 to match is the obvious
    attack. The seal recomputes with the canonical rule, so the pair is
    self-consistent but the digest is no longer the one the chain cites."""
    rel = _design_path(S, "PHASE1_DESIGN_SEALED.json")
    doc = json.loads((repo / rel).read_text())
    original = doc["design_sha256"]
    doc["purpose"] = "quietly different"
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    doc["design_sha256"] = S.sha_obj(body)
    _write_json(repo / rel, doc)
    # the design itself now re-derives, but the closure table cites the old
    # digest, so the seal refuses on the binding instead of passing
    with pytest.raises(S.SealRefusal, match="neither retained.*nor declared"):
        S.seal(repo)
    assert doc["design_sha256"] != original


def test_a_closure_that_binds_a_design_outside_the_inventory_is_a_refusal(
        S, repo):
    rel = next(r for kind, r, _ in S.INVENTORY if kind == "closure")
    doc = json.loads((repo / rel).read_text())
    doc["design_sha256"] = "f" * 64
    _write_json(repo / rel, doc)
    with pytest.raises(S.SealRefusal, match="not in the\\s+sealed inventory"):
        S.seal(repo)


def test_claiming_a_design_is_unretained_when_it_is_retained_is_a_refusal(
        S, repo):
    """The seal's own disclosure must be true: if the Huber design turned up
    here, the disclosure would be a lie and the seal refuses."""
    rel = next(r for kind, r, _ in S.INVENTORY
               if kind == "report_unbound_design")
    digest = next(iter(S.DESIGN_DIGESTS_WITHOUT_A_RETAINED_ARTIFACT))
    fake = repo / f"{EV}/RP66/PHASE2_DESIGN_SEALED_v2.json"
    doc = json.loads(fake.read_text())
    doc["design_sha256"] = digest
    _write_json(fake, doc)
    with pytest.raises(S.SealRefusal):
        S.seal(repo)
    assert rel  # the declaration exists and is what was contradicted


def test_a_closure_table_run_binding_an_unknown_design_is_a_refusal(S, repo):
    rel = next(r for kind, r, _ in S.INVENTORY if kind == "closure_table")
    doc = json.loads((repo / rel).read_text())
    run = next(iter(doc["runs"]))
    doc["runs"][run]["design_sha256"] = "a" * 64
    _write_json(repo / rel, doc)
    with pytest.raises(S.SealRefusal, match="neither retained"):
        S.seal(repo)


def test_a_block_declared_without_an_outcome_that_has_one_is_a_refusal(S, repo):
    rel = S.NO_OUTCOME_BLOCKS[0][0]
    _write_json(repo / rel / "REPORT.json", {"schema": "x", "block": "Q2"})
    with pytest.raises(S.SealRefusal, match="REPORT.json\\s+exists"):
        S.seal(repo)


def test_a_model_and_naive_that_do_not_share_rows_are_never_sealed(S, repo):
    rel = next(r for kind, r, _ in S.INVENTORY if kind == "closure_table")
    doc = json.loads((repo / rel).read_text())
    doc["rows"][0]["naive_population"] = doc["rows"][0]["model_population"] + 1
    _write_json(repo / rel, doc)
    with pytest.raises(S.SealRefusal, match="do not share"):
        S.seal(repo)


def test_a_closure_table_row_missing_a_required_column_flips_the_condition(
        S, repo):
    rel = next(r for kind, r, _ in S.INVENTORY if kind == "closure_table")
    doc = json.loads((repo / rel).read_text())
    doc["rows"][0]["literature_value_and_source"] = None
    _write_json(repo / rel, doc)
    out = S.seal(repo)
    state = {c["id"]: c["state"] for c in out["conditions"]}
    assert state["OWNER_CLOSURE_TABLE_FROM_ARTIFACTS"] == S.UNMET


def test_an_empty_closure_table_is_a_refusal(S, repo):
    rel = next(r for kind, r, _ in S.INVENTORY if kind == "closure_table")
    doc = json.loads((repo / rel).read_text())
    doc["rows"] = []
    _write_json(repo / rel, doc)
    with pytest.raises(S.SealRefusal, match="carries no rows"):
        S.seal(repo)


# --------------------------------------------- the states follow the evidence --

def test_sealing_13e_v1_turns_the_prohibition_from_observed_to_unmet(S, repo):
    sheet = repo / f"{PROG}/13E_E1_TASK_SHEET_2026_09_19.md"
    sheet.write_text(sheet.read_text().replace(
        "to be sealed after review", "SEALED"))
    state = {c["id"]: c["state"] for c in S.seal(repo)["conditions"]}
    assert state["RP17_RP24_NO_SEAL_13E_V1"] == S.UNMET


def test_an_outcome_for_the_context_block_would_discharge_the_factor_condition(
        S, repo):
    """The only thing standing between UNMET and DISCHARGED here is a fit that
    nobody has paid for. Proving that in a test is cheaper than running it."""
    rel, name, _state = S.NO_OUTCOME_BLOCKS[0]
    src = REPO / f"{EV}/RP82/blocks/e1_block_q3_volume_v1/REPORT.json"
    shutil.copy2(src, repo / rel / "REPORT.json")
    object.__setattr__(S, "NO_OUTCOME_BLOCKS", ())
    try:
        state = {c["id"]: c["state"] for c in S.seal(repo)["conditions"]}
        assert state["ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR"] \
            == S.DISCHARGED
    finally:
        S.NO_OUTCOME_BLOCKS = (
            (rel, name, "BUDGET_LIMITED_BEFORE_ANY_OUTCOME"),)


def test_an_r0_r1_r2_arm_after_huber_turns_the_prohibition_to_unmet(S, repo):
    rel = f"{EV}/RP82/blocks/e1_block_q3_volume_v1/REPORT.json"
    doc = json.loads((repo / rel).read_text())
    doc["summary"]["R0"] = {"n_seeds": 3, "mean_mae_kw": 0.5}
    _write_json(repo / rel, doc)
    state = {c["id"]: c["state"] for c in S.seal(repo)["conditions"]}
    assert state["HUBER_NO_RETURN_TO_R0_R1_R2_YET"] == S.UNMET


def _cond(S, repo):
    return next(c for c in S.seal(repo)["conditions"]
                if c["id"] == "MOD_E1_EXTERNAL_REVIEW")


def test_the_missing_external_review_condition_reads_the_absence(S, repo):
    for rel in S.EXPECTED_ABSENT_REVIEWS:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# a review that did not exist when the seal was written\n")
    cond = _cond(S, repo)
    # the state goes back to OUTSIDE — only a review's CONTENT can discharge it,
    # and this seal never reads a review's content as acceptance — but the
    # evidence line must stop claiming they are absent
    assert cond["state"] == S.OUTSIDE
    assert "absent []" in cond["evidence"]
    assert cond["blocks_module_dispatch"] is True


# ------------- the reserved external review: the ruling, its scope, its limits --

def test_the_current_ruling_is_derived_from_bytes_not_from_a_constant(S, repo):
    """The defect this replaces: the row was the literal OUTSIDE, so a programme
    state that declared the requirement discharged could not change it."""
    cond = _cond(S, repo)
    assert cond["state"] == S.BY_GRANT
    r = cond["ruling"]
    assert "MOD_E1_EXTERNAL_REVIEW_DISCHARGED" in r["ruling"]
    # identity, recomputed here from the documents' own bytes
    for rel in S.STANDING_IN_DISPOSITIONS:
        ident = r["identity"][rel]
        assert ident["file_sha256"] == S.sha_file(repo / rel)
        assert ident["bytes"] == (repo / rel).stat().st_size
    assert r["audited_commit"]
    # the scope is stated, and it stops where the reviewer begins
    assert "not the reviewer's signature" in r["scope"]
    assert "MOD-CONF" in r["scope"]
    assert r["what_would_discharge_it_fully"]


def test_the_grant_discharges_dispatch_and_never_promotes_the_seal(S, repo):
    doc = S.seal(repo)
    assert doc["verdict"] == S.PARTIAL
    assert "MOD_E1_EXTERNAL_REVIEW" in doc["gaps"]
    assert "MOD_E1_EXTERNAL_REVIEW" not in doc["dispatch_blocking_gaps"]
    assert set(doc["dispatch_blocking_gaps"]) < set(doc["gaps"])
    # and a documentary pass is never an authorization
    assert "Neither list is an" in doc["two_lists_reading"]


def test_every_condition_other_than_the_ruling_still_blocks_dispatch(S, repo):
    doc = S.seal(repo)
    assert doc["dispatch_blocking_gaps"] == [
        g for g in doc["gaps"] if g != "MOD_E1_EXTERNAL_REVIEW"]


# --- absence: answered with OUTSIDE and a named remedy, never with a refusal ---

def test_a_ruling_that_is_not_retained_leaves_the_requirement_outside(S, repo):
    for rel in S.STANDING_IN_DISPOSITIONS:
        (repo / rel).unlink()
    doc = json.loads((repo / S.PROGRAMME_STATE).read_text())
    doc.pop(S.DISPOSITION_BLOCK)
    _put_state(repo, S, doc)
    cond = _cond(S, repo)
    assert cond["state"] == S.OUTSIDE
    assert cond["ruling"]["ruling"] == "NO_RULING_RETAINED"
    assert "no standing-in ruling is retained either" in cond["evidence"]
    assert "Only Musashi" in cond["what_would_discharge_it"]
    assert cond["blocks_module_dispatch"] is True


def test_a_retained_ruling_the_programme_never_declared_is_not_promoted(S, repo):
    doc = json.loads((repo / S.PROGRAMME_STATE).read_text())
    doc.pop(S.DISPOSITION_BLOCK)
    _put_state(repo, S, doc)
    cond = _cond(S, repo)
    assert cond["state"] == S.OUTSIDE
    assert "does not declare" in cond["evidence"]


def test_an_absent_programme_state_is_absence_not_contradiction(S, repo):
    (repo / S.PROGRAMME_STATE).unlink()
    cond = _cond(S, repo)
    assert cond["state"] == S.OUTSIDE


# --- contradiction: a seal that would mislead is not emitted -------------------

def test_a_declared_discharge_whose_documents_are_gone_is_a_refusal(S, repo):
    (repo / S.STANDING_IN_DISPOSITIONS[0]).unlink()
    with pytest.raises(S.SealRefusal, match="are not retained"):
        S.seal(repo)


def test_a_declared_discharge_naming_other_documents_is_a_refusal(S, repo):
    doc = _state(repo, S)
    doc[S.DISPOSITION_BLOCK]["documents"] = ["../../audits/work_plan/OTHER.md"]
    _put_state(repo, S, doc)
    with pytest.raises(S.SealRefusal, match="own evidence disagree"):
        S.seal(repo)


def test_a_ruling_that_stands_in_for_a_different_review_is_a_refusal(S, repo):
    doc = _state(repo, S)
    doc[S.DISPOSITION_BLOCK]["stands_in_for"] = ["MUSASHI_RP1_RP8_REVIEW"]
    _put_state(repo, S, doc)
    with pytest.raises(S.SealRefusal, match="does not reach this requirement"):
        S.seal(repo)


def test_counts_the_documents_do_not_print_are_a_refusal(S, repo):
    doc = _state(repo, S)
    doc[S.DISPOSITION_BLOCK]["checks"]["REFUTED"] = 1234
    _put_state(repo, S, doc)
    with pytest.raises(S.SealRefusal, match="no retained disposition prints"):
        S.seal(repo)


def test_a_ruling_that_hides_its_authority_is_a_refusal(S, repo):
    p = repo / S.STANDING_IN_DISPOSITIONS[0]
    p.write_text(p.read_text().replace("owner's grant of 2026-09-26", "somehow"))
    with pytest.raises(S.SealRefusal, match="the authority it acts under"):
        S.seal(repo)


def test_a_ruling_written_in_the_reviewers_name_is_a_refusal(S, repo):
    p = repo / S.STANDING_IN_DISPOSITIONS[1]
    p.write_text(p.read_text().replace(
        "signed, quoted or attributed to Musashi", "written by the auditor"))
    with pytest.raises(S.SealRefusal,
                       match="not written in the reviewer's name"):
        S.seal(repo)


def test_a_ruling_signed_by_the_reviewer_is_a_refusal(S, repo):
    p = repo / S.STANDING_IN_DISPOSITIONS[0]
    p.write_text(p.read_text() + "\n— Musashi\n")
    with pytest.raises(S.SealRefusal, match="in the reviewer's name"):
        S.seal(repo)


def test_a_ruling_that_never_names_the_requirement_is_a_refusal(S, repo):
    for rel in S.STANDING_IN_DISPOSITIONS:
        p = repo / rel
        p.write_text(p.read_text().replace("MOD_E1_EXTERNAL_REVIEW", "something"))
    with pytest.raises(S.SealRefusal, match="the requirement ruled on"):
        S.seal(repo)


def test_the_markdown_carries_the_ruling_and_its_scope(S):
    md = S.markdown(S.seal(REPO))
    assert "## The ruling on the reserved external review" in md
    assert "Gaps that still block dispatch" in md
    assert "identity recomputed" in md


# --------------------------------------------------- the verdict rule itself --

@pytest.mark.parametrize("states,expected", [
    ((["DISCHARGED", "PROHIBITION_OBSERVED"]), "E1_SEALED"),
    ((["DISCHARGED", "UNMET"]), "E1_PARTIAL_SEAL"),
    ((["DISCHARGED", "NOT_DISCHARGEABLE_BY_ARTIFACTS"]), "E1_PARTIAL_SEAL"),
    ((["PROHIBITION_OBSERVED"]), "E1_SEALED"),
])
def test_the_verdict_follows_from_the_condition_states_alone(S, states,
                                                            expected):
    conds = [{"id": f"c{i}", "state": s} for i, s in enumerate(states)]
    v, gaps = S.verdict(conds)
    assert v == expected
    assert bool(gaps) == (expected == "E1_PARTIAL_SEAL")


def test_the_markdown_renders_the_gaps_and_the_disclosures(S):
    md = S.markdown(S.seal(REPO))
    assert "E1_PARTIAL_SEAL" in md
    assert "NOT_DISCHARGEABLE_BY_ARTIFACTS" in md
    assert "Q2_CONTEXT" in md and "not 'no effect'" in md
    assert "naive MAE kW" in md and "NOT_COMPARABLE" in md
    assert "has no retained artifact here" in md
