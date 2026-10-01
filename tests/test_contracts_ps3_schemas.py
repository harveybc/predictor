"""The two PS3 consumer contracts under `docs/contracts/`: the causal dossier and the representation candidate card.

Calendar and M5PHET consume these documents, never the producing implementations. These tests pin what the schemas
refuse, so that a document cannot say more than its evidence: an identified rung 2 with a failed placebo, a rung-3
counterfactual labelled model-based, a rung 3 that outruns its rung 2, a pooled latent that claims the temporal
contract, foreign pretrained weights labelled TRAIN_ONLY, and a transcribed comparator without its table.

Nothing here measures anything; the example documents are labelled as examples in their own limitations.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

CONTRACTS = Path(__file__).resolve().parents[1] / "docs" / "contracts"


def _load(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dossier_validator():
    schema = _load("causal_dossier.v1.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture(scope="module")
def card_validator():
    schema = _load("representation_candidate_card.v1.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.fixture
def dossier():
    return _load("examples/causal_dossier.v1.example.json")


@pytest.fixture
def card():
    return _load("examples/representation_candidate_card.v1.example.json")


def _errors(validator, document):
    return [error.message for error in validator.iter_errors(document)]


# ------------------------------------------------------------------------------------------------ causal dossier


def test_dossier_example_is_valid_and_not_identified(dossier_validator, dossier):
    assert _errors(dossier_validator, dossier) == []
    assert dossier["rung2"]["state"] == "NOT_IDENTIFIED"
    assert dossier["rung3"]["state"] == "NOT_IDENTIFIED"
    assert dossier["rung2"]["estimate"] is None
    assert dossier["selection"]["cf_eligible"] is False


def test_dossier_identified_rung2_needs_passed_placebo_and_support(dossier_validator, dossier):
    claimed = copy.deepcopy(dossier)
    claimed["data_manifest"]["publication_clock"] = "OBSERVED_PUBLICATION_CLOCK"
    claimed["rung2"]["state"] = "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS"
    claimed["rung2"]["reasons"] = []
    claimed["rung2"]["assumptions_declared"] = {"sufficient_adjustment": True, "positivity": True}
    claimed["rung2"]["estimate"] = {"value": 0.0012, "interval": [0.0001, 0.0023], "unit": "log return per unit surprise",
                                    "uncertainty": "HAC_newey_west_95"}
    # placebo FAILED and support NOT_EVALUATED are still in the document: the claim must be refused
    assert _errors(dossier_validator, claimed) != []
    claimed["rung2"]["placebo"] = {"state": "PASSED"}
    claimed["rung2"]["support"] = {"state": "SUPPORTED", "n_per_side": [40, 40]}
    assert _errors(dossier_validator, claimed) == []


def test_dossier_assumed_clock_cannot_carry_an_identified_rung(dossier_validator, dossier):
    claimed = copy.deepcopy(dossier)
    claimed["rung2"]["state"] = "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS"
    claimed["rung2"]["reasons"] = []
    claimed["rung2"]["assumptions_declared"] = {"sufficient_adjustment": True}
    claimed["rung2"]["placebo"] = {"state": "PASSED"}
    claimed["rung2"]["support"] = {"state": "SUPPORTED"}
    claimed["rung2"]["estimate"] = {"value": 0.0, "interval": [-0.1, 0.1], "unit": "u", "uncertainty": "x"}
    # publication_clock stays ASSUMED_SCHEDULED_PUBLICATION
    assert _errors(dossier_validator, claimed) != []


def test_dossier_model_based_expectation_cannot_carry_an_identified_rung(dossier_validator, dossier):
    claimed = copy.deepcopy(dossier)
    claimed["data_manifest"]["publication_clock"] = "OBSERVED_ACTUAL_PUBLICATION"
    claimed["data_manifest"]["expectation_kind"] = "MODEL_BASED_EXPECTATION"
    claimed["rung2"].update({"state": "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS", "reasons": [],
                             "assumptions_declared": {"sufficient_adjustment": True},
                             "placebo": {"state": "PASSED"}, "support": {"state": "SUPPORTED"},
                             "estimate": {"value": 0.0, "interval": [-0.1, 0.1], "unit": "u", "uncertainty": "x"}})
    assert _errors(dossier_validator, claimed) != []


def test_dossier_not_identified_rung2_releases_no_estimate(dossier_validator, dossier):
    leaked = copy.deepcopy(dossier)
    leaked["rung2"]["estimate"] = {"value": 0.0012, "interval": [0.0001, 0.0023], "unit": "u", "uncertainty": "x"}
    assert _errors(dossier_validator, leaked) != []


def test_dossier_rung3_cannot_outrun_rung2(dossier_validator, dossier):
    claimed = copy.deepcopy(dossier)
    claimed["rung3"] = {
        "state": "COUNTERFACTUAL_UNDER_DECLARED_SCM",
        "label": "SAME_EPISODE_COUNTERFACTUAL_UNDER_DECLARED_SCM",
        "scm": {"order": ["W_pre", "A", "M", "Y_h"], "equations": {"Y_h": "f(A, W_pre, M) + U_Y"}, "noise": "ADDITIVE",
                "invertible": True, "fit_digest": "0" * 64},
        "abduction": {"U_Y": 0.001}, "action": {"A": 0.0},
        "prediction": {"factual": 0.002, "counterfactual": 0.0015, "delta": 0.0005, "model_based": 0.0005},
    }
    # rung 2 is still NOT_IDENTIFIED
    assert _errors(dossier_validator, claimed) != []


def test_dossier_model_based_counterfactual_is_never_the_scm_state(dossier_validator, dossier):
    mislabelled = copy.deepcopy(dossier)
    mislabelled["data_manifest"]["publication_clock"] = "OBSERVED_PUBLICATION_CLOCK"
    mislabelled["rung2"].update({"state": "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS", "reasons": [],
                                 "assumptions_declared": {"sufficient_adjustment": True},
                                 "placebo": {"state": "PASSED"}, "support": {"state": "SUPPORTED"},
                                 "estimate": {"value": 0.0, "interval": [-0.1, 0.1], "unit": "u", "uncertainty": "x"}})
    mislabelled["rung3"] = {"state": "COUNTERFACTUAL_UNDER_DECLARED_SCM", "label": "MODEL_BASED_COUNTERFACTUAL",
                            "scm": {"order": ["A", "Y_h"], "equations": {}, "noise": "ADDITIVE", "invertible": True,
                                    "fit_digest": "0" * 64},
                            "abduction": {}, "action": {"A": 0.0}, "prediction": {"factual": 0.0, "counterfactual": 0.0}}
    assert _errors(dossier_validator, mislabelled) != []


def test_dossier_cf_eligible_needs_counterfactual_evidence_level(dossier_validator, dossier):
    claimed = copy.deepcopy(dossier)
    claimed["selection"]["cf_eligible"] = True
    assert _errors(dossier_validator, claimed) != []


def test_dossier_refuses_model_intervention_evidence(dossier_validator, dossier):
    """Subplan 5.5: intervening on the network is model evidence; it has no field in the causal dossier."""
    mixed = copy.deepcopy(dossier)
    mixed["model_intervention_evidence"] = [{"kind": "permute_latent", "scope": "branch_3", "delta_loss": 0.01,
                                             "paired_budget": True}]
    assert _errors(dossier_validator, mixed) != []


# ------------------------------------------------------------------------------------ representation candidate card


def test_card_example_is_valid(card_validator, card):
    assert _errors(card_validator, card) == []
    assert card["evaluation"]["reconstruction"]["state"] == "NOT_APPLICABLE"


def test_card_pooled_latent_cannot_satisfy_temporal_contract(card_validator, card):
    pooled = copy.deepcopy(card)
    pooled["latent"] = {"layout": "pooled", "time_steps": 1, "channels": 32, "grid_adapter": None}
    pooled["temporal_contract_state"] = "SATISFIED"
    assert _errors(card_validator, pooled) != []
    pooled["temporal_contract_state"] = "VIOLATED_POOLED"
    assert _errors(card_validator, pooled) == []


def test_card_patch_latent_needs_a_declared_adapter(card_validator, card):
    patched = copy.deepcopy(card)
    patched["latent"] = {"layout": "patch", "time_steps": 24, "channels": 16, "grid_adapter": None}
    patched["temporal_contract_state"] = "ADAPTER_DECLARED"
    assert _errors(card_validator, patched) != []
    patched["latent"]["grid_adapter"] = {"name": "overlapping_patch_to_step_projection", "validated_by": "FS04"}
    assert _errors(card_validator, patched) == []


def test_card_foreign_weights_are_never_train_only(card_validator, card):
    foreign = copy.deepcopy(card)
    foreign["family"] = "foundation_pretrained"
    foreign["corpus"] = {"kind": "TRAIN_ONLY", "pretrained_weights_source": "AutonLab/MOMENT-1-small",
                         "pretrained_weights_revision": None, "train_folds": ["fold"], "contamination_audit": "NOT_REQUIRED"}
    assert _errors(card_validator, foreign) != []
    foreign["corpus"] = {"kind": "FOREIGN_PRETRAINED", "pretrained_weights_source": "AutonLab/MOMENT-1-small",
                         "pretrained_weights_revision": None, "train_folds": [], "contamination_audit": "REQUIRED_NOT_DONE"}
    foreign["admissibility"] = {"verdict": "ADMISSIBLE", "reasons": []}
    assert _errors(card_validator, foreign) != [], "a foreign-pretrained corpus cannot be plainly ADMISSIBLE"
    foreign["admissibility"] = {"verdict": "ADMISSIBLE_AS_FOREIGN_COMPARATOR", "reasons": ["labelled foreign"]}
    assert _errors(card_validator, foreign) == []


def test_card_reconstruction_not_applicable_is_not_a_failure(card_validator, card):
    refused = copy.deepcopy(card)
    refused["admissibility"] = {"verdict": "NOT_ADMISSIBLE", "reasons": ["RECONSTRUCTION_FAILED"]}
    assert _errors(card_validator, refused) != [], "NOT_APPLICABLE reconstruction can never be cited as a failure"


def test_card_transcribed_comparator_needs_its_table_and_artifact(card_validator, card):
    bare = copy.deepcopy(card)
    bare["comparator"] = {"provenance": "TRANSCRIBED", "values": [{"metric": "MSE", "value": 0.126}]}
    assert _errors(card_validator, bare) != []
    bare["comparator"].update({"dataset": "ECL", "table": "Table 4", "artifact": "SOTA_REFERENCE_DOSSIER_2026_09_21.md",
                               "comparable_to_ours": "COMPARABLE"})
    assert _errors(card_validator, bare) == []


def test_card_code_revision_must_be_a_pin_not_a_branch(card_validator, card):
    loose = copy.deepcopy(card)
    loose["reference"]["code_revision"] = "main"
    assert _errors(card_validator, loose) != []


def test_card_generative_family_is_secondary(card_validator, card):
    generative = copy.deepcopy(card)
    generative["family"] = "generative_cvae"
    generative["evaluation"]["reconstruction"] = {"state": "NOT_EVALUATED"}
    generative["admissibility"] = {"verdict": "ADMISSIBLE", "reasons": []}
    assert _errors(card_validator, generative) != []
    generative["admissibility"] = {"verdict": "SECONDARY", "reasons": ["generation is evaluated separately"]}
    assert _errors(card_validator, generative) == []
