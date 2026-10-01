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
    claimed["data_manifest"]["asset_appearance"] = dict(CONTRACTED_EURUSD_SLOT)
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
    claimed["data_manifest"]["asset_appearance"] = dict(CONTRACTED_EURUSD_SLOT)
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
    claimed["data_manifest"]["asset_appearance"] = dict(CONTRACTED_EURUSD_SLOT)
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
    mislabelled["data_manifest"]["asset_appearance"] = dict(CONTRACTED_EURUSD_SLOT)
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


CONTRACTED_EURUSD_SLOT = {
    "state": "CONTRACTED",
    "appearance_id": "app_0123456789abcdef01234567",
    "dataset_id": "financial_data.census_appearance.app_0123456789abcdef01234567",
    "entity": "eurusd",
    "resource_sha256": "a" * 64,
    "contract_id": "c127:eurusd:5m:example",
    "contract_sha256": "b" * 64,
    "train_rows": [0, 931216],
    "frequency": "5m",
    "period": ["2005-01-03T00:00:00+00:00", "2025-12-31T16:55:00+00:00"],
    "contracts_document_sha256": "bfaf2cf63814a0be5756f648f3f5c949b97d96b1f0d0c29f674bc0d36a371406",
}


def test_dossier_not_executable_slot_keeps_every_rung_unevaluated(dossier_validator, dossier):
    """Open question 13: without a contracted price appearance, rung 1 is NOT_EVALUATED and no evidence level exists."""
    assert dossier["data_manifest"]["asset_appearance"]["state"] == "NOT_EXECUTABLE_NO_CONTRACTED_PRICE"
    claimed = copy.deepcopy(dossier)
    claimed["rung1"]["state"] = "ASSOCIATION_REPORTED"
    assert _errors(dossier_validator, claimed) != [], "an uncontracted price series cannot carry rung-1 evidence"
    claimed = copy.deepcopy(dossier)
    claimed["selection"]["causal_evidence_level"] = "ASSOCIATION"
    assert _errors(dossier_validator, claimed) != []


def test_dossier_contracted_slot_binds_without_redesign(dossier_validator, dossier):
    """Supplying the sealed appearance (id, resource sha, TRAIN rows, contract id) is the only change needed to run rung 1."""
    bound = copy.deepcopy(dossier)
    bound["data_manifest"]["asset_appearance"] = dict(CONTRACTED_EURUSD_SLOT)
    bound["rung1"]["state"] = "ASSOCIATION_REPORTED"
    bound["rung1"]["effective_n"] = 80
    bound["rung1"]["evidence"] = [{"measure": "pearson", "value": 0.12, "n": 80, "p": 0.29, "q": None, "regime": None,
                                   "signed_direction_stable": None}]
    bound["rung2"]["reasons"] = ["ASSUMED_PUBLICATION_CLOCK", "PLACEBO_FAILED"]
    bound["selection"]["causal_evidence_level"] = "ASSOCIATION"
    assert _errors(dossier_validator, bound) == []


@pytest.mark.parametrize("field, value", [
    ("appearance_id", "eurusd-5m"),
    ("resource_sha256", "d527b46a"),
    ("contract_sha256", "not-a-digest"),
    ("train_rows", [0]),
    ("frequency", "1m"),
    ("contract_id", ""),
])
def test_dossier_contracted_slot_refuses_a_loose_pin(dossier_validator, dossier, field, value):
    bound = copy.deepcopy(dossier)
    slot = dict(CONTRACTED_EURUSD_SLOT)
    slot[field] = value
    bound["data_manifest"]["asset_appearance"] = slot
    assert _errors(dossier_validator, bound) != []


def test_dossier_contracted_slot_requires_every_binding_field(dossier_validator, dossier):
    for missing in ("appearance_id", "resource_sha256", "train_rows", "contract_id", "contracts_document_sha256"):
        bound = copy.deepcopy(dossier)
        slot = dict(CONTRACTED_EURUSD_SLOT)
        del slot[missing]
        bound["data_manifest"]["asset_appearance"] = slot
        assert _errors(dossier_validator, bound) != [], missing


def test_dossier_slot_state_is_one_of_two(dossier_validator, dossier):
    other = copy.deepcopy(dossier)
    other["data_manifest"]["asset_appearance"] = {"state": "SEALED", "entity": "eurusd", "reason": "x"}
    assert _errors(dossier_validator, other) != []


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


PROXY_WAVELET_METHOD = {
    "method_id": "PROXY",
    "method_name": "MULTISCALE_ROLLING_MEAN_PROXY",
    "proxy_of": "DWT_DB4",
    "native_reference": None,
    "inputs_transformed": ["close"],
    "units": {"wavelet_detail_L1": "price"},
    "sample_interval_seconds": 300,
    "window_bars": 128,
    "window_frozen_from": None,
    "update_cadence_bars": 1,
    "max_feature_age_bars": 0,
    "timing_invariance": {"status": "NOT_EVALUATED", "tests": []},
    "native_coverage_claimed": False,
}


def test_card_requires_the_method_block(card_validator, card):
    bare = copy.deepcopy(card)
    del bare["method"]
    assert _errors(card_validator, bare) != []


def test_card_proxy_method_cannot_claim_native_coverage(card_validator, card):
    proxy = copy.deepcopy(card)
    proxy["method"] = dict(PROXY_WAVELET_METHOD)
    assert _errors(card_validator, proxy) == []
    proxy["method"] = dict(PROXY_WAVELET_METHOD, native_coverage_claimed=True)
    assert _errors(card_validator, proxy) != []
    proxy["method"] = dict(PROXY_WAVELET_METHOD, proxy_of=None)
    assert _errors(card_validator, proxy) != [], "a PROXY must name what it approximates"


def test_card_native_method_names_its_library_and_has_no_proxy_of(card_validator, card):
    native = copy.deepcopy(card)
    native["method"] = dict(PROXY_WAVELET_METHOD, method_id="NATIVE", method_name="DWT_DB4_TRAILING_WINDOW",
                            proxy_of=None, native_reference={"library": "pywt", "version": "1.7.0", "function": "wavedec"})
    assert _errors(card_validator, native) == []
    native["method"]["native_reference"] = None
    assert _errors(card_validator, native) != []
    native["method"]["native_reference"] = {"library": "pywt", "version": "1.7.0"}
    native["method"]["proxy_of"] = "DWT_DB4"
    assert _errors(card_validator, native) != []


def test_card_learned_method_needs_a_fit_period_bound_to_a_fold(card_validator, card):
    learned = copy.deepcopy(card)
    learned["method"] = dict(PROXY_WAVELET_METHOD, method_id="LEARNED", method_name="LEARNED_REGIME_FIXED_CONSTANTS",
                             proxy_of=None)
    assert _errors(card_validator, learned) != [], "LEARNED without fit_period must be refused"
    learned["method"]["fit_period"] = {"fold_id": "fold-1", "train_end": "2020-12-31T00:00:00Z"}
    assert _errors(card_validator, learned) == []


def test_card_violated_timing_cannot_be_admissible(card_validator, card):
    violated = copy.deepcopy(card)
    violated["method"] = dict(PROXY_WAVELET_METHOD, method_id="NATIVE", method_name="SCIPY_HILBERT_TRAILING_WINDOW",
                              proxy_of=None, native_reference={"library": "scipy", "version": "1.13.1"},
                              timing_invariance={"status": "VIOLATED",
                                                 "tests": ["test_required_prefix_invariance_below_1000_rows"],
                                                 "violations": ["WINDOW_DERIVED_FROM_INPUT_LENGTH"]})
    violated["admissibility"] = {"verdict": "ADMISSIBLE", "reasons": []}
    assert _errors(card_validator, violated) != []
    violated["admissibility"] = {"verdict": "NOT_ADMISSIBLE", "reasons": ["timing invariance violated"]}
    assert _errors(card_validator, violated) == []
    # a VIOLATED or VERIFIED status without the deciding tests is an assertion, not a measurement
    violated["method"]["timing_invariance"] = {"status": "VERIFIED", "tests": []}
    assert _errors(card_validator, violated) != []
    violated["method"]["timing_invariance"] = {"status": "VIOLATED", "tests": ["t"]}
    assert _errors(card_validator, violated) != [], "VIOLATED must name at least one violation"


def test_card_generative_family_is_secondary(card_validator, card):
    generative = copy.deepcopy(card)
    generative["family"] = "generative_cvae"
    generative["evaluation"]["reconstruction"] = {"state": "NOT_EVALUATED"}
    generative["admissibility"] = {"verdict": "ADMISSIBLE", "reasons": []}
    assert _errors(card_validator, generative) != []
    generative["admissibility"] = {"verdict": "SECONDARY", "reasons": ["generation is evaluated separately"]}
    assert _errors(card_validator, generative) == []
