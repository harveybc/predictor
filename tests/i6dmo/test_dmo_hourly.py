"""Section-3 preparation: sealed paired short HOURLY-input control (no fit, CPU only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tools.i6dmo import design as D
from tests.i6dmo.dmo_model_fixtures import DESIGN, load_design_doc

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
HOURLY_ID = "I6-DMO-SHORT-HOURLY-DS1-STATIC-R0"


@pytest.fixture()
def sealed(tmp_path):
    result = D.seal_hourly_control(DESIGN, tmp_path)
    return tmp_path / "EXPERIMENT.short_hourly.json", tmp_path / "COMMON_SUPPORT_RULES.json", result


def test_hourly_control_mirrors_the_short_design_except_the_declared_grid(sealed):
    design_path, rules_path, result = sealed
    doc = D.load_design(design_path)
    assert doc["experiment_id"] == HOURLY_ID
    assert doc["input"]["spacing_hours"] == 1 and doc["input"]["observations"] == 25
    assert doc["input"]["history_span_hours"] == 24 and doc["input"]["include_origin"] is True
    assert doc["horizons_hours"] == [2, 3, 4, 5, 6, 12]
    short = load_design_doc()
    body = {k: v for k, v in doc.items() if k not in ("derived_from", "seal_sha256")}
    assert {k for k in body if body[k] != short.get(k)} == {"experiment_id", "input"}
    assert {k for k in doc["input"] if doc["input"][k] != short["input"][k]} == \
        {"spacing_hours", "observations"}
    eff = D.effective_config(doc)
    assert eff["family"] == "short_hourly" and eff["mode"] == "VALIDATION"
    assert eff["offsets_hours"] == list(range(-24, 1, 1)) and len(eff["offsets_hours"]) == 25
    assert eff["splits"] == D.effective_config(load_design_doc())["splits"]  # same static cadence
    assert eff["design_sha256"] != D.design_sha256(short)  # its OWN identity
    report = D.validate_design_report(design_path)
    assert report["status"] == "PASS" and report["family"] == "short_hourly"
    assert result["design_sha256"] == D.design_sha256(doc)


def test_availability_differences_are_declared_and_counts_are_not_duplicated(sealed):
    design_path, _, _ = sealed
    doc = json.loads(design_path.read_text())
    declaration = doc["derived_from"]["availability_declaration"]
    assert "weekend" in declaration and "differ" in declaration
    assert "market-grid" in declaration and "data lane" in declaration  # coordination, not counts
    assert "not duplicated" in declaration.lower() or "owns" in declaration.lower()


def test_common_support_rules_are_frozen_before_any_scoring_and_bound_to_the_design(sealed, tmp_path):
    design_path, rules_path, _ = sealed
    rules = json.loads(rules_path.read_text())
    body = {k: v for k, v in rules.items() if k != "rules_sha256"}
    assert rules["rules_sha256"] == D._sha(body)
    assert rules["frozen_before_scoring"] is True
    assert rules["population"].startswith("intersection")
    assert "REUSED_MODEL_SUBSET_SCORE" in json.dumps(rules)
    assert "NOT a model trained on the common TRAIN population" in json.dumps(rules)
    assert "downsampling" in rules["distinct_contrasts"]
    doc = json.loads(design_path.read_text())
    assert doc["derived_from"]["common_support_rules_sha256"] == rules["rules_sha256"]
    # Changing the frozen rules breaks the sealed design identity (typed).
    tampered = dict(doc)
    tampered["derived_from"] = dict(doc["derived_from"], common_support_rules_sha256="0" * 64)
    with pytest.raises(D.DesignError, match="HOURLY_DESIGN_SEAL|FIXED_VALUE"):
        D.validate_design(tampered)


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda d: d["input"].update(spacing_hours=2), "FIXED_VALUE"),
    (lambda d: d["input"].update(observations=13), "FIXED_VALUE"),
    (lambda d: d["splits"]["test"].update(open=True), "TEST_FORBIDDEN"),
    (lambda d: d["model"]["core"].update(transformer_blocks=1), "FIXED_VALUE"),
    (lambda d: d.update(retraining_frequency="WEEKLY"), "FIXED_VALUE"),
    (lambda d: d.update(surprise=1), "UNKNOWN_FIELD"),
    (lambda d: d["training"].update(max_epochs=100), "FIXED_VALUE"),
    (lambda d: d["input"].update(history_span_hours="UNDECIDED"), "HOURLY_DESIGN_UNDECIDED"),
])
def test_hourly_typed_refusals(sealed, mutate, code):
    design_path, _, _ = sealed
    doc = json.loads(design_path.read_text())
    mutate(doc)
    with pytest.raises(D.DesignError, match=code + "|HOURLY_DESIGN_SEAL"):
        D.validate_design(doc)


def test_seal_is_deterministic_idempotent_and_refuses_foreign_bases(sealed, tmp_path):
    design_path, rules_path, result = sealed
    again = D.seal_hourly_control(DESIGN, design_path.parent)
    assert again["design_sha256"] == result["design_sha256"]
    other = tmp_path / "other"
    D.seal_hourly_control(DESIGN, other)
    assert (other / "EXPERIMENT.short_hourly.json").read_bytes() == design_path.read_bytes()
    with pytest.raises(D.DesignError, match="SHORT_DESIGN_REQUIRED"):
        D.seal_hourly_control(design_path, tmp_path / "x")  # hourly of an hourly
    doc = D.load_design(design_path)
    with pytest.raises(D.DesignError, match="FINAL_FROM_VALIDATION_ONLY|SHORT_DESIGN_REQUIRED"):
        D.final_refit_design(doc)  # a final of the control is a later, separate decision


def test_model_recipe_adjustment_is_only_what_25_observations_forces():
    tf = pytest.importorskip("tensorflow")
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    cfg = M.model_config(list(MANIFEST_NO_STLFSI4), family="short_hourly")
    bundle = M.build_dmo(cfg)
    model = bundle.forecast_model
    assert tuple(model.input_shape) == (None, 25, 19)
    assert list(model.output_names) == ["h2", "h3", "h4", "h5", "h6", "h12"]
    assert model.get_layer("positional_encoding").offsets_hours == tuple(range(-24, 1, 1))
    for branch in bundle.branch_models.values():  # recipe unchanged: 4 filters, kernel 3, stride 1
        conv = [l for l in branch.layers if isinstance(l, tf.keras.layers.Conv1D)][0].get_config()
        assert (conv["filters"], conv["kernel_size"], conv["strides"]) == (4, (3,), (1,))
    assert tuple(bundle.core_model.output_shape) == (None, 25, 4)
    # The ONLY forced change: each head flattens 25x4=100 (vs 13x4=52) core features.
    for head in bundle.head_models.values():
        hidden = [l for l in head.layers if isinstance(l, tf.keras.layers.Dense)][0]
        assert hidden.get_config()["units"] == 8 and hidden.input.shape[-1] == 100
    shared = sum(m.count_params() for m in bundle.branch_models.values()) + bundle.core_model.count_params()
    assert shared == 304 + 1932  # branches and core parameter counts identical to the 13-obs model


def test_trainer_refuses_a_13_slot_bundle_under_the_hourly_identity(sealed, tmp_path):
    pytest.importorskip("tensorflow")
    from tools.i6dmo import train as T
    from tests.i6dmo.dmo_model_fixtures import write_bundle
    design_path, _, _ = sealed
    eff = T.effective_from_design(design_path)
    with pytest.raises(T.TrainDisposition, match="BUNDLE_OFFSETS|BUNDLE_SCHEMA"):
        T.load_data_bundle(write_bundle(tmp_path / "b13"), eff)  # 13-slot bundle: typed refusal


def test_cli_seal_and_validate(tmp_path):
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "seal-hourly-control",
                          "--short-design", str(DESIGN), "--out-dir", str(tmp_path), "--json"],
                         capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = json.loads(run.stdout)
    assert out["status"] == "SEALED" and out["experiment_id"] == HOURLY_ID
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "validate-design", "--design",
                          str(tmp_path / "EXPERIMENT.short_hourly.json"), "--json"],
                         capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 0 and json.loads(run.stdout)["family"] == "short_hourly"
