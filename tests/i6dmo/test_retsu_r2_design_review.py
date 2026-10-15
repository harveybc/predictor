"""Retsu review of the frozen R0-vs-R2 design contract at 7488e899 (design layer only).

CPU only; models are BUILT (seed init) but never fitted; no pretraining runs; no data or TEST read.
``xfail(strict=True)`` tests state a REQUIRED behaviour the design does not have yet: each is a
minimal reproduction of a review finding and flips to a failure when it is fixed.

MODEL lane follow-up (D1-D4 fixed after the lead's ruling on review 93bff41d): the four strict
xfails now pass as ordinary tests; observation tests that documented a defect now assert the
fixed behaviour. Each edited line says so.
"""
import copy
import hashlib
import inspect
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.i6dmo import test_r2_design as L
from tools.i6dmo import r2_design as R2

REPO = Path(__file__).resolve().parents[2]
tf = pytest.importorskip("tensorflow")
from tools.i6dmo import model as M, train as T  # noqa: E402
from tools import modular_candidate_evaluator as E  # noqa: E402
from predictor_plugins.modular_temporal import objectives as O  # noqa: E402


# ======================================================================== item 1: objective O1

def test_o1_registered_component_has_no_end_to_end_caller_and_i6b_uses_its_own_loop():
    src = (REPO / "tools/i6b_branch_pretraining.py").read_text()
    assert "fit_objective" not in src and "AutoencoderReconstruction" not in src
    assert '"name": "branch_autoencoder_reconstruction"' in src           # a different identity
    assert '"decoder_channels": 16' in src and "keras.optimizers.Adam(" in src
    assert "MeanTrainTailEarlyStopping" in src and "shuffle=False" in src
    assert O.describe_objective("autoencoder_reconstruction")["defaults"] == {"decoder_channels": 32,
                                                                             "loss": "mse"}


def test_recommended_o1_ruling_is_not_executable_by_the_registered_fit_loop():
    """The 7488e899 recommendation (max_epochs 200) is not executable by the registered loop.
    MODEL fix D1: r2_design now refuses it at validation instead of returning READY."""
    ruled = R2.rule_objective(L._contract(), L._ruled_o1_v0())
    with pytest.raises(R2.R2DesignError):
        R2.validate_r2_design(ruled, for_execution=True)
    fs = L._ruled_o1_v0()["fit_settings"]
    with pytest.raises(ValueError, match=r"max_epochs must be an integer in \[1, 100\]"):
        E._settings({"evaluator": {"max_epochs": fs["max_epochs"], "patience": fs["patience"],
                                   "batch_size": fs["batch_size"], "learning_rate": fs["learning_rate"],
                                   "weight_decay": fs["weight_decay"], "min_delta": fs["min_delta"],
                                   "seed": 6102026, "loss": "mse"}})


def test_a_non_executable_ruling_must_be_refused_at_validation():      # was strict xfail; fixed (D1)
    objective = L._ruled_o1()
    objective["fit_settings"]["max_epochs"] = 200
    ruled = R2.rule_objective(L._contract(), objective)
    with pytest.raises(R2.R2DesignError):
        R2.validate_r2_design(ruled, for_execution=True)


def test_o1_fit_loop_ignores_the_seed_and_has_unruled_budget_stops():
    """The 'seed' setting reaches fit_with_early_stopping but no RNG is seeded there; max_updates and
    a WALL-CLOCK max_seconds stop default silently (the ruling does not fix them)."""
    body = inspect.getsource(E.fit_with_early_stopping)
    assert "set_random_seed" not in body and 'settings["seed"]' not in body
    assert O._SETTINGS["max_seconds"] == 3600.0 and O._SETTINGS["max_updates"] == 100000
    assert {"max_updates", "max_seconds"} <= set(R2.OBJECTIVE_FIT_SETTINGS)   # fixed (D1): ruled now
    assert "time.monotonic() - started >= settings[\"max_seconds\"]" in body


def test_o1_can_only_monitor_its_validation_argument():
    """mean(FIT, INNER) is not expressible through the registered objective (no 'monitor' key)."""
    with pytest.raises(ValueError, match="unknown objective fit settings"):
        O._settings({"monitor": "train_validation_mean"})
    assert E._settings({"evaluator": {}})["monitor"] == "validation_loss"
    # and the evaluator's train_validation_mean averages the RUNNING in-epoch train loss, not an
    # end-of-epoch FIT evaluation at the same weights (R0's monitor)
    assert "0.5 * (train_loss / len(x) + val_loss)" in inspect.getsource(E.fit_with_early_stopping)


# ======================================================================== item 3: forged identity

def _bundle_from(design):
    return M.build_dmo(T.effective_from_design(design)["model"])


def _forged(target, foreign_weights_by_feature):
    """Manifests copied from the TARGET identity with FOREIGN weights' digests (a forged block)."""
    ctx = L._context(target)
    out = {}
    for f in target.config["features"]:
        manifest = M.branch_donor_manifest(target, f, ctx)
        weights = [np.array(w) for w in foreign_weights_by_feature[f]]
        manifest["weights"] = M._array_digests(weights)
        out[f] = {"manifest": manifest, "weights": weights}
    return out, ctx


SOURCES = {
    "mg1_matched": L.MATCHED / "EXPERIMENT.mg1.json",
    "closed_full_population_mg2": L.MATCHED.parent / "EXPERIMENT.mg2.json",
}


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_observation_forged_identity_with_foreign_real_shaped_weights_is_imported(source):
    """Was ACCEPTED at 7488e899; MODEL fix D3: a self-declared manifest is now refused atomically."""
    foreign = L._pretend_pretrained(_bundle_from(SOURCES[source]))
    target = L._bundle()
    before = [np.array(w) for w in target.forecast_model.get_weights()]
    weights = {f: foreign.branch_models[f].get_weights() for f in target.config["features"]}
    donors, ctx = _forged(target, weights)
    with pytest.raises(M.DonorImportError, match="R2_DONOR_PROVENANCE_MISSING"):
        M.import_branch_donors(target, donors, ctx)
    assert all(np.array_equal(a, b) for a, b in zip(before, target.forecast_model.get_weights()))


def test_a_forged_identity_block_must_be_refused_before_any_weight_is_set():   # was strict xfail (D3)
    foreign = L._pretend_pretrained(_bundle_from(SOURCES["closed_full_population_mg2"]))
    target = L._bundle()
    donors, ctx = _forged(target, {f: foreign.branch_models[f].get_weights() for f in target.config["features"]})
    with pytest.raises(M.DonorImportError):
        M.import_branch_donors(target, donors, ctx)


def test_elapsed_and_wider_donors_are_refused_honestly_and_atomically():
    """Honest manifests from other recipes are refused before any weight is set (lane claim holds)."""
    target = L._bundle()
    before = [np.array(w) for w in target.forecast_model.get_weights()]
    elapsed = _bundle_from(L.MATCHED.parent / "EXPERIMENT.short.json")
    with pytest.raises(M.DonorImportError):
        M.branch_donor_context(elapsed, population_sha256=L.POP, scaler_sha256=L._scaler_sha(),
                               objective=L._ruled_o1())
    mg1 = L._pretend_pretrained(_bundle_from(SOURCES["mg1_matched"]))
    honest = {f: {"manifest": M.branch_donor_manifest(mg1, f, L._context(mg1)),
                  "weights": [np.array(w) for w in mg1.branch_models[f].get_weights()]}
              for f in target.config["features"]}
    with pytest.raises(M.DonorImportError, match="R2_DONOR_ARM_MISMATCH"):
        M.import_branch_donors(target, honest, L._context(target))
    wide = copy.deepcopy(honest)
    f0 = target.config["features"][0]
    wide[f0]["weights"] = [np.zeros((3, 1, 16), np.float32), np.zeros(16, np.float32)]
    with pytest.raises(M.DonorImportError):
        M.import_branch_donors(target, wide, L._context(target))
    assert all(np.array_equal(a, b) for a, b in zip(before, target.forecast_model.get_weights()))


# ======================================================================== item 4: R0 reuse

def test_r0_reuse_is_fresh_because_trainer_code_files_changed_since_the_deployed_pin():
    """The closed fit ran at 796ec8f4; design.py and model.py changed since (CODE_FILES of train)."""
    def at(rev, rel):
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{rel}"], capture_output=True, check=True)
        return hashlib.sha256(out.stdout).hexdigest()
    deployed = {rel: at("796ec8f4", rel) for rel in T.CODE_FILES}
    current = T._code_identity()
    changed = sorted(rel for rel in T.CODE_FILES if deployed[rel] != current[rel])
    # MODEL follow-up also touched train.py (GPU class recorded for R2-18): still FRESH_R0_REQUIRED.
    assert changed == ["tools/i6dmo/design.py", "tools/i6dmo/model.py", "tools/i6dmo/train.py"]
    receipt = L._agreeing_receipt(L._contract()["shared"])
    receipt["identity"]["code_files_sha256"] = deployed
    out = R2.r0_reuse_decision(receipt, dict(L.MONITOR), L._contract(), expected_code_files=current,
                               expected_scales_sha256=L.SCALES, current_monitor_identity=L.MONITOR,
                               r1_verdict="ACCEPT")
    assert out["decision"] == "FRESH_R0_REQUIRED" and out["failed"] == ["code_files_sha256"]


# ======================================================================== decision (item 6 gaps)

def test_decide_r2_must_refuse_non_scientific_fits():            # was strict xfail; fixed (D4)
    r0, r2 = L._metrics(-0.01), L._metrics(-0.02)
    r2["matched"].update(scientific_fit=False, fit_device="cpu")
    with pytest.raises(R2.R2DesignError):
        R2.decide_r2(r0, r2)


def test_decide_r2_must_require_the_import_record():             # was strict xfail; fixed (D4)
    with pytest.raises(R2.R2DesignError):
        R2.decide_r2(L._metrics(-0.01), L._metrics(-0.02))
