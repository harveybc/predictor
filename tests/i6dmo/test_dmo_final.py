"""AUD-DMO-02/03 corrections: typed FINAL_REFIT protocol and guarded final TEST path.

Synthetic dates/arrays only; no real TEST data is read and no scientific fit runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tools.i6dmo import design as D
from tests.i6dmo.dmo_model_fixtures import DESIGN, HORIZONS, short_terminal, split_arrays, write_bundle

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
FINAL_TRAIN = {"start": "2021-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"}


def final_bundle(root, **kw):
    """Synthetic bundle whose splits lie inside the typed final window."""
    splits = {
        "fit": split_arrays("2021-01-04T00:00:00", 96, 1, **kw),
        "inner": split_arrays("2024-11-11T00:00:00", 32, 2, **kw),
    }
    return write_bundle(root, splits=splits, **kw)


# ------------------------------------------------------------------ AUD-DMO-02

def test_final_refit_design_is_typed_and_changes_identity():
    base = D.load_design(DESIGN)
    final = D.final_refit_design(base)
    assert final["experiment_id"] == "I6-DMO-SHORT-DS2-STATIC-R0-FINAL"
    assert final["splits"]["train"] == FINAL_TRAIN
    assert "validation" not in final["splits"]
    assert final["splits"]["test"]["open"] is False
    assert final["derived_from"]["base_design_sha256"] == D.design_sha256(base)
    D.validate_design(final)  # sealed and self-consistent
    eff = D.effective_config(final)
    assert eff["mode"] == "FINAL_REFIT" and eff["family"] == "short"
    assert eff["splits"]["fit"] == {"start": "2021-01-01T00:00:00Z", "end": "2024-11-04T00:00:00Z"}
    assert eff["splits"]["inner"] == {"start": "2024-11-04T00:00:00Z", "end": "2025-01-01T00:00:00Z"}
    assert "validation" not in eff["splits"]
    # The final identity never matches the validation identity: no terminal can cross.
    val = D.effective_config(base)
    assert eff["design_sha256"] != val["design_sha256"]
    assert eff["effective_config_sha256"] != val["effective_config_sha256"]
    assert D.final_refit_design(base) == final  # deterministic


def test_final_refit_design_for_the_long_family(tmp_path):
    D.seal_long_design(DESIGN, short_terminal(tmp_path / "t.json"), tmp_path / "L.json")
    final = D.final_refit_design(D.load_design(tmp_path / "L.json"))
    assert final["experiment_id"] == "I6-DMO-LONG-DS12-STATIC-R0-FINAL"
    assert final["splits"]["train"] == FINAL_TRAIN
    assert final["horizons_hours"] == [18, 24, 36, 48, 60, 72]
    eff = D.effective_config(final)
    assert eff["mode"] == "FINAL_REFIT" and eff["family"] == "long"


def test_final_design_tamper_undecided_and_wrong_base_are_typed(tmp_path):
    final = D.final_refit_design(D.load_design(DESIGN))
    doc = json.loads(json.dumps(final))
    doc["splits"]["train"]["start"] = "2020-01-01T00:00:00Z"
    with pytest.raises(D.DesignError, match="FIXED_VALUE|FINAL_DESIGN_SEAL"):
        D.validate_design(doc)
    doc = json.loads(json.dumps(final))
    doc["training"]["max_epochs"] = 5
    with pytest.raises(D.DesignError, match="FINAL_DESIGN_SEAL|FIXED_VALUE"):
        D.validate_design(doc)
    with pytest.raises(D.DesignError, match="FINAL_FROM_VALIDATION_ONLY"):
        D.final_refit_design(final)  # final of a final
    assert D.FINAL_PLAN_OPEN_FIELDS == ()


def test_final_open_fields_emit_final_design_undecided(monkeypatch):
    monkeypatch.setattr(D, "FINAL_PLAN_OPEN_FIELDS", (("splits", "train"),))
    with pytest.raises(D.DesignError, match=r"FINAL_DESIGN_UNDECIDED.*splits\.train"):
        D.final_refit_design(D.load_design(DESIGN))


def test_unit_runner_selects_the_final_window(tmp_path):
    from tools.i6dmo import unit_runner
    design = json.loads(DESIGN.read_text())
    spec = {"kind": "final_refit", "family": "short", "effective_config": {"design": design},
            "window": {"train": [FINAL_TRAIN["start"], FINAL_TRAIN["end"]]}}
    selected = unit_runner._family_design(spec)["splits"]["train"]
    assert selected == FINAL_TRAIN  # AUD-DMO-02 no longer reproduces
    assert unit_runner._family_design({"kind": "fit", "family": "short",
                                       "effective_config": {"design": design}})["splits"]["train"] == \
        design["splits"]["train"]
    bad = dict(spec, window={"train": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]})
    with pytest.raises(ValueError, match="FINAL_WINDOW_MISMATCH"):
        unit_runner._family_design(bad)


def test_final_refit_real_cli_trains_on_the_final_window_only(tmp_path):
    from tools.i6dmo import unit_runner
    design_doc = unit_runner._family_design({"kind": "final_refit", "family": "short",
                                             "effective_config": {"design": json.loads(DESIGN.read_text())}})
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design_doc))
    bundle = final_bundle(tmp_path / "fb")
    cmd = [sys.executable, "-m", "tools.i6dmo.train", "--design", str(design_path), "--bundle", str(bundle),
           "--out", str(tmp_path / "u"), "--device", "cpu", "--json"]
    env = {**ENV, "I6DMO_TEST_TRAINING_OVERRIDES": json.dumps({"max_epochs": 2})}
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert run.returncode == 0, run.stdout[-1500:] + run.stderr[-1500:]
    receipt = json.loads(run.stdout)
    assert receipt["status"] == "COMPLETED" and receipt["scientific"] is False
    assert receipt["experiment_id"] == "I6-DMO-SHORT-DS2-STATIC-R0-FINAL"
    assert set(receipt["model_side_summary"]) & {"fit", "inner", "validation"} == {"fit", "inner"}
    scales = json.loads((tmp_path / "u" / "SCALES.json").read_text())
    assert scales["source_split"] == "fit" and scales["rows"] == 96  # final-window fit rows only
    assert not (tmp_path / "u" / "predictions" / "validation.npz").exists()


def test_validation_era_bundle_cannot_cross_into_the_final_identity(tmp_path):
    """DMO11 side: the 2020-2024 validation bundle and terminal never feed a final unit."""
    from tools.i6dmo import train as T
    final_eff = D.effective_config(D.final_refit_design(D.load_design(DESIGN)))
    validation_bundle = write_bundle(tmp_path / "vb")  # fit origins start 2020: outside the final window
    with pytest.raises(T.TrainDisposition, match="SPLIT_BOUNDARY_VIOLATION|BUNDLE_SCHEMA"):
        T.load_data_bundle(validation_bundle, final_eff)
    # And the final-window bundle is refused by the validation design (both directions).
    with pytest.raises(T.TrainDisposition, match="SPLIT_BOUNDARY_VIOLATION|BUNDLE_SCHEMA"):
        T.load_data_bundle(final_bundle(tmp_path / "fb"), T.effective_from_design(DESIGN))


def test_final_fit_labels_strictly_before_test(tmp_path):
    from tools.i6dmo import train as T
    final_eff = D.effective_config(D.final_refit_design(D.load_design(DESIGN)))
    splits = {"fit": split_arrays("2021-01-04T00:00:00", 24, 1),
              "inner": split_arrays("2024-12-31T20:00:00", 4, 2, invalid_rate=0.0)}  # +12h crosses 2025
    with pytest.raises(T.TrainDisposition, match="SPLIT_BOUNDARY_VIOLATION"):
        T.load_data_bundle(write_bundle(tmp_path / "x", splits=splits), final_eff)


# ------------------------------------------------------------------ AUD-DMO-03

def _gate(root, name, status="PASS", evidence=None):
    gates = root / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "i6dmo.gate.v1", "gate": name, "status": status, "splits_read": ["train", "validation"],
           "evidence": evidence or {}, "receipt_sha256": "cd" * 32}
    (gates / f"{name}.json").write_text(json.dumps(doc))
    return doc


def _ready_root(tmp_path):
    from tools.i6dmo import final as F
    root = tmp_path / "root"
    _gate(root, "PAIR_VALIDATION", evidence={"pair_status": "PASS"})
    _gate(root, "PROCEDURE_FREEZE", evidence={"final_fit_policy": {
        "train": [FINAL_TRAIN["start"], FINAL_TRAIN["end"]], "inner_tail_weeks": 8,
        "weights_frozen_through_test": True}})
    _gate(root, "DATA_PROVENANCE", evidence={"fred_clocks": {
        "fred.rates.dgs3mo.diff_5d": {"status": "VERIFIED"},
        "fred.rates.dgs7.logret_5d": {"status": "VERIFIED"}}})
    (root / "gates" / "TEST_READINESS.json").write_text(json.dumps(
        {"schema": "i6dmo.test_readiness.v1", "state": "READY", "receipts": {}, "readiness_sha256": "ef" * 32}))
    return root


def test_final_test_hook_is_registered_and_the_placeholder_is_gone(tmp_path):
    from tools.i6dmo import final as F, unit_runner
    assert unit_runner.HOOKS["final_test"] == "tools.i6dmo.final"
    terminal = F.run_unit({"kind": "final_test", "unit_id": "fixture.final", "identity_sha256": "a" * 64},
                          tmp_path / "out", tmp_path / "p.json")
    assert terminal["status"] == "FAILED"
    assert terminal["disposition"] != "FINAL_TEST_HOOK_NOT_PREPARED"  # AUD-DMO-03 no longer reproduces
    assert terminal["disposition"].startswith("FINAL_BLOCKED_GATE:")
    assert "TEST_READINESS" in terminal["disposition"]


@pytest.mark.parametrize(("spoil", "named"), [
    (lambda r: (r / "gates" / "TEST_READINESS.json").unlink(), "TEST_READINESS"),
    (lambda r: _gate(r, "PAIR_VALIDATION", status="FAIL"), "PAIR_VALIDATION"),
    (lambda r: _gate(r, "DATA_PROVENANCE", evidence={}), "DATA_PROVENANCE_FRED_CLOCKS"),
    (lambda r: _gate(r, "DATA_PROVENANCE", evidence={"fred_clocks": {
        "fred.rates.dgs3mo.diff_5d": {"status": "VERIFIED"},
        "fred.rates.dgs7.logret_5d": {"status": "UNVERIFIED"}}}), "DATA_PROVENANCE_FRED_CLOCKS"),
    (lambda r: _gate(r, "PROCEDURE_FREEZE", evidence={"final_fit_policy": {
        "train": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"], "inner_tail_weeks": 8,
        "weights_frozen_through_test": True}}), "PROCEDURE_FREEZE"),
])
def test_each_guard_refuses_with_the_missing_object(tmp_path, spoil, named):
    from tools.i6dmo import final as F
    root = _ready_root(tmp_path)
    spoil(root)
    with pytest.raises(F.FinalBlocked, match=named):
        F.guard_test(root)


def test_guarded_test_bundle_and_prediction_generation(tmp_path):
    from tools.i6dmo import final as F, train as T
    root = _ready_root(tmp_path)
    readiness = F.guard_test(root)
    final_design = D.final_refit_design(D.load_design(DESIGN))
    # Guard first: no readiness, no bundle.
    test_arrays = split_arrays("2025-01-06T00:00:00", 10, 7)
    with pytest.raises(F.FinalBlocked, match="TEST_READINESS"):
        F.build_test_bundle(final_design, {"test": test_arrays}, tmp_path / "nb", readiness=None)
    bundle = F.build_test_bundle(final_design, {"test": test_arrays}, tmp_path / "tb", readiness=readiness)
    manifest = json.loads((bundle / "BUNDLE.json").read_text())
    assert manifest["test_bundle"] is True and list(manifest["splits"]) == ["test"]
    assert manifest["readiness_sha256"] == readiness["readiness_sha256"]
    # The ordinary trainer refuses it: TEST arrays reach only the guarded final path.
    with pytest.raises(T.TrainDisposition, match="TEST_FORBIDDEN"):
        T.load_data_bundle(bundle, D.effective_config(final_design))
    # A trained final unit generates TEST predictions without retraining.
    unit = tmp_path / "unit"
    receipt = T.train_unit(F.design_file_for(final_design, tmp_path), final_bundle(tmp_path / "fb"), unit,
                           device="cpu", training_overrides={"max_epochs": 2})
    assert receipt["status"] == "COMPLETED"
    result = F.generate_test_predictions(unit, bundle, tmp_path / "preds", readiness=readiness)
    assert result["rows"] == 10 and result["weights_sha256"] == receipt["identity"]["weights_sha256"]
    with np.load(tmp_path / "preds" / "test.npz") as pz:
        assert pz["predicted_price"].shape == (10, 6)
        assert np.array_equal(pz["row_id"], test_arrays["row_id"])
        assert pz["horizons_hours"].tolist() == HORIZONS
    with pytest.raises(F.FinalBlocked, match="TEST_READINESS"):
        F.generate_test_predictions(unit, bundle, tmp_path / "p2", readiness=None)
