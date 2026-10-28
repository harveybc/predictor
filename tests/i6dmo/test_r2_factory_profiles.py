"""Factory consumes the ruled selected arm, not a hard-coded short alias."""
import json
from pathlib import Path

from tests.i6dmo.ops_fakes import private_spec
from tools.i6dmo import campaign as C, design as D, r2_design as RD

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/i6n/direct_multi_output_static"


def test_long_factory_uses_long_family_and_exact_arm(tmp_path, monkeypatch):
    design_path = BASE / "r2_downsampled/long_source/EXPERIMENT.mg12m.json"
    design = D.load_design(design_path)
    effective = D.effective_config(design)
    contract = json.loads((BASE / "matched_short/R2_CONTRAST_DESIGN.ruled.json").read_text())
    contract["shared"].update(arm="mg12m", experiment_seal_sha256=design["seal_sha256"],
        experiment_design_sha256=effective["design_sha256"], horizons_hours=[18,24,36,48,60,72],
        population_sha256="a" * 64)
    contract["contrast_id"] = "EXPLICIT_LONG_FACTORY_FIXTURE"
    # This isolates factory consumption; real design validation is covered separately.
    monkeypatch.setattr(RD, "validate_r2_design", lambda *a, **kw: None)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    deploy = tmp_path / "deployment.json"
    deploy.write_text(json.dumps(private_spec(tmp_path)))
    pins = dict(common_population=dict(base="experiment_storage", relpath="population", sha256="b"*64,
                                       population_sha256="a"*64),
                matched_input=dict(relpath="input", sha256="c"*64))
    root = tmp_path / "campaign"
    C.init_r2_contrast(path, design_path, deploy, root, data_pins=pins, code_commit="d"*40)
    units = C.load_plan(root)["units"]
    assert len(units) == 25
    assert all(u["arm"] == "mg12m" for u in units)
    assert all(u["family"] == "long" for u in units if u["kind"] != "comparison")
    e = C.load_effective(root)
    assert C.r2_features(e) == effective["feature_manifest"]["features"]
    assert e["comparison"]["horizons"] == [18,24,36,48,60,72]
    assert not any(u["test_open"] for u in units)


def test_recovered_donor_provenance_stays_original():
    original = dict(campaign_id="original", ruling_seal_sha256="a"*64,
                    code_pin="b"*40, units={"f": "r2.pretrain.00"})
    effective = dict(r2=dict(recovered_donor_provenance=original))
    result = C.r2_provenance(effective, dict(campaign_id="successor"))
    assert result == original
    result["units"]["f"] = "changed"
    assert effective["r2"]["recovered_donor_provenance"] == original


def test_recovery_pin_changes_r2_identity_but_not_control_identity():
    import copy
    effective = dict(arms={"mg2m": dict(design={"x": 1}, adapter_effective={"y": 2})},
        r2=dict(contract_seal_sha256="a"*64, objective={"name": "O1"},
                donor_recovery={"manifest_sha256": "b"*64, "original_donors": {"f": "c"*64}}))
    changed = copy.deepcopy(effective)
    changed["r2"]["donor_recovery"]["original_donors"]["f"] = "d"*64
    assert C._family_sha(effective, "r2_R0") == C._family_sha(changed, "r2_R0")
    assert C._family_sha(effective, "r2_R2") != C._family_sha(changed, "r2_R2")
    assert C._family_sha(effective, "r2_pretrain") != C._family_sha(changed, "r2_pretrain")


def test_invalid_recovery_refused_before_campaign_files(tmp_path, monkeypatch):
    import pytest
    from tools.i6dmo import r2_recovery as RR
    contract = json.loads((BASE / "matched_short/R2_CONTRAST_DESIGN.ruled.json").read_text())
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(private_spec(tmp_path)))
    pins = dict(common_population=dict(base="experiment_storage", relpath="population", sha256="b"*64,
        population_sha256=contract["shared"]["population_sha256"]),
        matched_input=dict(relpath="input", sha256="c"*64))
    def refuse(*args, **kwargs):
        raise RR.RecoveryError("RECOVERY_MANIFEST_FILE_DIGEST")
    monkeypatch.setattr(RR, "load_manifest", refuse)
    root = tmp_path / "new-campaign"
    with pytest.raises(RR.RecoveryError, match="RECOVERY_MANIFEST_FILE_DIGEST"):
        C.init_r2_contrast(BASE / "matched_short/R2_CONTRAST_DESIGN.ruled.json",
            BASE / "matched_short/EXPERIMENT.mg2m.json", spec, root, data_pins=pins,
            donor_recovery=dict(path="invalid", sha256="a"*64, r0_target={}))
    assert not root.exists()
