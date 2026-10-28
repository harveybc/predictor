"""I6-DMO typed design adapter (validate-design is pure) and DMO18 design side."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from tools.i6dmo import design as D
from tests.i6dmo.dmo_model_fixtures import DESIGN, load_design_doc

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _write(tmp_path, doc, name="design.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_committed_design_validates_and_effective_config_preserves_every_field():
    doc = D.load_design(DESIGN)
    eff = D.effective_config(doc)
    assert eff["design"] == load_design_doc()
    assert eff["design_sha256"] == D.design_sha256(load_design_doc())
    assert eff["horizons_hours"] == [2, 3, 4, 5, 6, 12]
    assert eff["offsets_hours"] == list(range(-24, 1, 2))
    assert eff["seed"] == 6102026
    assert eff["splits"]["fit"] == {"start": "2020-01-01T00:00:00Z", "end": "2023-11-06T00:00:00Z"}
    assert eff["splits"]["inner"] == {"start": "2023-11-06T00:00:00Z", "end": "2024-01-01T00:00:00Z"}
    assert eff["splits"]["test"]["open"] is False
    assert len(eff["feature_manifest"]["features"]) == 19
    assert eff["effective_config_sha256"] == D.effective_config_sha256(eff)
    # Any design field change changes the identity, including non-model fields.
    changed = load_design_doc()
    changed["execution"]["one_heavy_job_per_host"] = False
    assert D.design_sha256(changed) != eff["design_sha256"]


@pytest.mark.parametrize("path", [(), ("input",), ("model",), ("model", "core"), ("training",),
                                  ("splits",), ("business",), ("metrics",), ("execution",),
                                  ("feature_policy",), ("model", "heads")])
def test_unknown_fields_are_rejected_at_every_level(path):
    doc = load_design_doc()
    node = doc
    for key in path:
        node = node[key]
    node["surprise"] = 1
    with pytest.raises(D.DesignError, match="UNKNOWN_FIELD|SCHEMA"):
        D.validate_design(doc)


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda d: d.pop("seed"), "MISSING_FIELD"),
    (lambda d: d.update(horizons_hours=[2, 4, 6, 12, 24, 48]), "FIXED_VALUE"),
    (lambda d: d["model"]["branch"].update(filters=16), "FIXED_VALUE"),
    (lambda d: d["model"]["core"].update(transformer_blocks=2), "FIXED_VALUE"),
    (lambda d: d["model"]["heads"].update(hidden_units=[16]), "FIXED_VALUE"),
    (lambda d: d["training"].update(max_epochs=100), "FIXED_VALUE"),
    (lambda d: d["training"].update(outer_validation_for_checkpoint_selection=True), "FIXED_VALUE"),
    (lambda d: d["splits"]["test"].update(open=True), "TEST_FORBIDDEN"),
    (lambda d: d["input"].update(spacing_hours=1), "FIXED_VALUE"),
    (lambda d: d.update(seed=1), "FIXED_VALUE"),
    (lambda d: d["input"].update(smoothing=True), "FIXED_VALUE"),
    (lambda d: d["model"].update(regime="R2"), "FIXED_VALUE"),
    (lambda d: d.update(seed=True), "TYPE"),
])
def test_fixed_section_c_decisions_cannot_be_improvised(mutate, code):
    doc = load_design_doc()
    mutate(doc)
    with pytest.raises(D.DesignError, match=code):
        D.validate_design(doc)


@pytest.mark.parametrize("exclude", [[], ["STLFSI4"], ["fred.stress.stlfsi4"]])
def test_dmo18_stlfsi4_exclusion_cannot_be_renamed_or_dropped(exclude):
    doc = load_design_doc()
    doc["feature_policy"]["exclude"] = exclude
    with pytest.raises(D.DesignError, match="STLFSI4"):
        D.validate_design(doc)
    doc = load_design_doc()
    doc["feature_policy"]["reject_stlfsi4_aliases"] = False
    with pytest.raises(D.DesignError, match="FIXED_VALUE"):
        D.validate_design(doc)


@pytest.mark.parametrize("name", ["STLFSI4", "fred.stress.stlfsi4.level", "FRED:STLFSI4_lag24",
                                  "fred.stress.st-lfsi4.level", "x.StLfSi4.v2"])
def test_dmo18_stlfsi4_aliases_are_detected(name):
    with pytest.raises(D.DesignError, match="STLFSI4_ALIAS"):
        D.check_feature_identity([name])


def test_dmo18_feature_identity_is_exactly_the_gated_nineteen():
    from tools.i6n_contracts.feature_availability import LEGACY_MANIFEST_20, MANIFEST_NO_STLFSI4
    report = D.check_feature_identity(list(MANIFEST_NO_STLFSI4))
    assert report["features"] == list(MANIFEST_NO_STLFSI4) and report["count"] == 19
    with pytest.raises(D.DesignError, match="STLFSI4_ALIAS"):
        D.check_feature_identity(list(LEGACY_MANIFEST_20))
    reordered = list(MANIFEST_NO_STLFSI4)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(D.DesignError, match="FEATURE_IDENTITY_MISMATCH"):
        D.check_feature_identity(reordered)
    with pytest.raises(D.DesignError, match="FEATURE_IDENTITY_MISMATCH"):
        D.check_feature_identity(list(MANIFEST_NO_STLFSI4)[:18])


def test_duplicate_keys_and_nan_are_rejected(tmp_path):
    path = tmp_path / "dup.json"
    text = DESIGN.read_text(encoding="utf-8")
    path.write_text(text.replace('"seed": 6102026,', '"seed": 6102026, "seed": 6102026,'), encoding="utf-8")
    with pytest.raises(D.DesignError, match="DUPLICATE_KEY"):
        D.load_design(path)
    path.write_text(text.replace('"min_delta": 1e-8', '"min_delta": NaN'), encoding="utf-8")
    with pytest.raises(D.DesignError, match="NONFINITE"):
        D.load_design(path)


def test_validate_design_report_is_pure_and_typed(tmp_path):
    ok = D.validate_design_report(DESIGN)
    assert ok["status"] == "PASS" and ok["pure"] is True
    assert ok["design_sha256"] == D.design_sha256(load_design_doc())
    assert ok["feature_count"] == 19 and ok["test_open"] is False
    assert isinstance(ok["unverified_features"], list)
    bad = load_design_doc()
    bad["splits"]["test"]["open"] = True
    report = D.validate_design_report(_write(tmp_path, bad))
    assert report["status"] == "FAILED" and report["code"] == "TEST_FORBIDDEN"


def test_validate_design_cli_json(tmp_path):
    out = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "validate-design",
                          "--design", str(DESIGN), "--json"], capture_output=True, text=True,
                         check=False, cwd=REPO, env=ENV)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["status"] == "PASS"
    bad = copy.deepcopy(load_design_doc())
    bad["horizons_hours"] = [1]
    out = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "validate-design",
                          "--design", str(_write(tmp_path, bad)), "--json"],
                         capture_output=True, text=True, check=False, cwd=REPO, env=ENV)
    assert out.returncode == 2 and json.loads(out.stdout)["code"] == "FIXED_VALUE"
