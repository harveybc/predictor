"""O3: the next development design is sealed per variable, inherits its thresholds, controls
capacity, replicates independently, separates stages and executes nothing."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("df_utility_harness")
D = _load("df_utility_next_design")
ops = _load("df_d3_operators")
contract = _load("df_d3_contract")

PILOT = {"target": "return", "horizon": 1, "model": "ridge", "window": 4, "n_blocks": 4, "margin": 0.0,
         "alpha": 0.05, "min_rows_per_block": 30, "seed": 3}
PILOT_UNITS = ["bumps__seed11", "sinusoid__seed11", "steps__seed11"]
OPS = ["mad_extremes_trailing", "delta_run_length", "cusum_causal"]


def design(**over):
    kw = dict(variables=[{"unit": "bumps__seed21", "variable": "v0", "n": 2048}],
              replication=[{"unit": "bumps__seed31", "variable": "v0", "n": 2048},
                           {"unit": "bumps__seed41", "variable": "v0", "n": 2048}],
              operators=OPS, inherited_protocol=PILOT, pilot_units=PILOT_UNITS, cells_record="cells.json")
    kw.update(over)
    return D.build_design(**kw)


def test_O3_the_design_seals_one_family_per_variable_with_inherited_thresholds_and_derived_plans():
    doc = design()
    assert doc["schema"] == D.DESIGN_SCHEMA and doc["stage"] == "DEVELOPMENT_SELECTION"
    assert set(doc["stages_out_of_scope"]) == {"FLOW_DIAGNOSTIC", "PUBLIC_CONFIRMATION", "FINANCIAL_REVALIDATION"}
    assert len(doc["families"]) == 3 and [f["role"] for f in doc["families"]] == ["selection", "replication", "replication"]
    fam = doc["families"][0]
    assert fam["comparisons"] == 6 and fam["alpha_adjusted"] == pytest.approx(0.05 / 6)
    assert fam["calibration_plan"]["n_sims"] == H.sims_required_for_zero(0.05 / 6, 0.95)
    assert {m["hypothesis"] for m in fam["members"]} == {"H_T", "H_A"}
    aug = [m for m in fam["members"] if m["branch_b"] == "augmented"]
    assert aug and all(m["branch_a"] == "raw_wide" for m in aug)
    assert fam["protocol"]["margin"] == 0.0 and fam["protocol"]["horizon"] == 1 and fam["protocol"]["target"] == "return"
    # the protocol seals under the harness with the wide raw branch
    H.Protocol(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in fam["protocol"].items()
                  if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")})
    assert doc["design_sha256"] == D.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    assert doc["execution"].startswith("NONE")
    D.validate_design(doc, inherited_protocol=PILOT, pilot_units=PILOT_UNITS)


def test_O3_thresholds_chosen_after_results_are_refused():
    for change in ({"margin": 0.01}, {"alpha": 0.1}, {"horizon": 5}, {"target": "direction", "model": "logistic"}):
        with pytest.raises(D.DesignRefusal, match="inherited|frozen"):
            doc = design()
            D.validate_design(doc, inherited_protocol={**PILOT, **change}, pilot_units=PILOT_UNITS)


def test_O3_replication_must_be_independent_and_families_never_pool_units():
    with pytest.raises(D.DesignRefusal, match="independent"):
        design(replication=[{"unit": "bumps__seed21", "variable": "v0", "n": 2048}])
    with pytest.raises(D.DesignRefusal, match="pilot"):
        design(variables=[{"unit": PILOT_UNITS[0], "variable": "v0", "n": 2048}])
    with pytest.raises(D.DesignRefusal, match="replication"):
        design(replication=[])
    doc = design()
    pooled = json.loads(json.dumps(doc))
    pooled["families"][0]["members"][0]["unit"] = "other"
    pooled["design_sha256"] = D.sha_obj({k: v for k, v in pooled.items() if k != "design_sha256"})
    with pytest.raises(D.DesignRefusal, match="pools"):
        D.validate_design(pooled, inherited_protocol=PILOT, pilot_units=PILOT_UNITS)


def test_O3_augmented_without_capacity_control_and_other_stages_are_refused():
    doc = design()
    bad = json.loads(json.dumps(doc))
    bad["hypotheses"]["H_A"] = {**bad["hypotheses"]["H_A"], "branch_a": "raw"}
    for m in bad["families"][0]["members"]:
        if m["branch_b"] == "augmented":
            m["branch_a"] = "raw"
    bad["design_sha256"] = D.sha_obj({k: v for k, v in bad.items() if k != "design_sha256"})
    with pytest.raises(D.DesignRefusal, match="capacity control"):
        D.validate_design(bad, inherited_protocol=PILOT, pilot_units=PILOT_UNITS)
    for stage in ("PUBLIC_CONFIRMATION", "FINANCIAL_REVALIDATION", "FLOW_DIAGNOSTIC"):
        with pytest.raises(D.DesignRefusal, match="own order"):
            design(stage=stage)
    tampered = json.loads(json.dumps(doc))
    tampered["families"][0]["calibration_plan"]["n_sims"] = 10
    with pytest.raises(D.DesignRefusal, match="digest"):
        D.validate_design(tampered, inherited_protocol=PILOT, pilot_units=PILOT_UNITS)


def test_O3_the_capacity_control_branch_has_the_width_of_augmented_and_contrasts_run():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(size=600))
    s = H.series(x)
    fam = ("u__v0__mad_extremes_trailing__augmented",)
    p = H.Protocol(target="return", horizon=1, model="ridge", window=4, n_blocks=4, margin=0.0, seed=7,
                   family=fam, min_rows_per_block=30, branches=("raw", "transformed", "augmented", "raw_wide"))
    mad = ops.build("mad_extremes_trailing")
    rep = H.represent(mad, s, 300)
    Xw, okw, _ = H.features("raw_wide", s, None, p)
    Xa, oka, _ = H.features("augmented", s, rep, p)
    assert Xw.shape[1] == Xa.shape[1] == 2 * p.window
    elig = {"freeze_sha256": "f" * 64, "design_sha256": "d" * 64,
            "cells": {("u", "v0", mad.KIND): {"verdict": "MECHANICALLY_ACCEPTED",
                                             "spec_sha256": contract.spec_sha256(mad.describe())}}}
    out = H.contrast(s, mad, p, contrast_id=fam[0], eligibility=elig, unit="u", variable="v0",
                     branch_a="raw_wide", branch_b="augmented")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED and np.isfinite(out["delta_mean"])
    assert out["branch_a"] == "raw_wide" and out["representation"]["operator"] == mad.KIND
