"""O3/P2: the next development design is sealed per variable, inherits its thresholds on every
family protocol, controls capacity, calibrates each hypothesis on its own contract, maps
replication explicitly to the same regime, binds resources, separates stages, executes nothing."""
import hashlib
import importlib.util
import json
import sys
import tempfile
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
         "alpha": 0.05, "min_rows_per_block": 30, "ridge_lambda": 1.0, "seed": 3}
PILOT_UNITS = ["bumps__s11", "sinusoid__s11", "steps__s11"]
OPS = ["mad_extremes_trailing", "delta_run_length", "cusum_causal"]
_BANK = Path(tempfile.mkdtemp(prefix="utility-design-bank-"))


def _unit(bank, unit_id, family, seed, snr="10", n=400, missing="none"):
    d = bank / unit_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    np.save(d / "observed_signal.npy", x)
    np.save(d / "missing_mask.npy", np.zeros(n, dtype=bool))
    digest = hashlib.sha256(x.tobytes()).hexdigest()
    (d / "UNIT.json").write_text(json.dumps({
        "unit_id": unit_id, "family": family, "seed": seed, "n_samples": n, "n_variables": 1,
        "perturbation": "white", "declared_snr_db": snr, "missingness": {"kind": missing},
        "unit_params": {"family": family, "length": n, "snr_db": snr, "missingness": {"kind": missing}},
        "digests": {"observed_signal": digest}, "variable_names": ["v0"],
        "generator": {"version": "test.v1"}, "partitions": {"train": [0, n // 2]}}))
    return unit_id


def _cells(bank, units, operators=OPS, verdict="MECHANICALLY_ACCEPTED"):
    cells = [{"unit": u, "variable": "v0", "operator": k, "verdict": verdict,
              "spec_sha256": contract.spec_sha256(ops.build(k).describe())} for u in units for k in operators]
    p = bank / "cells.json"
    p.write_text(json.dumps({"schema": "d3_mechanics_cells.v1", "verified": True, "freeze_sha256": "f" * 64,
                             "design_sha256": "d" * 64, "run_id": "t", "cells": cells}))
    return p


def _bank():
    if not (_BANK / "cells.json").is_file():
        for fam in ("bumps", "sinusoid"):
            for seed in (12, 13):
                _unit(_BANK, f"{fam}__s{seed}", fam, seed)
        _unit(_BANK, "bumps__s14_snr0", "bumps", 14, snr="0")
        _unit(_BANK, "bumps__s12_copy", "bumps", 12)
        _cells(_BANK, [f"{f}__s{s}" for f in ("bumps", "sinusoid") for s in (12, 13)] + ["bumps__s14_snr0", "bumps__s12_copy"])
    return _BANK


def design(**over):
    bank = _bank()
    kw = dict(bank_root=bank, cells_record=bank / "cells.json",
              replication_map={"bumps__s12": "bumps__s13", "sinusoid__s12": "sinusoid__s13"},
              operators=OPS, inherited_protocol=PILOT, pilot_units=PILOT_UNITS)
    kw.update(over)
    return D.build_design(**kw)


def _resealed(doc):
    doc = json.loads(json.dumps(doc))
    doc["design_sha256"] = D.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def _validate(doc, **over):
    kw = dict(inherited_protocol=PILOT, pilot_units=PILOT_UNITS, bank_root=_bank())
    kw.update(over)
    D.validate_design(doc, **kw)


def test_P2_the_design_seals_one_family_per_variable_with_inherited_thresholds_and_one_contract_per_hypothesis():
    doc = design()
    assert doc["schema"] == D.DESIGN_SCHEMA and doc["stage"] == "DEVELOPMENT_SELECTION"
    assert [f["role"] for f in doc["families"]] == ["selection", "replication"] * 2
    fam = doc["families"][0]
    assert fam["comparisons"] == 6 and fam["alpha_adjusted"] == pytest.approx(0.05 / 6)
    assert fam["calibration_plan"]["n_sims"] == H.sims_required_for_zero(0.05 / 6, 0.95) == 358
    assert len(fam["calibration_contracts"]) == 6 and doc["calibration_contracts_total"] == 24
    ha = [c for c in fam["calibration_contracts"] if c["hypothesis"] == "H_A"][0]
    assert (ha["branch_a"], ha["branch_b"], ha["widths"]) == ("raw_wide", "augmented", {"a": 8, "b": 8})
    assert fam["n"] == 400 == fam["resource"]["n"] and fam["resource"]["data_sha256"]
    assert doc["families"][1]["replica_of"] == "bumps__s12"
    assert "IMPROVE" in doc["hypotheses"]["H_T"]["question"]
    assert "NOT a global control" in doc["multiplicity_policy"]
    for f in doc["families"]:
        p = D._protocol_from(f["protocol"])
        assert p.margin == 0.0 and p.horizon == 1 and p.ridge_lambda == 1.0 and p.calibration is None
    _validate(doc)


def test_P2_every_family_protocol_is_checked_recursively_not_only_the_outer_digest():
    doc = design()
    for field, value in (("margin", 99.0), ("horizon", 3), ("window", 6), ("ridge_lambda", 0.0),
                         ("model", "logistic"), ("n_blocks", 5), ("alpha", 0.2)):
        bad = json.loads(json.dumps(doc))
        bad["families"][0]["protocol"][field] = value
        with pytest.raises(D.DesignRefusal, match="protocol"):
            _validate(_resealed(bad))
    # members duplicated, removed, or hypotheses swapped — outer digest kept correct
    dup = json.loads(json.dumps(doc))
    dup["families"][0]["members"].append(dup["families"][0]["members"][0])
    with pytest.raises(D.DesignRefusal, match="duplicate or missing|cardinality"):
        _validate(_resealed(dup))
    gone = json.loads(json.dumps(doc))
    gone["families"][0]["members"].pop()
    with pytest.raises(D.DesignRefusal, match="duplicate or missing|cardinality|member list"):
        _validate(_resealed(gone))
    swapped = json.loads(json.dumps(doc))
    for m in swapped["families"][0]["members"]:
        m["hypothesis"] = "H_A" if m["hypothesis"] == "H_T" else "H_T"
    with pytest.raises(D.DesignRefusal, match="branch pair"):
        _validate(_resealed(swapped))
    plan = json.loads(json.dumps(doc))
    plan["families"][0]["calibration_plan"]["n_sims"] = 10
    with pytest.raises(D.DesignRefusal, match="simulations|plan"):
        _validate(_resealed(plan))
    with pytest.raises(D.DesignRefusal, match="digest"):
        D.validate_design(plan, inherited_protocol=PILOT, pilot_units=PILOT_UNITS)


def test_P2_thresholds_chosen_after_results_are_refused():
    doc = design()
    for change in ({"margin": 0.01}, {"alpha": 0.1}, {"horizon": 5}, {"ridge_lambda": 2.0},
                   {"target": "direction", "model": "logistic"}):
        with pytest.raises(D.DesignRefusal, match="inherited"):
            _validate(doc, inherited_protocol={**PILOT, **change})


def test_P2_replication_is_an_explicit_map_to_the_same_regime_with_different_seed_and_data():
    with pytest.raises(D.DesignRefusal, match="same generator family|regime|SNR"):
        design(replication_map={"bumps__s12": "bumps__s14_snr0", "sinusoid__s12": "sinusoid__s13"})
    with pytest.raises(D.DesignRefusal, match="same generator family|variable"):
        design(replication_map={"bumps__s12": "sinusoid__s13", "sinusoid__s12": "bumps__s13"})
    with pytest.raises(D.DesignRefusal, match="same seed or same data"):
        design(replication_map={"bumps__s12": "bumps__s12_copy", "sinusoid__s12": "sinusoid__s13"})
    with pytest.raises(D.DesignRefusal, match="pilot"):
        design(pilot_units=["bumps__s12"])
    doc = design()
    resource = json.loads(json.dumps(doc))
    resource["families"][0]["resource"]["n"] = 401
    resource["families"][0]["n"] = 401
    with pytest.raises(D.DesignRefusal, match="resource|length"):
        _validate(_resealed(resource))


def test_P2_eligibility_comes_from_the_bound_cells_record_and_other_stages_are_refused():
    bank = _bank()
    cells = bank / "cells_refused.json"
    body = json.loads((bank / "cells.json").read_text())
    for c in body["cells"]:
        if c["unit"] == "bumps__s13" and c["operator"] == "cusum_causal":
            c["verdict"] = "MECHANICALLY_REFUSED"
    cells.write_text(json.dumps(body))
    with pytest.raises(D.DesignRefusal, match="MECHANICALLY_ACCEPTED"):
        design(cells_record=cells)
    for stage in ("PUBLIC_CONFIRMATION", "FINANCIAL_REVALIDATION", "FLOW_DIAGNOSTIC"):
        with pytest.raises(D.DesignRefusal, match="own order"):
            design(stage=stage)
    doc = design()
    ran = json.loads(json.dumps(doc))
    ran["execution"] = "RUN"
    with pytest.raises(D.DesignRefusal, match="executes nothing"):
        _validate(_resealed(ran))


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
