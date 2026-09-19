"""RP18: the contrast-bound effects. Numeric oracles built from a known model of the cell values
(base + fusion + readout + interaction + donor), through the real `effects()` on a synthetic closure:
pure readout without fusion -> gamma of the common pair is exactly zero; additive effects -> zero
interaction and gamma common = fusion(r1) - fusion(r0); a known interaction is recovered with sign and
magnitude; permuting names/order changes nothing; a missing arm gives INCOMPLETE with n; a foreign
design, a duplicated member, an unexpected arm and a non-finite value refuse; the bootstrap reuses
the estimator (manual count); the old 'average of the survivors' estimator is a mutant that FAILS
these oracles for the expected reason."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AD = _load("df_mod_e0_arch_design")
AV = _load("df_mod_e0_arch_verify")


def _design(**kw):
    return AD.build(levels=[0, 3], replicates=[1, 2], random_assignments=2, max_updates=6, readout_controls_r=[1], hosts=["COORDINATOR"], **kw)


def _values(design, *, base=None, fusion=None, readout=None, interaction=None, donor=0.0, h2=None, seed_noise=0.0):
    """Cell value = base[arch] + fusion_r[r] * [sequence fusion] + readout[r] * [last readout] + interaction[r] * [seq & last]
    + donor * [dsum donor] + h2 profiles offset; a per-seed shift cancels in every within-replicate contrast."""
    base = base or {"A": 0.6, "B": 0.55, "C": 0.58, "0": 0.7}
    fusion = fusion or {0: 0.0, 1: 0.0}
    readout = readout or {0: 0.0, 1: 0.0}
    interaction = interaction or {0: 0.0, 1: 0.0}
    h2 = h2 or {0: 0.0, 3: 0.0}
    out = {}
    for c in design["cells"]:
        m = AV.parse_id(c["cell_id"])
        v = base[c["arch"]] + seed_noise * (c["seed"] - 1)
        if m["hypothesis"] == "H3" and c["arm"] not in ("extractor", "extractor_summary"):
            r = c["r"]
            seq = c["arm"] in ("sequence", "sequence_gap")
            last = c["arm"] in ("sequence", "summary_last")
            v += (fusion[r] if seq else 0.0) + (readout[r] if last else 0.0) + (interaction[r] if (seq and last) else 0.0)
            if m["donor"] == "summary":
                v += donor
        if m["hypothesis"] == "H2":
            v += h2[c["level"]] if c["arm"] == "profiles" else 0.0
        out[c["cell_id"]] = v
    return out


def _closure(design, values, status=None):
    units = {}
    for c in design["cells"]:
        cid = c["cell_id"]
        v = values[cid]
        rec = {"arch": c["arch"], "arm": c["arm"], "hypothesis": c["hypothesis"], "seed": c["seed"], "level": c["level"], "r": c["r"],
               "donor": c.get("donor"), "mase": {"validation": v, "test": v}, "mae": {"validation": v, "test": v},
               "naive_mase": {"validation": 1.0, "test": 1.0}, "linear_mase": {"validation": 0.62, "test": 0.62}, "oracle_mase": {"validation": 0.4, "test": 0.4},
               "denominator": [0.5] * 8, "updates": 10, "stop_reason": "EARLY_STOPPING", "cost": {"cpu_seconds": 1.0}}
        units[cid] = {"status": (status or {}).get(cid, "VERIFIED"), "role": "CELL", "record": rec, "problems": []}
    return {"design_sha256": design["design_sha256"], "population": {"members": [c["cell_id"] for c in design["cells"]]}, "closure": "TOTAL", "units": units}


def test_RP18_pure_readout_without_fusion_gives_zero_gamma_for_the_common_pair_and_the_readout_effect():
    d = _design()
    vals = _values(d, readout={0: -0.5, 1: -0.5})            # last readouts 0.5 better; identical at r = 0 and r = 1; no fusion effect
    eff = AV.effects(_closure(d, vals), d, n_boot=50)
    for a, p in eff["per_arch"].items():
        # the common pair (sequence = seq & last, summary = summ & pooled) carries the readout at BOTH r: constant -> gamma 0
        assert p["COMMON"]["d"][0]["value"] == pytest.approx(-0.5) and p["COMMON"]["d"][1]["value"] == pytest.approx(-0.5)
        assert p["COMMON"]["gamma_common_pair"]["state"] == AV.ESTIMATED and p["COMMON"]["gamma_common_pair"]["value"] == pytest.approx(0.0)
        assert p["FACT"][1]["fusion"]["value"] == pytest.approx(0.0) and p["FACT"][1]["readout"]["value"] == pytest.approx(-0.5)
        assert p["FACT"][1]["interaction"]["value"] == pytest.approx(0.0)
        assert p["FACT"]["gamma_factorial"]["state"] == AV.NOT_ESTIMABLE                      # the 2 x 2 exists at r = 1 only
        assert p["READOUT"][1]["sequence_fusion"]["value"] == pytest.approx(-0.5) and p["READOUT"][1]["summary_fusion"]["value"] == pytest.approx(-0.5)
        assert set(p["COMMON"]["d"][1]["per_replicate"]) == {1, 2} and p["COMMON"]["d"][1]["complete"]


def test_RP18_additive_effects_have_zero_interaction_and_gamma_common_equals_fusion_difference():
    d = _design()
    vals = _values(d, fusion={0: -0.02, 1: -0.09}, readout={0: -0.1, 1: -0.1}, seed_noise=0.01)
    eff = AV.effects(_closure(d, vals), d, n_boot=50)
    for a, p in eff["per_arch"].items():
        # the common pair (seq last vs summ gap) mixes fusion and readout: d_r = fusion_r + readout
        assert p["COMMON"]["d"][0]["value"] == pytest.approx(-0.02 - 0.1) and p["COMMON"]["d"][1]["value"] == pytest.approx(-0.09 - 0.1)
        assert p["COMMON"]["gamma_common_pair"]["value"] == pytest.approx(-0.07)         # the readout cancels in the SAME pair
        assert p["FACT"][1]["fusion"]["value"] == pytest.approx(-0.09) and p["FACT"][1]["readout"]["value"] == pytest.approx(-0.1)
        assert abs(p["FACT"][1]["interaction"]["value"]) < 1e-12
        assert p["COMMON"]["d"][1]["sd_replicates"] == pytest.approx(0.0)                  # the seed shift cancels within replicate


def test_RP18_a_known_interaction_is_recovered_with_sign_and_magnitude():
    d = _design()
    vals = _values(d, fusion={0: 0.0, 1: -0.05}, readout={0: 0.0, 1: -0.1}, interaction={0: 0.0, 1: +0.03})
    eff = AV.effects(_closure(d, vals), d, n_boot=0)
    p = eff["per_arch"]["A"]
    # 2 x 2 at r = 1: seq_last = f + rd + i, seq_gap = f, summ_last = rd, summ_gap = 0
    assert p["FACT"][1]["interaction"]["value"] == pytest.approx(0.03)
    assert p["FACT"][1]["fusion"]["value"] == pytest.approx(-0.05 + 0.03 / 2) and p["FACT"][1]["readout"]["value"] == pytest.approx(-0.1 + 0.03 / 2)
    assert p["COMMON"]["d"][1]["value"] == pytest.approx(-0.05 - 0.1 + 0.03) and p["COMMON"]["gamma_common_pair"]["value"] == pytest.approx(-0.12)


def test_RP18_donor_delta_uses_the_same_common_pair_and_H2_needs_every_control():
    d = _design()
    vals = _values(d, fusion={0: -0.02, 1: -0.09}, donor=+0.04, h2={0: -0.01, 3: +0.02})
    # the donor offset applies to both dsum arms: the pair difference is unchanged -> delta 0; make it pair-specific
    for cid in list(vals):
        if cid.endswith("__sequence__dsum"):
            vals[cid] += 0.05
    eff = AV.effects(_closure(d, vals), d, n_boot=0)
    assert eff["per_arch"]["A"]["DONOR"][1]["value"] == pytest.approx(0.05) and eff["per_arch"]["C"]["DONOR"] is None
    assert eff["per_arch"]["A"]["H2"]["e"][0]["value"] == pytest.approx(-0.01) and eff["per_arch"]["A"]["H2"]["e"][3]["value"] == pytest.approx(0.02)
    assert eff["per_arch"]["A"]["H2"]["slope"] == pytest.approx(0.03 / 3)
    # one random control missing -> e(h) INCOMPLETE for that replicate, slope not estimated, other levels intact
    c = _closure(d, vals, status={"H2__h3__s2__A__random_1": "PROBLEMS"})
    eff2 = AV.effects(c, d, n_boot=0)
    e3 = eff2["per_arch"]["A"]["H2"]["e"][3]
    assert e3["state"] == AV.INCOMPLETE and e3["n_expected"] == 2 and e3["n_observed"] == 1 and e3["missing"] == {2: ["H2__h3__s2__A__random_1"]}
    assert eff2["per_arch"]["A"]["H2"]["slope"] is None and eff2["per_arch"]["A"]["H2"]["e"][0]["state"] == AV.ESTIMATED
    assert eff2["per_arch"]["B"]["H2"]["e"][3]["state"] == AV.ESTIMATED                        # other architectures untouched


def test_RP18_a_missing_arm_is_INCOMPLETE_never_filled_and_the_other_pairs_keep_their_weights():
    d = _design()
    vals = _values(d, fusion={0: -0.02, 1: -0.09}, readout={0: -0.1, 1: -0.1})
    c = _closure(d, vals, status={"H3__r1__s2__A__summary": "PROBLEMS"})
    eff = AV.effects(c, d, n_boot=0)
    p = eff["per_arch"]["A"]
    assert p["COMMON"]["d"][1]["state"] == AV.INCOMPLETE and p["COMMON"]["d"][1]["n_observed"] == 1 and p["COMMON"]["d"][1]["value"] == pytest.approx(-0.19)
    assert p["COMMON"]["gamma_common_pair"]["state"] == AV.NOT_ESTIMABLE and p["COMMON"]["gamma_common_pair"]["value"] is None
    assert p["FACT"][1]["fusion"]["state"] == AV.INCOMPLETE and p["FACT"][1]["fusion"]["n_observed"] == 1
    assert p["COMMON"]["d"][0]["state"] == AV.ESTIMATED and eff["per_arch"]["B"]["COMMON"]["gamma_common_pair"]["value"] == pytest.approx(-0.07)
    assert eff["population"]["not_verified"] == {"H3__r1__s2__A__summary": "PROBLEMS"}
    assert eff["bootstrap"]["A"]["gamma_common_pair"] is None or eff["bootstrap"]["A"]["gamma_common_pair"]["n"] == 0


def test_RP18_names_and_order_do_not_matter_and_the_bootstrap_reuses_the_estimator():
    d = _design()
    vals = _values(d, fusion={0: -0.02, 1: -0.09}, readout={0: -0.1, 1: -0.1}, seed_noise=0.05)
    c = _closure(d, vals)
    eff = AV.effects(c, d, n_boot=200, seed=3)
    shuffled = {"design_sha256": c["design_sha256"], "population": c["population"], "closure": c["closure"], "units": dict(reversed(list(c["units"].items())))}
    eff2 = AV.effects(shuffled, d, n_boot=200, seed=3)
    def norm(o):
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        return o
    assert json.dumps(norm(eff["per_arch"]), default=str) == json.dumps(norm(eff2["per_arch"]), default=str)
    # manual count of the bootstrap: with two replicates the resample is one of {(1,1),(1,2),(2,1),(2,2)}; each gives the
    # per-replicate contrast mean of the drawn ids -> the 95 % interval lies within [min, max] of the two per-replicate values
    per = eff["per_arch"]["A"]["COMMON"]["d"][1]["per_replicate"]
    lo, hi = eff["bootstrap"]["A"]["d1_common"]["ci95"]
    assert min(per.values()) - 1e-12 <= lo <= hi <= max(per.values()) + 1e-12 and eff["bootstrap"]["A"]["d1_common"]["n"] == 200
    assert "two observations" in eff["bootstrap_note"]


def test_RP18_refusals_foreign_design_duplicate_unexpected_arm_and_non_finite():
    d = _design()
    vals = _values(d)
    c = _closure(d, vals)
    bad = dict(c, design_sha256="0" * 64)
    with pytest.raises(AV.EffectsRefusal, match="not"):
        AV.effects(bad, d, n_boot=0)
    dup = dict(c, population={"members": c["population"]["members"] + [c["population"]["members"][0]]})
    with pytest.raises(AV.EffectsRefusal):
        AV.effects(dup, d, n_boot=0)
    stranger = dict(c, units={**c["units"], "H3__r1__s1__A__pooled_mean": {**c["units"]["H3__r1__s1__A__sequence"]}})
    with pytest.raises(AV.EffectsRefusal, match="unexpected"):
        AV.effects(stranger, d, n_boot=0)
    nan = _closure(d, {**vals, "H3__r1__s1__A__sequence": float("nan")})
    with pytest.raises(AV.EffectsRefusal, match="non-finite"):
        AV.effects(nan, d, n_boot=0)
    swapped = _closure(d, vals)
    swapped["units"]["H3__r1__s1__A__sequence"]["record"]["arm"] = "summary"                    # the record contradicts its id
    with pytest.raises(AV.EffectsRefusal, match="identity"):
        AV.effects(swapped, d, n_boot=0)
    d1 = AD.build(levels=[0, 3], replicates=[1], random_assignments=2, max_updates=6, readout_controls_r=[1], hosts=["COORDINATOR"])
    other = _closure(d1, _values(d1))
    with pytest.raises(AV.EffectsRefusal, match="population"):
        AV.effects(dict(other, design_sha256=d["design_sha256"]), d, n_boot=0)


def test_RP18_the_survivor_average_estimator_is_a_mutant_that_fails_the_pure_readout_oracle():
    """The old estimator averaged whatever fusion arms were present at each r; under a pure readout effect
    it reports a non-zero gamma (mixing the pair at r = 1 with the 2 x 2 average). The behaviour test
    catches it, for the expected reason: the contrast is not the same pair at both r."""
    d = _design()
    vals = _values(d, readout={0: -0.5, 1: -0.5})
    SEQ, SUM = ("sequence", "sequence_gap"), ("summary", "summary_last")

    def survivor_gamma(a):
        by = {}
        for c in d["cells"]:
            m = AV.parse_id(c["cell_id"])
            if c["arch"] == a and m["hypothesis"] == "H3" and c["arm"] not in ("extractor", "extractor_summary") and m["donor"] != "summary":
                by.setdefault((c["r"], c["seed"]), {})[c["arm"]] = vals[c["cell_id"]]
        dd = {}
        for r in (0, 1):
            diffs = [np.mean([v for f, v in arms.items() if f in SEQ]) - np.mean([v for f, v in arms.items() if f in SUM])
                     for (rr, s), arms in by.items() if rr == r]
            dd[r] = float(np.mean(diffs))
        return dd[1] - dd[0]
    mutant = survivor_gamma("A")
    intact = AV.effects(_closure(d, vals), d, n_boot=0)["per_arch"]["A"]["COMMON"]["gamma_common_pair"]["value"]
    assert intact == pytest.approx(0.0)
    assert mutant == pytest.approx(0.5) and mutant != pytest.approx(0.0)       # at r = 1 the 2 x 2 averages cancel the readout, at r = 0 they do not: +0.5 (the dictum's control)


def test_RP18_the_real_stage_closure_reproduces_the_dictum_gamma_of_the_common_pair():
    ev = HERE.parent / "docs" / "audits" / "evidence" / "d3_k5_20260917"
    close = json.loads((ev / "RP14_ARCH_STAGE_CLOSE_LOCAL.json").read_text())["local"]
    design = json.loads((ev / "RP14_ARCH_STAGE_DESIGN.json").read_text())
    eff = AV.effects(close, design, n_boot=0)
    expected = {"A": -0.0748581942, "B": -0.0677836455, "C": -0.0675279072, "0": -0.0548738519}
    for a, g in expected.items():
        assert eff["per_arch"][a]["COMMON"]["gamma_common_pair"]["value"] == pytest.approx(g, abs=1e-9)
        assert eff["per_arch"][a]["FACT"]["gamma_factorial"]["state"] == AV.NOT_ESTIMABLE
    assert eff["population"]["not_verified"] and all(k.startswith("DX__") for k in eff["population"]["not_verified"])
