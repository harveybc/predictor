#!/usr/bin/env python3
"""Rules for the MOD-CORE-PRETRAIN resolution and for the monitor/budget repair.

Two batteries, and both are written so that they FAIL against the code as it stood before this round:

    the monitor repair    `df_e1_phase1._fit` previously hard-coded ``monitor="val_loss"`` -- the arm's
                          own trained loss -- and took no `monitor` or `budget_match` argument at all.
                          Every rule in `TestTheMonitorRepair` either passes those keywords (a
                          TypeError on the old signature) or asserts a field the old record did not
                          carry (`monitor_is_arm_independent`, `budget_match`,
                          `budget_is_matched_by_construction`), so none of them can pass on the old
                          behaviour. The behavioural rule goes further and fits two arms, asserting
                          that they receive the SAME number of optimiser updates -- which the old loop
                          cannot do, because its stopping epoch follows the loss under test.

    the resolution        the resolution must be DERIVED from the retained arrays and never typed:
                          the rules recompute it, mutate a stored prediction and require the number to
                          move, and require the ruling to flip when sigma is made small enough. A
                          resolution that survives a mutated artifact is a constant with a formula
                          drawn around it.

The rules that need the run roots (not committed) are skipped when the roots are absent, and say so.
The rules over `docs/audits/evidence/` are portable: they hold on any checkout.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TOOLS = REPO / "tools"
EVID = REPO / "docs" / "audits" / "evidence" / "d3_k5_20260917"
PHASE1_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_phase1_v1b"
SUCCESSOR_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_household_successor_v3"


def _load(name: str):
    """Load a tool under a PRIVATE name and register it nowhere.

    The RP49-RP64 audit found a defect of its own making by publishing tools into `sys.modules`: two
    rules then picked up another battery's copy of a module and failed while passing in isolation. A
    battery that only passes when it runs alone is not a battery, so nothing here is registered.
    """
    spec = importlib.util.spec_from_file_location(f"_respec_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # for dataclasses/typing inside the module only
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


ROOTS_PRESENT = (PHASE1_ROOT / "DATA.npz").is_file() and (SUCCESSOR_ROOT / "DATA.npz").is_file()
needs_roots = pytest.mark.skipif(not ROOTS_PRESENT, reason="the E1 run roots are not on this host")


# =====================================================================================================
# the monitor repair
# =====================================================================================================

class TestTheMonitorRepair:

    def test_the_retained_phase1_design_is_refused_by_name(self):
        """The design that produced the confound declares an arm-dependent monitor and no budget rule.

        It must be REFUSED, not defaulted. This is the rule that makes the defect unreachable by
        omission: before this round the same design ran without a word.
        """
        P = _load("df_e1_phase1")
        design = json.loads((EVID / "RP63" / "PHASE1_DESIGN_SEALED.json").read_text())
        assert design["training"]["monitor"].startswith("each arm monitors the loss it trains on")
        assert "budget_match" not in design["training"]
        with pytest.raises(P.ProtocolRefusal, match="not one of the arm-independent monitors"):
            P.resolve_protocol(design)

    def test_the_legacy_protocol_is_reachable_only_by_naming_it(self):
        P = _load("df_e1_phase1")
        design = json.loads((EVID / "RP63" / "PHASE1_DESIGN_SEALED.json").read_text())
        design["training"]["budget_match"] = P.LEGACY_PROTOCOL
        got = P.resolve_protocol(design)
        assert got["monitor"] == P.LEGACY_MONITOR
        assert got["monitor_is_arm_independent"] is False
        assert got["budget_is_matched_by_construction"] is False
        assert "confounded" in got["defect"]

    def test_a_fixed_monitor_without_a_budget_rule_is_still_refused(self):
        """Fixing the monitor is only HALF the repair, and the post-RP63 factorial did only that half."""
        P = _load("df_e1_phase1")
        with pytest.raises(P.ProtocolRefusal, match="declares no budget rule"):
            P.resolve_protocol({"training": {"monitor": P.FIXED_MONITOR}})
        ok = P.resolve_protocol({"training": {"monitor": P.FIXED_MONITOR, "budget_match": P.BUDGET_FIXED}})
        assert ok["monitor_is_arm_independent"] and ok["budget_is_matched_by_construction"]

    def test_the_retained_rp63_run_is_refused_as_unmatched_with_its_own_numbers(self):
        """The 1 492-update spread, read off the retained report and named in the refusal."""
        P = _load("df_e1_phase1")
        rep = json.loads((EVID / "RP63" / "PHASE1_REPORT.json").read_text())
        cells = {k: v for k, v in rep["cells"].items() if not k.startswith("pilot")}
        audit = P.budget_audit(cells)
        assert audit["total_updates_by_arm"] == {"core_mae": 11762, "core_mse": 10270, "tcn_mse": 11762}
        assert audit["matched_on_totals"] is False
        assert audit["largest_total_minus_smallest"] == 1492
        with pytest.raises(P.ProtocolRefusal, match="not budget-matched"):
            P.require_budget_match(cells)

    def test_a_matched_set_is_accepted_and_one_changed_cell_is_enough_to_refuse_it(self):
        P = _load("df_e1_phase1")
        matched = {f"{arm}_s{s}": {"arm": arm, "updates": 4000}
                   for arm in ("core_mae", "core_mse", "tcn_mse") for s in (1, 2, 3)}
        got = P.require_budget_match(matched)
        assert got["matched_on_totals"] and got["matched_per_seed"]
        assert got["largest_total_minus_smallest"] == 0
        matched["core_mse_s3"]["updates"] = 3999          # one update, one cell
        with pytest.raises(P.ProtocolRefusal, match="not budget-matched"):
            P.require_budget_match(matched)

    def test_the_fit_requires_the_monitor_and_the_budget_rule_to_be_named(self):
        """No default. The old signature had no such argument, so this rule cannot pass on it."""
        import inspect
        P = _load("df_e1_phase1")
        sig = inspect.signature(P._fit)
        for name in ("monitor", "budget_match"):
            assert name in sig.parameters, f"_fit does not take {name}"
            assert sig.parameters[name].default is inspect.Parameter.empty, \
                f"{name} has a default; the defect this repairs WAS a default"

    def test_an_arm_dependent_monitor_is_refused_inside_the_fit_too(self):
        P = _load("df_e1_phase1")
        with pytest.raises(P.ProtocolRefusal, match="not an arm-independent monitor"):
            P._fit("core_mse", None, None, None, max_updates=1, patience=1, lr=1e-3, seed=1,
                   monitor="val_something", budget_match=P.BUDGET_FIXED)

    def test_the_sealed_identity_separates_the_two_protocols(self):
        """A run under the repaired protocol may not share a design digest with the defective one."""
        if not (SUCCESSOR_ROOT / "DESIGN.json").is_file():
            pytest.skip("the successor run root is not on this host")
        P = _load("df_e1_phase1")
        fixed = P.seal(SUCCESSOR_ROOT)
        legacy = P.seal(SUCCESSOR_ROOT, budget_match=P.LEGACY_PROTOCOL)
        assert fixed["training"]["monitor"] == P.FIXED_MONITOR
        assert fixed["training"]["budget_match"] == P.BUDGET_FIXED
        assert "monitor" in fixed["factors_held"] and "optimiser_updates" in fixed["factors_held"]
        assert "monitor" not in legacy["factors_held"]
        assert fixed["design_sha256"] != legacy["design_sha256"]
        E = _load("df_mod_e0")
        for d in (fixed, legacy):
            assert E.sha_obj({k: v for k, v in d.items() if k != "design_sha256"}) == d["design_sha256"]

    @needs_roots
    def test_two_arms_under_the_repair_get_the_same_updates_and_the_same_monitor(self, tmp_path):
        """The behavioural rule. Tiny synthetic data, a real fit, two different losses.

        Under the repair both arms run exactly the ceiling and both select on the same curve. The old
        loop cannot satisfy this: its stopping epoch is chosen by the loss under test, which is the
        whole defect.
        """
        P = _load("df_e1_phase1")
        PI = _load("df_e1_pilot")
        E = _load("df_mod_e0")
        tf = E._tf()
        rng = np.random.default_rng(7)
        n, W, h, p = 260, 8, 4, 2
        Xs = rng.normal(size=(n, p)).astype(np.float32)
        Y = np.cumsum(rng.normal(size=n)) * 0.1 + 1.0
        origins = np.arange(W, n - h - 1, dtype=np.int64)
        Dataset = PI._dataset_class()
        kw = dict(scaler_mean=np.array([0.0, 0.0]), scaler_sd=np.array([1.0, 1.0]))
        records = {}
        for arm in ("core_mse", "core_mae"):
            tr = Dataset(Xs, Y, origins[:180], W, h, 0, 32, shuffle=True, seed=1, **kw)
            va = Dataset(Xs, Y, origins[180:], W, h, 0, 32, shuffle=False, seed=1, **kw)
            model = tf.keras.Sequential([tf.keras.layers.Input((W, p)), tf.keras.layers.Flatten(),
                                         tf.keras.layers.Dense(4, activation="relu"),
                                         tf.keras.layers.Dense(1)])
            records[arm] = P._fit(arm, model, tr, va, max_updates=18, patience=1, lr=0.01, seed=1,
                                  monitor=P.FIXED_MONITOR, budget_match=P.BUDGET_FIXED,
                                  ckpt_dir=tmp_path / arm)
        a, b = records["core_mse"], records["core_mae"]
        assert a["loss_trained"] == "mse" and b["loss_trained"] == "mae"        # the recipe DID move
        assert a["monitor"] == b["monitor"] == P.FIXED_MONITOR                  # the monitor did not
        assert a["monitor_is_arm_independent"] and b["monitor_is_arm_independent"]
        assert a["updates"] == b["updates"] == 18                              # matched by construction
        # The accounting must be read BEFORE the checkpoint restore. Keras 3's `save_weights` carries
        # the optimizer's variables, so a restored argmin checkpoint rewinds `optimizer.iterations`;
        # read afterwards, two arms that ran the same budget and restored different epochs report
        # different counts and look mismatched when they are not. This rule pins the order.
        assert a["optimizer_iterations"] == b["optimizer_iterations"] == 18
        assert a["updates_are_optimizer_iterations"] and b["updates_are_optimizer_iterations"]
        assert a["budget_is_matched_by_construction"] and not a["monitor_chose_the_budget"]
        assert a["restored_checkpoint_chosen_on"] == b["restored_checkpoint_chosen_on"] == P.FIXED_MONITOR
        assert a["censoring"]["verdict"] == "BUDGET_MATCHED_BY_CONSTRUCTION"
        assert P.require_budget_match({"core_mse_s1": {"arm": "core_mse", "updates": a["updates"]},
                                       "core_mae_s1": {"arm": "core_mae", "updates": b["updates"]}})

    @needs_roots
    def test_under_the_legacy_protocol_the_monitor_follows_the_loss_and_is_named_as_such(self, tmp_path):
        """The old behaviour, reproduced ON PURPOSE, and now labelled by the record itself."""
        P = _load("df_e1_phase1")
        PI = _load("df_e1_pilot")
        E = _load("df_mod_e0")
        tf = E._tf()
        rng = np.random.default_rng(11)
        n, W, h, p = 200, 8, 4, 2
        Xs = rng.normal(size=(n, p)).astype(np.float32)
        Y = np.cumsum(rng.normal(size=n)) * 0.1 + 1.0
        origins = np.arange(W, n - h - 1, dtype=np.int64)
        Dataset = PI._dataset_class()
        kw = dict(scaler_mean=np.array([0.0, 0.0]), scaler_sd=np.array([1.0, 1.0]))
        tr = Dataset(Xs, Y, origins[:140], W, h, 0, 32, shuffle=True, seed=1, **kw)
        va = Dataset(Xs, Y, origins[140:], W, h, 0, 32, shuffle=False, seed=1, **kw)
        model = tf.keras.Sequential([tf.keras.layers.Input((W, p)), tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(4, activation="relu"),
                                     tf.keras.layers.Dense(1)])
        rec = P._fit("core_mse", model, tr, va, max_updates=10, patience=1, lr=0.01, seed=1,
                     monitor=P.LEGACY_MONITOR, budget_match=P.BUDGET_EARLY, ckpt_dir=tmp_path)
        assert rec["monitor"] == "val_loss"
        assert rec["monitor_is_arm_independent"] is False
        assert rec["monitor_chose_the_budget"] is True
        assert rec["budget_is_matched_by_construction"] is False


# =====================================================================================================
# the resolution
# =====================================================================================================

class TestTheResolutionEstimator:

    def test_the_mde_falls_with_n_and_rises_with_sigma(self):
        R = _load("df_core_pretrain_resolution")
        base = R.mde(0.01, 6, 3, paired=False)["mde_kW"]
        assert R.mde(0.01, 6, 10, paired=False)["mde_kW"] < base
        assert R.mde(0.02, 6, 3, paired=False)["mde_kW"] > base
        assert math.isclose(R.mde(0.02, 6, 3, paired=False)["mde_kW"], 2 * base, rel_tol=1e-12)

    def test_seeds_required_inverts_the_mde_and_is_tight(self):
        """The returned n must reach the effect and n-1 must NOT: a sample size that is merely
        sufficient is not an answer to 'how many would it take'."""
        R = _load("df_core_pretrain_resolution")
        sigma, effect = 0.011363459833619234, 0.00980498867418004
        got = R.seeds_required(sigma, effect, paired=False, arms_pooled=3)
        n = got["n_per_arm"]
        assert n is not None and n > 3
        assert R.mde(sigma, 3 * (n - 1), n, paired=False)["mde_kW"] <= effect
        assert R.mde(sigma, 3 * (n - 2), n - 1, paired=False)["mde_kW"] > effect

    def test_a_non_positive_effect_gets_no_sample_size(self):
        R = _load("df_core_pretrain_resolution")
        assert R.seeds_required(0.01, 0.0)["n_per_arm"] is None
        assert R.seeds_required(0.0, 0.01)["n_per_arm"] is None

    def test_the_sigma_interval_brackets_sigma_and_widens_as_df_falls(self):
        R = _load("df_core_pretrain_resolution")
        wide, narrow = R.sd_interval(0.01, 6), R.sd_interval(0.01, 60)
        for iv in (wide, narrow):
            assert iv["lo"] < iv["sd"] < iv["hi"]
        assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])

    def test_the_estimator_band_quotes_its_most_favourable_member(self):
        """A ruling of 'below the resolution' must not rest on a pessimistic test."""
        R = _load("df_core_pretrain_resolution")
        band = R.estimator_band(0.0114, 6, 3)
        members = (band["paired_kW"], band["unpaired_kW"], band["unpaired_with_sigma_at_pooled_df_kW"])
        assert band["governing_kW"] == min(members)
        assert band["paired_kW"] > band["unpaired_kW"]        # pairing costs df at n=3


@needs_roots
class TestTheResolutionIsDerivedFromArtifacts:

    @pytest.fixture(scope="class")
    def doc(self):
        return _load("df_core_pretrain_resolution").build()

    def test_it_reports_no_problems_and_performed_no_training(self, doc):
        assert doc["problems"] == []
        assert doc["training_performed"] is False

    def test_every_arm_mean_is_the_float64_recomputation_and_the_records_agree(self, doc):
        for cid, c in doc["cells"].items():
            assert c["record_agrees"], f"{cid}: the record's score is not the recomputation"
        for arm, mean in doc["arm_means_kW"].items():
            cells = [c for c in doc["cells"].values() if c["arm"] == arm]
            assert math.isclose(mean, float(np.mean([c["mae_kW"] for c in cells])), rel_tol=0, abs_tol=1e-15)

    def test_the_naive_is_on_the_same_rows_as_the_model_in_every_row(self, doc):
        ns = {round(v["naive_kW"], 12) for v in doc["naive_same_rows"].values()}
        rows = {v["n_rows"] for v in doc["naive_same_rows"].values()}
        assert len(ns) == 1 and len(rows) == 1
        assert rows == {c["n_rows"] for c in doc["cells"].values()}

    def test_the_cross_runner_replicate_is_one_configuration_not_two(self, doc):
        """core_mse IS R0. If these were pooled as two arms the instrument would look twice as good
        as it is, so the rule pins both that they agree and that they are counted once."""
        rep = doc["cross_runner_replicate"]
        assert rep["updates_match_per_seed"] is True
        assert rep["max_absolute_difference_kW"] < 1e-5          # the repository's replay tolerance
        governing = doc["resolution"]["governing_arms"]
        assert "core_mse" not in governing or "R0" not in governing

    def test_the_governing_sigma_pools_only_arms_that_pass_a_variance_test(self, doc):
        pooled = doc["pooled_within_arm_sd"]
        assert pooled["the_modules_own_three_arms"]["homoscedasticity_bartlett"]["p_value"] > 0.05
        assert pooled["five_independent_arms"]["homoscedasticity_bartlett"]["p_value"] < 0.05
        assert doc["resolution"]["sigma_kW"] == pooled["the_modules_own_three_arms"]["pooled_sd"]
        assert doc["resolution"]["sigma_df"] == 6

    def test_the_scrambled_control_gap_reproduces_against_a_recomputed_partner(self, doc):
        c = doc["scrambled_label_control"]
        assert c["declared_gap_reproduces"] is True
        assert c["partner_is_budget_matched"] is False           # the control's own asymmetry, named
        assert c["custody"] == "READ_FROM_RETAINED_JSON_WEIGHTS_NOT_KEPT"

    def test_the_ruling_is_computed_and_flips_when_sigma_would_allow_it(self, doc):
        """The verdict must be a function of the measurements. Shrink sigma by a factor the
        measurements do not support and the same code must return the other answer."""
        R = _load("df_core_pretrain_resolution")
        ru = doc["ruling"]
        assert ru["verdict"] == "UNANSWERABLE_BY_THIS_PROTOCOL_AT_THIS_SEED_COUNT"
        assert ru["effect_is_above_the_resolution"] is False
        assert ru["resolution_kW"] > ru["effect_the_module_must_resolve_kW"]
        tiny = R.estimator_band(doc["resolution"]["sigma_kW"] / 20, 6, 3)["governing_kW"]
        assert tiny < ru["effect_the_module_must_resolve_kW"]

    def test_every_module_effect_is_below_the_resolution(self, doc):
        states = doc["ruling"]["every_module_effect_state"]
        assert set(states) == {"R1-R0", "R2-R0", "R2-R1"}
        assert all(v == "BELOW_THE_RESOLUTION" for v in states.values())

    def test_a_mutated_prediction_moves_the_resolution(self, tmp_path, doc):
        """The number is generated from artifacts. Change one stored prediction in a copy of one cell
        and the recomputed arm mean must move; a resolution that survives this is a typed constant."""
        R = _load("df_core_pretrain_resolution")
        src = SUCCESSOR_ROOT / "attempts" / "R1_s1"
        dst = tmp_path / "attempts" / "R1_s1"
        dst.mkdir(parents=True)
        (dst / "cell.json").write_text((src / "cell.json").read_text())
        with np.load(src / "arrays.npz", allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        before = R.cell_measurement(SUCCESSOR_ROOT, "R1_s1")["mae_kW"]
        arrs["validation_pred"] = arrs["validation_pred"] + np.float32(0.5)
        np.savez(dst / "arrays.npz", **arrs)
        after = R.cell_measurement(tmp_path, "R1_s1")
        assert abs(after["mae_kW"] - before) > 0.1
        assert after["record_agrees"] is False              # and the disagreement is REPORTED

    def test_a_label_that_is_not_Y_at_the_horizon_gets_no_naive(self, tmp_path):
        """The naive is refused rather than computed on a population it does not belong to."""
        R = _load("df_core_pretrain_resolution")
        dst = tmp_path / "attempts" / "R1_s1"
        dst.mkdir(parents=True)
        (tmp_path / "DATA.npz").write_bytes((SUCCESSOR_ROOT / "DATA.npz").read_bytes())
        with np.load(SUCCESSOR_ROOT / "attempts" / "R1_s1" / "arrays.npz", allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        arrs["validation_y"] = arrs["validation_y"] + 1.0
        np.savez(dst / "arrays.npz", **arrs)
        with pytest.raises(ValueError, match="not Y\\[origins\\+h\\]"):
            R.naive_on_the_same_rows(tmp_path, "R1_s1")


# =====================================================================================================
# the matched re-contrast, and the closure table over a root nothing governs
# =====================================================================================================

MATCHED_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_phase1_matched_v1"
needs_matched = pytest.mark.skipif(not (MATCHED_ROOT / "DESIGN.json").is_file(),
                                   reason="the matched re-contrast root is not on this host")


@needs_matched
class TestTheMatchedReContrast:

    @pytest.fixture(scope="class")
    def rep(self):
        return _load("df_core_pretrain_matched").report(MATCHED_ROOT)

    def test_every_arm_ran_the_same_number_of_updates(self, rep):
        a = rep["budget_audit"]
        assert a["matched_on_totals"] and a["matched_per_seed"]
        assert a["largest_total_minus_smallest"] == 0
        assert len(set(a["total_updates_by_arm"].values())) == 1

    def test_one_monitor_across_every_arm(self, rep):
        assert rep["monitor"] == "val_mae"
        assert rep["budget_match"] == "FIXED_UPDATES"
        assert {c["monitor"] for c in rep["cells"].values()} == {"val_mae"}

    def test_every_cell_replays_from_its_own_weights_in_a_fresh_graph(self, rep):
        assert rep["all_replays_within_tolerance"] is True
        for cid, r in rep["replay"].items():
            assert r["fresh_graph"] and r["rows"] == rep["evaluation_rows"], cid
            assert r["max_abs_error"] <= r["tolerance"], cid

    def test_the_arm_means_are_the_recomputation_and_the_naive_is_on_the_same_rows(self, rep):
        for arm, mean in rep["arm_means_kW"].items():
            assert math.isclose(mean, float(np.mean(rep["arm_mae_by_seed_kW"][arm])), abs_tol=1e-15)
        assert rep["evaluation_rows"] == 10020
        assert rep["naive_same_rows_kW"] > 0

    def test_the_report_refuses_an_unmatched_set_even_here(self):
        """The rule lives in the reporter, not only in the runner: a root whose arms drifted cannot be
        reported as a contrast at all."""
        P = _load("df_e1_phase1")
        with pytest.raises(P.ProtocolRefusal, match="not budget-matched"):
            P.require_budget_match({"core_mae_s1": {"arm": "core_mae", "updates": 4000},
                                    "core_mse_s1": {"arm": "core_mse", "updates": 3762}})


@needs_matched
class TestTheClosureTableOverAnUngovernedRoot:
    """A root with arrays and no terminal used to raise FileNotFoundError and produce NO table.

    A crash reports neither the measurement nor the missing custody. The repair produces the scored
    row, states that nothing anchors it, and forbids it from reading `verified` -- which is the tool's
    own stated rule ('a registered forecast unit without an accepted terminal is a PROBLEM, not an
    absence') applied to the case that used to bypass it.
    """

    @pytest.fixture(scope="class")
    def table(self):
        T = _load("df_closure_table")
        B = _load("df_benchmark_contract")
        return T.build([f"{MATCHED_ROOT}:e1_phase1_matched_v1"], registry=B.registry(),
                       warehouse=None, no_new_measurement=False)

    def test_it_produces_a_row_per_unit_instead_of_raising(self, table):
        rows = [r for r in table["rows"] if r["run"] == "e1_phase1_matched_v1"]
        assert len(rows) == 9

    def test_every_row_carries_its_number_its_naive_and_its_skill(self, table):
        for r in table["rows"]:
            assert isinstance(r["model_error"], float) and r["model_error"] > 0
            assert isinstance(r["naive_error"], float) and r["naive_error"] > 0
            assert r["model_population"] == r["naive_population"] == 10020
            assert r["model_horizon"] == r["naive_horizon"]
            assert r["model_scale"] == r["naive_scale"] == "kW"
            assert math.isclose(r["skill_vs_naive"]["value"], 1 - r["model_error"]/r["naive_error"], abs_tol=1e-12)
            assert r["comparability_status"] == "NOT_COMPARABLE"
            assert r["literature_value_and_source"]["why_not"]
            assert r["literature_value_and_source"]["placed_in_comparison_column"] is False

    def test_nothing_in_an_ungoverned_root_may_read_verified(self, table):
        for r in table["rows"]:
            assert r["verified"] is False
            assert r["preserved_with_qualified_scope"] is False
            assert r["custody"]["class"] == "UNANCHORED_NO_TERMINAL"
            assert any("NO accepted terminal receipt" in p for p in r["problems"])
        assert table["verified_rows"] == 0
        assert len(table["problems"]) >= 9
