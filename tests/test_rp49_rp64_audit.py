"""The audit of RP49-RP56 and RP57-RP64 as executable rules.

Written under the owner's grant of 2026-09-26, standing in place of the absent
``MUSASHI_RP49_RP56_REVIEW`` and ``MUSASHI_RP57_RP64_REVIEW``. These rules are the part of the
audit that outlives its document: every number below is RE-DERIVED from the bytes the two rounds
retained, and every assertion is about those bytes, never about a sentence. A later correction to
the prose changes nothing here; a change to the artifacts breaks a rule by name.

Three kinds of rule:

  recomputation   the audit's own arithmetic over the retained artifacts, asserted against what
                  the returns published (``tools/df_rp49_rp64_audit.py``)
  counterexample  the eight measurements that do NOT sustain their sentence, pinned so they cannot
                  be lost again: an off-by-one population, an off-by-one count of censored cells,
                  a figure quoted against a ceiling it is not charged to, an "exactly" that is
                  5e-07, a family quoted at the top of its own range, a signature that subtracts
                  across two row populations, a control with no matched arm, and a verification
                  that covers 5% of the rows its metric uses
  refusal         malformed cases built here, each refused BY NAME by the contract the rounds
                  touched, plus mutations of the retained artifacts that the audit must catch

Nothing here fits, trains or replays a model, and nothing contacts governance.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str, where: Path = TOOLS):
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("df_rp49_rp64_audit")
VERIFIED, REFUTED, ABSENT = A.VERIFIED, A.REFUTED, A.ABSENT

needs_run_root = pytest.mark.skipif(not (A.SUCCESSOR_ROOT / "DATA.npz").is_file(),
                                    reason="the successor run root is not on this host")


def _by_claim(rows, needle):
    hits = [r for r in rows if needle in r["claim"]]
    assert hits, f"no check matching {needle!r}"
    return hits[0]


# --- 1. identity, re-derived from bytes -------------------------------------------------------------

def test_the_sealed_successor_design_re_derives_its_own_digest():
    rows = A.identity_successor()
    c = _by_claim(rows, "re-derives its own digest")
    assert c["state"] == VERIFIED
    assert c["recomputed"] == c["declared"] == A.SUCCESSOR_DESIGN_SHA


@needs_run_root
def test_the_design_that_ran_is_byte_for_byte_the_design_that_was_sealed():
    c = _by_claim(A.identity_successor(), "IS the design that was sealed")
    assert c["state"] == VERIFIED
    assert c["sealed_file_sha256"] == c["run_file_sha256"]
    assert c["objects_identical"] is True


def test_the_phase1_design_re_derives_and_the_phase_ran_on_the_successors_own_rows():
    rows = A.phase1()
    assert _by_claim(rows, "sealed before running")["state"] == VERIFIED
    assert _by_claim(rows, "successor run's own rows")["state"] == VERIFIED


def test_an_edited_design_no_longer_re_derives(tmp_path):
    src = A.EVID / "RP38" / "E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json"
    doc = json.loads(src.read_text())
    doc["training"] = {**doc.get("training", {}), "batch": 999}
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(doc, indent=1))
    got = A.rederive_design(edited)
    assert got["re_derives"] is False
    assert got["declared"] == A.SUCCESSOR_DESIGN_SHA
    assert got["recomputed"] != A.SUCCESSOR_DESIGN_SHA


def test_an_edited_design_with_its_declaration_repaired_is_still_not_the_sealed_one(tmp_path):
    """The obvious attack: change the design AND its declared digest. It re-derives, and it is
    still not the identity the run and the closure bind to."""
    src = A.EVID / "RP38" / "E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json"
    doc = json.loads(src.read_text())
    doc["training"] = {**doc.get("training", {}), "batch": 999}
    doc["design_sha256"] = A.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(doc, indent=1))
    got = A.rederive_design(edited)
    assert got["re_derives"] is True
    assert got["declared"] != A.SUCCESSOR_DESIGN_SHA


# --- 2. the population ------------------------------------------------------------------------------

def test_the_closure_population_is_fifteen_units_none_absent_no_stranger():
    rows = A.population_successor()
    assert _by_claim(rows, "sealed population is 15 units")["state"] == VERIFIED
    assert _by_claim(rows, "15 units verified")["state"] == VERIFIED


def test_verified_plus_not_applicable_equals_declared_per_fact():
    c = _by_claim(A.population_successor(), "verified + NOT_APPLICABLE = declared")
    assert c["state"] == VERIFIED
    assert sorted(c["inference_not_applicable"]) == ["ae_s1", "ae_s2", "ae_s3", "pilot_ae"]
    assert c["regime_not_applicable"] == ["controls"]


def test_the_sixteenth_governed_unit_is_prepare_and_is_not_a_missing_cell():
    """COUNTEREXAMPLE 1. '16 / 16' beside '15 / 15' invites the reading that one cell did not
    verify. The sixteenth member of the governed population is the unit 'prepare', and the return
    names it nowhere — which is the one thing RP51 says must never happen to a registered unit."""
    c = _by_claim(A.population_successor(), "which the return never names")
    assert c["state"] == VERIFIED
    assert c["warehouse_population"] == 16
    assert c["the_sixteenth"] == ["prepare"]


# --- 3. the numbers ---------------------------------------------------------------------------------

def test_every_published_scaled_error_is_its_own_mae_over_the_runs_own_denominator():
    c = _by_claim(A.numbers_successor(), "over the run's own denominator")
    assert c["state"] == VERIFIED and c["cells"] == 9


@pytest.mark.parametrize("regime", ["R0", "R1", "R2"])
def test_the_regime_means_are_the_means_of_their_own_cells(regime):
    c = _by_claim(A.numbers_successor(), f"{regime} mean ")
    assert c["state"] == VERIFIED
    assert abs(c["recomputed_mean"] - c["published_mean"]) < 1e-12
    assert abs(c["recomputed_sd"] - c["published_sd"]) < 1e-12


@pytest.mark.parametrize("pair", ["R1_minus_R0", "R2_minus_R0", "R2_minus_R1"])
def test_the_paired_differences_are_the_per_seed_differences(pair):
    assert _by_claim(A.numbers_successor(), f"paired {pair}")["state"] == VERIFIED


def test_the_controls_and_the_censoring_and_the_restored_checkpoints():
    rows = A.numbers_successor()
    for name in ("linear_ridge", "persistence", "seasonal_naive_daily"):
        assert _by_claim(rows, f"control {name}")["state"] == VERIFIED
    c = _by_claim(rows, "CENSORED_BY_BUDGET")
    assert c["state"] == VERIFIED and len(c["censored"]) == 4 and c["at_the_ceiling"] is True
    assert _by_claim(rows, "argmin of its own curve")["state"] == VERIFIED


def test_the_cpu_figure_quoted_against_the_cap_is_not_the_one_charged_to_it():
    """COUNTEREXAMPLE 2. '2 535.5 CPU seconds of an 11 000-second cap' is the root-only total; the
    report charges root + already_spent = 2 785.504 to that same cap. Both are far inside it."""
    rows = A.numbers_successor()
    assert _by_claim(rows, "sum of its own components")["state"] == VERIFIED
    c = _by_claim(rows, "of an 11 000-second cap")
    assert c["state"] == REFUTED
    assert c["quoted_against_the_cap"] == pytest.approx(2535.504, abs=1e-3)
    assert c["the_runner_charges_to_the_cap"] == pytest.approx(2785.504, abs=1e-3)
    assert c["already_spent_seconds"] == pytest.approx(250.0)


# --- 4. the verification's own coverage --------------------------------------------------------------

def test_the_retained_closure_replay_covers_five_percent_of_the_metrics_rows():
    """COUNTEREXAMPLE 3. The closure's fresh-process replay is real, and it covers the FIRST 512 of
    10 020 evaluation origins. The published MAE is computed on all of them."""
    c = _by_claim(A.replay_successor(), "covers the rows the published metric is computed on")
    assert c["state"] == REFUTED
    assert c["replayed_windows_per_unit"] == 512 and c["metric_rows"] == 10020
    assert c["coverage_percent"] == pytest.approx(5.11, abs=0.01)


# --- 5. phase 1 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["core_mae", "tcn_mse", "core_mse"])
def test_the_phase1_arm_means_re_derive_from_the_retained_report(arm):
    assert _by_claim(A.phase1(), f"{arm} MAE")["state"] == VERIFIED


def test_the_reference_block_is_parameter_matched_and_the_cost_adds_up():
    rows = A.phase1()
    assert _by_claim(rows, "parameter-matched")["state"] == VERIFIED
    assert _by_claim(rows, "2 024.5 CPU seconds")["state"] == VERIFIED


def test_phase1_has_five_censored_cells_and_the_return_says_four():
    """COUNTEREXAMPLE 4, the off-by-one. Five of the nine phase-1 cells carry
    CENSORED_BY_BUDGET in the retained report; four is the successor run's count."""
    c = _by_claim(A.phase1(), "Four of nine cells are CENSORED_BY_BUDGET")
    assert c["state"] == REFUTED
    assert c["measured"] == 5
    assert c["censored"] == ["core_mae_s1", "core_mae_s2", "core_mse_s2", "tcn_mse_s2", "tcn_mse_s3"]


def test_the_winning_recipe_arm_also_got_a_bigger_training_budget():
    """COUNTEREXAMPLE 5. Changing the monitor with the loss changed when early stopping fired, so
    the arm that wins by 0.049 kW ran 11 762 optimiser updates against the arm it beats at 10 270."""
    c = _by_claim(A.phase1(), "trained to the same budget")
    assert c["state"] == REFUTED
    assert c["updates_by_arm"]["core_mae"] == 11762
    assert c["updates_by_arm"]["core_mse"] == 10270


def test_the_cross_runner_replication_is_five_times_ten_to_the_minus_seven_not_exact():
    """COUNTEREXAMPLE 6. 'reproducing R0_s1 exactly' is a mean agreement of 5.0e-07 kW; the
    per-row predictions differ by up to 6.5e-05, ABOVE this repository's own 1e-05 replay
    tolerance. The replication is real. 'Exactly' is not."""
    c = _by_claim(A.phase1(), "through a different runner")
    assert c["state"] == REFUTED
    assert 0 < c["difference"] < 1e-6


# --- 6. the reanalysis -------------------------------------------------------------------------------

def test_the_192_metric_rows_recount_under_the_publishers_own_rule():
    rows = A.reanalysis()
    c = _by_claim(rows, "192 metric rows")
    assert c["state"] == VERIFIED and c["recounted_under_the_publisher_s_own_rule"] == 192


def test_the_published_rows_bind_to_a_report_that_is_retained():
    c = _by_claim(A.reanalysis(), "bind to the report whose digest they carry")
    assert c["state"] == VERIFIED
    assert c["report_sha256"] == c["retained_file_sha256"]


def test_the_reanalysis_agrees_with_the_runs_own_results_for_all_nine_cells():
    assert _by_claim(A.reanalysis(), "identical for all nine cells")["state"] == VERIFIED


def test_a_zero_error_reference_yields_no_skill_and_no_epsilon_is_invented():
    """The rule the real rows never exercise, exercised here: a reference with zero error must
    produce no skill row at all, not an infinity and not a manufactured win."""
    rep = json.loads((A.EVID / "RP57" / "COMPARISON_REPORT.json").read_text())
    row = copy.deepcopy(rep["by_scale"]["raw_kW"]["R0_s1"])
    row["versus"]["perfect_oracle"] = {"mae_skill_percent": None, "status": "UNDEFINED_ZERO_ERROR_REFERENCE"}
    published = sum(1 for _r, a in row["versus"].items() if a.get("mae_skill_percent") is not None)
    assert published == len(row["versus"]) - 1


@needs_run_root
def test_the_three_scaled_error_denominators_re_derive_on_their_stated_supports():
    rows = A.denominators()
    assert _by_claim(rows, "over the HORIZON on the train origins")["state"] == VERIFIED
    c = _by_claim(rows, "0.0851237797 over 40 257 finite pairs")
    assert c["state"] == VERIFIED and c["finite_pairs"] == 40257
    m = _by_claim(rows, "m = 1 and m = 1440")
    assert m["state"] == VERIFIED and m["m1440_pairs"] == 38818


# --- 7. the legacy tables ----------------------------------------------------------------------------

def test_the_legacy_inventory_counts_re_derive_from_its_own_entries():
    rows = A.legacy()
    assert _by_claim(rows, "tables inventoried 167")["state"] == VERIFIED
    assert _by_claim(rows, "CAUSALITY_UNVERIFIED 114")["state"] == VERIFIED
    assert _by_claim(rows, "one published table ever carried a naive")["state"] == VERIFIED


def test_the_eighty_five_flagged_tables_re_derive_and_the_base_is_one_hundred_and_thirteen():
    """COUNTEREXAMPLE 7a. The flag set re-derives exactly by the artifact's own >=50% rule. The
    denominator does not: 54 of the 167 tables carry no reconstructable predictions and were never
    tested, so the rate is 85 of 113."""
    c = _by_claim(A.legacy(), "85 of 167 published legacy tables")
    assert c["state"] == VERIFIED
    assert c["flagged"] == c["re_derived_by_the_artifact_s_own_rule"] == 85
    assert c["measurable_base"] == 113 and c["unmeasurable"] == 54


def test_the_phase_3_family_is_not_at_ninety_six_percent_only_its_top_four_are():
    """COUNTEREXAMPLE 7b."""
    c = _by_claim(A.legacy(), "phase-3 family by 96-97%")
    assert c["state"] == REFUTED
    assert c["phase_3_tables_measured"] == 24
    assert len(c["at_or_above_96"]) == 4
    assert c["best_horizon_skill_range"][0] < 30.0


def test_the_legacy_skills_recompute_from_the_committed_predictions_on_identical_rows():
    rows = A.legacy_recomputed()
    assert rows, "no legacy table was recomputed"
    for r in rows:
        assert r["state"] == VERIFIED, r
        for h, v in r["horizons"].items():
            assert v["n"] == 5984, (r["claim"], h)


def test_a_skill_whose_model_and_naive_do_not_share_rows_is_caught(tmp_path, monkeypatch):
    """The refusal built by hand. The legacy recomputation is only meaningful because model and
    naive are read from the same rows of the same file. Drop one row from the prediction file and
    the recomputation no longer reproduces the artifact's own n, mae, naive or skill: the check
    turns REFUTED instead of quietly agreeing on a different population."""
    key = "examples/results/phase_3_3/phase_3_2_cnn_25200_1h_results.csv"
    inv = json.loads((A.EVID / "RP58" / "LEGACY_INVENTORY.json").read_text())
    pred = REPO / inv["entries"][key]["prediction_file"]
    lines = pred.read_text().splitlines()

    dst = _evidence_copy(tmp_path)
    short_rel = "examples/results/_audit_short_prediction.csv"
    short = REPO / short_rel
    try:
        short.write_text("\n".join([lines[0]] + lines[1:-1]) + "\n")
        doc = json.loads((dst / "RP58" / "LEGACY_INVENTORY.json").read_text())
        doc["entries"][key]["prediction_file"] = short_rel
        (dst / "RP58" / "LEGACY_INVENTORY.json").write_text(json.dumps(doc))
        monkeypatch.setattr(A, "EVID", dst)
        row = A.legacy_recomputed([key])[0]
        assert row["state"] == REFUTED
        assert row["horizons"]["H1"]["n"] == 5983
        assert row["horizons"]["H1"]["n"] != inv["entries"][key]["recomputed"]["horizons"]["H1"]["n"]
    finally:
        short.unlink(missing_ok=True)


def test_the_leak_signature_subtracts_across_two_row_populations():
    """COUNTEREXAMPLE 8. The causal trailing wavelet is scored on 1 512 rows, the leaking
    representations on 1 525, and the 'signature' field subtracts the two."""
    rows = A.leak_signature()
    assert _by_claim(rows, "re-derives from its own MAE")["state"] == VERIFIED
    c = _by_claim(rows, "measured on the same rows")
    assert c["state"] == REFUTED
    assert c["evaluation_rows"]["TRAILING_WAVELET"] == 1512
    assert c["evaluation_rows"]["CENTRED_SMOOTHER"] == 1525


# --- 8. the data audit and the optimisation probe -----------------------------------------------------

def test_the_grid_the_label_and_the_scaler_grain():
    rows = A.data_audit()
    assert _by_claim(rows, "perfect one-minute grid")["state"] == VERIFIED
    assert _by_claim(rows, "equals the panel column element by element")["state"] == VERIFIED
    assert _by_claim(rows, "centred on the window grain")["state"] == VERIFIED


@needs_run_root
def test_the_autocorrelations_re_derive_under_a_biased_estimator_whose_bias_grows_with_the_lag():
    """The four published autocorrelations re-derive EXACTLY, under an ACF with a fixed
    whole-series denominator. The day-below-hour reading survives the bias correction; the
    published list's apparent decay to the weekly lag does not."""
    c = _by_claim(A.data_audit(), "0.403 at sixty")
    assert c["state"] == VERIFIED
    assert c["train_rows_used"] == 40199
    assert c["bias_factor_n_minus_k_over_n"]["10080"] < 0.76
    assert c["lag_truncated_pearson"]["1440"] < c["lag_truncated_pearson"]["60"]
    assert c["lag_truncated_pearson"]["10080"] > c["lag_truncated_pearson"]["1440"]
    assert c["published"]["10080"] < c["published"]["1440"]


def test_the_route_the_capacity_and_the_update_counter():
    rows = A.optimisation()
    assert _by_claim(rows, "reproduces the persistence control")["state"] == VERIFIED
    assert _by_claim(rows, "256-window subset")["state"] == VERIFIED
    assert _by_claim(rows, "400 = optimizer.iterations")["state"] == VERIFIED


def test_the_full_scale_negative_control_has_no_matched_true_label_arm():
    """COUNTEREXAMPLE 9. 0.6015 against 0.5525 is measured, and the arms differ in budget and in
    selection point as well as in labels."""
    rows = A.optimisation()
    assert _by_claim(rows, "0.6015 scrambled against 0.5525 true")["state"] == VERIFIED
    c = _by_claim(rows, "matched true-label arm")
    assert c["state"] == REFUTED
    assert c["scrambled_updates"] == 4000 and c["partner_updates"] == 3762
    assert c["scrambled_restored_epoch"] != c["partner_best_epoch"]


def test_the_guards_and_the_suites_are_the_ones_the_returns_report():
    rows = A.mutants_and_controls()
    for needle in ("5 killed of 5", "all DETECTED", "RP49-RP56 full suite", "RP57-RP64 full suite"):
        assert _by_claim(rows, needle)["state"] == VERIFIED


# --- 9. mutations of the retained artifacts the audit must catch ---------------------------------------

def _evidence_copy(tmp_path) -> Path:
    dst = tmp_path / "evidence"
    dst.mkdir()
    for sub in ("RP38", "RP55", "RP57", "RP58", "RP59", "RP60", "RP63"):
        shutil.copytree(A.EVID / sub, dst / sub)
    for f in ("RP56_WAREHOUSE_CONTENT_CHECK.json", "RP56_MUTANTS_POST.json",
              "RP56_FULL_SUITE_SUMMARY.txt", "RP64_FULL_SUITE_SUMMARY.txt"):
        shutil.copy2(A.EVID / f, dst / f)
    return dst


def test_a_regime_mean_that_drifts_from_its_own_cells_is_refused(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    p = dst / "RP55" / "E1_SUCCESSOR_RESULTS.json"
    doc = json.loads(p.read_text())
    doc["means"]["R0"]["mase_mean"] = doc["means"]["R0"]["mase_mean"] + 0.01
    p.write_text(json.dumps(doc, indent=1))
    monkeypatch.setattr(A, "EVID", dst)
    assert _by_claim(A.numbers_successor(), "R0 mean ")["state"] == REFUTED


def test_a_cell_whose_scaled_error_is_not_its_own_mae_over_the_denominator_is_refused(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    p = dst / "RP55" / "E1_SUCCESSOR_RESULTS.json"
    doc = json.loads(p.read_text())
    doc["cells"]["R0_s1"]["mase"] = doc["cells"]["R0_s1"]["mase"] * 0.9
    p.write_text(json.dumps(doc, indent=1))
    monkeypatch.setattr(A, "EVID", dst)
    assert _by_claim(A.numbers_successor(), "over the run's own denominator")["state"] == REFUTED


def test_an_absent_unit_in_the_closure_population_is_refused(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    p = dst / "RP55" / "E1_SUCCESSOR_CLOSE.json"
    doc = json.loads(p.read_text())
    doc["population"]["absent_ids"] = ["R2_s3"]
    p.write_text(json.dumps(doc, indent=1))
    monkeypatch.setattr(A, "EVID", dst)
    assert _by_claim(A.population_successor(), "sealed population is 15 units")["state"] == REFUTED


def test_a_flag_set_that_no_longer_re_derives_from_its_own_skills_is_refused(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    p = dst / "RP58" / "CAUSAL_LINEAGE.json"
    doc = json.loads(p.read_text())
    key = next(k for k, v in doc["entries"].items()
               if v["anomaly_indicator"].get("measured") and not v["anomaly_indicator"]["flagged_as_extraordinary"])
    doc["entries"][key]["anomaly_indicator"]["flagged_as_extraordinary"] = True
    p.write_text(json.dumps(doc))
    monkeypatch.setattr(A, "EVID", dst)
    assert _by_claim(A.legacy(), "85 of 167 published legacy tables")["state"] == REFUTED


def test_a_metric_count_that_contradicts_the_report_is_refused(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    p = dst / "RP57" / "REANALYSIS_PUBLICATION.json"
    doc = json.loads(p.read_text())
    doc["metrics_published"] = 191
    p.write_text(json.dumps(doc, indent=1))
    monkeypatch.setattr(A, "EVID", dst)
    assert _by_claim(A.reanalysis(), "192 metric rows")["state"] == REFUTED


def test_a_report_whose_digest_no_longer_binds_is_refused(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    p = dst / "RP57" / "COMPARISON_REPORT.json"
    doc = json.loads(p.read_text())
    doc["n"] = doc["n"]
    p.write_text(json.dumps(doc, indent=1) + "\n")        # same content, different bytes
    monkeypatch.setattr(A, "EVID", dst)
    assert _by_claim(A.reanalysis(), "bind to the report whose digest they carry")["state"] == REFUTED


def test_an_absent_artifact_is_unverifiable_and_never_accepted(tmp_path, monkeypatch):
    dst = _evidence_copy(tmp_path)
    (dst / "RP63" / "PHASE1_REPORT.json").unlink()
    monkeypatch.setattr(A, "EVID", dst)
    rows = A.phase1()
    assert len(rows) == 1 and rows[0]["state"] == ABSENT


# --- 10. refusals of the contracts the two rounds touched ----------------------------------------------

def test_the_strict_normalization_contract_refuses_an_undeclared_space():
    """RP58's repair on an active path: the prediction space is a declaration. Under
    strict_normalization_contract an undeclared one is refused instead of guessed."""
    from pipeline_plugins import stl_norm
    with pytest.raises(ValueError, match="does not declare prediction_space"):
        stl_norm._space_or_refuse({"strict_normalization_contract": True})
    assert stl_norm._space_or_refuse({"prediction_space": "REAL"}) == "REAL"
    assert stl_norm._space_or_refuse({}) is None
    with pytest.raises(ValueError, match="prediction_space must be one of"):
        stl_norm._space_or_refuse({"prediction_space": "MAYBE"})


def _delivery_root(tmp_path, *, design_sha, unit="prepare", payload=b"panel-bytes", digest=None):
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    panel = root / "panel.parquet"
    panel.write_bytes(payload)
    import hashlib
    sha = digest or hashlib.sha256(payload).hexdigest()
    (root / "DELIVERIES.json").write_text(json.dumps({
        "design_sha256": design_sha, "lake": "public_panels", "resource": "panel.parquet",
        "units": {unit: {"campaign_key": "k", "campaign_sha256": "c" * 64, "delivery_id": "d" * 32,
                         "sha256": sha, "at": "2026-09-20T18:00:00Z", "host": "omega",
                         "code_identity": {"kind": "git_commit", "value": "f" * 40},
                         "path": str(panel), "cached": True,
                         "verification_state": "VERIFIED_CACHE"}}}))
    return root


def test_the_governed_delivery_refuses_another_design_and_changed_bytes_by_name(tmp_path):
    G = _load("df_e1_governed")
    design = {"design_sha256": A.SUCCESSOR_DESIGN_SHA}
    ok = _delivery_root(tmp_path / "a", design_sha=A.SUCCESSOR_DESIGN_SHA)
    assert G.require_delivery(ok, design)["campaign_key"] == "k"

    other = _delivery_root(tmp_path / "b", design_sha="0" * 64)
    with pytest.raises(SystemExit, match="belongs to another design"):
        G.require_delivery(other, design)

    missing_unit = _delivery_root(tmp_path / "c", design_sha=A.SUCCESSOR_DESIGN_SHA, unit="R0_s1")
    with pytest.raises(BaseException, match="has no governed delivery of its own"):
        G.require_delivery(missing_unit, design)

    tampered = _delivery_root(tmp_path / "d", design_sha=A.SUCCESSOR_DESIGN_SHA)
    (tampered / "panel.parquet").write_bytes(b"other-bytes")
    with pytest.raises(SystemExit, match="changed after the delivery was confirmed"):
        G.require_delivery(tampered, design)

    gone = _delivery_root(tmp_path / "e", design_sha=A.SUCCESSOR_DESIGN_SHA)
    (gone / "panel.parquet").unlink()
    with pytest.raises(BaseException, match="gone from the cache"):
        G.require_delivery(gone, design)

    empty = tmp_path / "f"
    empty.mkdir(parents=True, exist_ok=True)
    with pytest.raises(BaseException, match="no governed delivery"):
        G.require_delivery(empty, design)


def _governance_root(tmp_path, *, delivered_at="2026-09-20T18:00:00Z",
                     started_at="2026-09-20T18:28:40Z", accepted_at="2026-09-20T19:01:28+00:00",
                     status="COMPLETED", recon=None, terminal_sha="e" * 64):
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "DELIVERIES.json").write_text(json.dumps({
        "design_sha256": A.SUCCESSOR_DESIGN_SHA, "lake": "public_panels", "resource": "panel.parquet",
        "units": {"R0_s1": {"campaign_key": "k", "campaign_sha256": "c" * 64, "delivery_id": "d" * 32,
                            "sha256": "b" * 64, "at": delivered_at, "host": "omega",
                            "code_identity": {"kind": "git_commit", "value": "f" * 40},
                            "cached": True, "verification_state": "VERIFIED_CACHE"}}}))
    (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps({
        "design_sha256": A.SUCCESSOR_DESIGN_SHA,
        "units": {"R0_s1": {"campaign_sha256": "c" * 64, "terminal_sha256": terminal_sha,
                            "status": status, "accepted_at": accepted_at,
                            "work_started_at": started_at,
                            "reconciliation": recon if recon is not None else
                            {"http": 200, "missing_units": [], "accounting_only": [], "lake_only": []}}}}))
    return root


def test_governed_is_a_set_of_facts_and_each_malformed_case_is_refused_by_name(tmp_path):
    C = _load("df_e1_close")
    rec = {"design_sha256": A.SUCCESSOR_DESIGN_SHA, "data_sha256": "a" * 64}

    state, facts = C._governance(_governance_root(tmp_path / "ok"), "R0_s1", rec)
    assert state == "GOVERNED" and facts["problems"] == []

    # the same instant spelled two ways is not an inversion (RP55 defect 4)
    state, facts = C._governance(_governance_root(tmp_path / "spelling",
                                                  delivered_at="2026-09-20T19:01:29Z",
                                                  started_at="2026-09-20T19:01:29+00:00",
                                                  accepted_at="2026-09-20T19:05:00Z"),
                                 "R0_s1", rec)
    assert state == "GOVERNED", facts["problems"]

    # a real inversion in either direction is still refused
    state, facts = C._governance(_governance_root(tmp_path / "late_delivery",
                                                  delivered_at="2026-09-20T18:30:00Z"), "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("did not precede the work" in p for p in facts["problems"])

    state, facts = C._governance(_governance_root(tmp_path / "early_terminal",
                                                  accepted_at="2026-09-20T18:00:01Z"), "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("accepted before the work started" in p for p in facts["problems"])

    # an impossible terminal state
    state, facts = C._governance(_governance_root(tmp_path / "state", status="SORT_OF_DONE"),
                                 "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("impossible state" in p for p in facts["problems"])

    # a digest the service could not have returned
    state, facts = C._governance(_governance_root(tmp_path / "digest", terminal_sha="not-a-digest"),
                                 "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("no digest the service could have returned" in p for p in facts["problems"])

    # the reconciliation lists must be there, and empty
    state, facts = C._governance(_governance_root(tmp_path / "recon",
                                                  recon={"http": 200, "missing_units": ["R2_s3"],
                                                         "accounting_only": [], "lake_only": []}),
                                 "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("incomplete or unsuccessful" in p for p in facts["problems"])

    state, facts = C._governance(_governance_root(tmp_path / "lists", recon={"http": 200}), "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("does not list" in p for p in facts["problems"])

    # another design
    root = _governance_root(tmp_path / "design")
    state, facts = C._governance(root, "R0_s1", {"design_sha256": "0" * 64, "data_sha256": "a" * 64})
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("another design" in p for p in facts["problems"])

    # another unit: arbitrary local files do not grant GOVERNED
    state, facts = C._governance(_governance_root(tmp_path / "unit"), "R2_s3", rec)
    assert state == "HISTORICAL_UNGOVERNED"
    assert any("names no delivery" in p for p in facts["problems"])

    # absent receipts are silence, not governance
    bare = tmp_path / "bare"
    bare.mkdir()
    state, facts = C._governance(bare, "R0_s1", rec)
    assert state == "HISTORICAL_UNGOVERNED" and facts["delivery_receipt"] is None


# --- 11. the replay phase 1 never had --------------------------------------------------------------

@pytest.mark.skipif(not (A.PHASE1_ROOT / "DATA.npz").is_file(),
                    reason="the phase-1 run root is not on this host")
def test_phase_1_was_never_closed_and_its_weights_are_still_there():
    """COUNTEREXAMPLE 10. 'nine cells, every unit registered, delivered, reported and reconciled'
    is true, and it is not a closure: this root has no CLOSE.json and no closure_replays, so the
    successor run's own verification standard was never applied to the phase that carries the
    round's headline result. The saved weights are retained, which is why the audit could apply it."""
    c = _by_claim(A.phase1_replay(), "carries a closure of its own")
    assert c["state"] == REFUTED
    assert "CLOSE.json" not in c["files_in_the_run_root"]
    assert "closure_replays" not in c["files_in_the_run_root"]
    assert "attempts" in c["files_in_the_run_root"]
