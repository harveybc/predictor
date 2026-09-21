import numpy as np
import pytest
import json

from tools.forecast_comparison import compare, audit_run


def test_perfect_and_reference_have_unambiguous_skill():
    y = np.array([1., 3., 5.])
    p = np.array([0., 2., 4.])
    r = compare(y, {"perfect": y, "naive": p}, reference="naive")
    assert r["perfect"]["mae"] == 0
    assert r["perfect"]["mae_skill_percent"] == 100
    assert r["naive"]["mae_skill_percent"] == 0


def test_equal_affine_scale_preserves_skill_not_raw_mae():
    y, p, b = np.array([2., 4., 8.]), np.array([1., 3., 7.]), np.array([0., 2., 6.])
    r = compare(y, {"model": p, "naive": b}, reference="naive")
    z = compare((y-3)/2, {"model": (p-3)/2, "naive": (b-3)/2}, reference="naive")
    assert r["model"]["mae"] == 1
    assert z["model"]["mae"] == .5
    assert r["model"]["mae_skill_percent"] == z["model"]["mae_skill_percent"] == 50


def test_zero_reference_is_undefined_not_epsilon_success():
    r = compare(np.array([1., 2.]), {"naive": np.array([1., 2.])}, reference="naive")
    assert r["naive"]["mae_skill_percent"] is None
    assert r["naive"]["skill_status"] == "ZERO_REFERENCE_ERROR"


@pytest.mark.parametrize("bad", [np.array([1.]), np.array([1., np.nan]), np.array([1., np.inf]), np.array([True, False]), np.array([[1.], [2.]])])
def test_incompatible_arrays_refuse_without_trimming(bad):
    with pytest.raises(ValueError):
        compare(np.array([1., 2.]), {"naive": bad}, reference="naive")


def test_empty_refuses():
    with pytest.raises(ValueError):
        compare(np.array([]), {"naive": np.array([])}, reference="naive")


def test_log_transform_is_pointwise_not_log_of_mae():
    y = np.array([1., 100.])
    p = np.array([2., 90.])
    r = compare(np.log1p(y), {"naive": np.log1p(p)}, reference="naive")
    expected = np.mean(np.abs(np.log1p(y) - np.log1p(p)))
    assert r["naive"]["mae"] == expected
    assert not np.isclose(expected, np.log1p(np.mean(np.abs(y-p))))


@pytest.fixture
def saved_run(tmp_path):
    y = np.arange(1500, dtype=float)/100
    ev = np.array([1450, 1451])
    denom = np.array([.01])
    np.savez(tmp_path/"DATA.npz", Y=y, horizon=[1], window=[2], target_channel=[0],
             train_origins=np.arange(1,20), eval_origins=ev, denominator=denom,
             scaler_mean=[0.], scaler_sd=[1.])
    for cell in [f"{r}_s{s}" for r in ("R0","R1","R2") for s in (1,2,3)] + ["controls"]:
        d = tmp_path/"attempts"/cell
        d.mkdir(parents=True)
        pred = y[ev+1]
        arrays = {"validation_y":pred[:,None], "eval_origins":ev, "denominator":denom}
        if cell == "controls":
            arrays.update(validation_pred_persistence=y[ev],
                          validation_pred_seasonal_naive_daily=y[ev+1-1440],
                          validation_pred_linear_ridge=pred)
        else:
            arrays["validation_pred"] = pred[:,None]
        np.savez(d/"arrays.npz", **arrays)
        (d/"cell.json").write_text(json.dumps({"scores":{"validation":{"model":{"mae_mean":0,"mase_mean":0}}}}))
    return tmp_path


def test_saved_run_positive_and_sources_preserved(saved_run):
    r = audit_run(saved_run)
    assert r["sources_unchanged"]
    assert r["n"] == 2
    assert r["regime_means"]["R0"]["raw_kW"]["mae"] == 0


@pytest.mark.parametrize("field", ["eval_origins", "validation_y", "denominator"])
def test_saved_population_and_denominator_are_checked(saved_run, field):
    path = saved_run/"attempts"/"R0_s1"/"arrays.npz"
    with np.load(path) as z:
        data = {k:z[k] for k in z.files}
    data[field] = data[field]+1
    np.savez(path, **data)
    with pytest.raises(ValueError):
        audit_run(saved_run)


# --- RP57: every reference visible, the rows identical, and the scaled error named for what it is ---

def test_RP57_skill_is_reported_against_every_declared_reference_on_the_same_rows():
    """One reference was not enough to read the table: a method can beat persistence and lose to the
    linear control on the very same rows. Both readings must be present, and they must be paired."""
    y = np.array([1., 2., 3., 4.])
    preds = {"persistence": np.array([2., 3., 4., 5.]),      # off by 1 everywhere
             "linear": np.array([1.5, 2.5, 3.5, 4.5]),       # off by 0.5
             "model": np.array([0.4, 1.4, 2.4, 3.4])}        # off by 0.6
    r = compare(y, preds, reference="persistence", references=("linear",))
    assert r["model"]["versus"]["persistence"]["mae_skill_percent"] == pytest.approx(40.0)
    assert r["model"]["versus"]["linear"]["mae_skill_percent"] == pytest.approx(-20.0)
    paired = r["model"]["versus"]["linear"]["paired"]
    assert paired["n"] == 4 and paired["mean_abs_error_difference"] == pytest.approx(0.1)
    assert paired["rows_first_better"] == 0


def test_RP57_a_reference_that_is_absent_is_refused_not_skipped():
    y = np.array([1., 2.])
    with pytest.raises(ValueError, match="reference missing"):
        compare(y, {"naive": y+1}, reference="naive", references=("linear_ridge",))


def test_RP57_a_zero_error_reference_gives_no_skill_against_it_either():
    y = np.array([1., 2.])
    r = compare(y, {"naive": y+1, "perfect": y}, reference="naive", references=("perfect",))
    assert r["naive"]["versus"]["perfect"]["mae_skill_percent"] is None
    assert r["naive"]["versus"]["perfect"]["status"] == "ZERO_REFERENCE_ERROR"


def test_RP57_paired_difference_keeps_the_rows_and_refuses_a_different_population():
    from tools.forecast_comparison import paired_difference
    d = paired_difference(np.array([1., 0., 2.]), np.array([2., 1., 0.]))
    assert d["mean_abs_error_difference"] == pytest.approx(0.0)      # cancels on the mean...
    assert d["rows_first_better"] == 2 and d["rows_equal"] == 0      # ...but not row by row
    with pytest.raises(ValueError):
        paired_difference(np.array([1., 2.]), np.array([1.]))


def test_RP57_a_difference_confined_to_one_stretch_is_visible_by_block():
    """Ten thousand overlapping windows are not ten thousand situations. A method that only wins in
    one stretch must not disappear into one average."""
    from tools.forecast_comparison import temporal_blocks
    y = np.zeros(8)
    a = np.array([0., 0., 0., 0., 1., 1., 1., 1.])          # wrong only in the second half
    b = np.full(8, 0.5)                                     # wrong everywhere, by less on average
    out = temporal_blocks(y, {"a": a, "b": b}, block_rows=4, reference="b", arms=[("a", "b")])
    first, second = out["blocks"]
    assert first["best"] == "a" and second["best"] == "b"
    assert first["paired"]["a_minus_b"] == pytest.approx(-0.5)
    assert second["paired"]["a_minus_b"] == pytest.approx(0.5)
    assert [blk["rows"] for blk in out["blocks"]] == [[0, 4], [4, 8]]


def test_RP57_the_scaled_error_carries_its_name_its_period_and_its_gap_treatment(saved_run):
    r = audit_run(saved_run)
    row = r["by_scale"]["raw_kW"]["R0_s1"]
    assert "persistence_scaled_error_horizon_train" in row
    assert row["legacy_horizon_scaled_error"] == row["persistence_scaled_error_horizon_train"]
    erratum = row["legacy_field_erratum"]
    assert erratum["published_as"] == "MASE" and erratum["correct_name"] == "persistence_scaled_error_horizon_train"
    names = r["scaled_error_names"]
    assert names["conventional_mase_m1_train_slice"]["m"] == 1
    assert names["conventional_mase_m1440_train_slice"]["m"] == 1440
    assert "NOT the horizon" in names["conventional_mase_m1440_train_slice"]["justification"]
    for key in ("conventional_mase_m1_train_slice", "conventional_mase_m1440_train_slice"):
        assert "EXCLUDED" in names[key]["gaps"]


def test_RP57_log1p_is_dropped_with_its_reason_when_a_value_leaves_its_domain(saved_run):
    path = saved_run/"attempts"/"R0_s1"/"arrays.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["validation_pred"] = np.full_like(data["validation_pred"], -2.0)   # outside log1p's domain
    np.savez(path, **data)
    cell = saved_run/"attempts"/"R0_s1"/"cell.json"                          # the record follows the arrays
    body = json.loads(cell.read_text())
    mae = float(np.mean(np.abs(-2.0 - np.load(path)["validation_y"].reshape(-1))))
    body["scores"]["validation"]["model"] = {"mae_mean": mae, "mase_mean": mae/0.01}
    cell.write_text(json.dumps(body))
    r = audit_run(saved_run)
    assert "log1p_of_kW" not in r["by_scale"]
    assert r["log1p_domain"]["applied"] is False and r["log1p_domain"]["smallest_value_seen"] == -2.0
    assert "clipping" in r["log1p_domain"]["why"]


def test_RP57_a_local_report_that_disagrees_keeps_the_exact_difference(saved_run):
    (saved_run/"RESULTS.json").write_text(json.dumps({"cells": {"R0_s1": {"mae": 0.25, "mase": 25.0}}}))
    check = audit_run(saved_run)["local_report_check"]
    assert check["compared"] and not check["identical"]
    assert check["differences"]["R0_s1"]["difference_mae"] == pytest.approx(0.25)
    assert check["cells"]["R0_s1"]["recomputed_mae"] == 0.0


def test_RP57_costs_and_dispersion_travel_with_the_table(saved_run):
    (saved_run/"attempts"/"R0_s1"/"outcome.json").write_text(
        json.dumps({"summary": {"cost": {"cpu_seconds": 406.681, "wall_seconds": 342.3,
                                         "peak_rss_bytes": 958988288}}}))
    r = audit_run(saved_run)
    assert r["costs"]["R0_s1"]["cpu_seconds"] == 406.681
    assert r["costs"]["R0_s2"]["cpu_seconds"] is None            # absent is absent, not zero
    row = r["by_scale"]["raw_kW"]["R0_s1"]
    assert row["abs_error_median"] == 0 and row["abs_error_q90"] == 0
    assert "no iid interval" in row["dispersion_note"]
    assert r["regime_means"]["R0"]["raw_kW"]["mae_sd_across_seeds"] == 0
