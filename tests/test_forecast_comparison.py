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
