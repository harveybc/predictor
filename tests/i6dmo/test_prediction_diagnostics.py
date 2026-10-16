"""Independent toy cases for the read-only short/long discontinuity audit."""
import numpy as np
import pytest

from tools.i6dmo.prediction_diagnostics import common_origins, describe, validate, load_fit, reports


def sample(rows=(1704067200, 1704070800, 1704074400), horizon=12):
    p = np.ones(len(rows))
    return dict(row_id=np.array(rows, dtype=np.int64), horizons_hours=np.array([horizon]),
                origin_price=p, predicted_price=p[:, None].copy(),
                target_price=(p + .01)[:, None], mask=np.ones((len(rows), 1), bool))


def test_persistence_is_not_predictive_signal():
    result = describe(sample(), 12)
    assert result["relative_error"] == 0
    assert result["movement_ratio_to_naive"] == 0
    assert result["near_naive_fraction"] == 1
    assert result["movement_correlation"] is None


def test_perfect_predictor_and_own_naive():
    z = sample()
    z["target_price"][:, 0] += np.array([0, .01, -.02])
    z["predicted_price"] = z["target_price"].copy()
    r = describe(z, 12)
    assert r["mae"] == 0
    assert r["relative_error"] == -1
    assert r["movement_correlation"] == pytest.approx(1)
    assert r["direction_accuracy_nonzero_actual"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "shape", "nonfinite", "mask", "future"])
def test_invalid_arrays_refused(fault):
    z = sample()
    if fault == "duplicate":
        z["row_id"][1] = z["row_id"][0]
    elif fault == "shape":
        z["target_price"] = z["target_price"][:-1]
    elif fault == "nonfinite":
        z["predicted_price"][0, 0] = np.nan
    elif fault == "mask":
        z["mask"] = z["mask"].astype(int)
    else:
        z["row_id"] += 366 * 86400
    with pytest.raises(ValueError):
        validate(z, "validation")


def test_masked_nan_does_not_contaminate_metrics():
    z = sample()
    z["mask"][0, 0] = False
    z["target_price"][0, 0] = np.nan
    assert describe(z, 12)["n"] == 2
    z["mask"][:] = False
    assert describe(z, 12) == {"state": "NO_SUPPORT", "n": 0}


def test_common_join_uses_origins_not_positions_and_own_horizons():
    a = sample()
    b = sample((1704070800, 1704074400, 1704078000), 18)
    b["target_price"] += .01
    ids = common_origins(a, 12, b, 18)
    assert ids.tolist() == [1704070800, 1704074400]
    assert describe(b, 18, ids)["naive_mae"] == pytest.approx(2 * describe(a, 12, ids)["naive_mae"])
    b["origin_price"][0] += .1
    with pytest.raises(ValueError, match="ORIGIN_PRICE"):
        common_origins(a, 12, b, 18)


def test_supported_target_cannot_cross_test():
    z = sample((1735686000,), 12)
    with pytest.raises(ValueError, match="TARGET_IN_TEST"):
        validate(z, "validation")


def test_authentication_failure_precedes_prediction_read(tmp_path, monkeypatch):
    import json
    from tools.i6dmo import prediction_diagnostics as diag
    (tmp_path / "PLAN.json").write_text(json.dumps({"units": [dict(unit_id="r0.fit", identity_sha256="x")]}))
    def refused(*args):
        raise ValueError("ARTIFACT_DIGEST:mutation")
    def forbidden(*args, **kwargs):
        pytest.fail("prediction load before verification")
    monkeypatch.setattr(diag.worker, "_verify_stored", refused)
    monkeypatch.setattr(diag.np, "load", forbidden)
    with pytest.raises(ValueError, match="ARTIFACT_DIGEST"):
        load_fit(tmp_path, "r0")


def test_report_retry_is_idempotent_and_undefined_correlation_omitted(tmp_path):
    from tools.i6dmo.outbox import Outbox
    r = describe(sample(), 12)
    d = dict(results={"short.r0": {"validation": {"12": r}}}, common_origins={}, weekly_boundary_heads={})
    generated = reports(d, "a" * 40)
    assert not any(m["metric"].endswith("correlation") for m in generated[0]["metrics"])
    box = Outbox(tmp_path / "outbox.sqlite")
    try:
        for _ in range(2):
            for report in generated:
                box.enqueue(report, source_unit="diagnostic")
        assert box.summary()["expected"] == 1
    finally:
        box.close()


def audit_fixture():
    start = 1704067200
    ts = start + np.arange(1, 72, dtype=np.int64) * 3600
    x = np.column_stack((np.arange(len(ts)), np.arange(len(ts)) * 2)).astype(float)
    raw = dict(ts=ts, x=x, price=1 + np.arange(len(ts)) * .001)
    ix = np.arange(0, 25, 2)[None, :]
    slots = ts[ix]
    arrays = dict(row_id=slots[:, -1].copy(), slot_instants=slots, x=x[ix],
                  origin_price=raw["price"][[24]], target_price=raw["price"][[36]][:, None],
                  offsets_hours=(slots - slots[:, -1:]) / 3600,
                  adjacent_delta_hours=np.array([[0] + [2] * 12], float),
                  closure_crossed=np.zeros((1, 13), int), head_mask=np.ones((1, 1), bool),
                  management_mask=np.ones((1, 1), bool), entry_mask=np.ones((1, 1), bool))
    calendar = dict(entry_delay_after_open_hours=12, force_flat_before_close_hours=12,
                    market_calendar=dict(weeks=[dict(market_open="2024-01-01T00:00:00Z",
                        market_close="2024-01-05T00:00:00Z", entry_cutoff="2024-01-04T00:00:00Z")]))
    split = dict(start="2024-01-01T00:00:00Z", end="2024-01-04T00:00:00Z",
                 label_available_by="2024-01-04T00:00:00Z")
    return raw, arrays, calendar, split


def test_independent_temporal_audit_and_permutation_traps():
    from tools.i6dmo.prediction_causal_audit import audit_arrays
    raw, arrays, calendar, split = audit_fixture()
    assert audit_arrays(raw, arrays, calendar, split, [12], 2)["supported_labels"] == 1


@pytest.mark.parametrize("fault,code", [("future", "FUTURE_INPUT"), ("target", "TARGET_NOT_EXACT"),
    ("mask", "BUSINESS_HEAD"), ("opening", "BUSINESS_HEAD"), ("entry", "ENTRY_MASK"),
    ("origin", "ORIGIN_NOT_INCLUDED"), ("input", "INPUT_VALUES")])
def test_causal_and_business_mutants_rejected(fault, code):
    from tools.i6dmo.prediction_causal_audit import audit_arrays
    raw, arrays, calendar, split = audit_fixture()
    if fault == "future":
        arrays["slot_instants"][0, 0] = arrays["row_id"][0] + 3600
    elif fault == "target":
        arrays["target_price"] += .1
    elif fault == "mask":
        arrays["head_mask"][:] = False
    elif fault == "opening":
        calendar["entry_delay_after_open_hours"] = 36
    elif fault == "entry":
        calendar["market_calendar"]["weeks"][0]["entry_cutoff"] = "2024-01-02T00:00:00Z"
    elif fault == "input":
        arrays["x"][0, 0, 0] += .1
    else:
        arrays["row_id"] -= 3600
    with pytest.raises(ValueError, match=code):
        audit_arrays(raw, arrays, calendar, split, [12], 2)
