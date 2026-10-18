"""Part B of ORDEN_SATOSHI_OPERACION.md: explicit closure coverage (metrics unchanged). Synthetic only."""
import numpy as np
import pytest

from tools.i6dmo import prequential_closure as PC, weekly_test as P, weekly18 as W

HEADS = dict(short=[2, 3], long=[18])


def _period(wid, open_iso, close_iso):
    return dict(id=wid, train_end=open_iso, validation_end=close_iso, week_ids=[wid])


WEEKS = [  # both annual boundaries plus three ordinary weeks
    _period("WEEKLY-2024-12-29", "2024-12-29T22:00:00+00:00", "2025-01-03T22:00:00+00:00"),
    _period("WEEKLY-2025-01-05", "2025-01-05T22:00:00+00:00", "2025-01-10T22:00:00+00:00"),
    _period("WEEKLY-2025-01-12", "2025-01-12T22:00:00+00:00", "2025-01-17T22:00:00+00:00"),
    _period("WEEKLY-2025-12-21", "2025-12-21T22:00:00+00:00", "2025-12-26T22:00:00+00:00"),
    _period("WEEKLY-2025-12-28", "2025-12-28T22:00:00+00:00", "2026-01-02T22:00:00+00:00"),
]


def _spec():
    cells = [dict(id=f"{fam}__{p['id']}", family=fam, horizons=HEADS[fam], period=p)
             for fam in ("short", "long") for p in WEEKS]
    return dict(cells=cells)


def _row(h, n):
    return dict(horizon=h, n=n, mae=0.1 if n else None, naive_mae=0.2 if n else None,
                relative_error=-0.5 if n else None, sum_abs_error=0.1 * n, sum_abs_naive_error=0.2 * n,
                disposition=None if n else "NO_VALID_TARGET_IN_PERIOD")


def _results(spec, *, states, support=None):
    """states: {cell_id: state}; support: {(cell_id, h): n} overrides (default 4 pairs)."""
    out = {}
    for c in spec["cells"]:
        state = states.get(c["id"])
        if state is None:
            continue
        r = dict(state=state, cell=c, batch_id="b")
        if state == "COMPLETE":
            r["metrics"] = dict(TEST=[_row(h, (support or {}).get((c["id"], h), 4)) for h in c["horizons"]])
        if state == "FAILED":
            r["reason"] = "TRANSIENT_EXHAUSTED:TransientProcessError:GPU_NOT_VISIBLE"
        out[c["id"]] = r
    return out


def test_coverage_counts_both_annual_boundaries_as_no_pair_and_keeps_the_invariant():
    spec = _spec()
    states = {c["id"]: ("SKIPPED_NO_EVAL_ROWS" if c["period"]["id"] in ("WEEKLY-2024-12-29", "WEEKLY-2025-12-28")
                        else "COMPLETE") for c in spec["cells"]}
    cov = PC.coverage(spec, _results(spec, states=states), HEADS)
    for fam in ("short", "long"):
        f = cov["families"][fam]
        assert f["declared_weeks"] == 5 and f["scored_weeks"] == 3 and f["no_pair_weeks"] == 2
        assert f["failed_weeks"] == 0 and f["pending_weeks"] == 0 and f["invariant_holds"] and f["complete"]
        assert [w["id"] for w in f["no_pair_weeks_detail"]] == [f"{fam}__WEEKLY-2024-12-29", f"{fam}__WEEKLY-2025-12-28"]
        assert all(w["year_boundary"] for w in f["no_pair_weeks_detail"])
        assert f["no_pair_weeks_detail"][0]["market_open"] == "2024-12-29T22:00:00+00:00"
        assert f["no_pair_weeks_detail"][1]["market_close"] == "2026-01-02T22:00:00+00:00"
        assert f["year_boundary_week_ids"] == [f"{fam}__WEEKLY-2024-12-29", f"{fam}__WEEKLY-2025-12-28"]
        assert f["prediction_coverage"]["fraction"] == pytest.approx(0.6)
        for h in f["per_horizon"].values():
            assert h["weeks_with_pairs"] == 3 and h["weeks_without_support"] == 2 and h["pairs_scored"] == 12
    text = PC.markdown(cov)
    assert "100.00%" not in text and "60.00%" in text and "SIN_PARES" in text and "SKIPPED_NO_EVAL_ROWS" in text
    assert "WEEKLY-2024-12-29" in text and "WEEKLY-2025-12-28" in text and "| yes |" in text
    assert "NO_VALID_TARGET_IN_PERIOD" in text and "not a zero MAE" in text


def test_a_head_without_support_in_a_useful_week_is_counted_per_horizon_only():
    spec = _spec()
    states = {c["id"]: "COMPLETE" for c in spec["cells"]}
    cov = PC.coverage(spec, _results(spec, states=states, support={("short__WEEKLY-2025-01-12", 3): 0}), HEADS)
    f = cov["families"]["short"]
    assert f["scored_weeks"] == 5 and f["no_pair_weeks"] == 0           # the week itself IS scored
    assert f["per_horizon"]["2"] == dict(horizon=2, weeks_with_pairs=5, weeks_without_support=0,
                                         weeks_without_support_ids=[], pairs_scored=20,
                                         aggregation=f["per_horizon"]["2"]["aggregation"])
    assert f["per_horizon"]["3"]["weeks_with_pairs"] == 4 and f["per_horizon"]["3"]["pairs_scored"] == 16
    assert f["per_horizon"]["3"]["weeks_without_support_ids"] == ["short__WEEKLY-2025-01-12"]
    assert f["prediction_coverage"]["fraction"] == 1.0                    # no skipped week here: 100% is honest


def test_a_failure_and_an_incomplete_closure_are_explicit_and_never_100_percent():
    spec = _spec()
    states = {c["id"]: "COMPLETE" for c in spec["cells"] if c["family"] == "long"}
    states.update({"short__WEEKLY-2024-12-29": "SKIPPED_NO_EVAL_ROWS", "short__WEEKLY-2025-01-05": "COMPLETE",
                   "short__WEEKLY-2025-01-12": "FAILED"})                 # two short weeks pending
    cov = PC.coverage(spec, _results(spec, states=states), HEADS)
    f = cov["families"]["short"]
    assert (f["scored_weeks"], f["no_pair_weeks"], f["failed_weeks"], f["pending_weeks"]) == (1, 1, 1, 2)
    assert f["invariant_holds"] and not f["complete"]
    assert f["failed_weeks_detail"][0]["cause"].startswith("TRANSIENT_EXHAUSTED")
    assert f["pending_week_ids"] == ["short__WEEKLY-2025-12-21", "short__WEEKLY-2025-12-28"]
    assert f["prediction_coverage"]["fraction"] == pytest.approx(0.2)
    assert cov["families"]["long"]["complete"] and cov["families"]["long"]["prediction_coverage"]["fraction"] == 1.0
    text = PC.markdown(cov)
    assert "| short | 5 | 1 | 1 | 1 | 2 | 20.00% |" in text and "closure INCOMPLETE" in text
    assert "failed weeks" in text and "GPU_NOT_VISIBLE" in text
    with pytest.raises(ValueError, match="UNKNOWN_TERMINAL_STATE"):
        PC.coverage(spec, {"short__WEEKLY-2025-01-05": dict(state="WEIRD", cell=spec["cells"][1])}, HEADS)


def test_tick_seam_adds_coverage_without_changing_annual_metrics(tmp_path):
    """weekly_test.closure_coverage uses the real family heads; the pooled metrics are untouched."""
    spec = dict(cells=[dict(id="short__WEEKLY-2025-01-05", family="short", horizons=P.D.SHORT, period=WEEKS[1]),
                       dict(id="long__WEEKLY-2025-01-05", family="long", horizons=P.D.LONG, period=WEEKS[1])])
    results = _results(spec, states={c["id"]: "COMPLETE" for c in spec["cells"]})
    cov = P.closure_coverage(spec, results)
    assert set(cov["families"]["short"]["per_horizon"]) == {str(h) for h in P.D.SHORT}
    assert set(cov["families"]["long"]["per_horizon"]) == {str(h) for h in P.D.LONG}
    assert cov["glossary"]["SIN_PARES"].startswith("Descriptive name of the persisted week disposition SKIPPED_NO_EVAL_ROWS")
    # the annual pooled numbers come from W.pool_metrics on the same rows as before (coverage is additive)
    rows = results["short__WEEKLY-2025-01-05"]["metrics"]["TEST"]
    pooled = W.pool_metrics(rows, 2)
    assert pooled["n"] == 4 and pooled["mae"] == pytest.approx(0.1) and pooled["naive_mae"] == pytest.approx(0.2)
    assert P.enrich_status(dict(state="RUNNING"), spec, tmp_path, results) == dict(state="RUNNING")  # default seam no-op
