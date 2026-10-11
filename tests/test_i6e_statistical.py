"""Clock, causality and fixed-parameter statistical control acceptance tests."""

import numpy as np
import pytest

from tools.i6e_statistical import phase_grid, delayed_exog, forecast_origins, fit_phase


def test_phase_grid_preserves_calendar_gaps():
    assert np.array_equal(phase_grid(0, 12*3600, 3, 1), [3600,14400,25200,36000])
    with pytest.raises(ValueError):
        phase_grid(0, 100, 3, 3)


def test_exogenous_forecast_values_are_known_at_origin():
    ts = np.arange(200)*3600
    x = np.arange(200, dtype=float)[:,None]
    future = np.array([118,136,154,172])*3600
    z, observed = delayed_exog(ts,x,future)
    assert np.array_equal(z[:,0], [46,64,82,100])
    assert observed.all()
    changed = x.copy()
    changed[101:] = 999999
    assert np.array_equal(delayed_exog(ts,changed,future)[0],z)
    z, observed = delayed_exog(ts,x,np.array([0]))
    assert not observed[0] and np.isnan(z[0]).all()


@pytest.mark.parametrize("with_exog", [False, True])
def test_fixed_parameter_forecast_ignores_future_prices(with_exog):
    rng = np.random.default_rng(6102026)
    y = 1 + np.cumsum(rng.normal(0,0.002,170))
    x = rng.normal(size=(170,2)) if with_exog else None
    model = fit_phase(y[:140], None if x is None else x[:140])
    origins = np.array([141,145,150])
    first = forecast_origins(model,y,x,origins,4)
    altered = y.copy()
    altered[142:] += 100
    second = forecast_origins(model,altered,x,origins[:1],4)
    assert np.array_equal(first[:1],second)
    assert np.isfinite(first).all()
    assert np.array_equal(model.params, model.model.filter(model.params).params)


def test_missing_price_is_not_carried_forward():
    y = np.arange(150,dtype=float)*0.001+1
    y[20:24] = np.nan
    result = fit_phase(y,None)
    assert np.isnan(result.model.endog[20:24]).all()


def test_design_does_not_choose_its_own_population(tmp_path):
    import json
    from tools.i6e_statistical_campaign import code_digest, design_digest, read_design
    design=dict(code_sha256=code_digest(),expected_cells=312,weeks=[{}]*52,
                spacings=[72,36,18],models=["ARIMA","ARIMAX"],test_read=False)
    design["design_sha256"]=design_digest(design)
    path=tmp_path/"DESIGN.json"
    path.write_text(json.dumps(design))
    assert read_design(tmp_path)["expected_cells"]==312
    design["expected_cells"]=1
    design["design_sha256"]=design_digest(design)
    path.write_text(json.dumps(design))
    with pytest.raises(ValueError,match="population"):
        read_design(tmp_path)


def test_unknown_future_exog_is_not_used_by_dynamic_forecast():
    rng=np.random.default_rng(6102026)
    y=1+np.cumsum(rng.normal(0,0.002,170))
    x=rng.normal(size=(170,2))
    fitted=fit_phase(y[:140],x[:140])
    first=forecast_origins(fitted,y,x,np.array([141]),4)
    changed=x.copy()
    changed[146:]+=100000
    assert np.array_equal(first,forecast_origins(fitted,y,changed,np.array([141]),4))


def test_numerical_failure_does_not_repeat_or_block_independent_cells(tmp_path,monkeypatch):
    from tools import i6e_statistical_dispatch as dispatch
    folder=tmp_path/"cells/ARIMAX_w000_d72"
    folder.mkdir(parents=True)
    (folder/"CELL.json").write_text("{}")
    monkeypatch.setattr(dispatch,"validate",lambda *a:dict(status="FAILED_CONVERGENCE"))
    pending=list(dispatch.next_cells(tmp_path,"ARIMAX"))
    assert len(pending)==155 and (0,72) not in pending
    assert pending[0]==(0,36)


def test_interrupted_claim_is_not_a_fresh_experiment(tmp_path):
    from tools.i6e_statistical_dispatch import next_cells
    folder=tmp_path/"cells/ARIMAX_w000_d72"
    folder.mkdir(parents=True)
    (folder/"CLAIM").write_text("old")
    with pytest.raises(ValueError,match="interrupted"):
        list(next_cells(tmp_path,"ARIMAX"))


def test_two_workers_partition_weeks_without_repetition(tmp_path):
    from tools.i6e_statistical_dispatch import next_cells
    even=set(next_cells(tmp_path,"ARIMAX",0))
    odd=set(next_cells(tmp_path,"ARIMAX",1))
    assert len(even)==len(odd)==78
    assert not even.intersection(odd)
    assert len(even.union(odd))==156
