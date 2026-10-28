"""Origin-anchored SARIMAX controls with weekly fixed parameters.

Calendar gaps remain NaN endogenous observations. Exogenous values are delayed
72 physical hours, scaled and rank-filtered on TRAIN only. Missing regressors
have explicit source and market masks, not invented market prices.
"""

import numpy as np
from scipy.linalg import qr
from statsmodels.tsa.statespace.sarimax import SARIMAX


def phase_grid(start, end, spacing, phase):
    """UTC-hour phase grid on [start,end), preserving closed-market slots."""
    if spacing not in (3,18,36,72) or not 0 <= phase < spacing or end <= start:
        raise ValueError("invalid physical-hour phase grid")
    first_hour = (int(start)+3599)//3600
    first_hour += (phase-first_hour) % spacing
    return np.arange(first_hour*3600, int(end), spacing*3600, dtype=np.int64)


def delayed_exog(timestamps, values, grid):
    """Exact t-72h regressors; no as-of fill and no realized future inputs."""
    wanted = np.asarray(grid)-72*3600
    indices = np.searchsorted(timestamps,wanted)
    safe = np.minimum(indices,len(timestamps)-1)
    observed = (indices<len(timestamps)) & (timestamps[safe]==wanted)
    output = np.full((len(grid),values.shape[1]),np.nan)
    output[observed] = values[safe[observed]]
    return output,observed


def masked_exog(raw, market_observed):
    """Finite raw placeholders plus masks, before TRAIN-only scaling."""
    output = np.empty((len(raw),raw.shape[1]*2+1))
    output[:,0:-1:2] = np.where(np.isfinite(raw),raw,0)
    output[:,1:-1:2] = np.isfinite(raw)
    output[:,-1] = market_observed
    return output


def fit_phase(price, exog, maxiter=100):
    """One ML fit, no drift/order search, with fixed regression coefficients."""
    finite = np.asarray(price)[np.isfinite(price)]
    if len(finite)<100 or np.any(finite<=0):
        raise ValueError("insufficient positive TRAIN prices")
    center, scale = float(finite.mean()), float(finite.std())
    if scale<=0:
        raise ValueError("degenerate TRAIN price")
    transformed = None
    mean = sd = columns = None
    if exog is not None:
        if not np.isfinite(exog).all() or len(exog)!=len(price):
            raise ValueError("nonfinite or misaligned regressors")
        mean,sd = exog.mean(axis=0),exog.std(axis=0)
        sd = np.where(sd>1e-12,sd,1)
        z = (exog-mean)/sd
        _,r,pivot = qr(z,mode="economic",pivoting=True)
        diagonal = np.abs(np.diag(r))
        tolerance = max(z.shape)*np.finfo(float).eps*(diagonal.max() if len(diagonal) else 0)
        columns = np.sort(pivot[:int(np.sum(diagonal>tolerance))])
        transformed = z[:,columns] if len(columns) else None
    model = SARIMAX((np.asarray(price)-center)/scale,exog=transformed,
                    order=(1,1,0),trend="n",mle_regression=True,
                    enforce_stationarity=True,concentrate_scale=True)
    result = model.fit(disp=False,maxiter=maxiter)
    result._i6e_scaling = (center,scale,mean,sd,columns)
    return result


def forecast_origins(fitted, price, exog, origin_indices, steps):
    """Filter with fixed weekly parameters; each forecast stops conditioning at t."""
    center,scale,mean,sd,columns = fitted._i6e_scaling
    transformed = None if mean is None or not len(columns) else ((exog-mean)/sd)[:,columns]
    result = fitted.apply((np.asarray(price)-center)/scale,exog=transformed,refit=False)
    predictions = []
    for index in origin_indices:
        if index+steps>=len(price):
            raise ValueError("forecast endpoint outside physical grid")
        forecast = result.get_prediction(start=int(index+1),end=int(index+steps),dynamic=True)
        predictions.append(float(forecast.predicted_mean[-1])*scale+center)
    output = np.array(predictions)
    if not np.isfinite(output).all() or np.any(output<=0):
        raise ValueError("invalid statistical price forecast")
    return output
