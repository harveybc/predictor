"""act.p_hit forward-only (expanding) cross-fit: a block's predictions never depend on rows after it (mutation),
the first block is NaN (counted), rows after the allowed span use the full model, the default mode is unchanged."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from tools import activity_producer as AP  # noqa: E402


def synth(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f@1": rng.normal(size=n), "f@2": rng.normal(size=n)})
    y = (X["f@1"].to_numpy() + 0.3 * rng.normal(size=n) > 0).astype(float)
    res = np.arange(n) + 3
    allowed = np.zeros(n, bool)
    allowed[:2400] = True
    return X, y, res, allowed


def test_expanding_is_forward_only_and_first_block_nan():
    X, y, res, allowed = synth()
    p, rep = AP.scope_predictions(X, y, res, allowed, k=4, embargo=10, lookback=2, mode="expanding")
    assert rep["mode"] == "expanding" and rep["nan_first_block_rows"] == 600
    assert np.isnan(p[:600]).all() and np.isfinite(p[600:]).all()
    # mutation: perturb everything after block 1 (rows >= 1200): block 1's predictions must be identical
    X2, y2 = X.copy(), y.copy()
    X2.iloc[1200:] = X2.iloc[1200:] * -3 + 5
    y2[1200:] = 1 - y2[1200:]
    p2, _ = AP.scope_predictions(X2, y2, res, allowed, k=4, embargo=10, lookback=2, mode="expanding")
    assert np.array_equal(p[600:1200], p2[600:1200])
    # the purged k-fold (default) is NOT forward-only: the same perturbation moves block 1
    q, rq = AP.scope_predictions(X, y, res, allowed, k=4, embargo=10, lookback=2)
    q2, _ = AP.scope_predictions(X2, y2, res, allowed, k=4, embargo=10, lookback=2)
    assert rq["mode"] == "purged_kfold" and not np.array_equal(q[600:1200], q2[600:1200])


def test_unknown_mode_refused():
    X, y, res, allowed = synth(400)
    with pytest.raises(ValueError, match="ACTIVITY_MODE_UNKNOWN"):
        AP.scope_predictions(X, y, res, allowed, mode="leaky")
