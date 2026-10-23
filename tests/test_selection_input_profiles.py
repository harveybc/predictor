import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


def test_lags_preserve_missing_positions():
    from selection_input_profiles import temporal_profile
    x = np.array([1., 2., np.nan, 4., 5., 6., 7., 8.])
    r = temporal_profile(x, max_lag=2)
    assert r['acf_n_pairs'] == [5, 4]
    assert r['pacf_n_rows'] == 5
    assert r['missing_share_fit'] == 1 / 8


def test_fixed_fit_uses_availability_and_never_future_rows():
    from selection_input_profiles import fit_frame
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2023-06-30T22:00Z', '2023-06-30T23:00Z', '2023-07-01T00:00Z']),
                          'x': [1., 100., 200.]})
    assert fit_frame(frame)['x'].tolist() == [1.]
    frame.loc[1:, 'x'] = -999
    assert fit_frame(frame)['x'].tolist() == [1.]


def test_folds_retain_embargo_and_exact_coverage():
    from selection_input_profiles import purged_folds
    folds = list(purged_folds(120, 6, 3))
    assert np.array_equal(np.concatenate([v for _, _, v in folds]), np.arange(120))
    for _, train, val in folds:
        assert all(np.min(abs(val - i)) > 3 for i in train)


def test_constant_profile_is_explicit_not_fake_zero_correlation():
    from selection_input_profiles import temporal_profile
    p = temporal_profile(np.ones(100), max_lag=3)
    assert p['constant_fit'] and p['acf_lag1'] is None
    assert p['partial_acf_lags_significant'] == []
