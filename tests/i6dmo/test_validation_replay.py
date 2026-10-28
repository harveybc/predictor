import copy

import numpy as np
import pytest

from tools.i6dmo import validation_replay as R


def arrays():
    return dict(origin_price=np.array([1., 2.]), origin_ts=np.array([10, 20]),
                target_price=np.array([[1.1, np.nan], [2.2, np.nan]]),
                mask=np.array([[True, False], [True, False]]))


def test_exact_replay_and_nonzero_difference_are_distinguished():
    a = arrays()
    pred = np.array([[1.05, 0.], [2.1, 0.]])
    r = R.compare(a, pred, dict(a, prediction=pred.copy()), [2, 4])
    assert r['prediction_exact'] and r['metrics_exact'] and r['max_abs_prediction_delta'] == 0
    assert r['metrics'][1]['n'] == 0 and r['metrics'][1]['mae'] is None
    changed = pred.copy(); changed[0, 0] += 1e-10
    r = R.compare(a, changed, dict(a, prediction=pred), [2, 4])
    assert not r['prediction_exact'] and not r['metrics_exact']
    assert r['max_abs_prediction_delta'] > 0


def test_aggregation_uses_sums_and_own_naive():
    rows = [dict(horizon=2, n=1, sum_abs_error=1., sum_abs_naive_error=2.),
            dict(horizon=2, n=3, sum_abs_error=3., sum_abs_naive_error=3.),
            dict(horizon=4, n=1, sum_abs_error=3., sum_abs_naive_error=6.)]
    result = R.pool(rows, [2, 4])
    assert result[0]['mae'] == 1. and result[0]['naive_mae'] == 1.25
    assert result[0]['relative_error'] == pytest.approx(-.2)
    assert result[1]['relative_error'] == -.5


def test_future_donor_and_changed_rows_are_refused():
    with pytest.raises(ValueError, match='DONOR_NOT_YET_AVAILABLE'):
        R.check_available('2024-01-12T22:00:00+00:00', np.array([1]))
    a = arrays(); ref = copy.deepcopy(a); ref['origin_ts'][0] += 1
    with pytest.raises(ValueError, match='RETAINED_POPULATION_CHANGED'):
        R.compare(a, np.zeros((2, 2)), dict(ref, prediction=np.zeros((2, 2))), [2, 4])


def test_changed_bytes_fail_authentication(tmp_path):
    p = tmp_path / 'input'; p.write_bytes(b'original')
    expected = {'input': R.W.file_sha256(p)}
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):
        R.W.authenticate(tmp_path, expected)


def test_resume_rejects_different_unit(tmp_path):
    R.W.B.put(tmp_path / 'RESULT.json', dict(unit='first', input_digest='abc'))
    assert R.adopt(tmp_path, 'first', 'abc')['unit'] == 'first'
    with pytest.raises(ValueError, match='REPLAY_TERMINAL_IDENTITY'):
        R.adopt(tmp_path, 'second', 'abc')
