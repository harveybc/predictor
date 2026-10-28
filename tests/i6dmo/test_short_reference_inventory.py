import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2]/'tools/i6dmo/short_reference_inventory.py'


def module():
    spec = importlib.util.spec_from_file_location('short_reference_inventory', PATH)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def rows():
    return [dict(horizon=h, n=10, mae=0.9*h, naive_mae=h, relative_error=-.1)
            for h in (2, 3, 4, 5, 6, 12)]


def test_training_wins_are_not_validation_wins():
    result = module().extract({'metrics': {'FIT': rows(), 'VALIDATION': [dict(r, mae=r['naive_mae']*1.1,
        relative_error=.1) for r in rows()]}}, [])
    assert [(r['split'], r['wins']) for r in result] == [('fit', 6), ('validation', 0)]


def test_a_positive_head_prevents_all_six_claim():
    data = rows()
    data[0].update(mae=3, relative_error=.5)
    record = module().extract({'metrics': {'validation': data}}, [])[0]
    assert record['wins'] == 5


def test_legacy_receipt_needs_explicit_horizons_and_keeps_split():
    legacy = [dict(count=10, mae=.9*h, naive=h, relative_error=-.1) for h in (2,3,4,5,6,12)]
    data = {'model_side_summary': {'fit': legacy}}
    assert module().extract(data, []) == []
    assert module().extract(data, [2,3,4,5,6,12])[0]['split'] == 'fit'


def test_duplicate_or_long_horizons_are_not_a_short_vector():
    m = module()
    assert m.extract({'metrics': {'validation': [rows()[0]]*6}}, []) == []
    assert m.extract({'metrics': {'validation': [dict(r, horizon=r['horizon']+20) for r in rows()]}}, []) == []


def test_stored_sign_must_agree_with_mae_and_own_naive():
    data = copy.deepcopy(rows())
    data[0]['relative_error'] = .1
    with pytest.raises(ValueError, match='RELATIVE_ERROR_MISMATCH'):
        module().extract({'metrics': {'validation': data}}, [])


def test_count_and_positive_naive_required():
    data = copy.deepcopy(rows())
    data[0]['n'] = 0
    with pytest.raises(ValueError, match='INVALID_SCORE'):
        module().extract({'metrics': {'validation': data}}, [])
