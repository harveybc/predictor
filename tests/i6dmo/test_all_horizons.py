import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.i6dmo import all_horizons as A


def fixture():
    origins = np.array([100000, 110000], dtype=np.int64)
    times = np.unique(np.concatenate([origins, *[origins+h*3600 for h in A.HORIZONS]]))
    price = 1 + np.arange(len(times))*.001
    table = SimpleNamespace(origin_ts=origins, input_ok=np.ones(2, bool),
                            split=np.array(['FIT', 'FIT']),
                            label_ok=np.ones((2, 12), bool))
    old = dict(row_id=origins, x=np.ones((2, 13, 19)),
               offsets_hours=np.tile(np.arange(-24, 1, 2), (2, 1)),
               closure_crossed=np.zeros((2, 13)),
               origin_price=price[np.searchsorted(times, origins)],
               target_price=np.column_stack([price[np.searchsorted(times, origins+h*3600)]
                                             for h in A.SHORT]), mask=np.ones((2, 6), bool))
    return old, table, SimpleNamespace(ts=times, price=price)


def test_extension_keeps_every_short_byte_and_row():
    old, table, bundle = fixture()
    result = A.extend_arrays(old, table, bundle, 'FIT')
    for k in ('x', 'row_id', 'offsets_hours', 'closure_crossed', 'origin_price'):
        np.testing.assert_array_equal(old[k], result[k])
    np.testing.assert_array_equal(old['mask'], result['mask'][:, :6])
    np.testing.assert_array_equal(old['target_price'], result['target_price'][:, :6])
    assert result['mask'].shape == (2, 12)


@pytest.mark.parametrize('change', ['mask', 'price', 'split', 'input', 'absent'])
def test_rejects_changed_short_or_row_support(change):
    old, table, bundle = fixture()
    if change == 'mask':
        table.label_ok[0, 0] = False
    elif change == 'price':
        old['target_price'][0, 0] += .1
    elif change == 'split':
        table.split[0] = 'VALIDATION'
    elif change == 'input':
        table.input_ok[0] = False
    else:
        table.origin_ts[0] -= 1
    with pytest.raises(ValueError):
        A.extend_arrays(old, table, bundle, 'FIT')


def test_invalid_long_labels_stay_nan_without_dropping_short_rows():
    old, table, bundle = fixture()
    table.label_ok[:, 6:] = False
    result = A.extend_arrays(old, table, bundle, 'FIT')
    assert len(result['row_id']) == 2
    assert np.isnan(result['target_price'][:, 6:]).all()
    assert not result['mask'][:, 6:].any()


def test_extended_config_changes_only_horizons_and_initial_short_weights():
    from tools.i6dmo import model as M
    base = M.model_config(['f'+str(i) for i in range(19)], family='short_mg2')
    extended = A.extended_config(base)
    stripped = copy.deepcopy(extended)
    stripped['horizons_hours'] = A.SHORT
    assert stripped == base
    M.keras.backend.clear_session()
    original = M.build_dmo(base)
    M.keras.backend.clear_session()
    added = M.build_dmo(extended)
    proof = A.verify_model(original, added)
    assert proof['parameters'] == 7480
    assert proof['short_initial_forward'] == 'BITWISE_IDENTICAL'


def test_legacy_builder_and_trainer_structural_identity():
    proof = A.legacy_code_proof()
    assert not proof['trainer_file_unchanged']
    assert proof['other_trainer_functions_unchanged']
    assert proof['builder_unchanged']
