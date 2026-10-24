import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from close_t3_v5 import paired


def test_complete_paired_45():
    rows = {f't3_unit_{i:02d}': {'support_hash': {'TRAIN': 'a', 'INNER_A': 'b'},
            'consumed_features': ['x'], 'seeds': [1, 2, 3]} for i in range(45)}
    assert paired(rows) == rows['t3_unit_00']
    for field in ('support_hash', 'consumed_features', 'seeds'):
        changed = {k: dict(v) for k, v in rows.items()}
        changed['t3_unit_44'][field] = []
        with pytest.raises(ValueError):
            paired(changed)
    rows.pop('t3_unit_44')
    with pytest.raises(ValueError):
        paired(rows)
