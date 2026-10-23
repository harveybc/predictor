import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from replay_t3_checkpoints import compare


def test_replay_does_not_silently_tolerate_a_small_difference():
    assert compare([[.1, .2]], [[.1, .2]])["exact"]
    result = compare([[.1, .2]], [[.1, .2 + 1e-12]])
    assert not result["exact"] and result["max_abs_delta"] > 0
    assert compare(None, None)["exact"]
    assert not compare(None, [[1]])["exact"]
    with pytest.raises(ValueError, match="SHAPE"):
        compare([[1]], [[1, 2]])
