"""The collector cannot mistake an unfinished or failed worker for closure."""

import pytest

from tools.i6a_collect import complete_status, source_spec


def test_source_syntax_rejects_ambiguous_remote():
    assert source_spec("worker=host:/absolute/results") == ("worker", "host", "/absolute/results")
    for value in ("worker=host:relative", "worker=host", "bad label=host:/x"):
        with pytest.raises(ValueError, match="INVALID_REMOTE_SOURCE"):
            source_spec(value)


def test_completion_requires_exact_integer_denominator():
    assert complete_status({"state": "COMPLETE", "completed": 140, "total": 140})
    assert not complete_status({"state": "COMPLETE", "completed": 139, "total": 140})
    assert not complete_status({"state": "COMPLETE", "completed": True, "total": 1})
    assert not complete_status({"state": "RUNNING", "completed": 140, "total": 140})
