"""Tests for unattended deterministic I6-D weekly shard execution."""

import pytest

from tools import i6d_weekly_shard_runner as runner


def _design():
    return {
        "seeds": [0],
        "weeks": [{"ordinal": index} for index in range(5)],
        "arms": ["DENSE", "CONV"],
    }


def test_shards_partition_every_cell_exactly_once():
    all_cells = runner.enumerate_cells(_design())
    shards = [runner.shard_cells(_design(), index, 3) for index in range(3)]

    assert sorted(cell for shard in shards for cell in shard) == sorted(all_cells)
    assert not (set(shards[0]) & set(shards[1]))
    assert not (set(shards[0]) & set(shards[2]))
    assert not (set(shards[1]) & set(shards[2]))


def test_eta_uses_only_completed_cell_durations():
    status = runner.progress_report(
        worker_id="dragon", assigned=8, completed=3, failures=[],
        durations=[10.0, 20.0, 30.0], active=(0, 2, "DENSE"),
    )

    assert status["state"] == "RUNNING"
    assert status["pending"] == 5
    assert status["median_cell_seconds"] == 20.0
    assert status["eta_seconds"] == 100.0


def test_failed_cells_are_terminal_for_the_worker_and_not_counted_complete():
    failure = {"seed": 0, "week_ordinal": 1, "arm": "CONV", "reason": "boom"}
    status = runner.progress_report(
        worker_id="gamma", assigned=2, completed=1, failures=[failure],
        durations=[4.0], active=None,
    )

    assert status["state"] == "COMPLETE_WITH_FAILURES"
    assert status["completed"] == 1
    assert status["failed"] == 1
    assert status["pending"] == 0
    assert status["failures"] == [failure]


class _FakeTensorFlow:
    class config:
        @staticmethod
        def list_physical_devices(kind):
            assert kind == "GPU"
            return ["/physical_device:GPU:0"]


def test_gpu_preflight_accepts_exactly_one_visible_device():
    assert runner.gpu_preflight(_FakeTensorFlow) == ["/physical_device:GPU:0"]


@pytest.mark.parametrize("devices", [[], ["GPU:0", "GPU:1"]])
def test_gpu_preflight_rejects_cpu_fallback_and_ambiguous_placement(devices):
    class FakeTensorFlow:
        class config:
            @staticmethod
            def list_physical_devices(kind):
                assert kind == "GPU"
                return devices

    with pytest.raises(RuntimeError, match="exactly one visible GPU"):
        runner.gpu_preflight(FakeTensorFlow)


def test_tensorflow_runtime_is_released_between_cells():
    calls = []

    class FakeTensorFlow:
        class keras:
            class backend:
                @staticmethod
                def clear_session(*, free_memory):
                    calls.append(free_memory)

    runner.release_tensorflow_runtime(FakeTensorFlow)

    assert calls == [True]
