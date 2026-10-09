"""Tests for unattended deterministic I6-D weekly shard execution."""

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
