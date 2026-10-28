"""Tests for process-isolated weekly shard execution."""

from tools import i6d_weekly_isolated_shard_runner as isolated


def test_cell_command_keeps_every_input_and_exact_cell_identity(tmp_path):
    command = isolated.cell_command(
        output=tmp_path / "campaign",
        seed=3,
        week_ordinal=7,
        arm="CONV",
        feature_parquets=["train-a.parquet", "train-b.parquet"],
        target_parquet="target.parquet",
        validation_feature_parquets=["val-a.parquet", "val-b.parquet"],
        validation_target_parquet="val-target.parquet",
    )

    assert command.count("--feature-parquet") == 2
    assert command.count("--validation-feature-parquet") == 2
    assert command[command.index("--seed") + 1] == "3"
    assert command[command.index("--week-ordinal") + 1] == "7"
    assert command[command.index("--arm") + 1] == "CONV"


def test_child_environment_exposes_only_the_requested_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong")

    environment = isolated.child_environment("GPU-exact")

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-exact"
