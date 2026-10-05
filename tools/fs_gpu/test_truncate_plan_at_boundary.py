"""The in-place truncation must end a live bash read-loop after its current cell only."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truncate_plan_at_boundary as T  # noqa: E402

LOG = [
    "2026-10-05T00:00:00Z START worker_b UUID=x",
    "2026-10-05T00:00:01Z BEGIN cap=8000M tier_1 batch_002 a.one",
    "2026-10-05T00:10:00Z END rc=0 tier_1 batch_002 a.one",
    "2026-10-05T00:10:00Z BEGIN cap=8000M tier_1 batch_002 b.two",
]
PLAN = b"tier_1 a.one\ntier_1 b.two\ntier_1 c.three\nexploration d.four\n"


def test_current_cell_is_last_begin_without_done():
    assert T.current_cell_from_log(LOG) == "b.two"
    assert T.current_cell_from_log(LOG + ["2026-10-05T00:30:00Z DONE"]) is None
    assert T.current_cell_from_log(["2026-10-05T00:00:00Z START"]) is None


def test_consumed_prefix_is_byte_identical_and_refuses_ambiguity():
    prefix, k = T.consumed_prefix(PLAN, "b.two")
    assert k == 2 and prefix == b"tier_1 a.one\ntier_1 b.two\n"
    with pytest.raises(ValueError):
        T.consumed_prefix(PLAN, "zz.missing")
    with pytest.raises(ValueError):
        T.consumed_prefix(PLAN + b"tier_2 b.two\n", "b.two")


def test_live_bash_loop_ends_after_current_cell(tmp_path):
    plan = tmp_path / "plan.txt"
    plan.write_bytes(PLAN)
    out = tmp_path / "out.txt"
    script = tmp_path / "loop.sh"
    script.write_text(
        "#!/bin/bash\nwhile read -r s f; do echo \"BEGIN $s $f\"; sleep 1.5; echo \"END $s $f\"; done < \"$1\"; echo DONE\n"
    )
    inode_before = plan.stat().st_ino
    proc = subprocess.Popen(["bash", str(script), str(plan)], stdout=out.open("w"))
    time.sleep(2.0)  # inside cell b.two (a.one took 1.5 s)
    prefix, _ = T.consumed_prefix(PLAN, "b.two")
    T.truncate_in_place(plan, prefix)
    proc.wait(timeout=20)
    lines = out.read_text().splitlines()
    assert lines == ["BEGIN tier_1 a.one", "END tier_1 a.one", "BEGIN tier_1 b.two", "END tier_1 b.two", "DONE"]
    assert plan.stat().st_ino == inode_before
    assert plan.read_bytes() == prefix


def test_cli_dry_run_then_apply_with_backup(tmp_path, capsys):
    plan = tmp_path / "plan.txt"
    plan.write_bytes(PLAN)
    log = tmp_path / "driver.log"
    log.write_text("\n".join(LOG) + "\n")
    backup = tmp_path / "plan.backup"
    assert T.main(["--plan", str(plan), "--log", str(log)]) == 0
    assert plan.read_bytes() == PLAN
    assert T.main(["--plan", str(plan), "--log", str(log), "--apply", "--backup", str(backup)]) == 0
    assert plan.read_bytes() == b"tier_1 a.one\ntier_1 b.two\n" and backup.read_bytes() == PLAN
    log.write_text("\n".join(LOG) + "\n2026-10-05T01:00:00Z DONE\n")
    assert T.main(["--plan", str(plan), "--log", str(log), "--apply"]) == 3
