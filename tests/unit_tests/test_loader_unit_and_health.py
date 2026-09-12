"""R5 (order 2026-09-11): the loader unit file, verified as a unit file.

Two defects met here, and both had been *visible* the whole time:

  * `StartLimitIntervalSec=` and `StartLimitBurst=` were declared under
    `[Service]`. systemd parses the file, prints `Unknown key ...,
    ignoring`, and applies its own default — so the rate limit the
    comment promised was never in force. `systemd-analyze verify` said
    so on every run; nothing ever ran it;
  * nothing in the repository read the shipped unit at all, so a
    directive could move, be misspelled or be dropped without a single
    test noticing.

These tests read the shipped file, and — where a systemd manager is
reachable — ask systemd itself. They never start, stop, restart, enable
or disable anything.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UNIT = REPO / "deploy/crispdm-olap-loader.service"

#: directives systemd only honours in [Unit], and silently ignores in
#: [Service]. The rate limit is the whole point of Restart=always.
UNIT_ONLY = ("StartLimitIntervalSec", "StartLimitBurst",
             "StartLimitAction", "StartLimitInterval")


def sections() -> dict[str, str]:
    """Map every directive key to the section it was declared in."""
    placed: dict[str, str] = {}
    section = None
    for raw in UNIT.read_text().splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
        elif line and not line.startswith("#") and "=" in line:
            placed.setdefault(line.split("=", 1)[0], section)
    return placed


def test_rate_limit_directives_sit_in_the_unit_section():
    placed = sections()
    declared = {k: v for k, v in placed.items() if k in UNIT_ONLY}
    assert declared, "the unit declares no start rate limit at all"
    wrong = {k: v for k, v in declared.items() if v != "[Unit]"}
    assert not wrong, (
        f"systemd ignores these outside [Unit], so the limit is not in "
        f"force: {wrong}")


def test_restart_always_is_paired_with_a_rate_limit():
    placed = sections()
    if placed.get("Restart") == "[Service]":
        body = UNIT.read_text()
        assert "Restart=always" in body
        assert "StartLimitBurst" in placed, (
            "Restart=always without a burst limit is a busy loop")


def test_the_unit_keeps_the_accelerator_invisible():
    body = UNIT.read_text()
    assert "Environment=CUDA_VISIBLE_DEVICES=" in body, (
        "the loader is CPU-only by construction, not by convention")


def test_credentials_are_read_from_outside_the_repository():
    body = UNIT.read_text()
    assert "EnvironmentFile=" in body
    for line in body.splitlines():
        if line.startswith("EnvironmentFile="):
            value = line.split("=", 1)[1]
            assert "Documents/GitHub" not in value, (
                "credentials must not live inside a Git checkout")


def test_no_literal_home_path_is_baked_into_the_unit():
    """%h keeps the unit portable and keeps a private path out of Git."""
    body = UNIT.read_text()
    assert "/home/" not in body, "use %h, never a literal home directory"


# ---------------------------------------------------- systemd's own view
def _systemd_available() -> bool:
    if not shutil.which("systemd-analyze") or not os.environ.get(
            "XDG_RUNTIME_DIR"):
        return False
    return subprocess.run(("systemctl", "--user", "show", "-p", "Version"),
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _systemd_available(),
                    reason="no reachable systemd --user manager")
def test_systemd_analyze_reports_no_ignored_directive(tmp_path):
    """Ask systemd, not a parser of ours, whether it understood the file.

    The file is copied to a temp name first so the check is about the
    shipped bytes and cannot be confused with whatever is installed.
    """
    staged = tmp_path / "crispdm-olap-loader-verify.service"
    staged.write_text(UNIT.read_text())
    out = subprocess.run(("systemd-analyze", "--user", "verify", str(staged)),
                         capture_output=True, text=True, timeout=120)
    complaints = [ln for ln in (out.stdout + out.stderr).splitlines()
                  if staged.name in ln]
    assert not complaints, complaints
