# -*- coding: utf-8 -*-
"""The declared-coverage fixture must not break a file that is run ON ITS OWN.

RR02 added `tests/conftest.py::_crispdm_declared_coverage`, an autouse fixture that loads
`tools/crispdm_admission.py` by spec.  It executed the module without ever assigning
`sys.modules["crispdm_admission"]`, so `dataclasses` resolved the class's own module to None and
every `@dataclass` in that file raised at import.  Result: 122 tests erroring at setup.

**Why this test runs a subprocess, and why it runs ONE file alone.**  The fault only reaches the
first unmarked test in a process.  `tests/test_df_admission_guard.py` opts out with
`crispdm_uncovered`, and `tests/test_crispdm_admission.py` does `import crispdm_admission` at
module scope -- so any selection that begins with it puts the name in `sys.modules` before the
fixture ever runs, and the fixture then works.  Every combined selection of this campaign begins
with it.  A regression test that exercised the fixture from inside such a group would therefore
pass on the broken code too, and would reproduce exactly the blindness that hid this for two days.
The only thing that catches it is a covered file, on its own, in a fresh interpreter.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# A file that is COVERED (no `crispdm_uncovered` marker, so the fixture runs for it) and that does
# NOT import crispdm_admission itself (so it cannot mask the fault).  It is also the cheapest such
# file: 17 tests in about four seconds.
PROBE = "tests/test_df_d2_r4_comparator_guard.py"

# The exact failure the repair removes.
SIGNATURE = "AttributeError: 'NoneType' object has no attribute '__dict__'"


def _run_pytest_on(target: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = ""          # do not inherit the parent run's options
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900, env=env,
    )


def test_a_covered_file_passes_when_it_is_run_on_its_own():
    """The regression itself: one covered file, its own process, nothing ahead of it."""
    if not (REPO / PROBE).is_file():
        pytest.skip(f"the probe file {PROBE} is not present in this checkout")
    if "crispdm_uncovered" in (REPO / PROBE).read_text(encoding="utf-8"):
        pytest.skip(f"{PROBE} now opts out of declared coverage; it can no longer probe the fixture")

    result = _run_pytest_on(PROBE)
    out = result.stdout + result.stderr
    tail = "\n".join(out.strip().splitlines()[-25:])

    assert SIGNATURE not in out, (
        "the declared-coverage fixture broke a file run on its own: it executed "
        "tools/crispdm_admission.py without registering it in sys.modules, so dataclasses "
        f"resolved the module to None.\n--- last lines of the child run ---\n{tail}"
    )
    assert not re.search(r"\b\d+ errors?\b", out), (
        f"{PROBE} reported setup errors when run on its own.\n"
        f"--- last lines of the child run ---\n{tail}"
    )
    passed = re.search(r"(\d+) passed", out)
    assert passed and int(passed.group(1)) > 0, (
        f"{PROBE} ran no passing test on its own.\n"
        f"--- last lines of the child run ---\n{tail}"
    )
    assert result.returncode == 0, (
        f"{PROBE} exited {result.returncode} on its own.\n"
        f"--- last lines of the child run ---\n{tail}"
    )


def test_the_fixture_registers_the_admission_module_under_its_own_name():
    """The mechanism, checked directly, so a failure names the cause and not just the symptom.

    The autouse fixture has already run for THIS test, so the name must be bound to a module whose
    own `__name__` agrees -- which is the condition `dataclasses` relies on.
    """
    module = sys.modules.get("crispdm_admission")
    assert module is not None, (
        "sys.modules['crispdm_admission'] is unset after the declared-coverage fixture ran; "
        "dataclasses cannot resolve the module's annotations and every @dataclass in it will "
        "raise at import in any process that does not import it some other way first"
    )
    assert module.__name__ == "crispdm_admission"
    # the module really did execute: the dataclass that used to raise is present and usable
    assert hasattr(module, "Lease") and hasattr(module, "Request")


def test_a_file_that_opts_out_of_coverage_is_still_left_alone():
    """The repair must not quietly widen the fixture: an opted-out file still skips it."""
    guard = REPO / "tests/test_df_admission_guard.py"
    if not guard.is_file():
        pytest.skip("tests/test_df_admission_guard.py is not present in this checkout")
    assert "crispdm_uncovered" in guard.read_text(encoding="utf-8"), (
        "the opt-out marker this fixture honours has disappeared from the file whose subject is "
        "the ABSENCE of a covering reservation"
    )
