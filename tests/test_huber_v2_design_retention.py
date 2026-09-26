"""Retention pin for the Huber/AdamW v2 frozen design (MOD-CONF prerequisite).

The four-arm loss/optimizer factorial of 2026-09-21 rests on design digest
be2e776e...  The design lived only in a private run root and was not retained in
the repository, which blocked MOD-CONF.  It was recovered on 2026-09-26 and
committed next to the run's REPORT.json.  These tests pin that retention: the
bytes must stay present, must stay byte-identical, and the digest must keep
re-deriving under the corpus's own rule (df_mod_e0.sha_obj over the design
object minus its own design_sha256 field).

They are deliberately self-contained -- no TensorFlow import, no run root -- so
that a missing or altered design fails fast in any environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "audits" / "evidence" / "HUBER_ADAMW_2026_09_21"
DESIGN = EVIDENCE / "DESIGN.json"
REPORT = EVIDENCE / "REPORT.json"

# The digest the four arms were executed under, as published in
# docs/audits/work_plan/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md.
FROZEN_DESIGN_SHA256 = "be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965"

# sha256 of the recovered raw file bytes, recorded so that a re-serialization
# that happens to preserve the canonical digest is still visible as a change.
RECOVERED_BYTES_SHA256 = "cdd1611c41d7822169e481e585278ab9c5f59e68ab049c6bf411b01f303599a1"

# Execution revision named by the results work plan; the design's source_code
# digests are pinned against that revision, not against a moving HEAD.
EXECUTION_REVISION = "73f3bab"


def _sha_obj(obj) -> str:
    """The corpus's own design-digest rule, restated locally.

    Identical to tools/df_mod_e0.sha_obj; duplicated on purpose so this pin does
    not depend on the scientific source it is meant to outlive.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def test_frozen_design_is_retained():
    assert DESIGN.is_file(), (
        f"the frozen Huber/AdamW v2 design is not retained at {DESIGN.relative_to(ROOT)}; "
        "it blocks MOD-CONF and must not be deleted"
    )


def test_retained_bytes_are_unchanged():
    assert hashlib.sha256(DESIGN.read_bytes()).hexdigest() == RECOVERED_BYTES_SHA256


def test_digest_rederives_under_the_corpus_rule():
    design = json.loads(DESIGN.read_text())
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    assert _sha_obj(body) == FROZEN_DESIGN_SHA256
    assert design["design_sha256"] == FROZEN_DESIGN_SHA256


def test_local_digest_rule_agrees_with_the_shipped_helper():
    """Guard against the pin drifting away from df_mod_e0.sha_obj."""
    path = ROOT / "tools" / "df_mod_e0.py"
    if not path.is_file():
        pytest.skip("df_mod_e0.py not present")
    spec = importlib.util.spec_from_file_location("_df_mod_e0_pin", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment-dependent import
        pytest.skip(f"df_mod_e0 not importable here: {exc}")
    design = json.loads(DESIGN.read_text())
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    assert module.sha_obj(body) == _sha_obj(body) == FROZEN_DESIGN_SHA256


def test_design_is_the_four_arm_factorial_the_report_scored():
    design = json.loads(DESIGN.read_text())
    assert design["schema"] == "df_e1_huber_design.v1"
    assert design["purpose"] == "HUBER_ADAMW_FACTORIAL_DEVELOPMENT"
    assert design["arms"] == ["mae_adam", "mae_adamw", "huber_adam", "huber_adamw"]
    assert design["factors_moved"] == ["loss", "optimizer"]
    assert [c["cell_id"] for c in design["cells"]] == [
        f"{arm}_s{seed}" for seed in (1, 2, 3) for arm in design["arms"]
    ]


def test_report_binds_to_the_retained_design():
    assert REPORT.is_file()
    assert json.loads(REPORT.read_text())["design_sha256"] == FROZEN_DESIGN_SHA256


def test_source_code_binding_is_pinned_to_the_execution_revision():
    """The design seals five scientific sources; they are pinned to 73f3bab.

    Skipped where git history is unavailable.  This does NOT assert anything
    about current HEAD: the sources have moved on since, which is exactly why
    the binding is checked against the recorded execution revision.
    """
    import subprocess

    design = json.loads(DESIGN.read_text())
    sealed = design["source_code"]
    assert sorted(sealed) == [
        "df_e1_governed.py",
        "df_e1_huber.py",
        "df_e1_phase1.py",
        "df_e1_pilot.py",
        "df_mod_e0.py",
    ]
    probe = subprocess.run(
        ["git", "cat-file", "-t", EXECUTION_REVISION],
        cwd=ROOT, capture_output=True, text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "commit":
        pytest.skip(f"execution revision {EXECUTION_REVISION} not reachable here")
    for name, digest in sorted(sealed.items()):
        blob = subprocess.run(
            ["git", "show", f"{EXECUTION_REVISION}:tools/{name}"],
            cwd=ROOT, capture_output=True,
        )
        assert blob.returncode == 0, f"{name} missing at {EXECUTION_REVISION}"
        assert hashlib.sha256(blob.stdout).hexdigest() == digest, (
            f"{name} at {EXECUTION_REVISION} does not match the digest the design sealed"
        )
