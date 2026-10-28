"""Actual temporary Git worktree bytes, not mocked same-HEAD assertions."""
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from tools.i6dmo import raw_long_campaign as L


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    content = {"tools/__init__.py": "", "tools/i6dmo/__init__.py": "",
               "tools/i6dmo/model.py": "VALUE = 1\n", ".gitignore": "__pycache__/\n"}
    for rel, text in content.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    files = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
             for rel in content if rel.endswith(".py")}
    monkeypatch.setattr(L, "verify_release", lambda: dict(code_files=files))
    return root, git("rev-parse", "HEAD"), dict(python=sys.executable)


def test_actual_clean_runtime_verifies_bytes_and_imports(runtime):
    root, commit, role = runtime
    assert L._verify_runtime(role, str(root), commit) == dict(verified_files=3, code_commit=commit)


def test_same_head_dirty_source_is_rejected_without_reset(runtime):
    root, commit, role = runtime
    model = root / "tools/i6dmo/model.py"
    model.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="RUNTIME_DIRTY"):
        L._verify_runtime(role, str(root), commit)
    assert model.read_text() == "VALUE = 2\n"


def test_ignored_import_override_is_rejected(runtime):
    root, commit, role = runtime
    extra = root / "tools/i6dmo/__pycache__/unexpected.pyc"
    extra.parent.mkdir()
    extra.write_bytes(b"not an allowed source artifact")
    with pytest.raises(RuntimeError, match="RUNTIME_IMPORT_OVERRIDE"):
        L._verify_runtime(role, str(root), commit)


def test_committed_bytes_must_match_review_manifest(runtime):
    root, _, role = runtime
    model = root / "tools/i6dmo/model.py"
    model.write_text("VALUE = 2\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "changed"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(RuntimeError, match="RUNTIME_BYTES"):
        L._verify_runtime(role, str(root), commit)
