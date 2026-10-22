"""The deployment inventory must use Git's pinned blobs, not the dirty checkout."""
import ast
from pathlib import Path


def test_installer_uses_commit_blobs_and_reads_both_copies():
    source = (Path(__file__).parents[1] / "tools/install_shared_selection_worker.py").read_text()
    tree = ast.parse(source)
    assert "git\", \"show\", f\"{args.source_commit}:{p}" in source
    assert "own != peer" in source
    assert "SOURCE_IDENTITY_MISMATCH" in source
    assert "timer_enabled" in source
    assert not any(isinstance(n, ast.Attribute) and n.attr in {"fit", "train"} for n in ast.walk(tree))
