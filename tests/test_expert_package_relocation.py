import importlib.util
from pathlib import Path
import sys

import pytest


def module():
    root = Path(__file__).parents[1] / "tools"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("relocation", root / "prepare_expert_package_relocation.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w


def source(w, root):
    (root / "data").mkdir()
    f = root / "data/x"
    f.write_bytes(b"pinned")
    cfg = {"root": "/old", "pins": {"/old/data/x": w.digest(f)}, "runtime_versions": {"torch": "same"}}
    w.atomic(root / "source/CONFIG.private.json", cfg)
    w.atomic(root / "source/CLOSE.json", {"state": "COMPLETE_WITH_FAILURES", "config_sha256": w.identity(cfg)})
    w.atomic(root / "source/JOBS.json", {"cons_shortlist": {"state": "FAILED"}})


def test_relocation_rejects_changed_science_and_runtime(tmp_path):
    w = module()
    source(w, tmp_path)
    assert w.validate_source(tmp_path, {"torch": "same"})["verified_input_files"] == 1
    with pytest.raises(ValueError, match="RUNTIME"):
        w.validate_source(tmp_path, {"torch": "other"})
    (tmp_path / "data/x").write_bytes(b"different")
    with pytest.raises(ValueError, match="INPUT"):
        w.validate_source(tmp_path, {"torch": "same"})


def test_relocation_cannot_repeat_a_completed_arm(tmp_path):
    w = module()
    source(w, tmp_path)
    w.atomic(tmp_path / "source/JOBS.json", {"cons_shortlist": {"state": "DONE"}})
    with pytest.raises(ValueError, match="FAILED_ARM"):
        w.validate_source(tmp_path, {"torch": "same"})
