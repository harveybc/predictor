"""The worker pin is a file. It does not import TensorFlow and it does not name a host."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "docs/audits/evidence/GPU_ENV_REPAIR_20260930/pins-worker-prep.txt"
PREP = ROOT / "docs/audits/evidence/GPU_ENV_REPAIR_20260930/WORKER_PREP.json"
DOC = ROOT / "docs/audits/work_plan/RETSU_GPU_WORKER_VENV_2026_09_30.md"


def test_pin_is_the_tested_build_floor_and_names_no_host():
    text = PIN.read_text() + PREP.read_text() + DOC.read_text()
    assert "/home/" not in text
    assert os.uname().nodename not in text
    assert "CUDA_VISIBLE_DEVICES=" not in DOC.read_text()
    assert ".repair-env/bin" not in DOC.read_text()
    body = [line for line in PIN.read_text().splitlines() if line and not line.startswith("#")]
    assert len(body) == 44
    assert "tensorflow==2.21.0 --hash=sha256:" in PIN.read_text()
    assert "nvidia-cuda-runtime-cu12==12.5.82 --hash=sha256:" in PIN.read_text()
    assert "nvidia-cudnn-cu12==9.3.0.75 --hash=sha256:" in PIN.read_text()
    record = json.loads(PREP.read_text())
    assert record["diagnostic"] == "NOT_RUN"
    assert record["calibrations"] == "NOT_RUN"
    assert record["procedure"] == "WRITTEN_NOT_RUN"
    assert record["venv_created"] is False
    assert record["cuda_12_5_versus_cuda_13_soname_clash"] == "NOT_DEMONSTRATED"
    assert record["refusal_retained"]["missing_soname"] == "NOT_RETAINED"
    assert record["coordinator_prefix_filename_inventory"]["libcudart_so_13_present"] is False
    assert record["preparation_pin"]["wheel_bytes"] == 2576641397
    preferred = [item for item in record["occupancy"]["devices"]
                 if item["uuid"] == "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"]
    assert preferred[0]["pin_applied"] is False
    assert preferred[0]["inside_declared_compute_list"] is False
    assert ".worker-diag-env/bin/python" in DOC.read_text()
