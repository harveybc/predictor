"""Wrapper of tests/i6dmo/ops_matched_fake.py that first sleeps RETSU_SLOW_SECONDS (orchestration only)."""
import os
import runpy
import sys
import time
from pathlib import Path

time.sleep(float(os.environ.get("RETSU_SLOW_SECONDS", "0")))
sys.argv[0] = str(Path(__file__).resolve().parents[5] / "tests/i6dmo/ops_matched_fake.py")
runpy.run_path(sys.argv[0], run_name="__main__")
