"""No allocation and no real subprocess: two callers can spend the same headroom."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import patch


def reproduce(path):
    spec = importlib.util.spec_from_file_location("audited_gate", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    barrier = Barrier(2)
    lock = Lock()
    calls = []

    def fake_run(cmd, **kwargs):
        with lock:
            calls.append(cmd)
        barrier.wait(timeout=5)
        return SimpleNamespace(returncode=0)

    with TemporaryDirectory() as tmp, patch.object(gate, "mem_available_bytes", return_value=12 << 30), \
            patch.object(gate.subprocess, "run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
        def submit(i):
            return gate.main(["--root", str(Path(tmp) / str(i)), "--label", str(i),
                              "--peak-bytes", str(7 << 30), "--cap", "8G", "--cap-bytes", str(8 << 30),
                              "--wall", "1", "--", "NEVER_EXECUTED"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(submit, [1, 2]))
    return {"real_processes_started": 0, "mock_mem_available_gib": 12,
            "requested_gib_per_caller": 8, "aggregate_requested_gib": 16,
            "launches_accepted_before_either_finished": len(calls), "return_codes": codes,
            "claim": "The new wrapper does not serialize or reserve admission across callers"}


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[4] / "predictor-q2deep-20260926/tools/df_memory_gated_run.py"
    print(json.dumps(reproduce(target), indent=2))
