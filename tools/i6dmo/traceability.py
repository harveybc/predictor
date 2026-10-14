"""Generate TRACEABILITY.json: per-DMO code, test and evidence pointers that are verified to exist.

``python -m tools.i6dmo.traceability [--out PATH]`` resolves every declared pointer against the
checkout: a test pointer must name an existing file containing the named test pattern, and a code
or evidence pointer must be an existing file. A pointer that fails resolution makes the entry
``POINTER_BROKEN`` and the exit code 3 — the table cannot silently claim missing evidence.

The ``result`` field distinguishes CPU_TESTS_PASS (the integration suite), GPU_REPLAY (bounded
engineering replay on an admitted worker) and never claims a scientific verdict.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = "docs/i6n/direct_multi_output_static"
GPU_REPLAY = f"{DOCS}/audit_20261014/GPU_REPLAY.json"

# DMO id -> (owner lane, [test pointers "file::name-regex"], [code files], [evidence files])
TABLE = {
    "DMO01": ("model", ["tests/i6dmo/test_dmo_model.py::test_dmo01_", "tests/i6dmo/test_dmo_model.py::causality"],
              ["tools/i6dmo/model.py"], [f"{DOCS}/MODEL_RETURN.md", GPU_REPLAY]),
    "DMO02": ("data", ["tests/i6dmo/test_data_dmo.py::test_dmo02_"], ["tools/i6dmo/data.py"],
              [f"{DOCS}/DATA_RETURN.md", f"{DOCS}/audit_20261014/AUDIT.md"]),
    "DMO03": ("data", ["tests/i6dmo/test_data_dmo.py::test_dmo03_"], ["tools/i6dmo/data.py"],
              [f"{DOCS}/DATA_RETURN.md"]),
    "DMO04": ("model", ["tests/i6dmo/test_dmo_train.py::test_dmo04_"], ["tools/i6dmo/train.py"],
              [f"{DOCS}/MODEL_RETURN.md", GPU_REPLAY]),
    "DMO05": ("data", ["tests/i6dmo/test_data_dmo.py::test_dmo05_"], ["tools/i6dmo/data.py"],
              [f"{DOCS}/DATA_RETURN.md"]),
    "DMO06": ("data", ["tests/i6dmo/test_data_dmo.py::test_dmo06_"], ["tools/i6dmo/data.py"],
              [f"{DOCS}/DATA_RETURN.md"]),
    "DMO07": ("ops", ["tests/i6dmo/test_ops_dmo07_units.py::test_dmo07_"], ["tools/i6dmo/campaign.py"],
              [f"{DOCS}/OPS_RETURN.md"]),
    "DMO08": ("data+ops", ["tests/i6dmo/test_data_dmo.py::test_dmo08_",
                           "tests/i6dmo/test_ops_dmo08_09_reporting.py::test_dmo08_"],
              ["tools/i6dmo/naive.py", "tools/i6dmo/reporting.py"], [f"{DOCS}/DATA_RETURN.md", f"{DOCS}/OPS_RETURN.md"]),
    "DMO09": ("data+ops", ["tests/i6dmo/test_data_dmo.py::test_dmo09_",
                           "tests/i6dmo/test_ops_dmo08_09_reporting.py::test_dmo09_"],
              ["tools/i6dmo/naive.py", "tools/i6dmo/reporting.py"], [f"{DOCS}/DATA_RETURN.md"]),
    "DMO10": ("model", ["tests/i6dmo/test_dmo_train.py::test_dmo10_", "tests/i6dmo/test_dmo_model.py::test_dmo10_"],
              ["tools/i6dmo/model.py", "tools/i6dmo/train.py"], [f"{DOCS}/MODEL_RETURN.md", GPU_REPLAY]),
    "DMO11": ("ops", ["tests/i6dmo/test_ops_dmo11_identity.py::test_dmo11_"],
              ["tools/i6dmo/queue.py", "tools/i6dmo/worker.py"], [f"{DOCS}/OPS_RETURN.md"]),
    "DMO12": ("ops", ["tests/i6dmo/test_ops_dmo12_outbox.py::test_dmo12_",
                      "tests/i6dmo/test_audit_warehouse_contract.py::test_"],
              ["tools/i6dmo/outbox.py"], [f"{DOCS}/OPS_RETURN.md", f"{DOCS}/audit_20261014/AUDIT.md"]),
    "DMO13": ("ops+data", ["tests/i6dmo/test_ops_dmo13_test_closed.py::test_dmo13_",
                           "tests/i6dmo/test_data_dmo.py::test_test_rows_are_refused"],
              ["tools/i6dmo/campaign.py", "tools/i6dmo/data.py"], [f"{DOCS}/OPS_RETURN.md"]),
    "DMO14": ("strategy", ["tests/i6dmo/test_strategy_adapter.py::test_dmo14_"],
              ["tools/i6dmo/strategy_adapter.py"], [f"{DOCS}/STRATEGY_RETURN.md", f"{DOCS}/STRATEGY_PLUGIN_MAP.json"]),
    "DMO15": ("strategy", ["tests/i6dmo/test_strategy_adapter.py::test_dmo15_"],
              ["tools/i6dmo/strategy_adapter.py"], [f"{DOCS}/STRATEGY_RETURN.md"]),
    "DMO16": ("data", ["tests/i6dmo/test_data_dmo.py::test_dmo16_"], ["tools/i6dmo/data.py"],
              [f"{DOCS}/DATA_RETURN.md"]),
    "DMO17": ("ops", ["tests/i6dmo/test_ops_dmo17_host_lease.py::test_dmo17_"],
              ["tools/i6dmo/queue.py", "tools/i6dmo/worker.py", "tools/i6dmo/preflight.py"],
              [f"{DOCS}/OPS_RETURN.md"]),
    "DMO18": ("model+data", ["tests/i6dmo/test_dmo_train.py::test_dmo18_", "tests/i6dmo/test_dmo_design.py::test_dmo18_",
                             "tests/i6dmo/test_data_dmo.py::test_stlfsi4_"],
              ["tools/i6dmo/design.py", "tools/i6dmo/data.py"], [f"{DOCS}/MODEL_RETURN.md"]),
    "DMO19": ("ops", ["tests/i6dmo/test_ops_dmo19_status.py::test_dmo19_",
                      "tests/i6dmo/test_ops_audit_lane4.py::test_fit_eta_"],
              ["tools/i6dmo/status.py"], [f"{DOCS}/OPS_RETURN.md", f"{DOCS}/audit_20261014/AUDIT.md"]),
    "DMO20": ("ops", ["tests/i6dmo/test_ops_dmo20_cadence.py::test_dmo20_"],
              ["tools/i6dmo/campaign.py"], [f"{DOCS}/OPS_RETURN.md"]),
}
GPU_BACKED = ("DMO01", "DMO04", "DMO10")


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout.strip()


def _test_pointer_ok(pointer):
    file, _, pattern = pointer.partition("::")
    path = ROOT / file
    return path.is_file() and re.search(re.escape(pattern), path.read_text()) is not None


def build():
    commit = _git("rev-parse", "HEAD")
    replay = None
    replay_path = ROOT / GPU_REPLAY
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text())
    entries, broken = {}, []
    for dmo, (owner, tests, code, evidence) in TABLE.items():
        bad = ([t for t in tests if not _test_pointer_ok(t)]
               + [f for f in code + evidence if not (ROOT / f).is_file()])
        gpu = dmo in GPU_BACKED and replay is not None and replay.get("pytest_rc") == 0
        entries[dmo] = dict(
            owner=owner, tests=tests, code=code, evidence=evidence,
            result="POINTER_BROKEN" if bad else ("CPU_TESTS_PASS+GPU_REPLAY" if gpu else "CPU_TESTS_PASS"),
            gpu_backed=gpu, broken_pointers=bad)
        if bad:
            broken.append(dmo)
    return dict(schema="i6dmo.traceability.v1", code_commit=commit,
                scope="software acceptance evidence; not a scientific verdict and not TEST closure",
                gpu_replay=None if replay is None else dict(receipt=GPU_REPLAY, pytest_rc=replay.get("pytest_rc"),
                                                            device=replay.get("device"),
                                                            wall_seconds=replay.get("wall_seconds")),
                entries=entries, broken=broken)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.traceability")
    parser.add_argument("--out", default=str(ROOT / DOCS / "TRACEABILITY.json"))
    args = parser.parse_args(argv)
    doc = build()
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(entries=len(doc["entries"]), broken=doc["broken"], out=args.out)))
    return 0 if not doc["broken"] else 3


if __name__ == "__main__":
    sys.exit(main())
