"""Finite, resumable CPU verification of I6N integration components.

Run: ``python -m tools.i6n_verify run --output <artifact-directory>``.
Query once: use ``status`` instead of ``run``. Each suite uses the existing
admission launcher (2 GiB, 120 s); no monitoring agent or training is needed.
Results bind code, environment, commands, logs and JUnit reports. They are
engineering evidence, NOT market measurements or permission to run N1.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from tools.i6m_queue import atomic_json

BASE = Path(__file__).resolve().parents[1]
STEPS = (
    ("contracts", ("tests/test_i6n_eligibility.py", "tests/test_i6n_paired_metrics.py",
                   "tests/test_i6n_paired_metrics_r3.py", "tests/test_i6n_paired_metrics_r4.py")),
    ("population_bridge", ("tests/test_i6n_business_metrics.py",)),
    ("temporal", ("tests/test_i6n_temporal_ops.py",)),
    ("model_factory", ("tests/test_i6n_model.py",)),
    ("weekly_data", ("tests/test_i6n_weekly_data.py",)),
    ("masked_training", ("tests/test_i6n_training.py",)),
    ("joint_search", ("tests/test_i6n_search.py",)),
    ("source_recovery", ("tests/test_i6n_source_recovery.py",)),
    ("weekly_integration", ("tests/test_i6n_weekly_smoke.py",)),
    ("automation", ("tests/test_i6n_verify.py",)),
)
SCOPE = "INTEGRATED_CANDIDATE_COMPONENTS_ONLY"


def file_hash(path):
    """Hash one artifact without loading it all into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity():
    """Bind direct code, reused helpers, tests and installed package versions."""
    paths = {Path(__file__), BASE / "tools/i6m_queue.py", BASE / "tools/i6e_compact_run.py",
             BASE / "tools/i6e_masked_model.py", BASE / "tools/i6n_model.py"}
    paths.update(BASE / "tools" / name for name in (
        "i6n_weekly_data.py", "i6n_training.py", "i6n_search.py",
        "i6n_source_recovery.py", "i6n_weekly_smoke.py", "i6e_clock_windows.py",
        "modular_candidate_evaluator.py"))
    paths.update((BASE / "tools/i6n_contracts").glob("*.py"))
    paths.update((BASE / "predictor_plugins/modular_temporal").glob("*.py"))
    paths.update(BASE / name for _, tests in STEPS for name in tests)
    versions = {}
    for package in ("numpy", "tensorflow", "keras", "pytest", "deap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    body = {"files": {str(p.relative_to(BASE)): file_hash(p) for p in sorted(paths)},
            "python": sys.version, "packages": versions, "steps": STEPS}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def junit_count(path):
    """Require actual executed tests, with no failure, error or silent skip."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, "0")) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if not counts["tests"] or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise ValueError(f"unaccepted JUnit counts: {counts}")
    return counts["tests"]


def _verify_step(root, row):
    name = row["name"]
    if name not in dict(STEPS) or row["state"] != "COMPLETED":
        raise ValueError("unknown or incomplete verification step")
    for kind in ("log", "xml"):
        if file_hash(root / f"{name}.{kind}") != row[f"{kind}_sha256"]:
            raise ValueError(f"changed {name}.{kind}")
    if junit_count(root / f"{name}.xml") != row["tests"]:
        raise ValueError("test count differs from retained receipt")


def status(root, expected):
    """Read-only check against independently calculated expected code identity."""
    root = Path(root)
    common = {"scope": SCOPE, "campaign_ready": False, "scientific_measurement": False}
    if not (root / "STATUS.json").exists():
        return dict(common, state="NOT_RUN"), 3
    try:
        value = json.loads((root / "STATUS.json").read_text())
        if value["identity"] != expected:
            return dict(common, state="IDENTITY_MISMATCH"), 2
        for row in value.get("steps", []):
            _verify_step(root, row)
        names = [r["name"] for r in value.get("steps", [])]
        if names != [n for n, _ in STEPS[:len(names)]]:
            raise ValueError("steps must be an ordered prefix")
        if value["state"] == "COMPLETED":
            if [r["name"] for r in value["steps"]] != [n for n, _ in STEPS]:
                raise ValueError("missing/duplicate/out-of-order steps")
            return dict(value, **common), 0
        return dict(value, **common), 4
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        return dict(common, state="DAMAGED", error=str(exc)), 6


def run(root):
    """Run all suites in order, reusing only verified prior engineering steps."""
    root = Path(root).resolve()
    expected = identity()
    prior, code = status(root, expected)
    if code == 0:
        return prior, 0
    if code in (2, 6):
        return prior, code
    launcher = shutil.which("crispdm-run")
    if not launcher:
        return {"state": "FAILED", "error": "ADMISSION_LAUNCHER_MISSING"}, 2
    if shutil.disk_usage(root if root.exists() else root.parent).free < 512 * 1024**2:
        return {"state": "FAILED", "error": "INSUFFICIENT_ARTIFACT_SPACE"}, 2
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "RUNNING", "error": "VERIFIER_ALREADY_RUNNING"}, 4
        # Re-read under the lock; another invocation may have just completed.
        prior, code = status(root, expected)
        if code in (0, 2, 6):
            return prior, code
        value = {"schema": "predictor.i6n.verification.v1", "identity": expected,
                 "scope": SCOPE, "campaign_ready": False, "scientific_measurement": False,
                 "state": "RUNNING", "steps": prior.get("steps", []),
                 "started_at_epoch": time.time(), "eta_seconds": None}
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
                           TF_NUM_INTEROP_THREADS="1")
        started = time.monotonic()
        try:
            for name, tests in STEPS[len(value["steps"]):]:
                if identity() != expected:
                    raise ValueError("CODE_CHANGED_DURING_VERIFICATION")
                value.update(current_step=name, updated_at_epoch=time.time())
                atomic_json(root / "STATUS.json", value)
                command = [launcher, "-m", "2G", "-t", "120s", "-n", f"i6n-verify-{name}",
                           "--", sys.executable, "-m", "pytest", "-q", *tests,
                           f"--junitxml={root / (name + '.xml')}"]
                step_started = time.monotonic()
                with (root / f"{name}.log").open("w") as log:
                    process = subprocess.run(command, cwd=BASE, env=environment,
                                             stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise ValueError(f"{name}: verification exited {process.returncode}")
                row = {"name": name, "state": "COMPLETED", "command": command,
                       "tests": junit_count(root / f"{name}.xml"),
                       "elapsed_seconds": time.monotonic() - step_started,
                       "log_sha256": file_hash(root / f"{name}.log"),
                       "xml_sha256": file_hash(root / f"{name}.xml")}
                value["steps"].append(row)
                atomic_json(root / "STATUS.json", value)
            if identity() != expected:
                raise ValueError("CODE_CHANGED_DURING_VERIFICATION")
            value.update(state="COMPLETED", current_step=None, eta_seconds=0,
                         elapsed_seconds=time.monotonic() - started, updated_at_epoch=time.time())
            atomic_json(root / "STATUS.json", value)
            return value, 0
        except (OSError, ValueError) as exc:
            value.update(state="FAILED", error=str(exc), updated_at_epoch=time.time())
            atomic_json(root / "STATUS.json", value)
            return value, 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        value, code = (run(args.output) if args.command == "run"
                       else status(args.output, identity()))
    except OSError as exc:
        value, code = {"state": "INCOMPLETE_CODE", "error": str(exc)}, 2
    print(json.dumps(value, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
