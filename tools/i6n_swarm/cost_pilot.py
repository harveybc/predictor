"""One immutable, admitted TRAIN pilot; no implicit retry or campaign ranking.

Run with ``python -m tools.i6n_swarm.cost_pilot --config <private.json>``.
Read ``--status`` without importing TensorFlow or launching any child.
The measured peak belongs to this exact workload, not every model in a class.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from tools.i6n_swarm import common, runner


def retained_peak(ledger, label):
    """Read only a released cgroup measurement for this unique admission label."""
    matches = []
    with Path(ledger).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "LEASE_RELEASED" and row.get("label") == label:
                if row.get("peak_scope") != "cgroup":
                    raise ValueError("pilot peak is not a cgroup measurement")
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one released measurement, found {len(matches)}")
    peak = matches[0].get("observed_tree_peak_bytes")
    if type(peak) is not int or peak <= 0:
        raise ValueError("missing positive measured peak")
    return peak, matches[0]


def execute(path):
    cfg = json.loads(Path(path).read_text())
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = common.file_sha256(path)
        status_path = root / "STATUS.json"
        if status_path.exists():
            old = json.loads(status_path.read_text())
            if old["config_sha256"] != digest:
                raise ValueError("pilot configuration changed")
            if old["state"] == "COMPLETED":
                report = common.read_pinned(old["result"], "completed pilot")
                return report
            raise ValueError("existing attempt requires explicit reviewed recovery; not retraining")
        started = time.time()

        def status(state, **extra):
            body = dict(state=state, config_sha256=digest, started_at=started,
                        updated_at=time.time(), elapsed_seconds=time.time()-started,
                        eta_seconds=None, **extra)
            common.atomic_json(status_path, body, mode=0o600)

        status("PREPARING")
        try:
            cell = common.read_pinned(cfg["cell"], "pilot cell")
            if cell["evaluation_split"]["split_id"].startswith("STRUCTURAL_UNSCORED_") is False:
                raise ValueError("pilot requires a TRAIN-only structural fold")
            binding_root = root / "binding"
            freeze = [cfg["python"], "-m", "tools.i6n_historical_evaluator", "freeze",
                      "--config", cfg["cell"]["path"], "--output", str(binding_root),
                      "--experiment", cfg["experiment"], "--fold", cfg["fold"]]
            status("FREEZING")
            freeze_launcher = dict(cfg["launcher"], memory_bytes=2*1024**3, wall_seconds=300)
            env = dict(os.environ, **cfg.get("environment", {}))
            result = runner.run_attempt(runner.build_argv(freeze_launcher, "freeze", freeze),
                dict(env, CUDA_VISIBLE_DEVICES=""), "", wall_seconds=300,
                renew=lambda p=None: True, renew_interval=5, ttl=60,
                log_dir=root / "freeze_logs", cwd=cfg["workdir"], guard=True)
            if result["kind"] != "OK":
                raise RuntimeError(f"freeze failed: {result}")
            binding = json.loads((binding_root / "BINDING.json").read_text())
            request = dict(config=cfg["effective_config"], generation=0,
                           context=binding["i6n_evaluation_context"], support=binding["i6n_support"])
            common.atomic_json(root / "REQUEST.json", request, mode=0o600)
            status("RUNNING", binding=str(binding_root / "BINDING.json"))
            child = [cfg["python"], "-m", "tools.i6n_historical_evaluator", "evaluate"]
            argv = runner.build_argv(cfg["launcher"], "fit", child)
            env = runner.device_env(cfg["device"], env)
            result = runner.run_attempt(argv, env, json.dumps(request, allow_nan=False),
                wall_seconds=cfg["launcher"]["wall_seconds"], renew=lambda p=None: True,
                renew_interval=10, ttl=60, log_dir=root / "fit_logs", cwd=cfg["workdir"], guard=True)
            common.atomic_json(root / "ATTEMPT.json", result, mode=0o600)
            if result["kind"] != "OK":
                raise RuntimeError(f"fit failed: {result['kind']}: {result['cause']}")
            peak, lease = retained_peak(cfg["admission_ledger"], cfg["launcher"]["label_prefix"] + "-fit")
            report = dict(state="COMPLETED", config_sha256=digest, code_commit=cfg["code_commit"],
                          scope="EXACT_WORKLOAD_ONLY_NOT_CLASS_ENVELOPE", observed_peak_bytes=peak,
                          production_cap_bytes=(5*peak+3)//4, lease=lease,
                          elapsed_seconds=time.time()-started, request=request, receipt=result["receipt"])
            common.atomic_json(root / "RESULT.json", report, mode=0o600)
            status("COMPLETED", observed_peak_bytes=peak,
                   result=dict(path=str(root / "RESULT.json"), sha256=common.file_sha256(root / "RESULT.json")))
            return report
        except Exception as exc:
            status("FAILED", reason=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        cfg = json.loads(Path(args.config).read_text())
        path = Path(cfg["output"]) / "STATUS.json"
        print(path.read_text() if path.exists() else '{"state":"NOT_STARTED"}')
    else:
        result = execute(args.config)
        print(json.dumps({k: result[k] for k in ("state", "observed_peak_bytes", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
