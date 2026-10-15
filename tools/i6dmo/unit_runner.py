"""Child-side adapter between the ops unit protocol and the lanes' entry points.

``python -m tools.i6dmo.unit_runner --kind <kind> run-unit --spec S --out O --progress P``

- fit / final_refit: runs the model lane's ``python -m tools.i6dmo.train --design --bundle --out
  --device`` once as a subprocess, mirrors its PROGRESS.json (epochs) into P and converts its
  RECEIPT.json (the atomic terminal marker) into ``O/TERMINAL.json``. Exit 2/3/4 of the trainer are
  typed terminals (FAILED with the trainer's code), never retried here. A killed trainer (137/124)
  leaves no terminal, so the worker records FAILED_OOM/FAILED_TIMEOUT.
- report / pair / final_test: delegates to ``run_unit(spec, out_dir, progress_path) -> terminal``
  of the hook module named in HOOKS (owned by another lane). The worker checks the hook is
  importable before claiming, so a missing hook is a WAIT_ARTIFACT, never a burned attempt.
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOOKS = {"report": "tools.i6dmo.reporting", "pair": "tools.i6dmo.strategy_adapter",
         "strategy": "tools.i6dmo.strategy_adapter", "final_test": "tools.i6dmo.final"}
HOOK_OWNERS = {"tools.i6dmo.reporting": "ops lane (tools/i6dmo/reporting.py)",
               "tools.i6dmo.strategy_adapter": "strategy lane",
               "tools.i6dmo.final": "model lane (guarded final TEST path)"}
TRAINER_MODULE = "tools.i6dmo.train"
FIT_KINDS = ("fit", "final_refit")


def _atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hook_module(kind):
    return TRAINER_MODULE if kind in FIT_KINDS else HOOKS[kind]


def _family_design(spec):
    """The design the trainer CLI receives: the family design, or its typed FINAL_REFIT variant.

    A ``final_refit`` unit trains on the PLAN final window (TRAIN 2021-2025 with its inner tail),
    never the fixed validation adapter; the unit's declared window must match the typed protocol
    exactly (FINAL_WINDOW_MISMATCH otherwise, a typed terminal upstream, not an improvised window).
    """
    effective = spec["effective_config"]
    base = effective["long"]["design"] if spec["family"] == "long" else effective["design"]
    if spec.get("kind") != "final_refit":
        return base
    from tools.i6dmo import design as design_module  # model lane typed adapter
    final = design_module.final_refit_design(base)
    declared = [final["splits"]["train"]["start"], final["splits"]["train"]["end"]]
    window = (spec.get("windows") or spec.get("window") or {}).get("train")
    if window is not None and list(window) != declared:
        raise ValueError(f"FINAL_WINDOW_MISMATCH: unit window {window} != typed final window {declared}")
    return final


def _progress_from_trainer(src, dst):
    try:
        doc = json.loads(Path(src).read_text())
    except (OSError, ValueError):
        return
    _atomic(dst, dict(done=doc.get("epoch"), total=doc.get("max_epochs"), unit="epoch", phase="fit",
                      updated_at=doc.get("updated_at", time.time()), best_epoch=doc.get("best_epoch"),
                      monitored=doc.get("monitored")))


def run_fit(spec, out, progress, python=sys.executable, poll=5.0):
    out = Path(out)
    folder = out.parent
    bundle = spec.get("bundle") or {}
    manifest = Path(bundle.get("path", "")) / "BUNDLE.json"
    if not manifest.exists() or _sha(manifest) != bundle.get("sha256"):
        return _terminal(spec, out, "FAILED", "BUNDLE_PIN_MISMATCH", {}, {}, [])
    identity = json.loads(manifest.read_text()).get("bundle_sha256")
    if bundle.get("bundle_sha256") and identity != bundle["bundle_sha256"]:
        return _terminal(spec, out, "FAILED", "BUNDLE_PIN_MISMATCH", {}, {}, [])
    design_path = folder / "design.json"
    _atomic(design_path, _family_design(spec))
    train_out = out / "train"
    device = "gpu" if (spec.get("device") or {}).get("gpu_uuid") else "cpu"
    argv = [python, "-m", TRAINER_MODULE, "--design", str(design_path), "--bundle", str(manifest.parent),
            "--out", str(train_out), "--device", device, "--json"]
    started = time.time()
    with open(folder / "trainer.stdout", "ab") as so, open(folder / "trainer.stderr", "ab") as se:
        proc = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            _progress_from_trainer(train_out / "PROGRESS.json", progress)
            time.sleep(poll)
    _progress_from_trainer(train_out / "PROGRESS.json", progress)
    rc = proc.returncode
    receipt_path = train_out / "RECEIPT.json"
    if rc in (137, -9, 124) or (rc not in (0, 2, 3, 4)):
        return rc  # killed: no terminal; the worker types it (FAILED_OOM/FAILED_TIMEOUT/FAILED)
    if not receipt_path.exists():
        code = "CONTRACT_ERROR" if rc == 2 else f"NO_RECEIPT_RC_{rc}"
        return _terminal(spec, out, "FAILED", code, {}, dict(wall_seconds=time.time() - started), [])
    receipt = json.loads(receipt_path.read_text())
    status = "COMPLETE" if receipt.get("status") == "COMPLETED" and rc == 0 else "FAILED"
    artifacts = {rel: dict(path=f"train/{rel}", sha256=sha) for rel, sha in (receipt.get("artifacts") or {}).items()}
    artifacts["RECEIPT"] = dict(path="train/RECEIPT.json", sha256=_sha(receipt_path))
    timings, resources = receipt.get("timings") or {}, receipt.get("resources") or {}
    costs = dict(wall_seconds=time.time() - started, fit_seconds=timings.get("fit_s"),
                 score_seconds=timings.get("predict_s"), cgroup_peak_bytes=resources.get("cgroup_memory_peak_bytes"),
                 tf_gpu_memory_peak_bytes=resources.get("tf_gpu_memory_peak_bytes"),
                 max_rss_kib=resources.get("max_rss_kib"), cpu_seconds=(resources.get("cpu_user_s") or 0)
                 + (resources.get("cpu_sys_s") or 0), gpu_uuid=(spec.get("device") or {}).get("gpu_uuid"),
                 device=resources.get("device"))
    training = receipt.get("training") or {}
    reports = [_cost_report(spec, receipt, costs, training)] if status == "COMPLETE" else []
    return _terminal(spec, out, status, receipt.get("code") or receipt.get("status"), artifacts, costs, reports,
                     extra=dict(trainer_identity=receipt.get("identity"), training=training,
                                scientific=receipt.get("scientific")))


def _cost_report(spec, receipt, costs, training):
    def metric(name, value, unit):
        return dict(metric=name, value=value, split="train", horizon=None, unit=unit)
    metrics = [metric("Fit_Seconds", costs.get("fit_seconds"), "seconds"),
               metric("Wall_Seconds", costs.get("wall_seconds"), "seconds"),
               metric("Epochs_Completed", training.get("epochs_completed"), "count"),
               metric("Best_Epoch", training.get("best_epoch"), "count"),
               metric("Cgroup_Peak", costs.get("cgroup_peak_bytes"), "bytes")]
    return dict(experiment_key=f"i6dmo:{spec['campaign_id']}:{spec['unit_id']}:cost",
                experiment_set_key=f"i6dmo:{spec['campaign_id']}", actor="predictor", lake="LOCAL_RETAINED",
                lineage="UNVERIFIED", config_sha256=spec["identity_sha256"],
                code_commit=spec["effective_config"]["code_commit"], project="predictor", phase="I6DMO",
                tags={"unit": spec["unit_id"], "family": spec["family"], "kind": spec["kind"], "test_read": "false",
                      "cadence": str(spec.get("cadence")), "stop_reason": str(training.get("stop_reason")),
                      "weights_sha256": str((receipt.get("identity") or {}).get("weights_sha256"))},
                metrics=[m for m in metrics if m["value"] is not None])


def _terminal(spec, out, status, disposition, artifacts, costs, reports, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
               status=status, disposition=disposition, artifacts=artifacts, costs=costs, eligibility={}, gates={},
               reports=reports, **(extra or {}))
    _atomic(out / "TERMINAL.json", doc)
    print(json.dumps(dict(terminal_sha256=_sha(out / "TERMINAL.json"))))
    return 0


def run_hook(kind, spec, out, progress):
    module = importlib.import_module(HOOKS[kind])
    doc = module.run_unit(spec, Path(out), Path(progress))
    doc = dict(doc, schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"])
    Path(out).mkdir(parents=True, exist_ok=True)
    _atomic(Path(out) / "TERMINAL.json", doc)
    print(json.dumps(dict(terminal_sha256=_sha(Path(out) / "TERMINAL.json"))))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.unit_runner")
    parser.add_argument("--kind", required=True,
                        choices=("fit", "final_refit", "report", "pair", "strategy", "final_test"))
    parser.add_argument("command", choices=("run-unit",))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    if spec.get("kind") != args.kind:
        print("KIND_MISMATCH", file=sys.stderr)
        return 2
    if spec.get("test_open") and args.kind != "final_test":
        print("TEST_OPEN_OUTSIDE_FINAL_TEST", file=sys.stderr)
        return 2
    if args.kind in FIT_KINDS:
        return run_fit(spec, args.out, args.progress)
    return run_hook(args.kind, spec, args.out, args.progress)


if __name__ == "__main__":
    sys.exit(main())
