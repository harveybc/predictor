"""Bounded CPU review of actual candidate APIs; all writes use temporary roots.

Run with M5PHET's candidate src on PYTHONPATH. Optional --output records results.
No model, broker, network, production service or scientific array is touched.
"""
import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from m5phet.evidence import open_run
from m5phet.runtime import Registry, REQUEST_SCHEMA, run


class Provider:
    name = "probe"

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return {"operations": ["infer", "calibrate"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED"],
                "known_states": ["state-1"]}

    def load(self, ref):
        self.calls.append("load")
        return {"digest": "a" * 64}

    def infer(self, request, state):
        self.calls.append("infer")
        return {"outputs": {"tone": {"status": "OK", "payload": None, "uncertainty": "UNCALIBRATED"}}}

    def calibrate(self, request, state):
        return {"calibration_ref": "other-task", "population": {"rows": 5},
                "clocks": {"calibration_end": "2099-01-01T00:00:00Z"},
                "state_digest": "b" * 64, "task_id": "other-task"}


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--predictor", required=True)
    args.add_argument("--revision", default="5c2dc332")
    args.add_argument("--output")
    opts = args.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="m5phet-review-") as temporary:
        root = Path(temporary)
        kwargs = {"run_id": "local", "task": {"id": "A"}, "code_identity": {"rev": "A"}}
        original = open_run(root, **kwargs)
        pending = original.start_attempt(candidate="candidate-A")
        resumed = open_run(root, **{**kwargs, "task": {"id": "B"}, "code_identity": {"rev": "B"}}, resume=True)
        results["resume_changed_identity"] = {"requested_task": "B", "retained_task": resumed.manifest["task"],
                                               "same_digest": original.identity_sha256 == resumed.identity_sha256}
        resumed.recover()
        results["resume_loses_pending_attempt"] = {
            "pending_attempt_exists": bool(pending), "reported_open": resumed.close()["open_attempts_at_close"]}
        metric = {"metric": "mae", "definition": "mean absolute error", "definition_version": "1",
                  "unit": "1", "scale": "z", "aggregation": "mean", "task": "A", "split": "test",
                  "population": True, "status": "OK", "value": float("nan")}
        resumed.record_metric("unknown-attempt", metric)
        resumed.record_metric("unknown-attempt", metric)
        resumed.finish_attempt("unknown-attempt", status="COMPLETED")
        lines = (resumed.directory / "metrics.jsonl").read_text().splitlines()
        results["invalid_duplicate_metrics"] = {"rows": len(lines), "nonfinite_serialized": "NaN" in lines[0],
                                               "boolean_population": json.loads(lines[0])["population"],
                                               "unknown_attempt_finished": True}
        attempts_path = resumed.directory / "attempts.jsonl"
        old = attempts_path.read_text().splitlines()
        attempts_path.write_text(old[0] + "\n{broken-interior-record\n" + "\n".join(old[1:]) + "\n")
        recovery = resumed.recover()
        results["recovery_erases_interior_corruption"] = {
            "report": recovery, "corrupt_bytes_preserved_in_log": "broken-interior" in attempts_path.read_text()}
        denied = open_run(root, run_id="denied", task={"id": "A"}, code_identity={"rev": "A"},
                          governed=True, deliveries={"x": "resource"},
                          delivery_resolver=lambda _: {"status": "DENIED"})
        results["denied_receipt_accepted"] = {"authority": denied.authority, "directory_created": denied.directory.exists()}
        no_resolver = open_run(root, run_id="denied", task={"id": "B"}, code_identity={"rev": "B"}, resume=True)
        results["governed_resume_without_resolver"] = {"authority": no_resolver.authority}
    provider = Provider()
    registry = Registry()
    registry.register(provider)
    request = {"schema_version": REQUEST_SCHEMA, "request_id": "req", "task_id": "task",
               "operation": "infer", "family": "classification", "output_kind": "typed_questions",
               "as_of": "2026-09-24T00:00:00Z", "provider_ref": "probe", "fitted_state_ref": "state-1",
               "output_schema": {"questions": ["tone"]}}
    result = run(request, registry)
    results["empty_payload_success"] = {"status": result["status"], "facts": result["facts"], "outputs": result["outputs"]}
    provider.calls.clear()
    result = run({k: v for k, v in request.items() if k != "output_schema"}, registry)
    results["invalid_request_does_work"] = {"status": result["status"], "calls": list(provider.calls)}
    result = run({**request, "operation": "calibrate"}, registry)
    results["foreign_future_calibration_bound"] = {"status": result["status"], "facts": result["facts"]}
    class Entry:
        name = "probe"

        def load(self):
            return Provider
    with patch("importlib.metadata.entry_points", return_value=[Entry()]):
        results["class_entrypoint"] = Registry().load_entry_points()
    source = subprocess.check_output(["git", "-C", opts.predictor, "show",
                                      f"{opts.revision}:tools/df_ecl_modular.py"], text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "regime_checks")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-regime-checks", "exec"), namespace)
    cells = {"AE_s1": {}, "R0_s1": {"steps": 1, "observed_updates": 1},
             "R1_s1": {"donor": "d", "steps": 1, "observed_updates": 1, "detector_unchanged_by_the_fit": True},
             "R2_s1": {"donor": "d", "steps": 1, "observed_updates": 1}}
    results["missing_R2_evidence"] = namespace["regime_checks"](cells, [1])
    cells["R0_s999"] = dict(cells["R0_s1"])
    results["extra_regime_cell"] = namespace["regime_checks"](cells, [1])
    results["scope"] = {"m5phet_revision": "adba176", "predictor_revision": opts.revision,
                        "predictor_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "regime_probe": "exact AST function, no rewritten implementation",
                        "provider_probe": "recording double through actual runtime and entrypoint discovery",
                        "not_tested": ["live services", "GPU", "new scientific training", "DuckDB projection"]}
    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if opts.output:
        Path(opts.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
