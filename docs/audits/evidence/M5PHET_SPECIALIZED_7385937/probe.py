"""Actual candidate APIs with recording doubles; temporary evidence only, no GPU.

Run with candidate src on PYTHONPATH. This characterizes behavior, not ML quality.
"""
import argparse
import json
import tempfile
from pathlib import Path

from m5phet.runtime import REQUEST_SCHEMA, Registry, run
from m5phet.evidence import open_run


class Provider:
    name = "probe"

    def __init__(self):
        self.calls = []
        self.loaded_task = "task"
        self.rows = ["row-1"]
        self.probability = 0.9
        self.calibration = {"calibration_ref": "cal", "population": {"rows": 5},
                            "clocks": {"calibration_end": "2026-01-01T00:00:00Z"}}

    def capabilities(self):
        return {"operations": ["infer", "calibrate"], "families": ["classification", "regression_forecasting"],
                "output_kinds": ["typed_questions", "marginal_quantiles"], "uncertainty_methods": ["UNCALIBRATED"],
                "supported": [{"operation": op, "family": fam, "output_kind": out}
                              for op, fam, out in [("infer", "classification", "typed_questions"),
                                                   ("calibrate", "classification", "typed_questions"),
                                                   ("infer", "regression_forecasting", "marginal_quantiles")]]}

    def load(self, ref):
        self.calls.append("load")
        return {"digest": "a" * 64, "state_ref": ref, "model_sha256": "b" * 64, "task_id": self.loaded_task}

    def infer(self, request, state):
        self.calls.append("infer")
        return {"population": {"ids": self.rows},
                "outputs": {"tone": {"status": "OK", "uncertainty": "UNCALIBRATED",
                                     "payload": {"label": "neutral", "probability": self.probability}}}}

    def calibrate(self, request, state):
        return self.calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    provider, registry = Provider(), Registry()
    registry.register(provider)
    req = {"schema_version": REQUEST_SCHEMA, "request_id": "request", "task_id": "task", "operation": "infer",
           "family": "classification", "output_kind": "typed_questions", "provider_ref": "probe",
           "fitted_state_ref": "state", "as_of": "2026-09-24T00:00:00Z",
           "input_schema": {"population": {"ids": ["row-1"]}}, "output_schema": {"questions": ["tone"]}}
    results = {"revision": "7385937", "scope": "runtime recording doubles and temporary evidence; no live services/model"}
    cross = run({**req, "output_kind": "marginal_quantiles"}, registry)
    results["undeclared_combination"] = {"status": cross["status"], "calls": list(provider.calls)}
    for case in ("valid_baseline", "wrong_loaded_task", "foreign_population", "nonfinite_payload"):
        provider.loaded_task = "different-task" if case == "wrong_loaded_task" else "task"
        provider.rows = ["foreign-row"] if case == "foreign_population" else ["row-1"]
        provider.probability = float("nan") if case == "nonfinite_payload" else 0.9
        out = run(req, registry)
        results[case] = {
            "status": out["status"], "schema_valid": out["facts"]["schema_valid"],
            "requested_task_in_binding": out["binding"]["task_id"], "actual_loaded_task": provider.loaded_task,
            "requested_rows": ["row-1"], "returned_rows": provider.rows,
            "binding_contains_population": "population" in out["binding"],
            "payload_contains_nan": out["outputs"]["tone"]["payload"]["probability"] != out["outputs"]["tone"]["payload"]["probability"]}
    provider.loaded_task = "task"
    provider.rows, provider.probability = ["row-1"], 0.9
    cases = {"missing_calibration_bindings": dict(provider.calibration),
             "wrong_calibration_digest": {**provider.calibration, "task_id": "task", "state_ref": "state", "state_digest": "c" * 64},
             "offset_future_calibration": {**provider.calibration, "task_id": "task", "state_ref": "state",
                                            "clocks": {"calibration_end": "2026-09-23T23:30:00-05:00"}}}
    for name, calibration in cases.items():
        provider.calibration = calibration
        out = run({**req, "operation": "calibrate"}, registry)
        results[name] = {"status": out["status"], "calibration_bound": out["facts"]["calibration_bound"]}
    forecast = run({**req, "family": "regression_forecasting", "output_kind": "marginal_quantiles",
                    "output_schema": {"targets": ["price"], "horizons": [6, 72], "quantiles": [0.1, 0.5, 0.9]}}, registry)
    results["declared_forecast_without_questions"] = {"status": forecast["status"], "why": forecast["why"]}
    with tempfile.TemporaryDirectory(prefix="m5phet-738-review-") as root:
        kwargs = {"run_id": "local", "task": {"id": "A"}, "code_identity": {"commit": "A"}}
        original = open_run(Path(root), **kwargs)
        manifest_path = original.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["task"] = {"id": "B"}
        manifest["task"] = {"id": "B"}
        manifest["authority"] = "GOVERNED"
        manifest_path.write_text(json.dumps(manifest))
        resumed = open_run(Path(root), **kwargs, resume=True)
        results["modified_manifest_keeps_old_digest"] = {
            "task": resumed.manifest["task"], "authority": resumed.authority,
            "same_digest": original.identity_sha256 == resumed.identity_sha256}
        attempt = resumed.start_attempt(candidate="test")
        attempts_path = resumed.directory / "attempts.jsonl"
        with attempts_path.open("a") as handle:
            handle.write('{"broken":\n')
        resumed.finish_attempt(attempt, status="COMPLETED")
        reopened = open_run(Path(root), **kwargs, resume=True)
        results["corrupt_interior_without_explicit_recover"] = {
            "reported_open": reopened.close()["open_attempts_at_close"],
            "corrupt_bytes_still_present": "broken" in attempts_path.read_text()}
    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
