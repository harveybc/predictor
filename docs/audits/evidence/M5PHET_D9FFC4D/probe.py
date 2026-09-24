"""Independent POST continuation; original PRE stays untouched. CPU, temporary files only.

Use the reviewed candidate src on PYTHONPATH. The old recording provider is reused,
not the candidate's tests. Each case is isolated; an expected refusal cannot skip
later checks. Assertions freeze both repaired behavior and reproduced open defects.
"""
import argparse
import copy
import json
from pathlib import Path
import runpy
import tempfile

from m5phet import ContractError
from m5phet.evidence import open_run
from m5phet.runtime import REQUEST_SCHEMA, Registry, run

Provider = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                             "M5PHET_SPECIALIZED_7385937/probe.py"))["Provider"]


def setup():
    provider, registry = Provider(), Registry()
    registry.register(provider)
    provider.calibration.update(task_id="task", state_ref="state", state_digest="a" * 64)
    request = {"schema_version": REQUEST_SCHEMA, "request_id": "request", "task_id": "task",
               "operation": "infer", "family": "classification", "output_kind": "typed_questions",
               "provider_ref": "probe", "fitted_state_ref": "state", "as_of": "2026-09-24T00:00:00Z",
               "population": {"ids": ["row-1"]}, "output_schema": {"questions": ["tone"]}}
    return provider, registry, request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {"revision": "d9ffc4deaf7948d9432597e8c68bcd48e6ceb41c",
               "scope": "real public APIs, recording provider, temporary evidence; no model or live service",
               "cases": {}}
    cases = results["cases"]
    for case in ("baseline", "wrong_loaded_task", "foreign_top_level_population",
                 "foreign_nested_population", "nonfinite_payload", "missing_calibration_digest",
                 "wrong_calibration_digest", "future_offset", "past_offset", "crossed_combination"):
        p, registry, request = setup()
        expected = "OK"
        if case == "wrong_loaded_task":
            p.loaded_task, expected = "foreign-task", "INVALID_INPUT"
        if case in ("foreign_top_level_population", "foreign_nested_population"):
            p.rows = ["foreign-row"]
            if case == "foreign_nested_population":
                request["input_schema"] = {"population": request.pop("population")}
            else:
                expected = "INVALID_INPUT"
        if case == "nonfinite_payload":
            p.probability, expected = float("nan"), "INVALID_INPUT"
        if case in ("missing_calibration_digest", "wrong_calibration_digest", "future_offset", "past_offset"):
            request["operation"] = "calibrate"
            expected = "CALIBRATION_UNAVAILABLE"
            if case == "missing_calibration_digest":
                p.calibration.pop("state_digest")
            elif case == "wrong_calibration_digest":
                p.calibration["state_digest"] = "c" * 64
            else:
                p.calibration["clocks"]["calibration_end"] = (
                    "2026-09-23T23:30:00-05:00" if case == "future_offset" else "2026-09-23T18:00:00-05:00")
                if case == "past_offset":
                    expected = "OK"
        if case == "crossed_combination":
            request["output_kind"], expected = "marginal_quantiles", "UNSUPPORTED_TASK"
        out = run(request, registry)
        assert out["status"] == expected, (case, out)
        cases[case] = {"status": out["status"], "why": out.get("why"), "calls": p.calls,
                       "population_bound": bool(out.get("binding", {}).get("population_sha256"))}

    class Forecast(Provider):
        def infer(self, request, state):
            return {"population": copy.deepcopy(request["population"]),
                    "outputs": {"price": {"status": "OK", "uncertainty": "UNCALIBRATED",
                                          "payload": {"target": "price", "horizon": 6,
                                                      "quantiles": {"0.1": 1.0, "0.5": 1.1, "0.9": 1.2}}}}}

    for horizons in ([6], [6, 72]):
        _, _, request = setup()
        registry = Registry()
        registry.register(Forecast())
        request.update(family="regression_forecasting", output_kind="marginal_quantiles",
                       output_schema={"targets": ["price"], "horizons": horizons, "quantiles": [0.1, 0.5, 0.9]})
        out = run(request, registry)
        assert out["status"] == "OK", out
        cases["forecast_" + "_".join(map(str, horizons))] = {
            "requested_horizons": horizons, "returned_horizons": [out["outputs"]["price"]["payload"]["horizon"]],
            "status": out["status"], "schema_valid": out["facts"]["schema_valid"]}

    for case in ("manifest_rewritten", "interior_corruption", "torn_tail_then_append"):
        with tempfile.TemporaryDirectory(prefix="m5phet-d9-review-") as root:
            kwargs = {"run_id": "run", "task": {"id": "task"}, "code_identity": {"commit": "test"}}
            evidence = open_run(Path(root), **kwargs)
            first = evidence.start_attempt(candidate="first")
            evidence.finish_attempt(first, status="OK")
            path = evidence.directory / "attempts.jsonl"
            if case == "manifest_rewritten":
                path = evidence.directory / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["identity"]["task"] = {"id": "changed"}
                path.write_text(json.dumps(manifest))
            else:
                with path.open("a") as stream:
                    stream.write('{"event": "torn')
                    if case == "interior_corruption":
                        stream.write('\n{"event":"unrelated"}\n')
            try:
                reopened = open_run(Path(root), **kwargs, resume=True)
            except ContractError as error:
                assert case != "torn_tail_then_append"
                cases[case] = {"refused_on_open": True, "reason": str(error)}
                continue
            assert case == "torn_tail_then_append"
            second = reopened.start_attempt(candidate="second")
            reopened.finish_attempt(second, status="OK")
            valid_events = []
            invalid_lines = 0
            for line in path.read_text().splitlines():
                try:
                    valid_events.append(json.loads(line))
                except ValueError:
                    invalid_lines += 1
            starts = [e.get("attempt_id") for e in valid_events if e.get("event") == "attempt_started"]
            assert second not in starts and invalid_lines == 1
            try:
                open_run(Path(root), **kwargs, resume=True)
            except ContractError as error:
                cases[case] = {"append_and_finish_accepted": True, "new_start_parseable": False,
                               "subsequent_open_refused": True, "reason": str(error)}
            else:
                raise AssertionError("expected newly interior corruption")
    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(args.output).write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
