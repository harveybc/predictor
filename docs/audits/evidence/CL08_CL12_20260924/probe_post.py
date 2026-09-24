"""POST for the five reviewed counterexamples, through the PUBLIC application and store surface only.

This is Musashi's probe re-aimed at the repaired code. One deliberate difference: it never touches a private method. The
audit's `store._path(event_id, input_sha256)` no longer exists, because the repair removed exactly that -- a path derived
from an external identifier. Where the audit reached for it, this reads the record back through `find`, `records_for` and
`find_record`. All writes stay inside a TemporaryDirectory; no weights, no inference, no broker.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from m5phet.runtime import Registry
    from news_signal.application import classify_event
    from news_signal.core import digest
    from news_signal.provider import LayaNewsProvider
    from news_signal.shadow import ShadowStore

    fixture = load_module("news_fixture", args.news / "tests/test_classification_first.py")
    out = {"scope": "the same five cases the review reproduced, after repair; public surface only, CPU, no inference"}
    with tempfile.TemporaryDirectory(prefix="satoshi-cl08-post-") as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": "c" * 64, "files": {}}))
        registry = Registry()
        registry.register(LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest), "NEWS_SIGNAL_DEVICE": "cpu"},
                          backend_factory=lambda config, state: fixture.StubLaya()))
        event = json.loads((args.news / "examples/eurusd/events/00_relevant.json").read_text())

        def call(item, store, as_of=fixture.AS_OF, task=fixture.TASK):
            return classify_event(item, task_id=task, as_of=as_of, registry=registry, store=store)

        # 1. an identifier from a feed cannot choose a path
        escape_root = root / "escape"
        store = ShadowStore(escape_root / "store")
        escape = dict(event, event_id="../escaped")
        escaped = call(escape, store)
        written = sorted(p.relative_to(escape_root).as_posix() for p in escape_root.rglob("*.json"))
        out["path_escape"] = {"application_status": escaped["status"],
                              "written_outside_store": any(not p.startswith("store/") for p in written),
                              "files_written": written,
                              "retained_event_id": store.all_records()[0]["event_id"]}

        # 2a. a refusal does not answer for a later successful evaluation
        future = json.loads((args.news / "examples/eurusd/refusals/01_future_receipt.json").read_text())
        store = ShadowStore(root / "retry")
        refused = call(future, store)
        accepted = call(future, store, as_of=future["received_at"])
        kept = store.records_for(future["event_id"])
        out["refusal_then_success"] = {"first": refused["status"], "second": accepted["status"],
                                       "disposition": accepted["stored"]["disposition"],
                                       "retained_statuses": sorted(r["status"] for r in kept),
                                       "successful_is_retrievable":
                                           store.find(future["event_id"], digest(future))["status"]}

        # 2b. another task is another evaluation
        store = ShadowStore(root / "tasks")
        first = call(event, store)
        other = call(event, store, task="news_triage.v1")
        out["different_task"] = {"new_receipt_task": other["receipt"]["task_id"],
                                 "disposition": other["stored"]["disposition"],
                                 "retained_tasks": sorted(r["task_id"] for r in store.records_for(event["event_id"])),
                                 "receipt_points_at_its_own_result":
                                     store.find_record(other["stored"]["record_sha256"])["task_id"]}

        # 2c. a tampered record is not reused and not counted
        damaged_path = next(p for p in (root / "tasks").rglob("*.json"))
        damaged = json.loads(damaged_path.read_text())
        damaged["features"]["relevance"]["label"] = "TAMPERED"
        damaged_path.write_text(json.dumps(damaged))
        duplicate = call(event, store, task=damaged["task_id"])
        replay = store.replay()
        out["corrupted_duplicate"] = {"disposition": duplicate["stored"]["disposition"],
                                      "actionable_events": replay["actionable_events"],
                                      "integrity_failures": len(replay["integrity_failures"]),
                                      "quarantined": replay["quarantined"],
                                      "tampered_label_served":
                                          any(r.get("features", {}).get("relevance", {}).get("label") == "TAMPERED"
                                              for r in store.all_records() if r.get("integrity") == "OK")}

        # 4. the retained closure cannot be told its own expected population
        gates = load_module("retained_fixture", args.predictor / "tests/test_df_ecl_contrast_gates.py")

        def forbidden(*a, **k):
            raise AssertionError("no inference permitted in this probe")

        gates.M.score_cell = forbidden
        run = gates._retained_run(root / "retained")
        baseline = gates.M.close_retained_run(run)
        scoring_path = run / "SCORING.json"
        forged = copy.deepcopy(json.loads(scoring_path.read_text()))
        forged["child_binding"]["expected_windows_from_contract"] = {"complete_validation": 1,
                                                                    "label_disjoint_from_selection": 1}
        for child in forged["cells"].values():
            for population in child["populations"].values():
                population.update(windows=1, elements=96 * 321)
        scoring_path.write_text(json.dumps(forged))
        try:
            mutated = gates.M.close_retained_run(run)
            result = {"status": mutated["status"], "problems": mutated["problems"],
                      "summaries_present": mutated["by_regime"] is not None}
        except Exception as exc:                                    # noqa: BLE001
            result = {"status": "REFUSED", "refusal": f"{type(exc).__name__}: {exc}", "summaries_present": False}
        out["retained_population_rewrite"] = {"baseline": baseline["status"], "mutated": result,
                                              "authority": baseline["authority"]}
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
