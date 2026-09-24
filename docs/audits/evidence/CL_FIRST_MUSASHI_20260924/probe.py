"""CPU audit of retained evidence and public boundaries; no model inference.

Run with installed candidate news-signal/M5PHET and --news/--predictor paths.
Fixture backends below are explicitly not a repeat of the real-weight pilot.
All deliberately malformed writes remain inside TemporaryDirectory.
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

    fixture = load_module("news_audit_fixture", args.news / "tests/test_classification_first.py")
    evidence = args.predictor / "docs/audits/evidence/CL01_CL07_20260924"
    direct = json.loads((evidence / "direct.json").read_text())
    corpus = json.loads((args.news / "examples/eurusd/corpus.json").read_text())
    native = {r["input_sha256"]: r["response"]["answers"] for r in direct["single"]}
    parity = {}
    for phase in ("first", "restart"):
        receipts = [json.loads(p.read_text()) for p in (evidence / phase / "receipts").glob("*.json")]
        by_input = {r["input_sha256"]: {q: a["sdk_answer"] for q, a in r["features"].items()}
                    for r in receipts if r["status"] == "SHADOW_ONLY"}
        expected = {i["input_sha256"] for i in corpus["items"]}
        # JSON serialization without sorting also checks nested key order.
        parity[phase] = {"expected_distinct_inputs": len(expected),
                        "population_exact": set(by_input) == set(native) == expected,
                        "exact_answers": sum(json.dumps(native[k]) == json.dumps(by_input[k])
                                             for k in expected & set(by_input))}
    labels = sorted({r["label"] for r in corpus["items"]})
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for row in corpus["items"]:
        confusion[row["label"]][native[row["input_sha256"]]["relevance"]["choice"]] += 1
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[a][label] for a in labels)
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0)
    out = {"scope": "Retained JSON parity, fixture public application/store and fixture retained closure; CPU only",
           "parity": parity, "quality": {"confusion": confusion, "macro_f1": sum(f1s) / len(f1s),
           "correct": sum(confusion[a][a] for a in labels), "rows": len(corpus["items"]),
           "distinct_inputs": len(native), "always_related_accuracy": 7 / 13,
           "always_related_macro_f1": (14 / 20) / 3}}
    with tempfile.TemporaryDirectory(prefix="musashi-cl-") as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": "c" * 64, "files": {}}))
        registry = Registry()
        registry.register(LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest), "NEWS_SIGNAL_DEVICE": "cpu"},
                          backend_factory=lambda config, state: fixture.StubLaya()))
        event = json.loads((args.news / "examples/eurusd/events/00_relevant.json").read_text())

        def call(item, store, as_of=fixture.AS_OF, task=fixture.TASK):
            return classify_event(item, task_id=task, as_of=as_of, registry=registry, store=store)

        escape = dict(event, event_id="../escaped")
        escaped = call(escape, ShadowStore(root / "store"))
        out["path_escape"] = {"application_status": escaped["status"],
                              "written_outside_store": (root / "escaped" / f"{digest(escape)}.json").is_file()}
        future = json.loads((args.news / "examples/eurusd/refusals/01_future_receipt.json").read_text())
        store = ShadowStore(root / "retry")
        refused = call(future, store)
        accepted = call(future, store, as_of=future["received_at"])
        out["refusal_then_success"] = {"first": refused["status"], "second": accepted["status"],
            "disposition": accepted["stored"]["disposition"],
            "persisted_status": store.find(future["event_id"], digest(future))["status"]}
        store = ShadowStore(root / "tasks")
        call(event, store)
        other = call(event, store, task="news_triage.v1")
        out["different_task"] = {"new_receipt_task": other["receipt"]["task_id"],
            "disposition": other["stored"]["disposition"],
            "persisted_task": store.find(event["event_id"], digest(event))["task_id"]}
        path = store._path(event["event_id"], digest(event))
        damaged = json.loads(path.read_text())
        damaged["features"]["relevance"]["label"] = "TAMPERED"
        path.write_text(json.dumps(damaged))
        duplicate = call(event, store)
        replay = store.replay()
        out["corrupted_duplicate"] = {"disposition": duplicate["stored"]["disposition"],
            "actionable_events": replay["actionable_events"], "integrity_failures": len(replay["integrity_failures"])}

        gates = load_module("retained_audit_fixture", args.predictor / "tests/test_df_ecl_contrast_gates.py")
        def forbidden(*a, **k):
            raise AssertionError("No inference permitted by this audit")
        gates.M.score_cell = forbidden
        run = gates._retained_run(root / "retained")
        baseline = gates.M.close_retained_run(run)
        scoring_path = run / "SCORING.json"
        scoring = json.loads(scoring_path.read_text())
        forged = copy.deepcopy(scoring)
        forged["child_binding"]["expected_windows_from_contract"] = {
            "complete_validation": 1, "label_disjoint_from_selection": 1}
        for child in forged["cells"].values():
            for population in child["populations"].values():
                population.update(windows=1, elements=96 * 321)
        scoring_path.write_text(json.dumps(forged))
        mutated = gates.M.close_retained_run(run)
        out["retained_population_rewrite"] = {"baseline": baseline["status"], "mutated": mutated["status"],
            "mutated_problems": mutated["problems"], "design_and_contrast_changed": False,
            "summaries_present": mutated["by_regime"] is not None}
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
